// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::controller::{ControlResult, Controller, ControllerError};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub type SharedController = Arc<Mutex<Controller>>;

pub async fn serve(
    controller: SharedController,
    address: SocketAddr,
    schedule_interval: f64,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/management/capabilities", get(management_capabilities))
        .route("/v1/management/:action", post(management_call))
        .route("/v1/identity/me", get(identity_me))
        .route("/v1/identity/pools", get(identity_pools))
        .route("/v1/identity/tokens", get(identity_tokens).post(identity_register))
        .route("/v1/identity/tokens/:token_id/revoke", post(identity_revoke))
        .route("/v1/identity/admin", post(identity_admin))
        .route("/v1/metrics/status", get(metric_status))
        .route("/v1/metrics/query", get(metric_query))
        .route("/v1/metrics/service-config", get(metric_service_config))
        .route("/v1/agent/metrics/control", post(metric_control))
        .route("/v1/identity/nodes/enroll", post(identity_enroll))
        .route("/v1/identity/nodes", get(identity_nodes))
        .route("/v1/identity/nodes/:node_id/action", post(identity_node_action))
        .route("/v1/submit", post(submit))
        .route("/v1/sessions", get(logical_sessions))
        .route("/v1/executions/:execution_id/:action", post(execution_action))
        .route("/v1/requests/:request_id", get(status))
        .route("/v1/requests/:request_id/wait", get(wait))
        .route("/v1/requests/:request_id/logs", get(logs))
        .route("/v1/requests/:request_id/cancel", post(cancel))
        .route("/v1/workspaces/:workspace_id", get(workspace_status))
        .route("/v1/workspaces/:workspace_id/:action", post(workspace_operation))
        .route("/v1/pools/:pool/utilization", get(utilization))
        .route("/v1/agent/register", post(agent_register))
        .route("/v1/agent/announce", post(agent_announce))
        .route("/v1/agent/heartbeat", post(agent_heartbeat))
        .route("/v1/agent/report-command", post(agent_report))
        .route("/v1/admin/drain-allocation", post(drain_allocation))
        .route("/v1/admin/remove-allocation", post(remove_allocation))
        .route("/v1/admin/drain-node", post(drain_node))
        .route("/v1/admin/set-pool-config", post(set_pool_config))
        .route("/v1/admin/workspace-config", get(get_workspace_config).post(set_workspace_config))
        .with_state(controller.clone());
    let tick_controller = controller.clone();
    let tick = tokio::spawn(async move {
        let delay = Duration::from_secs_f64(schedule_interval.max(0.01));
        loop {
            tokio::time::sleep(delay).await;
            let mut controller = tick_controller.lock().unwrap();
            controller.check_heartbeats();
            controller.expire_idle_sessions();
            controller.schedule();
        }
    });
    let result = axum::Server::bind(&address)
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await;
    tick.abort();
    result?;
    let controller = controller.lock().unwrap();
    controller.flush_journal()?;
    if let Some((events, commits)) = controller.journal_stats() {
        eprintln!("[rscape-journal] events={events} commits={commits}");
    }
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(signal) => signal,
                Err(_) => {
                    let _ = tokio::signal::ctrl_c().await;
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

async fn health() -> Json<Value> {
    Json(json!({"ok":true}))
}

struct ApiError(ControllerError);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let payload = self
            .0
            .payload
            .unwrap_or_else(|| json!({"error":self.0.message}));
        (status, Json(payload)).into_response()
    }
}
fn api<T>(result: ControlResult<T>) -> Result<T, ApiError> {
    result.map_err(ApiError)
}
fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

fn identity_store(controller: &Controller) -> Result<crate::identity::IdentityStore, ApiError> {
    controller.authority.identity().map_err(|_| ApiError(ControllerError::internal("identity storage unavailable")))?
        .ok_or_else(|| ApiError(ControllerError::bad("persistent identity is not configured")))
}

fn identity_result<T>(result: anyhow::Result<T>) -> Result<T, ApiError> {
    result.map_err(|error| {
        if let Some(control) = error.downcast_ref::<ControllerError>() {
            return ApiError(ControllerError { status: control.status, message: control.message.clone(), payload: control.payload.clone() });
        }
        let message = error.to_string();
        let forbidden = ["invalid credential", "invalid node credential", "target identity mismatch", "enrollment belongs to another identity", "admin credential required", "credential not accessible", "cannot create another identity", "recovery requires owner", "missing bearer", "SSO bridge credential required", "SSO identity scope mismatch", "SSO account not provisioned", "SSO account unavailable", "SSO session required", "node not accessible"]
            .iter().any(|text| message.contains(text));
        ApiError(if forbidden { ControllerError::forbidden(message) } else { ControllerError::bad(message) })
    })
}

async fn identity_enroll(State(controller): State<SharedController>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    Ok(Json(identity_result(identity_store(&controller)?.enroll_node(bearer(&headers).unwrap_or(""), &body, crate::controller::now()))?))
}

async fn identity_node_action(State(controller): State<SharedController>, headers: HeaderMap, Path(node_id): Path<String>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let mut controller = controller.lock().unwrap();
    if body.as_object().map(|b|b.keys().any(|k| !["action","force","changes"].contains(&k.as_str()))).unwrap_or(true) {
        return Err(ApiError(ControllerError::bad("invalid node action")));
    }
    let force = body.get("force").map(Value::as_bool).unwrap_or(Some(false)).ok_or_else(||ApiError(ControllerError::bad("invalid force value")))?;
    if body["action"] == "edit" {
        if force { return Err(ApiError(ControllerError::bad("force applies to drain, not edit"))); }
        return Ok(Json(api(controller.managed_node_edit(bearer(&headers), &node_id, &body["changes"]))?));
    }
    Ok(Json(api(controller.managed_node_action(bearer(&headers), &node_id, body["action"].as_str().unwrap_or(""), force, false))?))
}

async fn identity_nodes(State(controller): State<SharedController>, headers: HeaderMap, Query(query): Query<std::collections::HashMap<String,String>>) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    let store = identity_store(&controller)?;
    identity_result(store.authenticate(bearer(&headers).unwrap_or(""), crate::controller::now()))?;
    let pool = query.get("pool").map(|p|store.pool_id(p)).transpose().map_err(|e|ApiError(ControllerError::bad(e.to_string())))?;
    let state = identity_result(store.snapshot())?;
    let mut result = Vec::new();
    if let Some(nodes) = state["nodes"].as_object() {
        for (id, node) in nodes {
            if node["desired"] == "removed" || pool.as_ref().map(|p|node["pool"] != *p).unwrap_or(false) { continue; }
            let mut metrics = Vec::new();
            if let Some(status) = controller.nodes.get(id) {
                if let Some(samples) = status.status["gpu_metrics"].as_array() {
                    for sample in samples {
                        if node.get("effective_resources").unwrap_or(&node["resources"])["gpu_uuids"].as_array().map(|ids|ids.contains(&sample["uuid"])).unwrap_or(false) {
                            if let Some(sample) = sample.as_object() {
                                metrics.push(Value::Object(sample.iter().filter(|(k,_)|["index","uuid","name","timestamp","gpu_utilization_percent","memory_utilization_percent",
                                    "memory_total_bytes","memory_used_bytes","memory_free_bytes","temperature_c"].contains(&k.as_str())).map(|(k,v)|(k.clone(),v.clone())).collect()));
                            }
                        }
                    }
                }
            }
            result.push(json!({"node_id":id,"pool_id":node["pool"],"resources":node.get("effective_resources").unwrap_or(&node["resources"]),
                "state":controller.nodes.get(id).map(|n|serde_json::to_value(n.state).unwrap()).unwrap_or(json!("enrolled")),"gpu_metrics":metrics}));
        }
    }
    Ok(Json(json!(result)))
}

async fn identity_me(State(controller): State<SharedController>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    let store = identity_store(&controller)?;
    let mut actor = identity_result(store.authenticate(bearer(&headers).unwrap_or(""), crate::controller::now()))?;
    let state = identity_result(store.snapshot())?;
    let mut permissions = serde_json::Map::new();
    for group in actor["groups"].as_array().unwrap() {
        if let Some(name) = group.as_str() { if let Some(policy) = state["groups"].get(name) { permissions.insert(name.to_owned(), policy.clone()); } }
    }
    actor["scheduler_id"] = state["scheduler_id"].clone();
    actor["permissions"] = Value::Object(permissions);
    Ok(Json(actor))
}

async fn identity_tokens(State(controller): State<SharedController>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    Ok(Json(json!(identity_result(identity_store(&controller)?.list_tokens(bearer(&headers).unwrap_or(""), crate::controller::now()))?)))
}

async fn identity_pools(State(controller): State<SharedController>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    let store = identity_store(&controller)?;
    identity_result(store.authenticate(bearer(&headers).unwrap_or(""), crate::controller::now()))?;
    let state = identity_result(store.snapshot())?;
    Ok(Json(Value::Array(state["pools"].as_object().unwrap().iter().map(|(id,pool)| {
        let status = if pool["draining"] == true { "draining" } else if controller.nodes.values().any(|n| n.pool == *id && n.state == crate::model::NodeState::Active) { "available" } else { "unavailable" };
        json!({"pool_id":id,"name":pool["name"],"state":status,"config":pool["config"],"max_nodes":pool["max_nodes"]})
    }).collect())))
}

async fn identity_register(State(controller): State<SharedController>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let mut controller = controller.lock().unwrap();
    let object = body.as_object().ok_or_else(|| ApiError(ControllerError::bad("registration object required")))?;
    if object.keys().any(|key| !["operation_id", "verifier", "name", "username", "expires_at", "user_id"].contains(&key.as_str())) {
        return Err(ApiError(ControllerError::bad("unknown registration parameter")));
    }
    if (!body["expires_at"].is_null() && !body["expires_at"].is_number()) || (!body["username"].is_null() && !body["username"].is_string()) || (!body["user_id"].is_null() && !body["user_id"].is_string()) {
        return Err(ApiError(ControllerError::bad("invalid registration parameter")));
    }
    if bearer(&headers).is_none() {
        let now = std::time::Instant::now();
        while controller.registration_attempts.front().map(|t|now.duration_since(*t).as_secs()>=60).unwrap_or(false) { controller.registration_attempts.pop_front(); }
        if controller.registration_attempts.len() >= 64 { return Err(ApiError(ControllerError { status:429, message:"anonymous registration rate limit exceeded".into(), payload:None })); }
        controller.registration_attempts.push_back(now);
    }
    Ok(Json(identity_result(identity_store(&controller)?.register_token_for(
        body["operation_id"].as_str().unwrap_or(""), body["verifier"].as_str().unwrap_or(""), body["name"].as_str().unwrap_or(""),
        body["username"].as_str(), bearer(&headers), body["expires_at"].as_f64(), crate::controller::now(), body["user_id"].as_str()))?))
}

async fn identity_revoke(State(controller): State<SharedController>, headers: HeaderMap, Path(token_id): Path<String>) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    identity_result(identity_store(&controller)?.revoke_token(bearer(&headers).unwrap_or(""), &token_id, crate::controller::now()))?;
    Ok(Json(json!({"revoked":true})))
}

async fn metric_query(State(controller): State<SharedController>, headers: HeaderMap, Query(params): Query<std::collections::BTreeMap<String,String>>) -> Result<Json<Value>, ApiError> {
    let (plan, url) = {
        let controller = controller.lock().unwrap();
        let store = identity_store(&controller)?;
        identity_result(store.authenticate(bearer(&headers).unwrap_or(""), crate::controller::now()))?;
        let plan = identity_result(crate::telemetry_query::prepare_query(&identity_result(store.snapshot())?, &params, crate::controller::now()))?;
        (plan, controller.telemetry_query_url.clone())
    };
    Ok(Json(crate::telemetry_query::fetch_query(plan, url).await))
}

async fn metric_service_config(State(controller): State<SharedController>, Query(params): Query<std::collections::BTreeMap<String,String>>) -> Result<Json<Value>, ApiError> {
    if params.len()!=1 || !params.contains_key("service_id"){return Err(ApiError(ControllerError::bad("service_id required")));}
    let controller=controller.lock().unwrap();
    let mut result=identity_result(crate::telemetry_control::public_config(&identity_store(&controller)?,&params["service_id"],crate::controller::now()))?;
    result["receiver_url"]=json!(controller.telemetry_receiver_url);
    Ok(Json(result))
}

async fn metric_status(State(controller): State<SharedController>, headers: HeaderMap, Query(params): Query<std::collections::BTreeMap<String,String>>) -> Result<Json<Value>, ApiError> {
    if params.keys().any(|k| k != "node_id") { return Err(ApiError(ControllerError::bad("unknown metrics status parameter"))); }
    let controller = controller.lock().unwrap();
    Ok(Json(identity_result(crate::telemetry_control::get_config(&identity_store(&controller)?, bearer(&headers).unwrap_or(""), params.get("node_id").map(String::as_str), false, crate::controller::now()))?))
}

async fn metric_control(State(controller): State<SharedController>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    let mut result = identity_result(crate::telemetry_control::poll_control(&mut identity_store(&controller)?, bearer(&headers).unwrap_or(""), &body))?;
    result["receiver_url"] = json!(controller.telemetry_receiver_url);
    Ok(Json(result))
}

async fn identity_admin(State(controller): State<SharedController>, headers: HeaderMap, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    let mut controller = controller.lock().unwrap();
    let mut store = identity_store(&controller)?;
    let token = bearer(&headers).unwrap_or("");
    let now = crate::controller::now();
    identity_result(store.require_admin(token, now))?;
    let name = body["name"].as_str().unwrap_or("");
    match body["action"].as_str().unwrap_or("") {
        action @ ("metric-get" | "metric-set") => {
            let valid = body.as_object().map(|o|o.keys().all(|k| ["action","node_id"].contains(&k.as_str())
                || (action == "metric-set" && ["changes","expected_version"].contains(&k.as_str())))).unwrap_or(false);
            if !valid { return Err(ApiError(ControllerError::bad("unknown telemetry parameter"))); }
            let node = match body.get("node_id") { None | Some(Value::Null) => None, Some(Value::String(v)) => Some(v.as_str()), _ => return Err(ApiError(ControllerError::bad("invalid node ID"))) };
            if action == "metric-get" { return Ok(Json(identity_result(crate::telemetry_control::get_config(&store, token, node, true, now))?)); }
            let expected = match body.get("expected_version") { None | Some(Value::Null) => None, Some(v) => Some(v.as_u64().ok_or_else(||ApiError(ControllerError::bad("invalid expected telemetry version")))?) };
            return Ok(Json(identity_result(crate::telemetry_control::set_config(&mut store, token, &body["changes"], node, expected, now))?));
        },
        "node-release-quarantine" => {
            if body.as_object().map(|b|b.len()!=4 || !b.contains_key("node_id") || !b.contains_key("reason") || !b.contains_key("confirm_stopped")).unwrap_or(true) {
                return Err(ApiError(ControllerError::bad("invalid quarantine release parameters")));
            }
            let id=body["node_id"].as_str().ok_or_else(||ApiError(ControllerError::bad("invalid node ID")))?;
            let reason=body["reason"].as_str().ok_or_else(||ApiError(ControllerError::bad("invalid release reason")))?;
            let confirmed=body["confirm_stopped"].as_bool().ok_or_else(||ApiError(ControllerError::bad("invalid confirm_stopped")))?;
            return Ok(Json(api(controller.managed_release_quarantine(Some(token),id,reason,confirmed))?));
        },
        "node-purge" => {
            if body.as_object().map(|b|b.len()!=3 || !b.contains_key("node_id") || !b.contains_key("reason")).unwrap_or(true) {
                return Err(ApiError(ControllerError::bad("invalid node purge parameters")));
            }
            let id=body["node_id"].as_str().ok_or_else(||ApiError(ControllerError::bad("invalid node ID")))?;
            let reason=body["reason"].as_str().ok_or_else(||ApiError(ControllerError::bad("invalid purge reason")))?;
            return Ok(Json(api(controller.managed_node_purge(Some(token),id,reason))?));
        },
        "audit" => {
            let limit: usize = serde_json::from_value(body.get("limit").cloned().unwrap_or(json!(100))).map_err(|_|ApiError(ControllerError::bad("invalid audit limit")))?;
            return Ok(Json(json!(identity_result(store.audit_events(token,limit,now))?)));
        },
        "user-list" => {
            let state = identity_result(store.snapshot())?;
            return Ok(Json(json!(state["users"].as_object().unwrap().iter().map(|(id,user)| {
                let mut user = user.clone(); user["user_id"] = json!(id); user
            }).collect::<Vec<_>>())));
        },
        "group-list" => return Ok(Json(identity_result(store.snapshot())?["groups"].clone())),
        "pool-drain" | "pool-remove" => {
            let pool_id = identity_result(store.pool_id(body["pool"].as_str().unwrap_or("")))?;
            let force = body.get("force").map(Value::as_bool).unwrap_or(Some(false)).ok_or_else(||ApiError(ControllerError::bad("invalid force value")))?;
            let action = if body["action"] == "pool-drain" {"drain"} else {"remove"};
            return Ok(Json(api(controller.managed_pool_action(bearer(&headers), &pool_id, action, force))?));
        },
        "pool-create" => {
            let limit: Option<u64> = serde_json::from_value(body["max_nodes"].clone()).map_err(|_|ApiError(ControllerError::bad("invalid node limit")))?;
            return Ok(Json(json!({"pool_id":identity_result(store.create_pool_config(token, name, body.get("config").filter(|c|!c.is_null()).unwrap_or(&json!({})), limit, now))?})));
        },
        "pool-edit" => {
            let pool = identity_result(store.pool_id(body["pool"].as_str().unwrap_or("")))?;
            return Ok(Json(api(controller.managed_pool_edit(bearer(&headers), &pool, &body["changes"]))?));
        },
        "group-set" => {
            let pools: Vec<String> = serde_json::from_value(body.get("pools").cloned().unwrap_or_else(|| json!([])))
                .map_err(|_| ApiError(ControllerError::bad("invalid pool IDs")))?;
            let pools = identity_result(pools.iter().map(|pool| store.pool_id(pool)).collect::<anyhow::Result<Vec<_>>>())?;
            let all = body.get("all_pools").map(|v| v.as_bool()).unwrap_or(Some(false))
                .ok_or_else(|| ApiError(ControllerError::bad("invalid group policy")))?;
            identity_result(store.set_group(token, name, all, &pools, now))?;
        },
        "group-delete" => identity_result(store.delete_group(token, name, now))?,
        "user-edit" => {
            let groups: Option<Vec<String>> = serde_json::from_value(body["groups"].clone()).map_err(|_| ApiError(ControllerError::bad("invalid groups")))?;
            let disabled: Option<bool> = serde_json::from_value(body["disabled"].clone()).map_err(|_| ApiError(ControllerError::bad("invalid disabled value")))?;
            identity_result(store.edit_user(token, body["user_id"].as_str().unwrap_or(""), groups, disabled, now))?;
        },
        _ => return Err(ApiError(ControllerError::bad("unknown identity management action"))),
    }
    Ok(Json(json!({"updated":true})))
}

async fn submit(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    if ["task_id", "session_key", "workspace_id"].iter().any(|key| body.get(*key).is_some()) {
        return Err(ApiError(ControllerError::bad("obsolete workload identity fields; use session_id and workspace.id")));
    }
    {
        let controller = controller.lock().unwrap();
        if controller.authority.identity_path.is_some() {
            let store = identity_store(&controller)?;
            let actor = identity_result(store.authenticate(bearer(&headers).unwrap_or(""), crate::controller::now()))?;
            if !body["user_id"].is_null() && body["user_id"] != "" && body["user_id"] != actor["user_id"] && body["user_id"] != actor["name"] {
                return Err(ApiError(ControllerError::forbidden("cannot submit for another identity")));
            }
            body["user_id"] = actor["user_id"].clone();
            body["pool"] = json!(identity_result(store.pool_id(body["pool"].as_str().unwrap_or("")))?);
        }
    }
    let spec = serde_json::from_value(body)
        .map_err(|error| ApiError(ControllerError::bad(error.to_string())))?;
    let (value, receipt) = api(controller
        .lock()
        .unwrap()
        .submit_unflushed(spec, bearer(&headers)))?;
    if let Some(receipt) = receipt {
        tokio::task::spawn_blocking(move || receipt.wait())
            .await
            .map_err(|error| {
                ApiError(ControllerError::internal(format!(
                    "journal acknowledgement task failed: {error}"
                )))
            })?
            .map_err(|error| {
                ApiError(ControllerError::internal(format!(
                    "durable submit failed: {error:#}"
                )))
            })?;
    }
    Ok(Json(value))
}

async fn logical_sessions(State(controller):State<SharedController>,headers:HeaderMap,
    Query(query):Query<std::collections::HashMap<String,String>>)->Result<Json<Value>,ApiError>{
    Ok(Json(api(controller.lock().unwrap().logical_session_list(bearer(&headers),query.get("session_id").map(String::as_str)))?))
}

async fn execution_action(State(controller):State<SharedController>,headers:HeaderMap,
    Path((id,action)):Path<(String,String)>,Json(body):Json<Value>)->Result<Json<Value>,ApiError>{
    if !body.as_object().map(|m|m.keys().all(|k|k=="force")).unwrap_or(false)
        || body.get("force").map(|v|!v.is_boolean()).unwrap_or(false){return Err(ApiError(ControllerError::bad("invalid execution action body")));}
    Ok(Json(api(controller.lock().unwrap().execution_action(&id,&action,body["force"].as_bool().unwrap_or(false),bearer(&headers)))?))
}
async fn status(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Path(request_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(api(controller
        .lock()
        .unwrap()
        .request_status(&request_id, bearer(&headers)))?))
}
async fn logs(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Path(request_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(api(controller
        .lock()
        .unwrap()
        .logs(&request_id, bearer(&headers)))?))
}

#[derive(Deserialize)]
struct WaitQuery {
    timeout: Option<f64>,
    running_timeout: Option<f64>,
}
async fn wait(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Path(request_id): Path<String>,
    Query(query): Query<WaitQuery>,
) -> Result<Json<Value>, ApiError> {
    let timeout = query.timeout.unwrap_or(30.0).max(0.0);
    let started = Instant::now();
    let mut running_since = None;
    loop {
        let value = api(controller
            .lock()
            .unwrap()
            .request_status(&request_id, bearer(&headers)))?;
        let terminal = value
            .get("state")
            .and_then(Value::as_str)
            .map(|state| {
                matches!(
                    state,
                    "COMPLETED" | "FAILED" | "CANCELLED" | "EXPIRED" | "LOST" | "INFEASIBLE"
                )
            })
            .unwrap_or(false);
        if terminal {
            return Ok(Json(value));
        }
        if value.get("started_at").is_some()
            && !value["started_at"].is_null()
            && running_since.is_none()
        {
            running_since = Some(Instant::now());
        }
        let overall_expired = started.elapsed().as_secs_f64() >= timeout;
        let running_expired = query
            .running_timeout
            .zip(running_since)
            .map(|(limit, instant)| instant.elapsed().as_secs_f64() >= limit.max(0.0))
            .unwrap_or(false);
        if overall_expired || running_expired {
            let mut value = value;
            value["wait_timed_out"] = Value::Bool(true);
            value["wait_timeout_kind"] = Value::String(
                if running_expired {
                    "running"
                } else {
                    "overall"
                }
                .into(),
            );
            return Ok(Json(value));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn cancel(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Path(request_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let reason = body
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("cancelled");
    Ok(Json(api(controller.lock().unwrap().cancel(
        &request_id,
        reason,
        bearer(&headers),
    ))?))
}
async fn utilization(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Path(pool): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let controller = controller.lock().unwrap();
    let pool = if controller.authority.identity_path.is_some() {
        let store = identity_store(&controller)?;
        identity_result(store.authenticate(bearer(&headers).unwrap_or(""), crate::controller::now()))?;
        identity_result(store.pool_id(&pool))?
    } else {pool};
    let mut result = controller.utilization(&pool);
    if controller.authority.identity_path.is_some() {
        if let Some(nodes) = result["node_states"].as_array_mut() {
            for node in nodes { if let Some(node) = node.as_object_mut() { node.remove("active_sessions"); } }
        }
    }
    Ok(Json(result))
}
async fn agent_register(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    if body.get("workspace_config").is_none() {
        body["workspace_config"] = json!({"state":"unavailable","applied_version":null,"error":"agent did not prove workspace capability"});
    }
    Ok(Json(api(controller.lock().unwrap().announce_or_register(
        &body,
        true,
        bearer(&headers),
    ))?))
}
async fn agent_announce(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    body["workspace_config"] = json!({"state":"pending","applied_version":null});
    Ok(Json(api(controller.lock().unwrap().announce_or_register(
        &body,
        false,
        bearer(&headers),
    ))?))
}
async fn agent_heartbeat(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let (response, receipt) = {
        let mut controller = controller.lock().unwrap();
        let mut response = api(controller.heartbeat(&body, bearer(&headers)))?;
        if let Some(metrics) = body["status"].get("metrics").filter(|v| !v.is_null()) {
            let result = (|| -> Result<Value, ApiError> {
                let mut store = identity_store(&controller)?;
                let payload = json!({"node_id":body["node_id"],"agent_epoch":body["agent_epoch"],
                    "service_id":metrics["service_id"],"applied_version":metrics.get("applied_version").unwrap_or(&json!(0)),
                    "state":metrics.get("state").unwrap_or(&json!("pending"))});
                let mut control = identity_result(crate::telemetry_control::poll_control(&mut store,bearer(&headers).unwrap_or(""),&payload))?;
                control["receiver_url"]=json!(controller.telemetry_receiver_url);
                Ok(control)
            })();
            response["metrics_control"]=result.unwrap_or_else(|_|json!({"state":"error","reason":"invalid_metrics_control"}));
        }
        let receipt = api(controller.workspace_barrier())?;
        (response, receipt)
    };
    if let Some(receipt) = receipt {
        tokio::task::spawn_blocking(move || receipt.wait()).await
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?;
    }
    Ok(Json(response))
}
async fn agent_report(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let (response, receipt) = {
        let mut controller = controller.lock().unwrap();
        let response = api(controller.report_command(&body, bearer(&headers)))?;
        let receipt = api(controller.workspace_barrier())?;
        (response, receipt)
    };
    if let Some(receipt) = receipt {
        tokio::task::spawn_blocking(move || receipt.wait()).await
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?;
    }
    Ok(Json(response))
}

async fn workspace_status(State(controller): State<SharedController>, headers: HeaderMap,
    Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    let (response,receipt)={
        let controller=controller.lock().unwrap();
        let response=api(controller.workspace_status(&id,bearer(&headers)))?;
        let receipt=if response.get("operation").is_some(){api(controller.workspace_barrier())?}else{None};
        (response,receipt)
    };
    if let Some(receipt)=receipt {
        tokio::task::spawn_blocking(move||receipt.wait()).await
            .map_err(|e|ApiError(ControllerError::internal(e.to_string())))?
            .map_err(|e|ApiError(ControllerError::internal(e.to_string())))?;
    }
    Ok(Json(response))
}

async fn workspace_operation(State(controller): State<SharedController>, headers: HeaderMap,
    Path((id, action)): Path<(String, String)>, body: axum::body::Bytes) -> Result<Json<Value>, ApiError> {
    let parameters:Value=if body.is_empty(){json!({})}else{serde_json::from_slice(&body).map_err(|e|ApiError(ControllerError::bad(e.to_string())))?};
    if !parameters.is_object() || parameters.as_object().unwrap().keys().any(|k|
        if action=="set" {k!="auto_reclaim"} else {!matches!(k.as_str(),"target_bytes"|"operation_id")} ) {
        return Err(ApiError(ControllerError::bad("invalid workspace operation body")));
    }
    let (response, receipt) = {
        let mut controller = controller.lock().unwrap();
        let response = api(controller.workspace_operation(&id, &action, bearer(&headers), &parameters))?;
        let receipt = api(controller.workspace_barrier())?;
        (response, receipt)
    };
    if let Some(receipt) = receipt {
        tokio::task::spawn_blocking(move || receipt.wait()).await
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?;
    }
    Ok(Json(response))
}
async fn drain_allocation(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let id = body
        .get("allocation_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError(ControllerError::bad("allocation_id is required")))?;
    let reason = body
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("allocation_drain");
    Ok(Json(api(controller.lock().unwrap().drain_allocation(
        id,
        reason,
        bearer(&headers),
    ))?))
}
async fn remove_allocation(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let id = body
        .get("allocation_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError(ControllerError::bad("allocation_id is required")))?;
    Ok(Json(api(controller.lock().unwrap().remove_allocation(
        id,
        body.get("force").and_then(Value::as_bool).unwrap_or(false),
        bearer(&headers),
    ))?))
}
async fn drain_node(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let id = body
        .get("node_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError(ControllerError::bad("node_id is required")))?;
    let reason = body
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("node_drain");
    Ok(Json(api(controller.lock().unwrap().drain_node(
        id,
        reason,
        bearer(&headers),
    ))?))
}
#[derive(Deserialize)]
struct WorkspaceConfigQuery {
    pool: String,
    node_id: Option<String>,
}

async fn get_workspace_config(
    State(controller): State<SharedController>, headers: HeaderMap,
    Query(query): Query<WorkspaceConfigQuery>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(api(controller.lock().unwrap().get_workspace_config(
        &query.pool, query.node_id.as_deref(), bearer(&headers),
    ))?))
}

async fn set_workspace_config(
    State(controller): State<SharedController>, headers: HeaderMap, Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let pool = body.get("pool").and_then(Value::as_str)
        .ok_or_else(|| ApiError(ControllerError::bad("pool must be a nonempty string")))?;
    let node = match body.get("node_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        _ => return Err(ApiError(ControllerError::bad("node_id must be a nonempty string"))),
    };
    let changes = body.get("changes").and_then(Value::as_object)
        .ok_or_else(|| ApiError(ControllerError::bad("workspace configuration overrides must be an object")))?;
    let (value, receipt) = api(controller.lock().unwrap().set_workspace_config_unflushed(pool, changes, node, bearer(&headers)))?;
    if let Some(receipt) = receipt {
        tokio::task::spawn_blocking(move || receipt.wait()).await
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?
            .map_err(|error| ApiError(ControllerError::internal(error.to_string())))?;
    }
    Ok(Json(value))
}

async fn set_pool_config(
    State(controller): State<SharedController>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let pool = body
        .get("pool")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError(ControllerError::bad("pool is required")))?;
    let config=serde_json::from_value(json!({"pool_mode":body.get("pool_mode").and_then(Value::as_str).unwrap_or("exclusive"),"gpu_slots":body.get("gpu_slots").and_then(Value::as_u64).unwrap_or(1),"placement":body.get("placement").and_then(Value::as_str).unwrap_or("pack")})).map_err(|error|ApiError(ControllerError::bad(error.to_string())))?;
    Ok(Json(api(controller.lock().unwrap().set_pool_config(
        pool,
        config,
        bearer(&headers),
    ))?))
}

async fn management_capabilities(State(controller): State<SharedController>) -> Json<Value> {
    Json(crate::management::capabilities(&controller.lock().unwrap()))
}
async fn management_call(State(controller): State<SharedController>, headers: HeaderMap,
    Path(action): Path<String>, Json(body): Json<Value>) -> Result<Json<Value>, ApiError> {
    Ok(Json(identity_result(crate::management::call(&mut controller.lock().unwrap(), &action, bearer(&headers).unwrap_or(""), &body))?))
}
