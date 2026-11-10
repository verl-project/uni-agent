// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::workspace::WorkspaceConfig;
use crate::workspace_fs::{probe_configuration, WorkspaceManager};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;

pub type SharedWorkspaces = Arc<Mutex<AgentWorkspaces>>;
type Manager = Arc<Mutex<WorkspaceManager>>;

#[cfg(test)]
mod maintenance_tests {
    use super::*;

    #[test]
    fn cleanup_projection_preserves_runtime_identity_for_resume() {
        let spec=json!({"runtime_adapter":"local-process","image":"image","isolation_policy":"none","env":{"secret":"not-retained"}});
        let projected=cleanup_spec(&spec);
        for field in ["runtime_adapter","image","isolation_policy"] {assert_eq!(projected[field],spec[field]);}
        assert!(projected.get("env").is_none());
    }
    #[test]
    fn old_ack_does_not_drop_new_same_revision_sample() {
        let mut state = AgentWorkspaces::default();
        state.updates.insert(
            "ws".into(),
            json!({"execution_session_id":"session","revision":8,"telemetry_id":"new"}),
        );
        state.acknowledge(&json!({"workspace_acks":{"ws":{"execution_session_id":"session","revision":8,"telemetry_id":"old"}}}));
        assert!(state.updates.contains_key("ws"));
        state.acknowledge(&json!({"workspace_acks":{"ws":{"execution_session_id":"session","revision":8,"telemetry_id":"new"}}}));
        assert!(!state.updates.contains_key("ws"));
    }
    #[test]
    fn completed_receipt_fences_delayed_operation_delivery() {
        let shared = Arc::new(Mutex::new(AgentWorkspaces::default()));
        shared.lock().unwrap().operation_receipts.insert(
            "ws".into(),
            json!({"sequence":2,"id":"latest","update":{"revision":10}}),
        );
        let runtime = crate::runtime::RuntimeConfig::test_config("local-process");
        response(
            &shared,
            &json!({"workspace_operations":[{"id":"ws","operation":{"id":"old","sequence":1}}]}),
            "node",
            "pool",
            &runtime,
        )
        .unwrap();
        assert!(shared.lock().unwrap().updates.is_empty());
        response(
            &shared,
            &json!({"workspace_operations":[{"id":"ws","operation":{"id":"latest","sequence":2}}]}),
            "node",
            "pool",
            &runtime,
        )
        .unwrap();
        assert_eq!(shared.lock().unwrap().updates["ws"], json!({"revision":10}));
    }
}

pub struct AgentWorkspaces {
    pub config_status: Value,
    pub contexts: BTreeMap<String, Value>,
    managers: BTreeMap<String, Manager>,
    updates: BTreeMap<String, Value>,
    operation_receipts: BTreeMap<String, Value>,
    bindings: BTreeMap<String, String>,
    acknowledged: BTreeMap<String, Value>,
    attempted_config: Option<Value>,
    probe_running: bool,
    probe_task: Option<JoinHandle<()>>,
    stopping: bool,
    operations_running: BTreeSet<String>,
    operation_tasks: Vec<JoinHandle<()>>,
    pub maintenance_sessions: BTreeSet<String>,
    pub stop_requested: BTreeSet<String>,
    sampling: bool,
}

impl Default for AgentWorkspaces {
    fn default() -> Self {
        Self {
            config_status: json!({"state":"pending","applied_version":null,"error":null,"protocol":1}),
            contexts: BTreeMap::new(),
            managers: BTreeMap::new(),
            updates: BTreeMap::new(),
            operation_receipts: BTreeMap::new(),
            bindings: BTreeMap::new(),
            acknowledged: BTreeMap::new(),
            attempted_config: None,
            probe_running: false,
            probe_task: None,
            stopping: false,
            operations_running: BTreeSet::new(),
            operation_tasks: Vec::new(),
            maintenance_sessions: BTreeSet::new(),
            stop_requested: BTreeSet::new(),
            sampling: false,
        }
    }
}

impl AgentWorkspaces {
    pub fn restore(value: &Value) -> Self {
        let mut state = Self::default();
        state.contexts = serde_json::from_value(
            value
                .get("session_contexts")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )
        .unwrap_or_default();
        state.updates = serde_json::from_value(
            value
                .get("workspace_updates")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )
        .unwrap_or_default();
        state.operation_receipts = serde_json::from_value(
            value
                .get("workspace_operation_receipts")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )
        .unwrap_or_default();
        state.bindings = serde_json::from_value(
            value
                .get("workspace_bindings")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )
        .unwrap_or_default();
        state.acknowledged = serde_json::from_value(
            value
                .get("workspace_acknowledged")
                .cloned()
                .unwrap_or_else(|| json!({})),
        )
        .unwrap_or_default();
        state
    }

    pub fn persist(&self) -> Value {
        json!({"session_contexts":self.contexts,"workspace_updates":self.updates,
               "workspace_operation_receipts":self.operation_receipts,
               "workspace_bindings":self.bindings,"workspace_acknowledged":self.acknowledged})
    }

    pub fn pending(&self) -> Result<Value> {
        let mut entries = Vec::new();
        let mut size = 0;
        for update in self.updates.values() {
            let bytes = serde_json::to_vec(update)?.len();
            if !entries.is_empty() && (entries.len() >= 64 || size + bytes > 512 * 1024) {
                break;
            }
            if bytes > 512 * 1024 {
                bail!("workspace update exceeds the bounded reporting budget");
            }
            entries.push(update.clone());
            size += bytes;
        }
        Ok(Value::Array(entries))
    }

    pub fn reclaim_ready(&self) -> bool {
        !self.stopping && !self.probe_running && self.operations_running.is_empty()
    }

    pub fn begin_shutdown(&mut self) -> Option<JoinHandle<()>> {
        self.stopping = true;
        self.probe_task.take()
    }

    pub fn take_operation_tasks(&mut self) -> Vec<JoinHandle<()>> {
        std::mem::take(&mut self.operation_tasks)
    }

    fn acknowledge(&mut self, response: &Value) {
        if let Some(acks) = response.get("workspace_acks").and_then(Value::as_object) {
            for (id, ack) in acks {
                if self
                    .updates
                    .get(id)
                    .map(|pending| {
                        pending["execution_session_id"] == ack["execution_session_id"]
                            && pending["telemetry_id"] == ack["telemetry_id"]
                            && pending["operation_result"]["id"] == ack["operation_id"]
                            && pending["revision"].as_u64().unwrap_or(0)
                                <= ack["revision"].as_u64().unwrap_or(0)
                    })
                    .unwrap_or(false)
                {
                    self.updates.remove(id);
                    self.acknowledged.insert(id.clone(), ack.clone());
                }
            }
        }
    }
}

pub async fn configure(
    shared: &SharedWorkspaces,
    values: Value,
    node_id: &str,
    pool: &str,
) -> Result<()> {
    let config = WorkspaceConfig::from_value(values.clone())?;
    let node = node_id.to_string();
    let pool = pool.to_string();
    shared.lock().unwrap().attempted_config = Some(values);
    let version = config.version;
    let outcome =
        tokio::task::spawn_blocking(move || probe_configuration(&config, &node, &pool)).await?;
    let mut state = shared.lock().unwrap();
    match outcome {
        Ok(mut status) => {
            status["desired_version"] = json!(version);
            state.config_status = status;
            Ok(())
        }
        Err(error) => {
            state.config_status["state"] = json!("failed");
            state.config_status["desired_version"] = json!(version);
            state.config_status["error"] = json!(error.to_string());
            Err(error)
        }
    }
}

pub fn response(
    shared: &SharedWorkspaces,
    response: &Value,
    node_id: &str,
    pool: &str,
    runtime: &crate::runtime::RuntimeConfig,
) -> Result<()> {
    let mut state = shared.lock().unwrap();
    state.acknowledge(response);
    if !state.stopping {
        for workspace in response
            .get("workspace_operations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let identity = workspace["id"].as_str().unwrap_or("");
            if let (Some(sequence), Some(receipt)) = (
                workspace["operation"]["sequence"].as_u64(),
                state.operation_receipts.get(identity).cloned(),
            ) {
                if sequence <= receipt["sequence"].as_u64().unwrap_or(0) {
                    if sequence == receipt["sequence"].as_u64().unwrap_or(0)
                        && workspace["operation"]["id"] == receipt["id"]
                    {
                        state
                            .updates
                            .entry(identity.into())
                            .or_insert(receipt["update"].clone());
                    }
                    continue;
                }
            }
            if state.probe_running || !state.operations_running.is_empty() {
                break;
            }
            let Some(id) = workspace["operation"]["id"].as_str() else {
                continue;
            };
            let identity = workspace["id"].as_str().unwrap_or("");
            if state.operations_running.contains(id)
                || state
                    .updates
                    .get(identity)
                    .map(|update| update["operation_result"]["id"] == id)
                    .unwrap_or(false)
            {
                continue;
            }
            state.operations_running.insert(id.into());
            let session = workspace["execution_session_id"].as_str().unwrap_or("").to_string();
            state.maintenance_sessions.insert(session.clone());
            let manager = state.managers.get(&session).cloned();
            let context = state.contexts.get(&session).cloned();
            let runtime = runtime.clone();
            let shared = shared.clone();
            let workspace = workspace.clone();
            let node = node_id.to_string();
            let pool = pool.to_string();
            state.operation_tasks.retain(|task| !task.is_finished());
            state.operation_tasks.push(tokio::spawn(async move {
                let work = workspace.clone();
                let control=shared.clone();
                let outcome = tokio::task::spawn_blocking(move || maintenance(&work, &node, &pool, manager, context, &runtime, &control)).await;
                let update = match outcome {
                    Ok(update) => update,
                    Err(error) => { let mut update = workspace.clone(); update["workspace_id"] = workspace["id"].clone();
                        update["operation_result"] = json!({"id":workspace["operation"]["id"],"action":workspace["operation"]["action"],"state":"failed","error":error.to_string()}); update }
                };
                let mut state = shared.lock().unwrap();
                state.maintenance_sessions.remove(&session);
                if update["operation_result"]["state"]=="completed" && workspace["operation"]["boundary"]==true {
                    if let Some(context)=state.contexts.get_mut(&session) {
                        context["runtime_generation"]=json!(context["runtime_generation"].as_u64().unwrap_or(0)+1);
                    }
                }
                if let Some(id) = workspace["operation"]["id"].as_str() { state.operations_running.remove(id); }
                if let (Some(id),Some(sequence))=(workspace["id"].as_str(),workspace["operation"]["sequence"].as_u64()) {
                    state.operation_receipts.insert(id.into(),json!({"sequence":sequence,"id":workspace["operation"]["id"],"update":update.clone()}));
                }
                if let Some(id) = workspace["id"].as_str() { state.updates.insert(id.into(), update); }
            }));
        }
    }
    if let Some(desired) = response.get("workspace_config") {
        if !state.stopping
            && !state.probe_running
            && state.operations_running.is_empty()
            && state.attempted_config.as_ref() != Some(desired)
        {
            WorkspaceConfig::from_value(desired.clone())?;
            state.probe_running = true;
            state.attempted_config = Some(desired.clone());
            state.config_status["state"] = json!("applying");
            state.config_status["desired_version"] = desired["version"].clone();
            state.config_status["error"] = Value::Null;
            let shared = shared.clone();
            let desired = desired.clone();
            let node = node_id.to_string();
            let pool = pool.to_string();
            state.probe_task = Some(tokio::spawn(async move {
                if let Err(error) = configure(&shared, desired, &node, &pool).await {
                    eprintln!("[rscape-node {node}] workspace configuration failed: {error:#}");
                }
                shared.lock().unwrap().probe_running = false;
            }));
        }
    }
    Ok(())
}

fn maintenance(
    workspace: &Value,
    node: &str,
    pool: &str,
    existing: Option<Manager>,
    context: Option<Value>,
    runtime: &crate::runtime::RuntimeConfig,
    control: &SharedWorkspaces,
) -> Value {
    let mut update = workspace.clone();
    update["workspace_id"] = workspace["id"].clone();
    let mut result = workspace["operation"].clone();
    result["state"] = json!("completed");
    result["error"] = Value::Null;
    let outcome = (|| -> Result<()> {
        let _priority = if workspace["operation"]["action"]=="reclaim" {
            let policy=crate::workspace_fs::MaintenancePolicy::from_value(workspace["operation"].get("policy").cloned().unwrap_or(json!({})))?;
            Some(crate::workspace_fs::MaintenancePriority::enter(policy.reclaim_nice)?)
        } else {None};
        if workspace["node_id"] != node || workspace["pool"] != pool {
            bail!("workspace maintenance belongs to another node or pool");
        }
        let mut fresh = WorkspaceManager::new(
            WorkspaceConfig::from_value(workspace["config"].clone())?,
            node.into(),
            pool.into(),
        )?;
        let id = workspace["id"].as_str().context("workspace id")?;
        let user = workspace["user_id"].as_str().context("workspace user")?;
        let boundary = workspace["operation"]["boundary"] == true;
        let session = workspace["execution_session_id"]
            .as_str()
            .context("workspace session")?;
        if boundary && (existing.is_none() || context.is_none()) {
            bail!("boundary maintenance requires the live owning agent and runtime");
        }
        let mut owned = if boundary {
            existing.as_ref().map(|m| m.lock().unwrap())
        } else {
            None
        };
        let manager = if let Some(ref mut locked) = owned {
            &mut **locked
        } else {
            &mut fresh
        };
        if boundary {
            crate::runtime::workspace_transition(
                runtime,
                session,
                context.as_ref().unwrap(),
                false,
            )?;
            if let Err(error) = manager.release(id) {
                let observed = manager.inspect(id, user)?;
                if workspace["operation"]["action"] != "recover"
                    || observed["mounted"] == true
                    || !matches!(
                        observed["state"].as_str(),
                        Some("needs_check" | "maintenance" | "needs_recovery")
                    )
                {
                    return Err(error);
                }
            }
        }
        let progress_control = control.clone();
        let progress_id = workspace["operation"]["id"].clone();
        let progress_workspace = id.to_string();
        let progress_session = session.to_string();
        manager.maintenance_observer = Some(Arc::new(move |manifest| {
            if let Ok(mut update) = serde_json::to_value(manifest) {
                update["state"] = json!("maintenance");
                update["execution_session_id"] = json!(progress_session);
                update["telemetry_id"] = json!(uuid::Uuid::new_v4().simple().to_string());
                update["operation_progress"] =
                    json!({"id":progress_id,"phase":update["maintenance"]["phase"]});
                progress_control
                    .lock()
                    .unwrap()
                    .updates
                    .insert(progress_workspace.clone(), update);
            }
        }));
        let output = match workspace["operation"]["action"].as_str() {
            Some("check") => manager.check(id, user),
            Some("delete") => manager.delete(id, user),
            Some(action @ ("grow" | "reclaim" | "shrink")) => {
                let policy = crate::workspace_fs::MaintenancePolicy::from_value(
                    workspace["operation"]
                        .get("policy")
                        .cloned()
                        .unwrap_or(json!({})),
                )?;
                let maximum = workspace["operation"]["max_image_size_bytes"]
                    .as_u64()
                    .or_else(|| workspace["config"]["max_image_size_bytes"].as_u64())
                    .context("capacity limit")?;
                let output = manager.maintain_with_limit(
                    id,
                    user,
                    action,
                    workspace["operation"]["target_bytes"].as_u64(),
                    workspace["operation"]["id"]
                        .as_str()
                        .context("operation id")?,
                    &policy,
                    maximum,
                );
                output.and_then(|completed| {
                    result["reclaimed_bytes"] = completed.get("reclaimed_bytes").cloned().unwrap_or(json!(0));
                    let value = manager.recover_maintenance(id, user)?;
                    result["details"] = value.clone();
                    Ok(value)
                })
            }
            Some("recover") => {
                let observed = manager.inspect(id, user)?;
                (if observed["maintenance"].is_object() {
                    manager.recover_maintenance(id, user)
                } else {
                    manager.check(id, user)
                })
                .map(|value| {
                    result["details"] = value.clone();
                    value
                })
            }
            _ => Err(anyhow::anyhow!("unknown workspace maintenance action")),
        };
        let output = output.and_then(|value| {
            let resume = {
                let state = control.lock().unwrap();
                !state.stopping && !state.stop_requested.contains(session)
            };
            if boundary && resume {
                manager.prepare(id, user, session, false)?;
                crate::runtime::workspace_transition(
                    runtime,
                    session,
                    context.as_ref().unwrap(),
                    true,
                )?;
            }
            Ok(value)
        });
        if let Ok(value) = manager.inspect(id, user) {
            update = value;
        }
        manager.maintenance_observer = None;
        if boundary {
            if let Some(path) = context.as_ref().and_then(|c| c.get("workspace_path")) {
                update["path"] = path.clone();
            }
        }
        let output = output?;
        result["exit_code"] = output.get("exit_code").cloned().unwrap_or(json!(0));
        if result["exit_code"] != 0 {
            bail!("offline filesystem check failed");
        }
        Ok(())
    })();
    if let Err(error) = outcome {
        result["state"] = json!("failed");
        result["error"] = json!(error.to_string());
        if let Some(existing) = existing {
            let mut manager = existing.lock().unwrap();
            manager.maintenance_observer = None;
            let id = workspace["id"].as_str().unwrap_or("");
            let user = workspace["user_id"].as_str().unwrap_or("");
            let session = workspace["execution_session_id"].as_str().unwrap_or("");
            if let Ok(observed) = manager.inspect(id, user) {
                if observed["state"] == "retained" {
                    let resume = {
                        let state = control.lock().unwrap();
                        !state.stopping && !state.stop_requested.contains(session)
                    };
                    if resume {
                        if let Some(context) = &context {
                            let _ = manager.prepare(id, user, session, false).and_then(|_| {
                                crate::runtime::workspace_transition(
                                    runtime, session, context, true,
                                )
                            });
                        }
                    }
                }
            }
            if let Ok(observed) = manager.inspect(id, user) {
                update = observed;
            }
        }
    }
    update["execution_session_id"] = workspace["execution_session_id"].clone();
    result["phase"] = json!(if result["state"] == "completed" {
        "completed"
    } else {
        "failed"
    });
    result["completed_at"] = json!(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64());
    update["operation_result"] = result;
    update
}

fn cleanup_spec(spec: &Value) -> Value {
    let mut result = serde_json::Map::new();
    for field in [
        "execution_session_id",
        "runtime_adapter",
        "image",
        "mounts",
        "cwd",
        "privileged",
        "cache_dir",
        "tmp_dir",
        "runtime_dir",
        "oci_bundle",
        "isolation_policy",
        "state_layer_id",
        "overlay_path",
        "overlay_upper",
        "overlay_work",
        "workspace",
    ] {
        if let Some(value) = spec.get(field) {
            result.insert(field.into(), value.clone());
        }
    }
    Value::Object(result)
}

pub async fn prepare(
    shared: &SharedWorkspaces,
    assignment: &mut Value,
    node_id: &str,
    pool: &str,
) -> Result<()> {
    let workspace = assignment["session_spec"]
        .get("workspace")
        .filter(|value| value.is_object())
        .context("assignment lacks a managed workspace; upgrade the controller protocol")?
        .clone();
    if workspace["node_id"] != node_id || workspace["pool"] != pool {
        bail!("workspace assignment belongs to another node or pool");
    }
    let id = workspace["id"]
        .as_str()
        .context("workspace id")?
        .to_string();
    let user = workspace["user_id"]
        .as_str()
        .context("workspace user")?
        .to_string();
    let session = assignment["execution_session_id"]
        .as_str()
        .context("execution_session_id")?
        .to_string();
    let adapter = assignment["session_spec"]["runtime_adapter"]
        .as_str()
        .unwrap_or("local-process")
        .to_string();
    let target = workspace["target"]
        .as_str()
        .unwrap_or("/workspace")
        .to_string();
    if adapter != "local-process"
        && assignment["session_spec"]["mounts"]
            .as_array()
            .map(|mounts| {
                mounts.iter().any(|mount| {
                    crate::workspace::workspace_bind_conflict(
                        mount["target"].as_str().unwrap_or("/"),
                        &target,
                    )
                })
            })
            .unwrap_or(false)
    {
        bail!(
            "managed workspace target conflicts with an explicit bind; select --workspace-target"
        );
    }
    let manager = {
        let mut state = shared.lock().unwrap();
        if let Some(previous) = state
            .contexts
            .get(&session)
            .and_then(|context| context["session_spec"].get("workspace"))
        {
            if !previous.is_null() && previous["id"] != id {
                bail!("session workspace identity changed");
            }
        }
        let manager = if let Some(manager) = state.managers.get(&session) {
            manager.clone()
        } else {
            let manager = Arc::new(Mutex::new(WorkspaceManager::new(
                WorkspaceConfig::from_value(workspace["config"].clone())?,
                node_id.into(),
                pool.into(),
            )?));
            state.managers.insert(session.clone(), manager.clone());
            manager
        };
        state.bindings.insert(id.clone(), session.clone());
        state.contexts.insert(session.clone(), json!({"runtime_adapter":adapter,"request_id":assignment["request_id"],
                                                    "session_spec":cleanup_spec(&assignment["session_spec"])}));
        manager
    };
    let work = manager.clone();
    let session_id = id.clone();
    let task_user = user.clone();
    let task_session = session.clone();
    let create = workspace["create"].as_bool().unwrap_or(false);
    let path = tokio::task::spawn_blocking(move || {
        work.lock()
            .unwrap()
            .prepare(&session_id, &task_user, &task_session, create)
    })
    .await??;
    let visible = if adapter == "local-process" {
        path.to_str()
            .context("workspace path is not UTF-8")?
            .to_string()
    } else {
        target.clone()
    };
    if adapter != "local-process" {
        let spec = assignment["session_spec"]
            .as_object_mut()
            .context("session_spec")?;
        let mounts = spec
            .entry("mounts")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .context("mounts must be an array")?;
        mounts.insert(0, json!({"source":path,"target":target,"read_only":false}));
    }
    for key in ["session_spec", "command_spec"] {
        let spec = assignment[key]
            .as_object_mut()
            .context("invalid assignment specification")?;
        let environment = spec.entry("env").or_insert_with(|| json!({}));
        if environment.is_null() {
            *environment = json!({});
        }
        environment
            .as_object_mut()
            .context("environment must be an object")?
            .insert("CAPE_WORKSPACE_DIR".into(), json!(visible));
        if spec.get("cwd").map(Value::is_null).unwrap_or(true) {
            spec.insert("cwd".into(), json!(visible));
        }
    }
    {
        let mut state = shared.lock().unwrap();
        let context = state
            .contexts
            .get_mut(&session)
            .context("session was stopped during workspace preparation")?;
        context["workspace_path"] = json!(visible);
        context["session_spec"] = cleanup_spec(&assignment["session_spec"]);
    }
    record(shared, &session, None).await?;
    Ok(())
}

pub async fn verify(shared: &SharedWorkspaces, session: &str) -> Result<()> {
    let (manager, id) = {
        let state = shared.lock().unwrap();
        let manager = state
            .managers
            .get(session)
            .context("workspace manager is unavailable")?
            .clone();
        let id = state.contexts[session]["session_spec"]["workspace"]["id"]
            .as_str()
            .context("workspace id")?
            .to_string();
        (manager, id)
    };
    tokio::task::spawn_blocking(move || manager.lock().unwrap().verify(&id)).await?
}

pub async fn unhealthy_sessions(shared: &SharedWorkspaces) -> Result<Vec<String>> {
    let managers: Vec<(String, Manager)> = shared
        .lock()
        .unwrap()
        .managers
        .iter()
        .map(|(session, manager)| (session.clone(), manager.clone()))
        .collect();
    let failed = tokio::task::spawn_blocking(move || {
        let mut failed = Vec::new();
        for (session, manager) in managers {
            if let Ok(mut manager) = manager.try_lock() {
                for identity in manager.mounted_ids() {
                    if let Err(error) = manager.verify(&identity) {
                        eprintln!("workspace health failure for {session}: {error:#}");
                        failed.push(session.clone());
                        break;
                    }
                }
            }
        }
        failed
    })
    .await?;
    // Launch sampling after the health worker releases its manager locks. Starting
    // both together can make every try_lock sample lose the same heartbeat race.
    start_sampler(shared);
    Ok(failed)
}

fn start_sampler(shared: &SharedWorkspaces) {
    let snapshots = {
        let mut state = shared.lock().unwrap();
        if state.stopping
            || state.sampling
            || state.config_status["config"]["maintenance"]["auto_grow"] != true
        {
            return;
        }
        state.sampling = true;
        state
            .managers
            .iter()
            .filter(|(session, _)| !state.maintenance_sessions.contains(*session))
            .filter_map(|(session, manager)| {
                state
                    .contexts
                    .get(session)
                    .map(|c| (session.clone(), manager.clone(), c.clone()))
            })
            .collect::<Vec<_>>()
    };
    let shared = shared.clone();
    tokio::spawn(async move {
        let updates = tokio::task::spawn_blocking(move || {
            let mut updates = Vec::new();
            for (session, manager, context) in snapshots {
                let workspace = &context["session_spec"]["workspace"];
                if let (Some(id), Some(user), Ok(manager)) = (
                    workspace["id"].as_str(),
                    workspace["user_id"].as_str(),
                    manager.try_lock(),
                ) {
                    if let Ok(mut update) = manager.inspect(id, user) {
                        update["execution_session_id"] = json!(session);
                        if !context["workspace_path"].is_null() {
                            update["path"] = context["workspace_path"].clone();
                        }
                        updates.push((id.to_string(), update));
                    }
                }
            }
            updates
        })
        .await;
        let mut state = shared.lock().unwrap();
        state.sampling = false;
        if let Ok(updates) = updates {
            for (id, mut update) in updates {
                if state
                    .bindings
                    .get(&id)
                    .map(|s| Some(s.as_str()) != update["execution_session_id"].as_str())
                    .unwrap_or(false)
                {
                    continue;
                }
                if let Some(previous) = state.updates.get(&id) {
                    if previous["revision"].as_u64() > update["revision"].as_u64() {
                        continue;
                    }
                    if previous.get("operation_result").is_some() {
                        update["operation_result"] = previous["operation_result"].clone();
                    }
                }
                update["telemetry_id"] = json!(uuid::Uuid::new_v4().simple().to_string());
                state.updates.insert(id, update);
            }
        }
    });
}

pub async fn record(shared: &SharedWorkspaces, session: &str, error: Option<String>) -> Result<()> {
    let (context, manager) = {
        let state = shared.lock().unwrap();
        let Some(context) = state.contexts.get(session).cloned() else {
            return Ok(());
        };
        (context, state.managers.get(session).cloned())
    };
    let workspace = context["session_spec"]["workspace"].clone();
    let Some(id) = workspace["id"].as_str() else {
        return Ok(());
    };
    let user = workspace["user_id"]
        .as_str()
        .context("workspace user")?
        .to_string();
    let identity = id.to_string();
    let inspected = if let Some(manager) = manager {
        tokio::task::spawn_blocking(move || manager.lock().unwrap().inspect(&identity, &user))
            .await?
            .ok()
    } else {
        None
    };
    let mut update = inspected.unwrap_or_else(|| json!({"workspace_id":id,"user_id":workspace["user_id"],
        "config":workspace["config"],"state":"needs_check","revision":workspace.get("revision").cloned().unwrap_or(json!(0)),
        "path":null,"last_error":error.clone().unwrap_or_else(|| "workspace preparation did not complete".into())}));
    update["execution_session_id"] = json!(session);
    if error.is_some()
        && matches!(
            update["state"].as_str(),
            Some("needs_check" | "creating" | "mounting")
        )
    {
        update["last_error"] = json!(error);
    }
    if !update["path"].is_null() && !context["workspace_path"].is_null() {
        update["path"] = context["workspace_path"].clone();
    }
    let mut state = shared.lock().unwrap();
    update["maintenance_deferred"] = json!(state.stopping);
    if state.config_status["config"]["maintenance"]["auto_grow"] == true {
        update["telemetry_id"] = json!(uuid::Uuid::new_v4().simple().to_string());
    }
    if let Some(previous) = state.updates.get(id) {
        if previous.get("operation_result").is_some() && update.get("operation_result").is_none() {
            update["operation_result"] = previous["operation_result"].clone();
        }
    }
    if state
        .bindings
        .get(id)
        .map(|owner| owner != session)
        .unwrap_or(false)
    {
        return Ok(());
    }
    if error.is_none()
        && update.get("space").is_none()
        && state
            .acknowledged
            .get(id)
            .map(|ack| {
                ack["execution_session_id"] == session
                    && ack["revision"].as_u64().unwrap_or(0)
                        >= update["revision"].as_u64().unwrap_or(0)
            })
            .unwrap_or(false)
    {
        return Ok(());
    }
    if state
        .updates
        .get(id)
        .map(|old| {
            old["execution_session_id"] != session
                || old["revision"].as_u64().unwrap_or(0) <= update["revision"].as_u64().unwrap_or(0)
        })
        .unwrap_or(true)
    {
        state.updates.insert(id.into(), update);
    }
    Ok(())
}

pub async fn release(shared: &SharedWorkspaces, session: &str) -> Result<()> {
    let (context, manager) = {
        let state = shared.lock().unwrap();
        let Some(context) = state.contexts.get(session).cloned() else {
            return Ok(());
        };
        (context, state.managers.get(session).cloned())
    };
    let workspace = &context["session_spec"]["workspace"];
    if workspace.is_null() {
        shared.lock().unwrap().contexts.remove(session);
        return Ok(());
    }
    let id = workspace["id"]
        .as_str()
        .context("workspace id")?
        .to_string();
    let recovered = manager.is_none();
    let manager = match manager {
        Some(manager) => manager,
        None => {
            let manager = Arc::new(Mutex::new(WorkspaceManager::new(
                WorkspaceConfig::from_value(workspace["config"].clone())?,
                workspace["node_id"]
                    .as_str()
                    .context("workspace node")?
                    .into(),
                workspace["pool"].as_str().context("workspace pool")?.into(),
            )?));
            shared
                .lock()
                .unwrap()
                .managers
                .insert(session.into(), manager.clone());
            manager
        }
    };
    let user = workspace["user_id"]
        .as_str()
        .context("workspace user")?
        .to_string();
    let outcome = tokio::task::spawn_blocking(move || {
        let mut manager = manager.lock().unwrap();
        if recovered {
            let checked = manager
                .check(&id, &user)
                .context("workspace ownership requires reconciliation after agent restart")?;
            if checked["exit_code"] != 0 {
                bail!("recovered workspace requires offline maintenance");
            }
            Ok(())
        } else if manager.mounted_ids().contains(&id) {
            manager.release(&id)
        } else {
            manager.reconcile_released(&id, &user)
        }
    })
    .await?;
    record(
        shared,
        session,
        outcome.as_ref().err().map(|error| error.to_string()),
    )
    .await?;
    if recovered && outcome.is_err() {
        shared.lock().unwrap().managers.remove(session);
    }
    outcome?;
    let mut state = shared.lock().unwrap();
    state.contexts.remove(session);
    state.managers.remove(session);
    Ok(())
}
