// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::runtime::{self, execute, write_log, RuntimeConfig};
use crate::workspace_agent::{self, AgentWorkspaces, SharedWorkspaces};
use anyhow::{bail, Context, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub metrics_enabled: bool,
    pub controller_url: String,
    pub node_id: String,
    pub agent_epoch: Option<String>,
    pub pool: String,
    pub pool_mode: String,
    pub gpu_slots: usize,
    pub placement: String,
    pub cpu_cores: f64,
    pub memory_gb: f64,
    pub gpu_uuids: Vec<String>,
    pub allocation_id: Option<String>,
    pub allocation_deadline: Option<f64>,
    pub drain_before_sec: f64,
    pub cleanup_buffer_sec: f64,
    pub runtime_cache_config: String,
    pub cached_images: Vec<String>,
    pub join_token: Option<String>,
    pub token: Option<String>,
    pub heartbeat_interval: f64,
    pub once: bool,
    pub exit_on_removal: bool,
    pub state_path: PathBuf,
    pub runtime: RuntimeConfig,
}

#[derive(Default)]
struct LocalState {
    heartbeat_changed: Arc<tokio::sync::Notify>,
    suspended_contexts: BTreeMap<String, Value>,
    resumed_versions: BTreeMap<String, u64>,
    purged_executions: BTreeSet<String>,
    stop_versions: BTreeMap<String, u64>,
    sessions: BTreeSet<String>,
    running: BTreeMap<String, (String, String)>,
    pending_reports: BTreeSet<String>,
    report_retries_exhausted: bool,
    workspaces: SharedWorkspaces,
    pending_stops: BTreeSet<String>,
    closed_sessions: BTreeSet<String>,
    stopping: bool,
    unreconciled_commands: bool,
}

const REPORT_FALLBACK_ATTEMPTS: usize = 3;

#[derive(Debug)]
struct AgentHttpError {
    status: reqwest::StatusCode,
    body: Value,
}

impl std::fmt::Display for AgentHttpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "CAPE HTTP {}: {}", self.status, self.body)
    }
}

impl std::error::Error for AgentHttpError {}

fn report_error_status(error: &anyhow::Error) -> Option<u16> {
    error.downcast_ref::<AgentHttpError>().map(|error| error.status.as_u16())
}

fn retryable_report_error(error: &anyhow::Error) -> bool {
    if let Some(status) = report_error_status(error) {
        return (500..600).contains(&status);
    }
    error.downcast_ref::<reqwest::Error>().map(|error| {
        !error.is_builder() && !error.is_decode()
            && (error.is_timeout() || error.is_connect() || error.is_request() || error.is_body())
    }).unwrap_or(false)
}

pub async fn run(mut config: AgentConfig) -> Result<()> {
    let client = Client::builder().timeout(Duration::from_secs(30)).build()?;
    let epoch = config.agent_epoch.clone().unwrap_or_else(|| Uuid::new_v4().simple().to_string());
    let metrics_link=if config.metrics_enabled {
        crate::telemetry_link::AgentLink::start(json!({"controller_url":config.controller_url,"node_id":config.node_id,
            "agent_epoch":epoch,"pool_id":config.pool,"gpu_uuids":config.gpu_uuids,"token":config.token,"parent_pid":std::process::id()})).ok()
    }else{None};
    let state = Arc::new(Mutex::new(load_state(&config.state_path)?));
    if state.lock().unwrap().unreconciled_commands {
        bail!("agent state contains interrupted workspace commands; fence predecessor process groups before restart");
    }
    let announcement = announcement_payload(&config, &epoch);
    let announcement_response = post(&client, &config, "/v1/agent/announce", &announcement).await?;
    let workspaces = state.lock().unwrap().workspaces.clone();
    let probe_config = config.clone();
    let probe_client = client.clone();
    let probe_epoch = epoch.clone();
    let probe_complete = Arc::new(AtomicBool::new(false));
    let probe_complete_task = probe_complete.clone();
    let probe_heartbeat = tokio::spawn(async move {
        while !probe_complete_task.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_secs_f64(
                probe_config.heartbeat_interval.max(0.01),
            ))
            .await;
            if probe_complete_task.load(Ordering::SeqCst) {
                break;
            }
            let payload = json!({"node_id":probe_config.node_id,"agent_epoch":probe_epoch,"state":"REGISTERING","status":{"authoritative":false,"phase":"runtime_probe"},"max_commands":0});
            let _ = post(
                &probe_client,
                &probe_config,
                "/v1/agent/heartbeat",
                &payload,
            )
            .await;
        }
    });
    let probe_result = async {
        runtime::probe(&mut config.runtime, &config.gpu_uuids).await?;
        workspace_agent::configure(&workspaces, announcement_response["workspace_config"].clone(), &config.node_id, &config.pool).await
    }.await;
    probe_complete.store(true, Ordering::SeqCst);
    probe_heartbeat.abort();
    if let Err(error) = probe_result {
        let payload = json!({"node_id":config.node_id,"agent_epoch":epoch,"state":"UNHEALTHY","status":{"authoritative":false,"phase":"runtime_probe_failed","error":error.to_string()},"max_commands":0});
        let _ = post(&client, &config, "/v1/agent/heartbeat", &payload).await;
        return Err(error);
    }
    eprintln!(
        "[rscape-node {}] proven runtime capabilities: {}",
        config.node_id,
        serde_json::to_string(&config.runtime.capabilities)?
    );
    let mut registration = registration_payload(&config, &epoch);
    registration["workspace_config"] = workspaces.lock().unwrap().config_status.clone();
    let registration_response = register_until_available(&client, &config, &registration).await;
    if shutdown_control(&registration_response, &config).is_some() && config.exit_on_removal {
        runtime::terminate_all(&config.runtime);
        return Ok(());
    }
    let mut tasks: Vec<JoinHandle<()>> = Vec::new();
    let terminate_requested = Arc::new(AtomicBool::new(false));
    let terminate_flag = terminate_requested.clone();
    let signal_task = tokio::spawn(async move {
        #[cfg(unix)]
        {
            if let Ok(mut signal) =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            {
                signal.recv().await;
                terminate_flag.store(true, Ordering::SeqCst);
            }
        }
    });
    let mut drain_announced = false;
    loop {
        tasks.retain(|task| !task.is_finished());
        for session in workspace_agent::unhealthy_sessions(&workspaces).await? {
            state.lock().unwrap().pending_stops.insert(session);
        }
        let pending_stops: Vec<String> = state.lock().unwrap().pending_stops.iter().cloned().collect();
        for session in pending_stops {
            if let Err(error) = stop_local_session(&config, &state, &session).await {
                eprintln!("[rscape-node {}] workspace cleanup pending for {session}: {error:#}", config.node_id);
            }
        }
        if state.lock().unwrap().report_retries_exhausted {
            eprintln!("[rscape-node {}] report_retry_exhausted; stopping heartbeats and local work", config.node_id);
            runtime::terminate_all(&config.runtime);
            break;
        }
        if terminate_requested.load(Ordering::SeqCst) && !drain_announced {
            match post(
                &client,
                &config,
                "/v1/admin/drain-node",
                &json!({"node_id":config.node_id,"reason":"operator_drain"}),
            )
            .await
            {
                Ok(_) => {
                    drain_announced = true;
                    eprintln!("[rscape-node {}] SIGTERM drain acknowledged; waiting for active work and cleanup",config.node_id);
                }
                Err(error) => eprintln!(
                    "[rscape-node {}] SIGTERM drain notification failed; retrying: {error:#}",
                    config.node_id
                ),
            }
        }
        let snapshot = state.lock().unwrap();
        let running_sessions: BTreeSet<String> = snapshot
            .running
            .values()
            .map(|value| value.0.clone())
            .collect();
        let mut sessions: Vec<Value> = snapshot
            .sessions
            .iter()
            .filter(|id| !running_sessions.contains(*id))
            .map(|id| json!({"execution_session_id":id,"state":"CACHED_IDLE"}))
            .collect();
        sessions.extend(snapshot.running.values().map(|(execution_session_id, request_id)| {
            json!({"execution_session_id":execution_session_id,"state":"RUNNING_COMMAND","active_request_id":request_id})
        }));
        let command_ids: BTreeSet<String> = snapshot.running.keys().cloned()
            .chain(snapshot.pending_reports.iter().cloned()).collect();
        drop(snapshot);
        let suspended: BTreeMap<String, Value> = state.lock().unwrap().suspended_contexts.iter()
            .map(|(id, context)|(id.clone(),context["suspend_version"].clone())).collect();
        let mut status = json!({"authoritative":true,"sessions":sessions,"command_ids":command_ids,"suspended_sessions":suspended});
        status["purged_executions"]=json!(state.lock().unwrap().purged_executions);
        status["workspace_config"] = workspaces.lock().unwrap().config_status.clone();
        status["workspace_updates"] = workspaces.lock().unwrap().pending()?;
        status["reclaim_ready"] = json!(command_ids.is_empty() && workspaces.lock().unwrap().reclaim_ready());
        if let Some(link)=metrics_link.as_ref(){
            let metric=link.status.lock().unwrap().clone();
            status["metrics"]=metric;
        } else if config.metrics_enabled {
            status["metrics"]=json!({"state":"error","reason":"metrics_link_unavailable"});
        }
        let heartbeat =
            json!({"node_id":config.node_id,"agent_epoch":epoch,"status":status,"max_commands":8});
        let response = match post(&client, &config, "/v1/agent/heartbeat", &heartbeat).await {
            Ok(response) => response,
            Err(error) => {
                eprintln!(
                    "[rscape-node {}] heartbeat failure; retrying: {error:#}",
                    config.node_id
                );
                if error.to_string().contains("CAPE HTTP 404") {
                    let response = register_until_available(&client, &config, &registration).await;
                    if shutdown_control(&response, &config).is_some() && config.exit_on_removal {
                        runtime::terminate_all(&config.runtime);
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_secs_f64(config.heartbeat_interval.max(0.01)))
                    .await;
                continue;
            }
        };
        prepare_received_response(&config, &state, &workspaces, &response)?;
        if let Some(control) = shutdown_control(&response, &config) {
            if config.exit_on_removal {
                runtime::terminate_all(&config.runtime);
                break;
            }
            eprintln!("[rscape-node {}] authoritative removal received but --stay-after-removal is active: {}",config.node_id,control.get("reason").and_then(Value::as_str).unwrap_or("unknown"));
        }
        for command in response
            .get("commands")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            match command
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("run")
            {
                "purge_session" => {
                    if let Err(error)=purge_local_session(&config,&state,&command) {eprintln!("execution purge deferred: {error:#}");}
                }
                "suspend_session" => {
                    if let Err(error) = suspend_local_session(&config, &state, &command).await {
                        eprintln!("execution suspension deferred: {error:#}");
                    }
                }
                "stop_session" => {
                    if let Some(id) = command.get("execution_session_id").and_then(Value::as_str) {
                        if let Err(error) = stop_command(&config, &state, &command).await {
                            eprintln!("[rscape-node {}] session cleanup deferred for {id}: {error:#}", config.node_id);
                        }
                    }
                }
                "cancel" => {
                    if let Some(id) = command.get("command_id").and_then(Value::as_str) {
                        runtime::cancel_command(&config.runtime, id);
                    }
                }
                _ => {
                    let config = config.clone();
                    let client = client.clone();
                    let state = state.clone();
                    let epoch = epoch.clone();
                    let node_id = config.node_id.clone();
                    tasks.push(tokio::spawn(async move {
                        if let Err(error) =
                            execute_and_report(client, config, state, epoch, command).await
                        {
                            eprintln!("[rscape-node {node_id}] assignment failed: {error:#}");
                        }
                    }));
                }
            }
        }
        if drain_announced && tasks.is_empty() && state.lock().unwrap().sessions.is_empty() {
            eprintln!(
                "[rscape-node {}] SIGTERM drain complete; node enroll exits 0",
                config.node_id
            );
            break;
        }
        if config.once {
            break;
        }
        let changed = state.lock().unwrap().heartbeat_changed.clone();
        tokio::select! {
            _ = changed.notified() => {},
            _ = tokio::time::sleep(Duration::from_secs_f64(config.heartbeat_interval.max(0.01))) => {},
        }
    }
    if !config.once || state.lock().unwrap().report_retries_exhausted {
        let commands = {
            let mut state = state.lock().unwrap();
            state.stopping = true;
            state.running.keys().cloned().collect::<Vec<_>>()
        };
        for command in commands { runtime::cancel_command(&config.runtime, &command); }
    }
    for task in tasks {
        let _ = task.await;
    }
    signal_task.abort();
    let probe_task = workspaces.lock().unwrap().begin_shutdown();
    if let Some(task) = probe_task { let _ = task.await; }
    let operation_tasks = workspaces.lock().unwrap().take_operation_tasks();
    for task in operation_tasks { let _ = task.await; }
    let sessions: Vec<String> = state.lock().unwrap().sessions.iter().cloned().collect();
    for execution_session_id in sessions {
        stop_local_session(&config, &state, &execution_session_id).await?;
    }
    state.lock().unwrap().sessions.clear();
    state.lock().unwrap().running.clear();
    save_state(&config.state_path, &state.lock().unwrap())?;
    if state.lock().unwrap().report_retries_exhausted {
        bail!("report_retry_exhausted; agent stopped");
    }
    let updates = workspaces.lock().unwrap().pending()?;
    if updates.as_array().map(|items| !items.is_empty()).unwrap_or(false) {
        let payload = json!({"node_id":config.node_id,"agent_epoch":epoch,"status":{"authoritative":true,"sessions":[],"workspace_updates":updates},"max_commands":0});
        let response = post(&client, &config, "/v1/agent/heartbeat", &payload).await?;
        prepare_received_response(&config, &state, &workspaces, &response)?;
        save_state(&config.state_path, &state.lock().unwrap())?;
    }
    Ok(())
}

fn reserve_received_commands(config: &AgentConfig, state: &Arc<Mutex<LocalState>>, response: &Value) -> Result<Vec<String>> {
    if shutdown_control(response, config).is_some() { return Ok(Vec::new()); }
    let mut state = state.lock().unwrap();
    if state.stopping || state.report_retries_exhausted { return Ok(Vec::new()); }
    let mut reserved = Vec::new();
    if let Some(commands) = response.get("commands").and_then(Value::as_array) {
        for command in commands {
            if command.get("action").and_then(Value::as_str).unwrap_or("run") != "run" { continue; }
            if let (Some(id), Some(session), Some(request)) = (
                command["command_id"].as_str(), command["execution_session_id"].as_str(), command["request_id"].as_str()) {
                let can_resume = state.suspended_contexts.get(session).map(|context|
                    context["suspend_version"] == command["session_spec"]["suspend_version"]).unwrap_or(false);
                if (!state.closed_sessions.contains(session) || can_resume) && !state.running.contains_key(id) {
                    state.sessions.insert(session.into());
                    state.running.insert(id.into(), (session.into(), request.into()));
                    reserved.push(id.to_owned());
                }
            }
        }
    }
    if !reserved.is_empty() {
        if let Err(error) = save_state(&config.state_path, &state) {
            for id in &reserved { state.running.remove(id); }
            return Err(error);
        }
    }
    Ok(reserved)
}

fn prepare_received_response(config: &AgentConfig, state: &Arc<Mutex<LocalState>>,
    workspaces: &SharedWorkspaces, response: &Value) -> Result<()> {
    let reserved = reserve_received_commands(config, state, response)?;
    if let Err(error) = workspace_agent::response(workspaces, response, &config.node_id, &config.pool, &config.runtime) {
        let mut state = state.lock().unwrap();
        for id in reserved { state.running.remove(&id); }
        return Err(error);
    }
    Ok(())
}

async fn execute_and_report(
    client: Client,
    config: AgentConfig,
    state: Arc<Mutex<LocalState>>,
    epoch: String,
    assignment: Value,
) -> Result<()> {
    let mut pending = VecDeque::from([assignment]);
    while let Some(mut assignment) = pending.pop_front() {
        if { let state = state.lock().unwrap(); state.report_retries_exhausted || state.stopping } {
            let mut state = state.lock().unwrap();
            for item in std::iter::once(&assignment).chain(pending.iter()) {
                if let Some(id) = item["command_id"].as_str() { state.running.remove(id); }
            }
            return Ok(());
        }
        let command_id = assignment
            .get("command_id")
            .and_then(Value::as_str)
            .context("command_id is required")?
            .to_string();
        let execution_session_id = assignment
            .get("execution_session_id")
            .and_then(Value::as_str)
            .context("execution_session_id is required")?
            .to_string();
        let request_id = assignment
            .get("request_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        {
            let mut state = state.lock().unwrap();
            state.sessions.insert(execution_session_id.clone());
            state
                .running
                .insert(command_id.clone(), (execution_session_id.clone(), request_id));
        }
        save_state(&config.state_path, &state.lock().unwrap())?;
        let workspaces = state.lock().unwrap().workspaces.clone();
        eprintln!("[rscape-node {}] command {command_id} workspace preparation starting", config.node_id);
        let invalid_visibility = runtime::validate_assignment_visibility(&assignment).is_err();
        let preparation = async {
            if invalid_visibility { return Ok(()); }
            let version = assignment["session_spec"]["suspend_version"].as_u64().unwrap_or(0);
            let retained = state.lock().unwrap().suspended_contexts.get(&execution_session_id).cloned();
            let previous = retained.clone().or_else(||workspaces.lock().unwrap().contexts.get(&execution_session_id).cloned());
            let fingerprint=runtime::retention_fingerprint(&config.runtime,&assignment["session_spec"])?;
            if let Some(previous)=previous {
                if previous["retention_fingerprint"]!=fingerprint {bail!("retained execution environment changed; explicit rebuild required");}
            }
            if let Some(context) = retained {
                validate_retained_environment(&context, &assignment["session_spec"], version)?;
                runtime::validate_retained_state(&config.runtime,&execution_session_id,
                    assignment["session_spec"]["runtime_adapter"].as_str().unwrap_or(""),
                    assignment["session_spec"]["isolation_policy"].as_str().unwrap_or("default"))?;
                let mut state=state.lock().unwrap();
                state.closed_sessions.remove(&execution_session_id);
                state.resumed_versions.insert(execution_session_id.clone(),version);
            } else if assignment["session_spec"]["resume_retained"] == true && state.lock().unwrap().resumed_versions.get(&execution_session_id) != Some(&version) {
                bail!("retained execution metadata missing; explicit rebuild required");
            }
            if assignment["session_spec"]["resume_retained"] == true {
                runtime::validate_retained_state(&config.runtime,&execution_session_id,
                    assignment["session_spec"]["runtime_adapter"].as_str().unwrap_or(""),
                    assignment["session_spec"]["isolation_policy"].as_str().unwrap_or("default"))?;
            }
            if state.lock().unwrap().closed_sessions.contains(&execution_session_id) { bail!("session is stopping"); }
            if config.runtime.cancelled_commands.lock().unwrap().contains(&command_id) { bail!("command cancelled before workspace preparation"); }
            workspace_agent::prepare(&workspaces, &mut assignment, &config.node_id, &config.pool).await?;
            if let Some(context)=workspaces.lock().unwrap().contexts.get_mut(&execution_session_id){
                context["retention_fingerprint"]=fingerprint;
            }
            state.lock().unwrap().suspended_contexts.remove(&execution_session_id);
            save_state(&config.state_path, &state.lock().unwrap())?;
            workspace_agent::verify(&workspaces, &execution_session_id).await
        }.await;
        let prepared = preparation.is_ok() && !invalid_visibility;
        eprintln!("[rscape-node {}] command {command_id} workspace prepared={prepared}", config.node_id);
        let mut result = match preparation {
            Ok(()) => execute(&assignment, &config.runtime).await,
            Err(error) => runtime::ExecutionResult { exit_code:127, stdout:String::new(), stderr:format!("workspace preparation failed: {error:#}"),
                failure_kind:"infrastructure".into(), argv:Vec::new(), adapter:assignment["session_spec"]["runtime_adapter"].as_str().unwrap_or("local-process").into(),
                isolation_policy:assignment["session_spec"]["isolation_policy"].as_str().unwrap_or("none").into() },
        };
        if prepared {
            if let Err(error) = workspace_agent::verify(&workspaces, &execution_session_id).await {
                result.exit_code = 127; result.failure_kind = "infrastructure".into();
                result.stderr.push_str(&format!("\nworkspace verification failed: {error:#}"));
            }
        }
        eprintln!("[rscape-node {}] command {command_id} execution finished exit={}", config.node_id, result.exit_code);
        if let Err(error) = workspace_agent::record(&workspaces, &execution_session_id, None).await {
            result.exit_code = 127; result.failure_kind = "infrastructure".into();
            result.stderr.push_str(&format!("\nworkspace status failed: {error:#}"));
        }
        {
            let mut state = state.lock().unwrap();
            state.pending_reports.insert(command_id.clone());
            state.running.remove(&command_id);
        }
        config.runtime.cancelled_commands.lock().unwrap().remove(&command_id);
        let log_path = config.runtime.log_dir.join(format!("{command_id}.log"));
        let mut exit_code = result.exit_code;
        let mut stderr = result.stderr.clone();
        let mut failure_kind = result.failure_kind.clone();
        if let Err(error) = write_log(&log_path, &assignment, &result) {
            exit_code = 127;
            failure_kind = "infrastructure".into();
            stderr.push_str(&format!("\nfailed to write node-agent log: {error}"));
        }
        save_state(&config.state_path, &state.lock().unwrap())?;
        if state.lock().unwrap().report_retries_exhausted { return Ok(()); }
        let mut payload = json!({"node_id":config.node_id,"agent_epoch":epoch,"command_id":command_id,"exit_code":exit_code,"stdout":result.stdout,"stderr":stderr,"failure_kind":failure_kind,"log_path":log_path,"artifact_paths":[],"accept_commands":true,"max_commands":8});
        payload["workspace_updates"] = workspaces.lock().unwrap().pending()?;
        let response = match post(&client, &config, "/v1/agent/report-command", &payload).await {
            Ok(response) => response,
            Err(error) => {
                if let Some(status) = report_error_status(&error) {
                    if status < 500 && status != 413 {
                        eprintln!("[rscape-node {}] report abandoned command_id={} reason=permanent_http status={} log_path={}", config.node_id, command_id, status, log_path.display());
                        state.lock().unwrap().pending_reports.remove(&command_id);
                        continue;
                    }
                }
                if report_error_status(&error) != Some(413) && !retryable_report_error(&error) {
                    state.lock().unwrap().pending_reports.remove(&command_id);
                    return Err(error);
                }
                payload["exit_code"] = json!(127);
                payload["stdout"] = json!("");
                payload["failure_kind"] = json!("infrastructure");
                payload["stderr"] = json!(json!({
                    "error": "command_report_delivery_failed",
                    "original_exit_code": exit_code,
                    "log_reference": "node-local; not remotely retrievable"
                }).to_string());
                let mut attempts = 0;
                loop {
                    attempts += 1;
                    match post(&client, &config, "/v1/agent/report-command", &payload).await {
                        Ok(response) => break response,
                        Err(error) => {
                            if let Some(status) = report_error_status(&error) {
                                if status < 500 {
                                    eprintln!("[rscape-node {}] report abandoned command_id={} reason=permanent_http status={} log_path={}", config.node_id, command_id, status, log_path.display());
                                    break json!({});
                                }
                            }
                            if !retryable_report_error(&error) {
                                state.lock().unwrap().pending_reports.remove(&command_id);
                                return Err(error);
                            }
                            if attempts == REPORT_FALLBACK_ATTEMPTS {
                                state.lock().unwrap().report_retries_exhausted = true;
                                eprintln!("[rscape-node {}] report abandoned command_id={} reason=retry_exhausted_agent_stopping log_path={}", config.node_id, command_id, log_path.display());
                                return Ok(());
                            }
                            eprintln!("[rscape-node {}] failure report pending for {}", config.node_id, command_id);
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    }
                }
            }
        };
        state.lock().unwrap().pending_reports.remove(&command_id);
        prepare_received_response(&config, &state, &workspaces, &response)?;
        for command in response
            .get("commands")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            match command
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("run")
            {
                "purge_session" => {
                    if let Err(error)=purge_local_session(&config,&state,&command) {eprintln!("execution purge deferred: {error:#}");}
                }
                "suspend_session" => {
                    if let Err(error) = suspend_local_session(&config, &state, &command).await {
                        eprintln!("execution suspension deferred: {error:#}");
                    }
                }
                "stop_session" => {
                    if let Some(id) = command.get("execution_session_id").and_then(Value::as_str) {
                        if let Err(error) = stop_command(&config, &state, &command).await {
                            eprintln!("[rscape-node {}] session cleanup deferred for {id}: {error:#}", config.node_id);
                        }
                    }
                }
                "cancel" => {
                    if let Some(id) = command.get("command_id").and_then(Value::as_str) {
                        runtime::cancel_command(&config.runtime, id);
                    }
                }
                _ => pending.push_back(command),
            }
        }
    }
    Ok(())
}

fn validate_retained_environment(context: &Value, spec: &Value, version: u64) -> Result<()> {
    if context["suspend_version"].as_u64() != Some(version) {
        bail!("retained suspension version mismatch");
    }
    // The adapter has always been retained at the context level; older cleanup
    // projections omitted its redundant session_spec copy.
    if context["runtime_adapter"].as_str().is_none() || context["runtime_adapter"] != spec["runtime_adapter"] {
        bail!("retained execution environment changed");
    }
    if let Some(adapter) = context["session_spec"].get("runtime_adapter") {
        if adapter != &context["runtime_adapter"] {
            bail!("inconsistent retained runtime adapter");
        }
    }
    for key in ["image", "isolation_policy"] {
        if context["session_spec"][key] != spec[key] {
            bail!("retained execution environment changed");
        }
    }
    Ok(())
}

async fn stop_command(config:&AgentConfig,state:&Arc<Mutex<LocalState>>,command:&Value)->Result<()> {
    let id=command["execution_session_id"].as_str().context("stop execution ID")?;
    let version=command["suspend_version"].as_u64().unwrap_or(0);
    {
        let mut state=state.lock().unwrap();
        if state.resumed_versions.get(id).map(|last|version<=*last).unwrap_or(false){return Ok(());}
        if let Some(context)=state.suspended_contexts.get_mut(id){
            if context["suspend_version"].as_u64().unwrap_or(0)>=version{return Ok(());}
            context["suspend_version"]=json!(version);save_state(&config.state_path,&state)?;return Ok(());
        }
        state.stop_versions.insert(id.into(),version);
    }
    stop_local_session(config,state,id).await
}

async fn stop_local_session(
    config: &AgentConfig,
    state: &Arc<Mutex<LocalState>>,
    execution_session_id: &str,
) -> Result<()> {
    let running: Vec<String> = {
        let mut state = state.lock().unwrap();
        state.pending_stops.insert(execution_session_id.into());
        state.closed_sessions.insert(execution_session_id.into());
        state.running.iter().filter(|(_, (session, _))| session == execution_session_id).map(|(id, _)| id.clone()).collect()
    };
    if !running.is_empty() {
        for command in running { runtime::cancel_command(&config.runtime, &command); }
        return Ok(());
    }
    let workspaces = state.lock().unwrap().workspaces.clone();
    {
        let mut workspaces=workspaces.lock().unwrap();workspaces.stop_requested.insert(execution_session_id.into());
        if workspaces.maintenance_sessions.contains(execution_session_id) {return Ok(());}
    }
    let context = workspaces.lock().unwrap().contexts.get(execution_session_id).cloned();
    if let Some(context)=&context {
        runtime::ensure_retained_unmounted(&config.runtime,execution_session_id,
            context["runtime_adapter"].as_str().unwrap_or("local-process"),
            context["session_spec"]["isolation_policy"].as_str().unwrap_or("none"))?;
    }
    workspace_agent::release(&workspaces, execution_session_id).await?;
    {
        let mut state = state.lock().unwrap();
        state.sessions.remove(execution_session_id);
        state.pending_stops.remove(execution_session_id);
        if let Some(mut context) = context {
            context["suspend_version"] = json!(state.stop_versions.remove(execution_session_id).unwrap_or(0));
            state.suspended_contexts.insert(execution_session_id.into(), context);
        }
    }
    save_state(&config.state_path, &state.lock().unwrap())?;
    Ok(())
}

async fn suspend_local_session(config: &AgentConfig, state: &Arc<Mutex<LocalState>>, command: &Value) -> Result<()> {
    let id = command["execution_session_id"].as_str().context("suspension execution ID")?;
    let version = command["suspend_version"].as_u64().filter(|v|*v>0).context("suspension version")?;
    let workspaces = {
        let state=state.lock().unwrap();
        if state.resumed_versions.get(id).copied().unwrap_or(0)>=version || state.suspended_contexts.get(id)
            .map(|c|c["suspend_version"].as_u64()==Some(version)).unwrap_or(false) {return Ok(());}
        if state.running.values().any(|(session,_)|session==id) {bail!("execution is still busy");}
        state.workspaces.clone()
    };
    let mut context=workspaces.lock().unwrap().contexts.get(id).cloned().context("missing execution metadata")?;
    runtime::ensure_retained_unmounted(&config.runtime,id,context["runtime_adapter"].as_str().unwrap_or("local-process"),
        context["session_spec"]["isolation_policy"].as_str().unwrap_or("none"))?;
    runtime::validate_retained_state(&config.runtime,id,context["runtime_adapter"].as_str().unwrap_or("local-process"),
        context["session_spec"]["isolation_policy"].as_str().unwrap_or("none"))?;
    workspace_agent::release(&workspaces,id).await?;
    context["suspend_version"]=json!(version);
    let mut state=state.lock().unwrap();
    state.sessions.remove(id); state.closed_sessions.insert(id.into());
    state.suspended_contexts.insert(id.into(),context);
    save_state(&config.state_path,&state)?;
    state.heartbeat_changed.notify_one();
    Ok(())
}

fn purge_local_session(config:&AgentConfig,state:&Arc<Mutex<LocalState>>,command:&Value)->Result<()> {
    let id=command["execution_session_id"].as_str().context("purge execution ID")?;
    let mut state=state.lock().unwrap();
    if state.purged_executions.contains(id){return Ok(());}
    if state.sessions.contains(id)||state.running.values().any(|(session,_)|session==id){bail!("cannot purge a busy execution");}
    let context=state.suspended_contexts.get(id).context("purge requires retained metadata")?;
    if context["suspend_version"]!=command["suspend_version"]{bail!("purge suspension version mismatch");}
    let adapter=context["runtime_adapter"].as_str().unwrap_or("local-process");
    let paths=runtime::retained_state_paths(&config.runtime,id,adapter,context["session_spec"]["isolation_policy"].as_str().unwrap_or("none"))?;
    runtime::ensure_retained_unmounted(&config.runtime,id,adapter,context["session_spec"]["isolation_policy"].as_str().unwrap_or("none"))?;
    runtime::cleanup_session_for(&config.runtime,id,adapter)?;
    if paths.iter().any(|path|path.exists()){bail!("runtime purge left state paths");}
    state.suspended_contexts.remove(id);state.purged_executions.insert(id.into());
    save_state(&config.state_path,&state)?;state.heartbeat_changed.notify_one();Ok(())
}

fn registration_payload(config: &AgentConfig, epoch: &str) -> Value {
    let capabilities = config.runtime.capabilities.clone();
    let aggregate = runtime::aggregate_capabilities(&capabilities);
    json!({"node_id":config.node_id,"pool":config.pool,"pool_mode":config.pool_mode,"gpu_slots":config.gpu_slots,"placement":config.placement,"resources":{"cpu_cores":config.cpu_cores,"memory_gb":config.memory_gb,"gpu_uuids":config.gpu_uuids},"capabilities":aggregate,"runtime_capabilities":capabilities,"allocation_id":config.allocation_id,"allocation_deadline":config.allocation_deadline,"drain_before_sec":config.drain_before_sec,"cleanup_buffer_sec":config.cleanup_buffer_sec,"runtime_cache_config":config.runtime_cache_config,"cached_images":config.cached_images,"join_token":config.join_token,"agent_epoch":epoch})
}

fn announcement_payload(config: &AgentConfig, epoch: &str) -> Value {
    json!({"node_id":config.node_id,"pool":config.pool,"pool_mode":config.pool_mode,"gpu_slots":config.gpu_slots,"placement":config.placement,"resources":{"cpu_cores":config.cpu_cores,"memory_gb":config.memory_gb,"gpu_uuids":config.gpu_uuids},"allocation_id":config.allocation_id,"allocation_deadline":config.allocation_deadline,"drain_before_sec":config.drain_before_sec,"cleanup_buffer_sec":config.cleanup_buffer_sec,"runtime_cache_config":config.runtime_cache_config,"cached_images":config.cached_images,"join_token":config.join_token,"agent_epoch":epoch})
}

async fn register_until_available(client: &Client, config: &AgentConfig, payload: &Value) -> Value {
    loop {
        match post(client, config, "/v1/agent/register", payload).await {
            Ok(response) => return response,
            Err(error) => eprintln!(
                "[rscape-node {}] registration failure; retrying: {error:#}",
                config.node_id
            ),
        }
        tokio::time::sleep(Duration::from_secs_f64(config.heartbeat_interval.max(0.01))).await;
    }
}

fn shutdown_control<'a>(payload: &'a Value, config: &AgentConfig) -> Option<&'a Value> {
    let control = payload.get("agent_control")?;
    if control.get("authoritative").and_then(Value::as_bool) != Some(true)
        || control.get("action").and_then(Value::as_str) != Some("shutdown")
    {
        return None;
    }
    if !matches!(
        control.get("reason").and_then(Value::as_str),
        Some("allocation_removed" | "node_removed")
    ) {
        return None;
    }
    if control.get("node_id").and_then(Value::as_str) != Some(config.node_id.as_str()) {
        return None;
    }
    if let Some(allocation) = &config.allocation_id {
        if control.get("allocation_id").and_then(Value::as_str) != Some(allocation.as_str()) {
            return None;
        }
    }
    Some(control)
}

async fn post(client: &Client, config: &AgentConfig, path: &str, payload: &Value) -> Result<Value> {
    let mut request = client
        .post(format!(
            "{}{}",
            config.controller_url.trim_end_matches('/'),
            path
        ))
        .json(payload);
    if let Some(token) = &config.token {
        request = request.bearer_auth(token);
    }
    let response = request.send().await?;
    let status = response.status();
    let value: Value = response.json().await.unwrap_or_else(|_| json!({}));
    if !status.is_success() {
        return Err(AgentHttpError { status, body: value }.into());
    }
    Ok(value)
}
fn load_state(path: &PathBuf) -> Result<LocalState> {
    let data = match std::fs::read(path) {
        Ok(data) => data,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(LocalState::default()),
        Err(error) => return Err(error).context("cannot safely restore agent state; preserve it for reconciliation"),
    };
    let value: Value = serde_json::from_slice(&data).context("cannot safely restore agent state; preserve it for reconciliation")?;
    if !value.is_object() { bail!("cannot safely restore agent state: expected an object"); }
    if value["version"] != 2 {bail!("unsupported agent state schema; use a fresh state path");}
    if value["running_commands"].as_array().map(|items|items.is_empty()).unwrap_or(true)
        &&value["local_sessions"].as_array().map(|items|items.iter().any(|id|id.as_str().map(|id|value["session_contexts"].get(id).is_none()).unwrap_or(true))).unwrap_or(false){
        bail!("cached execution metadata is missing; explicit recovery required");
    }
    for name in ["local_sessions", "running_commands", "closed_workspace_sessions"] {
        if let Some(field) = value.get(name) {
            if !field.as_array().map(|items| items.iter().all(Value::is_string)).unwrap_or(false) { bail!("invalid agent state field: {name}"); }
        }
    }
    for name in ["session_contexts", "workspace_updates", "workspace_bindings", "workspace_acknowledged", "workspace_operation_receipts"] {
        if value.get(name).map(|field| !field.is_object()).unwrap_or(false) { bail!("invalid agent state field: {name}"); }
    }
    if let Some(receipts)=value.get("workspace_operation_receipts").and_then(Value::as_object) {
        for (identity,receipt) in receipts {
            let valid_id=identity.len()==35 && identity.starts_with("ws-") && identity[3..].bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b));
            let valid_operation=receipt["id"].as_str().map(|id|id.len()==32&&id.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))).unwrap_or(false);
            if !valid_id || !valid_operation || !receipt["sequence"].as_u64().map(|n|n>0&&n<=i64::MAX as u64).unwrap_or(false) || !receipt["update"].is_object() {bail!("invalid workspace operation receipt");}
        }
    }
    let items = value.get("local_sessions").and_then(Value::as_array).cloned().unwrap_or_default();
    Ok(LocalState {
                    heartbeat_changed: Default::default(),
                    suspended_contexts: serde_json::from_value(value.get("suspended_contexts").cloned().unwrap_or(json!({})))?,
                    resumed_versions: serde_json::from_value(value.get("resumed_versions").cloned().unwrap_or(json!({})))?,
                    purged_executions: serde_json::from_value(value.get("purged_executions").cloned().unwrap_or(json!([])))?,
                    stop_versions: serde_json::from_value(value.get("stop_versions").cloned().unwrap_or(json!({})))?,
                    sessions: items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect(),
                    running: BTreeMap::new(),
                    pending_reports: BTreeSet::new(),
                    report_retries_exhausted: false,
                    workspaces: Arc::new(Mutex::new(AgentWorkspaces::restore(&value))),
                    pending_stops: BTreeSet::new(),
                    closed_sessions: serde_json::from_value(value.get("closed_workspace_sessions").cloned().unwrap_or_else(|| json!([]))).unwrap_or_default(),
                    stopping: false,
                    unreconciled_commands: value["running_commands"].as_array().map(|commands| !commands.is_empty()).unwrap_or(false)
                        && value["session_contexts"].as_object().map(|contexts| contexts.values().any(|context| context["session_spec"]["workspace"].is_object())).unwrap_or(false),
    })
}
fn save_state(path: &PathBuf, state: &LocalState) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("tmp.{}", Uuid::new_v4().simple()));
    let mut value = state.workspaces.lock().unwrap().persist();
    value["version"] = json!(2);
    value["local_sessions"] = json!(state.sessions);
    value["running_commands"] = json!(state.running.keys().collect::<Vec<_>>());
    value["closed_workspace_sessions"] = json!(state.closed_sessions);
    value["suspended_contexts"] = json!(state.suspended_contexts);
    value["resumed_versions"] = json!(state.resumed_versions);
    value["purged_executions"] = json!(state.purged_executions);
    value["stop_versions"] = json!(state.stop_versions);
    let result=(|| -> Result<()> {
        let mut file=std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600)
            .custom_flags(nix::libc::O_NOFOLLOW).open(&temp)?;
        file.write_all(&serde_json::to_vec(&value)?)?;file.sync_all()?;
        std::fs::rename(&temp,path)?;
        std::fs::File::open(path.parent().unwrap_or(std::path::Path::new(".")))?.sync_all()?;
        Ok(())
    })();
    if temp.exists(){let _=std::fs::remove_file(&temp);}
    result
}

#[cfg(test)]
fn parse_gpu_metrics(text: &str, allowed_uuids: &[String]) -> Vec<Value> {
    let allowed: BTreeSet<&str> = allowed_uuids.iter().map(String::as_str).collect();
    let timestamp = crate::controller::now();
    text.lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != 9 || (!allowed.is_empty() && !allowed.contains(fields[1])) {
                return None;
            }
            let index = fields[0].parse::<usize>().ok()?;
            let mib = |value: &str| value.parse::<u64>().ok().map(|v| v * 1024 * 1024);
            Some(json!({
                "index": index,
                "uuid": fields[1],
                "name": fields[2],
                "timestamp": timestamp,
                "gpu_utilization_percent": fields[3].parse::<u64>().ok(),
                "memory_utilization_percent": fields[4].parse::<u64>().ok(),
                "memory_total_bytes": mib(fields[5]),
                "memory_used_bytes": mib(fields[6]),
                "memory_free_bytes": mib(fields[7]),
                "temperature_c": fields[8].parse::<u64>().ok(),
            }))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_adapter_uses_canonical_context_and_rejects_environment_changes() {
        let spec=json!({"runtime_adapter":"local-process","image":"docker://ubuntu:latest","isolation_policy":"none"});
        let mut context=json!({"runtime_adapter":"local-process","suspend_version":1,
            "session_spec":{"image":"docker://ubuntu:latest","isolation_policy":"none"}});
        validate_retained_environment(&context,&spec,1).unwrap();
        for field in ["runtime_adapter","image","isolation_policy"] {
            let mut changed=spec.clone();changed[field]=json!("different");
            assert!(validate_retained_environment(&context,&changed,1).is_err());
        }
        assert!(validate_retained_environment(&context,&spec,2).is_err());
        context["session_spec"]["runtime_adapter"]=json!("apptainer");
        assert!(validate_retained_environment(&context,&spec,1).is_err());
        context["session_spec"]["runtime_adapter"]=json!("local-process");
        validate_retained_environment(&context,&spec,1).unwrap();
    }

    #[tokio::test]
    async fn only_suspension_controls_wake_heartbeat() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let mut config = config();
        config.state_path = temporary.path().join("state.json");
        let state = Arc::new(Mutex::new(LocalState::default()));
        state.lock().unwrap().sessions.insert("exec-noop".into());
        state.lock().unwrap().workspaces.lock().unwrap().contexts.insert("exec-noop".into(),
            json!({"runtime_adapter":"local-process", "session_spec":{"isolation_policy":"none"}}));
        let changed = state.lock().unwrap().heartbeat_changed.clone();
        stop_command(&config, &state, &json!({"execution_session_id":"exec-noop"})).await?;
        assert!(tokio::time::timeout(Duration::from_millis(10), changed.notified()).await.is_err());
        let command = json!({"execution_session_id":"exec-noop", "suspend_version":1});
        stop_command(&config, &state, &command).await?;
        assert!(tokio::time::timeout(Duration::from_millis(10), changed.notified()).await.is_err());
        state.lock().unwrap().sessions.insert("exec-suspend".into());
        state.lock().unwrap().workspaces.lock().unwrap().contexts.insert("exec-suspend".into(),
            json!({"runtime_adapter":"local-process", "session_spec":{"isolation_policy":"none"}}));
        let command = json!({"execution_session_id":"exec-suspend", "suspend_version":1});
        suspend_local_session(&config, &state, &command).await?;
        tokio::time::timeout(Duration::from_millis(100), changed.notified()).await?;
        suspend_local_session(&config, &state, &command).await?;
        assert!(tokio::time::timeout(Duration::from_millis(10), changed.notified()).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn retained_layer_survives_agent_reload_until_explicit_purge() -> Result<()> {
        let temporary=tempfile::tempdir()?;
        let mut config=config();
        config.state_path=temporary.path().join("state.json");
        config.runtime.adapters=vec!["apptainer".into()];
        config.runtime.apptainer_runtime_dir=temporary.path().join("runtime");
        let path=runtime::retained_state_paths(&config.runtime,"exec-retained","apptainer","image-stateful")?.remove(0);
        std::fs::create_dir_all(&path)?;
        std::fs::write(path.join("marker"),b"preserved")?;
        let state=Arc::new(Mutex::new(LocalState::default()));
        state.lock().unwrap().sessions.insert("exec-retained".into());
        state.lock().unwrap().workspaces.lock().unwrap().contexts.insert("exec-retained".into(),
            json!({"runtime_adapter":"apptainer","session_spec":{"runtime_adapter":"apptainer","isolation_policy":"image-stateful"}}));
        let command=json!({"execution_session_id":"exec-retained","suspend_version":1});
        suspend_local_session(&config,&state,&command).await?;
        let changed = state.lock().unwrap().heartbeat_changed.clone();
        tokio::time::timeout(Duration::from_millis(100), changed.notified()).await?;
        suspend_local_session(&config,&state,&command).await?;
        assert!(tokio::time::timeout(Duration::from_millis(10), changed.notified()).await.is_err());
        assert!(path.join("marker").exists());
        assert!(state.lock().unwrap().sessions.is_empty());
        let restored=Arc::new(Mutex::new(load_state(&config.state_path)?));
        assert_eq!(restored.lock().unwrap().suspended_contexts["exec-retained"]["suspend_version"],1);
        suspend_local_session(&config,&restored,&command).await?;
        purge_local_session(&config,&restored,&command)?;
        assert!(!path.exists());
        assert!(restored.lock().unwrap().purged_executions.contains("exec-retained"));
        Ok(())
    }

    #[test]
    fn received_batch_is_reserved_before_serial_dispatch() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let mut config = config();
        config.state_path = temporary.path().join("state.json");
        let state = Arc::new(Mutex::new(LocalState::default()));
        let response = json!({"commands":[
            {"command_id":"c1","execution_session_id":"s1","request_id":"r1"},
            {"command_id":"c2","execution_session_id":"s2","request_id":"r2"}]});
        reserve_received_commands(&config, &state, &response)?;
        assert_eq!(state.lock().unwrap().running.len(), 2);
        assert_eq!(state.lock().unwrap().sessions.len(), 2);
        let persisted: Value = serde_json::from_slice(&std::fs::read(&config.state_path)?)?;
        assert_eq!(persisted["running_commands"], json!(["c1","c2"]));
        reserve_received_commands(&config, &state, &response)?;
        assert_eq!(state.lock().unwrap().running.len(), 2);
        state.lock().unwrap().closed_sessions.insert("s3".into());
        reserve_received_commands(&config, &state, &json!({"commands":[
            {"command_id":"c3","execution_session_id":"s3","request_id":"r3"}]}))?;
        assert!(!state.lock().unwrap().running.contains_key("c3"));
        let bad_parent = temporary.path().join("not-directory");
        std::fs::write(&bad_parent, b"blocked")?;
        config.state_path = bad_parent.join("state.json");
        assert!(reserve_received_commands(&config, &state, &json!({"commands":[
            {"command_id":"c4","execution_session_id":"s4","request_id":"r4"}]})).is_err());
        assert!(!state.lock().unwrap().running.contains_key("c4"));
        assert!(state.lock().unwrap().running.contains_key("c1"));
        Ok(())
    }

    #[tokio::test]
    async fn stopped_dispatch_drops_unstarted_reservation() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let mut config = config();
        config.state_path = temporary.path().join("state.json");
        let state = Arc::new(Mutex::new(LocalState::default()));
        let command = json!({"command_id":"c1","execution_session_id":"s1","request_id":"r1"});
        reserve_received_commands(&config, &state, &json!({"commands":[command]}))?;
        state.lock().unwrap().stopping = true;
        execute_and_report(Client::new(), config, state.clone(), "epoch".into(), command).await?;
        assert!(state.lock().unwrap().running.is_empty());
        Ok(())
    }

    #[test]
    fn report_failure_classification_is_typed() {
        for status in [400, 401, 403, 404, 409, 413, 429] {
            let error: anyhow::Error = AgentHttpError { status: reqwest::StatusCode::from_u16(status).unwrap(), body: json!({}) }.into();
            assert_eq!(report_error_status(&error), Some(status));
            assert!(!retryable_report_error(&error));
        }
        for status in [500, 502, 503, 504] {
            let error: anyhow::Error = AgentHttpError { status: reqwest::StatusCode::from_u16(status).unwrap(), body: json!({}) }.into();
            assert!(retryable_report_error(&error));
        }
        assert!(!retryable_report_error(&anyhow::anyhow!("programming error")));
        let error = Client::new().post("://invalid").build().unwrap_err();
        assert!(!retryable_report_error(&error.into()));
    }

    fn config() -> AgentConfig {
        AgentConfig {
            metrics_enabled: false,
            controller_url: "http://127.0.0.1:1".into(),
            node_id: "node-a".into(),
            agent_epoch: None,
            pool: "pool-a".into(),
            pool_mode: "exclusive".into(),
            gpu_slots: 1,
            placement: "pack".into(),
            cpu_cores: 1.0,
            memory_gb: 1.0,
            gpu_uuids: vec![],
            allocation_id: Some("allocation-a".into()),
            allocation_deadline: None,
            drain_before_sec: 0.0,
            cleanup_buffer_sec: 0.0,
            runtime_cache_config: "default".into(),
            cached_images: vec![],
            join_token: None,
            token: None,
            heartbeat_interval: 0.01,
            once: false,
            exit_on_removal: true,
            state_path: "/tmp/rscape-agent-test-state.json".into(),
            runtime: RuntimeConfig::test_config("local-process"),
        }
    }

    #[test]
    fn shutdown_requires_authority_reason_and_matching_identity() {
        let config = config();
        let positive = json!({"agent_control":{"authoritative":true,"action":"shutdown","reason":"allocation_removed","node_id":"node-a","allocation_id":"allocation-a"}});
        assert!(shutdown_control(&positive, &config).is_some());
        for mutation in [
            json!({"agent_control":{"authoritative":false,"action":"shutdown","reason":"allocation_removed","node_id":"node-a","allocation_id":"allocation-a"}}),
            json!({"agent_control":{"authoritative":true,"action":"shutdown","reason":"unknown","node_id":"node-a","allocation_id":"allocation-a"}}),
            json!({"agent_control":{"authoritative":true,"action":"shutdown","reason":"allocation_removed","node_id":"node-b","allocation_id":"allocation-a"}}),
            json!({"agent_control":{"authoritative":true,"action":"shutdown","reason":"allocation_removed","node_id":"node-a","allocation_id":"allocation-b"}}),
        ] {
            assert!(shutdown_control(&mutation, &config).is_none());
        }
    }

    #[test]
    fn gpu_metric_parser_matches_python_field_shape_and_filters_uuid() {
        let metrics = parse_gpu_metrics(
            "0, GPU-a, NVIDIA GB200, 12, 3, 100, 25, 75, 41\n1, GPU-b, NVIDIA GB200, 0, 0, 200, 10, 190, 39\n",
            &["GPU-b".into()],
        );
        assert_eq!(metrics.len(), 1);
        let metric = &metrics[0];
        let keys: BTreeSet<&str> = metric
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            BTreeSet::from([
                "gpu_utilization_percent",
                "index",
                "memory_free_bytes",
                "memory_total_bytes",
                "memory_used_bytes",
                "memory_utilization_percent",
                "name",
                "temperature_c",
                "timestamp",
                "uuid",
            ])
        );
        assert_eq!(metric["uuid"], "GPU-b");
        assert_eq!(metric["memory_total_bytes"], 200 * 1024 * 1024);
    }
}
