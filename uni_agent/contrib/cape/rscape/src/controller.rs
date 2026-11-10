// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::auth::Authority;
use crate::journal::{DurableReceipt, Event, Journal};
use crate::model::*;
use crate::workspace::{WorkspaceConfig, WorkspaceConfigStore};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ControllerError {
    pub status: u16,
    pub message: String,
    pub payload: Option<Value>,
}

impl ControllerError {
    pub fn bad(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            payload: None,
        }
    }
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: 403,
            message: message.into(),
            payload: None,
        }
    }
    pub fn missing(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
            payload: None,
        }
    }
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
            payload: None,
        }
    }
}

pub type ControlResult<T> = std::result::Result<T, ControllerError>;

pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub struct Controller {
    pub telemetry_query_url: Option<String>,
    pub telemetry_receiver_url: Option<String>,
    pub node_purge_quarantine: BTreeSet<String>,
    projected_node_purges: BTreeSet<String>,
    managed_seen: BTreeMap<String, f64>,
    pub registration_attempts: std::collections::VecDeque<std::time::Instant>,
    pub requests: BTreeMap<String, RequestRecord>,
    pub sessions: BTreeMap<String, SessionRecord>,
    pub logical_sessions: BTreeMap<(String, String), Value>,
    pub nodes: BTreeMap<String, NodeRecord>,
    pub commands: BTreeMap<String, CommandAssignment>,
    pub request_commands: BTreeMap<String, String>,
    pub allocations: BTreeMap<String, AllocationRecord>,
    pub gpu_ledger: BTreeMap<String, GpuLedgerEntry>,
    pub pool_configs: BTreeMap<String, PoolConfig>,
    pub workspace_configs: WorkspaceConfigStore,
    pub workspaces: BTreeMap<String, Value>,
    pub queues: BTreeMap<String, NodeQueues>,
    pub authority: Authority,
    pub journal: Option<Journal>,
    pub heartbeat_timeout_seconds: f64,
    pub lost_timeout_seconds: f64,
    pub idle_session_timeout_seconds: f64,
    pub reconcile_missing_grace_seconds: f64,
    pub reconcile_missing_heartbeats: usize,
    pub gpu_contention_policy: String,
    pub gpu_shared_ratio: usize,
    pub gpu_share_overrides: BTreeMap<String, (String, usize)>,
    pub max_active_gpu_sessions_per_user: Option<usize>,
    pub max_queued_requests_per_user: Option<usize>,
    pub max_active_sessions_per_pool: Option<usize>,
    pub profiles: BTreeMap<String, Value>,
    gpu_shared_admissions: BTreeMap<String, usize>,
    reconcile_missing_observations: BTreeMap<String, (f64, usize)>,
    request_seq: u64,
    session_seq: u64,
    command_seq: u64,
}

impl Controller {
    pub fn new(authority: Authority, journal: Option<Journal>) -> Result<Self> {
        let mut controller = Self {
            telemetry_query_url: None,
            telemetry_receiver_url: None,
            managed_seen: BTreeMap::new(),
            node_purge_quarantine: BTreeSet::new(),
            projected_node_purges: BTreeSet::new(),
            registration_attempts: std::collections::VecDeque::new(),
            requests: BTreeMap::new(),
            sessions: BTreeMap::new(),
            logical_sessions: BTreeMap::new(),
            nodes: BTreeMap::new(),
            commands: BTreeMap::new(),
            request_commands: BTreeMap::new(),
            allocations: BTreeMap::new(),
            gpu_ledger: BTreeMap::new(),
            pool_configs: BTreeMap::new(),
            workspace_configs: WorkspaceConfigStore::default(),
            workspaces: BTreeMap::new(),
            queues: BTreeMap::new(),
            authority,
            journal,
            heartbeat_timeout_seconds: 30.0,
            lost_timeout_seconds: 90.0,
            idle_session_timeout_seconds: 3600.0,
            reconcile_missing_grace_seconds: 5.0,
            reconcile_missing_heartbeats: 2,
            gpu_contention_policy: "barrier-ratio".into(),
            gpu_shared_ratio: 4,
            gpu_share_overrides: BTreeMap::new(),
            max_active_gpu_sessions_per_user: None,
            max_queued_requests_per_user: None,
            max_active_sessions_per_pool: None,
            profiles: BTreeMap::new(),
            gpu_shared_admissions: BTreeMap::new(),
            reconcile_missing_observations: BTreeMap::new(),
            request_seq: 0,
            session_seq: 0,
            command_seq: 0,
        };
        controller.replay()?;
        if controller.authority.identity_path.is_some() { controller.managed_reconcile()?; }
        Ok(controller)
    }

    pub(crate) fn event(
        &mut self,
        event_type: &str,
        entity_id: &str,
        state: Option<&str>,
        reason: Option<&str>,
        details: Value,
    ) {
        if let Some(journal) = &mut self.journal {
            let _ = journal.append(&Event {
                timestamp: now(),
                event_type: event_type.to_string(),
                entity_id: entity_id.to_string(),
                state: state.map(str::to_string),
                reason: reason.map(str::to_string),
                details,
            });
        }
    }

    fn durable_event(
        &self,
        event_type: &str,
        entity_id: &str,
        state: Option<&str>,
        reason: Option<&str>,
        details: Value,
    ) -> ControlResult<Option<DurableReceipt>> {
        let Some(journal) = &self.journal else {
            return Ok(None);
        };
        journal
            .append_durable(&Event {
                timestamp: now(),
                event_type: event_type.to_string(),
                entity_id: entity_id.to_string(),
                state: state.map(str::to_string),
                reason: reason.map(str::to_string),
                details,
            })
            .map(Some)
            .map_err(|error| {
                ControllerError::internal(format!("journal enqueue failed: {error:#}"))
            })
    }

    #[cfg(test)]
    pub fn submit(&mut self, spec: WorkloadSpec, token: Option<&str>) -> ControlResult<Value> {
        let (value, receipt) = self.submit_unflushed(spec, token)?;
        if let Some(receipt) = receipt {
            receipt.wait().map_err(|error| {
                ControllerError::internal(format!("durable submit failed: {error:#}"))
            })?;
        }
        Ok(value)
    }

    pub fn submit_unflushed(
        &mut self,
        spec: WorkloadSpec,
        token: Option<&str>,
    ) -> ControlResult<(Value, Option<DurableReceipt>)> {
        let mut spec = self.resolve_profile(spec)?;
        if let Some(store) = self.authority.identity().map_err(|e| ControllerError::internal(e.to_string()))? {
            let actor = store.authenticate(token.unwrap_or(""), now()).map_err(|e| ControllerError::forbidden(e.to_string()))?;
            if actor["user_id"] != spec.user_id && actor["name"] != spec.user_id {
                return Err(ControllerError::forbidden("cannot submit for another identity"));
            }
            spec.user_id = actor["user_id"].as_str().unwrap().to_owned();
            spec.pool = store.pool_id(&spec.pool).map_err(|e| ControllerError::bad(e.to_string()))?;
        }
        spec.workspace.validate().map_err(|error| ControllerError::bad(error.to_string()))?;
        crate::session_identity::validate(&spec.session_id).map_err(|error| ControllerError::bad(error.to_string()))?;
        if spec.resources.runtime_adapter.as_deref() != Some("local-process") && spec.mounts.iter().any(|mount| crate::workspace::workspace_bind_conflict(&mount.target, &spec.workspace.target)) {
            return Err(ControllerError::bad("managed workspace target conflicts with an explicit bind; select --workspace-target"));
        }
        if spec.pool.is_empty()
            || spec.user_id.is_empty()
            || spec.session_id.is_empty()
            || spec.command.is_empty()
        {
            return Err(ControllerError::bad(
                "pool, user_id, session_id and command are required",
            ));
        }
        if spec.resources.max_duration_seconds <= 0.0
            || spec
                .resources
                .command_timeout_seconds
                .map(|value| value <= 0.0)
                .unwrap_or(false)
        {
            return Err(ControllerError::bad("duration bounds must be positive"));
        }
        self.authority
            .authorize(token, &spec.pool, Role::Submitter, Some(&spec.user_id))
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        let admission_token_id = if let Some(store) = self.authority.identity().map_err(|e| ControllerError::internal(e.to_string()))? {
            let actor = store.authenticate(token.unwrap_or(""), now()).map_err(|e| ControllerError::forbidden(e.to_string()))?;
            Some(actor["token_id"].as_str().unwrap().to_owned())
        } else { None };
        if let Some(id) = &spec.workspace.id {
            let workspace = self.workspaces.get(id).ok_or_else(|| ControllerError::bad("workspace does not exist"))?;
            if workspace["pool"] != spec.pool || workspace["user_id"] != spec.user_id {
                return Err(ControllerError::forbidden("workspace belongs to another user or pool"));
            }
            let reserved_empty=workspace["execution_session_id"].as_str().and_then(|id|self.sessions.get(id))
                .map(|s|s.empty_retained&&s.matches_reuse_identity(&spec)).unwrap_or(false);
            if matches!(workspace["state"].as_str(),Some("deleted"|"deleting"|"needs_check"|"needs_recovery")) || workspace["state"]=="absent"&&!reserved_empty {
                return Err(ControllerError::bad("workspace is unavailable; recover it or explicitly choose another workspace"));
            }
        }
        let quota_exceeded = self
            .max_queued_requests_per_user
            .map(|limit| {
                self.requests
                    .values()
                    .filter(|request| {
                        request.spec.user_id == spec.user_id
                            && request.state == RequestState::Queued
                    })
                    .count()
                    >= limit
            })
            .unwrap_or(false);
        self.request_seq += 1;
        let timestamp = now();
        let request_id = format!("req-{:06}", self.request_seq);
        let record = RequestRecord {
            admission_token_id,
            request_id: request_id.clone(),
            spec,
            state: if quota_exceeded {
                RequestState::Infeasible
            } else {
                RequestState::Queued
            },
            reason: Some(if quota_exceeded {
                "quota_exceeded".into()
            } else {
                "pool_empty".into()
            }),
            created_at: json!(timestamp),
            updated_at: json!(timestamp),
            leased_at: None,
            started_at: None,
            ended_at: quota_exceeded.then(|| json!(timestamp)),
            sequence: self.request_seq,
            execution_session_id: None,
            node_id: None,
            gpu_uuids: Vec::new(),
            gpu_mode: None,
            gpu_cotenant_count: 0,
            gpu_cotenant_counts: BTreeMap::new(),
            exit_code: None,
            failure_kind: None,
            stdout: String::new(),
            stderr: String::new(),
            log_path: None,
            artifact_paths: Vec::new(),
        };
        let submitted_state = if quota_exceeded {
            "INFEASIBLE"
        } else {
            "QUEUED"
        };
        let submitted_reason = if quota_exceeded {
            "quota_exceeded"
        } else {
            "pool_empty"
        };
        let details = request_details(&record);
        let identity = (record.spec.user_id.clone(), record.spec.session_id.clone());
        let created_logical = !self.logical_sessions.contains_key(&identity);
        if !self.logical_sessions.contains_key(&identity) {
            let logical = json!({"session_id":identity.1,"user_id":identity.0,"state":"alive","created_at":timestamp});
            self.logical_sessions.insert(identity.clone(), logical);
        }
        self.requests.insert(request_id.clone(), record);
        let receipt = self.durable_event(
            "request_submitted",
            &request_id,
            Some(submitted_state),
            Some(submitted_reason),
            details,
        )?;
        if created_logical {
            self.event("logical_session_created", &identity.1, Some("alive"), None, self.logical_sessions[&identity].clone());
        }
        if quota_exceeded {
            let details = request_details(self.requests.get(&request_id).unwrap());
            self.event(
                "request_state",
                &request_id,
                Some("INFEASIBLE"),
                Some("quota_exceeded"),
                details,
            );
        } else {
            self.schedule();
        }
        self.request_status(&request_id, token)
            .map(|value| (value, receipt))
    }

    pub fn flush_journal(&self) -> Result<()> {
        match &self.journal {
            Some(journal) => journal.flush(),
            None => Ok(()),
        }
    }

    pub fn journal_stats(&self) -> Option<(u64, u64)> {
        self.journal
            .as_ref()
            .map(|journal| (journal.event_count(), journal.commit_count()))
    }

    pub fn set_profiles(&mut self, profiles: BTreeMap<String, Value>) -> Result<()> {
        for (name, profile) in &profiles {
            validate_profile(name, profile)?;
        }
        self.profiles = profiles;
        Ok(())
    }

    fn resolve_profile(&self, mut spec: WorkloadSpec) -> ControlResult<WorkloadSpec> {
        let Some(name) = spec.profile.clone() else {
            return Ok(spec);
        };
        let profile = self.profiles.get(&name).ok_or_else(|| {
            ControllerError::missing(format!("unknown environment profile: {name}"))
        })?;
        if spec.image.is_none() {
            spec.image = profile
                .get("image")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| Some("docker://ubuntu:latest".into()));
        }
        if spec.resources.runtime_adapter.is_none() {
            spec.resources.runtime_adapter = profile
                .get("runtime_preference")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        if spec.resources.gpus == 0
            && profile
                .get("gpu_required")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        {
            spec.resources.gpus = 1;
        }
        let mut environment = BTreeMap::new();
        if let Some(values) = profile.get("env").and_then(Value::as_object) {
            for (key, value) in values {
                environment.insert(key.clone(), value.as_str().unwrap_or_default().to_string());
            }
        }
        environment.extend(spec.env.clone());
        spec.env = environment;
        let mut mounts = Vec::new();
        if let Some(values) = profile.get("default_mounts").and_then(Value::as_array) {
            for value in values {
                mounts.push(
                    profile_mount(value).map_err(|error| {
                        ControllerError::bad(format!("profile {name}: {error:#}"))
                    })?,
                );
            }
        }
        mounts.extend(spec.mounts.clone());
        spec.mounts = mounts;
        Ok(spec)
    }

    pub fn request_status(&self, request_id: &str, token: Option<&str>) -> ControlResult<Value> {
        let request = self
            .requests
            .get(request_id)
            .ok_or_else(|| ControllerError::missing(request_id))?;
        self.authority
            .authorize_owner(
                token,
                &request.spec.pool,
                &request.spec.user_id,
            )
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        Ok(self.status_value(request))
    }

    pub fn logs(&self, request_id: &str, token: Option<&str>) -> ControlResult<Value> {
        let request = self
            .requests
            .get(request_id)
            .ok_or_else(|| ControllerError::missing(request_id))?;
        self.authority
            .authorize_owner(
                token,
                &request.spec.pool,
                &request.spec.user_id,
            )
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        Ok(json!({
            "request_id": request_id,
            "stdout": request.stdout,
            "stderr": request.stderr,
            "log_path": request.log_path,
        }))
    }

    pub fn cancel(
        &mut self,
        request_id: &str,
        reason: &str,
        token: Option<&str>,
    ) -> ControlResult<Value> {
        self.cancel_impl(request_id, reason, token, true)
    }

    fn cancel_impl(&mut self, request_id: &str, reason: &str, token: Option<&str>, enforce_auth: bool) -> ControlResult<Value> {
        let (pool, owner, terminal) = {
            let request = self
                .requests
                .get(request_id)
                .ok_or_else(|| ControllerError::missing(request_id))?;
            (
                request.spec.pool.clone(),
                request.spec.user_id.clone(),
                request.state.terminal(),
            )
        };
        if enforce_auth { self.authority
            .authorize_owner(token, &pool, &owner)
            .map_err(|error| ControllerError::forbidden(error.to_string()))?; }
        if !terminal {
            let command_id = self.request_commands.get(request_id).cloned();
            let execution_session_id = self
                .requests
                .get(request_id)
                .and_then(|record| record.execution_session_id.clone());
            let timestamp = now();
            if let Some(request) = self.requests.get_mut(request_id) {
                request.state = RequestState::Cancelled;
                request.reason = Some(reason.to_string());
                request.updated_at = json!(timestamp);
                request.ended_at = Some(json!(timestamp));
            }
            if let Some(command_id) = command_id {
                self.cancel_command(&command_id, reason);
            }
            if let Some(execution_session_id) = execution_session_id {
                self.stop_session(&execution_session_id, reason);
            }
            let details = request_details(self.requests.get(request_id).unwrap());
            self.event(
                "request_state",
                request_id,
                Some("CANCELLED"),
                Some(reason),
                details,
            );
        }
        Ok(self.status_value(self.requests.get(request_id).unwrap()))
    }

    fn cancel_command(&mut self, command_id: &str, reason: &str) {
        let running_node = self.commands.get(command_id).and_then(|command| {
            (command.state == CommandState::Running).then(|| command.node_id.clone())
        });
        if let Some(command) = self.commands.get_mut(command_id) {
            let was_dispatched=if command.state==CommandState::Cancelled {command.result["was_dispatched"].as_bool().unwrap_or(true)}else{command.state!=CommandState::Pending};
            command.state = CommandState::Cancelled;
            command.updated_at = now();
            command.result = json!({"reason":reason,"was_dispatched":was_dispatched});
        } else {
            return;
        }
        if let Some(node_id) = running_node {
            let mut payload = self.commands.get(command_id).unwrap().wire();
            payload["action"] = json!("cancel");
            let controls = &mut self.queues.entry(node_id).or_default().controls;
            let duplicate = controls.iter().any(|item| {
                item.get("action").and_then(Value::as_str) == Some("cancel")
                    && item.get("command_id").and_then(Value::as_str) == Some(command_id)
            });
            if !duplicate {
                controls.push_back(payload);
            }
        }
        let details = self.commands.get(command_id).unwrap().wire();
        self.event(
            "command_state",
            command_id,
            Some("CANCELLED"),
            Some(reason),
            details,
        );
    }

    pub fn announce_or_register(
        &mut self,
        body: &Value,
        active: bool,
        bearer: Option<&str>,
    ) -> ControlResult<Value> {
        let node_id = required_string(body, "node_id")?;
        let pool = required_string(body, "pool")?;
        let runtime_config = body
            .get("runtime_cache_config")
            .and_then(Value::as_str)
            .unwrap_or("default");
        if let Some(mut store) = self.authority.identity().map_err(|e| ControllerError::internal(e.to_string()))? {
            let enrollment = store.validate_registration(bearer.unwrap_or(""), body).map_err(|e| ControllerError::forbidden(e.to_string()))?;
            if enrollment["desired"] == "removed" && !self.nodes.contains_key(&node_id) {
                return Err(ControllerError::forbidden("node enrollment is retired"));
            }
        } else { self.authority
            .validate_join(
                body.get("join_token").and_then(Value::as_str),
                &pool,
                &node_id,
                runtime_config,
            )
            .map_err(|error| ControllerError::forbidden(error.to_string()))?; }
        if self.authority.enabled {
            self.authority
                .authorize(bearer, &pool, Role::NodeAgent, Some(&node_id))
                .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        }
        let mut resources: NodeResources =
            serde_json::from_value(body.get("resources").cloned().unwrap_or_else(|| json!({})))
                .map_err(|error| ControllerError::bad(error.to_string()))?;
        if let Some(store) = self.authority.identity().map_err(|e|ControllerError::internal(e.to_string()))? {
            let state = store.snapshot().map_err(|e|ControllerError::internal(e.to_string()))?;
            if let Some(effective) = state["nodes"][&node_id].get("effective_resources") {
                resources = serde_json::from_value(effective.clone()).map_err(|e|ControllerError::bad(e.to_string()))?;
            }
        }
        let config = PoolConfig {
            pool_mode: body
                .get("pool_mode")
                .and_then(Value::as_str)
                .unwrap_or("exclusive")
                .to_string(),
            gpu_slots: body.get("gpu_slots").and_then(Value::as_u64).unwrap_or(1) as usize,
            placement: body
                .get("placement")
                .and_then(Value::as_str)
                .unwrap_or("pack")
                .to_string(),
        };
        if let Some(store) = self.authority.identity().map_err(|e| ControllerError::internal(e.to_string()))? {
            let state = store.snapshot().map_err(|e| ControllerError::internal(e.to_string()))?;
            let expected: PoolConfig = serde_json::from_value(state["pools"][&pool].get("config").cloned().unwrap_or_else(|| json!({})))
                .map_err(|e| ControllerError::bad(e.to_string()))?;
            if config != expected { return Err(ControllerError::forbidden("node cannot change administrator pool configuration")); }
            self.pool_configs.insert(pool.clone(), expected);
        }
        self.validate_pool_config(&pool, &config)?;
        let timestamp = now();
        let state = if active {
            NodeState::Active
        } else {
            NodeState::Registering
        };
        let allocation_id = body
            .get("allocation_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let declared_workspace = self.workspace_configs.get(&pool, Some(&node_id))
            .map_err(|error| ControllerError::bad(error.to_string()))?;
        let mut node = NodeRecord {
            node_id: node_id.clone(),
            pool: pool.clone(),
            state,
            resources: resources.clone(),
            runtime_capabilities: body
                .get("runtime_capabilities")
                .and_then(Value::as_object)
                .map(|items| {
                    items
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            capabilities: body
                .get("capabilities")
                .cloned()
                .unwrap_or_else(|| json!({})),
            allocation_id: allocation_id.clone(),
            allocation_deadline: body.get("allocation_deadline").and_then(Value::as_f64),
            drain_before_sec: body
                .get("drain_before_sec")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            cleanup_buffer_sec: body
                .get("cleanup_buffer_sec")
                .and_then(Value::as_f64)
                .unwrap_or(0.0),
            runtime_cache_config: runtime_config.to_string(),
            cached_images: body
                .get("cached_images")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            pool_config: config.clone(),
            agent_epoch: body
                .get("agent_epoch")
                .and_then(Value::as_str)
                .map(str::to_string),
            registered_at: timestamp,
            last_heartbeat_at: timestamp,
            used_cpu_cores: 0.0,
            used_memory_gb: 0.0,
            active_sessions: Vec::new(),
            status: json!({}),
            workspace_status: body.get("workspace_config").cloned().unwrap_or_else(|| json!({
                "protocol":1,"state":if active {"applied"} else {"pending"},"error":null,
                "applied_version":declared_workspace.version,"config":declared_workspace.to_value()})),
        };
        for session in self.sessions.values().filter(|s|s.node_id==node_id&&!s.state.terminal()
                &&!matches!(s.state,SessionState::Suspended|SessionState::Purging)) {
            node.used_cpu_cores += session.cpu_cores;
            node.used_memory_gb += session.memory_gb;
            node.active_sessions.push(session.execution_session_id.clone());
        }
        if node.used_cpu_cores>node.resources.cpu_cores || node.used_memory_gb>node.resources.memory_gb {
            return Err(ControllerError::bad("node re-registration cannot discard retained resource reservations"));
        }
        for gpu in &resources.gpu_uuids {
            self.gpu_ledger.entry(gpu.clone()).or_default();
        }
        if let Some(allocation_id) = &allocation_id {
            let allocation = self
                .allocations
                .entry(allocation_id.clone())
                .or_insert_with(|| AllocationRecord {
                    allocation_id: allocation_id.clone(),
                    pool: pool.clone(),
                    state: AllocationState::Active,
                    node_ids: BTreeSet::new(),
                    deadline: body.get("allocation_deadline").and_then(Value::as_f64),
                    created_at: timestamp,
                    updated_at: timestamp,
                    reason: None,
                });
            allocation.state = AllocationState::Active;
            allocation.node_ids.insert(node_id.clone());
            allocation.updated_at = timestamp;
        }
        let previous_config = self.pool_configs.insert(pool.clone(), config.clone());
        if previous_config
            .as_ref()
            .map(|previous| previous != &config)
            .unwrap_or(false)
        {
            self.event("pool_config_replaced", &pool, None, None, json!({"pool":pool,"old_config":previous_config,"new_config":config,"joining_node_id":node_id}));
        }
        self.nodes.insert(node_id.clone(), node);
        self.queues.entry(node_id.clone()).or_default();
        let details = node_details(self.nodes.get(&node_id).unwrap());
        self.event(
            if active {
                "node_registered"
            } else {
                "node_announced"
            },
            &node_id,
            Some(if active { "ACTIVE" } else { "REGISTERING" }),
            None,
            details,
        );
        self.schedule();
        let mut response = node_details(self.nodes.get(&node_id).unwrap());
        response["workspace_config"] = self.workspace_configs.get(&pool, Some(&node_id))
            .map_err(|error| ControllerError::bad(error.to_string()))?.to_value();
        if let Some(control) = self.agent_control(&node_id) {
            response
                .as_object_mut()
                .unwrap()
                .insert("agent_control".into(), control);
        }
        Ok(response)
    }

    fn validate_agent_epoch(&self, node_id: &str, body: &Value) -> ControlResult<()> {
        if self.authority.identity_path.is_some() && body.get("agent_epoch").and_then(Value::as_str).filter(|e| !e.is_empty()).is_none() {
            return Err(ControllerError::forbidden("node epoch required"));
        }
        let updates = body.get("workspace_updates").or_else(|| body.get("status").and_then(|status| status.get("workspace_updates")));
        if updates.and_then(Value::as_array).map(|items| !items.is_empty()).unwrap_or(false)
            && body.get("agent_epoch").and_then(Value::as_str).is_none() {
            return Err(ControllerError::forbidden("workspace updates require the current agent epoch"));
        }
        if let Some(expected) = self.nodes.get(node_id).and_then(|node| node.agent_epoch.as_ref()) {
            if let Some(received) = body.get("agent_epoch").filter(|value| !value.is_null()) {
                if received.as_str() != Some(expected.as_str()) {
                    return Err(ControllerError::forbidden("agent epoch does not match current node process"));
                }
            }
        }
        Ok(())
    }

    pub fn heartbeat(&mut self, body: &Value, bearer: Option<&str>) -> ControlResult<Value> {
        let node_id = required_string(body, "node_id")?;
        self.validate_agent_epoch(&node_id, body)?;
        let (pool, allocation_id, removed) = {
            let node = self
                .nodes
                .get(&node_id)
                .ok_or_else(|| ControllerError::missing(&node_id))?;
            (
                node.pool.clone(),
                node.allocation_id.clone(),
                node.state == NodeState::Removed,
            )
        };
        if self.authority.enabled {
            self.authority
                .authorize(bearer, &pool, Role::NodeAgent, Some(&node_id))
                .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        }
        let mut recovered = false;
        if self.authority.identity_path.is_some() { self.managed_seen.insert(node_id.clone(), now()); }
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.last_heartbeat_at = now();
            node.status = body.get("status").cloned().unwrap_or_else(|| json!({}));
            if let Some(status) = body.get("status").and_then(|status| status.get("workspace_config")).filter(|value| value.is_object()) {
                node.workspace_status = status.clone();
            }
            if node.state == NodeState::Unhealthy {
                node.state = NodeState::Active;
                recovered = true;
            }
        }
        if recovered {
            let details = node_details(self.nodes.get(&node_id).unwrap());
            self.event(
                "node_recovered",
                &node_id,
                Some("ACTIVE"),
                Some("heartbeat_recovered"),
                details,
            );
        }
        let updates = body.get("status").and_then(|status| status.get("workspace_updates")).cloned().unwrap_or_else(|| json!([]));
        let workspace_acks = self.apply_workspace_updates(&node_id, &updates)?;
        self.reconcile_agent_status(&node_id);
        self.schedule();
        let max_commands = body
            .get("max_commands")
            .and_then(Value::as_u64)
            .unwrap_or(8) as usize;
        let mut commands = Vec::new();
        if !removed {
            if let Some(queues) = self.queues.get_mut(&node_id) {
                while commands.len() < max_commands {
                    if let Some(control) = queues.controls.pop_front() {
                        commands.push(control);
                    } else if let Some(command_id) = queues.commands.pop_front() {
                        if let Some(command) = self
                            .commands
                            .get_mut(&command_id)
                            .filter(|command| command.state == CommandState::Pending)
                        {
                            command.state = CommandState::Running;
                            command.updated_at = now();
                            commands.push(command.wire());
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        let mut response = json!({"commands": commands});
        response["workspace_acks"] = workspace_acks;
        response["workspace_operations"] = crate::workspace_admission::dispatch(self, &node_id, body["status"]["reclaim_ready"]==true);
        response["workspace_config"] = self.workspace_configs.get(&pool, Some(&node_id))
            .map_err(|error| ControllerError::bad(error.to_string()))?.to_value();
        if removed {
            response["agent_control"] = json!({
                "authoritative": true,
                "action": "shutdown",
                "reason": "allocation_removed",
                "node_id": node_id,
                "allocation_id": allocation_id,
            });
        }
        Ok(response)
    }

    pub fn report_command(&mut self, body: &Value, bearer: Option<&str>) -> ControlResult<Value> {
        let node_id = required_string(body, "node_id")?;
        self.validate_agent_epoch(&node_id, body)?;
        let command_id = required_string(body, "command_id")?;
        let pool = self
            .nodes
            .get(&node_id)
            .ok_or_else(|| ControllerError::missing(&node_id))?
            .pool
            .clone();
        if self.authority.enabled {
            self.authority
                .authorize(bearer, &pool, Role::NodeAgent, Some(&node_id))
                .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        }
        let exit_code = body.get("exit_code").and_then(Value::as_i64).unwrap_or(0) as i32;
        let failure_kind = body
            .get("failure_kind")
            .and_then(Value::as_str)
            .unwrap_or("workload");
        if !matches!(failure_kind, "workload" | "infrastructure") {
            return Err(ControllerError::bad(
                "failure_kind must be workload or infrastructure",
            ));
        }
        if failure_kind == "infrastructure" && exit_code == 0 {
            return Err(ControllerError::bad(
                "infrastructure failure_kind requires a nonzero exit_code",
            ));
        }
        let request_id = self
            .commands
            .get(&command_id)
            .ok_or_else(|| ControllerError::missing(&command_id))?
            .request_id
            .clone();
        if self.requests.get(&request_id).map(|r| r.state.terminal()
            && r.reason.as_deref() == Some("assignment_missing")).unwrap_or(false) {
            let updates = body.get("workspace_updates").cloned().unwrap_or_else(|| json!([]));
            let acks = self.apply_workspace_updates(&node_id, &updates)?;
            let mut response = self.status_value(&self.requests[&request_id]);
            response["workspace_acks"] = acks;
            response["commands"] = json!([]);
            return Ok(response);
        }
        let cancelled = self
            .requests
            .get(&request_id)
            .map(|request| request.state == RequestState::Cancelled)
            .unwrap_or(false)
            || self
                .commands
                .get(&command_id)
                .map(|command| command.state == CommandState::Cancelled)
                .unwrap_or(false);
        let cancel_reason = cancelled.then(|| {
            self.commands
                .get(&command_id)
                .and_then(|command| command.result.get("reason"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    self.requests
                        .get(&request_id)
                        .and_then(|request| request.reason.clone())
                })
                .unwrap_or_else(|| "cancelled".into())
        });
        if let Some(command) = self.commands.get_mut(&command_id) {
            command.state = if cancelled {
                CommandState::Cancelled
            } else if exit_code == 0 {
                CommandState::Completed
            } else {
                CommandState::Failed
            };
            command.updated_at = now();
            command.log_path = body
                .get("log_path")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| command.log_path.clone());
            let mut result = json!({
                "exit_code": exit_code,
                "stdout": body.get("stdout").and_then(Value::as_str).unwrap_or(""),
                "stderr": body.get("stderr").and_then(Value::as_str).unwrap_or(""),
                "failure_kind": failure_kind,
                "artifact_paths": body.get("artifact_paths").cloned().unwrap_or_else(|| json!([])),
            });
            if let Some(reason) = &cancel_reason {
                result["reason"] = json!(reason);
            }
            command.result = result;
        }
        let command_details = self.commands.get(&command_id).unwrap().wire();
        self.event(
            "command_state",
            &command_id,
            Some(if cancelled {
                "CANCELLED"
            } else if exit_code == 0 {
                "COMPLETED"
            } else {
                "FAILED"
            }),
            cancel_reason.as_deref(),
            command_details,
        );
        if !cancelled {
            self.complete_request(&request_id, exit_code, failure_kind, body);
        }
        let updates = body.get("workspace_updates").cloned().unwrap_or_else(|| json!([]));
        let workspace_acks = self.apply_workspace_updates(&node_id, &updates)?;
        self.schedule();
        let mut response = self.status_value(self.requests.get(&request_id).unwrap());
        response["workspace_acks"] = workspace_acks;
        response["workspace_config"] = self.workspace_configs.get(&self.nodes[&node_id].pool, Some(&node_id))
            .map_err(|error| ControllerError::bad(error.to_string()))?.to_value();
        response["commands"] = json!([]);
        if body
            .get("accept_commands")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let heartbeat = self.heartbeat(&json!({"node_id":node_id,"agent_epoch":body.get("agent_epoch"),"max_commands":body.get("max_commands").and_then(Value::as_u64).unwrap_or(8)}), bearer)?;
            response["commands"] = heartbeat["commands"].clone();
            if heartbeat.get("agent_control").is_some() {
                response["agent_control"] = heartbeat["agent_control"].clone();
            }
        }
        Ok(response)
    }

    fn complete_request(
        &mut self,
        request_id: &str,
        exit_code: i32,
        failure_kind: &str,
        body: &Value,
    ) {
        let timestamp = now();
        let execution_session_id = self
            .requests
            .get(request_id)
            .and_then(|request| request.execution_session_id.clone());
        if let Some(request) = self.requests.get_mut(request_id) {
            if request.state.terminal() {
                return;
            }
            request.exit_code = Some(exit_code);
            request.failure_kind = Some(failure_kind.to_string());
            request.stdout = body
                .get("stdout")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            request.stderr = body
                .get("stderr")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            request.log_path = body
                .get("log_path")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| request.log_path.clone());
            request.artifact_paths = body
                .get("artifact_paths")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            request.state = if exit_code == 0 {
                RequestState::Completed
            } else {
                RequestState::Failed
            };
            request.reason = if exit_code == 0 {
                None
            } else if failure_kind == "workload" {
                Some("command_failed".into())
            } else {
                Some("infrastructure_failed".into())
            };
            request.updated_at = json!(timestamp);
            request.ended_at = Some(json!(timestamp));
        }
        if let Some(execution_session_id) = execution_session_id {
            let (node_id, healthy_keep, reason) = {
                let request = self.requests.get(request_id).unwrap();
                let session = self.sessions.get(&execution_session_id).unwrap();
                let node_draining = self
                    .nodes
                    .get(&session.node_id)
                    .map(|node| node.state != NodeState::Active)
                    .unwrap_or(true);
                let healthy = failure_kind != "infrastructure";
                (
                    session.node_id.clone(),
                    healthy
                        && !node_draining
                        && request.spec.reuse
                        && (exit_code == 0 || request.spec.keep_session_on_failure),
                    request.reason.clone(),
                )
            };
            self.release_gpu(&execution_session_id);
            if let Some(session) = self.sessions.get_mut(&execution_session_id) {
                session.active_request_id = None;
                session.last_used_at = timestamp;
                if healthy_keep {
                    session.state = SessionState::CachedIdle;
                    session.gpu_uuids.clear();
                } else if failure_kind == "infrastructure" {
                    session.state = SessionState::Failed;
                } else {
                    session.state = SessionState::Stopped;
                }
            }
            if !healthy_keep {
                self.release_session_resources(&execution_session_id);
                self.queue_stop(&node_id, &execution_session_id);
            }
            let details = session_details(self.sessions.get(&execution_session_id).unwrap());
            let state = serde_json::to_value(self.sessions.get(&execution_session_id).unwrap().state)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string();
            self.event(
                "session_state",
                &execution_session_id,
                Some(&state),
                reason.as_deref(),
                details,
            );
            if healthy_keep && self.sessions[&execution_session_id].suspend_requested {
                self.suspend_session(&execution_session_id,"operator_suspend");
            }
        }
        let details = request_details(self.requests.get(request_id).unwrap());
        let state = serde_json::to_value(self.requests.get(request_id).unwrap().state)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        let reason = self.requests.get(request_id).unwrap().reason.clone();
        self.event(
            "request_state",
            request_id,
            Some(&state),
            reason.as_deref(),
            details,
        );
    }

    pub fn drain_allocation(
        &mut self,
        allocation_id: &str,
        reason: &str,
        token: Option<&str>,
    ) -> ControlResult<Value> {
        let pool = self
            .allocations
            .get(allocation_id)
            .ok_or_else(|| ControllerError::missing(allocation_id))?
            .pool
            .clone();
        self.authority
            .authorize(token, &pool, Role::Admin, None)
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        let node_ids: Vec<String> = self
            .allocations
            .get(allocation_id)
            .unwrap()
            .node_ids
            .iter()
            .cloned()
            .collect();
        for node_id in node_ids {
            if self.authority.identity_path.is_some() {
                self.managed_node_action(token, &node_id, "drain", false, false)?;
            } else { self.drain_node_internal(&node_id, reason); }
        }
        let timestamp = now();
        if let Some(allocation) = self.allocations.get_mut(allocation_id) {
            allocation.state = AllocationState::Draining;
            allocation.reason = Some(reason.to_string());
            allocation.updated_at = timestamp;
        }
        let value = allocation_details(self.allocations.get(allocation_id).unwrap());
        self.event(
            "allocation_state",
            allocation_id,
            Some("DRAINING"),
            Some(reason),
            value.clone(),
        );
        Ok(value)
    }

    pub fn remove_allocation(
        &mut self,
        allocation_id: &str,
        force: bool,
        token: Option<&str>,
    ) -> ControlResult<Value> {
        if self.authority.identity_path.is_some() {
            return Err(ControllerError::bad("persistent allocations retire through node drain/exit; direct removal is not allowed"));
        }
        let pool = self
            .allocations
            .get(allocation_id)
            .ok_or_else(|| ControllerError::missing(allocation_id))?
            .pool
            .clone();
        self.authority
            .authorize(token, &pool, Role::Admin, None)
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        let node_ids: Vec<String> = self
            .allocations
            .get(allocation_id)
            .unwrap()
            .node_ids
            .iter()
            .cloned()
            .collect();
        for node_id in node_ids {
            let active: Vec<String> = self
                .nodes
                .get(&node_id)
                .map(|node| node.active_sessions.clone())
                .unwrap_or_default();
            if !force
                && active.iter().any(|execution_session_id| {
                    self.sessions
                        .get(execution_session_id)
                        .map(|session| session.state == SessionState::RunningCommand)
                        .unwrap_or(false)
                })
            {
                return Err(ControllerError::bad("allocation has active sessions"));
            }
            for execution_session_id in active {
                self.release_gpu(&execution_session_id);
                self.release_session_resources(&execution_session_id);
                if let Some(session) = self.sessions.get_mut(&execution_session_id) {
                    session.state = SessionState::Stopped;
                    session.active_request_id = None;
                }
                self.queue_stop(&node_id, &execution_session_id);
            }
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.state = NodeState::Removed;
                node.active_sessions.clear();
                node.used_cpu_cores = 0.0;
                node.used_memory_gb = 0.0;
            }
        }
        let timestamp = now();
        if let Some(allocation) = self.allocations.get_mut(allocation_id) {
            allocation.state = AllocationState::Removed;
            allocation.reason = Some("allocation_removed".into());
            allocation.updated_at = timestamp;
        }
        let value = allocation_details(self.allocations.get(allocation_id).unwrap());
        self.event(
            "allocation_state",
            allocation_id,
            Some("REMOVED"),
            Some("allocation_removed"),
            value.clone(),
        );
        self.schedule();
        Ok(value)
    }

    pub fn drain_node(
        &mut self,
        node_id: &str,
        reason: &str,
        token: Option<&str>,
    ) -> ControlResult<Value> {
        if self.authority.identity_path.is_some() {
            return self.managed_node_action(token, node_id, "drain", false, true);
        }
        let pool = self
            .nodes
            .get(node_id)
            .ok_or_else(|| ControllerError::missing(node_id))?
            .pool
            .clone();
        let node_agent = self
            .authority
            .authorize(token, &pool, Role::NodeAgent, Some(node_id));
        let admin = self.authority.authorize(token, &pool, Role::Admin, None);
        let owner = self.authority.identity().and_then(|store| store.context("legacy authority")?
            .owned_node(token.unwrap_or(""), node_id, now()));
        if node_agent.is_err() && admin.is_err() && owner.is_err() {
            return Err(ControllerError::forbidden("token cannot drain this node"));
        }
        self.drain_node_internal(node_id, reason);
        if self.authority.identity_path.is_some() {
            return Ok(json!({"node_id":node_id,"state":self.nodes[node_id].state}));
        }
        Ok(node_details(self.nodes.get(node_id).unwrap()))
    }

    fn drain_node_internal(&mut self, node_id: &str, reason: &str) {
        let cached: Vec<String> = self
            .nodes
            .get(node_id)
            .map(|node| {
                node.active_sessions
                    .iter()
                    .filter(|execution_session_id| {
                        self.sessions
                            .get(*execution_session_id)
                            .map(|session| {
                                matches!(
                                    session.state,
                                    SessionState::CachedIdle | SessionState::WarmIdle
                                )
                                && (self.authority.identity_path.is_none() || !self.commands.values().any(|c|
                                    c.execution_session_id == session.execution_session_id && matches!(c.state,CommandState::Pending|CommandState::Running)))
                            })
                            .unwrap_or(false)
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        for execution_session_id in cached {
            self.release_gpu(&execution_session_id);
            self.release_session_resources(&execution_session_id);
            if let Some(session) = self.sessions.get_mut(&execution_session_id) {
                session.state = SessionState::Stopped;
            }
            self.queue_stop(node_id, &execution_session_id);
        }
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.state = NodeState::Draining;
        }
        if let Some(node) = self.nodes.get(node_id) {
            self.event(
                "node_state",
                node_id,
                Some("DRAINING"),
                Some(reason),
                node_details(node),
            );
        }
    }

    pub fn set_pool_config(
        &mut self,
        pool: &str,
        config: PoolConfig,
        token: Option<&str>,
    ) -> ControlResult<Value> {
        if self.authority.identity_path.is_some() {
            let pool = self.authority.identity().and_then(|s|s.context("missing identity")?.pool_id(pool)).map_err(|e|ControllerError::bad(e.to_string()))?;
            self.managed_pool_edit(token, &pool, &json!({"config":config}))?;
            return Ok(json!({"pool":pool,"pool_config":config,"requires_matching_join":true}));
        }
        self.authority
            .authorize(token, pool, Role::Admin, None)
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        let blockers = self.pool_blockers(pool);
        if blockers
            .values()
            .any(|items| !items.as_array().unwrap().is_empty())
        {
            return Err(ControllerError {
                status: 400,
                message: "pool configuration has blockers".into(),
                payload: Some(json!({"pool":pool,"blockers":blockers})),
            });
        }
        let old = self.pool_configs.insert(pool.to_string(), config.clone());
        self.event(
            if old.is_some() {
                "pool_config_replaced"
            } else {
                "pool_config_set"
            },
            pool,
            None,
            None,
            json!({"pool":pool,"old_config":old,"new_config":config,"actor":"admin"}),
        );
        Ok(json!({"pool":pool,"pool_config":config,"requires_matching_join":true}))
    }

    pub fn get_workspace_config(&self, pool: &str, node_id: Option<&str>, token: Option<&str>) -> ControlResult<Value> {
        let managed_pool = self.authority.identity().and_then(|store| store.map(|s|s.pool_id(pool)).transpose())
            .map_err(|e|ControllerError::bad(e.to_string()))?;
        let pool = managed_pool.as_deref().unwrap_or(pool);
        self.authority.authorize(token, pool, Role::Admin, None)
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        let desired = self.workspace_configs.get(pool, node_id)
            .map_err(|error| ControllerError::bad(error.to_string()))?;
        if node_id.and_then(|id| self.nodes.get(id)).map(|node| node.pool != pool).unwrap_or(false) {
            return Err(ControllerError::bad("workspace node override belongs to another pool"));
        }
        let nodes: Vec<Value> = self.nodes.values()
            .filter(|node| node.pool == pool && node_id.map(|id| node.node_id == id).unwrap_or(true))
            .map(|node| {
                let status = &node.workspace_status;
                let ready = status["state"] == "applied" && status["applied_version"] == desired.version;
                json!({"node_id":node.node_id,"desired_version":desired.version,"applied_version":status["applied_version"],
                    "state":if ready {"applied"} else if !status["error"].is_null() {"failed"} else {"pending"},
                    "error":status["error"],"node_state":node.state,"applied_config":status["config"],
                    "agent_epoch":node.agent_epoch,"last_heartbeat_at":node.last_heartbeat_at})
            })
            .collect();
        Ok(json!({"pool":pool,"node_id":node_id,"desired_config":desired.to_value(),
                  "overrides":self.workspace_configs.record(pool).map_err(|error| ControllerError::bad(error.to_string()))?,
                  "nodes":nodes,"maintenance_defaults":crate::workspace_fs::MaintenancePolicy::default()}))
    }

    pub fn set_workspace_config_unflushed(&mut self, pool: &str, changes: &serde_json::Map<String, Value>,
                                         node_id: Option<&str>, token: Option<&str>) -> ControlResult<(Value, Option<DurableReceipt>)> {
        let managed_pool = self.authority.identity().and_then(|store| store.map(|s|s.pool_id(pool)).transpose())
            .map_err(|e|ControllerError::bad(e.to_string()))?;
        let pool = managed_pool.as_deref().unwrap_or(pool);
        self.get_workspace_config(pool, node_id, token)?;
        let mut candidate = self.workspace_configs.clone();
        let record = candidate.update(pool, changes, node_id)
            .map_err(|error| ControllerError::bad(error.to_string()))?;
        let receipt = self.durable_event("workspace_config_set", pool, None, Some("operator_set"),
                                        json!({"pool":pool,"new_config":record,"actor":"admin"}))?;
        self.workspace_configs = candidate;
        Ok((self.get_workspace_config(pool, node_id, token)?, receipt))
    }

    pub fn workspace_status(&self, id: &str, token: Option<&str>) -> ControlResult<Value> {
        let workspace = self.workspaces.get(id).ok_or_else(|| ControllerError::missing("workspace does not exist"))?;
        if self.authority.identity_path.is_some() {
            self.authority.authorize_owner(token, workspace["pool"].as_str().unwrap_or(""), workspace["user_id"].as_str().unwrap_or(""))
                .map_err(|e| ControllerError::forbidden(e.to_string()))?;
        } else {
            self.authority.authorize(token, workspace["pool"].as_str().unwrap_or(""), Role::Viewer, None)
            .or_else(|_| self.authority.authorize(token, workspace["pool"].as_str().unwrap_or(""), Role::Submitter, workspace["user_id"].as_str()))
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        }
        let mut result=workspace.clone();
        if result.get("auto_reclaim").is_none() {result["auto_reclaim"]=json!("inherit");}
        let config=self.workspace_configs.get(workspace["pool"].as_str().unwrap_or(""),workspace["node_id"].as_str())
            .map_err(|e|ControllerError::bad(e.to_string()))?;
        result["auto_reclaim_enabled"]=json!(result["auto_reclaim"]=="inherit" && config.maintenance.unwrap_or_default().reclaim_on_release);
        Ok(result)
    }

    pub fn workspace_operation(&mut self, id: &str, action: &str, token: Option<&str>, parameters: &Value) -> ControlResult<Value> {
        self.workspace_status(id, token)?;
        let mut record = self.workspaces[id].clone();
        self.authority.authorize(token, record["pool"].as_str().unwrap_or(""), Role::Admin, None)
            .or_else(|_| self.authority.authorize_owner(token, record["pool"].as_str().unwrap_or(""), record["user_id"].as_str().unwrap_or("")))
            .map_err(|error| ControllerError::forbidden(error.to_string()))?;
        let execution_session_id = record["execution_session_id"].as_str().unwrap_or("").to_string();
        if action == "set" {
            if !matches!(parameters["auto_reclaim"].as_str(),Some("off"|"inherit"))
                || parameters.as_object().map(|p|p.keys().any(|k|k!="auto_reclaim")).unwrap_or(true) {
                return Err(ControllerError::bad("set requires auto_reclaim off/inherit and no maintenance parameters"));
            }
            record["auto_reclaim"]=parameters["auto_reclaim"].clone();
            record["reclaim_candidate"]=Value::Null;
            record["reclaim_status"]=json!(if record["auto_reclaim"]=="off" {"workspace_disabled"} else {"awaiting_evaluation"});
            self.event("workspace_state",id,record["state"].as_str(),Some("auto_reclaim_set"),record.clone());
            self.workspaces.insert(id.into(),record);
            return self.workspace_status(id,token);
        }
        if parameters.get("auto_reclaim").is_some() {return Err(ControllerError::bad("auto_reclaim applies only to set"));}
        if action == "release" {
            if let Some(session) = self.sessions.get(&execution_session_id) {
                if session.active_request_id.is_some() { return Err(ControllerError::bad("cancel or finish the active command before releasing its workspace")); }
                if record["state"] != "retained" && record["state"] != "deleted" { self.stop_session(&execution_session_id, "workspace_release"); }
            }
            return self.workspace_status(id, token);
        }
        if !matches!(action, "check" | "delete" | "grow" | "reclaim" | "shrink" | "recover") { return Err(ControllerError::bad("unknown workspace operation")); }
        if action=="delete" && self.sessions.values().any(|s|s.execution_session_id!=execution_session_id&&s.state!=SessionState::Purged
            && s.workspace.as_ref().map(|w|w["id"]==id).unwrap_or(false)) {
            return Err(ControllerError::bad("workspace is referenced by another execution; purge that execution first"));
        }
        if let Some(operation_id)=parameters.get("operation_id").filter(|v|!v.is_null()) {
            if record["operation"]["id"]!=*operation_id || record["operation"]["action"]!=action || record["operation"]["target_bytes"]!=parameters["target_bytes"] {
                return Err(ControllerError::bad("operation id is stale or parameters differ"));
            }
            return Ok(record);
        }
        if record["operation"]["state"] == "pending" {
            if record["operation"]["action"] != action || record["operation"]["target_bytes"]!=parameters["target_bytes"] { return Err(ControllerError::bad("another workspace operation is pending")); }
            return Ok(record);
        }
        let modifying=matches!(action,"grow"|"reclaim"|"shrink"|"recover");
        let target=parameters.get("target_bytes").filter(|v|!v.is_null());
        if modifying {
            if self.nodes.get(record["node_id"].as_str().unwrap_or("")).map(|n|n.workspace_status["maintenance_protocol"]!=1).unwrap_or(true) {
                return Err(ControllerError::bad("node does not advertise verified workspace maintenance support"));
            }
            let desired=self.workspace_configs.get(record["pool"].as_str().unwrap_or(""),record["node_id"].as_str()).map_err(|e|ControllerError::bad(e.to_string()))?;
            if !self.nodes.get(record["node_id"].as_str().unwrap_or("")).map(|n|n.workspace_status["state"]=="applied"&&n.workspace_status["applied_version"]==desired.version).unwrap_or(false){return Err(ControllerError::bad("workspace maintenance configuration is not yet applied on node"));}
            if matches!(action,"grow"|"shrink") {
                let size=target.and_then(Value::as_u64).ok_or_else(||ControllerError::bad("target capacity must be a positive integer"))?;
                let maximum=self.workspace_configs.get(record["pool"].as_str().unwrap_or(""),record["node_id"].as_str()).map_err(|e|ControllerError::bad(e.to_string()))?.max_image_size_bytes;
                if size<32*1024*1024 || size>maximum || size%(1024*1024)!=0 {return Err(ControllerError::bad("target capacity must be whole MiB within deployment limits"));}
                let current=record["current_image_size_bytes"].as_u64().or_else(||record["config"]["image_size_bytes"].as_u64()).ok_or_else(||ControllerError::bad("missing image capacity"))?;
                if (action=="grow"&&size<=current)||(action=="shrink"&&size>=current){return Err(ControllerError::bad("target must change capacity in the requested direction"));}
            } else if target.is_some() {return Err(ControllerError::bad("this operation does not accept target capacity"));}
        } else if target.is_some() {return Err(ControllerError::bad("this operation does not accept target capacity"));}
        let boundary=(action=="grow" && record["state"]=="mounted")
            || (action=="recover" && self.sessions.get(&execution_session_id).map(|s|!s.state.terminal() && s.state != SessionState::Suspended).unwrap_or(false));
        if !boundary && !matches!(record["state"].as_str(), Some("retained" | "needs_check" | "deleted" | "maintenance" | "needs_recovery")) {
            return Err(ControllerError::bad("workspace must be released before offline maintenance"));
        }
        if !boundary && self.sessions.get(&execution_session_id).map(|session| !session.state.terminal() && session.state != SessionState::Suspended).unwrap_or(false) {
            return Err(ControllerError::bad("workspace still has a session owner"));
        }
        let sequence=record["operation_sequence"].as_u64().unwrap_or(0).checked_add(1).filter(|n|*n<=i64::MAX as u64).ok_or_else(||ControllerError::bad("workspace operation sequence exhausted"))?;
        record["operation_sequence"]=json!(sequence);
        record["operation"] = json!({"id":uuid::Uuid::new_v4().simple().to_string(),"action":action,"state":"pending","sequence":sequence,"dispatch_granted":false,"requested_at":now()});
        if modifying {
            if action!="recover" && record["state"]!="retained" && !boundary {return Err(ControllerError::bad("modifying maintenance requires a clean retained workspace"));}
            record["operation"]["target_bytes"]=parameters["target_bytes"].clone();
        let config=self.workspace_configs.get(record["pool"].as_str().unwrap_or(""),record["node_id"].as_str()).map_err(|e|ControllerError::bad(e.to_string()))?;
            record["operation"]["policy"]=json!(config.maintenance.unwrap_or_default());record["operation"]["source"]=json!("explicit");
            record["operation"]["boundary"]=json!(boundary);record["operation"]["requested_at"]=json!(now());
            record["operation"]["phase"]=json!(if boundary {"waiting_boundary"}else{"requested"});
            record["operation"]["max_image_size_bytes"]=json!(config.max_image_size_bytes);
            record["operation"]["policy_version"]=json!(config.version);
        }
        self.event("workspace_state", id, record["state"].as_str(), Some("maintenance_requested"), record.clone());
        self.workspaces.insert(id.into(), record.clone());
        Ok(record)
    }

    pub(crate) fn automatic_workspace_operation(&self, record:&mut Value) {
        if record["operation"]["state"]=="pending" || !self.nodes.get(record["node_id"].as_str().unwrap_or("")).map(|n|n.workspace_status["maintenance_protocol"]==1).unwrap_or(false) {return;}
        let Ok(config)=self.workspace_configs.get(record["pool"].as_str().unwrap_or(""),record["node_id"].as_str()) else{return;};
        if !self.nodes.get(record["node_id"].as_str().unwrap_or("")).map(|n|n.workspace_status["state"]=="applied"&&n.workspace_status["applied_version"]==config.version).unwrap_or(false){return;}
        let policy=config.maintenance.clone().unwrap_or_default();let time=now();
        let growth=match crate::workspace::automatic_grow_target(record,&config,time) {
            Ok(value)=>value,Err(error)=>{record["maintenance_warning"]=json!(error.to_string());record["last_maintenance_at"]=json!(time);return;}
        };
        if growth.is_none() {
            if record["state"]=="retained" && record["reclaim_release_generation"]!=record["usage_generation"] {
                record["reclaim_release_generation"]=record["usage_generation"].clone();
                record["reclaim_release_observed_at"]=json!(time);
            }
            let (candidate, reason)=crate::workspace_reclaim::candidate(record,&policy,time);
            record["reclaim_candidate"]=candidate.unwrap_or(Value::Null);
            record["reclaim_status"]=json!(reason);
        }
        if growth.is_some() {
            let Some(sequence)=record["operation_sequence"].as_u64().unwrap_or(0).checked_add(1).filter(|n|*n<=i64::MAX as u64) else {record["maintenance_warning"]=json!("workspace operation sequence exhausted");return;};
            record["operation_sequence"]=json!(sequence);
            let target=growth.as_ref().map(|g|g["target_bytes"].clone()).unwrap_or(Value::Null);
            let trigger=growth.as_ref().map(|g|g["reason"].clone()).unwrap_or(json!("release"));
            record["operation"]=json!({"id":uuid::Uuid::new_v4().simple().to_string(),"sequence":sequence,"action":if growth.is_some(){"grow"}else{"reclaim"},
                "state":"pending","target_bytes":target,"policy":policy,"source":"automatic","boundary":record["state"]=="mounted","requested_at":time,"trigger":trigger,
                "max_image_size_bytes":config.max_image_size_bytes,
                "policy_version":config.version,
                "dispatch_granted":false,
                "phase":if record["state"]=="mounted" {"waiting_boundary"} else {"requested"}});
            record["last_maintenance_at"]=json!(time);
        }
    }

    fn workspace_blocks_session(&self,session:&SessionRecord)->bool {
        if session.suspend_requested || matches!(session.state,SessionState::Suspending|SessionState::Purging) {return true;}
        session.workspace.as_ref().and_then(|w|w["id"].as_str()).and_then(|id|self.workspaces.get(id))
            .map(|w|w["operation"]["state"]=="pending"||matches!(w["state"].as_str(),Some("maintenance"|"needs_recovery"))
                ||(w["operation"]["boundary"]==true&&w["operation"]["state"]=="failed")
                ||(w["maintenance"].is_object()&&!self.nodes.get(&session.node_id).map(|n|n.workspace_status["maintenance_protocol"]==1).unwrap_or(false))).unwrap_or(false)
    }

    fn workspace_blocks_request(&self,spec:&WorkloadSpec)->bool {
        spec.reuse && self.sessions.values().any(|s|!s.state.terminal()&&s.matches_reuse_identity(spec)&&self.workspace_blocks_session(s))
    }

    fn failed_workspace_for_request(&self,spec:&WorkloadSpec)->Option<String> {
        if spec.workspace.id.as_ref().and_then(|id|self.workspaces.get(id)).map(|w|w["node_retired"]==true).unwrap_or(false) {
            return Some("workspace_node_unavailable".into());
        }
        let mut candidates=Vec::new();
        if let Some(record)=spec.workspace.id.as_ref().and_then(|id|self.workspaces.get(id)){candidates.push(record);}
        if spec.reuse {
            for session in self.sessions.values().filter(|s|!s.state.terminal()&&s.matches_reuse_identity(spec)) {
                if let Some(record)=session.workspace.as_ref().and_then(|w|w["id"].as_str()).and_then(|id|self.workspaces.get(id)){candidates.push(record);}
            }
        }
        candidates.into_iter().find(|w|w["operation"]["state"]=="failed"&&(w["operation"]["boundary"]==true||matches!(w["state"].as_str(),Some("maintenance"|"needs_recovery"))))
            .map(|w|w["operation"]["error"].as_str().or_else(||w["last_error"].as_str()).unwrap_or("workspace maintenance requires recovery").to_string())
    }

    fn reserve_workspace(&mut self, spec: &WorkloadSpec, node: &str, session: &str) -> Result<Value> {
        let id = spec.workspace.id.clone().unwrap_or_else(crate::workspace_fs::new_workspace_id);
        let mut record = if spec.workspace.id.is_none() {
            json!({"id":id,"pool":spec.pool,"user_id":spec.user_id,"node_id":node,
                   "config":self.workspace_configs.get(&spec.pool, Some(node))?.to_value(),
                   "revision":0,"path":null,"last_error":null})
        } else {
            let record = self.workspaces.get(&id).context("workspace does not exist")?;
            if record["state"] != "retained" || record["node_id"] != node || record["operation"]["state"] == "pending" { bail!("workspace is unavailable on the selected node"); }
            record.clone()
        };
        record["state"] = json!("pending"); record["execution_session_id"] = json!(session); record["target"] = json!(spec.workspace.target);
        self.event("workspace_state", &id, Some("pending"), None, record.clone());
        self.workspaces.insert(id, record.clone());
        record["create"] = json!(spec.workspace.id.is_none());
        Ok(record)
    }

    pub fn apply_workspace_updates(&mut self, node_id: &str, updates: &Value) -> ControlResult<Value> {
        let node = self.nodes.get(node_id).ok_or_else(|| ControllerError::missing(node_id))?;
        let pool = node.pool.clone();
        let updates = updates.as_array().filter(|items| items.len() <= 64)
            .ok_or_else(|| ControllerError::bad("workspace updates must be an array of at most 64 entries"))?;
        let mut acknowledgements = serde_json::Map::new();
        for update in updates {
            let id = update.get("workspace_id").and_then(Value::as_str)
                .ok_or_else(|| ControllerError::bad("workspace update requires workspace_id"))?;
            let Some(record) = self.workspaces.get(id) else { continue };
            if record["node_id"] != node_id || record["pool"] != pool {
                return Err(ControllerError::forbidden("workspace update belongs to another node or pool"));
            }
            let revision = update.get("revision").and_then(Value::as_u64).filter(|value| *value <= i64::MAX as u64)
                .ok_or_else(|| ControllerError::bad("invalid workspace revision"))?;
            let previous = record["revision"].as_u64().unwrap_or(0);
            let state = update.get("state").and_then(Value::as_str).unwrap_or("");
            let session = update.get("execution_session_id").and_then(Value::as_str)
                .ok_or_else(|| ControllerError::bad("workspace update requires execution_session_id"))?;
            let maintenance = record["operation"]["state"] == "pending" && update["operation_result"].is_object() && update["operation_result"]["id"] == record["operation"]["id"];
            let progress=update.get("operation_progress");
            if progress.is_some() && update["operation_progress"]["id"]!=record["operation"]["id"] {continue;}
            if revision <= previous && !maintenance && !(revision == previous && record["state"] == "pending" && matches!(state, "absent" | "needs_check" | "retained")) {
                if !update["operation_result"].is_object() && revision == previous && update["execution_session_id"] == record["execution_session_id"] && update["user_id"] == record["user_id"] && WorkspaceConfig::from_value(update["config"].clone()).ok().zip(WorkspaceConfig::from_value(record["config"].clone()).ok()).map(|(a,b)|a==b).unwrap_or(false) {
                    let mut refreshed=record.clone();refreshed["space"]=update.get("space").cloned().unwrap_or(Value::Null);
                    self.automatic_workspace_operation(&mut refreshed);
                    if refreshed["operation"]!=record["operation"] {self.event("workspace_state",id,refreshed["state"].as_str(),Some("automatic_maintenance_requested"),refreshed.clone());}
                    self.workspaces.insert(id.into(),refreshed);
                }
                let mut ack=json!({"revision":previous,"execution_session_id":session,"operation_id":update["operation_result"]["id"]});
                if let Some(value)=update.get("telemetry_id"){ack["telemetry_id"]=value.clone();}
                acknowledgements.insert(id.into(),ack); continue;
            }
            if update["execution_session_id"] != record["execution_session_id"] || update["user_id"] != record["user_id"] { continue; }
            if !WorkspaceConfig::from_value(update["config"].clone()).ok().zip(WorkspaceConfig::from_value(record["config"].clone()).ok()).map(|(a,b)|a==b).unwrap_or(false) { return Err(ControllerError::bad("workspace creation configuration must not change")); }
            if !matches!(state, "creating" | "mounting" | "mounted" | "retained" | "needs_check" | "deleting" | "deleted" | "absent" | "maintenance" | "needs_recovery") {
                return Err(ControllerError::bad("invalid workspace lifecycle state"));
            }
            if record["state"] == "deleted" && state != "deleted" { return Err(ControllerError::bad("deleted workspaces must not be resurrected")); }
            let mut updated = record.clone();
            if progress.is_some() && record["operation"]["state"]=="pending" {updated["operation"]["phase"]=update["operation_progress"]["phase"].clone();}
            if maintenance {
                if revision < previous || !matches!(update["operation_result"]["state"].as_str(), Some("completed" | "failed")) { return Err(ControllerError::bad("invalid workspace operation result")); }
                updated["operation"] = update["operation_result"].clone();
                updated["last_maintenance_at"]=json!(now());
            }
            updated["state"] = json!(state); updated["revision"] = json!(revision);
            updated["path"] = update.get("path").cloned().unwrap_or(Value::Null);
            updated["last_error"] = update.get("last_error").cloned().unwrap_or(Value::Null);
            updated["space"] = update.get("space").cloned().unwrap_or(Value::Null);
            updated["timings"] = update.get("timings").cloned().unwrap_or(Value::Null);
            for field in ["current_image_size_bytes","geometry","generation","maintenance","usage_generation",
                          "reclaim_reference","reclaim_sample","reclaim_sample_error"] {
                if let Some(value)=update.get(field) {updated[field]=value.clone();}
            }
            updated["maintenance_deferred"]=json!(update["maintenance_deferred"].as_bool().unwrap_or(false));
            if maintenance {crate::workspace_admission::complete(&mut updated,now());}
            if !maintenance {self.automatic_workspace_operation(&mut updated);}
            self.event("workspace_state", id, Some(state), None, updated.clone());
            self.workspaces.insert(id.into(), updated);
            if maintenance {if let Some(owner)=self.sessions.get_mut(session){owner.last_used_at=now();}}
            if state == "retained" && self.sessions.get(session).map(|session| matches!(session.state, SessionState::CachedIdle | SessionState::WarmIdle)).unwrap_or(false) {
                self.stop_session(session, "workspace_unmounted");
            }
            let mut ack=json!({"revision":revision,"execution_session_id":session,"operation_id":update["operation_result"]["id"]});
            if let Some(value)=update.get("telemetry_id"){ack["telemetry_id"]=value.clone();}
            acknowledgements.insert(id.into(),ack);
        }
        Ok(Value::Object(acknowledgements))
    }

    pub fn utilization(&self, pool: &str) -> Value {
        let visible: Vec<&NodeRecord> = self
            .nodes
            .values()
            .filter(|node| node.pool == pool && node.state != NodeState::Removed)
            .collect();
        let active: Vec<&NodeRecord> = visible
            .iter()
            .copied()
            .filter(|node| node.state == NodeState::Active)
            .collect();
        let gpus_total: usize = visible
            .iter()
            .map(|node| node.resources.gpu_uuids.len())
            .sum();
        let gpus_schedulable: usize = active
            .iter()
            .map(|node| node.resources.gpu_uuids.iter().filter(|g|!self.node_purge_quarantine.contains(*g)).count())
            .sum();
        let lease_count = self
            .gpu_ledger
            .values()
            .filter(|entry| entry.state() != GpuLeaseState::Free)
            .count();
        let queued = self
            .requests
            .values()
            .filter(|request| request.spec.pool == pool && request.state == RequestState::Queued)
            .count();
        let running = self
            .requests
            .values()
            .filter(|request| request.spec.pool == pool && request.state == RequestState::Running)
            .count();
        let config = self.pool_configs.get(pool).cloned().unwrap_or_default();
        let slots_total = gpus_total * config.gpu_slots;
        let slots_leased: usize = self
            .gpu_ledger
            .values()
            .map(|entry| {
                if config.pool_mode == "shared" {
                    entry.shared.len()
                } else {
                    usize::from(entry.exclusive.is_some())
                }
            })
            .sum();
        let slots_available = if self.node_purge_quarantine.is_empty() {slots_total.saturating_sub(slots_leased)} else {
            active.iter().map(|node|node.resources.gpu_uuids.iter().map(|gpu| {
                if self.node_purge_quarantine.contains(gpu) {return 0;}
                let entry=self.gpu_ledger.get(gpu).cloned().unwrap_or_default();
                if entry.exclusive.is_some() {0} else {node.pool_config.gpu_slots.saturating_sub(entry.shared.len())}
            }).sum::<usize>()).sum()
        };
        json!({
            "pool": pool,
            "nodes": visible.len(),
            "active_nodes": active.len(),
            "registering_nodes": visible.iter().filter(|node| node.state == NodeState::Registering).count(),
            "gpus_total": gpus_total,
            "gpus_schedulable": gpus_schedulable,
            "gpus_leased": lease_count,
            "gpus_available": active.iter().map(|node|node.resources.gpu_uuids.iter().filter(|uuid|!self.node_purge_quarantine.contains(*uuid)&&self.gpu_ledger.get(*uuid).map(|entry|entry.state()==GpuLeaseState::Free).unwrap_or(true)).count()).sum::<usize>(),
            "gpu_slots_total": slots_total,
            "gpu_slots_leased": slots_leased,
            "gpu_slots_available": slots_available,
            "cpu_total": visible.iter().map(|node| node.resources.cpu_cores).sum::<f64>(),
            "cpu_used": visible.iter().map(|node| node.used_cpu_cores).sum::<f64>(),
            "cpu_schedulable": active.iter().map(|node| node.resources.cpu_cores-node.used_cpu_cores).sum::<f64>(),
            "memory_total": visible.iter().map(|node| node.resources.memory_gb).sum::<f64>(),
            "memory_used": visible.iter().map(|node| node.used_memory_gb).sum::<f64>(),
            "memory_schedulable": active.iter().map(|node| node.resources.memory_gb-node.used_memory_gb).sum::<f64>(),
            "queued_requests": queued,
            "running_requests": running,
            "pool_mode": config.pool_mode,
            "pool_config": config,
            "requires_matching_join": self.authority.identity_path.is_some(),
            "node_states": visible.iter().map(|node| node_summary(node,&self.requests,&self.sessions,&self.gpu_ledger,&self.node_purge_quarantine)).collect::<Vec<_>>(),
            "physical": {
                "gpus_total": gpus_total,
                "gpus_free": self.gpu_ledger.iter().filter(|(uuid,entry)| !self.node_purge_quarantine.contains(*uuid)&&entry.state()==GpuLeaseState::Free).count(),
                "gpus_exclusive": self.gpu_ledger.values().filter(|entry| entry.state()==GpuLeaseState::Exclusive).count(),
                "gpus_shared": self.gpu_ledger.values().filter(|entry| entry.state()==GpuLeaseState::Shared).count(),
                "exclusive_leases": self.gpu_ledger.values().filter(|entry| entry.exclusive.is_some()).count(),
                "shared_slots_leased": self.gpu_ledger.values().map(|entry| entry.shared.len()).sum::<usize>(),
                "lease_count": lease_count,
                "gpus": self.gpu_ledger.iter().filter(|(uuid,_)|!self.node_purge_quarantine.contains(*uuid)||visible.iter().any(|n|n.resources.gpu_uuids.contains(*uuid)))
                    .map(|(uuid,entry)| json!({"gpu_uuid":uuid,"state":if self.node_purge_quarantine.contains(uuid){json!("QUARANTINED")}else{json!(entry.state())},"shared_refcount":entry.shared.len()})).collect::<Vec<_>>(),
            }
        })
    }

    pub fn schedule(&mut self) {
        if self.authority.identity_path.is_some() && self.managed_reconcile().is_err() {
            return;
        }
        self.refresh_deadlines_at(now());
        let mut pending: Vec<String> = self
            .requests
            .values()
            .filter(|request| request.state == RequestState::Queued)
            .map(|request| request.request_id.clone())
            .collect();
        pending.sort_by_key(|id| {
            let request = self.requests.get(id).unwrap();
            (
                -request.spec.priority,
                self.active_gpu_lease_count(Some(&request.spec.user_id)),
                request.sequence,
            )
        });
        let mut authority_blocked = BTreeSet::new();
        for request_id in &pending {
            if self.authority.identity_path.is_some() {
                let request = &self.requests[request_id];
                let allowed = self.authority.identity().and_then(|s| s.context("missing identity")?.pending_allowed(
                    &request.spec.user_id, &request.spec.pool, request.admission_token_id.as_deref()));
                if allowed.is_err() {
                    self.set_queued_reason(request_id, "authority_unavailable");
                    authority_blocked.insert(request_id.clone());
                    continue;
                }
                if !allowed.unwrap() {
                    let request = self.requests.get_mut(request_id).unwrap();
                    request.state = RequestState::Failed;
                    request.reason = Some("permission_revoked".into());
                    request.failure_kind = Some("workload".into());
                    request.updated_at = json!(now());
                    request.ended_at = Some(request.updated_at.clone());
                    let details = request_details(request);
                    self.event("request_state", request_id, Some("FAILED"), Some("permission_revoked"), details);
                    continue;
                }
            }
            if self.failed_workspace_for_request(&self.requests[request_id].spec).is_none()&&!self.workspace_blocks_request(&self.requests[request_id].spec) {self.try_reuse(request_id);}
        }
        for request_id in pending {
            if authority_blocked.contains(&request_id) { continue; }
            if self.requests[&request_id].state != RequestState::Queued {
                continue;
            }
            let spec = self.requests[&request_id].spec.clone();
            if self.try_resume_retained(&request_id) { continue; }
            if let Some(error)=self.failed_workspace_for_request(&spec) {
                let reason=if error=="workspace_node_unavailable" {"workspace_node_unavailable"}else{"workspace_maintenance_failed"};
                let request=self.requests.get_mut(&request_id).unwrap();
                request.state=RequestState::Failed;request.reason=Some(reason.into());request.failure_kind=Some("infrastructure".into());
                request.stderr=error;request.updated_at=json!(now());request.ended_at=Some(request.updated_at.clone());
                let details=request_details(request);
                self.event("request_state",&request_id,Some("FAILED"),Some(reason),details);
                continue;
            }
            if self.workspace_blocks_request(&spec) {self.set_queued_reason(&request_id,"waiting_for_session");continue;}
            if self.would_exceed_active_gpu_quota(&spec) {
                self.set_queued_reason(&request_id, "quota_exceeded");
                continue;
            }
            if self.would_exceed_pool_session_cap(&spec) {
                self.set_queued_reason(&request_id, "pool_session_limit");
                continue;
            }
            if self.try_new_session(&request_id) {
                continue;
            }
            let spec = &self.requests[&request_id].spec;
            let pool_nodes: Vec<&NodeRecord> = self
                .nodes
                .values()
                .filter(|node| node.pool == spec.pool)
                .collect();
            let reason = if pool_nodes.is_empty() {
                "pool_empty"
            } else if pool_nodes
                .iter()
                .any(|node| node.state == NodeState::Registering)
                && !pool_nodes
                    .iter()
                    .any(|node| node.state == NodeState::Active)
            {
                "node_probing"
            } else if pool_nodes.iter().any(|node| {
                node.state == NodeState::Active
                    && !Self::node_accepts_command(
                        node,
                        spec.resources.admission_timeout_seconds(),
                        now(),
                    )
            }) {
                "allocation_expiring"
            } else if pool_nodes
                .iter()
                .any(|node| node.state == NodeState::Draining)
            {
                "node_draining"
            } else if pool_nodes.iter().any(|node| {
                node.state == NodeState::Active
                    && Self::node_accepts_command(
                        node,
                        spec.resources.admission_timeout_seconds(),
                        now(),
                    )
                    && node_matches_locality(node, spec)
                    && !self.node_supports(node, spec)
            }) {
                "waiting_for_runtime_capability"
            } else if pool_nodes.iter().any(|node| {
                node.state == NodeState::Active
                    && Self::node_accepts_command(
                        node,
                        spec.resources.admission_timeout_seconds(),
                        now(),
                    )
                    && node_matches_locality(node, spec)
                    && self.node_supports(node, spec)
                    && self
                        .select_gpus(&node.node_id, spec, "reason-probe")
                        .is_none()
            }) {
                "waiting_for_gpu"
            } else if pool_nodes.iter().any(|node| {
                node.state == NodeState::Active
                    && Self::node_accepts_command(
                        node,
                        spec.resources.admission_timeout_seconds(),
                        now(),
                    )
                    && node_matches_locality(node, spec)
                    && self.node_supports(node, spec)
                    && (node.resources.cpu_cores - node.used_cpu_cores < spec.resources.cpu_cores
                        || node.resources.memory_gb - node.used_memory_gb
                            < spec.resources.memory_gb)
            }) {
                "waiting_for_cpu_or_memory"
            } else {
                "no_capable_node"
            };
            self.set_queued_reason(&request_id, reason);
        }
    }

    fn active_gpu_lease_count(&self, user_id: Option<&str>) -> usize {
        self.gpu_ledger
            .values()
            .flat_map(|entry| entry.exclusive.iter().chain(entry.shared.iter()))
            .filter(|execution_session_id| {
                self.sessions
                    .get(*execution_session_id)
                    .map(|session| user_id.map(|user| session.user_id == user).unwrap_or(true))
                    .unwrap_or(false)
            })
            .count()
    }

    fn would_exceed_active_gpu_quota(&self, spec: &WorkloadSpec) -> bool {
        let Some(limit) = self.max_active_gpu_sessions_per_user else {
            return false;
        };
        spec.resources.gpus > 0
            && self.active_gpu_lease_count(Some(&spec.user_id)) + spec.resources.gpus > limit
    }

    fn would_exceed_pool_session_cap(&self, spec: &WorkloadSpec) -> bool {
        let Some(limit) = self.max_active_sessions_per_pool else {
            return false;
        };
        self.sessions
            .values()
            .filter(|session| session.pool == spec.pool && !session.state.terminal()
                && !matches!(session.state, SessionState::Suspended | SessionState::Purging))
            .count()
            + 1
            > limit
    }

    fn set_queued_reason(&mut self, request_id: &str, reason: &str) {
        let changed = self
            .requests
            .get(request_id)
            .map(|request| request.reason.as_deref() != Some(reason))
            .unwrap_or(false);
        if !changed {
            return;
        }
        if let Some(request) = self.requests.get_mut(request_id) {
            request.reason = Some(reason.into());
            request.updated_at = json!(now());
        }
        let details = request_details(self.requests.get(request_id).unwrap());
        self.event(
            "request_state",
            request_id,
            Some("QUEUED"),
            Some(reason),
            details,
        );
    }

    fn try_reuse(&mut self, request_id: &str) -> bool {
        let spec = self.requests[request_id].spec.clone();
        let execution_session_id = self
            .sessions
            .values()
            .find(|session| {
                session.reusable_for(&spec)
                    && !self.workspace_blocks_session(session)
                    && self
                        .nodes
                        .get(&session.node_id)
                        .map(|node| {
                            node.state == NodeState::Active
                                && node_matches_locality(node, &spec)
                                && Self::node_accepts_command(
                                    node,
                                    spec.resources.admission_timeout_seconds(),
                                    now(),
                                )
                        })
                        .unwrap_or(false)
            })
            .map(|session| session.execution_session_id.clone());
        let Some(execution_session_id) = execution_session_id else {
            return false;
        };
        let node_id = self.sessions[&execution_session_id].node_id.clone();
        let Some(gpus) = self.select_gpus(&node_id, &spec, &execution_session_id) else {
            return false;
        };
        self.start_request(request_id, &execution_session_id, &node_id, gpus);
        true
    }

    fn try_resume_retained(&mut self, request_id: &str) -> bool {
        let spec = self.requests[request_id].spec.clone();
        let Some(session) = self.sessions.values().find(|s| matches!(s.state, SessionState::Suspending | SessionState::Suspended | SessionState::Stopped | SessionState::Failed | SessionState::Lost)
            && s.matches_reuse_identity(&spec)).cloned() else { return false; };
        if session.state == SessionState::Suspending {
            self.set_queued_reason(request_id,"waiting_for_session"); return true;
        }
        let Some(node) = self.nodes.get(&session.node_id) else {
            self.set_queued_reason(request_id,"retained_node_unavailable"); return true;
        };
        if node.state != NodeState::Active || !Self::node_accepts_command(node,spec.resources.admission_timeout_seconds(),now()) {
            self.set_queued_reason(request_id,"retained_node_unavailable"); return true;
        }
        let id = session.workspace.as_ref().and_then(|w|w["id"].as_str()).unwrap_or("").to_owned();
        let workspace = self.workspaces.get(&id).cloned();
        if workspace.as_ref().map(|w|matches!(w["state"].as_str(),Some("deleted"|"needs_check"|"needs_recovery"))||(w["state"]=="absent"&&!session.empty_retained)
            ||(matches!(session.state,SessionState::Failed|SessionState::Lost)&&w["state"]!="retained")).unwrap_or(true) {
            let request = self.requests.get_mut(request_id).unwrap();
            request.state = RequestState::Failed; request.reason = Some("retained_state_unavailable".into());
            request.failure_kind = Some("infrastructure".into()); request.ended_at = Some(json!(now()));
            let details = request_details(request);
            self.event("request_state",request_id,Some("FAILED"),Some("retained_state_unavailable"),details);
            return true;
        }
        let mut workspace = workspace.unwrap();
        if (workspace["state"] != "retained"&&!session.empty_retained) || workspace["operation"]["state"] == "pending" {
            self.set_queued_reason(request_id,"waiting_for_session"); return true;
        }
        let valid=node.status["authoritative"]==true&&node.status["sessions"].as_array().map(|items|items.iter().all(|item|item["execution_session_id"].is_string())).unwrap_or(false);
        let present=node.status["sessions"].as_array().map(|items|items.iter().any(|item|item["execution_session_id"]==session.execution_session_id)).unwrap_or(false);
        let retained=&node.status["suspended_sessions"];
        let receipt=retained[&session.execution_session_id].as_u64();
        if !session.empty_retained&&valid&&retained.is_object()&&!present&&(receipt.is_none()||receipt.unwrap()>session.suspend_version){
            let request=self.requests.get_mut(request_id).unwrap();request.state=RequestState::Failed;
            request.reason=Some("retained_state_unavailable".into());request.failure_kind=Some("infrastructure".into());
            request.stderr="agent cannot confirm retained execution metadata; explicit rebuild required".into();request.ended_at=Some(json!(now()));
            let details=request_details(request);self.event("request_state",request_id,Some("FAILED"),Some("retained_state_unavailable"),details);return true;
        }
        if !session.empty_retained&&(!valid||receipt!=Some(session.suspend_version)||present){
            let command=json!({"action":"stop_session","execution_session_id":session.execution_session_id,"suspend_version":session.suspend_version});
            let queue=&mut self.queues.entry(session.node_id.clone()).or_default().controls;
            if !queue.contains(&command){queue.push_back(command);}
            self.set_queued_reason(request_id,"waiting_for_session");return true;
        }
        if self.would_exceed_active_gpu_quota(&spec) {
            self.set_queued_reason(request_id, "quota_exceeded"); return true;
        }
        if self.would_exceed_pool_session_cap(&spec) {
            self.set_queued_reason(request_id, "pool_session_limit"); return true;
        }
        if node.resources.cpu_cores-node.used_cpu_cores < session.cpu_cores || node.resources.memory_gb-node.used_memory_gb < session.memory_gb {
            let can_reclaim = node.resources.cpu_cores >= session.cpu_cores
                && node.resources.memory_gb >= session.memory_gb
                && self.select_gpus(&session.node_id, &spec, &session.execution_session_id).is_some();
            let cached = self.sessions.values().filter(|candidate| {
                can_reclaim &&
                candidate.node_id == session.node_id
                    && matches!(candidate.state, SessionState::CachedIdle | SessionState::WarmIdle)
                    && !candidate.workspace.as_ref().and_then(|w| w["id"].as_str())
                        .and_then(|id| self.workspaces.get(id))
                        .map(|w| w["operation"]["state"] == "pending").unwrap_or(false)
            }).min_by(|a, b| a.last_used_at.total_cmp(&b.last_used_at))
                .map(|candidate| candidate.execution_session_id.clone());
            if let Some(cached) = cached {
                self.suspend_session(&cached, "capacity_reclaimed");
            }
            self.set_queued_reason(request_id,"waiting_for_cpu_or_memory"); return true;
        }
        let Some(gpus) = self.select_gpus(&session.node_id,&spec,&session.execution_session_id) else {
            self.set_queued_reason(request_id,"waiting_for_gpu"); return true;
        };
        workspace["state"] = json!("pending"); workspace["execution_session_id"] = json!(session.execution_session_id);
        self.workspaces.insert(id.clone(),workspace.clone());
        self.event("workspace_state",&id,Some("pending"),Some("execution_resume"),workspace.clone());
        workspace["create"] = json!(session.empty_retained&&session.workspace.as_ref().map(|w|w["create"]==true).unwrap_or(false));
        self.sessions.get_mut(&session.execution_session_id).unwrap().workspace = Some(workspace);
        self.sessions.get_mut(&session.execution_session_id).unwrap().resume_required = !session.empty_retained;
        self.sessions.get_mut(&session.execution_session_id).unwrap().empty_retained = false;
        self.sessions.get_mut(&session.execution_session_id).unwrap().activated_at = now();
        let node = self.nodes.get_mut(&session.node_id).unwrap();
        node.used_cpu_cores += session.cpu_cores; node.used_memory_gb += session.memory_gb;
        node.active_sessions.push(session.execution_session_id.clone());
        self.start_request(request_id,&session.execution_session_id,&session.node_id,gpus);
        true
    }

    fn workspace_node_ready(&self, node: &NodeRecord) -> bool {
        self.workspace_configs.get(&node.pool, Some(&node.node_id)).map(|desired|
            node.workspace_status["state"] == "applied" && node.workspace_status["applied_version"] == desired.version
                && node.workspace_status["config"] == desired.to_value()).unwrap_or(false)
    }

    pub fn workspace_barrier(&self) -> ControlResult<Option<DurableReceipt>> {
        self.journal.as_ref().map(|journal| journal.barrier()).transpose()
            .map_err(|error| ControllerError::internal(error.to_string()))
    }

    fn try_new_session(&mut self, request_id: &str) -> bool {
        let spec = self.requests[request_id].spec.clone();
        let workspace = spec.workspace.id.as_ref().and_then(|id| self.workspaces.get(id)).cloned();
        if workspace.as_ref().map(|record| record["state"] != "retained" || record["operation"]["state"] == "pending").unwrap_or(false) { return false; }
        if workspace.as_ref().map(|record|record["maintenance"].is_object()&&!self.nodes.get(record["node_id"].as_str().unwrap_or("")).map(|n|n.workspace_status["maintenance_protocol"]==1).unwrap_or(false)).unwrap_or(false){return false;}
        let reclaim_nodes: Vec<String> = self
            .nodes
            .values()
            .filter(|node| {
                node.pool == spec.pool
                    && (workspace.is_some() || self.workspace_node_ready(node))
                    && workspace.as_ref().map(|record| record["node_id"] == node.node_id).unwrap_or(true)
                    && node.state == NodeState::Active
                    && node_matches_locality(node, &spec)
                    && Self::node_accepts_command(
                        node,
                        spec.resources.admission_timeout_seconds(),
                        now(),
                    )
                    && self.node_supports(node, &spec)
                    && self
                        .select_gpus(&node.node_id, &spec, "pending-session")
                        .is_some()
            })
            .map(|node| node.node_id.clone())
            .collect();
        for node_id in reclaim_nodes {
            loop {
                let enough = self
                    .nodes
                    .get(&node_id)
                    .map(|node| {
                        node.resources.cpu_cores - node.used_cpu_cores >= spec.resources.cpu_cores
                            && node.resources.memory_gb - node.used_memory_gb
                                >= spec.resources.memory_gb
                    })
                    .unwrap_or(false);
                if enough {
                    break;
                }
                let cached = self.nodes.get(&node_id).and_then(|node| {
                    node.active_sessions
                        .iter()
                        .find(|execution_session_id| {
                            self.sessions
                                .get(*execution_session_id)
                                .map(|session| {
                                    matches!(
                                        session.state,
                                        SessionState::CachedIdle | SessionState::WarmIdle
                                    )
                                    && !session.workspace.as_ref().and_then(|w|w["id"].as_str()).and_then(|id|self.workspaces.get(id)).map(|w|w["operation"]["state"]=="pending").unwrap_or(false)
                                })
                                .unwrap_or(false)
                        })
                        .cloned()
                });
                let Some(cached) = cached else {
                    break;
                };
                self.suspend_session(&cached, "capacity_reclaimed");
            }
        }
        let mut candidates: Vec<String> = self
            .nodes
            .values()
            .filter(|node| {
                node.pool == spec.pool
                    && (workspace.is_some() || self.workspace_node_ready(node))
                    && workspace.as_ref().map(|record| record["node_id"] == node.node_id).unwrap_or(true)
                    && node.state == NodeState::Active
                    && node_matches_locality(node, &spec)
                    && Self::node_accepts_command(
                        node,
                        spec.resources.admission_timeout_seconds(),
                        now(),
                    )
                    && node.resources.cpu_cores - node.used_cpu_cores >= spec.resources.cpu_cores
                    && node.resources.memory_gb - node.used_memory_gb >= spec.resources.memory_gb
                    && self.node_supports(node, &spec)
            })
            .map(|node| node.node_id.clone())
            .collect();
        candidates.sort();
        for node_id in candidates {
            if workspace.as_ref().map(|record| record["node_id"] != node_id).unwrap_or(false) { continue; }
            let prospective = format!("exec-{}", uuid::Uuid::new_v4().simple());
            let Some(gpus) = self.select_gpus(&node_id, &spec, &prospective) else {
                continue;
            };
            self.session_seq += 1;
            let timestamp = now();
            let config = self
                .pool_configs
                .get(&spec.pool)
                .cloned()
                .unwrap_or_default();
            let runtime_adapter = self
                .resolve_runtime(self.nodes.get(&node_id).unwrap(), &spec)
                .unwrap_or_else(|| "local-process".into());
            let enforcement_mode =
                capability_enforcement(self.nodes.get(&node_id).unwrap(), &runtime_adapter, &spec);
            let workspace = match self.reserve_workspace(&spec, &node_id, &prospective) { Ok(value) => value, Err(_) => return false };
            let session = SessionRecord { suspend_version: 0, suspend_requested: false, resume_required: false,
                empty_retained: false,
                activated_at: timestamp,
                state_layer_id: Some(prospective.clone()),
                workspace: Some(workspace),
                execution_session_id: prospective.clone(),
                pool: spec.pool.clone(),
                user_id: spec.user_id.clone(),
                session_id: spec.session_id.clone(),
                node_id: node_id.clone(),
                runtime_adapter,
                image_identity: spec.image_identity(),
                isolation_policy: spec.resources.isolation_policy.clone(),
                enforcement_mode,
                mounts: spec.mounts.clone(),
                privileged: spec.privileged,
                gpu_count: spec.resources.gpus,
                gpu_uuids: Vec::new(),
                cpu_cores: spec.resources.cpu_cores,
                memory_gb: spec.resources.memory_gb,
                state: SessionState::Creating,
                active_request_id: None,
                created_at: timestamp,
                last_used_at: timestamp,
                max_lifetime_seconds: spec.resources.max_duration_seconds,
                gpu_mode: config.pool_mode.clone(),
                gpu_slots: config.gpu_slots,
            };
            self.sessions.insert(prospective.clone(), session);
            let details = session_details(self.sessions.get(&prospective).unwrap());
            self.event(
                "session_state",
                &prospective,
                Some("CREATING"),
                None,
                details,
            );
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.used_cpu_cores += spec.resources.cpu_cores;
                node.used_memory_gb += spec.resources.memory_gb;
                node.active_sessions.push(prospective.clone());
            }
            self.start_request(request_id, &prospective, &node_id, gpus);
            return true;
        }
        false
    }

    fn start_request(
        &mut self,
        request_id: &str,
        execution_session_id: &str,
        node_id: &str,
        gpus: Vec<String>,
    ) {
        self.grant_gpu(execution_session_id, &gpus);
        let timestamp = now();
        if let Some(request) = self.requests.get_mut(request_id) {
            request.state = RequestState::Leased;
            if request.leased_at.is_none() {
                request.leased_at = Some(json!(timestamp));
            }
            request.reason = None;
            request.execution_session_id = Some(execution_session_id.into());
            request.node_id = Some(node_id.into());
            request.gpu_uuids = gpus.clone();
            request.gpu_mode = Some(
                self.pool_configs
                    .get(&request.spec.pool)
                    .cloned()
                    .unwrap_or_default()
                    .pool_mode,
            );
            request.updated_at = json!(timestamp);
        }
        let details = request_details(self.requests.get(request_id).unwrap());
        self.event("request_state", request_id, Some("LEASED"), None, details);
        if let Some(request) = self.requests.get_mut(request_id) {
            request.state = RequestState::PreparingImage;
            request.updated_at = json!(timestamp);
        }
        let details = request_details(self.requests.get(request_id).unwrap());
        self.event(
            "request_state",
            request_id,
            Some("PREPARING_IMAGE"),
            None,
            details,
        );
        if let Some(request) = self.requests.get_mut(request_id) {
            request.state = RequestState::StartingSession;
            request.updated_at = json!(timestamp);
        }
        let details = request_details(self.requests.get(request_id).unwrap());
        self.event(
            "request_state",
            request_id,
            Some("STARTING_SESSION"),
            None,
            details,
        );
        if let Some(session) = self.sessions.get_mut(execution_session_id) {
            session.state = SessionState::RunningCommand;
            session.active_request_id = Some(request_id.into());
            session.gpu_uuids = gpus.clone();
            session.last_used_at = timestamp;
        }
        if let Some(request) = self.requests.get_mut(request_id) {
            request.state = RequestState::Running;
            request.started_at = Some(json!(timestamp));
            request.updated_at = json!(timestamp);
        }
        self.command_seq += 1;
        let command_id = format!("cmd-{:06}", self.command_seq);
        let request = self.requests.get(request_id).unwrap();
        let session = self.sessions.get(execution_session_id).unwrap();
        let session_spec = json!({"empty_retained":session.empty_retained,"activated_at":session.activated_at,"resume_required":session.resume_required,"suspend_requested":session.suspend_requested,"resume_retained":session.resume_required,"state_layer_id":session.state_layer_id,"suspend_version":session.suspend_version,"execution_session_id":execution_session_id,"image":session.image_identity,"mounts":session.mounts,"cwd":request.spec.cwd,"env":request.spec.env,"gpu_uuids":gpus,"privileged":session.privileged,"isolation_policy":session.isolation_policy,"runtime_adapter":session.runtime_adapter,"workspace":session.workspace});
        let command_spec =
            json!({"command":request.spec.command,"cwd":request.spec.cwd,"env":request.spec.env});
        let assignment = CommandAssignment {
            command_id: command_id.clone(),
            request_id: request_id.into(),
            execution_session_id: execution_session_id.into(),
            node_id: node_id.into(),
            session_spec,
            command_spec,
            state: CommandState::Pending,
            created_at: timestamp,
            updated_at: timestamp,
            log_path: request.spec.log_path.clone(),
            result: json!({}),
        };
        self.commands.insert(command_id.clone(), assignment);
        self.request_commands
            .insert(request_id.into(), command_id.clone());
        self.queues
            .entry(node_id.into())
            .or_default()
            .commands
            .push_back(command_id.clone());
        let details = self.commands.get(&command_id).unwrap().wire();
        self.event("command_state", &command_id, Some("PENDING"), None, details);
        let details = session_details(self.sessions.get(execution_session_id).unwrap());
        self.event(
            "session_state",
            execution_session_id,
            Some("RUNNING_COMMAND"),
            None,
            details,
        );
        let details = request_details(self.requests.get(request_id).unwrap());
        self.event("request_state", request_id, Some("RUNNING"), None, details);
    }

    fn contention_policy_for_gpu(&self, gpu_uuid: &str) -> (&str, usize) {
        self.gpu_share_overrides
            .get(gpu_uuid)
            .map(|(policy, ratio)| (policy.as_str(), *ratio))
            .unwrap_or((self.gpu_contention_policy.as_str(), self.gpu_shared_ratio))
    }

    fn request_can_target_gpu(&self, request: &RequestRecord, gpu_uuid: &str) -> bool {
        if request.state != RequestState::Queued || request.spec.resources.gpus == 0 {
            return false;
        }
        let spec = &request.spec;
        self.nodes.values().any(|node| {
            node.pool == spec.pool
                && node.resources.gpu_uuids.iter().any(|uuid| uuid == gpu_uuid)
                && node.resources.gpu_uuids.len() >= spec.resources.gpus
                && node.state == NodeState::Active
                && node_matches_locality(node, spec)
                && Self::node_accepts_command(
                    node,
                    spec.resources.admission_timeout_seconds(),
                    now(),
                )
                && node.resources.cpu_cores - node.used_cpu_cores >= spec.resources.cpu_cores
                && node.resources.memory_gb - node.used_memory_gb >= spec.resources.memory_gb
                && self.node_supports(node, spec)
        })
    }

    fn has_queued_gpu_mode_demand(&self, gpu_uuid: &str, pool_mode: &str) -> bool {
        self.requests.values().any(|request| {
            self.pool_configs
                .get(&request.spec.pool)
                .map(|config| config.pool_mode == pool_mode)
                .unwrap_or(false)
                && self.request_can_target_gpu(request, gpu_uuid)
        })
    }

    fn gpu_admission_allowed(&self, node: &NodeRecord, gpu_uuid: &str, execution_session_id: &str) -> bool {
        if self.node_purge_quarantine.contains(gpu_uuid) {return false;}
        let entry = self.gpu_ledger.get(gpu_uuid).cloned().unwrap_or_default();
        if entry.exclusive.as_deref() == Some(execution_session_id) || entry.shared.contains(execution_session_id) {
            return true;
        }
        if node.pool_config.pool_mode == "exclusive" {
            if entry.state() != GpuLeaseState::Free {
                return false;
            }
            if !self.has_queued_gpu_mode_demand(gpu_uuid, "shared") {
                return true;
            }
            let (policy, ratio) = self.contention_policy_for_gpu(gpu_uuid);
            return policy == "exclusive-first"
                || self
                    .gpu_shared_admissions
                    .get(gpu_uuid)
                    .copied()
                    .unwrap_or(0)
                    >= ratio;
        }
        if entry.state() == GpuLeaseState::Exclusive
            || entry
                .shared_capacity
                .map(|capacity| capacity != node.pool_config.gpu_slots)
                .unwrap_or(false)
            || entry.shared.len() >= node.pool_config.gpu_slots
        {
            return false;
        }
        if !self.has_queued_gpu_mode_demand(gpu_uuid, "exclusive") {
            return true;
        }
        let (policy, ratio) = self.contention_policy_for_gpu(gpu_uuid);
        if policy == "exclusive-first" {
            return false;
        }
        let admissions = self
            .gpu_shared_admissions
            .get(gpu_uuid)
            .copied()
            .unwrap_or(0);
        if policy == "barrier-ratio" {
            admissions < ratio
        } else {
            entry.state() == GpuLeaseState::Shared || admissions < ratio
        }
    }

    fn select_gpus(
        &self,
        node_id: &str,
        spec: &WorkloadSpec,
        execution_session_id: &str,
    ) -> Option<Vec<String>> {
        if spec.resources.gpus == 0 {
            return Some(Vec::new());
        }
        let node = self.nodes.get(node_id)?;
        let mut available = Vec::new();
        for uuid in &node.resources.gpu_uuids {
            if self.gpu_admission_allowed(node, uuid, execution_session_id) {
                available.push(uuid.clone());
            }
        }
        if node.pool_config.pool_mode == "shared" {
            if node.pool_config.placement == "pack" {
                available.sort_by_key(|uuid| {
                    let count = self
                        .gpu_ledger
                        .get(uuid)
                        .map(|entry| entry.shared.len())
                        .unwrap_or(0);
                    (
                        if count > 0 { 0 } else { 1 },
                        usize::MAX - count,
                        uuid.clone(),
                    )
                });
            } else {
                available.sort_by_key(|uuid| {
                    (
                        self.gpu_ledger
                            .get(uuid)
                            .map(|entry| entry.shared.len())
                            .unwrap_or(0),
                        uuid.clone(),
                    )
                });
            }
        }
        if available.len() < spec.resources.gpus {
            None
        } else {
            Some(available.into_iter().take(spec.resources.gpus).collect())
        }
    }

    fn node_accepts_command(
        node: &NodeRecord,
        command_timeout_seconds: f64,
        timestamp: f64,
    ) -> bool {
        node.allocation_deadline
            .map(|deadline| {
                deadline - timestamp - node.cleanup_buffer_sec >= command_timeout_seconds
            })
            .unwrap_or(true)
    }

    fn refresh_deadlines_at(&mut self, timestamp: f64) {
        let draining: Vec<String> = self
            .nodes
            .values()
            .filter(|node| {
                node.state == NodeState::Active
                    && node
                        .allocation_deadline
                        .map(|deadline| timestamp >= deadline - node.drain_before_sec)
                        .unwrap_or(false)
            })
            .map(|node| node.node_id.clone())
            .collect();
        for node_id in draining {
            self.drain_node_internal(&node_id, "allocation_expiring");
        }
        let removing: Vec<String> = self
            .nodes
            .values()
            .filter(|node| {
                !matches!(node.state, NodeState::Removed | NodeState::Lost)
                    && node
                        .allocation_deadline
                        .map(|deadline| timestamp >= deadline)
                        .unwrap_or(false)
            })
            .map(|node| node.node_id.clone())
            .collect();
        for node_id in removing {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.state = NodeState::Removing;
            }
            if let Some(node) = self.nodes.get(&node_id) {
                self.event(
                    "node_state",
                    &node_id,
                    Some("REMOVING"),
                    Some("allocation_deadline"),
                    node_details(node),
                );
            }
        }
    }

    pub fn check_heartbeats(&mut self) {
        self.check_heartbeats_at(now());
    }

    pub fn expire_idle_sessions(&mut self) {
        self.expire_idle_sessions_at(now());
    }

    fn expire_idle_sessions_at(&mut self, timestamp: f64) {
        let expired: Vec<String> = self
            .sessions
            .values()
            .filter(|session| {
                matches!(
                    session.state,
                    SessionState::CachedIdle | SessionState::WarmIdle
                ) && timestamp - session.last_used_at >= self.idle_session_timeout_seconds
                    && !session.workspace.as_ref().and_then(|w|w["id"].as_str()).and_then(|id|self.workspaces.get(id)).map(|w|w["operation"]["state"]=="pending").unwrap_or(false)
            })
            .map(|session| session.execution_session_id.clone())
            .collect();
        for execution_session_id in expired {
            self.suspend_session(&execution_session_id, "idle_timeout");
        }
    }

    fn check_heartbeats_at(&mut self, timestamp: f64) {
        let node_ids: Vec<String> = self.nodes.keys().cloned().collect();
        for node_id in node_ids {
            let Some(node) = self.nodes.get(&node_id) else {
                continue;
            };
            if matches!(node.state, NodeState::Removed | NodeState::Lost) {
                continue;
            }
            let age = timestamp - node.last_heartbeat_at;
            if age >= self.lost_timeout_seconds {
                self.mark_node_lost(&node_id, timestamp);
            } else if age >= self.heartbeat_timeout_seconds && node.state == NodeState::Active {
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.state = NodeState::Unhealthy;
                }
                if let Some(node) = self.nodes.get(&node_id) {
                    self.event(
                        "node_state",
                        &node_id,
                        Some("UNHEALTHY"),
                        Some("heartbeat_timeout"),
                        node_details(node),
                    );
                }
            }
        }
    }

    fn mark_node_lost(&mut self, node_id: &str, timestamp: f64) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.state = NodeState::Lost;
        }
        let execution_session_ids: Vec<String> = self
            .sessions
            .values()
            .filter(|session| session.node_id == node_id && !session.state.terminal())
            .map(|session| session.execution_session_id.clone())
            .collect();
        for execution_session_id in execution_session_ids {
            let request_id = self
                .sessions
                .get(&execution_session_id)
                .and_then(|session| session.active_request_id.clone());
            self.release_gpu(&execution_session_id);
            self.release_session_resources(&execution_session_id);
            if let Some(session) = self.sessions.get_mut(&execution_session_id) {
                session.state = SessionState::Lost;
                session.active_request_id = None;
                session.gpu_uuids.clear();
            }
            if let Some(request_id) = request_id {
                if let Some(request) = self.requests.get_mut(&request_id) {
                    if !request.state.terminal() {
                        request.state = RequestState::Lost;
                        request.reason = Some("node_lost".into());
                        request.failure_kind = Some("infrastructure".into());
                        request.updated_at = json!(timestamp);
                        request.ended_at = Some(json!(timestamp));
                    }
                }
                if let Some(request) = self.requests.get(&request_id) {
                    self.event(
                        "request_state",
                        &request_id,
                        Some("LOST"),
                        Some("node_lost"),
                        request_details(request),
                    );
                }
            }
            if let Some(session) = self.sessions.get(&execution_session_id) {
                self.event(
                    "session_state",
                    &execution_session_id,
                    Some("LOST"),
                    Some("node_lost"),
                    session_details(session),
                );
            }
        }
        if let Some(node) = self.nodes.get(node_id) {
            self.event(
                "node_state",
                node_id,
                Some("LOST"),
                Some("heartbeat_lost"),
                node_details(node),
            );
        }
        self.schedule();
    }

    fn grant_gpu(&mut self, execution_session_id: &str, gpus: &[String]) {
        self.grant_gpu_with_count(execution_session_id, gpus, true);
    }

    fn grant_gpu_with_count(&mut self, execution_session_id: &str, gpus: &[String], count: bool) {
        let pool = self
            .sessions
            .get(execution_session_id)
            .map(|s| s.pool.clone())
            .or_else(|| {
                self.requests
                    .values()
                    .find(|r| r.execution_session_id.as_deref() == Some(execution_session_id))
                    .map(|r| r.spec.pool.clone())
            })
            .unwrap_or_default();
        let config = self.pool_configs.get(&pool).cloned().unwrap_or_default();
        let mut shared_events = Vec::new();
        for uuid in gpus {
            let entry = self.gpu_ledger.entry(uuid.clone()).or_default();
            if config.pool_mode == "shared" {
                entry.shared.insert(execution_session_id.into());
                entry.shared_capacity = Some(config.gpu_slots);
                if count {
                    let admissions = self.gpu_shared_admissions.entry(uuid.clone()).or_insert(0);
                    *admissions += 1;
                    shared_events.push((uuid.clone(), *admissions));
                }
            } else {
                entry.exclusive = Some(execution_session_id.into());
            }
        }
        for (uuid, admissions) in shared_events {
            self.event(
                "gpu_contention_state",
                &uuid,
                None,
                Some("shared_admitted"),
                json!({"gpu_uuid":uuid,"shared_admissions":admissions}),
            );
        }
    }
    fn release_gpu(&mut self, execution_session_id: &str) {
        let mut exclusive_released = Vec::new();
        for (uuid, entry) in &mut self.gpu_ledger {
            if entry.exclusive.as_deref() == Some(execution_session_id) {
                entry.exclusive = None;
                self.gpu_shared_admissions.insert(uuid.clone(), 0);
                exclusive_released.push(uuid.clone());
            }
            entry.shared.remove(execution_session_id);
            if entry.shared.is_empty() {
                entry.shared_capacity = None;
            }
        }
        for uuid in exclusive_released {
            self.event(
                "gpu_contention_state",
                &uuid,
                None,
                Some("exclusive_released"),
                json!({"gpu_uuid":uuid,"shared_admissions":0}),
            );
        }
    }
    fn release_session_resources(&mut self, execution_session_id: &str) {
        let Some(session) = self.sessions.get(execution_session_id).cloned() else {
            return;
        };
        if let Some(node) = self.nodes.get_mut(&session.node_id) {
            node.used_cpu_cores = (node.used_cpu_cores - session.cpu_cores).max(0.0);
            node.used_memory_gb = (node.used_memory_gb - session.memory_gb).max(0.0);
            node.active_sessions.retain(|value| value != execution_session_id);
        }
    }
    fn stop_session(&mut self, execution_session_id: &str, reason: &str) {
        self.release_gpu(execution_session_id);
        self.release_session_resources(execution_session_id);
        let node_id = self.sessions.get(execution_session_id).map(|s| s.node_id.clone());
        if let Some(session) = self.sessions.get_mut(execution_session_id) {
            session.state = SessionState::Stopped;
            session.active_request_id = None;
        }
        if let Some(node_id) = node_id { self.queue_stop(&node_id, execution_session_id); }
        if let Some(session) = self.sessions.get(execution_session_id) {
            self.event(
                "session_state",
                execution_session_id,
                Some("STOPPED"),
                Some(reason),
                session_details(session),
            );
        }
    }

    fn queue_stop(&mut self,node_id:&str,id:&str){
        let version=if let Some(session)=self.sessions.get_mut(id){session.suspend_version+=1;session.suspend_version}else{1};
        self.queues.entry(node_id.into()).or_default().controls.push_back(json!({"action":"stop_session","execution_session_id":id,"suspend_version":version}));
    }

    fn suspend_session(&mut self, id: &str, reason: &str) {
        let Some(session) = self.sessions.get_mut(id) else { return; };
        if !matches!(session.state,SessionState::CachedIdle | SessionState::WarmIdle) { return; }
        session.state = SessionState::Suspending;
        session.suspend_requested = false;
        session.suspend_version += 1;
        let node = session.node_id.clone(); let version = session.suspend_version;
        let details = session_details(session);
        self.event("session_state",id,Some("SUSPENDING"),Some(reason),details);
        self.queues.entry(node).or_default().controls.push_back(json!({"action":"suspend_session","execution_session_id":id,"suspend_version":version}));
    }

    fn suspend_unstarted(&mut self,id:&str){
        self.release_gpu(id);self.release_session_resources(id);
        let session=self.sessions.get_mut(id).unwrap();
        session.state=SessionState::Suspended;session.empty_retained=true;session.suspend_requested=false;session.resume_required=false;
        let workspace=session.workspace.as_ref().unwrap();let workspace_id=workspace["id"].as_str().unwrap().to_owned();
        let state=if workspace["create"]==true{"absent"}else{"retained"};
        self.workspaces.get_mut(&workspace_id).unwrap()["state"]=json!(state);
        self.event("workspace_state",&workspace_id,Some(state),None,self.workspaces[&workspace_id].clone());
        self.event("session_state",id,Some("SUSPENDED"),Some("suspended"),session_details(&self.sessions[id]));
    }
    pub fn logical_session_list(&self, token: Option<&str>, selected: Option<&str>) -> ControlResult<Value> {
        let store = self.authority.identity().map_err(|e|ControllerError::internal(e.to_string()))?
            .ok_or_else(||ControllerError::forbidden("persistent identity is required"))?;
        let actor = store.authenticate(token.unwrap_or(""),now()).map_err(|e|ControllerError::forbidden(e.to_string()))?;
        let mut result=Vec::new();
        for ((owner,id),value) in &self.logical_sessions {
            if actor["user_id"] != *owner || selected.map(|wanted|wanted!=id).unwrap_or(false) {continue;}
            let mut item=value.clone();
            item["executions"]=json!(self.sessions.values().filter(|s|s.user_id==*owner&&s.session_id==*id).map(session_details).collect::<Vec<_>>());
            result.push(item);
        }
        if selected.is_some()&&result.is_empty(){return Err(ControllerError::bad("logical session not found"));}
        Ok(json!({"sessions":result}))
    }

    pub fn execution_action(&mut self,id:&str,action:&str,force:bool,token:Option<&str>)->ControlResult<Value>{
        if !matches!(action,"suspend"|"purge") || action=="purge"&&force {return Err(ControllerError::bad("invalid execution action"));}
        let session=self.sessions.get(id).cloned().ok_or_else(||ControllerError::bad("execution not found"))?;
        let store=self.authority.identity().map_err(|e|ControllerError::internal(e.to_string()))?
            .ok_or_else(||ControllerError::forbidden("persistent identity is required"))?;
        store.require_owner(token.unwrap_or(""),&session.user_id,now()).map_err(|e|ControllerError::forbidden(e.to_string()))?;
        if action=="suspend" {
            if matches!(session.state,SessionState::Suspended|SessionState::Suspending){return Ok(session_details(&session));}
            if session.state==SessionState::RunningCommand {
                self.sessions.get_mut(id).unwrap().suspend_requested=true;
                if force {
                    let history:Vec<_>=self.commands.values().filter(|c|c.execution_session_id==id).collect();
                    let never_dispatched=!history.is_empty()&&history.iter().all(|c|c.state==CommandState::Pending
                        ||(c.state==CommandState::Cancelled&&c.result["was_dispatched"]==false));
                    if let Some(request_id)=&session.active_request_id {
                        let request=self.requests.get_mut(request_id).unwrap();
                        request.state=RequestState::Cancelled;request.reason=Some("forced_suspend".into());request.ended_at=Some(json!(now()));
                        let details=request_details(request);
                        if let Some(command)=self.request_commands.get(request_id).cloned(){self.cancel_command(&command,"forced_suspend");}
                        self.event("request_state",request_id,Some("CANCELLED"),Some("forced_suspend"),details);
                    }
                    let state=self.sessions.get_mut(id).unwrap();state.state=SessionState::CachedIdle;state.active_request_id=None;
                    if never_dispatched {
                        self.suspend_unstarted(id);
                        self.flush_journal().map_err(|e|ControllerError::internal(e.to_string()))?;
                        return Ok(session_details(&self.sessions[id]));
                    }
                } else {
                    self.event("session_state",id,Some("RUNNING_COMMAND"),Some("suspend_requested"),session_details(&self.sessions[id]));
                    self.flush_journal().map_err(|e|ControllerError::internal(e.to_string()))?;
                    return Ok(session_details(&self.sessions[id]));
                }
            }
            if !matches!(self.sessions[id].state,SessionState::CachedIdle|SessionState::WarmIdle){return Err(ControllerError::bad("execution is busy"));}
            self.suspend_session(id,"operator_suspend");
        } else {
            if session.state==SessionState::Purged{return Ok(session_details(&session));}
            if !matches!(session.state,SessionState::Suspended|SessionState::Purging){return Err(ControllerError::bad("purge requires suspended execution"));}
            if self.requests.values().any(|r|!r.state.terminal()&&session.matches_reuse_identity(&r.spec)) {
                return Err(ControllerError::bad("execution is referenced by pending requests"));
            }
            if session.empty_retained {
                self.sessions.get_mut(id).unwrap().state=SessionState::Purged;
                self.event("session_state",id,Some("PURGED"),Some("purged"),session_details(&self.sessions[id]));
                self.flush_journal().map_err(|e|ControllerError::internal(e.to_string()))?;
                return Ok(session_details(&self.sessions[id]));
            }
            self.sessions.get_mut(id).unwrap().state=SessionState::Purging;
            self.event("session_state",id,Some("PURGING"),Some("purge_requested"),session_details(&self.sessions[id]));
            let command=json!({"action":"purge_session","execution_session_id":id,"suspend_version":session.suspend_version});
            let queue=&mut self.queues.entry(session.node_id).or_default().controls;
            if !queue.contains(&command){queue.push_back(command);}
        }
        self.flush_journal().map_err(|e|ControllerError::internal(e.to_string()))?;
        Ok(session_details(&self.sessions[id]))
    }

    fn node_supports(&self, node: &NodeRecord, spec: &WorkloadSpec) -> bool {
        self.resolve_runtime(node, spec).is_some()
    }
    fn resolve_runtime(&self, node: &NodeRecord, spec: &WorkloadSpec) -> Option<String> {
        if let Some(requested) = &spec.resources.runtime_adapter {
            return node
                .runtime_capabilities
                .get(requested)
                .filter(|value| supports_request(value, spec))
                .map(|_| requested.clone());
        }
        for preferred in ["bubblewrap", "apptainer", "podman", "local-process"] {
            if node
                .runtime_capabilities
                .get(preferred)
                .map(|value| supports_request(value, spec))
                .unwrap_or(false)
            {
                return Some(preferred.into());
            }
        }
        None
    }
    fn validate_pool_config(&self, pool: &str, config: &PoolConfig) -> ControlResult<()> {
        if config.gpu_slots == 0 || (config.pool_mode == "exclusive" && config.gpu_slots != 1) {
            return Err(ControllerError::bad("invalid pool configuration"));
        }
        if let Some(existing) = self.pool_configs.get(pool) {
            if existing != config {
                let blockers = self.pool_blockers(pool);
                if blockers
                    .values()
                    .any(|value| !value.as_array().unwrap().is_empty())
                {
                    return Err(ControllerError {
                        status: 400,
                        message: "pool configuration conflict".into(),
                        payload: Some(
                            json!({"pool":pool,"established_config":existing,"requested_config":config,"blockers":blockers}),
                        ),
                    });
                }
            }
        }
        Ok(())
    }
    fn pool_blockers(&self, pool: &str) -> BTreeMap<String, Value> {
        let members = self
            .nodes
            .values()
            .filter(|node| node.pool == pool && node.state != NodeState::Removed)
            .map(|node| Value::String(node.node_id.clone()))
            .collect();
        let requests = self
            .requests
            .values()
            .filter(|request| request.spec.pool == pool && !request.state.terminal())
            .map(|request| Value::String(request.request_id.clone()))
            .collect();
        let sessions = self
            .sessions
            .values()
            .filter(|session| session.pool == pool && !session.state.terminal()
                && !matches!(session.state, SessionState::Suspended | SessionState::Purging))
            .map(|session| Value::String(session.execution_session_id.clone()))
            .collect();
        BTreeMap::from([
            ("member_node_ids".into(), Value::Array(members)),
            ("request_ids".into(), Value::Array(requests)),
            ("execution_session_ids".into(), Value::Array(sessions)),
        ])
    }
    fn status_value(&self, request: &RequestRecord) -> Value {
        let session = request
            .execution_session_id
            .as_ref()
            .and_then(|id| self.sessions.get(id));
        let node = request.node_id.as_ref().and_then(|id| self.nodes.get(id));
        json!({"request_id":request.request_id,"state":request.state,"reason":request.reason,"status_message":status_message(request),"owner":request.spec.user_id,"pool":request.spec.pool,"session_id":request.spec.session_id,"node_id":request.node_id,"allocation_id":node.and_then(|n|n.allocation_id.clone()),"execution_session_id":request.execution_session_id,"command_id":self.request_commands.get(&request.request_id),"failure_kind":request.failure_kind,"keep_session_on_failure":request.spec.keep_session_on_failure,"gpu_uuids":request.gpu_uuids,"gpu_mode":request.gpu_mode,"gpu_cotenant_count":request.gpu_cotenant_count,"gpu_cotenant_counts":request.gpu_cotenant_counts,"runtime_adapter":session.map(|s|s.runtime_adapter.clone()).or_else(||request.spec.resources.runtime_adapter.clone()),"isolation_policy":session.map(|s|s.isolation_policy.clone()).unwrap_or_else(||request.spec.resources.isolation_policy.clone()),"enforcement_mode":session.map(|s|s.enforcement_mode.as_str()).unwrap_or("none"),"image":session.map(|s|s.image_identity.clone()).unwrap_or_else(||request.spec.image_identity()),"profile":request.spec.profile,"created_at":request.created_at,"updated_at":request.updated_at,"leased_at":request.leased_at,"started_at":request.started_at,"ended_at":request.ended_at,"exit_code":request.exit_code,"log_path":request.log_path,"log_handle":format!("/v1/requests/{}/logs",request.request_id),"artifact_paths":request.artifact_paths,"deadline":node.and_then(|n|n.allocation_deadline),"ports":request.spec.resources.ports,"locality":request.spec.resources.locality,"workspace":session.and_then(|s|s.workspace.as_ref()).and_then(|w|w.get("id")).and_then(Value::as_str).and_then(|id|self.workspaces.get(id))})
    }
    fn agent_control(&self, node_id: &str) -> Option<Value> {
        let node = self.nodes.get(node_id)?;
        if node.state != NodeState::Removed {
            return None;
        }
        Some(
            json!({"authoritative":true,"action":"shutdown","reason":"allocation_removed","node_id":node_id,"allocation_id":node.allocation_id}),
        )
    }
    fn fail_unreported_command(&mut self, request_id: &str, reason: &str) {
        let Some(id) = self.request_commands.get(request_id).cloned() else { return; };
        let Some(command) = self.commands.get_mut(&id) else { return; };
        if matches!(command.state, CommandState::Pending | CommandState::Running) {
            command.state = CommandState::Failed;
            command.updated_at = now();
            command.result = json!({"reason":reason,"failure_kind":"infrastructure","exit_code":127});
            let details = command.wire();
            self.event("command_state", &id, Some("FAILED"), Some(reason), details);
        }
    }

    fn reconcile_missing_live_assignments(&mut self, node_id: &str) {
        let commands = &self.commands;
        self.reconcile_missing_observations.retain(|key, _| key.strip_prefix("live:")
            .map(|id| commands.get(id).map(|c| c.state == CommandState::Running).unwrap_or(false)).unwrap_or(true));
        let status = self.nodes.get(node_id).map(|n| n.status.clone()).unwrap_or_default();
        let items = status.get("sessions").and_then(Value::as_array);
        let mut authoritative = status["authoritative"] == true && items.map(|items| items.iter().all(|item|
            item.get("execution_session_id").and_then(Value::as_str).map(|s| !s.is_empty()).unwrap_or(false))).unwrap_or(false);
        let command_ids = status.get("command_ids").and_then(Value::as_array);
        if status.get("command_ids").is_some() {
            authoritative &= command_ids.map(|ids| ids.iter().all(|id|
                id.as_str().map(|s| !s.is_empty()).unwrap_or(false))).unwrap_or(false);
        }
        let reported: BTreeMap<String, Value> = if authoritative {
            items.unwrap().iter().map(|item| (item["execution_session_id"].as_str().unwrap().to_owned(), item.clone())).collect()
        } else { BTreeMap::new() };
        let accepted: BTreeSet<String> = if authoritative {
            command_ids.into_iter().flatten().filter_map(Value::as_str).map(str::to_owned).collect()
        } else { BTreeSet::new() };
        let sessions: Vec<String> = self.sessions.values().filter(|s|
            s.node_id == node_id && s.state == SessionState::RunningCommand).map(|s| s.execution_session_id.clone()).collect();
        for execution_session_id in sessions {
            let command_id = self.sessions[&execution_session_id].active_request_id.as_ref()
                .and_then(|r| self.request_commands.get(r)).cloned();
            let running = command_id.as_ref().and_then(|id| self.commands.get(id))
                .map(|c| c.state == CommandState::Running).unwrap_or(false);
            let observation_key = format!("live:{}", command_id.as_deref().unwrap_or(""));
            let protected = reported.get(&execution_session_id).map(|item| command_ids.is_none()
                || !matches!(item["state"].as_str(), Some("CACHED_IDLE" | "WARM_IDLE" | "IDLE"))).unwrap_or(false);
            if !authoritative || protected || !running || command_id.as_ref().map(|id| accepted.contains(id)).unwrap_or(false) {
                self.reconcile_missing_observations.remove(&observation_key);
                continue;
            }
            let command_id = command_id.unwrap();
            let timestamp = now();
            let observation = self.reconcile_missing_observations.entry(observation_key.clone()).or_insert((timestamp, 0));
            observation.1 += 1;
            if timestamp - self.commands[&command_id].updated_at < self.reconcile_missing_grace_seconds
                || observation.1 < self.reconcile_missing_heartbeats
                || timestamp - observation.0 < self.reconcile_missing_grace_seconds { continue; }
            let request_id = self.commands[&command_id].request_id.clone();
            if self.requests[&request_id].state.terminal() { continue; }
            let message = "dispatched assignment missing from authoritative agent state; outcome unknown; not automatically replayed";
            let command = self.commands.get_mut(&command_id).unwrap();
            command.state = CommandState::Failed;
            command.updated_at = timestamp;
            command.result = json!({"reason":"assignment_missing","failure_kind":"infrastructure","exit_code":127,"stderr":message});
            let details = command.wire();
            self.event("command_state", &command_id, Some("FAILED"), Some("assignment_missing"), details);
            let request = self.requests.get_mut(&request_id).unwrap();
            request.state = RequestState::Failed;
            request.reason = Some("assignment_missing".into());
            request.failure_kind = Some("infrastructure".into());
            request.exit_code = Some(127);
            request.stderr = message.into();
            request.updated_at = json!(timestamp);
            request.ended_at = Some(json!(timestamp));
            let details = request_details(request);
            self.event("request_state", &request_id, Some("FAILED"), Some("assignment_missing"), details);
            self.reconcile_missing_observations.remove(&observation_key);
            self.stop_session(&execution_session_id, "assignment_missing");
        }
    }

    fn reconcile_agent_status(&mut self, node_id: &str) {
        let status = self.nodes.get(node_id).map(|n|n.status.clone()).unwrap_or_default();
        if status["authoritative"] == true {
            let valid_snapshot=status["sessions"].as_array().map(|items|items.iter().all(|item|
                item["execution_session_id"].as_str().map(|id|!id.is_empty()).unwrap_or(false))).unwrap_or(false);
            let reported_active: BTreeSet<String>=status["sessions"].as_array().into_iter().flatten()
                .filter_map(|v|v["execution_session_id"].as_str().map(str::to_owned)).collect();
            let stopped:Vec<_>=self.sessions.values().filter(|s|s.node_id==node_id
                &&matches!(s.state,SessionState::Stopped|SessionState::Failed|SessionState::Lost)).cloned().collect();
            for session in stopped {
                let id=&session.execution_session_id;
                let retained=session.workspace.as_ref().and_then(|w|w["id"].as_str()).and_then(|key|self.workspaces.get(key))
                    .map(|w|w["state"]=="retained").unwrap_or(false);
                if valid_snapshot&&!reported_active.contains(id)&&retained
                    &&status["suspended_sessions"][id].as_u64()==Some(session.suspend_version){
                    if self.nodes[node_id].active_sessions.contains(id){self.release_session_resources(id);}
                    self.release_gpu(id);self.sessions.get_mut(id).unwrap().state=SessionState::Suspended;
                    self.event("session_state",id,Some("SUSPENDED"),Some("retained_after_stop"),session_details(&self.sessions[id]));
                }
            }
            let purging:Vec<_>=self.sessions.values().filter(|s|s.node_id==node_id&&s.state==SessionState::Purging).cloned().collect();
            for session in purging {
                let id=&session.execution_session_id;
                if valid_snapshot&&!reported_active.contains(id)&&status["purged_executions"].as_array().map(|ids|ids.iter().any(|v|v==id)).unwrap_or(false) {
                    self.sessions.get_mut(id).unwrap().state=SessionState::Purged;
                    self.event("session_state",id,Some("PURGED"),Some("purged"),session_details(&self.sessions[id]));
                } else {
                    let command=json!({"action":"purge_session","execution_session_id":id,"suspend_version":session.suspend_version});
                    let queue=&mut self.queues.entry(node_id.into()).or_default().controls;
                    if !queue.contains(&command){queue.push_back(command);}
                }
            }
            let pending: Vec<_> = self.sessions.values().filter(|s|s.node_id==node_id && s.state==SessionState::Suspending).cloned().collect();
            for session in pending {
                let id = &session.execution_session_id;
                let retained = session.workspace.as_ref().and_then(|w|w["id"].as_str()).and_then(|key|self.workspaces.get(key))
                    .map(|w|w["state"]=="retained").unwrap_or(false);
                if valid_snapshot&&!reported_active.contains(id)&&status["suspended_sessions"][id].as_u64()==Some(session.suspend_version) && retained {
                    self.release_gpu(id); self.release_session_resources(id);
                    self.sessions.get_mut(id).unwrap().state=SessionState::Suspended;
                    self.event("session_state",id,Some("SUSPENDED"),Some("suspended"),session_details(&self.sessions[id]));
                } else {
                    let command=json!({"action":"suspend_session","execution_session_id":id,"suspend_version":session.suspend_version});
                    let queue=&mut self.queues.entry(node_id.into()).or_default().controls;
                    if !queue.contains(&command) {queue.push_back(command);}
                }
            }
        }
        self.reconcile_missing_live_assignments(node_id);
        let status = self
            .nodes
            .get(node_id)
            .map(|node| node.status.clone())
            .unwrap_or_else(|| json!({}));
        let authoritative = status
            .get("authoritative")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let reported: BTreeMap<String, Value> = status
            .get("sessions")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        item.get("execution_session_id")
                            .and_then(Value::as_str)
                            .map(|id| (id.to_string(), item.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let reconciling: Vec<String> = self
            .sessions
            .values()
            .filter(|session| {
                session.node_id == node_id && session.state == SessionState::Reconciling
            })
            .map(|session| session.execution_session_id.clone())
            .collect();
        for execution_session_id in reconciling {
            let item = reported.get(&execution_session_id).cloned();
            if item.is_none() && !authoritative {
                continue;
            }
            let timestamp = now();
            if item.is_none() {
                let observation = self
                    .reconcile_missing_observations
                    .entry(execution_session_id.clone())
                    .or_insert((timestamp, 0));
                observation.1 += 1;
                let active_request = self
                    .sessions
                    .get(&execution_session_id)
                    .and_then(|session| session.active_request_id.as_ref())
                    .and_then(|request_id| self.request_commands.get(request_id))
                    .and_then(|command_id| self.commands.get(command_id));
                let dispatch_in_grace = active_request
                    .map(|command| {
                        timestamp - command.updated_at < self.reconcile_missing_grace_seconds
                    })
                    .unwrap_or(false);
                if dispatch_in_grace
                    || observation.1 < self.reconcile_missing_heartbeats
                    || timestamp - observation.0 < self.reconcile_missing_grace_seconds
                {
                    continue;
                }
            } else {
                self.reconcile_missing_observations.remove(&execution_session_id);
            }
            let reported_state = item
                .as_ref()
                .and_then(|value| value.get("state"))
                .and_then(Value::as_str)
                .unwrap_or("missing");
            let active_request_id = item
                .as_ref()
                .and_then(|value| value.get("active_request_id"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    self.sessions
                        .get(&execution_session_id)
                        .and_then(|session| session.active_request_id.clone())
                });
            if matches!(reported_state, "RUNNING" | "RUNNING_COMMAND") {
                if let Some(session) = self.sessions.get_mut(&execution_session_id) {
                    session.state = SessionState::RunningCommand;
                    session.active_request_id = active_request_id.clone();
                }
                if let Some(request_id) = &active_request_id {
                    if let Some(request) = self.requests.get_mut(request_id) {
                        if !request.state.terminal() {
                            request.state = RequestState::Running;
                            request.reason = None;
                            request.updated_at = json!(timestamp);
                            if request.started_at.is_none() {
                                request.started_at = Some(json!(timestamp));
                            }
                        }
                    }
                    if let Some(request) = self.requests.get(request_id) {
                        self.event(
                            "request_state",
                            request_id,
                            Some("RUNNING"),
                            None,
                            request_details(request),
                        );
                    }
                }
                let details = session_details(self.sessions.get(&execution_session_id).unwrap());
                self.event(
                    "session_state",
                    &execution_session_id,
                    Some("RUNNING_COMMAND"),
                    Some("reconciled"),
                    details,
                );
            } else if matches!(reported_state, "CACHED_IDLE" | "WARM_IDLE" | "IDLE") {
                self.release_gpu(&execution_session_id);
                if let Some(session) = self.sessions.get_mut(&execution_session_id) {
                    session.state = SessionState::CachedIdle;
                    session.active_request_id = None;
                    session.gpu_uuids.clear();
                    session.last_used_at = timestamp;
                }
                let details = session_details(self.sessions.get(&execution_session_id).unwrap());
                self.event(
                    "session_state",
                    &execution_session_id,
                    Some("CACHED_IDLE"),
                    Some("reconciled"),
                    details,
                );
                if let Some(request_id) = active_request_id {
                    let should_close = self
                        .requests
                        .get(&request_id)
                        .map(|request| request.state == RequestState::Reconciling)
                        .unwrap_or(false);
                    if should_close {
                        if let Some(request) = self.requests.get_mut(&request_id) {
                            request.state = RequestState::Lost;
                            request.reason = Some("command_status_unknown".into());
                            request.updated_at = json!(timestamp);
                            request.ended_at = Some(json!(timestamp));
                        }
                        let details = request_details(self.requests.get(&request_id).unwrap());
                        self.event(
                            "request_state",
                            &request_id,
                            Some("LOST"),
                            Some("command_status_unknown"),
                            details,
                        );
                        self.fail_unreported_command(&request_id, "command_status_unknown");
                    }
                }
            } else {
                self.reconcile_missing_observations.remove(&execution_session_id);
                self.release_gpu(&execution_session_id);
                self.release_session_resources(&execution_session_id);
                if let Some(session) = self.sessions.get_mut(&execution_session_id) {
                    session.state = SessionState::Failed;
                    session.active_request_id = None;
                    session.gpu_uuids.clear();
                }
                let details = session_details(self.sessions.get(&execution_session_id).unwrap());
                self.event(
                    "session_state",
                    &execution_session_id,
                    Some("FAILED"),
                    Some("reconcile_missing"),
                    details,
                );
                if let Some(request_id) = active_request_id {
                    let should_fail = self
                        .requests
                        .get(&request_id)
                        .map(|request| !request.state.terminal())
                        .unwrap_or(false);
                    if should_fail {
                        if let Some(request) = self.requests.get_mut(&request_id) {
                            request.state = RequestState::Failed;
                            request.reason = Some("reconcile_missing".into());
                            request.failure_kind = Some("infrastructure".into());
                            request.updated_at = json!(timestamp);
                            request.ended_at = Some(json!(timestamp));
                        }
                        let details = request_details(self.requests.get(&request_id).unwrap());
                        self.event(
                            "request_state",
                            &request_id,
                            Some("FAILED"),
                            Some("reconcile_missing"),
                            details,
                        );
                        self.fail_unreported_command(&request_id, "reconcile_missing");
                    }
                }
            }
        }
    }

    fn replay(&mut self) -> Result<()> {
        let events = match &self.journal {
            Some(journal) => journal.events()?,
            None => return Ok(()),
        };
        for event in events {
            let details = event.details;
            match event.event_type.as_str() {
                "logical_session_created" => {
                    self.logical_sessions.insert((details["user_id"].as_str().context("logical owner")?.into(),
                        details["session_id"].as_str().context("logical session")?.into()), details.clone());
                }
                "workspace_state" => { self.workspaces.insert(event.entity_id.clone(), details.clone()); }
                "workspace_config_set" => {
                    let pool = details.get("pool").and_then(Value::as_str).unwrap_or(&event.entity_id);
                    self.workspace_configs.restore(pool, details["new_config"].clone())?;
                }
                "request_submitted" => {
                    if let Ok(record) = serde_json::from_value::<RequestRecord>(details.clone()) {
                        self.logical_sessions.entry((record.spec.user_id.clone(),record.spec.session_id.clone()))
                            .or_insert_with(||json!({"user_id":record.spec.user_id,"session_id":record.spec.session_id,
                                "state":"alive","created_at":record.created_at}));
                        self.request_seq = self.request_seq.max(record.sequence);
                        self.requests.insert(record.request_id.clone(), record);
                    }
                }
                "request_state" | "request_queued" => {
                    if let Some(request) = self.requests.get_mut(&event.entity_id) {
                        apply_request_details(
                            request,
                            &details,
                            event.state.as_deref(),
                            event.reason.as_deref(),
                        );
                    }
                }
                "session_state" => {
                    if let Ok(session) = session_from_value(&details) {
                        if let Some(number) = session
                            .execution_session_id
                            .rsplit('-')
                            .next()
                            .and_then(|value| value.parse().ok())
                        {
                            self.session_seq = self.session_seq.max(number);
                        }
                        self.sessions.insert(session.execution_session_id.clone(), session);
                    }
                }
                "command_state" => {
                    if let Ok(command) =
                        serde_json::from_value::<CommandAssignment>(details.clone())
                    {
                        if let Some(number) = command
                            .command_id
                            .rsplit('-')
                            .next()
                            .and_then(|value| value.parse().ok())
                        {
                            self.command_seq = self.command_seq.max(number);
                        }
                        self.request_commands
                            .insert(command.request_id.clone(), command.command_id.clone());
                        self.commands.insert(command.command_id.clone(), command);
                    }
                }
                "node_registered" | "node_announced" | "node_refreshed" | "node_recovered"
                | "node_state" => {
                    if let Ok(node) = node_from_value(&details) {
                        self.pool_configs
                            .entry(node.pool.clone())
                            .or_insert(node.pool_config.clone());
                        for gpu in &node.resources.gpu_uuids {
                            self.gpu_ledger.entry(gpu.clone()).or_default();
                        }
                        self.nodes.insert(node.node_id.clone(), node);
                    }
                }
                "allocation_state" => {
                    if let Ok(allocation) =
                        serde_json::from_value::<AllocationRecord>(details.clone())
                    {
                        self.allocations
                            .insert(allocation.allocation_id.clone(), allocation);
                    }
                }
                "pool_config_set" | "pool_config_replaced" => {
                    if let Some(config) = details
                        .get("new_config")
                        .and_then(|value| serde_json::from_value(value.clone()).ok())
                    {
                        self.pool_configs.insert(event.entity_id.clone(), config);
                    }
                }
                "gpu_contention_state" => {
                    let gpu_uuid = details
                        .get("gpu_uuid")
                        .and_then(Value::as_str)
                        .unwrap_or(&event.entity_id)
                        .to_string();
                    let admissions = details
                        .get("shared_admissions")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as usize;
                    self.gpu_shared_admissions.insert(gpu_uuid, admissions);
                }
                _ => {}
            }
        }
        let recovery = now();
        for command in self.commands.values_mut() {
            if matches!(command.state, CommandState::Pending | CommandState::Running) {
                command.updated_at = recovery;
            }
        }
        for node in self.nodes.values_mut() {
            if !matches!(node.state, NodeState::Removed | NodeState::Lost) {
                node.state = NodeState::Unhealthy;
                node.last_heartbeat_at = recovery;
            }
        }
        for request in self.requests.values_mut() {
            if !request.state.terminal() && request.state != RequestState::Queued {
                request.state = RequestState::Reconciling;
                request.reason = Some("controller_restart".into());
            }
        }
        for session in self.sessions.values_mut() {
            if !session.state.terminal()
                && !matches!(
                    session.state,
                    SessionState::CachedIdle | SessionState::WarmIdle | SessionState::Suspending | SessionState::Suspended | SessionState::Purging
                )
            {
                session.state = SessionState::Reconciling;
            }
        }
        self.rebuild();
        Ok(())
    }
    fn rebuild(&mut self) {
        for node in self.nodes.values_mut() {
            node.used_cpu_cores = 0.0;
            node.used_memory_gb = 0.0;
            node.active_sessions.clear();
            self.queues.entry(node.node_id.clone()).or_default();
        }
        let ids: Vec<String> = self.sessions.keys().cloned().collect();
        for id in ids {
            let session = self.sessions.get(&id).unwrap().clone();
            if !session.state.terminal() && !matches!(session.state,SessionState::Suspended|SessionState::Purging) {
                if let Some(node) = self.nodes.get_mut(&session.node_id) {
                    node.used_cpu_cores += session.cpu_cores;
                    node.used_memory_gb += session.memory_gb;
                    node.active_sessions.push(id.clone());
                }
                self.grant_gpu_with_count(&id, &session.gpu_uuids, false);
            }
        }
    }

    pub fn assert_invariants(&self) -> Result<()> {
        for session in self.sessions.values() {
            // Terminal and nonresident retained records preserve historical GPU
            // assignments but own no lease, including after journal reconstruction.
            if session.state.terminal() || matches!(session.state,SessionState::Suspended|SessionState::Purging) {
                if self.gpu_ledger.values().any(|entry| {
                    entry.exclusive.as_deref() == Some(&session.execution_session_id)
                        || entry.shared.contains(&session.execution_session_id)
                }) {
                    bail!("inactive session holds GPU lease");
                }
                continue;
            }
            if matches!(
                session.state,
                SessionState::CachedIdle | SessionState::WarmIdle
            ) {
                if !session.gpu_uuids.is_empty() {
                    bail!("cached session holds GPU UUIDs");
                }
                if session.active_request_id.is_some() {
                    bail!("idle session has active request");
                }
            }
            for uuid in &session.gpu_uuids {
                let entry = self
                    .gpu_ledger
                    .get(uuid)
                    .ok_or_else(|| anyhow!("missing GPU ledger entry"))?;
                if !entry.shared.contains(&session.execution_session_id)
                    && entry.exclusive.as_deref() != Some(&session.execution_session_id)
                {
                    bail!("ledger/session disagreement");
                }
            }
        }
        Ok(())
    }
}

fn required_string(body: &Value, key: &str) -> ControlResult<String> {
    body.get(key)
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ControllerError::bad(format!("{key} is required")))
}

fn profile_mount(value: &Value) -> Result<MountSpec> {
    if let Some(text) = value.as_str() {
        let parts: Vec<&str> = text.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            bail!("mount string must be source:target[:ro|rw]");
        }
        if parts.len() == 3 && !matches!(parts[2], "ro" | "rw") {
            bail!("mount mode must be ro or rw");
        }
        return Ok(MountSpec {
            source: parts[0].into(),
            target: parts[1].into(),
            read_only: parts.get(2) == Some(&"ro"),
        });
    }
    serde_json::from_value(value.clone()).context("profile mount must be an object or string")
}

fn validate_profile(name: &str, value: &Value) -> Result<()> {
    let object = value
        .as_object()
        .with_context(|| format!("profile {name} must be an object"))?;
    for key in ["image", "runtime_preference"] {
        if object
            .get(key)
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            bail!("profile {name} {key} must be a string");
        }
    }
    if object
        .get("gpu_required")
        .is_some_and(|value| !value.is_boolean())
    {
        bail!("profile {name} gpu_required must be boolean");
    }
    if let Some(environment) = object.get("env") {
        let environment = environment
            .as_object()
            .with_context(|| format!("profile {name} env must be an object"))?;
        if environment.values().any(|value| !value.is_string()) {
            bail!("profile {name} env values must be strings");
        }
    }
    if let Some(mounts) = object.get("default_mounts") {
        let mounts = mounts
            .as_array()
            .with_context(|| format!("profile {name} default_mounts must be an array"))?;
        for mount in mounts {
            profile_mount(mount).with_context(|| format!("profile {name}"))?;
        }
    }
    Ok(())
}
fn supports_policy(capabilities: &Value, policy: &str) -> bool {
    capabilities
        .get("isolation_policies")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .any(|value| value == policy || (policy == "default" && value == "default"))
        })
        .unwrap_or(policy == "default")
}
fn node_matches_locality(node: &NodeRecord, spec: &WorkloadSpec) -> bool {
    spec.resources
        .locality
        .as_ref()
        .map(|locality| {
            locality == &node.node_id
                || node
                    .allocation_id
                    .as_ref()
                    .map(|value| value == locality)
                    .unwrap_or(false)
        })
        .unwrap_or(true)
}
fn supports_request(capabilities: &Value, spec: &WorkloadSpec) -> bool {
    let policy = &spec.resources.isolation_policy;
    if !supports_policy(capabilities, policy) {
        return false;
    }
    if spec.resources.gpus == 0 {
        return true;
    }
    let gpu_policy = capabilities
        .get("gpu_isolation_policies")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .any(|value| value == policy)
        })
        .unwrap_or(false);
    if !gpu_policy
        || !capabilities
            .get("can_select_gpus")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return false;
    }
    !spec.resources.hard_gpu_enforcement
        || capability_enforcement_mode(capabilities) == EnforcementMode::Hard
}
fn capability_enforcement_mode(capabilities: &Value) -> EnforcementMode {
    capabilities
        .get("gpu_enforcement")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or(EnforcementMode::None)
}
fn capability_enforcement(node: &NodeRecord, adapter: &str, spec: &WorkloadSpec) -> String {
    if spec.resources.gpus == 0 {
        return "none".into();
    }
    match node
        .runtime_capabilities
        .get(adapter)
        .map(capability_enforcement_mode)
    {
        Some(EnforcementMode::Hard) => "hard",
        Some(EnforcementMode::Advisory) => "advisory",
        _ => "none",
    }
    .into()
}
fn request_reason_message(reason: &str) -> Option<&'static str> {
    match reason{
    "pool_empty"=>Some("no active node is ready in the pool yet; waiting for a node to register"),
    "permission_revoked"=>Some("current account or permission groups no longer allow this queued request"),
    "pool_draining"=>Some("pool is draining; this unassigned request will not be dispatched"),
    "workspace_node_unavailable"=>Some("the workspace node is retiring; its retained image has not been migrated or deleted"),
    "authority_unavailable"=>Some("authorization state is unavailable; assignment is paused"),
    "node_probing"=>Some("a node has joined but is still probing runtime capabilities; waiting for it to become active"),
    "no_capable_node"=>Some("nodes are present, but none can satisfy the request"),
    "waiting_for_gpu"=>Some("waiting for enough leased-GPU capacity"),
    "waiting_for_cpu_or_memory"=>Some("waiting for enough CPU or memory capacity"),
    "waiting_for_runtime_capability"=>Some("waiting for a node with the required runtime capability"),
    "waiting_for_image_cache_or_prepare"=>Some("waiting for image preparation or a suitable cached image"),
    "waiting_for_session"=>Some("waiting for the matching reusable session to become idle"),
    "workspace_maintenance_failed"=>Some("workspace maintenance failed; recover or release the workspace before retrying"),
    "allocation_expiring"=>Some("matching allocation capacity is too close to its deadline"),
    "node_draining"=>Some("matching nodes are draining and no longer accept new work"),
    "quota_exceeded"=>Some("waiting for user or pool quota capacity"),
    "pool_session_limit"=>Some("waiting for the pool's active-session limit"),
    "fairness_delay"=>Some("queued behind other work under the fairness policy"),
    "node_reregistered"=>Some("the assigned node restarted and CAPE is reconciling its session"),
    "controller_restart"=>Some("the controller restarted and CAPE is reconciling the request"),
    "reconcile_missing"=>Some("the assigned session was absent after the reconciliation grace period"),
    "command_failed"=>Some("the workload command exited unsuccessfully"),
    "infrastructure_failed"=>Some("the node agent or runtime infrastructure failed"),
    "command_status_unknown"=>Some("the node reported an idle session but the command result is unknown"),
    "assignment_missing"=>Some("the agent no longer reports the dispatched command; execution outcome is unknown and it was not automatically retried"),
    "retained_node_unavailable"=>Some("waiting for the original node that holds this execution's retained state"),
    "retained_state_unavailable"=>Some("retained execution state cannot be confirmed; explicit recovery or rebuild is required"),
    "forced_suspend"=>Some("active work was cancelled by an explicit forced suspension"),
    "cancelled"=>Some("the request was cancelled before normal completion"),
    "node_lost"=>Some("the assigned node stopped heartbeating and exceeded the loss timeout"),
    "max_lifetime"=>Some("the reusable session reached its declared maximum lifetime"),
    "allocation_removed"=>Some("the allocation was removed before the request could complete"),
    "node_purged"=>Some("an administrator retired the lost node without remote cleanup confirmation"),
    "node_removed"=>Some("the assigned node was removed before the request could complete"),
    "allocation_drain"=>Some("the allocation is draining and no longer accepts new work"),
    "operator_drain"=>Some("an operator placed the assigned capacity into drain mode"),
    "stopped"=>Some("the request or its session was stopped"),
    _=>None,
}
}
#[allow(dead_code)]
fn session_reason_message(reason: &str) -> Option<&'static str> {
    match reason {
        "node_purged" => Some("an administrator retired the lost node without remote cleanup confirmation"),
        "command_failed" => Some("the workload command exited unsuccessfully"),
        "infrastructure_failed" => Some("the node agent or runtime infrastructure failed"),
        "node_lost" => Some("the assigned node exceeded the loss timeout"),
        "idle_timeout" => Some("the cached session exceeded its idle timeout"),
        "max_lifetime" => Some("the session reached its declared maximum lifetime"),
        "stopped" => Some("the session was stopped"),
        "evict_for_queued_work" => Some("the cached session was reclaimed for queued work"),
        "allocation_drain" => Some("the allocation entered drain mode"),
        "allocation_removed" => Some("the allocation was removed"),
        "node_removed" => Some("the node was removed"),
        "operator_drain" => Some("an operator placed the node into drain mode"),
        "allocation_expiring" => {
            Some("the declared allocation deadline entered its advisory drain window")
        }
        "reconcile_missing" => {
            Some("the session was absent from authoritative node status after reconciliation")
        }
        "reconciled" => Some("authoritative node status reconciled the session"),
        "node_reregistered" => {
            Some("the node re-registered and the session requires reconciliation")
        }
        "lost_session_cleanup_confirmed" => {
            Some("the recovered agent confirmed cleanup of a previously lost session")
        }
        _ => None,
    }
}
#[allow(dead_code)]
fn node_reason_message(reason: &str) -> Option<&'static str> {
    match reason {
        "node_purged" => Some("an administrator retired the lost node without remote cleanup confirmation"),
        "heartbeat_timeout" => Some("the node missed heartbeats and is temporarily unhealthy"),
        "heartbeat_lost" => Some("the node exceeded the loss timeout"),
        "allocation_expiring" => {
            Some("the declared allocation deadline entered its advisory drain window")
        }
        "allocation_deadline" => Some("the declared allocation deadline was reached"),
        "allocation_drain" => Some("the allocation entered drain mode"),
        "allocation_removed" => Some("the allocation was removed"),
        "node_removed" => Some("the node was removed"),
        "operator_drain" => Some("an operator placed the node into drain mode"),
        "remove_pending_sessions" => Some("node removal is waiting for active sessions"),
        _ => None,
    }
}
fn request_state_message(state: &str) -> &'static str {
    match state {
        "SUBMITTED" => "the request was accepted",
        "QUEUED" => "the request is waiting for schedulable capacity",
        "INFEASIBLE" => "the request cannot currently be admitted",
        "LEASED" => "resources have been leased",
        "PREPARING_IMAGE" => "the runtime is preparing the image",
        "STARTING_SESSION" => "the runtime is starting the session",
        "RUNNING" => "the workload is executing",
        "RECONCILING" => "CAPE is reconciling state with the node",
        "COMPLETED" => "the workload completed successfully",
        "FAILED" => "the request failed",
        "CANCELLED" => "the request was cancelled",
        "EXPIRED" => "the request expired",
        "LOST" => "the assigned node or session was lost",
        _ => "request state",
    }
}
fn status_message(request: &RequestRecord) -> String {
    let state = serde_json::to_value(request.state)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "UNKNOWN".into());
    if let Some(reason) = &request.reason {
        format!(
            "{state} ({reason}): {}",
            request_reason_message(reason).unwrap_or("request state reason")
        )
    } else {
        format!("{state}: {}", request_state_message(&state))
    }
}
impl Controller {
    pub fn managed_node_edit(&mut self, token: Option<&str>, node_id: &str, changes: &Value) -> ControlResult<Value> {
        let mut store = self.authority.identity().map_err(|e|ControllerError::internal(e.to_string()))?
            .ok_or_else(||ControllerError::bad("persistent identity is not configured"))?;
        let secret = token.unwrap_or("");
        let record = store.owned_node(secret,node_id,now()).map_err(|e|ControllerError::forbidden(e.to_string()))?;
        let result = (|| -> Result<()> {
            anyhow::ensure!(matches!(record["desired"].as_str(),Some("active"|"reconfigure")), "retiring nodes cannot be edited");
            let changes = changes.as_object().context("invalid node changes")?;
            anyhow::ensure!(changes.keys().all(|k|["cpu_cores","memory_gb","gpu_uuids"].contains(&k.as_str())), "invalid node changes");
            let old = record.get("effective_resources").unwrap_or(&record["resources"]);
            let mut merged = old.clone();
            for (key,value) in changes { merged[key] = value.clone(); }
            let changed = crate::identity_nodes::resources(&merged)?;
            anyhow::ensure!(changed["cpu_cores"].as_f64() <= old["cpu_cores"].as_f64() && changed["memory_gb"].as_f64() <= old["memory_gb"].as_f64()
                && changed["gpu_uuids"].as_array().unwrap().iter().all(|g|old["gpu_uuids"].as_array().unwrap().contains(g)),
                "node edit supports reductions only; enroll a new contribution to add resources");
            if record["desired"] == "reconfigure" { anyhow::ensure!(record["pending_resources"] == changed, "node reconfiguration is already pending"); }
            if record["desired"] == "active" && &changed == old { return Ok(()); }
            store.mutate(|state| {
                let actor = crate::identity::authenticate(state,secret,now())?;
                anyhow::ensure!(actor["is_admin"] == true || actor["user_id"] == state["nodes"][node_id]["owner_user_id"], "node not accessible");
                let current = &mut state["nodes"][node_id];
                if current["desired"] != "reconfigure" {
                    current["desired"] = json!("reconfigure"); current["pending_resources"] = changed.clone(); current["force"] = json!(false); current["intent_at"] = json!(now());
                }
                crate::identity::audit(state,actor["user_id"].as_str().unwrap(),"node.edit",node_id,now());
                Ok(())
            })?;
            self.managed_reconcile()?;
            Ok(())
        })();
        result.map_err(|e|ControllerError::bad(e.to_string()))?;
        Ok(json!({"node_id":node_id,"state":self.nodes.get(node_id).map(|n|serde_json::to_value(n.state).unwrap()).unwrap_or(json!("enrolled")),
            "desired":store.snapshot().map_err(|e|ControllerError::internal(e.to_string()))?["nodes"][node_id]["desired"]}))
    }

    pub fn managed_pool_edit(&mut self, token: Option<&str>, pool_id: &str, changes: &Value) -> ControlResult<Value> {
        let mut store = self.authority.identity().map_err(|e|ControllerError::internal(e.to_string()))?
            .ok_or_else(||ControllerError::bad("persistent identity is not configured"))?;
        let secret = token.unwrap_or("");
        store.require_admin(secret, now()).map_err(|e|ControllerError::forbidden(e.to_string()))?;
        let result = (|| -> Result<Value> {
            let changes = changes.as_object().context("invalid pool changes")?;
            anyhow::ensure!(changes.keys().all(|k|["name","config","max_nodes"].contains(&k.as_str())), "invalid pool changes");
            let state = store.snapshot()?;
            let current = state["pools"].get(pool_id).context("unknown pool")?;
            anyhow::ensure!(current["draining"] == false, "cannot edit a draining pool");
            let config = crate::identity::pool_configuration(changes.get("config").unwrap_or(&current["config"]))?;
            let active = state["nodes"].as_object().map(|nodes|nodes.values().filter(|n|n["pool"] == pool_id && n["desired"] != "removed").count()).unwrap_or(0);
            if config != current["config"] {
                anyhow::ensure!(active == 0 && self.pool_blockers(pool_id).values().all(|v|v.as_array().unwrap().is_empty()), "pool configuration requires an empty pool");
            }
            let limit: Option<u64> = serde_json::from_value(changes.get("max_nodes").unwrap_or(&current["max_nodes"]).clone())?;
            anyhow::ensure!(limit.map(|n| n>0 && n <= i64::MAX as u64 && n >= active as u64).unwrap_or(true), "invalid node limit or limit below enrolled capacity");
            let name = changes.get("name").unwrap_or(&current["name"]).as_str().context("invalid pool name")?;
            crate::identity::name(name)?;
            anyhow::ensure!(!state["pools"].as_object().unwrap().iter().any(|(id,p)|id != pool_id && p["name"] == name), "pool name already exists");
            store.mutate(|state| {
                let actor = crate::identity::authenticate(state, secret, now())?;
                anyhow::ensure!(actor["is_admin"] == true, "admin credential required");
                state["pools"][pool_id]["name"] = json!(name);
                state["pools"][pool_id]["config"] = config.clone();
                state["pools"][pool_id]["max_nodes"] = json!(limit);
                crate::identity::audit(state, actor["user_id"].as_str().unwrap(), "pool.edit", pool_id, now());
                Ok(())
            })?;
            self.pool_configs.insert(pool_id.to_owned(), serde_json::from_value(config.clone())?);
            Ok(json!({"pool_id":pool_id,"name":name,"config":config,"max_nodes":limit}))
        })();
        result.map_err(|e|ControllerError::bad(e.to_string()))
    }

    fn managed_clean(&self, node_id: &str, after: f64) -> bool {
        let Some(node) = self.nodes.get(node_id) else { return false; };
        if !node.active_sessions.is_empty() { return false; }
        let mut latest = 0_f64;
        for command in self.commands.values().filter(|c| c.node_id == node_id) {
            if matches!(command.state, CommandState::Pending | CommandState::Running) { return false; }
            latest = latest.max(command.updated_at);
        }
        let received = self.managed_seen.get(node_id).copied().unwrap_or(0.);
        received > after.max(latest) && now() - received < self.heartbeat_timeout_seconds
            && node.status["authoritative"] == true && node.status["sessions"] == json!([])
            && node.status.get("running_commands").map(|v| v == &json!([])).unwrap_or(true)
            && node.status.get("command_ids").map(|v| v == &json!([])).unwrap_or(true)
    }

    pub fn managed_node_action(&mut self, token: Option<&str>, node_id: &str, action: &str, force: bool, allow_agent: bool) -> ControlResult<Value> {
        if !matches!(action, "drain" | "exit") || (action == "exit" && force) { return Err(ControllerError::bad("invalid node action")); }
        let mut store = self.authority.identity().map_err(|e| ControllerError::internal(e.to_string()))?
            .ok_or_else(|| ControllerError::bad("persistent identity is not configured"))?;
        let secret = token.unwrap_or("");
        let personal = store.owned_node(secret, node_id, now());
        let agent = personal.is_err() && allow_agent && !force && store.authenticate_agent(secret, None, Some(node_id), None).is_ok();
        if personal.is_err() && !agent { return Err(ControllerError::forbidden("node not accessible")); }
        let state = store.snapshot().map_err(|e| ControllerError::internal(e.to_string()))?;
        if state["nodes"][node_id]["desired"] == "removed" { return Ok(json!({"node_id":node_id,"state":"REMOVED"})); }
        if action == "exit" && (self.nodes.contains_key(node_id) || state["nodes"][node_id]["announced"] == true) && !self.managed_clean(node_id, 0.) {
            return Err(ControllerError::bad("node is busy or cleanup is unconfirmed; use drain"));
        }
        store.mutate(|state| {
            let actor_id;
            if !agent {
                let actor = crate::identity::authenticate(state, secret, now())?;
                anyhow::ensure!(actor["is_admin"] == true || actor["user_id"] == state["nodes"][node_id]["owner_user_id"], "node not accessible");
                actor_id = actor["user_id"].as_str().unwrap().to_owned();
            } else {
                actor_id = format!("node:{node_id}");
            }
            let record = &mut state["nodes"][node_id];
            if record["desired"] != "draining" || (force && record["force"] != true) {
                record["desired"] = json!("draining"); record["force"] = json!(force); record["intent_at"] = json!(now());
                record.as_object_mut().unwrap().remove("pending_resources");
            }
            if action == "exit" { record["intent_at"] = json!(0.); }
            crate::identity::audit(state, &actor_id, &format!("node.{action}{}",if force {".force"}else{""}), node_id, now());
            Ok(())
        }).map_err(|e| ControllerError::bad(e.to_string()))?;
        self.managed_reconcile().map_err(|e| ControllerError::internal(e.to_string()))?;
        Ok(json!({"node_id":node_id,"state":self.nodes.get(node_id).map(|n|serde_json::to_value(n.state).unwrap()).unwrap_or(json!("REMOVED"))}))
    }

    pub fn managed_pool_action(&mut self, token: Option<&str>, pool_id: &str, action: &str, force: bool) -> ControlResult<Value> {
        if !matches!(action, "drain" | "remove") || (action == "remove" && force) { return Err(ControllerError::bad("invalid pool action")); }
        let mut store = self.authority.identity().map_err(|e| ControllerError::internal(e.to_string()))?
            .ok_or_else(|| ControllerError::bad("persistent identity is not configured"))?;
        let secret = token.unwrap_or("");
        store.require_admin(secret, now()).map_err(|e| ControllerError::forbidden(e.to_string()))?;
        let state = store.snapshot().map_err(|e| ControllerError::internal(e.to_string()))?;
        if state["pools"].get(pool_id).is_none() { return Err(ControllerError::bad("unknown pool")); }
        if action == "remove" && state["nodes"].as_object().map(|nodes| nodes.values().any(|n| n["pool"] == pool_id && n["desired"] != "removed")).unwrap_or(false) {
            return Err(ControllerError::bad("pool is not empty; drain it before removal"));
        }
        store.mutate(|state| {
            let actor = crate::identity::authenticate(state, secret, now())?;
            anyhow::ensure!(actor["is_admin"] == true, "admin credential required");
            state["pools"][pool_id]["draining"] = json!(true);
            if let Some(nodes) = state["nodes"].as_object_mut() {
                for node in nodes.values_mut().filter(|n| n["pool"] == pool_id && n["desired"] != "removed") {
                    if node["desired"] != "draining" || (force && node["force"] != true) {
                        node["desired"] = json!("draining"); node["force"] = json!(force); node["intent_at"] = json!(now());
                        node.as_object_mut().unwrap().remove("pending_resources");
                    }
                }
            }
            crate::identity::audit(state, actor["user_id"].as_str().unwrap(), &format!("pool.{action}{}",if force {".force"}else{""}), pool_id, now());
            Ok(())
        }).map_err(|e| ControllerError::bad(e.to_string()))?;
        self.managed_reconcile().map_err(|e| ControllerError::internal(e.to_string()))?;
        if action == "remove" {
            self.flush_journal().map_err(|e| ControllerError::internal(e.to_string()))?;
            store.mutate(|state| { state["pools"].as_object_mut().context("invalid pools")?.remove(pool_id); Ok(()) })
                .map_err(|e| ControllerError::internal(e.to_string()))?;
            return Ok(json!({"pool_id":pool_id,"state":"removed"}));
        }
        let state = store.snapshot().map_err(|e| ControllerError::internal(e.to_string()))?;
        Ok(json!({"pool_id":pool_id,"state":if state["pools"][pool_id]["draining"]==true {"draining"} else {"unavailable"}}))
    }

    pub fn managed_node_purge(&mut self, token: Option<&str>, node_id: &str, reason: &str) -> ControlResult<Value> {
        let mut store=self.authority.identity().map_err(|e|ControllerError::internal(e.to_string()))?
            .ok_or_else(||ControllerError::bad("persistent identity is not configured"))?;
        let token=token.unwrap_or("");
        store.require_admin(token,now()).map_err(|e|ControllerError::forbidden(e.to_string()))?;
        if reason.trim().is_empty() || reason.len()>1024 {return Err(ControllerError::bad("purge requires a nonblank reason of at most 1024 UTF-8 bytes"));}
        let snapshot=store.snapshot().map_err(|e|ControllerError::internal(e.to_string()))?;
        let intent=snapshot["nodes"].get(node_id).ok_or_else(||ControllerError::bad("unknown node"))?;
        if intent.get("purge").is_none() && !self.nodes.get(node_id).map(|n|n.state==NodeState::Lost).unwrap_or(false) {
            return Err(ControllerError::bad("node purge requires LOST state"));
        }
        let effective=intent.get("effective_resources").unwrap_or(&intent["resources"]);
        let mut gpus:BTreeSet<String>=effective["gpu_uuids"].as_array().unwrap().iter().filter_map(|v|v.as_str().map(str::to_string)).collect();
        if let Some(node)=self.nodes.get(node_id) {gpus.extend(node.resources.gpu_uuids.iter().cloned());}
        let result=store.mutate(|state| {
            let actor=crate::identity::authenticate(state,token,now())?;
            anyhow::ensure!(actor["is_admin"]==true,"admin credential required");
            if state["nodes"][node_id].get("purge").is_none() {
                state["nodes"][node_id]["purge"]=json!({"node_id":node_id,"state":"REMOVED","purged":true,
                    "cleanup_confirmed":false,"reason":reason.trim(),"purged_at":now(),"actor_user_id":actor["user_id"],
                    "quarantined_gpu_uuids":gpus});
                state["nodes"][node_id]["desired"]=json!("removed");
                state["nodes"][node_id]["force"]=json!(false);
                state["nodes"][node_id].as_object_mut().unwrap().remove("pending_resources");
                crate::identity::audit(state,actor["user_id"].as_str().unwrap(),"node.purge",node_id,now());
            }
            Ok(state["nodes"][node_id]["purge"].clone())
        }).map_err(|e|ControllerError::internal(e.to_string()))?;
        self.managed_reconcile().map_err(|e|ControllerError::internal(e.to_string()))?;
        self.flush_journal().map_err(|e|ControllerError::internal(e.to_string()))?;
        Ok(result)
    }

    pub fn managed_release_quarantine(&mut self, token:Option<&str>, node_id:&str, reason:&str, confirm_stopped:bool) -> ControlResult<Value> {
        let mut store=self.authority.identity().map_err(|e|ControllerError::internal(e.to_string()))?
            .ok_or_else(||ControllerError::bad("persistent identity is not configured"))?;
        let token=token.unwrap_or("");
        store.require_admin(token,now()).map_err(|e|ControllerError::forbidden(e.to_string()))?;
        if !confirm_stopped {return Err(ControllerError::bad("quarantine release requires explicit confirm_stopped=true"));}
        if reason.trim().is_empty() || reason.len()>1024 {return Err(ControllerError::bad("quarantine release requires a nonblank reason of at most 1024 UTF-8 bytes"));}
        let snapshot=store.snapshot().map_err(|e|ControllerError::internal(e.to_string()))?;
        let intent=snapshot["nodes"].get(node_id).ok_or_else(||ControllerError::bad("unknown node"))?;
        if intent.get("purge").is_none() {return Err(ControllerError::bad("quarantine release requires a purged node"));}
        let scope:BTreeSet<String>=intent["purge"]["quarantined_gpu_uuids"].as_array().unwrap().iter().filter_map(|g|g.as_str().map(str::to_string)).collect();
        let receipt=store.mutate(|state| {
            let actor=crate::identity::authenticate(state,token,now())?;
            anyhow::ensure!(actor["is_admin"]==true,"admin credential required");
            if state["nodes"][node_id].get("quarantine_release").is_none() {
                state["nodes"][node_id]["quarantine_release"]=json!({"actor_user_id":actor["user_id"],"released_at":now(),
                    "reason":reason.trim(),"confirmed_stopped":true,"confirmation_source":"administrator"});
                crate::identity::audit(state,actor["user_id"].as_str().unwrap(),"node.release-quarantine",node_id,now());
            }
            Ok(state["nodes"][node_id]["quarantine_release"].clone())
        }).map_err(|e|ControllerError::internal(e.to_string()))?;
        self.managed_reconcile().map_err(|e|ControllerError::internal(e.to_string()))?;
        self.flush_journal().map_err(|e|ControllerError::internal(e.to_string()))?;
        Ok(json!({"node_id":node_id,"state":"REMOVED","released":true,"cleanup_confirmed":false,"release_receipt":receipt,
            "released_source_gpu_uuids":scope,"still_quarantined_gpu_uuids":scope.intersection(&self.node_purge_quarantine).collect::<Vec<_>>(),
            "unquarantined_gpu_uuids":scope.difference(&self.node_purge_quarantine).collect::<Vec<_>>()}))
    }

    fn project_node_purge(&mut self, node_id:&str, intent:&Value) {
        let mut events=Vec::new();
        for request in self.requests.values_mut().filter(|r|!r.state.terminal()) {
            let bound=request.state==RequestState::Queued && request.spec.workspace.id.as_ref()
                .and_then(|id|self.workspaces.get(id)).map(|w|w["node_id"]==node_id).unwrap_or(false);
            if request.node_id.as_deref()!=Some(node_id) && !bound {continue;}
            request.state=if request.state==RequestState::Queued {RequestState::Failed}else{RequestState::Lost};
            request.reason=Some(if request.state==RequestState::Failed {"workspace_node_unavailable"}else{"node_purged"}.into());request.failure_kind=Some("infrastructure".into());
            request.updated_at=json!(now());request.ended_at=Some(json!(now()));
            events.push(("request_state",request.request_id.clone(),serde_json::to_value(request.state).unwrap(),request_details(request)));
        }
        for command in self.commands.values_mut().filter(|c|c.node_id==node_id && matches!(c.state,CommandState::Pending|CommandState::Running)) {
            command.state=CommandState::Failed;command.updated_at=now();
            command.result=json!({"reason":"node_purged","failure_kind":"infrastructure","exit_code":127});
            events.push(("command_state",command.command_id.clone(),json!("FAILED"),command.wire()));
        }
        let sessions:Vec<_>=self.sessions.values().filter(|s|s.node_id==node_id&&!s.state.terminal()).map(|s|s.execution_session_id.clone()).collect();
        for id in sessions {
            self.release_gpu(&id);self.release_session_resources(&id);
            let session=self.sessions.get_mut(&id).unwrap();session.state=SessionState::Lost;session.active_request_id=None;
            events.push(("session_state",id,json!("LOST"),session_details(session)));
        }
        for (id,workspace) in self.workspaces.iter_mut().filter(|(_,w)|w["node_id"]==node_id&&w["node_retired"]!=true) {
            workspace["node_retired"]=json!(true);workspace["cleanup_confirmed"]=json!(false);
            events.push(("workspace_state",id.clone(),workspace["state"].clone(),workspace.clone()));
        }
        if let Some(node)=self.nodes.get_mut(node_id) {
            if node.state!=NodeState::Removed {node.state=NodeState::Removed;events.push(("node_state",node_id.to_string(),json!("REMOVED"),node_details(node)));}
        }
        if let Some(allocation)=intent["allocation_id"].as_str().and_then(|id|self.allocations.get_mut(id)) {
            if allocation.state!=AllocationState::Removed && allocation.node_ids.iter().all(|id|self.nodes.get(id).map(|n|n.state==NodeState::Removed).unwrap_or(false)) {
                allocation.state=AllocationState::Removed;allocation.reason=Some("node_purged".into());allocation.updated_at=now();
                events.push(("allocation_state",allocation.allocation_id.clone(),json!("REMOVED"),allocation_details(allocation)));
            }
        }
        for (kind,id,state,details) in events {
            let reason=if kind=="request_state" {details["reason"].as_str().unwrap_or("node_purged")}else{"node_purged"}.to_string();
            self.event(kind,&id,state.as_str(),Some(&reason),details);
        }
    }

    fn managed_reconcile(&mut self) -> Result<()> {
        let Some(mut store) = self.authority.identity()? else { return Ok(()); };
        let state = store.snapshot()?;
        self.node_purge_quarantine=state["nodes"].as_object().into_iter().flat_map(|nodes|nodes.values())
            .filter(|n|n.get("quarantine_release").is_none())
            .flat_map(|n|n["purge"]["quarantined_gpu_uuids"].as_array().into_iter().flatten())
            .filter_map(|g|g.as_str().map(str::to_string)).collect();
        for (pool, config) in state["pools"].as_object().context("invalid pools")? {
            self.pool_configs.insert(pool.clone(), serde_json::from_value(config.get("config").cloned().unwrap_or_else(|| json!({})))?);
        }
        let queued: Vec<String> = self.requests.values().filter(|r| r.state == RequestState::Queued
            && state["pools"][&r.spec.pool]["draining"] == true).map(|r| r.request_id.clone()).collect();
        for id in queued { self.managed_fail_queued(&id, "pool_draining", "workload"); }
        let mut retired = Vec::new();
        let mut reconfigured = Vec::new();
        if let Some(nodes) = state["nodes"].as_object() {
            for (node_id, intent) in nodes {
                if intent.get("purge").is_some() {
                    if !self.projected_node_purges.contains(node_id) {
                        self.project_node_purge(node_id,intent);
                        self.projected_node_purges.insert(node_id.clone());
                    }
                    continue;
                }
                if intent["desired"] == "active" { continue; }
                let Some(node) = self.nodes.get(node_id) else {
                    if intent["desired"] == "reconfigure" && intent["announced"] != true { reconfigured.push(node_id.clone()); continue; }
                    if intent["desired"] != "removed" && intent["announced"] != true { retired.push(node_id.clone()); }
                    continue;
                };
                if node.state == NodeState::Removed {
                    if intent["desired"] != "removed" { retired.push(node_id.clone()); }
                    continue;
                }
                if !matches!(node.state, NodeState::Draining | NodeState::Lost) { self.drain_node_internal(node_id, "managed_drain"); }
                let idle: Vec<_> = self.sessions.values().filter(|s| s.node_id == *node_id
                    && matches!(s.state,SessionState::CachedIdle|SessionState::WarmIdle)
                    && !self.commands.values().any(|c|c.execution_session_id==s.execution_session_id&&matches!(c.state,CommandState::Pending|CommandState::Running)))
                    .map(|s|s.execution_session_id.clone()).collect();
                for id in idle { self.stop_session(&id,"managed_drain"); }
                if intent["force"] == true {
                    let requests: Vec<_> = self.requests.values().filter(|r| r.node_id.as_deref() == Some(node_id) && !r.state.terminal()).map(|r|r.request_id.clone()).collect();
                    for id in requests { self.cancel_impl(&id, "cancelled", None, false).map_err(|e|anyhow!(e.to_string()))?; }
                }
                let queued: Vec<_> = self.requests.values().filter(|r| intent["desired"] != "reconfigure" && r.state == RequestState::Queued && r.spec.workspace.id.as_ref()
                    .and_then(|id|self.workspaces.get(id)).map(|w|w["node_id"] == *node_id).unwrap_or(false)).map(|r|r.request_id.clone()).collect();
                for id in queued { self.managed_fail_queued(&id, "workspace_node_unavailable", "infrastructure"); }
                if intent["desired"] == "removed" || self.managed_clean(node_id, intent["intent_at"].as_f64().unwrap_or(f64::INFINITY)) {
                    let node = self.nodes.get_mut(node_id).unwrap();
                    if intent["desired"] == "reconfigure" {
                        node.resources = serde_json::from_value(intent["pending_resources"].clone())?;
                        node.state = NodeState::Active;
                        let details = node_details(node);
                        self.event("node_state",node_id,Some("ACTIVE"),Some("managed_reconfigured"),details);
                        reconfigured.push(node_id.clone());
                        continue;
                    }
                    node.state = NodeState::Removed;
                    let allocation_id = node.allocation_id.clone();
                    let details = node_details(node);
                    self.event("node_state", node_id, Some("REMOVED"), Some("managed_drain_complete"), details);
                    retired.push(node_id.clone());
                    if let Some(id) = allocation_id {
                        if let Some(allocation) = self.allocations.get(&id) {
                            if allocation.node_ids.iter().all(|n|self.nodes.get(n).map(|node|node.state==NodeState::Removed).unwrap_or(false)) {
                                let allocation = self.allocations.get_mut(&id).unwrap();
                                allocation.state = AllocationState::Removed; allocation.reason = Some("allocation_removed".into()); allocation.updated_at = now();
                                let details = allocation_details(allocation);
                                self.event("allocation_state", &id, Some("REMOVED"), Some("allocation_removed"), details);
                            }
                        }
                    }
                }
            }
        }
        if !retired.is_empty() {
            self.flush_journal()?;
            store.mutate(|state| { for id in retired {
                state["nodes"][&id]["desired"] = json!("removed");
                crate::identity::audit(state, "controller", "node.retired", &id, now());
            } Ok(()) })?;
        }
        if !reconfigured.is_empty() {
            self.flush_journal()?;
            store.mutate(|state| { for id in reconfigured {
                let node = &mut state["nodes"][&id];
                let effective = node.as_object_mut().unwrap().remove("pending_resources").context("missing pending resources")?;
                node["effective_resources"] = effective;
                node["desired"] = json!("active");
                crate::identity::audit(state,"controller","node.reconfigured",&id,now());
            } Ok(()) })?;
        }
        let state = store.snapshot()?;
        let complete: Vec<_> = state["pools"].as_object().context("invalid pools")?.iter().filter(|(id,p)| p["draining"] == true
            && !state["nodes"].as_object().map(|nodes|nodes.values().any(|n|n["pool"] == **id && n["desired"] != "removed")).unwrap_or(false))
            .map(|(id,_)|id.clone()).collect();
        if !complete.is_empty() {
            self.flush_journal()?;
            store.mutate(|state| { for id in complete {
                state["pools"][&id]["draining"] = json!(false);
                crate::identity::audit(state, "controller", "pool.drained", &id, now());
            } Ok(()) })?;
        }
        Ok(())
    }

    fn managed_fail_queued(&mut self, id: &str, reason: &str, failure_kind: &str) {
        let request = self.requests.get_mut(id).unwrap();
        request.state = RequestState::Failed; request.reason = Some(reason.into()); request.failure_kind = Some(failure_kind.into());
        request.updated_at = json!(now()); request.ended_at = Some(request.updated_at.clone());
        let details = request_details(request);
        self.event("request_state", id, Some("FAILED"), Some(reason), details);
    }
}

fn request_details(request: &RequestRecord) -> Value {
    serde_json::to_value(request).unwrap_or_else(|_| json!({}))
}
fn session_details(session: &SessionRecord) -> Value {
    json!({"empty_retained":session.empty_retained,"activated_at":session.activated_at,"resume_required":session.resume_required,"suspend_requested":session.suspend_requested,"suspend_version":session.suspend_version,"execution_session_id":session.execution_session_id,"pool":session.pool,"user_id":session.user_id,"session_id":session.session_id,"node_id":session.node_id,"runtime_adapter":session.runtime_adapter,"image_identity":session.image_identity,"mounts":session.mounts,"privileged":session.privileged,"gpu_uuids":session.gpu_uuids,"gpu_count":session.gpu_count,"cpu_cores":session.cpu_cores,"memory_gb":session.memory_gb,"enforcement_mode":session.enforcement_mode,"state":session.state,"created_at":session.created_at,"last_used_at":session.last_used_at,"max_duration_seconds":session.max_lifetime_seconds,"active_request_id":session.active_request_id,"isolation_policy":session.isolation_policy,"state_layer_id":session.state_layer_id,"gpu_mode":session.gpu_mode,"gpu_slots":session.gpu_slots,"gpu_cotenant_count":0,"gpu_cotenant_counts":{},"workspace":session.workspace})
}
fn node_details(node: &NodeRecord) -> Value {
    json!({"node_id":node.node_id,"pool":node.pool,"resources":node.resources,"capabilities":node.capabilities,"runtime_capabilities":node.runtime_capabilities,"pool_mode":node.pool_config.pool_mode,"gpu_slots":node.pool_config.gpu_slots,"placement":node.pool_config.placement,"state":node.state,"agent_epoch":node.agent_epoch,"registered_at":node.registered_at,"last_heartbeat_at":node.last_heartbeat_at,"allocation_id":node.allocation_id,"allocation_deadline":node.allocation_deadline,"drain_before_sec":node.drain_before_sec,"cleanup_buffer_sec":node.cleanup_buffer_sec,"runtime_cache_config":node.runtime_cache_config,"cached_images":node.cached_images,"used_cpu_cores":node.used_cpu_cores,"used_memory_gb":node.used_memory_gb,"active_sessions":node.active_sessions,"status":node.status,"workspace_status":node.workspace_status})
}
fn node_summary(
    node: &NodeRecord,
    requests: &BTreeMap<String, RequestRecord>,
    sessions: &BTreeMap<String, SessionRecord>,
    ledger: &BTreeMap<String, GpuLedgerEntry>,
    quarantined: &BTreeSet<String>,
) -> Value {
    let schedulable = node.state == NodeState::Active;
    let execution_session_ids: BTreeSet<String> = node.active_sessions.iter().cloned().collect();
    let leased = node
        .resources
        .gpu_uuids
        .iter()
        .filter(|uuid| {
            ledger
                .get(*uuid)
                .map(|entry| {
                    entry
                        .exclusive
                        .as_ref()
                        .map(|id| execution_session_ids.contains(id))
                        .unwrap_or(false)
                        || entry.shared.iter().any(|id| execution_session_ids.contains(id))
                })
                .unwrap_or(false)
        })
        .count();
    let available = node
        .resources
        .gpu_uuids
        .iter()
        .filter(|uuid| {
            !quarantined.contains(*uuid) && ledger
                .get(*uuid)
                .map(|entry| entry.state() == GpuLeaseState::Free)
                .unwrap_or(true)
        })
        .count();
    let nonterminal = requests
        .values()
        .filter(|request| {
            request.node_id.as_deref() == Some(node.node_id.as_str()) && !request.state.terminal()
        })
        .count();
    let _ = sessions;
    json!({"node_id":node.node_id,"allocation_id":node.allocation_id,"allocation_deadline":node.allocation_deadline,"state":node.state,"gpus_total":node.resources.gpu_uuids.len(),"gpus_leased":leased,"gpus_available":if schedulable{available}else{0},"cpu_total":node.resources.cpu_cores,"cpu_available":if schedulable{(node.resources.cpu_cores-node.used_cpu_cores).max(0.0)}else{0.0},"memory_total":node.resources.memory_gb,"memory_available":if schedulable{(node.resources.memory_gb-node.used_memory_gb).max(0.0)}else{0.0},"last_heartbeat_at":node.last_heartbeat_at,"active_sessions":node.active_sessions,"active_session_count":node.active_sessions.len(),"nonterminal_requests":nonterminal})
}
fn allocation_details(allocation: &AllocationRecord) -> Value {
    json!({"allocation_id":allocation.allocation_id,"pool":allocation.pool,"state":allocation.state,"node_ids":allocation.node_ids,"deadline":allocation.deadline,"created_at":allocation.created_at,"updated_at":allocation.updated_at,"reason":allocation.reason})
}
fn apply_request_details(
    request: &mut RequestRecord,
    details: &Value,
    state: Option<&str>,
    reason: Option<&str>,
) {
    if let Some(state) =
        state.and_then(|value| serde_json::from_value(Value::String(value.into())).ok())
    {
        request.state = state;
    }
    request.reason = reason.map(str::to_string);
    if let Ok(updated) = serde_json::from_value::<RequestRecord>(details.clone()) {
        request.updated_at = updated.updated_at;
        if request.started_at.is_none() {
            request.started_at = updated.started_at;
        }
        if request.leased_at.is_none() {
            request.leased_at = updated.leased_at;
        }
        if request.ended_at.is_none() {
            request.ended_at = updated.ended_at;
        }
        request.execution_session_id = updated.execution_session_id;
        request.node_id = updated.node_id;
        request.gpu_uuids = updated.gpu_uuids;
        request.gpu_mode = updated.gpu_mode;
        request.gpu_cotenant_count = updated.gpu_cotenant_count;
        request.gpu_cotenant_counts = updated.gpu_cotenant_counts;
        request.exit_code = updated.exit_code;
        request.failure_kind = updated.failure_kind;
        request.stdout = updated.stdout;
        request.stderr = updated.stderr;
        request.log_path = updated.log_path;
        request.artifact_paths = updated.artifact_paths;
    }
}
fn node_from_value(value: &Value) -> Result<NodeRecord> {
    let resources =
        serde_json::from_value(value.get("resources").cloned().unwrap_or_else(|| json!({})))?;
    let runtime_capabilities = value
        .get("runtime_capabilities")
        .and_then(Value::as_object)
        .map(|items| {
            items
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default();
    let pool_config = PoolConfig {
        pool_mode: value
            .get("pool_mode")
            .and_then(Value::as_str)
            .unwrap_or("exclusive")
            .into(),
        gpu_slots: value.get("gpu_slots").and_then(Value::as_u64).unwrap_or(1) as usize,
        placement: value
            .get("placement")
            .and_then(Value::as_str)
            .unwrap_or("pack")
            .into(),
    };
    Ok(NodeRecord {
        node_id: value["node_id"].as_str().context("node_id")?.into(),
        pool: value["pool"].as_str().context("pool")?.into(),
        state: serde_json::from_value(
            value
                .get("state")
                .cloned()
                .unwrap_or_else(|| json!("ACTIVE")),
        )?,
        resources,
        runtime_capabilities,
        capabilities: value
            .get("capabilities")
            .cloned()
            .unwrap_or_else(|| json!({})),
        allocation_id: value
            .get("allocation_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        allocation_deadline: value.get("allocation_deadline").and_then(Value::as_f64),
        drain_before_sec: value
            .get("drain_before_sec")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        cleanup_buffer_sec: value
            .get("cleanup_buffer_sec")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        runtime_cache_config: value
            .get("runtime_cache_config")
            .and_then(Value::as_str)
            .unwrap_or("default")
            .into(),
        cached_images: value
            .get("cached_images")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        pool_config,
        agent_epoch: value
            .get("agent_epoch")
            .and_then(Value::as_str)
            .map(str::to_string),
        registered_at: value
            .get("registered_at")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        last_heartbeat_at: value
            .get("last_heartbeat_at")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        used_cpu_cores: value
            .get("used_cpu_cores")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        used_memory_gb: value
            .get("used_memory_gb")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        active_sessions: value
            .get("active_sessions")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        status: value.get("status").cloned().unwrap_or_else(|| json!({})),
        workspace_status: value.get("workspace_status").cloned().unwrap_or_else(|| json!({"protocol":0,"state":"unavailable","applied_version":null})),
    })
}
fn session_from_value(value: &Value) -> Result<SessionRecord> {
    let state: SessionState = serde_json::from_value(
        value
            .get("state")
            .cloned()
            .unwrap_or_else(|| json!("CREATING")),
    )?;
    let mut gpu_uuids: Vec<String> = value
        .get("gpu_uuids")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if state == SessionState::CachedIdle {
        gpu_uuids.clear();
    }
    Ok(SessionRecord { suspend_version: value["suspend_version"].as_u64().unwrap_or(0), suspend_requested: value["suspend_requested"].as_bool().unwrap_or(false), resume_required: value["resume_required"].as_bool().unwrap_or(false),
        empty_retained: value["empty_retained"].as_bool().unwrap_or(false),
        activated_at: value["activated_at"].as_f64().or_else(||value["created_at"].as_f64()).unwrap_or(0.0),
        state_layer_id: value["state_layer_id"].as_str().map(str::to_owned),
        workspace: value.get("workspace").cloned().filter(|value| !value.is_null()),
        execution_session_id: value["execution_session_id"].as_str().context("execution_session_id")?.into(),
        pool: value["pool"].as_str().context("pool")?.into(),
        user_id: value["user_id"].as_str().context("user_id")?.into(),
        session_id: value
            .get("session_id")
            .and_then(Value::as_str)
            .context("session_id")?
            .into(),
        node_id: value["node_id"].as_str().context("node_id")?.into(),
        runtime_adapter: value
            .get("runtime_adapter")
            .and_then(Value::as_str)
            .unwrap_or("local-process")
            .into(),
        image_identity: value
            .get("image_identity")
            .and_then(Value::as_str)
            .unwrap_or("docker://ubuntu:latest")
            .into(),
        isolation_policy: value
            .get("isolation_policy")
            .and_then(Value::as_str)
            .unwrap_or("default")
            .into(),
        enforcement_mode: value
            .get("enforcement_mode")
            .and_then(Value::as_str)
            .unwrap_or("none")
            .into(),
        mounts: serde_json::from_value(value.get("mounts").cloned().unwrap_or_else(|| json!([])))?,
        privileged: value
            .get("privileged")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        gpu_count: value
            .get("gpu_count")
            .and_then(Value::as_u64)
            .unwrap_or(gpu_uuids.len() as u64) as usize,
        gpu_uuids,
        cpu_cores: value
            .get("cpu_cores")
            .and_then(Value::as_f64)
            .unwrap_or(1.0),
        memory_gb: value
            .get("memory_gb")
            .and_then(Value::as_f64)
            .unwrap_or(1.0),
        state,
        active_request_id: value
            .get("active_request_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        created_at: value
            .get("created_at")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        last_used_at: value
            .get("last_used_at")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        max_lifetime_seconds: value
            .get("max_duration_seconds")
            .or_else(|| value.get("max_lifetime_seconds"))
            .and_then(Value::as_f64)
            .unwrap_or(3600.0),
        gpu_mode: value
            .get("gpu_mode")
            .and_then(Value::as_str)
            .unwrap_or("exclusive")
            .into(),
        gpu_slots: value.get("gpu_slots").and_then(Value::as_u64).unwrap_or(1) as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn managed_report_fixture() -> Result<(tempfile::TempDir, Controller, String, String, String, String)> {
        use crate::identity::{new_secret, verifier, IdentityStore};
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("identity.sqlite3");
        let admin = new_secret()?;
        let mut store = IdentityStore::create(&path,"admin",&admin)?;
        let owner = new_secret()?;
        let user = new_secret()?;
        for (name,secret) in [("alice",&owner),("bob",&user)] {
            store.register_token(&uuid::Uuid::new_v4().to_string(),&verifier(secret)?,"client",Some(name),None,None,now())?;
        }
        let pool = store.create_pool(&admin,"pool",now())?;
        let agent = new_secret()?;
        let mut body=json!({"operation_id":uuid::Uuid::new_v4().to_string(),"verifier":verifier(&agent)?,"node_id":"node-a","pool":pool,
            "allocation_id":"allocation-a","agent_epoch":"epoch-a","resources":{"cpu_cores":2,"memory_gb":2,"gpu_uuids":[]}});
        store.enroll_node(&owner,&body,now())?;
        body["runtime_capabilities"]=json!({"local-process":{"adapter":"local-process","nested_container":true,"can_bind_mounts":true,"image_formats":["default"],"isolation_policies":["default","none"]}});
        let mut controller=Controller::new(Authority::from_identity(path),None)?;
        let desired=controller.workspace_configs.get(&pool,Some("node-a"))?;
        body["workspace_config"]=json!({"protocol":1,"state":"applied","applied_version":desired.version,"config":desired.to_value()});
        controller.announce_or_register(&body,true,Some(&agent))?;
        Ok((temp,controller,owner,user,agent,pool))
    }

    #[test]
    fn managed_purge_retains_gpu_fence_without_projected_journal() -> Result<()> {
        let (_temp,mut c,owner,_user,agent,_pool)=managed_report_fixture()?;
        let admin=crate::identity::new_secret()?;
        let gpu="GPU-00000000-0000-0000-0000-000000000001";
        let path=c.authority.identity_path.clone().unwrap();
        let mut store=c.authority.identity()?.unwrap();
        store.mutate(|state| {
            let token=state["tokens"].as_object_mut().unwrap().values_mut().find(|t|t["kind"]=="admin").unwrap();
            token["verifier"]=json!(crate::identity::verifier(&admin)?);
            state["nodes"]["node-a"]["resources"]["gpu_uuids"]=json!([gpu,"GPU-00000000-0000-0000-0000-000000000002"]);
            state["nodes"]["node-a"]["effective_resources"]["gpu_uuids"]=json!([gpu]);
            Ok(())
        })?;
        c.nodes.get_mut("node-a").unwrap().resources.gpu_uuids=vec![gpu.into()];
        assert_eq!(c.managed_node_purge(Some(&owner),"node-a","expired").unwrap_err().status,403);
        assert_eq!(c.managed_node_purge(Some(&admin),"node-a","expired").unwrap_err().status,400);
        c.nodes.get_mut("node-a").unwrap().state=NodeState::Lost;
        let result=c.managed_node_purge(Some(&admin),"node-a","expired")?;
        assert_eq!(result["cleanup_confirmed"],false);
        assert_eq!(result["quarantined_gpu_uuids"],json!([gpu]));
        assert!(store.authenticate_agent(&agent,None,None,None).is_err());
        assert!(!c.gpu_admission_allowed(&c.nodes["node-a"],gpu,"new-execution"));
        // Identity is durable even when no derived node journal is available.
        let mut restored=Controller::new(Authority::from_identity(path.clone()),None)?;
        assert!(restored.node_purge_quarantine.contains(gpu));
        assert_eq!(restored.managed_node_purge(Some(&admin),"node-a","retry")?,result);
        assert_eq!(restored.managed_release_quarantine(Some(&owner),"node-a","stopped",true).unwrap_err().status,403);
        assert_eq!(restored.managed_release_quarantine(Some(&admin),"node-a","stopped",false).unwrap_err().status,400);
        let released=restored.managed_release_quarantine(Some(&admin),"node-a","stopped",true)?;
        assert_eq!(released["unquarantined_gpu_uuids"],json!([gpu]));
        assert!(restored.node_purge_quarantine.is_empty());
        assert!(store.authenticate_agent(&agent,None,None,None).is_err());
        let mut reopened=Controller::new(Authority::from_identity(path),None)?;
        assert!(reopened.node_purge_quarantine.is_empty());
        assert_eq!(reopened.managed_release_quarantine(Some(&admin),"node-a","retry",true)?,released);
        assert_eq!(reopened.managed_node_purge(Some(&admin),"node-a","retry")?,result);
        assert!(reopened.node_purge_quarantine.is_empty());
        Ok(())
    }

    #[test]
    fn managed_report_piggyback_preserves_verified_epoch() -> Result<()> {
        let (_temp,mut controller,_owner,user,agent,pool)=managed_report_fixture()?;
        let actor=controller.authority.identity()?.unwrap().authenticate(&user,now())?;
        let request=controller.submit(serde_json::from_value(json!({"pool":pool,"user_id":actor["user_id"],"session_id":"t","command":["true"],
            "resources":{"gpus":0,"runtime_adapter":"local-process","isolation_policy":"none"}}))?,Some(&user))?;
        let pulse=controller.heartbeat(&json!({"node_id":"node-a","agent_epoch":"epoch-a","status":{"authoritative":false}}),Some(&agent))?;
        let command=pulse["commands"][0]["command_id"].as_str().context("command was not dispatched")?;
        let result=controller.report_command(&json!({"node_id":"node-a","agent_epoch":"epoch-a","command_id":command,"exit_code":0,"accept_commands":true}),Some(&agent))?;
        assert_eq!(result["state"],"COMPLETED");
        assert_eq!(result["request_id"],request["request_id"]);
        assert!(result["commands"].is_array());
        Ok(())
    }

    #[test]
    fn managed_owner_force_finishes_without_user_token_in_internal_cancel() -> Result<()> {
        let (_temp,mut controller,owner,user,agent,pool)=managed_report_fixture()?;
        let actor=controller.authority.identity()?.unwrap().authenticate(&user,now())?;
        let request=controller.submit(serde_json::from_value(json!({"pool":pool,"user_id":actor["user_id"],"session_id":"t","command":["true"],
            "resources":{"gpus":0,"runtime_adapter":"local-process","isolation_policy":"none"}}))?,Some(&user))?;
        controller.heartbeat(&json!({"node_id":"node-a","agent_epoch":"epoch-a","status":{"authoritative":false}}),Some(&agent))?;
        let id=request["request_id"].as_str().unwrap();
        assert!(controller.request_status(id,Some(&owner)).is_err());
        let result=controller.managed_node_action(Some(&owner),"node-a","drain",true,false)?;
        assert_eq!(result.as_object().unwrap().len(),2);
        assert_eq!(controller.request_status(id,Some(&user))?["state"],"CANCELLED");
        assert!(controller.request_status(id,Some(&owner)).is_err());
        Ok(())
    }

    #[test]
    fn managed_drain_sweeps_late_idle_without_stopping_pending_assignment() -> Result<()> {
        let (_temp,mut controller,owner,user,_agent,pool)=managed_report_fixture()?;
        let actor=controller.authority.identity()?.unwrap().authenticate(&user,now())?;
        let response=controller.submit(serde_json::from_value(json!({"pool":pool,"user_id":actor["user_id"],"session_id":"t","command":["true"],
            "resources":{"gpus":0,"runtime_adapter":"local-process","isolation_policy":"none"}}))?,Some(&user))?;
        let id=response["request_id"].as_str().unwrap();
        let execution_session_id=controller.requests[id].execution_session_id.clone().context("session not assigned")?;
        let command_id=controller.request_commands[id].clone();
        controller.managed_node_action(Some(&owner),"node-a","drain",false,false)?;
        controller.sessions.get_mut(&execution_session_id).unwrap().state=SessionState::CachedIdle;
        controller.schedule();
        assert_eq!(controller.sessions[&execution_session_id].state,SessionState::CachedIdle);
        controller.commands.get_mut(&command_id).unwrap().state=CommandState::Completed;
        controller.requests.get_mut(id).unwrap().state=RequestState::Completed;
        controller.sessions.get_mut(&execution_session_id).unwrap().active_request_id=None;
        controller.schedule();
        assert_eq!(controller.sessions[&execution_session_id].state,SessionState::Stopped);
        assert!(controller.nodes["node-a"].active_sessions.is_empty());
        Ok(())
    }

    #[test]
    fn managed_drain_requires_command_completion_and_fresh_cleanup() -> Result<()> {
        use crate::identity::{new_secret, verifier, IdentityStore};
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("identity.sqlite3");
        let admin = new_secret()?;
        let mut store = IdentityStore::create(&path, "admin", &admin)?;
        let owner = new_secret()?;
        store.register_token(&uuid::Uuid::new_v4().to_string(), &verifier(&owner)?, "owner", Some("alice"), None, None, now())?;
        let pool = store.create_pool(&admin, "pool", now())?;
        let agent = new_secret()?;
        let body = json!({"operation_id":uuid::Uuid::new_v4().to_string(),"verifier":verifier(&agent)?,"node_id":"node-a",
            "pool":pool,"allocation_id":"allocation-a","agent_epoch":"epoch-a","resources":{"cpu_cores":1,"memory_gb":1,"gpu_uuids":[]}});
        store.enroll_node(&owner, &body, now())?;
        let mut controller = Controller::new(Authority::from_identity(path), None)?;
        controller.announce_or_register(&body, false, Some(&agent))?;
        let pulse = json!({"node_id":"node-a","agent_epoch":"epoch-a","status":{"authoritative":true,"sessions":[]}});
        controller.heartbeat(&pulse, Some(&agent))?;
        controller.commands.insert("command".into(), CommandAssignment { command_id:"command".into(), request_id:"request".into(),
            execution_session_id:"session".into(), node_id:"node-a".into(), session_spec:json!({}),command_spec:json!({}),state:CommandState::Pending,
            created_at:now(),updated_at:now(),log_path:None,result:Value::Null });
        controller.managed_node_action(Some(&owner), "node-a", "drain", false, false)?;
        controller.heartbeat(&pulse, Some(&agent))?;
        controller.schedule();
        assert_eq!(controller.nodes["node-a"].state, NodeState::Draining);
        let command = controller.commands.get_mut("command").unwrap();
        command.state = CommandState::Completed; command.updated_at = now();
        controller.schedule();
        assert_eq!(controller.nodes["node-a"].state, NodeState::Draining);
        controller.heartbeat(&pulse, Some(&agent))?;
        controller.schedule();
        assert_eq!(controller.nodes["node-a"].state, NodeState::Removed);
        assert_eq!(controller.heartbeat(&pulse, Some(&agent))?["agent_control"]["action"], "shutdown");
        Ok(())
    }

    fn spec() -> WorkloadSpec {
        serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"t","command":["false"],"resources":{"gpus":1}})).unwrap()
    }

    fn spec_for(user: &str, task: &str) -> WorkloadSpec {
        serde_json::from_value(json!({"pool":"p","user_id":user,"session_id":task,"command":["false"],"resources":{"gpus":1}})).unwrap()
    }

    #[test]
    fn cached_idle_never_holds_gpu() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        controller
            .gpu_ledger
            .insert("GPU-a".into(), GpuLedgerEntry::default());
        controller.sessions.insert(
            "sess-000001".into(),
            SessionRecord { suspend_version: 0, suspend_requested: false, resume_required: false,
                empty_retained: false,
                activated_at: 0.0,
                state_layer_id: None,
                workspace: None,
                execution_session_id: "sess-000001".into(),
                pool: "p".into(),
                user_id: "u".into(),
                session_id: "t".into(),
                node_id: "n".into(),
                runtime_adapter: "local-process".into(),
                image_identity: "docker://ubuntu:latest".into(),
                isolation_policy: "default".into(),
                enforcement_mode: "none".into(),
                mounts: vec![],
                privileged: false,
                gpu_count: 1,
                gpu_uuids: vec![],
                cpu_cores: 1.0,
                memory_gb: 1.0,
                state: SessionState::CachedIdle,
                active_request_id: None,
                created_at: 0.0,
                last_used_at: 0.0,
                max_lifetime_seconds: 3600.0,
                gpu_mode: "exclusive".into(),
                gpu_slots: 1,
            },
        );
        controller.assert_invariants().unwrap();
    }

    #[test]
    fn retained_historical_gpu_assignment_does_not_require_a_live_lease() {
        let mut controller=Controller::new(Authority::default(),None).unwrap();
        register(&mut controller,"exclusive",1,"allocation");
        let request=controller.submit(spec(),None).unwrap();
        let id=controller.requests[request["request_id"].as_str().unwrap()].execution_session_id.clone().unwrap();
        let historical=controller.sessions[&id].gpu_uuids.clone();
        assert!(!historical.is_empty());
        controller.release_gpu(&id);
        controller.release_session_resources(&id);
        controller.sessions.get_mut(&id).unwrap().active_request_id=None;
        for state in [SessionState::Suspended,SessionState::Purging] {
            controller.sessions.get_mut(&id).unwrap().state=state;
            controller.rebuild();
            controller.assert_invariants().unwrap();
            controller.grant_gpu_with_count(&id,&historical,false);
            assert!(controller.assert_invariants().is_err());
            controller.release_gpu(&id);
        }
    }

    #[test]
    fn workload_spec_defaults_match_python() {
        let spec = spec();
        assert!(spec.reuse);
        assert!(spec.keep_session_on_failure);
        assert_eq!(spec.resources.cpu_cores, 1.0);
        assert_eq!(spec.resources.max_duration_seconds, 3600.0);
        assert_eq!(spec.resources.command_timeout_seconds, None);
        assert_eq!(spec.resources.admission_timeout_seconds(), 3600.0);
    }

    fn register(controller: &mut Controller, pool_mode: &str, gpu_slots: usize, allocation: &str) {
        let body = json!({"node_id":"node-a","pool":"p","pool_mode":pool_mode,"gpu_slots":gpu_slots,"placement":"pack","resources":{"cpu_cores":8,"memory_gb":64,"gpu_uuids":["GPU-a"]},"capabilities":{"adapter":"local-process","isolation_policies":["default","none"]},"runtime_capabilities":{"local-process":{"adapter":"local-process","image_formats":["default"],"isolation_policies":["default","none"],"can_select_gpus":true,"gpu_isolation_policies":["default","none"]}},"allocation_id":allocation,"runtime_cache_config":"default"});
        controller.announce_or_register(&body, true, None).unwrap();
    }

    fn register_pool(
        controller: &mut Controller,
        node_id: &str,
        pool: &str,
        pool_mode: &str,
        gpu_slots: usize,
        gpu_uuids: &[&str],
    ) {
        let body = json!({"node_id":node_id,"pool":pool,"pool_mode":pool_mode,"gpu_slots":gpu_slots,"placement":"pack","resources":{"cpu_cores":16,"memory_gb":128,"gpu_uuids":gpu_uuids},"capabilities":{"adapter":"local-process","isolation_policies":["default","none"]},"runtime_capabilities":{"local-process":{"adapter":"local-process","image_formats":["default"],"isolation_policies":["default","none"],"can_select_gpus":true,"gpu_isolation_policies":["default","none"]}},"allocation_id":format!("allocation-{node_id}"),"runtime_cache_config":"default"});
        controller.announce_or_register(&body, true, None).unwrap();
    }

    fn pool_spec(pool: &str, user: &str, task: &str) -> WorkloadSpec {
        serde_json::from_value(json!({"pool":pool,"user_id":user,"session_id":task,"command":["false"],"resources":{"gpus":1}})).unwrap()
    }

    fn finish_request(controller: &mut Controller, request_id: &str) {
        let command_id = controller.request_commands[request_id].clone();
        let node_id = controller.requests[request_id].node_id.clone().unwrap();
        controller
            .report_command(
                &json!({"node_id":node_id,"command_id":command_id,"exit_code":0,"failure_kind":"workload"}),
                None,
            )
            .unwrap();
    }

    #[test]
    fn running_cancel_is_delivered_before_stop_and_late_report_stays_cancelled() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register(&mut controller, "exclusive", 1, "a1");
        let submitted = controller.submit(spec(), None).unwrap();
        let request_id = submitted["request_id"].as_str().unwrap().to_string();
        let assignment = controller
            .heartbeat(&json!({"node_id":"node-a","max_commands":8}), None)
            .unwrap()["commands"][0]
            .clone();
        let command_id = assignment["command_id"].as_str().unwrap().to_string();
        assert_eq!(
            controller.commands[&command_id].state,
            CommandState::Running
        );

        controller.cancel(&request_id, "cancelled", None).unwrap();
        assert_eq!(
            controller.commands[&command_id].state,
            CommandState::Cancelled
        );
        let controls = &controller.queues["node-a"].controls;
        assert_eq!(controls[0]["action"], "cancel");
        assert_eq!(controls[0]["command_id"], command_id);
        assert_eq!(controls[1]["action"], "stop_session");

        let response = controller
            .report_command(
                &json!({"node_id":"node-a","command_id":command_id,"exit_code":0,"failure_kind":"workload","accept_commands":true,"max_commands":8}),
                None,
            )
            .unwrap();
        assert_eq!(
            controller.requests[&request_id].state,
            RequestState::Cancelled
        );
        assert_eq!(
            controller.commands[&command_id].state,
            CommandState::Cancelled
        );
        assert_eq!(response["commands"][0]["action"], "cancel");
        assert_eq!(response["commands"][1]["action"], "stop_session");
    }

    #[test]
    fn queued_and_pending_cancel_never_emit_a_node_cancel() {
        let mut queued = Controller::new(Authority::default(), None).unwrap();
        let value = queued.submit(spec(), None).unwrap();
        let request_id = value["request_id"].as_str().unwrap();
        queued.cancel(request_id, "cancelled", None).unwrap();
        assert_eq!(queued.requests[request_id].started_at, None);
        assert!(queued.request_commands.get(request_id).is_none());

        let mut pending = Controller::new(Authority::default(), None).unwrap();
        register(&mut pending, "exclusive", 1, "a1");
        let value = pending.submit(spec(), None).unwrap();
        let request_id = value["request_id"].as_str().unwrap().to_string();
        let command_id = pending.request_commands[&request_id].clone();
        assert_eq!(pending.commands[&command_id].state, CommandState::Pending);
        pending.cancel(&request_id, "cancelled", None).unwrap();
        assert!(pending.queues["node-a"]
            .controls
            .iter()
            .all(|control| control["action"] != "cancel"));
        let response = pending
            .heartbeat(&json!({"node_id":"node-a","max_commands":8}), None)
            .unwrap();
        assert!(response["commands"]
            .as_array()
            .unwrap()
            .iter()
            .all(|command| command["action"] != "run" && command["action"] != "cancel"));
    }

    #[test]
    fn contention_policies_are_work_conserving_nonpreemptive_and_distinct() {
        for (policy, expected_shared) in [
            ("barrier-ratio", 1usize),
            ("ratio-only", 2usize),
            ("exclusive-first", 0usize),
        ] {
            let mut controller = Controller::new(Authority::default(), None).unwrap();
            controller.gpu_contention_policy = policy.into();
            controller.gpu_shared_ratio = 1;
            register_pool(
                &mut controller,
                "shared-node",
                "shared",
                "shared",
                2,
                &["GPU-a"],
            );
            register_pool(
                &mut controller,
                "exclusive-node",
                "exclusive",
                "exclusive",
                1,
                &["GPU-a"],
            );
            controller.nodes.get_mut("shared-node").unwrap().state = NodeState::Draining;
            controller.nodes.get_mut("exclusive-node").unwrap().state = NodeState::Draining;
            let shared_a = controller
                .submit(pool_spec("shared", "s1", "a"), None)
                .unwrap();
            let shared_b = controller
                .submit(pool_spec("shared", "s2", "b"), None)
                .unwrap();
            let exclusive = controller
                .submit(pool_spec("exclusive", "e", "e"), None)
                .unwrap();
            controller.nodes.get_mut("shared-node").unwrap().state = NodeState::Active;
            controller.nodes.get_mut("exclusive-node").unwrap().state = NodeState::Active;
            controller.schedule();
            let shared_running = [shared_a, shared_b]
                .iter()
                .filter(|value| {
                    let id = value["request_id"].as_str().unwrap();
                    controller.requests[id].state == RequestState::Running
                })
                .count();
            assert_eq!(shared_running, expected_shared, "policy={policy}");
            let exclusive_id = exclusive["request_id"].as_str().unwrap();
            assert_eq!(
                controller.requests[exclusive_id].state == RequestState::Running,
                policy == "exclusive-first",
                "policy={policy}"
            );
            if expected_shared > 0 {
                let running_id = controller
                    .requests
                    .values()
                    .find(|request| {
                        request.spec.pool == "shared" && request.state == RequestState::Running
                    })
                    .unwrap()
                    .request_id
                    .clone();
                assert_eq!(
                    controller.requests[&running_id].state,
                    RequestState::Running
                );
                if policy == "barrier-ratio" {
                    finish_request(&mut controller, &running_id);
                    assert_eq!(
                        controller.requests[exclusive_id].state,
                        RequestState::Running
                    );
                }
            }
        }
        let mut work_conserving = Controller::new(Authority::default(), None).unwrap();
        work_conserving.gpu_contention_policy = "exclusive-first".into();
        register_pool(
            &mut work_conserving,
            "shared-node",
            "shared",
            "shared",
            2,
            &["GPU-a"],
        );
        let value = work_conserving
            .submit(pool_spec("shared", "s", "only"), None)
            .unwrap();
        assert_eq!(
            work_conserving.requests[value["request_id"].as_str().unwrap()].state,
            RequestState::Running
        );
    }

    #[test]
    fn fairness_prefers_user_without_an_active_gpu_lease() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register_pool(
            &mut controller,
            "node-a",
            "p",
            "exclusive",
            1,
            &["GPU-a", "GPU-b"],
        );
        controller
            .submit(pool_spec("p", "holder", "first"), None)
            .unwrap();
        controller.nodes.get_mut("node-a").unwrap().state = NodeState::Draining;
        let holder = controller
            .submit(pool_spec("p", "holder", "second"), None)
            .unwrap();
        let newcomer = controller
            .submit(pool_spec("p", "newcomer", "first"), None)
            .unwrap();
        controller.nodes.get_mut("node-a").unwrap().state = NodeState::Active;
        controller.schedule();
        assert_eq!(
            controller.requests[newcomer["request_id"].as_str().unwrap()].state,
            RequestState::Running
        );
        assert_eq!(
            controller.requests[holder["request_id"].as_str().unwrap()].state,
            RequestState::Queued
        );
    }

    #[test]
    fn live_missing_assignment_converges_without_replay() {
        for draining in [false, true] {
            let mut controller = Controller::new(Authority::default(), None).unwrap();
            controller.reconcile_missing_grace_seconds = 0.0;
            controller.reconcile_missing_heartbeats = 2;
            register(&mut controller, "exclusive", 1, "a1");
            let response = controller.submit(spec(), None).unwrap();
            let id = response["request_id"].as_str().unwrap().to_owned();
            let heartbeat = json!({"node_id":"node-a","status":{"authoritative":true,"sessions":[],"command_ids":[]},"max_commands":0});
            for _ in 0..3 { controller.heartbeat(&heartbeat, None).unwrap(); }
            assert_eq!(controller.requests[&id].state, RequestState::Running);
            let mut heartbeat = heartbeat;
            heartbeat["max_commands"] = json!(8);
            let delivered = controller.heartbeat(&heartbeat, None).unwrap();
            let command_id = delivered["commands"][0]["command_id"].as_str().unwrap().to_owned();
            let execution_session_id = controller.requests[&id].execution_session_id.clone().unwrap();
            // A pending result in an idle reusable session is still protected.
            let mut reporting = heartbeat.clone();
            reporting["status"]["sessions"] = json!([{"execution_session_id":execution_session_id,"state":"CACHED_IDLE"}]);
            reporting["status"]["command_ids"] = json!([command_id]);
            for _ in 0..3 { controller.heartbeat(&reporting, None).unwrap(); }
            assert_eq!(controller.requests[&id].state, RequestState::Running);
            if draining { controller.nodes.get_mut("node-a").unwrap().state = NodeState::Draining; }
            controller.heartbeat(&heartbeat, None).unwrap();
            assert_eq!(controller.requests[&id].state, RequestState::Running);
            let mut partial = heartbeat.clone();
            partial["status"]["authoritative"] = json!(false);
            controller.heartbeat(&partial, None).unwrap();
            controller.heartbeat(&heartbeat, None).unwrap();
            assert_eq!(controller.requests[&id].state, RequestState::Running);
            let response = controller.heartbeat(&heartbeat, None).unwrap();
            assert!(response["commands"].as_array().unwrap().iter().all(|c| c["action"] != "run"));
            assert_eq!(controller.requests[&id].reason.as_deref(), Some("assignment_missing"));
            assert_eq!(controller.commands[&command_id].state, CommandState::Failed);
            assert!(controller.nodes["node-a"].active_sessions.is_empty());
            assert!(controller.gpu_ledger.values().all(|entry| entry.exclusive.is_none() && entry.shared.is_empty()));
            controller.report_command(&json!({"node_id":"node-a","command_id":command_id,"exit_code":0}), None).unwrap();
            assert_eq!(controller.requests[&id].state, RequestState::Failed);
            assert_eq!(controller.commands[&command_id].state, CommandState::Failed);
        }
    }

    #[test]
    fn idle_expiry_and_reconciliation_grace_match_python() {
        let mut idle = Controller::new(Authority::default(), None).unwrap();
        register(&mut idle, "exclusive", 1, "a1");
        let value = idle.submit(spec(), None).unwrap();
        let request_id = value["request_id"].as_str().unwrap().to_string();
        finish_request(&mut idle, &request_id);
        let execution_session_id = idle.requests[&request_id].execution_session_id.clone().unwrap();
        let last_used = idle.sessions[&execution_session_id].last_used_at;
        idle.idle_session_timeout_seconds = 10.0;
        idle.expire_idle_sessions_at(last_used + 9.0);
        assert_eq!(idle.sessions[&execution_session_id].state, SessionState::CachedIdle);
        idle.expire_idle_sessions_at(last_used + 10.0);
        assert_eq!(idle.sessions[&execution_session_id].state, SessionState::Suspending);
        assert!(idle.nodes["node-a"].used_cpu_cores > 0.0);
        let workspace_id=idle.sessions[&execution_session_id].workspace.as_ref().unwrap()["id"].as_str().unwrap().to_owned();
        idle.workspaces.get_mut(&workspace_id).unwrap()["state"]=json!("retained");
        idle.nodes.get_mut("node-a").unwrap().status=json!({"authoritative":true,"sessions":[],"suspended_sessions":{execution_session_id.clone():1}});
        idle.reconcile_agent_status("node-a");
        assert_eq!(idle.sessions[&execution_session_id].state,SessionState::Suspended);
        assert_eq!(idle.nodes["node-a"].used_cpu_cores,0.0);
        let resumed=idle.submit(spec(),None).unwrap();
        assert_eq!(resumed["execution_session_id"],execution_session_id);
        assert_eq!(idle.sessions[&execution_session_id].workspace.as_ref().unwrap()["id"],workspace_id);

        let mut reconcile = Controller::new(Authority::default(), None).unwrap();
        register(&mut reconcile, "exclusive", 1, "a1");
        let value = reconcile.submit(spec(), None).unwrap();
        let request_id = value["request_id"].as_str().unwrap().to_string();
        let execution_session_id = reconcile.requests[&request_id].execution_session_id.clone().unwrap();
        reconcile.requests.get_mut(&request_id).unwrap().state = RequestState::Reconciling;
        reconcile.sessions.get_mut(&execution_session_id).unwrap().state = SessionState::Reconciling;
        reconcile.reconcile_missing_grace_seconds = 0.0;
        reconcile.reconcile_missing_heartbeats = 2;
        let heartbeat = json!({"node_id":"node-a","status":{"authoritative":true,"sessions":[]}});
        reconcile.heartbeat(&heartbeat, None).unwrap();
        assert_eq!(
            reconcile.sessions[&execution_session_id].state,
            SessionState::Reconciling
        );
        reconcile.heartbeat(&heartbeat, None).unwrap();
        assert_eq!(reconcile.sessions[&execution_session_id].state, SessionState::Failed);
        assert_eq!(
            reconcile.requests[&request_id].reason.as_deref(),
            Some("reconcile_missing")
        );
        assert_eq!(reconcile.commands[&reconcile.request_commands[&request_id]].state, CommandState::Failed);
    }

    #[test]
    fn profile_and_quota_policy_are_reachable() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        controller
            .set_profiles(BTreeMap::from([(
                "gpu".into(),
                json!({"image":"docker://example/profile:1","runtime_preference":"local-process","gpu_required":true,"env":{"PROFILE":"1"},"default_mounts":[{"source":"/data","target":"/data","read_only":true}]}),
            )]))
            .unwrap();
        register(&mut controller, "exclusive", 1, "a1");
        let profile: WorkloadSpec = serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"profile","command":["true"],"profile":"gpu","resources":{"gpus":0}})).unwrap();
        let value = controller.submit(profile, None).unwrap();
        let request = &controller.requests[value["request_id"].as_str().unwrap()];
        assert_eq!(request.spec.resources.gpus, 1);
        assert_eq!(
            request.spec.image.as_deref(),
            Some("docker://example/profile:1")
        );
        assert_eq!(
            request.spec.env.get("PROFILE").map(String::as_str),
            Some("1")
        );
        assert_eq!(request.spec.mounts[0].target, "/data");

        let mut queued = Controller::new(Authority::default(), None).unwrap();
        queued.max_queued_requests_per_user = Some(0);
        let value = queued.submit(spec(), None).unwrap();
        assert_eq!(
            queued.requests[value["request_id"].as_str().unwrap()].state,
            RequestState::Infeasible
        );
    }

    #[test]
    fn workload_failure_keeps_and_reuses_session() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register(&mut controller, "exclusive", 1, "a1");
        let first = controller.submit(spec(), None).unwrap();
        let first_id = first["request_id"].as_str().unwrap().to_string();
        let command = controller
            .heartbeat(&json!({"node_id":"node-a"}), None)
            .unwrap()["commands"][0]
            .clone();
        let session = controller.requests[&first_id].execution_session_id.clone().unwrap();
        controller.report_command(&json!({"node_id":"node-a","command_id":command["command_id"],"exit_code":7,"failure_kind":"workload"}),None).unwrap();
        assert_eq!(controller.requests[&first_id].state, RequestState::Failed);
        assert_eq!(
            controller.sessions[&session].state,
            SessionState::CachedIdle
        );
        assert!(controller.sessions[&session].gpu_uuids.is_empty());
        let second = controller.submit(spec(), None).unwrap();
        let second_id = second["request_id"].as_str().unwrap();
        assert_eq!(
            controller.requests[second_id].execution_session_id.as_deref(),
            Some(session.as_str())
        );
        controller.assert_invariants().unwrap();
    }

    #[test]
    fn failed_boundary_does_not_destroy_session_or_rewrite_previous_result() {
        let mut controller=Controller::new(Authority::default(),None).unwrap();
        register(&mut controller,"exclusive",1,"a1");
        let first=controller.submit(spec(),None).unwrap();let first_id=first["request_id"].as_str().unwrap().to_string();
        let command=controller.heartbeat(&json!({"node_id":"node-a"}),None).unwrap()["commands"][0].clone();
        let session=controller.requests[&first_id].execution_session_id.clone().unwrap();
        controller.report_command(&json!({"node_id":"node-a","command_id":command["command_id"],"exit_code":0}),None).unwrap();
        let workspace=controller.sessions[&session].workspace.as_ref().unwrap()["id"].as_str().unwrap().to_string();
        controller.workspaces.get_mut(&workspace).unwrap()["operation"]=json!({"id":"failed","boundary":true,"state":"failed","error":"injected"});
        let second=controller.submit(spec(),None).unwrap();let second_id=second["request_id"].as_str().unwrap();
        controller.schedule();
        assert_eq!(controller.requests[second_id].reason.as_deref(),Some("workspace_maintenance_failed"));
        assert_eq!(controller.requests[second_id].failure_kind.as_deref(),Some("infrastructure"));
        assert!(controller.requests[second_id].started_at.is_none());
        assert_eq!(controller.requests[&first_id].state,RequestState::Completed);
        assert_eq!(controller.sessions[&session].state,SessionState::CachedIdle);
    }

    #[test]
    fn reusable_work_runs_before_reclaiming_a_cpu_saturated_cache() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register(&mut controller, "exclusive", 1, "a1");
        let first = controller.submit(spec_for("u1", "t1"), None).unwrap();
        let first_id = first["request_id"].as_str().unwrap().to_string();
        let first_command = controller
            .heartbeat(&json!({"node_id":"node-a"}), None)
            .unwrap()["commands"][0]
            .clone();
        let other = controller.submit(spec_for("u2", "t2"), None).unwrap();
        let other_id = other["request_id"].as_str().unwrap().to_string();
        let reuse = controller.submit(spec_for("u1", "t1"), None).unwrap();
        let reuse_id = reuse["request_id"].as_str().unwrap().to_string();
        let session = controller.requests[&first_id].execution_session_id.clone().unwrap();
        controller.report_command(&json!({"node_id":"node-a","command_id":first_command["command_id"],"exit_code":0,"failure_kind":"workload"}),None).unwrap();
        assert_eq!(controller.requests[&reuse_id].state, RequestState::Running);
        assert_eq!(
            controller.requests[&reuse_id].execution_session_id.as_deref(),
            Some(session.as_str())
        );
        assert_eq!(controller.requests[&other_id].state, RequestState::Queued);
        let reuse_command = controller
            .heartbeat(&json!({"node_id":"node-a"}), None)
            .unwrap()["commands"][0]
            .clone();
        controller.report_command(&json!({"node_id":"node-a","command_id":reuse_command["command_id"],"exit_code":0,"failure_kind":"workload"}),None).unwrap();
        assert_eq!(controller.requests[&other_id].state, RequestState::Running);
        assert_ne!(
            controller.requests[&other_id].execution_session_id.as_deref(),
            Some(session.as_str())
        );
        controller.assert_invariants().unwrap();
    }

    #[test]
    fn infrastructure_failure_tears_session_down() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register(&mut controller, "exclusive", 1, "a1");
        let first = controller.submit(spec(), None).unwrap();
        let id = first["request_id"].as_str().unwrap().to_string();
        let command = controller
            .heartbeat(&json!({"node_id":"node-a"}), None)
            .unwrap()["commands"][0]
            .clone();
        let session = controller.requests[&id].execution_session_id.clone().unwrap();
        controller.report_command(&json!({"node_id":"node-a","command_id":command["command_id"],"exit_code":127,"failure_kind":"infrastructure"}),None).unwrap();
        assert_eq!(controller.sessions[&session].state, SessionState::Failed);
        assert!(controller.nodes["node-a"].active_sessions.is_empty());
    }

    #[test]
    fn zero_exit_cannot_report_infrastructure_failure() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register(&mut controller, "exclusive", 1, "a1");
        controller.submit(spec(), None).unwrap();
        let command = controller
            .heartbeat(&json!({"node_id":"node-a"}), None)
            .unwrap()["commands"][0]
            .clone();
        let error=controller.report_command(&json!({"node_id":"node-a","command_id":command["command_id"],"exit_code":0,"failure_kind":"infrastructure"}),None).unwrap_err();
        assert!(error.message.contains("nonzero exit_code"));
    }

    #[test]
    fn empty_pool_can_change_shared_slot_count() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register(&mut controller, "shared", 1, "a1");
        controller.remove_allocation("a1", true, None).unwrap();
        register(&mut controller, "shared", 4, "a2");
        assert_eq!(controller.pool_configs["p"].gpu_slots, 4);
        assert_eq!(controller.utilization("p")["gpu_slots_total"], 4);
    }

    #[test]
    fn command_timeout_controls_admission_with_legacy_fallback() {
        let deadline = now() + 900.0;
        let body = json!({"node_id":"node-a","pool":"p","pool_mode":"exclusive","gpu_slots":1,"placement":"pack","resources":{"cpu_cores":8,"memory_gb":64,"gpu_uuids":["GPU-a"]},"capabilities":{"adapter":"local-process","isolation_policies":["default","none"]},"runtime_capabilities":{"local-process":{"adapter":"local-process","image_formats":["default"],"isolation_policies":["default","none"],"can_select_gpus":true,"gpu_isolation_policies":["default","none"]}},"allocation_id":"a1","allocation_deadline":deadline,"runtime_cache_config":"default"});
        let mut accepted = Controller::new(Authority::default(), None).unwrap();
        accepted.announce_or_register(&body, true, None).unwrap();
        let short:WorkloadSpec=serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"t","command":["true"],"resources":{"gpus":1,"max_duration_seconds":14400,"command_timeout_seconds":10}})).unwrap();
        let value = accepted.submit(short, None).unwrap();
        let id = value["request_id"].as_str().unwrap();
        assert_eq!(accepted.requests[id].state, RequestState::Running);
        assert_eq!(
            accepted.sessions[accepted.requests[id].execution_session_id.as_ref().unwrap()]
                .max_lifetime_seconds,
            14400.0
        );

        let mut rejected = Controller::new(Authority::default(), None).unwrap();
        rejected.announce_or_register(&body, true, None).unwrap();
        let long:WorkloadSpec=serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"t","command":["true"],"resources":{"gpus":1,"max_duration_seconds":14400,"command_timeout_seconds":1200}})).unwrap();
        let value = rejected.submit(long, None).unwrap();
        let id = value["request_id"].as_str().unwrap();
        assert_eq!(rejected.requests[id].state, RequestState::Queued);
        assert_eq!(
            rejected.requests[id].reason.as_deref(),
            Some("allocation_expiring")
        );

        let mut legacy = Controller::new(Authority::default(), None).unwrap();
        legacy.announce_or_register(&body, true, None).unwrap();
        let old:WorkloadSpec=serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"t","command":["true"],"resources":{"gpus":1,"max_duration_seconds":14400}})).unwrap();
        let value = legacy.submit(old, None).unwrap();
        let id = value["request_id"].as_str().unwrap();
        assert_eq!(
            legacy.requests[id].reason.as_deref(),
            Some("allocation_expiring")
        );
    }

    #[test]
    fn no_deadline_stays_active_and_node_loss_cascades() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        register(&mut controller, "exclusive", 1, "a1");
        assert_eq!(controller.heartbeat_timeout_seconds, 30.0);
        assert_eq!(controller.lost_timeout_seconds, 90.0);
        assert_eq!(controller.nodes["node-a"].allocation_deadline, None);
        let value = controller.submit(spec(), None).unwrap();
        let id = value["request_id"].as_str().unwrap().to_string();
        let heartbeat = controller.nodes["node-a"].last_heartbeat_at;
        controller.check_heartbeats_at(heartbeat + 31.0);
        assert_eq!(controller.nodes["node-a"].state, NodeState::Unhealthy);
        controller.check_heartbeats_at(heartbeat + 91.0);
        assert_eq!(controller.nodes["node-a"].state, NodeState::Lost);
        assert_eq!(controller.requests[&id].state, RequestState::Lost);
        assert_eq!(
            controller.requests[&id].reason.as_deref(),
            Some("node_lost")
        );
        assert_eq!(
            controller.requests[&id].failure_kind.as_deref(),
            Some("infrastructure")
        );
        assert_eq!(controller.requests[&id].exit_code, None);
        assert!(controller.nodes["node-a"].active_sessions.is_empty());
        assert_eq!(controller.gpu_ledger["GPU-a"].state(), GpuLeaseState::Free);
    }

    #[test]
    fn emitted_request_reasons_have_messages() {
        for reason in [
            "pool_empty",
            "node_probing",
            "no_capable_node",
            "allocation_expiring",
            "node_draining",
            "command_failed",
            "infrastructure_failed",
            "cancelled",
            "node_lost",
        ] {
            assert!(
                request_reason_message(reason).is_some(),
                "missing message for {reason}"
            );
        }
    }

    #[test]
    fn queued_reasons_distinguish_gpu_runtime_and_locality() {
        let mut gpu = Controller::new(Authority::default(), None).unwrap();
        register(&mut gpu, "exclusive", 1, "a1");
        gpu.submit(spec_for("u1", "t1"), None).unwrap();
        let second = gpu.submit(spec_for("u2", "t2"), None).unwrap();
        let second_id = second["request_id"].as_str().unwrap();
        assert_eq!(
            gpu.requests[second_id].reason.as_deref(),
            Some("waiting_for_gpu")
        );

        let mut runtime = Controller::new(Authority::default(), None).unwrap();
        register(&mut runtime, "exclusive", 1, "a1");
        let unsupported:WorkloadSpec=serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"t","command":["true"],"resources":{"gpus":1,"runtime_adapter":"bubblewrap","isolation_policy":"host-stateless"}})).unwrap();
        let value = runtime.submit(unsupported, None).unwrap();
        let id = value["request_id"].as_str().unwrap();
        assert_eq!(
            runtime.requests[id].reason.as_deref(),
            Some("waiting_for_runtime_capability")
        );

        let mismatched:WorkloadSpec=serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"locality","command":["true"],"resources":{"gpus":1,"locality":"another-node"}})).unwrap();
        let value = runtime.submit(mismatched, None).unwrap();
        let id = value["request_id"].as_str().unwrap();
        assert_eq!(
            runtime.requests[id].reason.as_deref(),
            Some("no_capable_node")
        );
    }

    #[test]
    fn hard_gpu_request_refuses_advisory_runtime() {
        let mut controller = Controller::new(Authority::default(), None).unwrap();
        let body = json!({"node_id":"node-a","pool":"p","pool_mode":"exclusive","gpu_slots":1,"placement":"pack","resources":{"cpu_cores":8,"memory_gb":64,"gpu_uuids":["GPU-a"]},"capabilities":{"adapter":"bubblewrap","isolation_policies":["default","host-stateless"]},"runtime_capabilities":{"bubblewrap":{"adapter":"bubblewrap","image_formats":["host"],"isolation_policies":["default","host-stateless"],"can_select_gpus":true,"gpu_enforcement":"advisory","gpu_isolation_policies":["default","host-stateless"]}},"allocation_id":"a1","runtime_cache_config":"default"});
        controller.announce_or_register(&body, true, None).unwrap();
        let hard:WorkloadSpec=serde_json::from_value(json!({"pool":"p","user_id":"u","session_id":"hard","command":["true"],"resources":{"gpus":1,"runtime_adapter":"bubblewrap","isolation_policy":"host-stateless","hard_gpu_enforcement":true}})).unwrap();
        let value = controller.submit(hard, None).unwrap();
        let id = value["request_id"].as_str().unwrap();
        assert_eq!(
            controller.requests[id].reason.as_deref(),
            Some("waiting_for_runtime_capability")
        );
    }

    #[test]
    fn emitted_session_and_node_reasons_have_messages() {
        for reason in [
            "command_failed",
            "infrastructure_failed",
            "node_lost",
            "idle_timeout",
            "max_lifetime",
            "stopped",
            "evict_for_queued_work",
            "allocation_drain",
            "allocation_removed",
            "node_removed",
            "operator_drain",
            "allocation_expiring",
            "reconcile_missing",
            "lost_session_cleanup_confirmed",
        ] {
            assert!(
                session_reason_message(reason).is_some(),
                "missing session message for {reason}"
            );
        }
        for reason in [
            "heartbeat_timeout",
            "heartbeat_lost",
            "allocation_expiring",
            "allocation_deadline",
            "allocation_drain",
            "allocation_removed",
            "node_removed",
            "operator_drain",
            "remove_pending_sessions",
        ] {
            assert!(
                node_reason_message(reason).is_some(),
                "missing node message for {reason}"
            );
        }
    }
}
