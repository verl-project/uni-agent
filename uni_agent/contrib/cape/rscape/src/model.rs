// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RequestState {
    Submitted,
    Queued,
    Infeasible,
    Leased,
    PreparingImage,
    StartingSession,
    Running,
    Completed,
    Failed,
    Cancelled,
    Expired,
    Lost,
    Reconciling,
}

impl RequestState {
    pub fn terminal(self) -> bool {
        matches!(
            self,
            Self::Infeasible
                | Self::Completed
                | Self::Failed
                | Self::Cancelled
                | Self::Expired
                | Self::Lost
        )
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SessionState {
    Creating,
    WarmIdle,
    CachedIdle,
    Suspending,
    Suspended,
    Purging,
    Purged,
    RunningCommand,
    Draining,
    Stopping,
    Stopped,
    Failed,
    Lost,
    Reconciling,
}

impl SessionState {
    pub fn terminal(self) -> bool {
        matches!(self, Self::Stopped | Self::Failed | Self::Lost | Self::Purged)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeState {
    Registering,
    Active,
    Draining,
    Unhealthy,
    Removing,
    Removed,
    Lost,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CommandState {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AllocationState {
    Active,
    Draining,
    Removing,
    Removed,
    Lost,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GpuLeaseState {
    Free,
    Exclusive,
    Shared,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EnforcementMode {
    Hard,
    Advisory,
    None,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum Role {
    #[serde(rename = "submitter")]
    Submitter,
    #[serde(rename = "viewer")]
    Viewer,
    #[serde(rename = "operator")]
    Operator,
    #[serde(rename = "admin")]
    Admin,
    #[serde(rename = "node-agent")]
    NodeAgent,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MountSpec {
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub read_only: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResourceRequest {
    #[serde(default)]
    pub gpus: usize,
    #[serde(default = "one_f64")]
    pub cpu_cores: f64,
    #[serde(default = "one_f64")]
    pub memory_gb: f64,
    #[serde(default = "default_duration")]
    pub max_duration_seconds: f64,
    #[serde(default)]
    pub command_timeout_seconds: Option<f64>,
    #[serde(default)]
    pub hard_gpu_enforcement: bool,
    #[serde(default)]
    pub runtime_adapter: Option<String>,
    #[serde(default)]
    pub ports: Vec<u16>,
    #[serde(default)]
    pub locality: Option<String>,
    #[serde(default = "default_policy")]
    pub isolation_policy: String,
}

fn one_f64() -> f64 {
    1.0
}
fn default_duration() -> f64 {
    3600.0
}
fn default_policy() -> String {
    "default".to_string()
}
fn true_value() -> bool {
    true
}

impl Default for ResourceRequest {
    fn default() -> Self {
        Self {
            gpus: 0,
            cpu_cores: 1.0,
            memory_gb: 1.0,
            max_duration_seconds: 3600.0,
            command_timeout_seconds: None,
            hard_gpu_enforcement: false,
            runtime_adapter: None,
            ports: Vec::new(),
            locality: None,
            isolation_policy: default_policy(),
        }
    }
}

impl ResourceRequest {
    pub fn admission_timeout_seconds(&self) -> f64 {
        self.command_timeout_seconds
            .unwrap_or(self.max_duration_seconds)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WorkloadSpec {
    pub pool: String,
    pub user_id: String,
    pub session_id: String,
    pub command: Vec<String>,
    #[serde(default)]
    pub resources: ResourceRequest,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub mounts: Vec<MountSpec>,
    #[serde(default = "true_value")]
    pub reuse: bool,
    #[serde(default = "true_value")]
    pub keep_session_on_failure: bool,
    #[serde(default)]
    pub priority: i64,
    #[serde(default)]
    pub privileged: bool,
    #[serde(default)]
    pub log_path: Option<String>,
    #[serde(default)]
    pub artifact_paths: Vec<String>,
    #[serde(default)]
    pub workspace: crate::workspace::WorkspaceSelection,
}

impl WorkloadSpec {
    pub fn image_identity(&self) -> String {
        if let Some(image) = &self.image {
            image.clone()
        } else if let Some(profile) = &self.profile {
            format!("profile:{profile}")
        } else {
            "docker://ubuntu:latest".to_string()
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PoolConfig {
    #[serde(default = "exclusive_mode")]
    pub pool_mode: String,
    #[serde(default = "one_usize")]
    pub gpu_slots: usize,
    #[serde(default = "pack_placement")]
    pub placement: String,
}
fn exclusive_mode() -> String {
    "exclusive".into()
}
fn one_usize() -> usize {
    1
}
fn pack_placement() -> String {
    "pack".into()
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            pool_mode: exclusive_mode(),
            gpu_slots: 1,
            placement: pack_placement(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct NodeResources {
    #[serde(default)]
    pub cpu_cores: f64,
    #[serde(default)]
    pub memory_gb: f64,
    #[serde(default)]
    pub gpu_uuids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RequestRecord {
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission_token_id: Option<String>,
    pub spec: WorkloadSpec,
    pub state: RequestState,
    pub reason: Option<String>,
    pub created_at: Value,
    pub updated_at: Value,
    #[serde(default)]
    pub leased_at: Option<Value>,
    pub started_at: Option<Value>,
    pub ended_at: Option<Value>,
    pub sequence: u64,
    pub execution_session_id: Option<String>,
    pub node_id: Option<String>,
    pub gpu_uuids: Vec<String>,
    pub gpu_mode: Option<String>,
    pub gpu_cotenant_count: usize,
    pub gpu_cotenant_counts: BTreeMap<String, usize>,
    pub exit_code: Option<i32>,
    pub failure_kind: Option<String>,
    pub stdout: String,
    pub stderr: String,
    pub log_path: Option<String>,
    pub artifact_paths: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SessionRecord {
    #[serde(default)]
    pub state_layer_id: Option<String>,
    #[serde(default)]
    pub suspend_version: u64,
    #[serde(default)]
    pub suspend_requested: bool,
    #[serde(default)]
    pub resume_required: bool,
    #[serde(default)]
    pub activated_at: f64,
    #[serde(default)]
    pub empty_retained: bool,
    pub execution_session_id: String,
    pub pool: String,
    pub user_id: String,
    pub session_id: String,
    pub node_id: String,
    pub runtime_adapter: String,
    pub image_identity: String,
    pub isolation_policy: String,
    #[serde(default = "no_enforcement")]
    pub enforcement_mode: String,
    pub mounts: Vec<MountSpec>,
    pub privileged: bool,
    pub gpu_count: usize,
    pub gpu_uuids: Vec<String>,
    pub cpu_cores: f64,
    pub memory_gb: f64,
    pub state: SessionState,
    pub active_request_id: Option<String>,
    pub created_at: f64,
    pub last_used_at: f64,
    pub max_lifetime_seconds: f64,
    pub gpu_mode: String,
    pub gpu_slots: usize,
    #[serde(default)]
    pub workspace: Option<Value>,
}

fn no_enforcement() -> String {
    "none".into()
}

impl SessionRecord {
    pub fn reusable_for(&self, spec: &WorkloadSpec) -> bool {
        self.state == SessionState::CachedIdle
            && self.matches_reuse_identity(spec)
    }

    pub fn matches_reuse_identity(&self, spec: &WorkloadSpec) -> bool {
        self.workspace.as_ref().map(|workspace| {
                workspace.get("target").and_then(Value::as_str).unwrap_or("/workspace") == spec.workspace.target
                    && spec.workspace.id.as_ref().map(|id| workspace.get("id").and_then(Value::as_str) == Some(id.as_str())).unwrap_or(true)
            }).unwrap_or(spec.workspace.id.is_none())
            && spec.reuse
            && self.pool == spec.pool
            && self.user_id == spec.user_id
            && self.session_id == spec.session_id
            && self.image_identity == spec.image_identity()
            && self.isolation_policy == spec.resources.isolation_policy
            && self.mounts == spec.mounts
            && self.privileged == spec.privileged
            && self.gpu_count == spec.resources.gpus
            && self.cpu_cores >= spec.resources.cpu_cores
            && self.memory_gb >= spec.resources.memory_gb
            && spec.resources.ports.is_empty()
            && spec
                .resources
                .runtime_adapter
                .as_ref()
                .map(|value| value == &self.runtime_adapter)
                .unwrap_or(true)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeRecord {
    pub node_id: String,
    pub pool: String,
    pub state: NodeState,
    pub resources: NodeResources,
    pub runtime_capabilities: BTreeMap<String, Value>,
    pub capabilities: Value,
    pub allocation_id: Option<String>,
    pub allocation_deadline: Option<f64>,
    pub drain_before_sec: f64,
    pub cleanup_buffer_sec: f64,
    pub runtime_cache_config: String,
    pub cached_images: Vec<String>,
    pub pool_config: PoolConfig,
    pub agent_epoch: Option<String>,
    pub registered_at: f64,
    pub last_heartbeat_at: f64,
    pub used_cpu_cores: f64,
    pub used_memory_gb: f64,
    pub active_sessions: Vec<String>,
    pub status: Value,
    #[serde(default)]
    pub workspace_status: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CommandAssignment {
    pub command_id: String,
    pub request_id: String,
    pub execution_session_id: String,
    pub node_id: String,
    pub session_spec: Value,
    pub command_spec: Value,
    pub state: CommandState,
    pub created_at: f64,
    pub updated_at: f64,
    pub log_path: Option<String>,
    pub result: Value,
}

impl CommandAssignment {
    pub fn wire(&self) -> Value {
        json!({
            "action": "run",
            "command_id": self.command_id,
            "request_id": self.request_id,
            "execution_session_id": self.execution_session_id,
            "node_id": self.node_id,
            "session_spec": self.session_spec,
            "command_spec": self.command_spec,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "log_path": self.log_path,
            "result": self.result,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AllocationRecord {
    pub allocation_id: String,
    pub pool: String,
    pub state: AllocationState,
    pub node_ids: BTreeSet<String>,
    pub deadline: Option<f64>,
    pub created_at: f64,
    pub updated_at: f64,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct GpuLedgerEntry {
    pub exclusive: Option<String>,
    pub shared: BTreeSet<String>,
    pub shared_capacity: Option<usize>,
}

impl GpuLedgerEntry {
    pub fn state(&self) -> GpuLeaseState {
        if self.exclusive.is_some() {
            GpuLeaseState::Exclusive
        } else if !self.shared.is_empty() {
            GpuLeaseState::Shared
        } else {
            GpuLeaseState::Free
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NodeQueues {
    pub commands: VecDeque<String>,
    pub controls: VecDeque<Value>,
}
