// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub fn workspace_bind_conflict(bind: &str, workspace: &str) -> bool {
    let mut components = Vec::new();
    for component in bind.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            value => components.push(value),
        }
    }
    let target = format!("/{}", components.join("/"));
    target == "/" || target == workspace || workspace.starts_with(&format!("{target}/"))
}
use std::collections::BTreeMap;

const MIN_IMAGE_SIZE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_INTEGER: u64 = i64::MAX as u64;

pub fn automatic_grow_target(
    record: &Value,
    config: &WorkspaceConfig,
    now: f64,
) -> Result<Option<Value>> {
    let policy = config.maintenance.clone().unwrap_or_default();
    if !policy.auto_grow
        || record["state"] != "mounted"
        || record["operation"]["state"] == "pending"
        || now - record["last_maintenance_at"].as_f64().unwrap_or(-1e30)
            < policy.grow_cooldown_seconds as f64
    {
        return Ok(None);
    }
    let space = &record["space"]["internal"];
    if space.is_null() {
        return Ok(None);
    }
    if !space.is_object()
        || [
            "total_bytes",
            "available_bytes",
            "total_inodes",
            "free_inodes",
        ]
        .iter()
        .any(|k| {
            space[*k]
                .as_u64()
                .map(|v| v > i64::MAX as u64)
                .unwrap_or(true)
        })
    {
        bail!("invalid capacity telemetry");
    }
    if space["available_bytes"].as_u64() > space["total_bytes"].as_u64()
        || space["free_inodes"].as_u64() > space["total_inodes"].as_u64()
    {
        bail!("invalid capacity telemetry");
    }
    let total = space["total_bytes"].as_u64().unwrap_or(0) as u128;
    let used = total.saturating_sub(space["available_bytes"].as_u64().unwrap_or(0) as u128);
    let inodes = space["total_inodes"].as_u64().unwrap_or(0) as u128;
    let used_inodes = inodes.saturating_sub(space["free_inodes"].as_u64().unwrap_or(0) as u128);
    let bytes = total > 0 && used * 100 >= total * policy.grow_threshold_percent as u128;
    let inode_pressure =
        inodes > 0 && used_inodes * 100 >= inodes * policy.grow_threshold_percent as u128;
    if !bytes && !inode_pressure {
        return Ok(None);
    }
    let current = record["current_image_size_bytes"]
        .as_u64()
        .or_else(|| record["config"]["image_size_bytes"].as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing image capacity"))? as u128;
    let desired = policy.grow_target_percent as u128;
    let mut target = current + policy.grow_min_bytes as u128;
    if bytes {
        target = target.max((used * 100 + desired - 1) / desired + current.saturating_sub(total));
    }
    if inode_pressure {
        let group = record["geometry"]["blocks_per_group"].as_u64().unwrap_or(0) as u128
            * record["geometry"]["block_size"].as_u64().unwrap_or(0) as u128;
        if group == 0 {
            bail!("inode-aware growth requires recorded filesystem geometry");
        }
        let groups = (current + group - 1) / group;
        let per_group = inodes / groups;
        if per_group == 0 {
            bail!("invalid inode geometry");
        }
        let wanted = (used_inodes * 100 + desired - 1) / desired;
        target = target.max((groups + 1).max((wanted + per_group - 1) / per_group) * group);
    }
    let unit = 1024_u128 * 1024;
    target = ((target + unit - 1) / unit) * unit;
    if target > config.max_image_size_bytes as u128 {
        bail!("automatic growth exceeds max_image_size_bytes");
    }
    Ok(Some(
        serde_json::json!({"target_bytes":target as u64,"reason":if inode_pressure {"inodes"} else {"bytes"},"policy":policy}),
    ))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct WorkspaceSelection {
    pub id: Option<String>,
    pub target: String,
}

impl Default for WorkspaceSelection {
    fn default() -> Self {
        Self {
            id: None,
            target: "/workspace".into(),
        }
    }
}

impl WorkspaceSelection {
    pub fn validate(&mut self) -> Result<()> {
        if let Some(id) = &self.id {
            if id.len() != 35
                || !id.starts_with("ws-")
                || !id[3..]
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                bail!("invalid workspace.id");
            }
        }
        self.target = WorkspaceConfig::from_value(serde_json::json!({"image_root":self.target}))
            .map_err(|error| anyhow::anyhow!("workspace target: {error}"))?
            .image_root;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct WorkspaceConfig {
    pub version: u64,
    pub image_root: String,
    pub image_size_bytes: u64,
    pub max_image_size_bytes: u64,
    pub bytes_per_inode: u64,
    pub mount_timeout_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maintenance: Option<crate::workspace_fs::MaintenancePolicy>,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            version: 0,
            image_root: "/tmp".into(),
            image_size_bytes: 10 * 1024_u64.pow(3),
            max_image_size_bytes: 1024_u64.pow(4),
            bytes_per_inode: 16384,
            mount_timeout_seconds: 30,
            maintenance: None,
        }
    }
}

impl WorkspaceConfig {
    pub fn from_value(value: Value) -> Result<Self> {
        let mut config: Self = serde_json::from_value(value)?;
        config.validate()?;
        Ok(config)
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("workspace config contains only strings and integers")
    }

    #[cfg(test)]
    pub fn updated(&self, changes: &Map<String, Value>) -> Result<Self> {
        if changes.contains_key("version") {
            bail!("workspace configuration version is controller-owned");
        }
        if changes.is_empty() {
            bail!("workspace configuration update must not be empty");
        }
        let mut value = self.to_value();
        let values = value
            .as_object_mut()
            .expect("serialized workspace configuration is an object");
        values.extend(changes.clone());
        values.insert(
            "version".into(),
            Value::from(
                self.version
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("workspace configuration version overflow"))?,
            ),
        );
        Self::from_value(value)
    }

    fn validate(&mut self) -> Result<()> {
        if let Some(policy) = &self.maintenance {
            crate::workspace_fs::MaintenancePolicy::from_value(serde_json::to_value(policy)?)?;
        }
        for (name, value, minimum, maximum) in [
            ("version", self.version, 0, MAX_INTEGER),
            (
                "image_size_bytes",
                self.image_size_bytes,
                MIN_IMAGE_SIZE_BYTES,
                MAX_INTEGER,
            ),
            (
                "max_image_size_bytes",
                self.max_image_size_bytes,
                MIN_IMAGE_SIZE_BYTES,
                MAX_INTEGER,
            ),
            (
                "bytes_per_inode",
                self.bytes_per_inode,
                1024,
                1024_u64.pow(3),
            ),
            ("mount_timeout_seconds", self.mount_timeout_seconds, 1, 300),
        ] {
            if value < minimum || value > maximum {
                bail!("{name} must be an integer in [{minimum}, {maximum}]");
            }
        }
        if self.image_size_bytes > self.max_image_size_bytes {
            bail!("image_size_bytes exceeds max_image_size_bytes");
        }
        if !self.image_root.starts_with('/') {
            bail!("image_root must be an absolute node-side path");
        }
        if self.image_root.chars().any(|c| c < ' ' || c == '\u{7f}')
            || self.image_root.split('/').any(|part| part == "..")
        {
            bail!("image_root contains forbidden path components");
        }
        let parts: Vec<&str> = self
            .image_root
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .collect();
        let normalized = format!("/{}", parts.join("/"));
        if matches!(normalized.as_str(), "/" | "/dev" | "/run")
            || matches!(parts.first(), Some(&"proc") | Some(&"sys"))
        {
            bail!("image_root must not be a filesystem root or virtual kernel tree");
        }
        self.image_root = normalized;
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceConfigRecord {
    pub version: u64,
    pub defaults: Map<String, Value>,
    pub nodes: BTreeMap<String, Map<String, Value>>,
}

#[derive(Clone, Default)]
pub struct WorkspaceConfigStore {
    pools: BTreeMap<String, WorkspaceConfigRecord>,
}

impl WorkspaceConfigStore {
    fn identity(value: &str, name: &str) -> Result<()> {
        if value.is_empty() {
            bail!("{name} must be a nonempty string");
        }
        Ok(())
    }

    fn effective(record: &WorkspaceConfigRecord, node_id: Option<&str>) -> Result<WorkspaceConfig> {
        let mut values = record.defaults.clone();
        if let Some(overrides) = node_id.and_then(|node| record.nodes.get(node)) {
            let mut policy = values
                .get("maintenance")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            if let Some(fields) = overrides.get("maintenance").and_then(Value::as_object) {
                policy.extend(fields.clone());
            }
            values.extend(overrides.clone());
            if !policy.is_empty() {
                values.insert("maintenance".into(), Value::Object(policy));
            }
        }
        values.insert("version".into(), Value::from(record.version));
        WorkspaceConfig::from_value(Value::Object(values))
    }

    pub fn get(&self, pool: &str, node_id: Option<&str>) -> Result<WorkspaceConfig> {
        Self::identity(pool, "pool")?;
        if let Some(node) = node_id {
            Self::identity(node, "node_id")?;
        }
        Self::effective(&self.pools.get(pool).cloned().unwrap_or_default(), node_id)
    }

    fn validate_overrides(values: &Map<String, Value>) -> Result<()> {
        let allowed = [
            "image_root",
            "image_size_bytes",
            "max_image_size_bytes",
            "bytes_per_inode",
            "mount_timeout_seconds",
            "maintenance",
        ];
        let invalid: Vec<&str> = values
            .keys()
            .map(String::as_str)
            .filter(|key| !allowed.contains(key))
            .collect();
        if !invalid.is_empty() {
            bail!("invalid workspace override fields: {}", invalid.join(", "));
        }
        if let Some(policy) = values.get("maintenance") {
            if !policy.is_object() {
                bail!("maintenance policy must be an object");
            }
            crate::workspace_fs::MaintenancePolicy::from_value(policy.clone())?;
        }
        Ok(())
    }

    fn validate_record(record: &WorkspaceConfigRecord) -> Result<()> {
        Self::validate_overrides(&record.defaults)?;
        Self::effective(record, None)?;
        for (node, overrides) in &record.nodes {
            Self::identity(node, "node_id")?;
            Self::validate_overrides(overrides)?;
            Self::effective(record, Some(node))?;
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        pool: &str,
        changes: &Map<String, Value>,
        node_id: Option<&str>,
    ) -> Result<Value> {
        self.get(pool, node_id)?;
        Self::validate_overrides(changes)?;
        if changes.is_empty() {
            bail!("workspace configuration update must not be empty");
        }
        let mut record = self.pools.get(pool).cloned().unwrap_or_default();
        let target = match node_id {
            Some(node) => record.nodes.entry(node.into()).or_default(),
            None => &mut record.defaults,
        };
        let mut changes = changes.clone();
        if let Some(policy) = changes
            .get_mut("maintenance")
            .and_then(Value::as_object_mut)
        {
            let mut merged = target
                .get("maintenance")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            merged.extend(policy.clone());
            *policy = merged;
        }
        target.extend(changes);
        record.version = record
            .version
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("workspace configuration version overflow"))?;
        Self::validate_record(&record)?;
        let value = serde_json::to_value(&record)?;
        self.pools.insert(pool.into(), record);
        Ok(value)
    }

    pub fn restore(&mut self, pool: &str, value: Value) -> Result<()> {
        Self::identity(pool, "pool")?;
        let record: WorkspaceConfigRecord = serde_json::from_value(value)?;
        Self::validate_record(&record)?;
        self.pools.insert(pool.into(), record);
        Ok(())
    }

    pub fn record(&self, pool: &str) -> Result<Value> {
        Self::identity(pool, "pool")?;
        Ok(serde_json::to_value(
            self.pools.get(pool).cloned().unwrap_or_default(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pool_node_precedence_replay_and_atomic_validation() {
        let mut store = WorkspaceConfigStore::default();
        store
            .update(
                "pool",
                json!({"image_root":"/scratch/pool"}).as_object().unwrap(),
                None,
            )
            .unwrap();
        let snapshot = store
            .update(
                "pool",
                json!({"image_root":"/scratch/node", "image_size_bytes":20 * 1024_u64.pow(3)})
                    .as_object()
                    .unwrap(),
                Some("node"),
            )
            .unwrap();
        assert_eq!(
            store.get("pool", Some("node")).unwrap().image_root,
            "/scratch/node"
        );
        assert_eq!(
            store.get("pool", Some("other")).unwrap().image_root,
            "/scratch/pool"
        );
        assert_eq!(store.get("pool", Some("node")).unwrap().version, 2);
        assert!(store
            .update(
                "pool",
                json!({"max_image_size_bytes":15 * 1024_u64.pow(3)})
                    .as_object()
                    .unwrap(),
                None
            )
            .is_err());
        assert_eq!(store.record("pool").unwrap(), snapshot);
        let mut restored = WorkspaceConfigStore::default();
        restored.restore("pool", snapshot).unwrap();
        assert_eq!(
            restored.get("pool", Some("node")).unwrap(),
            store.get("pool", Some("node")).unwrap()
        );
        assert!(store
            .update("pool", json!({"version":10}).as_object().unwrap(), None)
            .is_err());
        assert!(store
            .update("pool", json!({"bad":1}).as_object().unwrap(), None)
            .is_err());
        assert!(store.update("pool", &Map::new(), None).is_err());
        assert!(store
            .update(
                "pool",
                json!({"image_root":"/tmp"}).as_object().unwrap(),
                Some("")
            )
            .is_err());
        assert!(store
            .restore(
                "pool",
                json!({"version":1,"defaults":{},"nodes":{"n":{"version":5}}})
            )
            .is_err());
    }

    #[test]
    fn defaults_roundtrip_and_updates_leave_old_snapshot_unchanged() {
        let original = WorkspaceConfig::default();
        assert_eq!(
            WorkspaceConfig::from_value(original.to_value()).unwrap(),
            original
        );
        let update =
            json!({"image_root":"/scratch//work/./", "image_size_bytes":20 * 1024_u64.pow(3)});
        let changed = original.updated(update.as_object().unwrap()).unwrap();
        assert_eq!(changed.image_root, "/scratch/work");
        assert_eq!(changed.version, 1);
        assert_eq!(original.image_root, "/tmp");
        assert_eq!(original.image_size_bytes, 10 * 1024_u64.pow(3));
        assert_eq!(changed.bytes_per_inode, original.bytes_per_inode);
    }

    #[test]
    fn rejects_invalid_config_and_controller_version_overrides() {
        for value in [
            json!({"image_size_bytes":true}),
            json!({"image_size_bytes":1}),
            json!({"image_size_bytes":1.5}),
            json!({"image_size_bytes":-1}),
            json!({"image_size_bytes":20 * 1024_u64.pow(4)}),
            json!({"image_size_bytes":1_u64 << 63}),
            json!({"bytes_per_inode":0}),
            json!({"mount_timeout_seconds":301}),
            json!({"unknown":1}),
            Value::Null,
        ] {
            assert!(
                WorkspaceConfig::from_value(value.clone()).is_err(),
                "{value}"
            );
        }
        assert!(WorkspaceConfig::default().updated(&Map::new()).is_err());
        assert!(WorkspaceConfig::default()
            .updated(json!({"version":2}).as_object().unwrap())
            .is_err());
    }

    #[test]
    fn paths_are_node_side_lexical_and_safe() {
        for target in [
            "/",
            "/workspace",
            "/workspace/",
            "/workspace/./",
            "/workspace/../workspace",
        ] {
            assert!(workspace_bind_conflict(target, "/workspace"));
        }
        assert!(workspace_bind_conflict("/work", "/work/session"));
        assert!(!workspace_bind_conflict(
            "/workspace/external",
            "/workspace"
        ));
        assert!(!workspace_bind_conflict("/workspaces", "/workspace"));
        for path in [
            "relative",
            "",
            "/",
            "/tmp/../etc",
            "/sys",
            "/proc/self",
            "/dev",
            "/run",
            "/tmp/bad\npath",
            "/tmp/bad\u{7f}path",
        ] {
            assert!(
                WorkspaceConfig::from_value(json!({"image_root":path})).is_err(),
                "{path}"
            );
        }
        for (path, expected) in [
            ("//tmp//cape", "/tmp/cape"),
            ("/dev/shm/cape", "/dev/shm/cape"),
            (
                "/not-present-on-submitter/work space",
                "/not-present-on-submitter/work space",
            ),
        ] {
            assert_eq!(
                WorkspaceConfig::from_value(json!({"image_root":path}))
                    .unwrap()
                    .image_root,
                expected
            );
        }
    }
}
