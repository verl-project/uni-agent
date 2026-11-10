// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure telemetry contracts. Runtime integration must not initialize GPU collectors by default.

use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CollectionConfig {
    pub enabled: bool,
    pub basic: bool,
    pub profiling: bool,
    pub interval_seconds: f64,
    pub stale_periods: u32,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self { enabled: false, basic: true, profiling: false, interval_seconds: 5.0, stale_periods: 3 }
    }
}

impl CollectionConfig {
    pub fn parse(value: serde_json::Value) -> Result<Self> {
        ensure!(value.is_object(), "telemetry configuration must be an object");
        let config: Self = serde_json::from_value(value)?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.interval_seconds.is_finite() && (1.0..=3600.0).contains(&self.interval_seconds), "invalid telemetry interval");
        ensure!((1..=120).contains(&self.stale_periods), "invalid telemetry stale periods");
        Ok(())
    }
}

fn identifier(value: &str) -> Result<()> {
    ensure!(!value.is_empty() && value.len() <= 256 && !value.chars().any(|c| c < '!' || c == '\u{7f}'), "invalid telemetry identifier");
    Ok(())
}

struct Registration {
    epoch: String,
    pool: String,
    gpus: BTreeSet<String>,
    deadline: f64,
}

pub struct RegistrationSet {
    scheduler_id: String,
    pub host_id: String,
    ttl: f64,
    capacity: usize,
    entries: BTreeMap<String, Registration>,
}

impl RegistrationSet {
    pub fn new(scheduler_id: &str, host_id: &str, ttl: f64, capacity: usize) -> Result<Self> {
        identifier(scheduler_id)?;
        identifier(host_id)?;
        ensure!(ttl.is_finite() && (1.0..=3600.0).contains(&ttl), "invalid registration ttl");
        ensure!((1..=65536).contains(&capacity), "invalid registration capacity");
        Ok(Self { scheduler_id: scheduler_id.into(), host_id: host_id.into(), ttl, capacity, entries: BTreeMap::new() })
    }

    pub fn expire(&mut self, now: f64) -> Result<usize> {
        ensure!(now.is_finite() && (0.0..=1e20).contains(&now), "invalid monotonic time");
        let before = self.entries.len();
        self.entries.retain(|_, entry| entry.deadline > now);
        Ok(before - self.entries.len())
    }

    pub fn register(&mut self, scheduler: &str, node: &str, epoch: &str, pool: &str, gpus: &[String], now: f64) -> Result<()> {
        ensure!(scheduler == self.scheduler_id, "telemetry scheduler mismatch");
        for value in [node, epoch, pool] { identifier(value)?; }
        ensure!(gpus.len() <= 1024, "invalid telemetry GPU list");
        for gpu in gpus { identifier(gpu)?; }
        self.expire(now)?;
        if let Some(previous) = self.entries.get(node) {
            ensure!(previous.epoch == epoch, "telemetry registration incarnation conflict");
        } else if self.entries.len() >= self.capacity {
            bail!("telemetry registration limit reached");
        }
        self.entries.insert(node.into(), Registration {
            epoch: epoch.into(), pool: pool.into(), gpus: gpus.iter().cloned().collect(), deadline: now + self.ttl,
        });
        Ok(())
    }

    pub fn remove(&mut self, node: &str, epoch: &str, now: f64) -> Result<bool> {
        self.expire(now)?;
        if self.entries.get(node).map(|entry| entry.epoch.as_str()) != Some(epoch) { return Ok(false); }
        self.entries.remove(node);
        Ok(true)
    }

    pub fn gpu_uuids(&mut self, pool: Option<&str>, node: Option<&str>, now: f64) -> Result<Vec<String>> {
        self.expire(now)?;
        Ok(self.entries.iter().filter(|(key, entry)| {
            node.map(|n| n == key.as_str()).unwrap_or(true) && pool.map(|p| p == entry.pool).unwrap_or(true)
        }).flat_map(|(_, entry)| entry.gpus.iter().cloned()).collect::<BTreeSet<_>>().into_iter().collect())
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Snapshot {
    pub gpu_uuid: String,
    pub state: String,
    pub sampled_at: Option<f64>,
    pub values: BTreeMap<String, Option<f64>>,
}

pub struct LatestSnapshots {
    capacity: usize,
    values: BTreeMap<String, (f64, f64, BTreeMap<String, Option<f64>>)>,
}

impl LatestSnapshots {
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!((1..=65536).contains(&capacity), "invalid snapshot capacity");
        Ok(Self { capacity, values: BTreeMap::new() })
    }

    pub fn retain(&mut self, allowed: &BTreeSet<String>) {
        self.values.retain(|key, _| allowed.contains(key));
    }

    pub fn put(&mut self, gpu: &str, sampled_at: f64, values: BTreeMap<String, Option<f64>>, now: f64) -> Result<bool> {
        identifier(gpu)?;
        ensure!([sampled_at, now].iter().all(|v| v.is_finite() && (0.0..=1e20).contains(v)), "invalid sample time");
        ensure!(values.len() <= 64, "invalid telemetry values");
        for (name, value) in &values {
            identifier(name)?;
            ensure!(value.map(|v| v.is_finite() && (-1e20..=1e20).contains(&v)).unwrap_or(true), "invalid telemetry value");
        }
        if let Some((previous, _, _)) = self.values.get(gpu) {
            if sampled_at <= *previous { return Ok(false); }
        } else {
            ensure!(self.values.len() < self.capacity, "telemetry snapshot limit reached");
        }
        self.values.insert(gpu.into(), (sampled_at, now, values));
        Ok(true)
    }

    pub fn get(&self, gpu: &str, stale_after: f64, now: f64) -> Result<Snapshot> {
        ensure!(stale_after.is_finite() && (0.001..=432000.0).contains(&stale_after), "invalid stale duration");
        ensure!(now.is_finite() && (0.0..=1e20).contains(&now), "invalid monotonic time");
        let mut result = Snapshot { gpu_uuid: gpu.into(), state: "unavailable".into(), sampled_at: None, values: BTreeMap::new() };
        if let Some((sampled_at, received_at, values)) = self.values.get(gpu) {
            result.sampled_at = Some(*sampled_at);
            let age = now - received_at;
            if age < 0.0 || age >= stale_after { result.state = "stale".into(); }
            else { result.state = "ready".into(); result.values = values.clone(); }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strict_defaults() {
        let config = CollectionConfig::default();
        assert!(!config.enabled && config.basic && !config.profiling);
        assert_eq!(config.interval_seconds, 5.0);
        config.validate().unwrap();
        for value in [json!({"enabled":1}), json!({"token":"secret"}), json!({"stale_periods":3.0}), json!([])] {
            assert!(CollectionConfig::parse(value.clone()).is_err(), "accepted {value}");
        }
        let bad: CollectionConfig = serde_json::from_value(json!({"interval_seconds":0})).unwrap();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn union_scopes_expiry_and_old_exit() {
        let mut r = RegistrationSet::new("s", "h", 30.0, 2).unwrap();
        assert_eq!(r.host_id, "h");
        r.register("s", "a", "old", "shared", &["GPU-A".into()], 10.0).unwrap();
        r.register("s", "b", "b", "exclusive", &["GPU-A".into(), "MIG-B".into()], 20.0).unwrap();
        assert_eq!(r.gpu_uuids(None, None, 20.0).unwrap(), vec!["GPU-A", "MIG-B"]);
        assert_eq!(r.gpu_uuids(Some("shared"), None, 20.0).unwrap(), vec!["GPU-A"]);
        assert!(r.register("other", "c", "c", "p", &[], 20.0).is_err());
        assert!(r.register("s", "a", "new", "p", &[], 20.0).is_err());
        assert!(r.register("s", "c", "c", "p", &[], 20.0).is_err());
        assert_eq!(r.expire(40.0).unwrap(), 1);
        r.register("s", "a", "new", "shared", &["GPU-A".into()], 40.0).unwrap();
        assert!(!r.remove("a", "old", 40.0).unwrap());
        assert_eq!(r.gpu_uuids(None, Some("a"), 40.0).unwrap(), vec!["GPU-A"]);
        assert_eq!(r.gpu_uuids(None, None, 70.0).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn latest_only_and_replays_do_not_extend_freshness() {
        let mut cache = LatestSnapshots::new(1).unwrap();
        assert_eq!(cache.get("GPU-A", 15.0, 10.0).unwrap().state, "unavailable");
        let values = BTreeMap::from([("sm_active".into(), Some(0.0)), ("tensor_active".into(), None)]);
        assert!(cache.put("GPU-A", 1000.0, values.clone(), 10.0).unwrap());
        assert!(!cache.put("GPU-A", 1000.0, values.clone(), 24.0).unwrap());
        assert_eq!(cache.get("GPU-A", 15.0, 24.0).unwrap().values["sm_active"], Some(0.0));
        assert_eq!(cache.get("GPU-A", 15.0, 25.0).unwrap().state, "stale");
        assert!(cache.get("GPU-A", 15.0, 25.0).unwrap().values.is_empty());
        assert!(cache.put("GPU-B", 1001.0, values.clone(), 25.0).is_err());
        assert!(cache.put("GPU-A", 1001.0, BTreeMap::from([("bad".into(), Some(f64::NAN))]), 25.0).is_err());
        cache.retain(&BTreeSet::new());
        assert_eq!(cache.get("GPU-A", 15.0, 25.0).unwrap().state, "unavailable");
    }
}
