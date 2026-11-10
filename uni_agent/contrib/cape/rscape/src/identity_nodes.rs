// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::identity::{audit, authenticate, name, verifier, IdentityStore};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use uuid::Uuid;

pub fn resources(value: &Value) -> Result<Value> {
    let object = value.as_object().context("invalid resource declaration")?;
    ensure!(
        object.len() == 3
            && ["cpu_cores", "memory_gb", "gpu_uuids"]
                .iter()
                .all(|key| object.contains_key(*key)),
        "invalid resource declaration"
    );
    let mut result = json!({});
    for (key, limit) in [("cpu_cores", 100000.), ("memory_gb", 10000000.)] {
        let number = value[key].as_f64().context("invalid resource capacity")?;
        ensure!(
            number.is_finite() && number > 0. && number <= limit,
            "invalid resource capacity"
        );
        result[key] = json!(number);
    }
    let gpus = value["gpu_uuids"].as_array().context("invalid GPU UUIDs")?;
    ensure!(gpus.len() <= 64, "invalid GPU UUIDs");
    let mut canonical = BTreeSet::new();
    for gpu in gpus {
        let value = gpu
            .as_str()
            .and_then(|s| s.strip_prefix("GPU-"))
            .context("invalid GPU UUID")?;
        let value = format!("GPU-{}", Uuid::parse_str(value)?);
        ensure!(canonical.insert(value), "duplicate GPU UUID");
    }
    result["gpu_uuids"] = json!(canonical);
    Ok(result)
}

impl IdentityStore {
    pub fn enroll_node(&mut self, bearer: &str, body: &Value, now: f64) -> Result<Value> {
        let allowed = [
            "operation_id",
            "verifier",
            "node_id",
            "pool",
            "agent_epoch",
            "allocation_id",
            "resources",
        ];
        let object = body.as_object().context("invalid enrollment parameters")?;
        ensure!(
            allowed.iter().all(|key| object.contains_key(*key))
                && object
                    .keys()
                    .all(|k| allowed.contains(&k.as_str()) || k == "owner_user_id"),
            "invalid enrollment parameters"
        );
        let operation = body["operation_id"]
            .as_str()
            .context("invalid enrollment operation")?;
        ensure!(
            Uuid::parse_str(operation)?.to_string() == operation,
            "invalid enrollment operation"
        );
        let hash = body["verifier"]
            .as_str()
            .context("invalid enrollment verifier")?;
        ensure!(
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid enrollment verifier"
        );
        for key in ["node_id", "agent_epoch", "allocation_id"] {
            name(body[key].as_str().context("invalid node identity")?)?;
        }
        let pool = body["pool"].as_str().context("invalid pool")?;
        let node_id = body["node_id"].as_str().unwrap();
        let mut params = body.clone();
        params["resources"] = resources(&body["resources"])?;
        self.mutate(|state| {
            let actor = authenticate(state, bearer, now)?;
            if let Some(owner) = body.get("owner_user_id") {
                ensure!(owner == &actor["user_id"], "node enrollment target identity mismatch");
            }
            let key = format!("node:{operation}");
            if let Some(previous) = state["operations"].get(&key) {
                ensure!(previous["owner_user_id"] == actor["user_id"], "enrollment belongs to another identity");
                ensure!(previous["params"] == params, "operation parameters differ");
                return Ok(previous["result"].clone());
            }
            ensure!(state["pools"][pool]["draining"] == false, "pool does not allow enrollment");
            if state.get("nodes").is_none() { state["nodes"] = json!({}); }
            let nodes = state["nodes"].as_object().context("invalid nodes")?;
            if let Some(limit) = state["pools"][pool]["max_nodes"].as_u64() {
                ensure!((nodes.values().filter(|n|n["pool"] == pool && n["desired"] != "removed").count() as u64) < limit, "pool node limit reached");
            }
            ensure!(!nodes.contains_key(node_id), "node identity already enrolled; use the saved operation to recover");
            ensure!(!nodes.values().any(|n| n["verifier"] == hash || n["allocation_id"] == body["allocation_id"]), "node credential or allocation already registered");
            ensure!(!crate::management_sso::contains_verifier(state,hash), "SSO credential cannot be reused as node credential");
            ensure!(!state["tokens"].as_object().context("invalid tokens")?.values().any(|t| t["verifier"] == hash), "personal credential cannot be reused as node credential");
            let mut node = params.clone();
            node["owner_user_id"] = actor["user_id"].clone();
            node["desired"] = json!("active");
            node["announced"] = json!(false);
            node["effective_resources"] = params["resources"].clone();
            node["intent_at"] = json!(now);
            state["nodes"][node_id] = node;
            let result = json!({"node_id":node_id,"pool":pool,"allocation_id":body["allocation_id"],"agent_epoch":body["agent_epoch"],"operation_id":operation,"status":"enrolled"});
            state["operations"][&key] = json!({"owner_user_id":actor["user_id"],"params":params,"result":result});
            audit(state, actor["user_id"].as_str().unwrap(), "node.enroll", node_id, now);
            Ok(result)
        })
    }

    pub fn authenticate_agent(
        &self,
        secret: &str,
        pool: Option<&str>,
        node_id: Option<&str>,
        epoch: Option<&str>,
    ) -> Result<Value> {
        let hash = verifier(secret)?;
        let state = self.snapshot()?;
        if let Some(nodes) = state["nodes"].as_object() {
            for (id, node) in nodes {
                let digest = node["verifier"].as_str().context("invalid node verifier")?;
                if digest.len() != hash.len()
                    || digest
                        .bytes()
                        .zip(hash.bytes())
                        .fold(0_u8, |a, (b, c)| a | (b ^ c))
                        != 0
                {
                    continue;
                }
                ensure!(
                    node.get("purge").is_none() && pool.map(|v| node["pool"] == v).unwrap_or(true)
                        && node_id.map(|v| id == v).unwrap_or(true)
                        && epoch.map(|v| node["agent_epoch"] == v).unwrap_or(true),
                    "invalid node credential"
                );
                return Ok(node.clone());
            }
        }
        anyhow::bail!("invalid node credential")
    }

    pub fn validate_registration(&mut self, secret: &str, body: &Value) -> Result<Value> {
        let epoch = body["agent_epoch"]
            .as_str()
            .filter(|e| !e.is_empty())
            .context("node epoch required")?;
        let pool = body["pool"].as_str().context("node pool required")?;
        let id = body["node_id"].as_str().context("node ID required")?;
        let node = self.authenticate_agent(secret, Some(pool), Some(id), Some(epoch))?;
        ensure!(
            node["allocation_id"] == body["allocation_id"]
                && node["resources"] == resources(&body["resources"])?,
            "node registration differs from enrolled scope"
        );
        ensure!(
            node["desired"] != "removed" || node["announced"] == true,
            "node enrollment is retired"
        );
        if node["announced"] != true {
            self.mutate(|state| {
                state["nodes"][id]["announced"] = json!(true);
                audit(
                    state,
                    &format!("node:{id}"),
                    "node.registration-started",
                    id,
                    crate::controller::now(),
                );
                Ok(())
            })?;
        }
        Ok(node)
    }

    pub fn owned_node(&self, bearer: &str, node_id: &str, now: f64) -> Result<Value> {
        let actor = self.authenticate(bearer, now)?;
        let state = self.snapshot()?;
        let node = state["nodes"].get(node_id).context("node not accessible")?;
        ensure!(
            actor["is_admin"] == true || actor["user_id"] == node["owner_user_id"],
            "node not accessible"
        );
        Ok(node.clone())
    }
}
