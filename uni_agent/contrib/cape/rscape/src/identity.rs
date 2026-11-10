// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native personal authority store. Its versioned SQLite snapshot is shared with
//! Python; identity-provider/browser state is deliberately outside this module.

use anyhow::{bail, ensure, Context, Result};
use ring::{
    digest,
    rand::{SecureRandom, SystemRandom},
};
use rusqlite::{Connection, OpenFlags, TransactionBehavior};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;
use uuid::Uuid;

pub fn new_secret() -> Result<String> {
    let mut bytes = [0_u8; 32];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("secure random unavailable"))?;
    Ok(format!("cap1_{}", hex(&bytes)))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn verifier(secret: &str) -> Result<String> {
    ensure!(
        secret
            .strip_prefix("cap1_")
            .map(valid_digest)
            .unwrap_or(false),
        "invalid credential"
    );
    Ok(hex(
        digest::digest(&digest::SHA256, secret.as_bytes()).as_ref()
    ))
}

pub(crate) fn name(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 64
            && value.as_bytes()[0].is_ascii_alphanumeric()
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
        "invalid name"
    );
    Ok(())
}

pub fn pool_configuration(value: &Value) -> Result<Value> {
    let object = value.as_object().context("invalid pool configuration")?;
    ensure!(
        object
            .keys()
            .all(|k| ["pool_mode", "gpu_slots", "placement"].contains(&k.as_str())),
        "invalid pool configuration"
    );
    let config: crate::model::PoolConfig = serde_json::from_value(value.clone())?;
    ensure!(
        matches!(config.pool_mode.as_str(), "exclusive" | "shared")
            && matches!(config.placement.as_str(), "pack" | "spread")
            && config.gpu_slots > 0
            && config.gpu_slots <= i64::MAX as usize
            && (config.pool_mode != "exclusive" || config.gpu_slots == 1),
        "invalid pool configuration"
    );
    Ok(serde_json::to_value(config)?)
}

fn id() -> String {
    Uuid::new_v4().to_string()
}

pub(crate) fn audit(state: &mut Value, actor: &str, action: &str, target: &str, now: f64) {
    if state.get("_pending_audit").is_none() {
        state["_pending_audit"] = json!([]);
    }
    state["_pending_audit"]
        .as_array_mut()
        .unwrap()
        .push(json!([id(), now, actor, action, target]));
}

fn validate_state(state: &Value) -> Result<()> {
    if let Some(telemetry) = state.get("telemetry") { crate::telemetry_control::validate_state(telemetry)?; }
    ensure!(
        (state["version"] == 2 || state["version"] == 3) && state.get("_pending_audit").is_none(),
        "invalid authority version"
    );
    Uuid::parse_str(
        state["scheduler_id"]
            .as_str()
            .context("invalid scheduler identity")?,
    )?;
    for key in ["users", "tokens", "groups", "pools", "operations"] {
        ensure!(state[key].is_object(), "invalid authority records");
    }
    ensure!(
        state["groups"].get("default").is_some(),
        "missing default group"
    );
    for user in state["users"].as_object().unwrap().values() {
        name(user["name"].as_str().context("invalid user name")?)?;
        ensure!(
            matches!(user["role"].as_str(), Some("admin" | "regular"))
                && user["disabled"].is_boolean(),
            "invalid user policy"
        );
        for group in user["groups"].as_array().context("invalid memberships")? {
            ensure!(
                group
                    .as_str()
                    .and_then(|g| state["groups"].get(g))
                    .is_some(),
                "unknown membership"
            );
        }
    }
    for token in state["tokens"].as_object().unwrap().values() {
        ensure!(
            token["user_id"]
                .as_str()
                .and_then(|u| state["users"].get(u))
                .is_some()
                && matches!(token["kind"].as_str(), Some("admin" | "regular"))
                && token["revoked"].is_boolean(),
            "invalid token policy"
        );
        ensure!(
            token["verifier"]
                .as_str()
                .map(valid_digest)
                .unwrap_or(false),
            "invalid verifier"
        );
        ensure!(
            token["expires_at"].is_null()
                || token["expires_at"]
                    .as_f64()
                    .map(f64::is_finite)
                    .unwrap_or(false),
            "invalid token expiry"
        );
    }
    for group in state["groups"].as_object().unwrap().values() {
        ensure!(
            group["all_pools"].is_boolean()
                && group["pools"]
                    .as_array()
                    .map(|p| p.iter().all(Value::is_string))
                    .unwrap_or(false),
            "invalid group policy"
        );
    }
    for pool in state["pools"].as_object().unwrap().values() {
        name(pool["name"].as_str().context("invalid pool name")?)?;
        ensure!(pool["draining"].is_boolean(), "invalid pool state");
        pool_configuration(pool.get("config").unwrap_or(&json!({})))?;
        ensure!(
            pool["max_nodes"].is_null()
                || pool["max_nodes"]
                    .as_u64()
                    .map(|n| n > 0 && n <= i64::MAX as u64)
                    .unwrap_or(false),
            "invalid node limit"
        );
    }
    if let Some(nodes) = state.get("nodes") {
        for node in nodes.as_object().context("invalid nodes")?.values() {
            if let Some(release)=node.get("quarantine_release") {
                ensure!(node.get("purge").is_some() && release.is_object()
                    && release["confirmed_stopped"]==true && release["confirmation_source"]=="administrator"
                    && release["reason"].as_str().map(|s|!s.trim().is_empty()&&s.len()<=1024).unwrap_or(false)
                    && release["actor_user_id"].as_str().and_then(|id|state["users"].get(id)).is_some()
                    && release["released_at"].as_f64().map(|t|t.is_finite()&&t>=0.).unwrap_or(false),"invalid quarantine release receipt");
            }
            if let Some(purge)=node.get("purge") {
                ensure!(purge.is_object() && purge["node_id"]==node["node_id"] && node["desired"]=="removed"
                    && purge["state"]=="REMOVED" && purge["purged"]==true && purge["cleanup_confirmed"]==false
                    && purge["reason"].as_str().map(|s|!s.trim().is_empty()&&s.len()<=1024).unwrap_or(false),"invalid node purge fence");
                let mut scoped=node["resources"].clone();scoped["gpu_uuids"]=purge["quarantined_gpu_uuids"].clone();
                crate::identity_nodes::resources(&scoped)?;
                let effective=node.get("effective_resources").unwrap_or(&node["resources"]);
                ensure!(effective["gpu_uuids"].as_array().context("invalid purge resources")?.iter().all(|g|
                    purge["quarantined_gpu_uuids"].as_array().map(|gs|gs.contains(g)).unwrap_or(false)),"invalid purge GPU fence");
            }
            ensure!(
                node["owner_user_id"]
                    .as_str()
                    .and_then(|u| state["users"].get(u))
                    .is_some()
                    && matches!(
                        node["desired"].as_str(),
                        Some("active" | "draining" | "removed" | "reconfigure")
                    ),
                "invalid node policy"
            );
            ensure!(
                node["announced"].is_boolean()
                    && node.get("force").map(Value::is_boolean).unwrap_or(true),
                "invalid node fence"
            );
            ensure!(
                node["intent_at"]
                    .as_f64()
                    .map(|n| n.is_finite() && n >= 0.)
                    .unwrap_or(false),
                "invalid node intent"
            );
            ensure!(
                node["verifier"].as_str().map(valid_digest).unwrap_or(false),
                "invalid node verifier"
            );
            crate::identity_nodes::resources(&node["resources"])?;
            if let Some(effective) = node.get("effective_resources") {
                crate::identity_nodes::resources(effective)?;
            }
            if node["desired"] == "reconfigure" {
                crate::identity_nodes::resources(&node["pending_resources"])?;
            }
        }
    }
    crate::management_sso::validate(state)?;
    Ok(())
}

pub(crate) fn authenticate(state: &Value, secret: &str, now: f64) -> Result<Value> {
    let hash = verifier(secret)?;
    for (token_id, token) in state["tokens"].as_object().context("invalid tokens")? {
        let stored = token["verifier"].as_str().context("invalid verifier")?;
        // Fixed-length digest comparison; verifier strings themselves are never credentials.
        if stored.len() != hash.len()
            || stored
                .bytes()
                .zip(hash.bytes())
                .fold(0_u8, |acc, (a, b)| acc | (a ^ b))
                != 0
        {
            continue;
        }
        let user_id = token["user_id"].as_str().context("invalid owner")?;
        let user = &state["users"][user_id];
        ensure!(
            token["revoked"] == false
                && user["disabled"] == false
                && token["expires_at"]
                    .as_f64()
                    .map(|expiry| now < expiry)
                    .unwrap_or(token["expires_at"].is_null()),
            "invalid credential"
        );
        let mut actor = user.clone();
        actor["user_id"] = json!(user_id);
        actor["token_id"] = json!(token_id);
        actor["token_kind"] = token["kind"].clone();
        actor["is_admin"] = json!(token["kind"] == "admin" && user["role"] == "admin");
        return Ok(actor);
    }
    crate::management_sso::authenticate_session(state, secret, now)
}

fn admin(state: &Value, secret: &str, now: f64) -> Result<Value> {
    let actor = authenticate(state, secret, now)?;
    ensure!(actor["is_admin"] == true, "admin credential required");
    Ok(actor)
}

pub struct IdentityStore {
    db: Connection,
}

impl IdentityStore {
    pub fn open(path: &Path) -> Result<Self> {
        let meta = fs::symlink_metadata(path)?;
        ensure!(
            meta.is_file()
                && meta.nlink() == 1
                && meta.uid() == unsafe { nix::libc::getuid() }
                && meta.mode() & 0o077 == 0,
            "authority must be an owner-only regular file"
        );
        let db = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        db.busy_timeout(std::time::Duration::from_secs(5))?;
        db.execute_batch("PRAGMA synchronous=FULL;")?;
        let store = Self { db };
        store.snapshot()?;
        store.db.prepare("SELECT event_id FROM audit LIMIT 0")?;
        Ok(store)
    }

    pub fn create(path: &Path, admin_name: &str, secret: &str) -> Result<Self> {
        name(admin_name)?;
        let hash = verifier(secret)?;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?;
        let user_id = id();
        let token_id = id();
        let mut state = json!({"version":2, "scheduler_id":id(), "users":{}, "tokens":{},
            "groups":{"default":{"all_pools":true,"pools":[]}},"pools":{},"operations":{}});
        state["users"][&user_id] =
            json!({"name":admin_name,"role":"admin","disabled":false,"groups":["default"]});
        state["tokens"][&token_id] = json!({"user_id":user_id,"name":"bootstrap","kind":"admin", "verifier":hash,"expires_at":null,"revoked":false});
        let db = Connection::open(path)?;
        db.execute_batch("PRAGMA synchronous=FULL; CREATE TABLE authority (singleton INTEGER PRIMARY KEY CHECK(singleton=1), body TEXT NOT NULL); CREATE TABLE audit (event_id TEXT PRIMARY KEY, timestamp REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL);")?;
        db.execute("INSERT INTO authority VALUES (1, ?)", [state.to_string()])?;
        db.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
            rusqlite::params![
                id(),
                crate::controller::now(),
                user_id,
                "controller.init",
                state["scheduler_id"].as_str().unwrap()
            ],
        )?;
        drop(db);
        File::open(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?
        .sync_all()?;
        Self::open(path)
    }

    pub fn snapshot(&self) -> Result<Value> {
        Self::read(&self.db)
    }

    pub fn pool_id(&self, value: &str) -> Result<String> {
        let state = self.snapshot()?;
        let pools = state["pools"].as_object().context("invalid pools")?;
        if pools.contains_key(value) {
            return Ok(value.to_owned());
        }
        pools
            .iter()
            .find(|(_, pool)| pool["name"] == value)
            .map(|(id, _)| id.clone())
            .context("unknown pool")
    }

    pub fn pending_allowed(
        &self,
        user_id: &str,
        pool_id: &str,
        token_id: Option<&str>,
    ) -> Result<bool> {
        let state = self.snapshot()?;
        let user = &state["users"][user_id];
        let pool = &state["pools"][pool_id];
        let token = state["tokens"].get(token_id.unwrap_or("")).unwrap_or(&state["web_auth"]["sessions"][token_id.unwrap_or("")]);
        if user["disabled"] != false || pool["draining"] != false || token["user_id"] != user_id {
            return Ok(false);
        }
        Ok(((token["kind"] == "admin" || token["kind"] == "session") && user["role"] == "admin")
            || user["groups"]
                .as_array()
                .context("invalid membership")?
                .iter()
                .any(|g| {
                    let group = &state["groups"][g.as_str().unwrap_or("")];
                    group["all_pools"] == true
                        || group["pools"]
                            .as_array()
                            .map(|p| p.contains(&json!(pool_id)))
                            .unwrap_or(false)
                }))
    }

    fn read(db: &Connection) -> Result<Value> {
        let body: String =
            db.query_row("SELECT body FROM authority WHERE singleton=1", [], |row| {
                row.get(0)
            })?;
        let state: Value = serde_json::from_str(&body)?;
        validate_state(&state)
            .map_err(|_| anyhow::anyhow!("unsupported or invalid authority state"))?;
        Ok(state)
    }

    pub(crate) fn mutate<T>(&mut self, action: impl FnOnce(&mut Value) -> Result<T>) -> Result<T> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut state = Self::read(&tx)?;
        let result = action(&mut state)?;
        if let Some(events) = state.as_object_mut().unwrap().remove("_pending_audit") {
            for event in events.as_array().context("invalid audit events")? {
                tx.execute(
                    "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
                    rusqlite::params![
                        event[0].as_str().unwrap(),
                        event[1].as_f64().unwrap(),
                        event[2].as_str().unwrap(),
                        event[3].as_str().unwrap(),
                        event[4].as_str().unwrap()
                    ],
                )?;
            }
        }
        tx.execute(
            "UPDATE authority SET body=? WHERE singleton=1",
            [state.to_string()],
        )?;
        tx.commit()?;
        Ok(result)
    }

    pub fn authenticate(&self, secret: &str, now: f64) -> Result<Value> {
        authenticate(&self.snapshot()?, secret, now)
    }
    pub fn require_admin(&self, secret: &str, now: f64) -> Result<Value> {
        admin(&self.snapshot()?, secret, now)
    }

    pub fn register_token(
        &mut self,
        operation: &str,
        hash: &str,
        token_name: &str,
        username: Option<&str>,
        bearer: Option<&str>,
        expiry: Option<f64>,
        now: f64,
    ) -> Result<Value> {
        self.register_token_for(
            operation, hash, token_name, username, bearer, expiry, now, None,
        )
    }

    pub fn register_token_for(
        &mut self,
        operation: &str,
        hash: &str,
        token_name: &str,
        username: Option<&str>,
        bearer: Option<&str>,
        expiry: Option<f64>,
        now: f64,
        target_user_id: Option<&str>,
    ) -> Result<Value> {
        ensure!(
            Uuid::parse_str(operation)?.to_string() == operation && valid_digest(hash),
            "invalid registration operation"
        );
        name(token_name)?;
        if let Some(username) = username {
            name(username)?;
        }
        ensure!(expiry.map(f64::is_finite).unwrap_or(true), "invalid expiry");
        let mut params =
            json!({"verifier":hash,"name":token_name,"username":username,"expires_at":expiry});
        if let Some(target) = target_user_id {
            ensure!(
                Uuid::parse_str(target)?.to_string() == target,
                "invalid target identity"
            );
            params["user_id"] = json!(target);
        }
        self.mutate(|state| {
            let actor = bearer.map(|secret| authenticate(state, secret, now)).transpose()?;
            if let Some(target) = target_user_id {
                ensure!(actor.as_ref().map(|a|a["user_id"]==target).unwrap_or(false), "token target identity mismatch");
            }
            if let Some(old) = state["operations"].get(operation) {
                let actor = actor.as_ref().context("registration recovery requires owner credential")?;
                ensure!(actor["user_id"] == old["result"]["user_id"], "registration recovery requires owner credential");
                ensure!(params == old["params"], "operation parameters differ");
                let result = old["result"].clone();
                audit(state, actor["user_id"].as_str().unwrap(), "token.recover", result["token_id"].as_str().unwrap(), now);
                return Ok(result);
            }
            ensure!(expiry.map(|e| e > now).unwrap_or(true), "expiry must be in the future");
            ensure!(!crate::management_sso::contains_verifier(state,hash), "SSO credential cannot be reused as personal credential");
            ensure!(!state["tokens"].as_object().context("invalid tokens")?.values().any(|t| t["verifier"] == hash), "credential already registered");
            ensure!(!state["nodes"].as_object().map(|nodes| nodes.values().any(|n| n["verifier"] == hash)).unwrap_or(false), "node credential cannot be reused as personal credential");
            let user_id = if let Some(actor) = actor {
                ensure!(username.map(|u| actor["name"] == u).unwrap_or(true), "cannot create another identity");
                actor["user_id"].as_str().context("invalid user")?.to_owned()
            } else {
                let username = username.context("registration unavailable")?;
                ensure!(!state["users"].as_object().context("invalid users")?.values().any(|u| u["name"] == username), "registration unavailable");
                let user_id = id();
                state["users"][&user_id] = json!({"name":username,"role":"regular","disabled":false,"groups":["default"]});
                user_id
            };
            let token_id = id();
            state["tokens"][&token_id] = json!({"user_id":user_id,"name":token_name,"kind":"regular","verifier":hash,"expires_at":expiry,"revoked":false});
            let result = json!({"operation_id":operation,"user_id":user_id,"token_id":token_id,"status":"registered"});
            state["operations"][operation] = json!({"params":params,"result":result});
            audit(state, &user_id, "token.create", &token_id, now);
            Ok(result)
        })
    }

    pub fn list_tokens(&self, bearer: &str, now: f64) -> Result<Vec<Value>> {
        let state = self.snapshot()?;
        let actor = authenticate(&state, bearer, now)?;
        let mut result = Vec::new();
        for (key, token) in state["tokens"].as_object().context("invalid tokens")? {
            if token["user_id"] != actor["user_id"]
                || (token["kind"] != "regular" && actor["is_admin"] != true)
            {
                continue;
            }
            let mut record = token.clone();
            record.as_object_mut().unwrap().remove("verifier");
            record["token_id"] = json!(key);
            result.push(record);
        }
        Ok(result)
    }

    pub fn revoke_token(&mut self, bearer: &str, token_id: &str, now: f64) -> Result<()> {
        self.mutate(|state| {
            let actor = authenticate(state, bearer, now)?;
            let token = state["tokens"]
                .get_mut(token_id)
                .context("credential not accessible")?;
            ensure!(
                token["user_id"] == actor["user_id"]
                    && (token["kind"] != "admin" || actor["is_admin"] == true),
                "credential not accessible"
            );
            ensure!(
                token["kind"] != "admin",
                "bootstrap admin replacement must precede revocation"
            );
            token["revoked"] = json!(true);
            audit(
                state,
                actor["user_id"].as_str().unwrap(),
                "token.revoke",
                token_id,
                now,
            );
            Ok(())
        })
    }

    pub fn set_group(
        &mut self,
        bearer: &str,
        group: &str,
        all: bool,
        pools: &[String],
        now: f64,
    ) -> Result<()> {
        name(group)?;
        self.mutate(|state| {
            let actor = admin(state, bearer, now)?;
            ensure!(
                pools.iter().all(|p| state["pools"].get(p).is_some()),
                "unknown pool ID"
            );
            let mut pools = pools.to_vec();
            pools.sort();
            pools.dedup();
            state["groups"][group] = json!({"all_pools":all,"pools":pools});
            audit(
                state,
                actor["user_id"].as_str().unwrap(),
                "group.set",
                group,
                now,
            );
            Ok(())
        })
    }

    pub fn delete_group(&mut self, bearer: &str, group: &str, now: f64) -> Result<()> {
        self.mutate(|state| {
            let actor = admin(state, bearer, now)?;
            ensure!(group != "default", "default group cannot be deleted");
            ensure!(
                state["groups"]
                    .as_object_mut()
                    .context("invalid groups")?
                    .remove(group)
                    .is_some(),
                "unknown group"
            );
            for user in state["users"]
                .as_object_mut()
                .context("invalid users")?
                .values_mut()
            {
                user["groups"]
                    .as_array_mut()
                    .context("invalid membership")?
                    .retain(|g| g != group);
            }
            audit(
                state,
                actor["user_id"].as_str().unwrap(),
                "group.delete",
                group,
                now,
            );
            Ok(())
        })
    }

    pub fn edit_user(
        &mut self,
        bearer: &str,
        user_id: &str,
        groups: Option<Vec<String>>,
        disabled: Option<bool>,
        now: f64,
    ) -> Result<()> {
        self.mutate(|state| {
            let actor = admin(state, bearer, now)?;
            if let Some(groups) = &groups {
                ensure!(
                    groups.iter().all(|g| state["groups"].get(g).is_some()),
                    "unknown group"
                );
            }
            let user = state["users"].get_mut(user_id).context("unknown user")?;
            if let Some(mut groups) = groups {
                groups.sort();
                groups.dedup();
                user["groups"] = json!(groups);
            }
            if let Some(disabled) = disabled {
                ensure!(
                    !disabled || user["role"] != "admin",
                    "cannot disable bootstrap admin"
                );
                user["disabled"] = json!(disabled);
            }
            audit(
                state,
                actor["user_id"].as_str().unwrap(),
                "user.edit",
                user_id,
                now,
            );
            Ok(())
        })
    }

    pub fn create_pool(&mut self, bearer: &str, pool_name: &str, now: f64) -> Result<String> {
        self.create_pool_config(bearer, pool_name, &json!({}), None, now)
    }

    pub fn create_pool_config(
        &mut self,
        bearer: &str,
        pool_name: &str,
        config: &Value,
        max_nodes: Option<u64>,
        now: f64,
    ) -> Result<String> {
        name(pool_name)?;
        let config = pool_configuration(config)?;
        ensure!(
            max_nodes
                .map(|n| n > 0 && n <= i64::MAX as u64)
                .unwrap_or(true),
            "invalid node limit"
        );
        self.mutate(|state| {
            let actor = admin(state, bearer, now)?;
            ensure!(
                !state["pools"]
                    .as_object()
                    .context("invalid pools")?
                    .values()
                    .any(|p| p["name"] == pool_name),
                "pool already exists"
            );
            let pool_id = id();
            state["pools"][&pool_id] =
                json!({"name":pool_name,"draining":false,"config":config,"max_nodes":max_nodes});
            audit(
                state,
                actor["user_id"].as_str().unwrap(),
                "pool.create",
                &pool_id,
                now,
            );
            Ok(pool_id)
        })
    }

    pub fn can_submit(&self, bearer: &str, pool_id: &str, now: f64) -> Result<bool> {
        let state = self.snapshot()?;
        let actor = authenticate(&state, bearer, now)?;
        let Some(pool) = state["pools"].get(pool_id) else {
            return Ok(false);
        };
        if pool["draining"] != false {
            return Ok(false);
        }
        Ok(actor["is_admin"] == true
            || actor["groups"]
                .as_array()
                .context("invalid membership")?
                .iter()
                .any(|group| {
                    let policy = &state["groups"][group.as_str().unwrap_or("")];
                    policy["all_pools"] == true
                        || policy["pools"]
                            .as_array()
                            .map(|p| p.contains(&json!(pool_id)))
                            .unwrap_or(false)
                }))
    }

    pub fn require_owner(&self, bearer: &str, owner: &str, now: f64) -> Result<Value> {
        let actor = self.authenticate(bearer, now)?;
        ensure!(
            actor["user_id"] == owner || actor["is_admin"] == true,
            "resource not accessible"
        );
        if actor["user_id"] != owner {
            // Reserve the writer before reading: a deferred read-to-write upgrade
            // can fail with SQLITE_BUSY_SNAPSHOT while the journal commits.
            let tx = rusqlite::Transaction::new_unchecked(&self.db, TransactionBehavior::Immediate)?;
            admin(&Self::read(&tx)?, bearer, now)?;
            tx.execute(
                "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
                rusqlite::params![
                    id(),
                    now,
                    actor["user_id"].as_str().unwrap(),
                    "admin.owner-access",
                    owner
                ],
            )?;
            tx.commit()?;
        }
        Ok(actor)
    }

    pub fn audit_events(&self, bearer: &str, limit: usize, now: f64) -> Result<Vec<Value>> {
        self.require_admin(bearer, now)?;
        ensure!(
            (1..=1000).contains(&limit),
            "audit limit must be between 1 and 1000"
        );
        let mut statement = self.db.prepare(
            "SELECT event_id,timestamp,actor,action,target FROM audit ORDER BY rowid DESC LIMIT ?",
        )?;
        let rows = statement.query_map([limit as i64], |row| Ok(json!({"event_id":row.get::<_,String>(0)?,"timestamp":row.get::<_,f64>(1)?,
            "actor":row.get::<_,String>(2)?,"action":row.get::<_,String>(3)?,"target":row.get::<_,String>(4)?})))?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "cross-language fixture; invoked by tests.test_identity with an explicit test binary"]
    fn cross_language_fixture() -> Result<()> {
        let path = std::env::var("CAPE_IDENTITY_FIXTURE")?;
        let input: Value = serde_json::from_slice(&fs::read(&path)?)?;
        if let Some(source) = input["credential_input"].as_str() {
            let envelope = crate::credential_file::read_registration(Path::new(source))?;
            ensure!(
                envelope["registration"]["username"] == "python-client",
                "unexpected client identity"
            );
        }
        if let Some(destination) = input["credential_output"].as_str() {
            crate::credential_file::prepare_registration(
                Path::new(destination),
                "https://controller.example",
                Some("rust-client"),
                "laptop",
                Some(200.),
            )?;
        }
        let database = Path::new(input["database"].as_str().context("database")?);
        let root = input["admin_secret"].as_str().context("admin secret")?;
        let mut store = if input["create"] == true {
            IdentityStore::create(database, "admin", root)?
        } else {
            IdentityStore::open(database)?
        };
        store.require_admin(root, 100.)?;
        let secret = input["regular_secret"].as_str().context("regular secret")?;
        let operation = input["operation_id"].as_str().context("operation")?;
        let result = store.register_token(
            operation,
            &verifier(secret)?,
            "laptop",
            Some("rust-user"),
            if input["retry"] == true {
                Some(secret)
            } else {
                None
            },
            Some(200.),
            100.,
        )?;
        assert_eq!(
            store.register_token(
                operation,
                &verifier(secret)?,
                "laptop",
                Some("rust-user"),
                Some(secret),
                Some(200.),
                100.
            )?,
            result
        );
        store.create_pool(root, "rust-pool", 100.)?;
        Ok(())
    }

    #[test]
    fn registration_groups_and_credentials() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("identity.sqlite3");
        let root = new_secret()?;
        let mut store = IdentityStore::create(&path, "admin", &root)?;
        let secret = new_secret()?;
        let operation = id();
        let hash = verifier(&secret)?;
        let result =
            store.register_token(&operation, &hash, "laptop", Some("alice"), None, None, 100.)?;
        assert_eq!(
            store.register_token(
                &operation,
                &hash,
                "laptop",
                Some("alice"),
                Some(&secret),
                None,
                100.
            )?,
            result
        );
        assert!(store
            .register_token(&operation, &hash, "laptop", Some("alice"), None, None, 100.)
            .is_err());
        assert!(store.authenticate(&hash, 100.).is_err());
        let pool = store.create_pool(&root, "pool", 100.)?;
        assert!(store.can_submit(&secret, &pool, 100.)?);
        store.set_group(&root, "default", false, &[], 100.)?;
        assert!(!store.can_submit(&secret, &pool, 100.)?);
        store.set_group(&root, "access", false, &[pool.clone()], 100.)?;
        let user = result["user_id"].as_str().unwrap();
        store.edit_user(&root, user, Some(vec!["access".into()]), None, 100.)?;
        assert!(store.can_submit(&secret, &pool, 100.)?);
        store.delete_group(&root, "access", 100.)?;
        assert!(!store.can_submit(&secret, &pool, 100.)?);
        assert!(store.delete_group(&root, "default", 100.).is_err());
        store.require_owner(&secret, user, 100.)?;
        assert!(store.require_admin(&secret, 100.).is_err());
        store.revoke_token(&secret, result["token_id"].as_str().unwrap(), 100.)?;
        assert!(store.authenticate(&secret, 100.).is_err());
        assert!(store
            .register_token(
                &operation,
                &hash,
                "laptop",
                Some("alice"),
                Some(&secret),
                None,
                100.
            )
            .is_err());
        assert!(IdentityStore::create(&path, "other", &root).is_err());
        assert_eq!(IdentityStore::open(&path)?.snapshot()?, store.snapshot()?);
        assert!(!String::from_utf8_lossy(&fs::read(&path)?).contains(&secret));
        Ok(())
    }

    #[test]
    fn admin_regular_isolation_and_atomic_rollback() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("identity.sqlite3");
        let root = new_secret()?;
        let mut store = IdentityStore::create(&path, "admin", &root)?;
        let regular = new_secret()?;
        store.register_token(
            &id(),
            &verifier(&regular)?,
            "regular",
            None,
            Some(&root),
            None,
            100.,
        )?;
        assert!(store.require_admin(&regular, 100.).is_err());
        assert_eq!(store.list_tokens(&regular, 100.)?.len(), 1);
        let root_id = store.require_admin(&root, 100.)?["token_id"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(store.revoke_token(&regular, &root_id, 100.).is_err());
        let before = store.snapshot()?;
        store.db.execute_batch("CREATE TRIGGER reject_update BEFORE UPDATE ON authority BEGIN SELECT RAISE(ABORT, 'injected'); END;")?;
        assert!(store
            .register_token(
                &id(),
                &verifier(&new_secret()?)?,
                "laptop",
                Some("bob"),
                None,
                None,
                100.
            )
            .is_err());
        assert_eq!(store.snapshot()?, before);
        Ok(())
    }

    #[test]
    fn administrative_owner_reads_audit_while_another_writer_commits() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("authority.sqlite3");
        let root = new_secret()?;
        let store = IdentityStore::create(&path, "admin", &root)?;
        store.db.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE concurrent_events (value INTEGER);")?;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let writer_barrier = barrier.clone();
        let writer = std::thread::spawn(move || -> Result<()> {
            let db = Connection::open(path)?;
            db.busy_timeout(std::time::Duration::from_secs(5))?;
            writer_barrier.wait();
            for value in 0..500 {
                db.execute("INSERT INTO concurrent_events VALUES (?)", [value])?;
                std::thread::yield_now();
            }
            Ok(())
        });
        barrier.wait();
        let checked = (|| -> Result<()> {
            for _ in 0..500 {
                store.require_owner(&root, "another-owner", 100.)?;
                std::thread::yield_now();
            }
            Ok(())
        })();
        writer.join().expect("writer panicked")?;
        checked?;
        let count: i64 = store.db.query_row("SELECT count(*) FROM audit WHERE action='admin.owner-access'", [], |row| row.get(0))?;
        assert_eq!(count, 500);
        Ok(())
    }
}
