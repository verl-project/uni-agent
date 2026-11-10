// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::model::Role;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use crate::identity::IdentityStore;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UserTokenRecord {
    pub token: String,
    pub user_id: String,
    pub pool: String,
    #[serde(default)]
    pub roles: BTreeSet<Role>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JoinTokenRecord {
    pub token: String,
    pub pool: String,
    #[serde(default)]
    pub allocation_id: Option<String>,
    #[serde(default)]
    pub max_nodes: Option<usize>,
    #[serde(default)]
    pub allowed_node_prefixes: Vec<String>,
    #[serde(default)]
    pub allowed_runtime_configs: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Authority {
    pub enabled: bool,
    pub identity_path: Option<PathBuf>,
    users: BTreeMap<String, UserTokenRecord>,
    joins: BTreeMap<String, JoinTokenRecord>,
    used_join_nodes: BTreeMap<String, BTreeSet<String>>,
}

impl Authority {
    pub fn from_identity(path: PathBuf) -> Self {
        Self { enabled: true, identity_path: Some(path), ..Self::default() }
    }

    pub fn identity(&self) -> Result<Option<IdentityStore>> {
        self.identity_path.as_deref().map(IdentityStore::open).transpose()
    }

    pub fn authorize_owner(&self, token: Option<&str>, pool: &str, owner: &str) -> Result<String> {
        if let Some(store) = self.identity()? {
            let actor = store.require_owner(token.context("missing bearer token")?, owner, crate::controller::now())?;
            Ok(actor["user_id"].as_str().context("invalid identity")?.to_owned())
        } else { self.authorize(token, pool, Role::Submitter, Some(owner)) }
    }
    pub fn authorize(
        &self,
        token: Option<&str>,
        pool: &str,
        role: Role,
        owner: Option<&str>,
    ) -> Result<String> {
        if let Some(store) = self.identity()? {
            let token = token.context("missing bearer token")?;
            if role == Role::NodeAgent {
                let node = store.authenticate_agent(token, Some(pool), owner, None)?;
                return Ok(node["node_id"].as_str().context("invalid node identity")?.to_owned());
            }
            let actor = store.authenticate(token, crate::controller::now())?;
            let allowed = match role {
                Role::Submitter => store.can_submit(token, &store.pool_id(pool)?, crate::controller::now())?,
                Role::NodeAgent => false,
                _ => actor["is_admin"] == true,
            };
            if !allowed { bail!("missing required permission"); }
            if let Some(owner) = owner {
                if actor["user_id"] != owner { bail!("cannot operate on another identity"); }
            }
            return Ok(actor["user_id"].as_str().context("invalid identity")?.to_owned());
        }
        if !self.enabled {
            return Ok(owner.unwrap_or("anonymous").to_string());
        }
        let token = token.ok_or_else(|| anyhow::anyhow!("missing bearer token"))?;
        let record = self
            .users
            .get(token)
            .ok_or_else(|| anyhow::anyhow!("invalid user token"))?;
        if record.pool != pool {
            bail!("token pool mismatch");
        }
        if !record.roles.contains(&role) && !record.roles.contains(&Role::Admin) {
            bail!("missing required role");
        }
        if let Some(owner) = owner {
            if role == Role::Submitter
                && record.user_id != owner
                && !record.roles.contains(&Role::Admin)
            {
                bail!("cannot operate on another user's request");
            }
        }
        Ok(record.user_id.clone())
    }

    pub fn validate_join(
        &mut self,
        token: Option<&str>,
        pool: &str,
        node_id: &str,
        runtime_config: &str,
    ) -> Result<()> {
        if self.identity_path.is_some() { bail!("persistent instances require node enrollment credentials"); }
        if !self.enabled {
            return Ok(());
        }
        let token = token.ok_or_else(|| anyhow::anyhow!("missing join token"))?;
        let record = self
            .joins
            .get(token)
            .ok_or_else(|| anyhow::anyhow!("invalid join token"))?;
        if record.pool != pool {
            bail!("join token pool mismatch");
        }
        if !record.allowed_node_prefixes.is_empty()
            && !record
                .allowed_node_prefixes
                .iter()
                .any(|prefix| node_id.starts_with(prefix))
        {
            bail!("node id not allowed by join token");
        }
        if !record.allowed_runtime_configs.is_empty()
            && !record
                .allowed_runtime_configs
                .iter()
                .any(|value| value == runtime_config)
        {
            bail!("runtime cache config not allowed by join token");
        }
        let used = self.used_join_nodes.entry(token.to_string()).or_default();
        if !used.contains(node_id)
            && record
                .max_nodes
                .map(|limit| used.len() >= limit)
                .unwrap_or(false)
        {
            bail!("join token node limit reached");
        }
        used.insert(node_id.to_string());
        Ok(())
    }
}
