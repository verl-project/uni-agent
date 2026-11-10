// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::agent::AgentConfig;
use crate::credential_file::{lock_registration, prepare_operation, read_registration};
use crate::identity_nodes::resources;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
#[error("enrollment result unknown; retain the credential file and resume")]
pub struct EnrollmentUncertain {
    pub path: std::path::PathBuf,
    pub operation_id: String,
}

pub async fn enroll(
    config: &mut AgentConfig,
    output: &Path,
    resume: bool,
) -> Result<std::fs::File> {
    let declared = resources(
        &json!({"cpu_cores":config.cpu_cores,"memory_gb":config.memory_gb,"gpu_uuids":config.gpu_uuids}),
    )?;
    let cpus = std::thread::available_parallelism()?.get() as f64;
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let mut memory = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|value| value.split_whitespace().next())
        .context("host memory unavailable")?
        .parse::<f64>()?
        / 1024_f64.powi(2);
    let mut candidates = std::collections::BTreeSet::from([
        std::path::PathBuf::from("/sys/fs/cgroup/memory.max"),
        std::path::PathBuf::from("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]);
    if let Ok(groups) = std::fs::read_to_string("/proc/self/cgroup") {
        for line in groups.lines() {
            let fields: Vec<_> = line.splitn(3, ':').collect();
            if fields.len() != 3 {
                continue;
            }
            let (base, filename) = if fields[1].is_empty() {
                (Path::new("/sys/fs/cgroup"), "memory.max")
            } else if fields[1].split(',').any(|c| c == "memory") {
                (Path::new("/sys/fs/cgroup/memory"), "memory.limit_in_bytes")
            } else {
                continue;
            };
            let mut path = base.join(fields[2].trim_start_matches('/'));
            while path.starts_with(base) {
                candidates.insert(path.join(filename));
                if path == base || !path.pop() {
                    break;
                }
            }
        }
    }
    for path in candidates {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(limit) = text.trim().parse::<u64>() {
                if limit > 0 {
                    memory = memory.min(limit as f64 / 1024_f64.powi(3));
                }
            }
        }
    }
    ensure!(
        config.cpu_cores <= cpus && config.memory_gb <= memory,
        "declared CPU/memory exceeds detected host capacity"
    );
    let gpus: Vec<String> = serde_json::from_value(declared["gpu_uuids"].clone())?;
    if !gpus.is_empty() {
        let output = tokio::time::timeout(
            Duration::from_secs(15),
            tokio::process::Command::new("nvidia-smi")
                .args([
                    "-i",
                    &gpus.join(","),
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ])
                .output(),
        )
        .await??;
        ensure!(output.status.success(), "GPU resource probe failed");
        let text = String::from_utf8(output.stdout)?;
        let mut actual: Vec<_> = text
            .lines()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .collect();
        actual.sort();
        ensure!(
            actual == gpus,
            "declared GPUs differ from detected physical devices"
        );
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;
    let url = config.controller_url.trim_end_matches('/');
    let owner = config
        .token
        .as_deref()
        .context("personal credential required for enrollment")?;
    let actor: Value = client
        .get(format!("{url}/v1/identity/me"))
        .bearer_auth(owner)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let pools: Value = client
        .get(format!("{url}/v1/identity/pools"))
        .bearer_auth(owner)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let selected_pool = pools
        .as_array()
        .context("invalid pool inventory")?
        .iter()
        .find(|p| p["pool_id"] == config.pool || p["name"] == config.pool)
        .context("unknown pool")?;
    let pool = selected_pool["pool_id"].as_str().context("invalid pool")?;
    let envelope = if resume {
        let saved = read_registration(output)?;
        let body = &saved["registration"];
        ensure!(
            body["owner_user_id"].is_string() && body["owner_user_id"] == actor["user_id"],
            "saved enrollment target identity mismatch or missing binding"
        );
        ensure!(
            saved["controller_url"] == url
                && body["resources"] == declared
                && body["pool"] == pool
                && body["node_id"] == config.node_id,
            "enrollment recovery differs from saved operation"
        );
        if let Some(allocation) = &config.allocation_id {
            ensure!(
                body["allocation_id"] == *allocation,
                "enrollment allocation differs from saved operation"
            );
        }
        saved
    } else {
        prepare_operation(
            output,
            url,
            json!({"node_id":config.node_id,"pool":pool,"agent_epoch":Uuid::new_v4().simple().to_string(),"owner_user_id":actor["user_id"],
            "allocation_id":config.allocation_id.clone().unwrap_or_else(|| format!("alloc-{}",Uuid::new_v4().simple())),"resources":declared}),
        )?
    };
    let body = &envelope["registration"];
    let guard = lock_registration(output)?;
    let response: Result<Value> = async {
        Ok(client
            .post(format!("{url}/v1/identity/nodes/enroll"))
            .bearer_auth(owner)
            .json(body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?)
    }
    .await;
    let accepted = response
        .as_ref()
        .map(|r| r["operation_id"] == body["operation_id"] && r["status"] == "enrolled")
        .unwrap_or(false);
    if !accepted {
        return Err(EnrollmentUncertain {
            path: output.to_path_buf(),
            operation_id: body["operation_id"].as_str().unwrap().to_owned(),
        }
        .into());
    }
    config.pool = pool.to_owned();
    let policy: crate::model::PoolConfig = serde_json::from_value(selected_pool["config"].clone())?;
    config.pool_mode = policy.pool_mode;
    config.gpu_slots = policy.gpu_slots;
    config.placement = policy.placement;
    config.allocation_id = Some(
        body["allocation_id"]
            .as_str()
            .context("invalid enrollment allocation")?
            .to_owned(),
    );
    config.agent_epoch = Some(
        body["agent_epoch"]
            .as_str()
            .context("invalid enrollment epoch")?
            .to_owned(),
    );
    config.token = Some(
        envelope["secret"]
            .as_str()
            .context("invalid enrollment credential")?
            .to_owned(),
    );
    config.join_token = None;
    Ok(guard)
}
