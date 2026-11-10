// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::workspace_fs::MaintenancePolicy;
use anyhow::{ensure, Result};
use serde_json::{json, Value};

pub fn maintenance_host_key() -> Option<String> {
    let raw=std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
    let boot=uuid::Uuid::parse_str(raw.trim()).ok()?.to_string();
    let digest=ring::digest::digest(&ring::digest::SHA256,boot.as_bytes());
    Some(digest.as_ref().iter().map(|byte|format!("{byte:02x}")).collect())
}

fn counter(value: &Value) -> Option<u64> {
    value.as_u64().filter(|n| *n <= i64::MAX as u64)
}

pub fn sample(allocated: u64, geometry: &Value, image: u64, usage: u64) -> Result<Value> {
    let block = counter(&geometry["block_size"]).unwrap_or(0);
    let blocks = counter(&geometry["blocks"]).unwrap_or(0);
    let free = counter(&geometry["free_blocks"]).ok_or_else(|| anyhow::anyhow!("invalid reclaim geometry"))?;
    let logical = counter(&geometry["size_bytes"]).unwrap_or(0);
    ensure!(block > 0 && blocks > 0 && free <= blocks && block.checked_mul(blocks) == Some(logical),
        "inconsistent reclaim geometry");
    ensure!([allocated, image, usage].iter().all(|n| *n <= i64::MAX as u64), "invalid reclaim sample counter");
    Ok(json!({"allocated_bytes":allocated,"used_bytes":(blocks-free)*block,
        "logical_bytes":logical,"image_generation":image,"usage_generation":usage}))
}

pub fn signal(record: &Value) -> std::result::Result<u64, &'static str> {
    let reference = &record["reclaim_reference"];
    let sample = &record["reclaim_sample"];
    if reference.is_null() { return Err("reference_missing"); }
    for item in [reference, sample] {
        if !item.is_object() || ["allocated_bytes", "used_bytes", "logical_bytes", "image_generation", "usage_generation"]
            .iter().any(|key| counter(&item[*key]).is_none())
            || item["logical_bytes"] == 0 || counter(&item["used_bytes"]) > counter(&item["logical_bytes"]) {
            return Err("invalid_sample");
        }
    }
    let usage = counter(&record["usage_generation"]).filter(|n| *n > 0);
    let generation = record.get("generation").map(counter).unwrap_or(Some(0));
    let logical = record.get("current_image_size_bytes").map(counter)
        .unwrap_or_else(|| counter(&record["config"]["image_size_bytes"]));
    if usage.is_none() || generation.is_none() || counter(&sample["usage_generation"]) != usage
        || counter(&reference["usage_generation"]) > usage
        || counter(&reference["image_generation"]) != generation || counter(&sample["image_generation"]) != generation
        || counter(&reference["logical_bytes"]) != logical || counter(&sample["logical_bytes"]) != logical {
        return Err("stale_sample");
    }
    let allocated = counter(&sample["allocated_bytes"]).unwrap();
    let delta = (allocated as i128 - counter(&reference["allocated_bytes"]).unwrap() as i128)
        - (counter(&sample["used_bytes"]).unwrap() as i128 - counter(&reference["used_bytes"]).unwrap() as i128);
    Ok(delta.max(0).min(allocated as i128) as u64)
}

pub fn candidate(record: &Value, policy: &MaintenancePolicy, now: f64) -> (Option<Value>, &'static str) {
    if record.get("auto_reclaim").map(|v|v != "inherit").unwrap_or(false) { return (None, "workspace_disabled"); }
    if !policy.reclaim_on_release { return (None, "disabled"); }
    if record["state"] != "retained" || record["maintenance_deferred"] == true { return (None, "not_retained"); }
    if record["operation"]["state"] == "pending" { return (None, "operation_pending"); }
    let amount = match signal(record) { Ok(n) => n, Err(reason) => return (None, reason) };
    if record["reclaim_success_generation"] == record["usage_generation"] { return (None, "generation_completed"); }
    if amount == 0 || amount < policy.reclaim_min_allocated_bytes { return (None, "below_relative_threshold"); }
    let released = record["reclaim_release_observed_at"].as_f64().unwrap_or(now);
    let not_before = (released + policy.reclaim_idle_seconds as f64)
        .max(record["reclaim_last_success_at"].as_f64().unwrap_or(-1e30) + policy.reclaim_cooldown_seconds as f64)
        .max(record["reclaim_retry_after"].as_f64().unwrap_or(-1e30))
        .max(record["reclaim_low_yield_until"].as_f64().unwrap_or(-1e30));
    (Some(json!({"usage_generation":record["usage_generation"],"estimated_bytes":amount,
        "first_seen_at":released,"not_before":not_before})),
        if now >= not_before { "ready" } else { "waiting_idle_or_backoff" })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record() -> Value {
        let reference = json!({"allocated_bytes":260*1024*1024_u64,"used_bytes":260*1024*1024_u64,
            "logical_bytes":32*1024*1024*1024_u64,"image_generation":0,"usage_generation":0});
        let mut sample = reference.clone(); sample["usage_generation"] = json!(1);
        json!({"state":"retained","generation":0,"usage_generation":1,
            "config":{"image_size_bytes":32*1024*1024*1024_u64},"reclaim_reference":reference,
            "reclaim_sample":sample,"reclaim_release_observed_at":1000.0})
    }
    #[test]
    fn metadata_live_data_and_deleted_data_are_distinguished() {
        let mut value = record(); assert_eq!(signal(&value), Ok(0));
        for key in ["allocated_bytes","used_bytes"] { value["reclaim_sample"][key] = json!(value["reclaim_sample"][key].as_u64().unwrap()+1024*1024*1024); }
        assert_eq!(signal(&value), Ok(0));
        value["reclaim_sample"]["used_bytes"] = value["reclaim_reference"]["used_bytes"].clone();
        assert_eq!(signal(&value), Ok(1024*1024*1024));
        value["reclaim_reference"]["allocated_bytes"] = value["reclaim_sample"]["allocated_bytes"].clone();
        value["reclaim_reference"]["used_bytes"] = json!(value["reclaim_sample"]["used_bytes"].as_u64().unwrap()+1024*1024*1024);
        assert_eq!(signal(&value), Ok(1024*1024*1024));
    }
    #[test]
    fn missing_and_stale_samples_never_guess_a_baseline() {
        let mut value = record(); value["reclaim_reference"] = Value::Null;
        assert_eq!(signal(&value), Err("reference_missing"));
        let mut value = record(); value["usage_generation"] = json!(2);
        assert_eq!(signal(&value), Err("stale_sample"));
        value["reclaim_sample"]["allocated_bytes"] = json!(true);
        assert_eq!(signal(&value), Err("invalid_sample"));
        for (field, invalid) in [("generation",Value::Null),("current_image_size_bytes",Value::Null),("current_image_size_bytes",json!(true))] {
            let mut value=record();value[field]=invalid;
            assert_eq!(signal(&value),Err("stale_sample"));
        }
    }
    #[test]
    fn generation_dedup_and_delay_do_not_create_an_operation() {
        let mut value = record(); value["reclaim_sample"]["allocated_bytes"] = json!(2*1024*1024*1024_u64);
        let policy = MaintenancePolicy { reclaim_on_release:true, ..Default::default() };
        assert_eq!(candidate(&value,&policy,1001.0).1,"waiting_idle_or_backoff");
        value["reclaim_success_generation"] = json!(1);
        assert!(candidate(&value,&policy,2000.0).0.is_none());
        value["usage_generation"] = json!(2); value["reclaim_sample"]["usage_generation"] = json!(2);
        assert_eq!(candidate(&value,&policy,2000.0).1,"ready");
        assert!(value["operation"].is_null());
    }
}
