// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use crate::telemetry_contract::CollectionConfig;

const METRICS: &[&str] = &["gpu_utilization_percent","memory_utilization_percent","memory_used_bytes",
    "memory_total_bytes","temperature_c","power_watts","sm_active_ratio","sm_occupancy_ratio","tensor_active_ratio","dram_active_ratio"];

pub struct QueryPlan {
    pub path: &'static str,
    pub params: BTreeMap<String, String>,
    pub mode: String,
    pub metric: String,
    pub gpus: BTreeSet<String>,
    pub scheduler_id: String,
    pub enabled: bool,
    pub service_id: Option<String>,
}

pub fn prepare_query(state: &Value, params: &BTreeMap<String,String>, now: f64) -> Result<QueryPlan> {
    ensure!(params.keys().all(|k|["metric","mode","node_id","pool","gpu_uuid","start","end","step"].contains(&k.as_str())), "unknown metric query parameter");
    let metric = params.get("metric").context("metric required")?;
    let mode = params.get("mode").map(String::as_str).unwrap_or("latest");
    ensure!(METRICS.contains(&metric.as_str()) && ["latest","history"].contains(&mode), "invalid metric or query mode");
    let initial = crate::telemetry_control::initial_state();
    let telemetry = state.get("telemetry").unwrap_or(&initial);
    let mut config = CollectionConfig::parse(telemetry["default"].clone())?;
    let pool = if let Some(pool) = params.get("pool") {
        Some(state["pools"].as_object().context("invalid pools")?.iter().find(|(id,p)| *id == pool || p["name"] == pool.as_str()).context("unknown pool")?.0.as_str())
    } else { None };
    let node_id = params.get("node_id");
    if let Some(node) = node_id { ensure!(state["nodes"].get(node).is_some(), "unknown node"); }
    let mut gpus = BTreeSet::new();
    if let Some(nodes) = state["nodes"].as_object() {
        for (id,node) in nodes {
            if node_id.map(|n| n != id).unwrap_or(false) || pool.map(|p| node["pool"] != p).unwrap_or(false) { continue; }
            let resources = node.get("effective_resources").unwrap_or(&node["resources"]);
            for gpu in resources["gpu_uuids"].as_array().context("invalid GPUs")? { gpus.insert(gpu.as_str().context("invalid GPU")?.to_owned()); }
        }
    }
    if let Some(gpu) = params.get("gpu_uuid") {
        ensure!(gpus.contains(gpu), "GPU not in requested resource scope");
        gpus.retain(|g| g == gpu);
    }
    ensure!(gpus.len() <= 1000, "metric query scope exceeds limit");
    let service = node_id.and_then(|n|telemetry["bindings"][n].as_str());
    if let Some(service) = service { config = CollectionConfig::parse(telemetry["overrides"].get(service).unwrap_or(&telemetry["default"]).clone())?; }
    let escaped: Vec<String> = gpus.iter().map(|gpu|gpu.chars().map(|c|if "\\.+*?()|[]{}^$-".contains(c) { format!("\\{c}") } else { c.to_string() }).collect()).collect();
    let name = format!("{}{metric}",if metric.starts_with("gpu_") {"cape_"} else {"cape_gpu_"});
    let mut selector = format!("{name}{{scheduler_id={},gpu_uuid=~{}", state["scheduler_id"], json!(escaped.join("|")));
    if let Some(service) = service { selector.push_str(&format!(",service_id={}",json!(service))); }
    selector.push('}');
    let freshness=format!("cape_gpu_sample_expires_at_seconds{}",&selector[selector.find('{').unwrap()..]);
    let mut options = BTreeMap::from([("query".into(), format!("({selector}) and on(scheduler_id,service_id,gpu_uuid) ({freshness} > time())")), ("timeout".into(), "5s".into())]);
    if mode == "history" {
        let start: f64 = params.get("start").context("history requires start/end/step")?.parse()?;
        let end: f64 = params.get("end").context("history requires start/end/step")?.parse()?;
        let step: f64 = params.get("step").context("history requires start/end/step")?.parse()?;
        ensure!([start,end,step].iter().all(|v|v.is_finite()) && start >= 0.0 && end >= start && end <= now+60.0 && end-start <= 31.0*86400.0 && step >= 1.0
            && (((end-start)/step).floor()+1.0) * gpus.len().max(1) as f64 <= 100000.0, "metric query range exceeds limit");
        options.insert("start".into(), start.to_string()); options.insert("end".into(), end.to_string()); options.insert("step".into(), step.to_string());
    } else { ensure!(!["start","end","step"].iter().any(|k|params.contains_key(*k)), "time range requires history mode"); }
    Ok(QueryPlan { path: if mode == "history" {"/api/v1/query_range"} else {"/api/v1/query"}, params: options,
        mode:mode.into(), metric:metric.clone(), gpus, scheduler_id:state["scheduler_id"].as_str().context("invalid scheduler")?.into(), enabled:config.enabled, service_id:service.map(str::to_owned) })
}

pub fn sanitize_result(plan: &QueryPlan, body: &Value) -> Result<Value> {
    let expected = if plan.mode == "history" {"matrix"} else {"vector"};
    ensure!(body["status"] == "success" && body["data"]["resultType"] == expected, "invalid metric backend response");
    let result = body["data"]["result"].as_array().context("invalid metric backend response")?;
    ensure!(result.len() <= 1000, "metric backend result limit exceeded");
    let mut series = Vec::new(); let mut count = 0;
    for item in result {
        let labels = &item["metric"];
        ensure!(labels.is_object() && (labels["service_id"].is_null() || labels["service_id"].is_string()), "invalid metric labels");
        let gpu = labels["gpu_uuid"].as_str().context("invalid GPU label")?;
        ensure!(plan.gpus.contains(gpu) && labels["scheduler_id"] == plan.scheduler_id && plan.service_id.as_ref().map(|s|labels["service_id"]==s.as_str()).unwrap_or(true), "metric backend scope mismatch");
        let points: Vec<&Value> = if expected == "matrix" {item["values"].as_array().context("invalid metric points")?.iter().collect()} else {vec![&item["value"]]};
        let mut clean = Vec::new();
        for point in points {
            count += 1;
            ensure!(count <= 100000 && point.as_array().map(|v|v.len()==2).unwrap_or(false), "invalid metric backend points");
            let timestamp = point[0].as_f64().context("invalid metric timestamp")?;
            let value: f64 = point[1].as_str().context("invalid metric value")?.parse()?;
            ensure!(timestamp.is_finite(), "invalid metric timestamp");
            clean.push(json!([timestamp, if value.is_finite() {Some(value)} else {None}]));
        }
        series.push(json!({"gpu_uuid":gpu,"service_id":labels["service_id"],"samples":clean}));
    }
    Ok(json!({"state":if series.is_empty() {"missing"} else {"ready"},"metric":plan.metric,"mode":plan.mode,"series":series}))
}

pub async fn fetch_query(plan: QueryPlan, base: Option<String>) -> Value {
    let disabled = !plan.enabled && plan.mode == "latest";
    let empty = json!({"state":if disabled {"disabled"} else {"unavailable"},"metric":plan.metric,"mode":plan.mode,"series":[]});
    if disabled || base.is_none() { return empty; }
    if plan.gpus.is_empty() { let mut missing = empty; missing["state"]=json!("missing"); return missing; }
    async fn fetch(plan: &QueryPlan, base: &str) -> Result<Value> {
        let client = reqwest::Client::builder().no_proxy().timeout(std::time::Duration::from_secs(7)).redirect(reqwest::redirect::Policy::none()).build()?;
        let mut response = client.get(format!("{}{}",base.trim_end_matches('/'),plan.path)).query(&plan.params).send().await?;
        ensure!(response.status().as_u16() == 200, "metric backend unavailable");
        let mut data = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            ensure!(data.len()+chunk.len() <= 8*1024*1024, "metric backend response too large");
            data.extend_from_slice(&chunk);
        }
        sanitize_result(plan, &serde_json::from_slice(&data)?)
    }
    fetch(&plan, &base.unwrap()).await.unwrap_or(empty)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scoped_query_rejects_raw_promql_and_foreign_samples() {
        let state = json!({"scheduler_id":"s","pools":{"p":{"name":"pool"}},
            "nodes":{"n":{"pool":"p","resources":{"gpu_uuids":["GPU-A"]}}}});
        let mut params = BTreeMap::from([("metric".into(),"sm_active_ratio".into()),("node_id".into(),"n".into())]);
        let plan = prepare_query(&state,&params,1000.0).unwrap();
        assert!(!plan.enabled && plan.gpus.contains("GPU-A"));
        assert!(plan.params["query"].contains("sample_expires_at_seconds"));
        let mut body = json!({"status":"success","data":{"resultType":"vector","result":[
            {"metric":{"scheduler_id":"s","gpu_uuid":"GPU-A","secret":"hidden"},"value":[100,"NaN"]}]}});
        let clean = sanitize_result(&plan,&body).unwrap();
        assert!(clean["series"][0]["samples"][0][1].is_null());
        assert!(!clean.to_string().contains("hidden"));
        body["data"]["result"][0]["metric"]["gpu_uuid"] = json!("GPU-other");
        assert!(sanitize_result(&plan,&body).is_err());
        params.insert("query".into(),"up".into());
        assert!(prepare_query(&state,&params,1000.0).is_err());
    }
    #[test]
    fn history_limits_and_no_client_backend_url() {
        let state = json!({"scheduler_id":"s","pools":{},"nodes":{}});
        let mut params = BTreeMap::from([("metric".into(),"sm_active_ratio".into()),("mode".into(),"history".into()),
            ("start".into(),"0".into()),("end".into(),"1000".into()),("step".into(),"1".into())]);
        assert!(prepare_query(&state,&params,1000.0).is_ok());
        params.insert("step".into(),"NaN".into());
        assert!(prepare_query(&state,&params,1000.0).is_err());
        params.insert("step".into(),"1".into());
        params.insert("url".into(),"http://other".into());
        assert!(prepare_query(&state,&params,1000.0).is_err());
    }
}
