// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use crate::identity::{IdentityStore, authenticate, audit, verifier};
use crate::telemetry_contract::CollectionConfig;

const MAX_VERSION: u64 = 9007199254740991;

fn identifier(value: &str) -> Result<()> {
    ensure!(!value.is_empty() && value.len() <= 256 && !value.chars().any(|c| c < '!' || c == '\u{7f}'), "invalid telemetry identifier");
    Ok(())
}

pub fn initial_state() -> Value {
    json!({"version":0,"default":CollectionConfig::default(),"overrides":{},"bindings":{},"receipts":{}})
}

pub fn validate_state(value: &Value) -> Result<()> {
    let object = value.as_object().context("invalid telemetry state")?;
    ensure!(["version","default","overrides","bindings","receipts"].iter().all(|k| object.contains_key(*k))
        && object.keys().all(|k|["version","default","overrides","bindings","receipts","locations"].contains(&k.as_str())), "invalid telemetry state");
    let version = value["version"].as_u64().filter(|v| *v <= MAX_VERSION).context("invalid telemetry version")?;
    CollectionConfig::parse(value["default"].clone())?;
    for field in ["overrides", "bindings", "receipts"] {
        ensure!(value[field].as_object().map(|v| v.len() <= 65536).unwrap_or(false), "invalid telemetry records");
    }
    for (service, config) in value["overrides"].as_object().unwrap() {
        identifier(service)?;
        CollectionConfig::parse(config.clone())?;
    }
    for (node, service) in value["bindings"].as_object().unwrap() {
        identifier(node)?;
        identifier(service.as_str().context("invalid telemetry service")?)?;
    }
    for (node, receipt) in value["receipts"].as_object().unwrap() {
        identifier(node)?;
        ensure!(receipt.as_object().map(|r| r.len() == 3 && ["version","state","service_id"].iter().all(|k| r.contains_key(*k))).unwrap_or(false)
            && receipt["version"].as_u64().map(|v| v <= version).unwrap_or(false)
            && matches!(receipt["state"].as_str(), Some("pending" | "applied" | "error"))
            && receipt["service_id"].is_string() && receipt["service_id"] == value["bindings"][node], "invalid telemetry receipt");
    }
    if let Some(locations)=value.get("locations") {
        let locations=locations.as_object().context("invalid telemetry locations")?;
        ensure!(locations.len()<=65536,"too many telemetry locations");
        for (key,location) in locations {
            identifier(key)?;
            ensure!(location.as_object().map(|v|v.len()==3).unwrap_or(false),"invalid telemetry location");
            location_url(location["url"].as_str().context("invalid service URL")?)?;
            identifier(location["incarnation"].as_str().context("invalid service incarnation")?)?;
            ensure!(location["expires_at"].as_f64().map(|v|v.is_finite() && (0.0..=1e20).contains(&v)).unwrap_or(false),"invalid location expiry");
        }
    }
    Ok(())
}

fn location_url(value:&str)->Result<()> {
    ensure!(value.len()<=4096 && !value.chars().any(|c|c<'!'),"invalid service URL");
    let url=url::Url::parse(value)?;
    ensure!(url.scheme()=="http" && url.host_str().is_some() && url.username().is_empty() && url.password().is_none()
        && url.query().is_none() && url.fragment().is_none() && url.path()=="/" && !value.split("://").nth(1).unwrap_or("").split('/').next().unwrap_or("").contains('@'),"invalid service URL");
    Ok(())
}

pub fn public_config(store:&IdentityStore,service:&str,now:f64)->Result<Value> {
    identifier(service)?;
    let state=store.snapshot()?;let initial=initial_state();let telemetry=state.get("telemetry").unwrap_or(&initial);
    ensure!(telemetry["bindings"].as_object().unwrap().values().any(|v|v==service),"unknown telemetry service");
    let config=telemetry["overrides"].get(service).unwrap_or(&telemetry["default"]);
    let location=&telemetry["locations"][service];
    Ok(json!({"scheduler_id":state["scheduler_id"],"service_id":service,"version":telemetry["version"],"config":config,
        "location":location,"lease_remaining_seconds":location["expires_at"].as_f64().map(|t|(t-now).max(0.0)).unwrap_or(0.0)}))
}

fn view(state: &Value, node: Option<&str>) -> Result<Value> {
    let initial = initial_state();
    let telemetry = state.get("telemetry").unwrap_or(&initial);
    validate_state(telemetry)?;
    let service = if let Some(node) = node {
        identifier(node)?;
        ensure!(state["nodes"].get(node).is_some(), "unknown node");
        Some(telemetry["bindings"][node].as_str().context("node has not registered a metrics service")?)
    } else { None };
    let config = service.and_then(|s| telemetry["overrides"].get(s)).unwrap_or(&telemetry["default"]);
    let receipts: serde_json::Map<String, Value> = telemetry["receipts"].as_object().unwrap().iter()
        .filter(|(_, r)| service.map(|s| r["service_id"] == s).unwrap_or(true)).map(|(k,v)| (k.clone(),v.clone())).collect();
    Ok(json!({"scheduler_id":state["scheduler_id"],"service_id":service,"version":telemetry["version"],"config":config,"receipts":receipts}))
}

pub fn get_config(store: &IdentityStore, token: &str, node: Option<&str>, admin: bool, now: f64) -> Result<Value> {
    if admin { store.require_admin(token, now)?; } else { store.authenticate(token, now)?; }
    view(&store.snapshot()?, node)
}

pub fn set_config(store: &mut IdentityStore, token: &str, changes: &Value, node: Option<&str>, expected: Option<u64>, now: f64) -> Result<Value> {
    store.require_admin(token, now)?;
    let changes = changes.as_object().filter(|v| !v.is_empty()).context("telemetry changes must be a nonempty object")?;
    ensure!(expected.map(|v| v <= MAX_VERSION).unwrap_or(true), "invalid expected telemetry version");
    store.mutate(|state| {
        let actor = authenticate(state, token, now)?;
        ensure!(actor["is_admin"] == true, "admin credential required");
        let current = view(state, node)?;
        let version = current["version"].as_u64().unwrap();
        ensure!(expected.map(|v| v == version).unwrap_or(true), "telemetry configuration version conflict");
        let mut config = current["config"].clone();
        config.as_object_mut().unwrap().extend(changes.clone());
        let config = serde_json::to_value(CollectionConfig::parse(config)?)?;
        if config != current["config"] {
            if state.get("telemetry").is_none() { state["telemetry"] = initial_state(); }
            ensure!(version < MAX_VERSION, "telemetry version exhausted");
            state["telemetry"]["version"] = json!(version+1);
            if let Some(service) = current["service_id"].as_str() { state["telemetry"]["overrides"][service] = config; }
            else { state["telemetry"]["default"] = config; }
            let target = current["service_id"].as_str().unwrap_or(state["scheduler_id"].as_str().unwrap()).to_owned();
            audit(state, actor["user_id"].as_str().unwrap(), "metric.configure", &target, now);
        }
        view(state, node)
    })
}

pub fn poll_control(store: &mut IdentityStore, token: &str, body: &Value) -> Result<Value> {
    let object = body.as_object().context("invalid telemetry control request")?;
    if object.len()==3 && object.contains_key("node_id") && object.contains_key("agent_epoch") && body["discover"]==true {
        let node=store.authenticate_agent(token,None,Some(body["node_id"].as_str().context("invalid node ID")?),Some(body["agent_epoch"].as_str().context("invalid epoch")?))?;
        ensure!(node["desired"]!="removed","invalid node credential");
        return Ok(json!({"scheduler_id":store.snapshot()?["scheduler_id"]}));
    }
    ensure!(["node_id","agent_epoch","service_id"].iter().all(|k|object.contains_key(*k))
        && object.keys().all(|k| ["node_id","agent_epoch","service_id","applied_version","state","service_url","incarnation"].contains(&k.as_str())), "invalid telemetry control request");
    let node_id = body["node_id"].as_str().context("invalid node ID")?;
    let epoch = body["agent_epoch"].as_str().context("invalid agent epoch")?;
    let service = body["service_id"].as_str().context("invalid service ID")?;
    for value in [node_id, epoch, service] { identifier(value)?; }
    let node = store.authenticate_agent(token, None, Some(node_id), Some(epoch))?;
    ensure!(node["desired"] != "removed", "invalid node credential: retired agent");
    let version = body.get("applied_version").unwrap_or(&json!(0)).as_u64().filter(|v| *v <= MAX_VERSION).context("invalid telemetry acknowledgement")?;
    let status = body.get("state").and_then(Value::as_str).unwrap_or(if body.get("state").is_some() { "invalid" } else { "pending" });
    ensure!(["pending","applied","error"].contains(&status), "invalid telemetry acknowledgement");
    ensure!(object.contains_key("service_url")==object.contains_key("incarnation"),"service claim requires URL and incarnation");
    if object.contains_key("service_url") {location_url(body["service_url"].as_str().context("invalid service URL")?)?;identifier(body["incarnation"].as_str().context("invalid incarnation")?)?;}
    let snapshot = store.snapshot()?;
    if !object.contains_key("service_url") && snapshot["telemetry"]["bindings"][node_id] == service && snapshot["telemetry"]["receipts"][node_id] == json!({"version":version,"state":status,"service_id":service}) {
        let mut result = view(&snapshot, Some(node_id))?;
        result.as_object_mut().unwrap().remove("receipts");
        return Ok(result);
    }
    let digest = verifier(token)?;
    store.mutate(|state| {
        let current = &state["nodes"][node_id];
        ensure!(current["verifier"] == digest && current["agent_epoch"] == epoch && current["desired"] != "removed" && current.get("purge").is_none(), "invalid node credential");
        if state.get("telemetry").is_none() { state["telemetry"] = initial_state(); }
        let telemetry = &mut state["telemetry"];
        ensure!(telemetry["bindings"].get(node_id).map(|v| v == service).unwrap_or(true), "telemetry service binding conflict");
        ensure!(version <= telemetry["version"].as_u64().unwrap(), "future telemetry acknowledgement");
        telemetry["bindings"][node_id] = json!(service);
        if telemetry["receipts"].get(node_id).map(|r|version >= r["version"].as_u64().unwrap()).unwrap_or(true) {
            telemetry["receipts"][node_id] = json!({"version":version,"state":status,"service_id":service});
        }
        if object.contains_key("service_url") {
            if telemetry.get("locations").is_none(){telemetry["locations"]=json!({});}
            let location=&telemetry["locations"][service];let now=crate::controller::now();
            if location.is_null() || location["expires_at"].as_f64().unwrap_or(0.0)<=now || location["incarnation"]==body["incarnation"] {
                telemetry["locations"][service]=json!({"url":body["service_url"],"incarnation":body["incarnation"],"expires_at":now+30.0});
            }
        }
        let mut result = view(state, Some(node_id))?;
        result.as_object_mut().unwrap().remove("receipts");
        result["location"]=state["telemetry"]["locations"][service].clone();
        Ok(result)
    })
}
