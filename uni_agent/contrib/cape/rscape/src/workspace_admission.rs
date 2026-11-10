// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::controller::{Controller, now};
use crate::model::NodeState;
use crate::workspace_fs::MaintenancePolicy;
use serde_json::{json, Value};

fn host(controller: &Controller, node: &str) -> Option<String> {
    controller.nodes.get(node)?.workspace_status["maintenance_host_key"].as_str()
        .filter(|s| s.len()==64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
        .map(str::to_owned)
}

fn overlaps(a: &Option<String>, b: &Option<String>) -> bool { a.is_none() || b.is_none() || a==b }

pub fn dispatch(controller: &mut Controller, node_id: &str, ready: bool) -> Value {
    let Some(node)=controller.nodes.get(node_id) else {return json!([])};
    if node.state==NodeState::Removed {return json!([]);}
    let capable=node.workspace_status["maintenance_protocol"]==1;
    let active=node.state==NodeState::Active;
    let epoch=json!(node.agent_epoch);
    let key=host(controller,node_id);
    let time=now();
    let pending:Vec<Value>=controller.workspaces.values().filter(|r|r["operation"]["state"]=="pending").cloned().collect();
    let granted:Vec<&Value>=pending.iter().filter(|r|r["operation"]["dispatch_granted"]!=false).collect();
    if let Some(own)=granted.iter().find(|r|r["node_id"]==node_id) {return json!([own]);}
    if granted.iter().any(|r| {
        let owner=controller.nodes.get(r["node_id"].as_str().unwrap_or(""));
        let grant=if owner.map(|n|json!(n.agent_epoch)==r["operation"]["grant_epoch"]).unwrap_or(false) {
            r["operation"]["grant_host_key"].as_str().map(str::to_owned)
        } else {None};
        overlaps(&key,&grant)
    }) {return json!([]);}
    let mut eligible:Vec<Value>=pending.into_iter().filter(|r| {
        let Some(owner)=controller.nodes.get(r["node_id"].as_str().unwrap_or("")) else{return false;};
        let legacy_safe=matches!(r["operation"]["action"].as_str(),Some("check"|"delete")) && !r["maintenance"].is_object();
        if !matches!(owner.state,NodeState::Active|NodeState::Draining) || (owner.workspace_status["maintenance_protocol"]!=1 && !legacy_safe) {return false;}
        overlaps(&key,&host(controller,r["node_id"].as_str().unwrap_or(""))) &&
        !r["execution_session_id"].as_str().and_then(|id|controller.sessions.get(id)).map(|s|s.active_request_id.is_some()).unwrap_or(false)
    }).collect();
    eligible.sort_by(|a,b|a["operation"]["requested_at"].as_f64().unwrap_or(0.).total_cmp(&b["operation"]["requested_at"].as_f64().unwrap_or(0.))
        .then_with(||a["id"].as_str().cmp(&b["id"].as_str())));
    let mut selected=if let Some(first)=eligible.first() {
        let legacy_safe=matches!(first["operation"]["action"].as_str(),Some("check"|"delete")) && !first["maintenance"].is_object();
        if first["node_id"]!=node_id || !(capable || legacy_safe) {return json!([]);}
        first.clone()
    } else {
        if !ready || !capable || !active {return json!([]);}
        let pools:Vec<String>=controller.nodes.iter().filter(|(id,n)|!matches!(n.state,NodeState::Removed|NodeState::Lost) && overlaps(&key,&host(controller,id))).map(|(_,n)|n.pool.clone()).collect();
        if controller.requests.values().any(|r|pools.contains(&r.spec.pool)&&!r.state.terminal()) {return json!([]);}
        let mut candidates=Vec::new();let mut interval=0;let mut window=1024;
        let ids:Vec<String>=controller.workspaces.keys().cloned().collect();
        for id in ids {
            let mut record=controller.workspaces[&id].clone();
            let owner_id=record["node_id"].as_str().unwrap_or("");
            if !overlaps(&key,&host(controller,owner_id)) {continue;}
            let Some(owner)=controller.nodes.get(owner_id) else {continue;};
            let Ok(config)=controller.workspace_configs.get(record["pool"].as_str().unwrap_or(""),Some(owner_id)) else {continue;};
            if owner.state!=NodeState::Active || owner.workspace_status["maintenance_protocol"]!=1 || owner.workspace_status["state"]!="applied" || owner.workspace_status["applied_version"]!=config.version {continue;}
            let policy=config.maintenance.unwrap_or_default();
            interval=interval.max(policy.reclaim_start_interval_seconds);window=window.min(policy.reclaim_max_candidates);
            controller.automatic_workspace_operation(&mut record);
            let candidate=&record["reclaim_candidate"];
            if candidate.is_object() && candidate["not_before"].as_f64().unwrap_or(f64::INFINITY)<=time &&
                !record["execution_session_id"].as_str().and_then(|id|controller.sessions.get(id)).map(|s|s.active_request_id.is_some()).unwrap_or(false) {
                candidates.push(record.clone());
                candidates.sort_by(|a,b|a["reclaim_candidate"]["first_seen_at"].as_f64().unwrap_or(0.).total_cmp(&b["reclaim_candidate"]["first_seen_at"].as_f64().unwrap_or(0.))
                    .then_with(||a["id"].as_str().cmp(&b["id"].as_str())));
            }
            candidates.truncate(window as usize);
            controller.workspaces.insert(id,record);
        }
        if controller.workspaces.values().any(|r|overlaps(&key,&r["reclaim_attempt_host_key"].as_str().map(str::to_owned)) &&
            time < r["reclaim_last_attempt_at"].as_f64().unwrap_or(-1e30)+interval as f64) {return json!([]);}
        candidates.sort_by(|a,b|a["reclaim_candidate"]["first_seen_at"].as_f64().unwrap_or(0.).total_cmp(&b["reclaim_candidate"]["first_seen_at"].as_f64().unwrap_or(0.))
            .then_with(||a["id"].as_str().cmp(&b["id"].as_str())));
        candidates.truncate(window as usize);
        let Some(mut record)=candidates.into_iter().next() else{return json!([])};
        if record["node_id"]!=node_id {return json!([]);}
        let Ok(config)=controller.workspace_configs.get(record["pool"].as_str().unwrap_or(""),Some(node_id)) else {return json!([])};
        let Some(sequence)=record["operation_sequence"].as_u64().unwrap_or(0).checked_add(1).filter(|v|*v<=i64::MAX as u64) else{return json!([])};
        record["operation_sequence"]=json!(sequence);
        record["operation"]=json!({"id":uuid::Uuid::new_v4().simple().to_string(),"action":"reclaim","state":"pending",
            "sequence":sequence,"source":"automatic","boundary":false,"requested_at":time,"phase":"requested","target_bytes":null,
            "policy":config.maintenance.unwrap_or_default(),"policy_version":config.version,"max_image_size_bytes":config.max_image_size_bytes,
            "usage_generation":record["usage_generation"]});
        record["reclaim_last_attempt_at"]=json!(time);record["reclaim_attempt_host_key"]=json!(key);
        record["reclaim_candidate"]=Value::Null;record["reclaim_status"]=json!("granted");
        record
    };
    selected["operation"]["dispatch_granted"]=json!(true);
    selected["operation"]["grant_host_key"]=json!(key);selected["operation"]["grant_epoch"]=epoch;
    let id=selected["id"].as_str().unwrap().to_owned();
    controller.event("workspace_state",&id,selected["state"].as_str(),Some("maintenance_granted"),selected.clone());
    controller.workspaces.insert(id,selected.clone());
    json!([selected])
}

pub fn complete(record:&mut Value,time:f64) {
    let operation=&record["operation"];
    if operation["action"]!="reclaim" {return;}
    let Ok(policy)=MaintenancePolicy::from_value(operation.get("policy").cloned().unwrap_or(json!({}))) else{return;};
    if operation["state"]=="completed" {
        let reclaimed=operation["reclaimed_bytes"].as_u64().unwrap_or(0);
        record["reclaim_last_success_at"]=json!(time);record["reclaim_success_generation"]=record["usage_generation"].clone();
        if reclaimed<policy.reclaim_min_allocated_bytes {record["reclaim_low_yield_until"]=json!(time+policy.reclaim_low_yield_backoff_seconds as f64);}
    } else if operation["state"]=="failed" {record["reclaim_retry_after"]=json!(time+policy.reclaim_retry_backoff_seconds as f64);}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> Controller {
        let mut c=Controller::new(crate::auth::Authority::default(),None).unwrap();
        c.workspace_configs.update("pool",json!({"maintenance":{"reclaim_on_release":true,"reclaim_idle_seconds":0}}).as_object().unwrap(),None).unwrap();
        let config=c.workspace_configs.get("pool",None).unwrap();
        for id in ["a","b"] {
            c.announce_or_register(&json!({"node_id":id,"pool":"pool","resources":{"cpu_cores":2,"memory_gb":4,"gpu_uuids":[]},
                "capabilities":{"adapter":"local-process","isolation_policies":["default","none"]},
                "allocation_id":format!("allocation-{id}"),"runtime_cache_config":"default","agent_epoch":"epoch"}),true,None).unwrap();
            c.nodes.get_mut(id).unwrap().workspace_status=json!({"maintenance_protocol":1,"state":"applied","applied_version":config.version,"maintenance_host_key":"a".repeat(64)});
        }
        c
    }
    fn workspace(c:&mut Controller,id:&str,node:&str) {
        let config=c.workspace_configs.get("pool",None).unwrap();
        let reference=json!({"allocated_bytes":268435456,"used_bytes":268435456,"logical_bytes":config.image_size_bytes,"image_generation":0,"usage_generation":0});
        let mut sample=reference.clone();sample["allocated_bytes"]=json!(1073741824);sample["usage_generation"]=json!(1);
        let mut record=json!({"id":id,"node_id":node,"pool":"pool","user_id":"alice","execution_session_id":id,"state":"retained",
            "usage_generation":1,"generation":0,"config":config.to_value(),"reclaim_reference":reference,"reclaim_sample":sample});
        c.automatic_workspace_operation(&mut record);c.workspaces.insert(id.into(),record);
    }
    #[test]
    fn workspace_policy_set_orders_with_grant_and_preserves_explicit_operations() {
        let mut c=setup();workspace(&mut c,"w1","a");
        assert_eq!(c.workspace_status("w1",None).unwrap()["auto_reclaim"],"inherit");
        c.workspaces.get_mut("w1").unwrap()["state"]=json!("mounted");
        let result=c.workspace_operation("w1","set",None,&json!({"auto_reclaim":"off"})).unwrap();
        assert_eq!(result["state"],"mounted");assert_eq!(result["auto_reclaim_enabled"],false);
        assert!(result["operation"].is_null());
        c.workspaces.get_mut("w1").unwrap()["state"]=json!("retained");
        assert_eq!(dispatch(&mut c,"a",true),json!([]));
        c.workspace_operation("w1","set",None,&json!({"auto_reclaim":"inherit"})).unwrap();
        let grant=dispatch(&mut c,"a",true);assert_eq!(grant.as_array().unwrap().len(),1);
        let result=c.workspace_operation("w1","set",None,&json!({"auto_reclaim":"off"})).unwrap();
        assert_eq!(result["operation"],grant[0]["operation"]);
        assert_eq!(dispatch(&mut c,"a",true)[0]["operation"],grant[0]["operation"]);
        c.workspaces.get_mut("w1").unwrap()["operation"]["state"]=json!("completed");
        let result=c.workspace_operation("w1","reclaim",None,&json!({})).unwrap();
        assert_eq!(result["auto_reclaim"],"off");assert_eq!(result["operation"]["source"],"explicit");
        assert_eq!(dispatch(&mut c,"a",true).as_array().unwrap().len(),1);
    }
    #[test]
    fn workspace_policy_validation_and_backoff() {
        let mut c=setup();workspace(&mut c,"w1","a");
        let before=c.workspaces.clone();
        for parameters in [json!({}),json!({"auto_reclaim":"on"}),json!({"auto_reclaim":true}),
            json!({"auto_reclaim":"off","operation_id":"id"}),json!({"auto_reclaim":null})] {
            assert!(c.workspace_operation("w1","set",None,&parameters).is_err());
        }
        assert_eq!(before,c.workspaces);
        c.workspaces.get_mut("w1").unwrap()["reclaim_retry_after"]=json!(now()+3600.);
        c.workspace_operation("w1","set",None,&json!({"auto_reclaim":"off"})).unwrap();
        c.workspace_operation("w1","set",None,&json!({"auto_reclaim":"inherit"})).unwrap();
        assert_eq!(dispatch(&mut c,"a",true),json!([]));
        assert!(c.workspaces["w1"].get("auto_reclaim_enabled").is_none());
    }
    #[test]
    fn readiness_duplicate_delivery_host_budget_and_rate() {
        let mut c=setup();workspace(&mut c,"w1","a");workspace(&mut c,"w2","b");
        assert!(c.workspaces["w1"]["operation"].is_null());
        assert_eq!(dispatch(&mut c,"a",false),json!([]));
        let grant=dispatch(&mut c,"a",true);assert_eq!(grant.as_array().unwrap().len(),1);
        assert_eq!(dispatch(&mut c,"a",true),grant);
        assert_eq!(dispatch(&mut c,"b",true),json!([]));
        let first=c.workspaces.get_mut("w1").unwrap();first["operation"]["state"]=json!("completed");first["operation"]["reclaimed_bytes"]=json!(100*1024*1024);
        complete(first,now());
        assert_eq!(dispatch(&mut c,"b",true),json!([]));
        c.workspaces.get_mut("w1").unwrap()["reclaim_last_attempt_at"]=json!(now()-31.);
        assert_eq!(dispatch(&mut c,"b",true).as_array().unwrap().len(),1);
    }
    #[test]
    fn unknown_host_conflicts_but_distinct_known_hosts_do_not() {
        let mut c=setup();workspace(&mut c,"w1","a");workspace(&mut c,"w2","b");dispatch(&mut c,"a",true);
        c.nodes.get_mut("b").unwrap().workspace_status["maintenance_host_key"]=Value::Null;
        assert_eq!(dispatch(&mut c,"b",true),json!([]));
        c.nodes.get_mut("b").unwrap().workspace_status["maintenance_host_key"]=json!("b".repeat(64));
        assert_eq!(dispatch(&mut c,"b",true).as_array().unwrap().len(),1);
    }
    #[test]
    fn failure_low_yield_and_later_generation() {
        let mut c=setup();workspace(&mut c,"w1","a");dispatch(&mut c,"a",true);
        let r=c.workspaces.get_mut("w1").unwrap();r["operation"]["state"]=json!("failed");complete(r,1000.);
        assert_eq!(r["reclaim_retry_after"],1300.);
        r["operation"]["state"]=json!("completed");r["operation"]["reclaimed_bytes"]=json!(0);complete(r,1000.);
        assert_eq!(r["reclaim_success_generation"],1);assert_eq!(r["reclaim_low_yield_until"],4600.);
        r["usage_generation"]=json!(2);r["reclaim_sample"]["usage_generation"]=json!(2);
        let policy=MaintenancePolicy::default();
        let policy=MaintenancePolicy {reclaim_on_release:true,..policy};
        assert!(crate::workspace_reclaim::candidate(r,&policy,5000.).0.is_some());
    }
    #[test]
    fn outstanding_user_work_and_unapplied_policy_preempt_candidates() {
        let mut c=setup();workspace(&mut c,"w1","a");
        let spec=serde_json::from_value(json!({"pool":"pool","user_id":"alice","session_id":"work",
            "command":["true"],"resources":{"gpus":0,"runtime_adapter":"local-process","isolation_policy":"none"}})).unwrap();
        c.submit(spec,None).unwrap();
        assert_eq!(dispatch(&mut c,"a",true),json!([]));
        assert!(c.workspaces["w1"]["operation"].is_null());
        c.requests.clear();
        c.workspace_configs.update("pool",json!({"maintenance":{"reclaim_on_release":false}}).as_object().unwrap(),None).unwrap();
        assert_eq!(dispatch(&mut c,"a",true),json!([]));
    }
    #[test]
    fn unavailable_ungranted_owner_does_not_block_but_uncertain_grant_does() {
        let mut c=setup();workspace(&mut c,"w1","a");workspace(&mut c,"w2","b");
        c.workspaces.get_mut("w1").unwrap()["operation"]=json!({"id":"explicit","state":"pending","action":"reclaim","dispatch_granted":false,"requested_at":1});
        c.nodes.get_mut("a").unwrap().state=NodeState::Unhealthy;
        assert_eq!(dispatch(&mut c,"b",true).as_array().unwrap().len(),1);
        assert_eq!(c.workspaces["w1"]["operation"]["dispatch_granted"],false);
        let mut c=setup();workspace(&mut c,"w1","a");workspace(&mut c,"w2","b");dispatch(&mut c,"a",true);
        c.nodes.get_mut("a").unwrap().state=NodeState::Unhealthy;
        c.nodes.get_mut("a").unwrap().agent_epoch=Some("new-epoch".into());
        c.nodes.get_mut("b").unwrap().workspace_status["maintenance_host_key"]=json!("b".repeat(64));
        assert_eq!(dispatch(&mut c,"b",true),json!([]));
    }
}
