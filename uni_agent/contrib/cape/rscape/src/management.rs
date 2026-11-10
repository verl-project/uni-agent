// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Website projection over the native identity store, not a separate authority.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use crate::controller::{Controller, now};
use crate::identity::{authenticate, audit, name, new_secret, verifier};

pub fn capabilities(controller: &Controller) -> Value {
    json!({"management_api":2,"managed":controller.authority.identity_path.is_some(),
        "identity_model":"persistent-groups", "features":["identity","groups","sessions","sso-sessions","resources","node-enrollment","node-controls"]})
}
pub(crate) fn public_user(state: &Value, uid: &str, admin: bool) -> Value {
    let mut user=state["users"][uid].clone();
    user["id"]=json!(uid); user["platform_admin"]=json!(admin);
    let mut grants=serde_json::Map::new();
    for pid in state["pools"].as_object().unwrap().keys() {
        if admin || user["groups"].as_array().unwrap().iter().any(|g| {
            let group=&state["groups"][g.as_str().unwrap()];
            group["all_pools"]==true || group["pools"].as_array().unwrap().contains(&json!(pid))
        }) { grants.insert(pid.clone(),json!(["submit"])); }
    }
    user["grants"]=Value::Object(grants); user
}
pub(crate) fn public_token(id: &str, row: &Value) -> Value {
    let mut row=row.clone(); row.as_object_mut().unwrap().remove("verifier"); row["id"]=json!(id); row
}
fn text<'a>(body: &'a Value, key: &str) -> Result<&'a str> { body[key].as_str().context("missing string parameter") }
fn flag(body: &Value,key:&str) -> Result<bool> { body.get(key).map(Value::as_bool).unwrap_or(Some(false)).context("invalid boolean") }
fn control(result: crate::controller::ControlResult<Value>) -> Result<Value> {
    result.map_err(anyhow::Error::new)
}
pub fn call(controller: &mut Controller, action: &str, token: &str, body: &Value) -> Result<Value> {
    ensure!(body.is_object(),"management body must be an object");
    let mut store=controller.authority.identity()?.context("persistent identity is not configured")?;
    let time=now();
    if action.starts_with("sso.") || action.starts_with("session.") {
        return crate::management_sso::call(&mut store,action,token,body,time);
    }
    let actor=store.authenticate(token,time)?; let state=store.snapshot()?;
    let uid=actor["user_id"].as_str().unwrap();
    match action {
        "me" => {
            let mut cred=json!({"id":actor["token_id"],"kind":actor["token_kind"]});
            if actor["token_kind"]=="session" {cred["auth_method"]=json!("sso");}
            return Ok(json!({"user":public_user(&state,uid,actor["is_admin"]==true),"credential":cred,"scheduler_id":state["scheduler_id"]}));
        },
        "resources" => {
            let mut pools=Vec::new();
            for (id,pool) in state["pools"].as_object().unwrap() {
                let mut pool=pool.clone(); pool["id"]=json!(id);
                let mut utilization=controller.utilization(id);
                if let Some(nodes)=utilization["node_states"].as_array_mut() { for node in nodes { if let Some(row)=node.as_object_mut(){row.remove("active_sessions");} } }
                pool["utilization"]=utilization;
                let mut nodes=Vec::new();
                if let Some(rows)=state["nodes"].as_object() { for (nid,row) in rows {
                    if row["pool"]!=*id || row["desired"]=="removed" {continue;}
                    let mut node=row.get("effective_resources").unwrap_or(&row["resources"]).clone(); node["id"]=json!(nid);
                    node["state"]=controller.nodes.get(nid).map(|n|serde_json::to_value(n.state).unwrap()).unwrap_or(json!("enrolled"));
                    nodes.push(node);
                } }
                pool["nodes"]=json!(nodes); pools.push(pool);
            }
            return Ok(json!({"pools":pools,"timestamp":time}));
        },
        "pools.list" => return Ok(json!({"pools":state["pools"].as_object().unwrap().iter().map(|(id,p)| {let mut p=p.clone();p["id"]=json!(id);p}).collect::<Vec<_>>()})),
        "tokens.list" => return Ok(json!({"tokens":state["tokens"].as_object().unwrap().iter().filter(|(_,t)|t["user_id"]==uid && t["kind"]=="regular").map(|(id,t)|public_token(id,t)).collect::<Vec<_>>()})),
        "tokens.create" => {
            let op=text(body,"operation_id")?; ensure!(!op.is_empty() && op.len()<=128,"operation_id required");
            let label=body.get("name").unwrap_or(&Value::Null).as_str().unwrap_or("personal").to_owned(); name(&label)?;
            let expiry:Option<f64>=serde_json::from_value(body["expires_at"].clone())?;
            let params=json!({"name":label,"expires_at":expiry});
            return store.mutate(|current| {
                let actor=authenticate(current,token,time)?; let uid=actor["user_id"].as_str().unwrap();
                let key=format!("web-token:{uid}:{op}");
                if let Some(previous)=current["operations"].get(&key) {
                    ensure!(previous["params"]==params,"operation parameters differ");
                    let mut result=previous["result"].clone(); result["already_issued"]=json!(true); return Ok(result);
                }
                ensure!(expiry.map(|e|e.is_finite() && e>time).unwrap_or(true),"expiry must be in the future");
                let secret=new_secret()?; let id=uuid::Uuid::new_v4().to_string();
                let row=json!({"user_id":uid,"name":label,"kind":"regular","verifier":verifier(&secret)?,"expires_at":expiry,"revoked":false,"created_at":time});
                current["tokens"][&id]=row.clone(); let mut result=public_token(&id,&row);
                current["operations"][&key]=json!({"owner_user_id":uid,"params":params,"result":result});
                audit(current,uid,"token.create",&id,time); result["token"]=json!(secret); Ok(result)
            });
        },
        "tokens.revoke" => {store.revoke_token(token,text(body,"token_id")?,time)?;return Ok(json!({"revoked":true}));},
        "nodes.list" => {
            let rows=state["nodes"].as_object().map(|rows| rows.iter().filter(|(_,row)|actor["is_admin"]==true || row["owner_user_id"]==uid).map(|(id,row)|public_token(id,row)).collect::<Vec<_>>()).unwrap_or_default();
            return Ok(json!({"nodes":rows}));
        },
        "nodes.enroll" => return store.enroll_node(token,body,time),
        "nodes.action" => {
            let id=text(body,"node_id")?;let action=text(body,"action")?;
            return if action=="edit" {ensure!(!flag(body,"force")?,"force applies to drain, not edit");control(controller.managed_node_edit(Some(token),id,&body["changes"]))}
                else {control(controller.managed_node_action(Some(token),id,action,flag(body,"force")?,false))};
        },
        _=>{}
    }
    store.require_admin(token,time)?;
    match action {
        "users.list"=>return Ok(json!({"users":state["users"].as_object().unwrap().keys().map(|id|public_user(&state,id,false)).collect::<Vec<_>>()})),
        "users.update"=>{
            ensure!(body.as_object().unwrap().keys().all(|k|["operation_id","user_id","groups","disabled"].contains(&k.as_str())),"use group membership, not legacy grants or platform_admin");
            store.edit_user(token,text(body,"user_id")?,serde_json::from_value(body["groups"].clone())?,serde_json::from_value(body["disabled"].clone())?,time)?;
        },
        "groups.list"=>return Ok(json!({"groups":state["groups"]})),
        "groups.set"=>store.set_group(token,text(body,"name")?,flag(body,"all_pools")?,&serde_json::from_value::<Vec<String>>(body.get("pools").cloned().unwrap_or(json!([])))?,time)?,
        "groups.delete"=>store.delete_group(token,text(body,"name")?,time)?,
        "pools.create"=>return Ok(json!({"id":store.create_pool_config(token,text(body,"name")?,body.get("config").filter(|v|!v.is_null()).unwrap_or(&json!({})),serde_json::from_value(body["max_nodes"].clone())?,time)?})),
        "pools.edit"=>return control(controller.managed_pool_edit(Some(token),&store.pool_id(text(body,"pool")?)?,&body["changes"])),
        "pools.drain"|"pools.remove"=>return control(controller.managed_pool_action(Some(token),&store.pool_id(text(body,"pool")?)?,action.split('.').nth(1).unwrap(),flag(body,"force")?)),
        "audit.list"=>return Ok(json!({"items":store.audit_events(token,serde_json::from_value(body.get("limit").cloned().unwrap_or(json!(100)))?,time)?})),
        _=>bail!("unsupported management v2 action; use identity groups and node enrollment")
    }
    Ok(json!({"updated":true}))
}
