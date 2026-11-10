// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Version-fenced SSO extension of the native authority snapshot.
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use crate::identity::{IdentityStore, authenticate, audit, new_secret, verifier};
use crate::management::{public_user, public_token};

fn text<'a>(v:&'a Value,key:&str)->Result<&'a str>{v[key].as_str().context("missing string parameter")}
fn guid(v:&str)->Result<&str>{ensure!(uuid::Uuid::parse_str(v)?.to_string()==v,"expected canonical UUID");Ok(v)}
fn matches(a:&str,b:&str)->bool {a.len()==b.len() && a.bytes().zip(b.bytes()).fold(0u8,|v,(a,b)|v|(a^b))==0}
pub(crate) fn contains_verifier(state:&Value,digest:&str)->bool {
    ["bridges","sessions"].iter().any(|kind|state["web_auth"][*kind].as_object().map(|rows|rows.values().any(|r|r["verifier"]==digest)).unwrap_or(false))
}
fn active_bridge(state:&Value,row:&Value)->bool {
    let owner=&state["users"][row["user_id"].as_str().unwrap_or("")];
    row["revoked"]==false && owner["disabled"]==false && owner["role"]=="admin"
}
pub(crate) fn validate(state:&Value)->Result<()> {
    let Some(auth)=state.get("web_auth") else {return Ok(());};
    ensure!(state["version"]==3 && auth.as_object().map(|m|m.len()==2).unwrap_or(false),"invalid SSO authority");
    for kind in ["bridges","sessions"] {
        for (key,row) in auth[kind].as_object().context("invalid SSO records")? {
            guid(key)?;
            ensure!(state["users"].get(text(row,"user_id")?).is_some() && row["revoked"].is_boolean(),"invalid SSO owner");
            let digest=text(row,"verifier")?;
            ensure!(digest.len()==64 && digest.bytes().all(|b|b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),"invalid SSO verifier");
            if kind=="bridges" {
                guid(text(row,"tenant_id")?)?;guid(text(row,"client_id")?)?;
                ensure!(row["auto_provision"].is_boolean(),"invalid SSO provisioning policy");
            } else {
                let parent=auth["bridges"].get(text(row,"parent_id")?).context("invalid SSO parent")?;
                ensure!(row["expires_at"].as_f64().map(f64::is_finite).unwrap_or(false),"invalid SSO expiry");
                let user=&state["users"][text(row,"user_id")?];
                guid(text(user,"sso_tenant")?)?;guid(text(user,"sso_subject")?)?;
                ensure!(user["sso_tenant"]==parent["tenant_id"],"invalid SSO session scope");
            }
        }
    }
    Ok(())
}
pub(crate) fn authenticate_session(state:&Value,secret:&str,now:f64)->Result<Value>{
    let digest=verifier(secret)?;
    if let Some(sessions)=state["web_auth"]["sessions"].as_object(){
        for (id,row) in sessions {
            if !matches(text(row,"verifier")?,&digest){continue;}
            let uid=text(row,"user_id")?;let user=&state["users"][uid];
            ensure!(row["revoked"]==false && user["disabled"]==false && now<row["expires_at"].as_f64().context("invalid expiry")?
                && active_bridge(state,&state["web_auth"]["bridges"][text(row,"parent_id")?]),"invalid credential");
            let mut actor=user.clone();actor["user_id"]=json!(uid);actor["token_id"]=json!(id);
            actor["token_kind"]=json!("session");actor["is_admin"]=json!(user["role"]=="admin");return Ok(actor);
        }
    }
    bail!("invalid credential")
}
fn provision(state:&mut Value,tenant:&str,subject:&str,allow:bool,target:Option<&str>)->Result<String>{
    let users=state["users"].as_object().unwrap();
    let found:Vec<String>=users.iter().filter(|(_,u)|u["sso_tenant"]==tenant && u["sso_subject"]==subject).map(|(id,_)|id.clone()).collect();
    let uid=if !found.is_empty(){
        ensure!(found.len()==1 && target.map(|t|t==found[0]).unwrap_or(true),"SSO identity already bound");found[0].clone()
    }else if let Some(target)=target{
        ensure!(users.get(target).map(|u|u["sso_subject"].is_null()).unwrap_or(false),"SSO target unavailable");
        state["users"][target]["sso_tenant"]=json!(tenant);state["users"][target]["sso_subject"]=json!(subject);target.to_owned()
    }else{
        ensure!(allow,"SSO account not provisioned");
        let uid=uuid::Uuid::new_v4().to_string();
        state["users"][&uid]=json!({"name":format!("sso-{uid}"),"role":"regular","disabled":false,"groups":[],"sso_tenant":tenant,"sso_subject":subject});uid
    };
    ensure!(state["users"][&uid]["disabled"]==false,"SSO account unavailable");Ok(uid)
}
pub fn call(store:&mut IdentityStore,action:&str,token:&str,body:&Value,now:f64)->Result<Value>{
    let fields: &[&str] = match action {
        "session.sso" => &["tenant_id", "client_id", "subject"],
        "session.revoke" => &[],
        "sso.bridge.create" => &["tenant_id", "client_id", "auto_provision"],
        "sso.bridge.revoke" => &["bridge_id"],
        "sso.users.create" => &["tenant_id", "subject"],
        "sso.users.bind" => &["tenant_id", "subject", "user_id"],
        _ => bail!("unknown SSO action"),
    };
    ensure!(body.as_object().context("SSO object required")?.keys().all(|key| key=="operation_id" || fields.contains(&key.as_str())), "unknown SSO parameter");
    store.mutate(|state|{
        let actor_id:String;let mut uid=String::new();let mut parent=String::new();
        let tenant:String;
        if action=="session.sso"{
            let digest=verifier(token)?;
            let pair=state["web_auth"]["bridges"].as_object().and_then(|rows|rows.iter().find(|(_,r)|r["verifier"].as_str().map(|s|matches(s,&digest)).unwrap_or(false) && active_bridge(state,r)))
                .map(|(id,row)|(id.clone(),row.clone())).context("SSO bridge credential required")?;
            parent=pair.0;let bridge=pair.1;
            tenant=guid(text(body,"tenant_id")?)?.to_owned();let subject=guid(text(body,"subject")?)?;
            ensure!(bridge["tenant_id"]==tenant && bridge["client_id"]==body["client_id"],"SSO identity scope mismatch");
            uid=provision(state,&tenant,subject,bridge["auto_provision"]==true,None)?;actor_id=parent.clone();
        }else{
            let actor=authenticate(state,token,now)?;actor_id=text(&actor,"user_id")?.to_owned();
            if action=="session.revoke"{
                ensure!(actor["token_kind"]=="session","SSO session required");
                let id=text(&actor,"token_id")?;state["web_auth"]["sessions"][id]["revoked"]=json!(true);
                audit(state,&actor_id,action,id,now);return Ok(json!({"revoked":true}));
            }
            ensure!(actor["is_admin"]==true,"admin credential required");
            if action=="sso.bridge.revoke"{
                let id=text(body,"bridge_id")?;let row=state["web_auth"]["bridges"].get_mut(id).context("unknown bridge")?;
                row["revoked"]=json!(true);audit(state,&actor_id,action,id,now);return Ok(json!({"revoked":true}));
            }
            ensure!(["sso.bridge.create","sso.users.create","sso.users.bind"].contains(&action),"unsupported SSO action");
            tenant=guid(text(body,"tenant_id")?)?.to_owned();
            if action=="sso.bridge.create"{
                guid(text(body,"client_id")?)?;ensure!(body["auto_provision"].is_boolean(),"auto_provision must be explicitly boolean");
            }else{
                let subject=guid(text(body,"subject")?)?;
                let target=if action=="sso.users.bind" {Some(guid(text(body,"user_id")?)?)}else{None};
                uid=provision(state,&tenant,subject,true,target)?;state["version"]=json!(3);
                audit(state,&actor_id,action,&uid,now);return Ok(json!({"id":uid,"user":public_user(state,&uid,false)}));
            }
        }
        let op=text(body,"operation_id")?;ensure!(!op.is_empty() && op.len()<=128,"operation_id required");
        let key=format!("web-auth:{actor_id}:{action}:{op}");
        if let Some(old)=state["operations"].get(&key){
            ensure!(old["params"]==*body,"operation parameters differ");let mut result=old["result"].clone();result["already_issued"]=json!(true);return Ok(result);
        }
        state["version"]=json!(3);
        if state.get("web_auth").is_none(){state["web_auth"]=json!({"bridges":{},"sessions":{}});}
        let secret=new_secret()?;let id=uuid::Uuid::new_v4().to_string();let secret_key;let mut result;
        if action=="sso.bridge.create"{
            ensure!(state["web_auth"]["bridges"].as_object().unwrap().values().filter(|r|r["revoked"]==false).count()<8,"SSO bridge limit reached");
            let row=json!({"user_id":actor_id,"verifier":verifier(&secret)?,"revoked":false,"tenant_id":tenant,"client_id":body["client_id"],"auto_provision":body["auto_provision"]});
            state["web_auth"]["bridges"][&id]=row.clone();result=public_token(&id,&row);secret_key="token";
        }else{
            ensure!(state["web_auth"]["sessions"].as_object().unwrap().values().filter(|r|r["user_id"]==uid && r["revoked"]==false && r["expires_at"].as_f64().unwrap_or(0.0)>now).count()<20,"active session limit reached");
            let row=json!({"user_id":uid,"verifier":verifier(&secret)?,"revoked":false,"parent_id":parent,"expires_at":now+3600.0,"auth_method":"sso","kind":"session"});
            state["web_auth"]["sessions"][&id]=row.clone();result=public_token(&id,&row);result["user"]=public_user(state,&uid,state["users"][&uid]["role"]=="admin");secret_key="session_token";
        }
        state["operations"][&key]=json!({"params":body,"result":result});audit(state,&actor_id,action,&id,now);
        result[secret_key]=json!(secret);Ok(result)
    })
}
