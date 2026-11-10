// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-agent isolated control worker. It never collects GPUs or shares agent secrets.
use anyhow::{ensure, Context, Result};
use serde_json::{json,Value};
use std::{io::Write, process::Stdio, time::Duration};
use tokio::io::{AsyncBufReadExt,AsyncWriteExt,BufReader};

pub struct AgentLink {
    child: std::process::Child,
    pub status: std::sync::Arc<std::sync::Mutex<Value>>,
}

impl AgentLink {
    pub fn start(boot:Value)->Result<Self>{
        use std::io::{BufRead,Write};
        let mut child=std::process::Command::new(std::env::current_exe()?).arg("__metric-link")
            .env("TOKIO_WORKER_THREADS","2")
            .env_remove("CAPE_TOKEN").env_remove("CAPE_NODE_TOKEN")
            .env_remove("CAPE_ADMIN_TOKEN").env_remove("CAPE_JOIN_TOKEN")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
        let mut input=child.stdin.take().context("link stdin")?;
        let output=child.stdout.take().context("link stdout")?;
        let status=std::sync::Arc::new(std::sync::Mutex::new(json!({"state":"pending"})));let latest=status.clone();
        std::thread::spawn(move||{
            let written=writeln!(input,"{boot}").is_ok();drop(input);
            if written {for line in std::io::BufReader::new(output).lines().map_while(std::result::Result::ok){
                if line.len()>16384{continue;}if let Ok(value)=serde_json::from_str::<Value>(&line){*latest.lock().unwrap()=value;}
            }}
            let mut last=latest.lock().unwrap();last["state"]=json!("error");last["reason"]=json!("metrics_link_stopped");
        });
        Ok(Self{child,status})
    }
}

impl Drop for AgentLink {
    fn drop(&mut self){
        if self.child.try_wait().ok().flatten().is_none(){
            unsafe{nix::libc::kill(self.child.id() as i32,nix::libc::SIGTERM);}
            let deadline=std::time::Instant::now()+Duration::from_secs(5);
            while self.child.try_wait().ok().flatten().is_none() && std::time::Instant::now()<deadline{std::thread::sleep(Duration::from_millis(20));}
            if self.child.try_wait().ok().flatten().is_none(){let _=self.child.kill();}
            let _=self.child.wait();
        }
    }
}

async fn post(client:&reqwest::Client,base:&str,path:&str,body:&Value,token:Option<&str>)->Result<Value>{
    let mut request=client.post(format!("{}{}",base.trim_end_matches('/'),path)).json(body);
    if let Some(token)=token{request=request.bearer_auth(token);}
    let response=request.send().await?;ensure!(response.status().is_success(),"metrics request unavailable");
    let data=response.bytes().await?;ensure!(data.len()<=1024*1024,"metrics response too large");Ok(serde_json::from_slice(&data)?)
}

async fn alive(client:&reqwest::Client,location:&Value,id:&str)->bool {
    let Some(url)=location["url"].as_str() else{return false};
    if crate::telemetry_service::controller_url(url).is_err(){return false;}
    let result=async {Ok::<Value,anyhow::Error>(client.get(format!("{}/snapshot",url.trim_end_matches('/'))).send().await?.json().await?)}.await;
    result.map(|v|v["service_id"]==id && v["incarnation"]==location["incarnation"]).unwrap_or(false)
}

async fn spawn_service(controller:&str,scheduler:&str)->Result<(tokio::process::Child,Value)> {
    let mut child=tokio::process::Command::new(std::env::current_exe()?).arg("__metric-service")
        .env_remove("CAPE_TOKEN").env_remove("CAPE_NODE_TOKEN")
        .env_remove("CAPE_ADMIN_TOKEN").env_remove("CAPE_JOIN_TOKEN")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    let result=async {
        let mut input=child.stdin.take().context("service stdin")?;
        input.write_all(format!("{}\n",json!({"controller_url":controller,"scheduler_id":scheduler})).as_bytes()).await?;drop(input);
        let mut line=String::new();let mut output=BufReader::new(child.stdout.take().context("service stdout")?);
        output.read_line(&mut line).await?;ensure!(line.len()<=16384,"invalid service startup");
        Ok::<Value,anyhow::Error>(serde_json::from_str(&line)?)
    };
    match tokio::time::timeout(Duration::from_secs(4),result).await {
        Ok(Ok(record))=>Ok((child,record)),
        _=>{let _=child.kill().await;let _=child.wait().await;anyhow::bail!("metrics helper unavailable")}
    }
}

pub async fn run(boot:Value)->Result<()> {
    let controller=boot["controller_url"].as_str().context("controller_url required")?;
    crate::telemetry_service::controller_url(controller)?;
    let node=boot["node_id"].as_str().context("node_id required")?;
    let epoch=boot["agent_epoch"].as_str().context("agent_epoch required")?;
    let token=boot["token"].as_str().context("agent credential required")?;
    let parent=boot["parent_pid"].as_i64().context("parent PID required")?;
    let client=reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(2)).redirect(reqwest::redirect::Policy::none()).build()?;
    let mut location=Value::Null;let mut id=String::new();let mut applied=0;let mut acknowledged="pending";
    let mut termination=tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    loop {
        if unsafe{nix::libc::getppid()} as i64 != parent{break;}
        let cycle=async {
            let discovery=post(&client,controller,"/v1/agent/metrics/control",&json!({"node_id":node,"agent_epoch":epoch,"discover":true}),Some(token)).await?;
            let scheduler=discovery["scheduler_id"].as_str().context("missing scheduler")?;
            id=crate::telemetry_service::host_service_id(scheduler)?;
            let mut body=json!({"node_id":node,"agent_epoch":epoch,"service_id":id,"applied_version":applied,"state":acknowledged});
            let mut control=post(&client,controller,"/v1/agent/metrics/control",&body,Some(token)).await?;
            if !alive(&client,&location,&id).await {
                let public:Value=client.get(format!("{}/v1/metrics/service-config",controller.trim_end_matches('/'))).query(&[("service_id",&id)]).send().await?.json().await?;
                location=public["location"].clone();
            }
            if !alive(&client,&location,&id).await {
                // Discovery records are public metadata only; credentials never go to their URLs.
                if let Ok(entries)=std::fs::read_dir(crate::telemetry_service::runtime_root(&id)?) {
                    for entry in entries.take(256).filter_map(|e|e.ok()) {
                        if entry.path().extension().and_then(|v|v.to_str())!=Some("json"){continue;}
                        let Ok(bytes)=std::fs::read(entry.path()) else{continue};if bytes.len()>16384{continue;}
                        if let Ok(record)=serde_json::from_slice::<Value>(&bytes) {if alive(&client,&record,&id).await{location=record;break;}}
                    }
                }
            }
            let mut spawned=None;
            if !alive(&client,&location,&id).await {
                let (child,record)=spawn_service(controller,scheduler).await?;spawned=Some(child);location=record;
            }
            body["service_url"]=location["url"].clone();body["incarnation"]=location["incarnation"].clone();
            let claimed=post(&client,controller,"/v1/agent/metrics/control",&body,Some(token)).await;
            if claimed.as_ref().map(|c|c["location"]["incarnation"]!=location["incarnation"]).unwrap_or(true) {
                if let Some(child)=spawned.as_mut(){let _=child.kill().await;let _=child.wait().await;}
            }
            control=claimed?;
            location=control["location"].clone();
            ensure!(alive(&client,&location,&id).await,"shared metrics service unreachable");
            let response=post(&client,location["url"].as_str().context("missing service URL")?,"/register",&json!({
                "node_id":node,"agent_epoch":epoch,"pool_id":boot["pool_id"],"gpu_uuids":boot["gpu_uuids"]}),None).await?;
            applied=response["applied_version"].as_u64().unwrap_or(0);
            acknowledged=if response["state"]=="pending"{"pending"}else if response["state"]=="unavailable"{"error"}else{"applied"};
            Ok::<Value,anyhow::Error>(json!({"service_id":id,"url":location["url"],"incarnation":location["incarnation"],"applied_version":applied,"state":acknowledged,"collection_state":response["state"]}))
        }.await;
        let status=cycle.unwrap_or_else(|_|json!({"service_id":id,"applied_version":applied,"state":"error","reason":"metrics_unavailable"}));
        println!("{status}");std::io::stdout().flush()?;
        tokio::select!{_=termination.recv()=>break,_=tokio::time::sleep(Duration::from_secs(2))=>{}}
    }
    if let Some(url)=location["url"].as_str(){let _=post(&client,url,"/register",&json!({"node_id":node,"agent_epoch":epoch,"remove":true}),None).await;}
    Ok(())
}
