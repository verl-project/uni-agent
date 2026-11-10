// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One native collector per host/scheduler, shared by Python and Rust agents.
use anyhow::{ensure, Context, Result};
use axum::{extract::{Query, State}, routing::{get, post}, Json, Router};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs::{self, OpenOptions}, io::Write, os::{fd::AsRawFd, unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt}}, path::PathBuf, sync::{Arc, Mutex}, time::{Duration, Instant}};
use crate::telemetry_contract::{CollectionConfig, LatestSnapshots, RegistrationSet};

type Shared = Arc<Mutex<Service>>;
struct Service {
    scheduler: String, id: String, incarnation: String, origin: Instant,
    registry: RegistrationSet, snapshots: LatestSnapshots, config: CollectionConfig,
    version: u64, state: String, last_send: String,
    shutdown: Arc<tokio::sync::Notify>,
}

pub fn host_service_id(scheduler: &str) -> Result<String> {
    uuid::Uuid::parse_str(scheduler)?;
    let boot=fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    uuid::Uuid::parse_str(boot.trim())?;
    Ok(ring::digest::digest(&ring::digest::SHA256,format!("{scheduler}\0{}",boot.trim()).as_bytes()).as_ref().iter().map(|b|format!("{b:02x}")).collect())
}

pub fn runtime_root(service:&str)->Result<PathBuf> {
    ensure!(service.len()==64 && service.bytes().all(|c|c.is_ascii_hexdigit()),"invalid metrics service ID");
    Ok(PathBuf::from(std::env::var_os("CAPE_METRICS_RUNTIME_DIR").unwrap_or_else(||"/tmp".into())).join(format!("cape-metrics-{service}")))
}

pub fn hostname()->Result<String> {
    let mut bytes=[0_u8;256];
    ensure!(unsafe{nix::libc::gethostname(bytes.as_mut_ptr().cast(),bytes.len())}==0,"hostname unavailable");
    Ok(std::str::from_utf8(&bytes[..bytes.iter().position(|v|*v==0).context("hostname too long")?])?.into())
}

pub fn controller_url(value:&str)->Result<()> {
    let url=url::Url::parse(value)?;
    ensure!(["http","https"].contains(&url.scheme()) && url.host_str().is_some() && url.username().is_empty() && url.password().is_none()
        && url.query().is_none() && url.fragment().is_none(),"invalid metrics controller URL"); Ok(())
}

async fn registration(State(state):State<Shared>,Json(body):Json<Value>)->(axum::http::StatusCode,Json<Value>) {
    let result=(||->Result<Value>{
        let mut state=state.lock().unwrap();let now=state.origin.elapsed().as_secs_f64();
        let object=body.as_object().context("invalid registration")?;
        ensure!(object.keys().all(|k|["node_id","agent_epoch","pool_id","gpu_uuids","remove"].contains(&k.as_str())),"unknown registration field");
        let node=body["node_id"].as_str().context("node_id required")?;
        let epoch=body["agent_epoch"].as_str().context("agent_epoch required")?;
        if body["remove"]==true {state.registry.remove(node,epoch,now)?;}
        else {
            let gpus:Vec<String>=serde_json::from_value(body["gpu_uuids"].clone())?;
            let scheduler=state.scheduler.clone();
            state.registry.register(&scheduler,node,epoch,body["pool_id"].as_str().context("pool_id required")?,&gpus,now)?;
        }
        Ok(json!({"service_id":state.id,"incarnation":state.incarnation,"applied_version":state.version,"state":state.state}))
    })();
    match result {Ok(value)=>(axum::http::StatusCode::OK,Json(value)),Err(_)=>(axum::http::StatusCode::BAD_REQUEST,Json(json!({"error":"invalid metrics registration"})))}
}

async fn snapshot(State(state):State<Shared>,Query(params):Query<BTreeMap<String,String>>)->(axum::http::StatusCode,Json<Value>) {
    let mut state=state.lock().unwrap();let now=state.origin.elapsed().as_secs_f64();
    if params.keys().any(|k|!["node_id","pool","gpu_uuid","metric"].contains(&k.as_str())) {
        return (axum::http::StatusCode::BAD_REQUEST,Json(json!({"error":"unknown local metrics filter"})));
    }
    let mut gpus=state.registry.gpu_uuids(params.get("pool").map(String::as_str),params.get("node_id").map(String::as_str),now).unwrap_or_default();
    if let Some(gpu)=params.get("gpu_uuid"){gpus.retain(|g|g==gpu);}
    let stale=state.config.interval_seconds*state.config.stale_periods as f64;
    let samples:Vec<_>=gpus.iter().filter_map(|gpu|state.snapshots.get(gpu,stale,now).ok()).map(|mut sample|{
        if let Some(metric)=params.get("metric"){sample.values.retain(|k,_|k==metric);}
        sample
    }).collect();
    (axum::http::StatusCode::OK,Json(json!({"scheduler_id":state.scheduler,"service_id":state.id,"incarnation":state.incarnation,
        "pid":std::process::id(),"state":state.state,"applied_version":state.version,"config":state.config,"last_send":state.last_send,"samples":samples})))
}

const BASIC: &[(u32,&str,f64,f64,f64)] = &[(203,"gpu_utilization_percent",1.,0.,100.),(204,"memory_utilization_percent",1.,0.,100.),
    (250,"memory_total_bytes",1048576.,0.,1e9),(252,"memory_used_bytes",1048576.,0.,1e9),(150,"temperature_c",1.,-50.,150.),(155,"power_watts",1.,0.,10000.)];
const PROF: &[(u32,&str,f64,f64,f64)] = &[(1002,"sm_active_ratio",1.,0.,1.),(1003,"sm_occupancy_ratio",1.,0.,1.),(1004,"tensor_active_ratio",1.,0.,1.),(1005,"dram_active_ratio",1.,0.,1.)];

fn parse_rows(text:&str,fields:&[(u32,&str,f64,f64,f64)])->BTreeMap<String,Option<f64>> {
    let mut latest=BTreeMap::new();
    for line in text.lines() {
        let words:Vec<_>=line.split_whitespace().collect();
        if words.len()!=fields.len()+2 || !["GPU","GI","CI"].contains(&words[0]) || words[1].parse::<u32>().is_err(){continue;}
        latest=fields.iter().zip(&words[2..]).map(|((_,name,scale,min,max),word)|{
            let value=word.parse::<f64>().ok().filter(|v|v.is_finite() && v>=min && v<=max).map(|v|v*scale);
            (name.to_string(),value)
        }).collect();
    }
    latest
}

async fn collect(gpu:&str,config:&CollectionConfig,host:&str)->Result<BTreeMap<String,Option<f64>>> {
    ensure!(!gpu.starts_with('-') && !gpu.contains(',') && !gpu.chars().any(char::is_whitespace),"invalid GPU selection");
    let mut fields=Vec::new();if config.basic{fields.extend_from_slice(BASIC);}if config.profiling{fields.extend_from_slice(PROF);}
    if fields.is_empty(){return Ok(BTreeMap::new());}
    let ids=fields.iter().map(|f|f.0.to_string()).collect::<Vec<_>>().join(",");
    let mut command=tokio::process::Command::new(std::env::var("CAPE_DCGMI_BIN").unwrap_or_else(|_|"dcgmi".into()));
    let delay=if config.profiling{"1000"}else{"100"};
    command.args(["dmon","--host",host,"-i",gpu,"-e",&ids,"-c","2","-d",delay]).env_remove("CUDA_VISIBLE_DEVICES").kill_on_drop(true);
    let output=tokio::time::timeout(Duration::from_secs(3),command.output()).await??;
    ensure!(output.status.success() && output.stdout.len()<=65536,"DCGM sample unavailable");
    let values=parse_rows(std::str::from_utf8(&output.stdout)?,&fields);
    ensure!(!values.is_empty(),"DCGM entity/fields unsupported");Ok(values)
}

async fn collect_loop(state:Shared,controller:String,root:PathBuf) {
    let client=reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(2)).redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let mut next_sample=Instant::now();
    let mut engine:Option<tokio::process::Child>=None;
    let mut engine_host=std::env::var("CAPE_DCGM_HOST").unwrap_or_else(|_|"127.0.0.1:5555".into());
    let mut engine_gpus=Vec::new();
    loop {
        let (id,incarnation)= {let s=state.lock().unwrap();(s.id.clone(),s.incarnation.clone())};
        let desired=async {
            let response=client.get(format!("{}/v1/metrics/service-config",controller.trim_end_matches('/'))).query(&[("service_id",&id)]).send().await?;
            ensure!(response.status().is_success(),"metrics control unavailable");
            let bytes=response.bytes().await?;ensure!(bytes.len()<65536,"control too large");
            Ok::<Value,anyhow::Error>(serde_json::from_slice(&bytes)?)
        }.await;
        if let Ok(desired)=desired {
            if desired["location"]["incarnation"].as_str().map(|i|i!=incarnation).unwrap_or(false) {
                // A different network namespace won the central lease. Never collect twice.
                state.lock().unwrap().shutdown.notify_one();
                return;
            }
            if let Ok(config)=CollectionConfig::parse(desired["config"].clone()) {
                let (gpus,scheduler)={let mut s=state.lock().unwrap();if s.config!=config{next_sample=Instant::now();}s.config=config.clone();s.version=desired["version"].as_u64().unwrap_or(0);
                    let now=s.origin.elapsed().as_secs_f64();let gpus=s.registry.gpu_uuids(None,None,now).unwrap_or_default();
                    s.snapshots.retain(&gpus.iter().cloned().collect());s.state=if config.enabled{"ready"}else{"disabled"}.into();(gpus,s.scheduler.clone())};
                if config.enabled && !gpus.is_empty() && Instant::now()>=next_sample && desired["lease_remaining_seconds"].as_f64().unwrap_or(0.)>5. {
                    next_sample=Instant::now()+Duration::from_secs_f64(config.interval_seconds);
                    if std::env::var_os("CAPE_DCGM_HOST").is_none() && std::env::var_os("CAPE_DCGMI_BIN").is_none() {
                        if engine.as_mut().map(|c|c.try_wait().ok().flatten().is_some()).unwrap_or(false){engine=None;engine_host="127.0.0.1:5555".into();}
                        if engine.is_some() && engine_gpus!=gpus {if let Some(mut child)=engine.take(){let _=child.kill().await;let _=child.wait().await;}engine_host="127.0.0.1:5555".into();}
                        let connected=if engine.is_some(){true}else{
                            let mut probe=tokio::process::Command::new("dcgmi");
                            probe.args(["discovery","--host",&engine_host,"-l"]).env_remove("CUDA_VISIBLE_DEVICES").kill_on_drop(true);
                            matches!(tokio::time::timeout(Duration::from_secs(3),probe.output()).await,Ok(Ok(output)) if output.status.success())
                        };
                        if !connected && engine.is_none() {
                            if let Ok(socket)=std::net::TcpListener::bind("127.0.0.1:0") {
                                let port=socket.local_addr().unwrap().port();drop(socket);
                                let home=root.join(format!("engine-{incarnation}"));let _=fs::DirBuilder::new().mode(0o700).create(&home);
                                let mut command=tokio::process::Command::new("nv-hostengine");
                                command.args(["--no-daemon","--port",&port.to_string(),"--bind-interface","127.0.0.1","--log-level","NONE","--log-filename","-"])
                                    .current_dir(&home).env("DCGM_HOME_DIR",&home).env("CUDA_VISIBLE_DEVICES",gpus.join(","))
                                    .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).kill_on_drop(true);
                                let owner=unsafe{nix::libc::getpid()};
                                unsafe {command.pre_exec(move||{
                                    if nix::libc::prctl(nix::libc::PR_SET_PDEATHSIG,nix::libc::SIGTERM)!=0 {
                                        return Err(std::io::Error::last_os_error());
                                    }
                                    if nix::libc::getppid()!=owner {return Err(std::io::Error::new(std::io::ErrorKind::Interrupted,"metrics owner exited"));}
                                    Ok(())
                                });}
                                if let Ok(child)=command.spawn(){engine=Some(child);engine_host=format!("127.0.0.1:{port}");engine_gpus=gpus.clone();tokio::time::sleep(Duration::from_millis(500)).await;}
                            }
                        }
                    }
                    let mut batch=crate::telemetry_sender::LatestBatch::default();
                    for chunk in gpus.chunks(8) {
                        let mut pending=tokio::task::JoinSet::new();
                        for gpu in chunk {let gpu=gpu.clone();let config=config.clone();let host=engine_host.clone();pending.spawn(async move {let result=collect(&gpu,&config,&host).await;(gpu,result)});}
                        while let Some(Ok((gpu,result)))=pending.join_next().await {
                        match result {
                            Ok(values)=>{
                                let mut exposition=String::new();
                                let sampled=crate::controller::now();
                                {let mut s=state.lock().unwrap();let now=s.origin.elapsed().as_secs_f64();let _=s.snapshots.put(&gpu,sampled,values.clone(),now);}
                                let labels=format!("scheduler_id={},service_id={},gpu_uuid={}",json!(scheduler),json!(id),json!(gpu));
                                for (name,value) in values {
                                    if let Some(value)=value {let prefix=if name.starts_with("gpu_"){"cape_"}else{"cape_gpu_"};exposition.push_str(&format!("{prefix}{name}{{{labels}}} {value} {}\n",(sampled*1000.) as i64));}
                                }
                                exposition.push_str(&format!("cape_gpu_sample_expires_at_seconds{{{labels}}} {} {}\n",sampled+config.interval_seconds*config.stale_periods as f64,(sampled*1000.) as i64));
                                batch.push(exposition);
                            },
                            Err(_)=>{state.lock().unwrap().state="unavailable".into();}
                        }
                        }
                    }
                    if let Some(receiver)=desired["receiver_url"].as_str() {
                        let exposition=batch.exposition();
                        let receiver=receiver.to_owned();let root=root.clone();
                        let binary=PathBuf::from(std::env::var("CAPE_TELEGRAF_BIN").unwrap_or_else(|_|"telegraf".into()));
                        let sent=tokio::task::spawn_blocking(move||crate::telemetry_sender::send(&receiver,&binary,&exposition,2.,&root)).await;
                        state.lock().unwrap().last_send=match sent {Ok(Ok(crate::telemetry_sender::SendOutcome::Sent))=>"sent",_=>"dropped"}.into();
                    }
                }
            }
        } else {state.lock().unwrap().state="unavailable".into();}
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

pub async fn serve(boot:Value)->Result<()> {
    let scheduler=boot["scheduler_id"].as_str().context("scheduler required")?.to_owned();
    let controller=boot["controller_url"].as_str().context("controller required")?.to_owned();controller_url(&controller)?;
    let id=host_service_id(&scheduler)?;let root=runtime_root(&id)?;
    match fs::DirBuilder::new().mode(0o1777).create(&root){Ok(())=>fs::set_permissions(&root,fs::Permissions::from_mode(0o1777))?,Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>(),Err(e)=>return Err(e.into())}
    let meta=fs::symlink_metadata(&root)?;ensure!(meta.is_dir() && meta.mode() & 0o1777 == 0o1777,"metrics runtime directory must be shared/sticky");
    let lock_path=root.join("service.lock");
    match OpenOptions::new().write(true).create_new(true).mode(0o444).custom_flags(nix::libc::O_NOFOLLOW).open(&lock_path){Ok(file)=>file.set_permissions(fs::Permissions::from_mode(0o444))?,Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>(),Err(e)=>return Err(e.into())}
    let lock=OpenOptions::new().read(true).custom_flags(nix::libc::O_NOFOLLOW).open(lock_path)?;
    let meta=lock.metadata()?;ensure!(meta.is_file() && meta.nlink()==1 && meta.mode() & 0o022==0,"invalid metrics singleton lock");
    nix::fcntl::flock(lock.as_raw_fd(),nix::fcntl::FlockArg::LockExclusiveNonblock)?;
    let socket=std::net::TcpListener::bind("0.0.0.0:0")?;socket.set_nonblocking(true)?;
    let host=std::env::var("CAPE_METRIC_ADVERTISE_HOST").unwrap_or(hostname()?);
    ensure!(!host.chars().any(|c|c.is_whitespace() || "/?#@".contains(c)),"invalid metric advertise host");
    let url=format!("http://{}:{}",host,socket.local_addr()?.port());let incarnation=uuid::Uuid::new_v4().to_string();
    let record=json!({"scheduler_id":scheduler,"service_id":id,"url":url,"incarnation":incarnation,"pid":std::process::id()});
    let path=root.join(format!("{incarnation}.json"));let mut file=OpenOptions::new().write(true).create_new(true).mode(0o644).open(&path)?;
    file.write_all(record.to_string().as_bytes())?;file.set_permissions(fs::Permissions::from_mode(0o644))?;file.sync_all()?;
    let shutdown=Arc::new(tokio::sync::Notify::new());
    let state=Arc::new(Mutex::new(Service {scheduler:scheduler.clone(),id:id.clone(),incarnation:incarnation.clone(),origin:Instant::now(),
        registry:RegistrationSet::new(&scheduler,&id,30.,4096)?,snapshots:LatestSnapshots::new(1024)?,config:CollectionConfig::default(),version:0,state:"pending".into(),last_send:"none".into(),shutdown:shutdown.clone()}));
    let collector=tokio::spawn(collect_loop(state.clone(),controller,root));
    println!("{record}");std::io::stdout().flush()?;
    let app=Router::new().route("/register",post(registration)).route("/snapshot",get(snapshot)).with_state(state);
    let result=axum::Server::from_tcp(socket)?.serve(app.into_make_service()).with_graceful_shutdown(async move {
        if let Ok(mut term)=tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            tokio::select!{_=term.recv()=>{},_=tokio::signal::ctrl_c()=>{},_=shutdown.notified()=>{}}
        }else{let _=tokio::signal::ctrl_c().await;}
    }).await;
    collector.abort();let _=fs::remove_file(path);drop(lock);result?;Ok(())
}

#[cfg(test)]mod tests {
    use super::*;
    #[test]fn dcgm_unavailable_is_not_zero_and_units_are_bytes(){let v=parse_rows("GPU 0 N/A 0 2 1 40 100\n",BASIC);assert_eq!(v["gpu_utilization_percent"],None);assert_eq!(v["memory_total_bytes"],Some(2097152.));assert_eq!(v["memory_utilization_percent"],Some(0.));}
    #[test]fn profiler_sentinels_are_missing(){let v=parse_rows("GPU 0 0.5 9223372036854775794 NaN 2\n",PROF);assert_eq!(v["sm_active_ratio"],Some(0.5));assert_eq!(v["sm_occupancy_ratio"],None);assert_eq!(v["tensor_active_ratio"],None);assert_eq!(v["dram_active_ratio"],None);}
}
