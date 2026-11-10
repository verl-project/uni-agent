// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::agent::{self, AgentConfig};
use crate::auth::Authority;
use crate::controller::Controller;
use crate::http;
use crate::journal::Journal;
use crate::instance::Instance;
use crate::credential_file::{prepare_registration, prepare_registration_for, read_registration};
use crate::runtime::RuntimeConfig;
use anyhow::{anyhow, bail, Context, Result};
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use reqwest::{Client, Method};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub async fn run() -> Result<i32> {
    let matches = parser().get_matches();
    let controller_admin = std::env::var("CAPE_ADMIN_TOKEN").ok().or_else(||std::env::var("CAPE_TOKEN").ok());
    // Credentials read from the environment must not reach workloads that inherit this process's environment.
    clear_credential_environment();
    match matches.subcommand() {
        Some(("__metric-service", _)) | Some(("__metric-link", _)) => {
            use std::io::Read;
            let mut input=String::new();std::io::stdin().take(65537).read_to_string(&mut input)?;
            if input.len()>65536{bail!("metrics bootstrap too large");}
            let boot:Value=serde_json::from_str(&input)?;
            if matches.subcommand_name()==Some("__metric-service"){crate::telemetry_service::serve(boot).await?;}
            else{crate::telemetry_link::run(boot).await?;}
            Ok(0)
        },
        Some(("run", args)) => run_workload(args).await,
        Some(("status", args)) => request_status("status", args).await,
        Some(("logs", args)) => request_status("logs", args).await,
        Some(("wait", args)) => request_status("wait", args).await,
        Some(("cancel", args)) => request_status("cancel", args).await,
        Some(("utilization", args)) => utilization(args).await,
        Some(("metric", args)) => {
            let (action, leaf) = args.subcommand().context("metric command required")?;
            if action != "history" {
                if let Some(local)=leaf.get_one::<String>("local_url") {
                    crate::telemetry_service::controller_url(local)?;
                    let mut query=Vec::new();
                    if let Some(node)=leaf.get_one::<String>("node_id"){query.push(("node_id",node.clone()));}
                    if action=="latest"{for key in ["metric","pool","gpu_uuid"]{if let Some(value)=leaf.get_one::<String>(key){query.push((key,value.clone()));}}}
                    println!("{}",request_json(Method::GET,local,"/snapshot",None,None,Some(query)).await?);
                    return Ok(0);
                }
            }
            let token = identity_cli_token(leaf)?;
            let mut query = Vec::new();
            if let Some(id) = leaf.get_one::<String>("node_id") { query.push(("node_id", id.clone())); }
            if action != "status" {
                query.push(("mode", action.into()));
                for key in ["metric","pool","gpu_uuid"] { if let Some(value) = leaf.get_one::<String>(key) { query.push((key,value.clone())); } }
                if action == "history" {
                    for key in ["start","end","step"] { query.push((key,leaf.get_one::<f64>(key).unwrap().to_string())); }
                }
            }
            let path = if action == "status" {"/v1/metrics/status"} else {"/v1/metrics/query"};
            println!("{}", request_json(Method::GET, &controller_url(leaf), path, token.as_deref(), None, Some(query)).await?);
            Ok(0)
        },
        Some(("workspace", args)) => workspace(args).await,
        Some(("session", args)) => session_command(args).await,
        Some(("user", args)) => user_command(args).await,
        Some(("node", args)) => match args.subcommand() {
            Some(("enroll", args)) => node_enroll(args).await,
            Some(("edit", args)) => {
                let token = identity_cli_token(args)?;
                let mut changes = json!({});
                if let Some(cpu) = args.get_one::<f64>("cpu_cores") { changes["cpu_cores"] = json!(cpu); }
                if let Some(memory) = args.get_one::<f64>("memory_gb") { changes["memory_gb"] = json!(memory); }
                if args.get_flag("no_gpus") { changes["gpu_uuids"] = json!([]); }
                else if args.contains_id("gpu_uuid") { changes["gpu_uuids"] = json!(values(args,"gpu_uuid")); }
                println!("{}", request_json(Method::POST,&controller_url(args),&format!("/v1/identity/nodes/{}/action",string(args,"node_id")),
                    token.as_deref(),Some(json!({"action":"edit","changes":changes})),None).await?);
                Ok(0)
            },
            Some((action @ ("drain" | "exit"), args)) => {
                let token = identity_cli_token(args)?;
                let force = action == "drain" && args.get_flag("force");
                println!("{}", request_json(Method::POST, &controller_url(args),
                    &format!("/v1/identity/nodes/{}/action", string(args, "node_id")), token.as_deref(),
                    Some(json!({"action":action,"force":force})), None).await?);
                Ok(0)
            },
            _ => Ok(2),
        },
        Some(("controller", args)) => match args.subcommand() {
            Some(("init", args)) => {
                let directory = PathBuf::from(string(args, "state_dir"));
                if std::fs::symlink_metadata(&directory).is_ok() { bail!("instance directory already exists"); }
                let output = PathBuf::from(string(args, "token_out"));
                let envelope = prepare_registration(&output, "local:controller-init", Some(&string(args, "user")), "bootstrap", None)?;
                let scheduler_id = Instance::initialize(&directory, &string(args, "user"), envelope["secret"].as_str().unwrap())?;
                let mut result = json!({"scheduler_id":scheduler_id,"state_dir":directory,"token_file":output});
                if args.get_flag("metrics") {
                    let mut store = crate::identity::IdentityStore::open(&directory.join("state.sqlite3"))?;
                    crate::telemetry_control::set_config(&mut store,envelope["secret"].as_str().unwrap(),&json!({"enabled":true}),None,None,crate::controller::now())?;
                    let mut supervisor = crate::telemetry_supervisor::PrometheusSupervisor::new(&directory,&scheduler_id,std::env::var("CAPE_PROMETHEUS_BIN").unwrap_or_else(|_|"prometheus".into()))?;
                    result["metrics"] = supervisor.start().await.unwrap_or_else(|_|json!({"state":"unavailable","reason":"prometheus_start_failed"}));
                }
                println!("{}", result);
                Ok(0)
            },
            Some(("start", args)) => controller_start(args, controller_admin).await,
            _ => Ok(2),
        },
        Some(("admin", args)) => admin(args).await,
        _ => Ok(2),
    }
}

fn clear_credential_environment() {
    for name in [
        "CAPE_TOKEN", "CAPE_JOIN_TOKEN", "CAPE_ADMIN_TOKEN", "CAPE_PROBE_NODE_TOKEN", "CAPE_NODE_TOKEN",
        "CAPE_USER_TOKENS_INLINE", "CAPE_SUBMIT_TOKEN", "CAPE_SUBMIT_USER_TOKENS_INLINE",
        "CAPE_STATUS_TOKEN", "CAPE_RETIRE_TOKEN", "CAPE_CONCURRENCY_STATUS_TOKEN",
    ] {
        std::env::remove_var(name);
    }
}

fn identity_connection(command: Command) -> Command {
    command.arg(url_arg()).arg(auth_token_arg()).arg(Arg::new("token_file").long("token-file").conflicts_with("token"))
}

fn user_parser() -> Command {
    Command::new("user").subcommand_required(true)
        .subcommand(identity_connection(Command::new("whoami")))
        .subcommand(identity_connection(Command::new("permissions")))
        .subcommand(identity_connection(Command::new("pools")))
        .subcommand(identity_connection(Command::new("nodes")).arg(Arg::new("pool").long("pool")))
        .subcommand(Command::new("token").subcommand_required(true)
            .subcommand(identity_connection(Command::new("create"))
                .arg(Arg::new("user").long("user"))
                .arg(Arg::new("name").long("name").default_value("default"))
                .arg(Arg::new("expires_at").long("expires-at").value_parser(value_parser!(f64)))
                .arg(Arg::new("token_out").long("token-out").required(true)))
            .subcommand(identity_connection(Command::new("resume")).arg(Arg::new("token_out").long("token-out").required(true)))
            .subcommand(identity_connection(Command::new("list")))
            .subcommand(identity_connection(Command::new("revoke")).arg(Arg::new("token_id").required(true))))
}

fn identity_cli_token(args: &ArgMatches) -> Result<Option<String>> {
    if let Some(path) = option(args, "token_file") {
        Ok(Some(read_registration(&PathBuf::from(path))?["secret"].as_str().context("invalid credential")?.to_owned()))
    } else { Ok(option(args, "token")) }
}

async fn identity_register_http(url: &str, body: &Value, token: Option<&str>) -> Result<(u16, Value)> {
    let client = Client::builder().timeout(std::time::Duration::from_secs(60)).build()?;
    let mut request = client.post(format!("{}/v1/identity/tokens", url.trim_end_matches('/'))).json(body);
    if let Some(token) = token { request = request.bearer_auth(token); }
    let response = request.send().await?;
    let status = response.status().as_u16();
    Ok((status, response.json().await?))
}

async fn user_command(args: &ArgMatches) -> Result<i32> {
    let (verb, args) = args.subcommand().context("user command required")?;
    let (verb, args) = if verb == "token" { args.subcommand().context("token command required")? } else { (verb, args) };
    let token = identity_cli_token(args)?;
    if matches!(verb, "create" | "resume") {
        let resume = verb == "resume";
        let path = PathBuf::from(string(args, "token_out"));
        let envelope = if resume { read_registration(&path)? } else {
            let actor = if token.is_some() { Some(request_json(Method::GET,&controller_url(args),"/v1/identity/me",token.as_deref(),None,None).await?) } else {None};
            let target = actor.as_ref().map(|a|a["user_id"].as_str().context("invalid identity response")).transpose()?;
            prepare_registration_for(&path, controller_url(args).trim_end_matches('/'), option(args, "user").as_deref(), &string(args, "name"), args.get_one::<f64>("expires_at").copied(),
                target)?
        };
        let url = envelope["controller_url"].as_str().context("invalid controller URL")?;
        if !url.starts_with("http://") && !url.starts_with("https://") { bail!("this credential is not a resumable user registration"); }
        if let Some(explicit) = option(args, "controller_url") { if explicit.trim_end_matches('/') != url { bail!("registration controller differs from saved operation"); } }
        let body = &envelope["registration"];
        let mut outcome = identity_register_http(url, body, if resume { envelope["secret"].as_str() } else { token.as_deref() }).await;
        if resume && matches!(&outcome, Ok((403, _))) && (!body["user_id"].is_null() || !body["username"].is_null()) {
            outcome = identity_register_http(url, body, token.as_deref()).await;
        }
        let (code, result) = match outcome {
            Ok((status, mut result)) if (200..300).contains(&status) && result["operation_id"] == body["operation_id"]
                && result["status"] == "registered" && result["token_id"].is_string() && result["user_id"].is_string() => {
                    result["token_file"] = json!(path); (0, result)
                },
            Ok((status, _)) if (400..500).contains(&status) && !resume => (1, json!({"outcome":"rejected"})),
            _ => (3, json!({"outcome":"unknown"})),
        };
        if code == 0 { println!("{result}"); } else {
            eprintln!("{}", json!({"outcome":result["outcome"],"operation_id":body["operation_id"],"token_file":path,"recovery":"user token resume"}));
        }
        return Ok(code);
    }
    let (method, path) = match verb {
        "whoami" | "permissions" => (Method::GET, "/v1/identity/me".to_owned()),
        "pools" => (Method::GET, "/v1/identity/pools".to_owned()),
        "nodes" => (Method::GET, "/v1/identity/nodes".to_owned()),
        "list" => (Method::GET, "/v1/identity/tokens".to_owned()),
        "revoke" => (Method::POST, format!("/v1/identity/tokens/{}/revoke", string(args, "token_id"))),
        _ => bail!("unknown user command"),
    };
    let query = if verb == "nodes" { option(args, "pool").map(|pool|vec![("pool", pool)]) } else { None };
    println!("{}", request_json(method, &controller_url(args), &path, token.as_deref(), None, query).await?);
    Ok(0)
}

fn parser() -> Command { credential_file_options(base_parser()) }

fn credential_file_options(mut command: Command) -> Command {
    for child in command.get_subcommands_mut() {
        *child = credential_file_options(child.clone());
    }
    if command.get_arguments().any(|arg| arg.get_id().as_str() == "token") {
        if !command.get_arguments().any(|arg| arg.get_id().as_str() == "token_file") {
            command = command.arg(Arg::new("token_file").long("token-file"));
        }
        command = command.mut_arg("token", |arg| arg.conflicts_with("token_file"));
    }
    command
}

fn base_parser() -> Command {
    Command::new("rscape")
        .subcommand(Command::new("__metric-service").hide(true))
        .subcommand(Command::new("__metric-link").hide(true))
        .version(env!("CARGO_PKG_VERSION"))
        .subcommand_required(true)
        .subcommand(run_command())
        .subcommand(user_parser())
        .subcommand(request_command("status"))
        .subcommand(request_command("logs"))
        .subcommand(wait_command())
        .subcommand(cancel_command())
        .subcommand(
            Command::new("utilization")
                .arg(url_arg())
                .arg(auth_token_arg())
                .arg(Arg::new("pool").long("pool").required(true)),
        )
        .subcommand(
            Command::new("node")
                .subcommand_required(true)
                .subcommand(identity_connection(Command::new("edit")).arg(Arg::new("node_id").required(true))
                    .arg(Arg::new("cpu_cores").long("cpu-cores").value_parser(value_parser!(f64)))
                    .arg(Arg::new("memory_gb").long("memory-gb").value_parser(value_parser!(f64)))
                    .arg(Arg::new("gpu_uuid").long("gpu-uuid").action(ArgAction::Append).conflicts_with("no_gpus"))
                    .arg(Arg::new("no_gpus").long("no-gpus").action(ArgAction::SetTrue)))
                .subcommand(identity_connection(Command::new("drain")).arg(Arg::new("node_id").required(true))
                    .arg(Arg::new("force").long("force").action(ArgAction::SetTrue)))
                .subcommand(identity_connection(Command::new("exit")).arg(Arg::new("node_id").required(true)))
                .subcommand(node_enroll_command()
                    .arg(Arg::new("node_token_out").long("node-token-out").required(true))
                    .arg(Arg::new("token_file").long("token-file").conflicts_with("token"))
                    .arg(Arg::new("resume").long("resume").action(ArgAction::SetTrue))),
        )
        .subcommand(
            Command::new("controller")
                .subcommand_required(true)
                .subcommand(Command::new("init")
                    .arg(Arg::new("metrics").long("metrics").action(ArgAction::SetTrue))
                    .arg(Arg::new("state_dir").long("state-dir").required(true))
                    .arg(Arg::new("user").long("user").required(true))
                    .arg(Arg::new("token_out").long("token-out").required(true)))
                .subcommand(controller_start_command()),
        )
        .subcommand(admin_command())
        .subcommand(metric_command())
        .subcommand(Command::new("session").subcommand_required(true)
            .subcommand(identity_connection(Command::new("list")))
            .subcommand(identity_connection(Command::new("status")).arg(Arg::new("session_id").required(true)))
            .subcommand(identity_connection(Command::new("suspend")).arg(Arg::new("execution_id").required(true))
                .arg(Arg::new("force").long("force").action(ArgAction::SetTrue)))
            .subcommand(identity_connection(Command::new("purge")).arg(Arg::new("execution_id").required(true))))
        .subcommand(Command::new("workspace").arg(Arg::new("action").required(true).value_parser(["status", "set", "release", "check", "delete", "grow", "reclaim", "shrink", "recover"]))
            .arg(Arg::new("auto_reclaim").long("auto-reclaim").value_parser(["off", "inherit"]))
            .arg(Arg::new("workspace_id").required(true)).arg(Arg::new("size_gib").long("size-gib").value_parser(clap::value_parser!(u64)))
            .arg(Arg::new("operation_id").long("operation-id")).arg(url_arg()).arg(auth_token_arg()))
}

async fn workspace(args: &ArgMatches) -> Result<i32> {
    let action = string(args, "action");
    let mut body=json!({});
    if action == "set" {
        let policy=option(args,"auto_reclaim").context("set requires --auto-reclaim")?;
        if option(args,"operation_id").is_some() {bail!("set does not accept --operation-id");}
        body["auto_reclaim"]=json!(policy);
    } else if option(args,"auto_reclaim").is_some() {bail!("--auto-reclaim applies only to set");}
    let size=args.get_one::<u64>("size_gib").copied();
    if matches!(action.as_str(),"grow"|"shrink") {
        let bytes=size.and_then(|v|v.checked_mul(1024_u64.pow(3))).filter(|v|*v>0 && *v<=i64::MAX as u64).context("grow/shrink require a positive bounded --size-gib")?;
        body["target_bytes"]=json!(bytes);
    } else if size.is_some() {bail!("--size-gib applies only to grow/shrink");}
    if let Some(id)=option(args,"operation_id") {body["operation_id"]=json!(id);}
    let mut path = format!("/v1/workspaces/{}", string(args, "workspace_id"));
    if action != "status" { path.push('/'); path.push_str(&action); }
    let credential = identity_cli_token(args)?;
    let value = request_json(if action == "status" { Method::GET } else { Method::POST },
        &controller_url(args), &path, credential.as_deref(), if body.as_object().unwrap().is_empty(){None}else{Some(body)}, None).await?;
    println!("{}", serde_json::to_string(&value)?);
    Ok(0)
}

fn url_arg() -> Arg {
    Arg::new("controller_url").long("controller-url")
}
fn token_arg() -> Arg {
    Arg::new("token").long("token")
}
/// A caller's credential. `--token` wins; otherwise CAPE_TOKEN, which keeps the secret out of the world-readable
/// process command line. `admin add-*` keep the plain argument: there `--token` is the value being granted.
fn auth_token_arg() -> Arg {
    token_arg().env("CAPE_TOKEN").hide_env_values(true)
}
fn request_command(name: &'static str) -> Command {
    Command::new(name)
        .arg(Arg::new("request_id").required(true))
        .arg(url_arg())
        .arg(auth_token_arg())
}
fn wait_command() -> Command {
    request_command("wait")
        .arg(
            Arg::new("timeout")
                .long("timeout")
                .default_value("30")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("running_timeout")
                .long("running-timeout")
                .value_parser(value_parser!(f64)),
        )
}
fn cancel_command() -> Command {
    request_command("cancel").arg(Arg::new("reason").long("reason").default_value("cancelled"))
}
fn run_command() -> Command {
    Command::new("run")
        .arg(url_arg())
        .arg(auth_token_arg())
        .arg(Arg::new("pool").long("pool").required(true))
        .arg(Arg::new("user").long("user").default_value(""))
        .arg(Arg::new("session_id").long("session-id"))
        .arg(Arg::new("image").long("image").conflicts_with("profile"))
        .arg(Arg::new("profile").long("profile"))
        .arg(
            Arg::new("gpus")
                .long("gpus")
                .default_value("0")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("cpu_cores")
                .long("cpu-cores")
                .default_value("1")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("memory_gb")
                .long("memory-gb")
                .default_value("1")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("max_duration_seconds")
                .long("max-duration-seconds")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("command_timeout_seconds")
                .long("command-timeout-seconds")
                .value_parser(value_parser!(f64)),
        )
        .arg(Arg::new("runtime_adapter").long("runtime-adapter"))
        .arg(
            Arg::new("hard_gpu_enforcement")
                .long("hard-gpu-enforcement")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .action(ArgAction::Append)
                .value_parser(value_parser!(u16)),
        )
        .arg(Arg::new("locality").long("locality"))
        .arg(
            Arg::new("isolation_policy")
                .long("isolation-policy")
                .default_value("default"),
        )
        .arg(Arg::new("cwd").long("cwd"))
        .arg(Arg::new("workspace_id").long("workspace-id"))
        .arg(Arg::new("workspace_target").long("workspace-target").default_value("/workspace"))
        .arg(Arg::new("env").long("env").action(ArgAction::Append))
        .arg(Arg::new("bind").long("bind").action(ArgAction::Append))
        .arg(Arg::new("log_path").long("log-path"))
        .arg(
            Arg::new("artifact_path")
                .long("artifact-path")
                .action(ArgAction::Append),
        )
        .arg(
            Arg::new("suspend_on_failure")
                .long("destroy-session-on-failure")
                .action(ArgAction::SetTrue),
        )
        .arg(Arg::new("wait").long("wait").action(ArgAction::SetTrue))
        .arg(
            Arg::new("timeout")
                .long("timeout")
                .default_value("30")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("running_timeout")
                .long("running-timeout")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("cancel_on_timeout")
                .long("cancel-on-timeout")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("cancel_reason")
                .long("cancel-reason")
                .default_value("wait_timeout"),
        )
        .arg(
            Arg::new("workload_command")
                .num_args(1..)
                .required(true)
                .trailing_var_arg(true)
                .allow_hyphen_values(true),
        )
}

fn controller_start_command() -> Command {
    Command::new("start")
        .arg(Arg::new("metrics").long("metrics").action(ArgAction::SetTrue))
        .arg(Arg::new("state_dir").long("state-dir").required(true))
        .arg(Arg::new("token").long("token"))
        .arg(Arg::new("token_file").long("token-file").conflicts_with("token"))
        .arg(Arg::new("host").long("host").default_value("127.0.0.1"))
        .arg(
            Arg::new("port")
                .long("port")
                .default_value("8765")
                .value_parser(value_parser!(u16)),
        )
        .arg(
            Arg::new("schedule_interval")
                .long("schedule-interval")
                .default_value("0.05"),
        )
        .arg(
            Arg::new("heartbeat_timeout_seconds")
                .long("heartbeat-timeout-seconds")
                .default_value("30")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("lost_timeout_seconds")
                .long("lost-timeout-seconds")
                .default_value("90")
                .value_parser(value_parser!(f64)),
        )
        .arg(Arg::new("profiles_file").long("profiles-file"))
        .arg(
            Arg::new("max_active_gpu_sessions_per_user")
                .long("max-active-gpu-sessions-per-user")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("max_queued_requests_per_user")
                .long("max-queued-requests-per-user")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("max_active_sessions_per_pool")
                .long("max-active-sessions-per-pool")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("idle_session_timeout_seconds")
                .long("idle-session-timeout-seconds")
                .default_value("3600")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("reconcile_missing_grace_seconds")
                .long("reconcile-missing-grace-seconds")
                .default_value("5")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("reconcile_missing_heartbeats")
                .long("reconcile-missing-heartbeats")
                .default_value("2")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("gpu_contention_policy")
                .long("gpu-contention-policy")
                .default_value("barrier-ratio")
                .value_parser(["barrier-ratio", "ratio-only", "exclusive-first"]),
        )
        .arg(
            Arg::new("gpu_shared_ratio")
                .long("gpu-shared-ratio")
                .default_value("4")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("gpu_share_override")
                .long("gpu-share-override")
                .action(ArgAction::Append),
        )
}

fn node_enroll_command() -> Command {
    Command::new("enroll")
        .arg(Arg::new("metrics").long("metrics").action(ArgAction::SetTrue))
        .arg(url_arg().required(true))
        .arg(Arg::new("node_id").long("node-id"))
        .arg(Arg::new("pool").long("pool").required(true))
        .arg(
            Arg::new("pool_mode")
                .long("pool-mode")
                .default_value("exclusive"),
        )
        .arg(
            Arg::new("gpu_slots")
                .long("gpu-slots")
                .default_value("1")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("pool_placement")
                .long("pool-placement")
                .default_value("pack"),
        )
        .arg(auth_token_arg())
        .arg(
            Arg::new("cpu_cores")
                .long("cpu-cores")
                .default_value("1")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("memory_gb")
                .long("memory-gb")
                .default_value("1")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("gpu_uuid")
                .long("gpu-uuid")
                .action(ArgAction::Append),
        )
        .arg(Arg::new("allocation_id").long("allocation-id"))
        .arg(
            Arg::new("allocation_deadline")
                .long("allocation-deadline")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("drain_before_sec")
                .long("drain-before-sec")
                .default_value("0")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("cleanup_buffer_sec")
                .long("cleanup-buffer-sec")
                .default_value("0")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("runtime_adapter")
                .long("runtime-adapter")
                .value_parser(["apptainer", "bubblewrap", "podman", "local-process"])
                .help("Repeatable runtime filter; default: auto-probe apptainer, bubblewrap and podman; local-process is explicit-only")
                .action(ArgAction::Append),
        )
        .arg(
            Arg::new("runtime_cache_config")
                .long("runtime-cache-config")
                .default_value("default"),
        )
        .arg(Arg::new("runtime_cache").long("runtime-cache"))
        .arg(Arg::new("runtime_tmp").long("runtime-tmp"))
        .arg(Arg::new("runtime_dir").long("runtime-dir"))
        .arg(Arg::new("execution_state_dir").long("execution-state-dir"))
        .arg(Arg::new("execution_state_min_free_bytes").long("execution-state-min-free-bytes").value_parser(value_parser!(u64)).default_value("536870912"))
        .arg(Arg::new("execution_state_max_bytes").long("execution-state-max-bytes").value_parser(value_parser!(u64)).default_value("0"))
        .arg(
            Arg::new("apptainer_binary")
                .long("apptainer-binary")
                .default_value("apptainer"),
        )
        .arg(Arg::new("apptainer_cache").long("apptainer-cache"))
        .arg(Arg::new("apptainer_tmp").long("apptainer-tmp"))
        .arg(Arg::new("apptainer_runtime_dir").long("apptainer-runtime-dir"))
        .arg(
            Arg::new("bubblewrap_binary")
                .long("bubblewrap-binary")
                .default_value("bwrap"),
        )
        .arg(Arg::new("bubblewrap_cache").long("bubblewrap-cache"))
        .arg(Arg::new("bubblewrap_tmp").long("bubblewrap-tmp"))
        .arg(Arg::new("bubblewrap_runtime_dir").long("bubblewrap-runtime-dir"))
        .arg(
            Arg::new("bubblewrap_system_path")
                .long("bubblewrap-system-path")
                .action(ArgAction::Append),
        )
        .arg(
            Arg::new("podman_binary")
                .long("podman-binary")
                .default_value("podman"),
        )
        .arg(
            Arg::new("bubblewrap_read_only_path")
                .long("bubblewrap-read-only-path")
                .help("Optional virtual-kernel subpath bound read-only; roots and symlink escapes are refused; absent paths are skipped")
                .action(ArgAction::Append),
        )
        .arg(Arg::new("podman_graphroot").long("podman-graphroot"))
        .arg(Arg::new("podman_runroot").long("podman-runroot"))
        .arg(Arg::new("podman_tmpdir").long("podman-tmpdir"))
        .arg(
            Arg::new("cached_image")
                .long("cached-image")
                .action(ArgAction::Append),
        )
        .arg(Arg::new("log_dir").long("log-dir"))
        .arg(Arg::new("state_path").long("state-path"))
        .arg(
            Arg::new("heartbeat_interval")
                .long("heartbeat-interval")
                .default_value("1")
                .value_parser(value_parser!(f64)),
        )
        .arg(
            Arg::new("stay_after_removal")
                .long("stay-after-removal")
                .action(ArgAction::SetTrue),
        )
        .arg(Arg::new("once").long("once").action(ArgAction::SetTrue))
        .arg(Arg::new("apptainer_probe_image").long("apptainer-probe-image").default_value("docker://docker.io/library/ubuntu:24.04"))
        .arg(Arg::new("podman_probe_image").long("podman-probe-image").default_value("docker://docker.io/library/ubuntu:24.04"))
        .arg(
            Arg::new("apptainer_unsquash")
                .long("apptainer-unsquash")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no_contain")
                .long("no-contain")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no_warm_sessions")
                .long("no-warm-sessions")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log_redact_env_key")
                .long("log-redact-env-key")
                .action(ArgAction::Append),
        )
}

fn workspace_admin_command(name: &'static str, update: bool) -> Command {
    let mut command = Command::new(name).arg(url_arg()).arg(auth_token_arg())
        .arg(Arg::new("pool").long("pool").required(true))
        .arg(Arg::new("node_id").long("node-id"));
    if update {
        command = command.arg(Arg::new("image_root").long("image-root"));
        for (name, flag) in [("image_size_gib", "image-size-gib"), ("max_image_size_gib", "max-image-size-gib"),
                             ("bytes_per_inode", "bytes-per-inode"), ("mount_timeout_seconds", "mount-timeout-seconds")] {
            command = command.arg(Arg::new(name).long(flag).value_parser(value_parser!(u64)));
        }
        for (name,flag) in [("auto_grow","auto-grow"),("reclaim_on_release","reclaim-on-release")] {
            command=command.arg(Arg::new(name).long(flag).value_parser(["on","off"]));
        }
        for (name,flag) in [("grow_threshold_percent","grow-threshold-percent"),("grow_target_percent","grow-target-percent"),
            ("grow_min_mib","grow-min-mib"),("grow_cooldown_seconds","grow-cooldown-seconds"),
            ("reclaim_min_mib","reclaim-min-mib"),("reclaim_cooldown_seconds","reclaim-cooldown-seconds"),
            ("reclaim_idle_seconds","reclaim-idle-seconds"),("reclaim_start_interval_seconds","reclaim-start-interval-seconds"),
            ("reclaim_max_candidates","reclaim-max-candidates"),("reclaim_retry_backoff_seconds","reclaim-retry-backoff-seconds"),
            ("reclaim_low_yield_backoff_seconds","reclaim-low-yield-backoff-seconds"),("reclaim_nice","reclaim-nice"),
            ("maintenance_timeout_seconds","maintenance-timeout-seconds"),("backing_reserve_mib","backing-reserve-mib")] {
            command=command.arg(Arg::new(name).long(flag).value_parser(value_parser!(u64)));
        }
    }
    command
}

fn metric_command() -> Command {
    let mut command = Command::new("metric").subcommand_required(true);
    for action in ["status","latest","history"] {
        let mut leaf = identity_connection(Command::new(action)).arg(Arg::new("node_id").long("node-id"));
        if action != "history"{leaf=leaf.arg(Arg::new("local_url").long("local-url"));}
        if action != "status" {
            leaf = leaf.arg(Arg::new("metric").long("metric").required(true))
                .arg(Arg::new("pool").long("pool")).arg(Arg::new("gpu_uuid").long("gpu-uuid"));
        }
        if action == "history" {
            for key in ["start","end","step"] { leaf = leaf.arg(Arg::new(key).long(key).value_parser(value_parser!(f64)).required(true)); }
        }
        command = command.subcommand(leaf);
    }
    command
}

fn admin_command() -> Command {
    Command::new("admin")
        .subcommand(Command::new("metric").subcommand_required(true)
            .subcommand(identity_connection(Command::new("get")).arg(Arg::new("node_id").long("node-id")))
            .subcommand(identity_connection(Command::new("set"))
                .arg(Arg::new("node_id").long("node-id"))
                .arg(Arg::new("expected_version").long("expected-version").value_parser(value_parser!(u64)))
                .arg(Arg::new("interval_seconds").long("interval-seconds").value_parser(value_parser!(f64)))
                .arg(Arg::new("stale_periods").long("stale-periods").value_parser(value_parser!(u32)))
                .arg(Arg::new("enabled").long("enabled").value_parser(["on","off"]))
                .arg(Arg::new("basic").long("basic").value_parser(["on","off"]))
                .arg(Arg::new("profiling").long("profiling").value_parser(["on","off"]))))
        .subcommand(Command::new("node").subcommand_required(true)
            .subcommand(identity_connection(Command::new("purge").about("Retire a LOST node from pool inventory; preserve data/history and GPU quarantine. Does not kill remote processes."))
                .arg(Arg::new("node_id").required(true))
                .arg(Arg::new("reason").long("reason").required(true)))
            .subcommand(identity_connection(Command::new("release-quarantine").about("Release only this purged node's GPU quarantine source after an administrator confirms its processes stopped. Does not restore old credentials or certify workspace cleanup."))
                .arg(Arg::new("node_id").required(true))
                .arg(Arg::new("reason").long("reason").required(true))
                .arg(Arg::new("confirm_stopped").long("confirm-stopped").action(ArgAction::SetTrue))))
        .subcommand(identity_connection(Command::new("audit")).arg(Arg::new("limit").long("limit").value_parser(value_parser!(usize)).default_value("100")))
        .subcommand(Command::new("pool").subcommand_required(true)
            .subcommand(identity_connection(Command::new("create")).arg(Arg::new("name").required(true))
                .arg(Arg::new("max_nodes").long("max-nodes").value_parser(value_parser!(u64)))
                .arg(Arg::new("pool_mode").long("pool-mode").value_parser(["exclusive","shared"]).default_value("exclusive"))
                .arg(Arg::new("gpu_slots").long("gpu-slots").value_parser(value_parser!(u64)).default_value("1"))
                .arg(Arg::new("placement").long("placement").value_parser(["pack","spread"]).default_value("pack")))
            .subcommand(identity_connection(Command::new("edit")).arg(Arg::new("pool").required(true))
                .arg(Arg::new("name").long("name"))
                .arg(Arg::new("max_nodes").long("max-nodes").value_parser(value_parser!(u64)).conflicts_with("unlimited_nodes"))
                .arg(Arg::new("unlimited_nodes").long("unlimited-nodes").action(ArgAction::SetTrue))
                .arg(Arg::new("pool_mode").long("pool-mode").value_parser(["exclusive","shared"]))
                .arg(Arg::new("gpu_slots").long("gpu-slots").value_parser(value_parser!(u64)))
                .arg(Arg::new("placement").long("placement").value_parser(["pack","spread"])))
            .subcommand(identity_connection(Command::new("drain")).arg(Arg::new("pool").required(true))
                .arg(Arg::new("force").long("force").action(ArgAction::SetTrue)))
            .subcommand(identity_connection(Command::new("remove")).arg(Arg::new("pool").required(true)))
            .subcommand(Command::new("group").subcommand_required(true)
                .subcommand(identity_connection(Command::new("list")))
                .subcommand(identity_connection(Command::new("set"))
                    .arg(Arg::new("name").required(true))
                    .arg(Arg::new("pool").long("pool").action(ArgAction::Append))
                    .arg(Arg::new("all_pools").long("all-pools").action(ArgAction::SetTrue)))
                .subcommand(identity_connection(Command::new("remove")).arg(Arg::new("name").required(true)))))
        .subcommand(Command::new("user").subcommand_required(true)
            .subcommand(identity_connection(Command::new("list")))
            .subcommand(identity_connection(Command::new("edit"))
                .arg(Arg::new("user_id").required(true))
                .arg(Arg::new("group").long("group").action(ArgAction::Append).conflicts_with("clear_groups"))
                .arg(Arg::new("clear_groups").long("clear-groups").action(ArgAction::SetTrue))
                .arg(Arg::new("disable").long("disable").action(ArgAction::SetTrue).conflicts_with("enable"))
                .arg(Arg::new("enable").long("enable").action(ArgAction::SetTrue)))
            .subcommand(identity_connection(Command::new("remove")).arg(Arg::new("user_id").required(true))))
        .subcommand_required(true)
        .subcommand(workspace_admin_command("get-workspace-config", false))
        .subcommand(workspace_admin_command("set-workspace-config", true))

}

async fn session_command(args: &ArgMatches) -> Result<i32> {
    let (action,args)=args.subcommand().context("session action required")?;
    let url=controller_url(args);let credential=identity_cli_token(args)?;
    let result=if matches!(action,"list"|"status") {
        let query=if action=="status" {Some(vec![("session_id",string(args,"session_id"))])} else {None};
        request_json(Method::GET,&url,"/v1/sessions",credential.as_deref(),None,query).await?
    } else {
        request_json(Method::POST,&url,&format!("/v1/executions/{}/{}",string(args,"execution_id"),action),
            credential.as_deref(),Some(json!({"force":action=="suspend"&&args.get_flag("force")})),None).await?
    };
    println!("{}",serde_json::to_string(&result)?);Ok(0)
}

async fn run_workload(args: &ArgMatches) -> Result<i32> {
    let url = controller_url(args);
    let credential = identity_cli_token(args)?;
    let token = credential.as_deref();
    let identity = request_json(Method::GET, &url, "/v1/identity/me", token, None, None).await?;
    let selection = crate::session_identity::Selection::new(&identity, args.get_one::<String>("session_id").map(String::as_str), None, None)?;
    let mut env: BTreeMap<String, String> = BTreeMap::new();
    for item in values(args, "env") {
        let (key, value) = item
            .split_once('=')
            .ok_or_else(|| anyhow!("env entries must be KEY=VALUE"))?;
        env.insert(key.into(), value.into());
    }
    let mounts: Vec<Value> = values(args, "bind")
        .into_iter()
        .map(|item| parse_bind(&item))
        .collect::<Result<_>>()?;
    let command = values(args, "workload_command");
    let payload = json!({"pool":string(args,"pool"),"user_id":string(args,"user"),"session_id":selection.session_id,"command":command,"cwd":option(args,"cwd"),"env":env,"image":option(args,"image"),"profile":option(args,"profile"),"keep_session_on_failure":!args.get_flag("suspend_on_failure"),"resources":{"gpus":*args.get_one::<usize>("gpus").unwrap(),"cpu_cores":*args.get_one::<f64>("cpu_cores").unwrap(),"memory_gb":*args.get_one::<f64>("memory_gb").unwrap(),"max_duration_seconds":args.get_one::<f64>("max_duration_seconds").copied().unwrap_or(3600.0),"command_timeout_seconds":args.get_one::<f64>("command_timeout_seconds").copied(),"runtime_adapter":option(args,"runtime_adapter"),"hard_gpu_enforcement":args.get_flag("hard_gpu_enforcement"),"ports":args.get_many::<u16>("port").map(|v|v.copied().collect::<Vec<_>>()).unwrap_or_default(),"locality":option(args,"locality"),"isolation_policy":string(args,"isolation_policy")},"mounts":mounts,"log_path":option(args,"log_path"),"artifact_paths":values(args,"artifact_path"),"workspace":{"id":option(args,"workspace_id"),"target":string(args,"workspace_target")}});
    let mut status =
        request_json(Method::POST, &url, "/v1/submit", token, Some(payload), None).await?;
    selection.accepted(&status)?;
    eprintln!("session_id={}", selection.session_id);
    drop(selection);
    if !args.get_flag("wait") {
        println!("{}", status["request_id"].as_str().unwrap_or(""));
        return Ok(0);
    }
    let request_id = status["request_id"]
        .as_str()
        .context("missing request_id")?
        .to_string();
    let mut query = vec![(
        "timeout",
        args.get_one::<f64>("timeout").unwrap().to_string(),
    )];
    if let Some(value) = args.get_one::<f64>("running_timeout") {
        query.push(("running_timeout", value.to_string()));
    }
    status = request_json(
        Method::GET,
        &url,
        &format!("/v1/requests/{request_id}/wait"),
        token,
        None,
        Some(query),
    )
    .await?;
    if status
        .get("wait_timed_out")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && args.get_flag("cancel_on_timeout")
    {
        let kind = status.get("wait_timeout_kind").cloned();
        status = request_json(
            Method::POST,
            &url,
            &format!("/v1/requests/{request_id}/cancel"),
            token,
            Some(json!({"reason":string(args,"cancel_reason")})),
            None,
        )
        .await?;
        status["wait_timed_out"] = Value::Bool(true);
        if let Some(kind) = kind {
            status["wait_timeout_kind"] = kind;
        }
    }
    println!("{}", serde_json::to_string(&status)?);
    Ok(status.get("exit_code").and_then(Value::as_i64).unwrap_or(0) as i32)
}

async fn request_status(verb: &str, args: &ArgMatches) -> Result<i32> {
    let url = controller_url(args);
    let credential = identity_cli_token(args)?;
    let token = credential.as_deref();
    let id = string(args, "request_id");
    let (method, path, body, query) = match verb {
        "status" => (Method::GET, format!("/v1/requests/{id}"), None, None),
        "logs" => (Method::GET, format!("/v1/requests/{id}/logs"), None, None),
        "wait" => {
            let mut q = vec![(
                "timeout",
                args.get_one::<f64>("timeout").unwrap().to_string(),
            )];
            if let Some(v) = args.get_one::<f64>("running_timeout") {
                q.push(("running_timeout", v.to_string()));
            }
            (
                Method::GET,
                format!("/v1/requests/{id}/wait"),
                None,
                Some(q),
            )
        }
        _ => (
            Method::POST,
            format!("/v1/requests/{id}/cancel"),
            Some(json!({"reason":string(args,"reason")})),
            None,
        ),
    };
    let value = request_json(method, &url, &path, token, body, query).await?;
    if verb == "logs" {
        print!(
            "{}",
            value.get("stdout").and_then(Value::as_str).unwrap_or("")
        );
        eprint!(
            "{}",
            value.get("stderr").and_then(Value::as_str).unwrap_or("")
        );
        return Ok(0);
    }
    println!("{}", serde_json::to_string(&value)?);
    if verb == "status" {
        eprintln!(
            "{}",
            value
                .get("status_message")
                .and_then(Value::as_str)
                .unwrap_or("")
        );
    }
    Ok(if verb == "wait" {
        value.get("exit_code").and_then(Value::as_i64).unwrap_or(0) as i32
    } else {
        0
    })
}
async fn utilization(args: &ArgMatches) -> Result<i32> {
    let credential = identity_cli_token(args)?;
    let value = request_json(
        Method::GET,
        &controller_url(args),
        &format!("/v1/pools/{}/utilization", string(args, "pool")),
        credential.as_deref(),
        None,
        None,
    )
    .await?;
    println!("{}", serde_json::to_string(&value)?);
    Ok(0)
}

async fn controller_start(args: &ArgMatches, inherited_admin: Option<String>) -> Result<i32> {
    let token = identity_cli_token(args)?.or(inherited_admin).context("admin credential required")?;
    let mut instance = Instance::open(&PathBuf::from(string(args, "state_dir")), &token, crate::controller::now())?;
    if args.get_flag("metrics") {
        crate::telemetry_control::set_config(&mut instance.store,&token,&json!({"enabled":true}),None,None,crate::controller::now())?;
    }
    let metric_state = instance.store.snapshot()?;
    let authority = Authority::from_identity(instance.database.clone());
    let journal = Some(Journal::open(&instance.database)?);
    let mut controller = Controller::new(authority, journal)?;
    controller.heartbeat_timeout_seconds =
        *args.get_one::<f64>("heartbeat_timeout_seconds").unwrap();
    controller.lost_timeout_seconds = *args.get_one::<f64>("lost_timeout_seconds").unwrap();
    controller.idle_session_timeout_seconds =
        *args.get_one::<f64>("idle_session_timeout_seconds").unwrap();
    controller.reconcile_missing_grace_seconds = *args
        .get_one::<f64>("reconcile_missing_grace_seconds")
        .unwrap();
    controller.reconcile_missing_heartbeats = *args
        .get_one::<usize>("reconcile_missing_heartbeats")
        .unwrap();
    controller.gpu_contention_policy = string(args, "gpu_contention_policy");
    controller.gpu_shared_ratio = *args.get_one::<usize>("gpu_shared_ratio").unwrap();
    controller.max_active_gpu_sessions_per_user = args
        .get_one::<usize>("max_active_gpu_sessions_per_user")
        .copied();
    controller.max_queued_requests_per_user = args
        .get_one::<usize>("max_queued_requests_per_user")
        .copied();
    controller.max_active_sessions_per_pool = args
        .get_one::<usize>("max_active_sessions_per_pool")
        .copied();
    for value in values(args, "gpu_share_override") {
        let (gpu_uuid, policy, ratio) =
            parse_gpu_share_override(&value, controller.gpu_shared_ratio)?;
        if controller
            .gpu_share_overrides
            .insert(gpu_uuid.clone(), (policy, ratio))
            .is_some()
        {
            bail!("duplicate GPU share override: {gpu_uuid}");
        }
    }
    if let Some(path) = args.get_one::<String>("profiles_file") {
        controller.set_profiles(load_profiles(path)?)?;
    }
    if controller.heartbeat_timeout_seconds <= 0.0
        || controller.lost_timeout_seconds < controller.heartbeat_timeout_seconds
        || controller.idle_session_timeout_seconds < 0.0
        || controller.reconcile_missing_grace_seconds < 0.0
        || controller.reconcile_missing_heartbeats == 0
        || controller.gpu_shared_ratio == 0
    {
        bail!("controller timeout, reconciliation, and GPU ratio values are invalid");
    }
    controller.assert_invariants()?;
    let mut metrics_supervisor = None;
    let mut metrics_status = None;
    if metric_state["telemetry"]["default"]["enabled"] == true {
        let mut supervisor = crate::telemetry_supervisor::PrometheusSupervisor::new(&PathBuf::from(string(args,"state_dir")),
            metric_state["scheduler_id"].as_str().context("invalid scheduler")?,
            std::env::var("CAPE_PROMETHEUS_BIN").unwrap_or_else(|_|"prometheus".into()))?;
        let status = supervisor.start().await.unwrap_or_else(|_|json!({"state":"unavailable","reason":"prometheus_start_failed"}));
        if status["state"] == "ready" {
            supervisor.stop_on_drop = true;
            controller.telemetry_query_url = supervisor.query_url.clone();
            controller.telemetry_receiver_url = supervisor.receiver_url.clone();
        }
        metrics_status = Some(status);
        metrics_supervisor = Some(supervisor);
    }
    let address = SocketAddr::new(
        string(args, "host").parse::<IpAddr>()?,
        *args.get_one::<u16>("port").unwrap(),
    );
    let schedule_interval = args
        .get_one::<String>("schedule_interval")
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.05);
    let mut ready = json!({"url":format!("http://{address}")});
    if let Some(status) = metrics_status {ready["metrics"] = status;}
    println!("{}", ready);
    std::io::stdout().flush()?;
    http::serve(Arc::new(Mutex::new(controller)), address, schedule_interval).await?;
    if let Some(supervisor) = metrics_supervisor.as_mut() {if supervisor.stop_on_drop {supervisor.stop()?;}}
    Ok(0)
}

fn runtime_selection(args: &ArgMatches) -> (bool, Vec<String>) {
    let mut selected = Vec::new();
    for name in values(args, "runtime_adapter") {
        if !selected.contains(&name) {
            selected.push(name);
        }
    }
    let explicit = !selected.is_empty();
    if !explicit {
        selected = vec!["apptainer".into(), "bubblewrap".into(), "podman".into()];
    }
    (explicit, selected)
}

async fn node_enroll(args: &ArgMatches) -> Result<i32> {
    let mut node_id = string(args, "node_id");
    if args.get_flag("resume") {
        let saved = read_registration(&PathBuf::from(string(args, "node_token_out")))?;
        let stored_id = saved["registration"]["node_id"].as_str().context("invalid enrollment node")?;
        if !node_id.is_empty() && node_id != stored_id { bail!("enrollment node differs from saved operation"); }
        node_id = stored_id.to_owned();
    } else if node_id.is_empty() { node_id = format!("node-{}", uuid::Uuid::new_v4().simple()); }
    let (explicit_adapters, adapters) = runtime_selection(args);
    let log_dir = option(args, "log_dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".rscape-agent-logs"));
    let state_path = option(args, "state_path")
        .map(PathBuf::from)
        .unwrap_or_else(|| log_dir.join(format!("state-{}.json", node_id)));
    let execution_state_dir = option(args,"execution_state_dir").map(PathBuf::from)
        .unwrap_or_else(||state_path.parent().unwrap_or(std::path::Path::new(".")).join("execution-state"));
    let bubblewrap_runtime_dir = option(args, "bubblewrap_runtime_dir")
        .map(PathBuf::from)
        .or_else(|| option(args, "runtime_dir").map(|root| PathBuf::from(root).join("bubblewrap")))
        .unwrap_or_else(|| log_dir.join("runtime/bubblewrap"));
    let apptainer_runtime_dir = option(args, "apptainer_runtime_dir")
        .map(PathBuf::from)
        .or_else(|| option(args, "runtime_dir").map(|root| PathBuf::from(root).join("apptainer")))
        .unwrap_or_else(|| log_dir.join("runtime/apptainer"));
    let mut config = AgentConfig {
        metrics_enabled: args.get_flag("metrics"),
        controller_url: controller_url(args),
        node_id,
        agent_epoch: None,
        pool: string(args, "pool"),
        pool_mode: string(args, "pool_mode"),
        gpu_slots: *args.get_one::<usize>("gpu_slots").unwrap(),
        placement: string(args, "pool_placement"),
        cpu_cores: *args.get_one::<f64>("cpu_cores").unwrap(),
        memory_gb: *args.get_one::<f64>("memory_gb").unwrap(),
        gpu_uuids: values(args, "gpu_uuid"),
        allocation_id: option(args, "allocation_id"),
        allocation_deadline: args.get_one::<f64>("allocation_deadline").copied(),
        drain_before_sec: *args.get_one::<f64>("drain_before_sec").unwrap(),
        cleanup_buffer_sec: *args.get_one::<f64>("cleanup_buffer_sec").unwrap(),
        runtime_cache_config: string(args, "runtime_cache_config"),
        cached_images: values(args, "cached_image"),
        join_token: None,
        token: identity_cli_token(args)?,
        heartbeat_interval: *args.get_one::<f64>("heartbeat_interval").unwrap(),
        once: args.get_flag("once"),
        exit_on_removal: !args.get_flag("stay_after_removal"),
        state_path,
        runtime: RuntimeConfig {
            execution_state_dir: Some(execution_state_dir),
            execution_state_min_free_bytes: *args.get_one::<u64>("execution_state_min_free_bytes").unwrap(),
            execution_state_max_bytes: *args.get_one::<u64>("execution_state_max_bytes").unwrap(),
            adapters,
            explicit_adapters,
            apptainer_binary: string(args, "apptainer_binary"),
            bubblewrap_binary: string(args, "bubblewrap_binary"),
            podman_binary: string(args, "podman_binary"),
            apptainer_probe_image: option(args, "apptainer_probe_image"),
            apptainer_cache_dir: option(args, "apptainer_cache").map(PathBuf::from),
            apptainer_tmp_dir: option(args, "apptainer_tmp").map(PathBuf::from),
            apptainer_runtime_dir,
            apptainer_overlay_mode: None,
            apptainer_no_contain: args.get_flag("no_contain"),
            apptainer_unsquash: args.get_flag("apptainer_unsquash"),
            podman_probe_image: option(args, "podman_probe_image"),
            podman_graphroot: option(args, "podman_graphroot").map(PathBuf::from),
            podman_runroot: option(args, "podman_runroot").map(PathBuf::from),
            podman_tmpdir: option(args, "podman_tmpdir").map(PathBuf::from),
            bubblewrap_runtime_dir,
            bubblewrap_system_paths: values(args, "bubblewrap_system_path")
                .into_iter()
                .map(PathBuf::from)
                .collect(),
            bubblewrap_user_namespace: None,
            bubblewrap_read_only_paths: values(args, "bubblewrap_read_only_path")
                .into_iter().map(PathBuf::from).collect(),
            bubblewrap_overlay_mode: None,
            bubblewrap_gpu_discovery: Arc::new(Mutex::new(None)),
            no_warm_sessions: args.get_flag("no_warm_sessions"),
            capabilities: BTreeMap::new(),
            active_processes: Arc::new(Mutex::new(BTreeMap::new())),
            cancel_commands: Arc::new(Mutex::new(BTreeMap::new())),
            cancelled_commands: Arc::new(Mutex::new(BTreeSet::new())),
            log_dir,
        },
    };
    let enrollment_guard = match crate::node_enrollment::enroll(&mut config, &PathBuf::from(string(args, "node_token_out")), args.get_flag("resume")).await {
            Ok(guard) => guard,
            Err(error) => {
            if let Some(uncertain) = error.downcast_ref::<crate::node_enrollment::EnrollmentUncertain>() {
                eprintln!("{}",json!({"outcome":"unknown","operation_id":uncertain.operation_id,"token_file":uncertain.path,"recovery":"node enroll --resume"}));
                return Ok(3);
            }
            return Err(error);
            }
    };
    agent::run(config).await?;
    drop(enrollment_guard);
    Ok(0)
}

async fn admin(args: &ArgMatches) -> Result<i32> {
    let Some((name, sub)) = args.subcommand() else {
        return Ok(2);
    };
    if name == "metric" {
        let (action, leaf) = sub.subcommand().context("metric action required")?;
        let token = identity_cli_token(leaf)?;
        let mut body = json!({"action":format!("metric-{action}"),"node_id":leaf.get_one::<String>("node_id")});
        if action == "set" {
            let mut changes = json!({});
            if let Some(value) = leaf.get_one::<f64>("interval_seconds") { changes["interval_seconds"] = json!(value); }
            if let Some(value) = leaf.get_one::<u32>("stale_periods") { changes["stale_periods"] = json!(value); }
            for key in ["enabled","basic","profiling"] { if let Some(value) = leaf.get_one::<String>(key) { changes[key] = json!(value == "on"); } }
            body["changes"] = changes;
            body["expected_version"] = json!(leaf.get_one::<u64>("expected_version"));
        }
        println!("{}", request_json(Method::POST,&controller_url(leaf),"/v1/identity/admin",token.as_deref(),Some(body),None).await?);
        return Ok(0);
    }
    if name == "audit" {
        let token = identity_cli_token(sub)?;
        println!("{}", request_json(Method::POST,&controller_url(sub),"/v1/identity/admin",token.as_deref(),
            Some(json!({"action":"audit","limit":sub.get_one::<usize>("limit")})),None).await?);
        return Ok(0);
    }
    if name == "node" {
        let (action, leaf) = sub.subcommand().context("node command required")?;
        let mut body=json!({"action":format!("node-{action}"),"node_id":string(leaf,"node_id"),"reason":string(leaf,"reason")});
        if action=="release-quarantine" {
            if !leaf.get_flag("confirm_stopped") {bail!("release-quarantine requires --confirm-stopped");}
            body["confirm_stopped"]=json!(true);
        }
        let token = identity_cli_token(leaf)?;
        println!("{}", request_json(Method::POST, &controller_url(leaf), "/v1/identity/admin", token.as_deref(),
            Some(body), None).await?);
        return Ok(0);
    }
    if matches!(name, "pool" | "user") {
        let (verb, mut leaf) = sub.subcommand().context("management command required")?;
        let mut body = if name == "pool" {
            if verb == "create" { json!({"action":"pool-create","name":string(leaf,"name"),"max_nodes":leaf.get_one::<u64>("max_nodes"),
                "config":{"pool_mode":string(leaf,"pool_mode"),"gpu_slots":leaf.get_one::<u64>("gpu_slots"),"placement":string(leaf,"placement")}}) }
            else if verb == "edit" {
                let mut changes = json!({});
                if let Some(name) = option(leaf,"name") { changes["name"] = json!(name); }
                if leaf.get_flag("unlimited_nodes") || leaf.get_one::<u64>("max_nodes").is_some() { changes["max_nodes"] = json!(leaf.get_one::<u64>("max_nodes")); }
                if option(leaf,"pool_mode").is_some() || option(leaf,"placement").is_some() || leaf.get_one::<u64>("gpu_slots").is_some() {
                    let token = identity_cli_token(leaf)?;
                    let pools = request_json(Method::GET,&controller_url(leaf),"/v1/identity/pools",token.as_deref(),None,None).await?;
                    let pool = string(leaf,"pool");
                    let record = pools.as_array().context("invalid inventory")?.iter().find(|p|p["pool_id"]==pool || p["name"]==pool).context("unknown pool")?;
                    let mut config = record["config"].clone();
                    if let Some(mode) = option(leaf,"pool_mode") { config["pool_mode"] = json!(mode); }
                    if let Some(placement) = option(leaf,"placement") { config["placement"] = json!(placement); }
                    if let Some(slots) = leaf.get_one::<u64>("gpu_slots") { config["gpu_slots"] = json!(slots); }
                    changes["config"] = config;
                }
                json!({"action":"pool-edit","pool":string(leaf,"pool"),"changes":changes})
            }
            else if matches!(verb, "drain" | "remove") { json!({"action":format!("pool-{verb}"),"pool":string(leaf,"pool"),"force":verb=="drain" && leaf.get_flag("force")}) }
            else {
                let (group_verb, group) = leaf.subcommand().context("group command required")?;
                leaf = group;
                if group_verb == "list" { json!({"action":"group-list"}) }
                else if group_verb == "set" { json!({"action":"group-set","name":string(leaf,"name"),"all_pools":leaf.get_flag("all_pools"),"pools":values(leaf,"pool")}) }
                else { json!({"action":"group-delete","name":string(leaf,"name")}) }
            }
        } else if verb == "list" { json!({"action":"user-list"}) }
        else { json!({"action":"user-edit","user_id":string(leaf,"user_id")}) };
        if name == "user" && verb != "list" {
            if verb == "remove" { body["disabled"] = json!(true); } else {
                if leaf.get_flag("clear_groups") { body["groups"] = json!([]); }
                else if leaf.contains_id("group") { body["groups"] = json!(values(leaf,"group")); }
                if leaf.get_flag("disable") || leaf.get_flag("enable") { body["disabled"] = json!(leaf.get_flag("disable")); }
            }
        }
        let token = identity_cli_token(leaf)?;
        println!("{}", request_json(Method::POST, &controller_url(leaf), "/v1/identity/admin", token.as_deref(), Some(body), None).await?);
        return Ok(0);
    }
    match name {
        "get-workspace-config" | "set-workspace-config" => {
            let url = controller_url(sub);
            let credential = identity_cli_token(sub)?;
            let token = credential.as_deref();
            let pool = string(sub, "pool");
            let node = sub.get_one::<String>("node_id");
            let value = if name == "get-workspace-config" {
                let mut query = url::form_urlencoded::Serializer::new(String::new());
                query.append_pair("pool", &pool);
                if let Some(node) = node { query.append_pair("node_id", node); }
                request_json(Method::GET, &url, &format!("/v1/admin/workspace-config?{}", query.finish()), token, None, None).await?
            } else {
                let mut changes = serde_json::Map::new();
                for (option, field) in [("image_size_gib", "image_size_bytes"), ("max_image_size_gib", "max_image_size_bytes")] {
                    if let Some(value) = sub.get_one::<u64>(option) {
                        let bytes = value.checked_mul(1024_u64.pow(3)).filter(|value| *value > 0 && *value <= i64::MAX as u64)
                            .ok_or_else(|| anyhow!("{option} is out of range"))?;
                        changes.insert(field.into(), json!(bytes));
                    }
                }
                if let Some(root) = sub.get_one::<String>("image_root") { changes.insert("image_root".into(), json!(root)); }
                for field in ["bytes_per_inode", "mount_timeout_seconds"] {
                    if let Some(value) = sub.get_one::<u64>(field) { changes.insert(field.into(), json!(value)); }
                }
                let mut maintenance=serde_json::Map::new();
                for field in ["auto_grow","reclaim_on_release"] {
                    if let Some(value)=sub.get_one::<String>(field) {maintenance.insert(field.into(),json!(value=="on"));}
                }
                for field in ["grow_threshold_percent","grow_target_percent","grow_cooldown_seconds","reclaim_cooldown_seconds",
                    "reclaim_idle_seconds","reclaim_start_interval_seconds","reclaim_max_candidates","reclaim_retry_backoff_seconds",
                    "reclaim_low_yield_backoff_seconds","reclaim_nice"] {
                    if let Some(value)=sub.get_one::<u64>(field) {maintenance.insert(field.into(),json!(value));}
                }
                for (option,field,scale) in [("grow_min_mib","grow_min_bytes",1024*1024),("reclaim_min_mib","reclaim_min_allocated_bytes",1024*1024),
                    ("backing_reserve_mib","backing_reserve_bytes",1024*1024),("maintenance_timeout_seconds","timeout_seconds",1)] {
                    if let Some(value)=sub.get_one::<u64>(option) {maintenance.insert(field.into(),json!(value.checked_mul(scale).filter(|n|*n<=i64::MAX as u64).context("maintenance option overflow")?));}
                }
                if !maintenance.is_empty(){changes.insert("maintenance".into(),Value::Object(maintenance));}
                request_json(Method::POST, &url, "/v1/admin/workspace-config", token,
                             Some(json!({"pool":pool,"node_id":node,"changes":changes})), None).await?
            };
            println!("{}", serde_json::to_string(&value)?);
        }
        _ => bail!("unknown admin command"),
    }
    Ok(0)
}

async fn request_json(
    method: Method,
    url: &str,
    path: &str,
    token: Option<&str>,
    body: Option<Value>,
    query: Option<Vec<(&str, String)>>,
) -> Result<Value> {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(3660))
        .build()?;
    let mut request = client.request(method, format!("{}{}", url.trim_end_matches('/'), path));
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    if let Some(body) = body {
        request = request.json(&body);
    }
    if let Some(query) = query {
        request = request.query(&query);
    }
    let response = request.send().await?;
    let status = response.status();
    let text = response.text().await?;
    let value: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({"error":text}));
    if !status.is_success() {
        bail!("CAPE HTTP {}: {}", status, value);
    }
    Ok(value)
}
fn controller_url(args: &ArgMatches) -> String {
    option(args, "controller_url")
        .or_else(|| std::env::var("CAPE_CONTROLLER_URL").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8765".into())
}
fn string(args: &ArgMatches, key: &str) -> String {
    args.get_one::<String>(key).cloned().unwrap_or_default()
}
fn option(args: &ArgMatches, key: &str) -> Option<String> {
    args.get_one::<String>(key).cloned()
}
fn values(args: &ArgMatches, key: &str) -> Vec<String> {
    args.get_many::<String>(key)
        .map(|items| items.cloned().collect())
        .unwrap_or_default()
}
fn parse_gpu_share_override(value: &str, default_ratio: usize) -> Result<(String, String, usize)> {
    let (gpu_uuid, policy_ratio) = value
        .split_once('=')
        .filter(|(gpu_uuid, policy_ratio)| !gpu_uuid.is_empty() && !policy_ratio.is_empty())
        .context("expected GPU_UUID=POLICY[:RATIO]")?;
    let (policy, ratio) = match policy_ratio.split_once(':') {
        Some((policy, ratio)) => (
            policy,
            ratio
                .parse::<usize>()
                .context("override ratio must be a positive integer")?,
        ),
        None => (policy_ratio, default_ratio),
    };
    if !matches!(policy, "barrier-ratio" | "ratio-only" | "exclusive-first") {
        bail!("override policy must be barrier-ratio, ratio-only, or exclusive-first");
    }
    if ratio == 0 {
        bail!("override ratio must be a positive integer");
    }
    Ok((gpu_uuid.into(), policy.into(), ratio))
}
fn load_profiles(path: &str) -> Result<BTreeMap<String, Value>> {
    let value: Value = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read profiles file {path}"))?,
    )
    .with_context(|| format!("parse profiles file {path}"))?;
    let profiles = value.get("profiles").unwrap_or(&value);
    let profiles = profiles
        .as_object()
        .context("profiles file must contain an object or a profiles object")?;
    Ok(profiles
        .iter()
        .map(|(name, profile)| (name.clone(), profile.clone()))
        .collect())
}
fn parse_bind(value: &str) -> Result<Value> {
    let parts: Vec<&str> = value.split(':').collect();
    if !(parts.len() == 2 || parts.len() == 3) || parts[0].is_empty() || parts[1].is_empty() {
        bail!("bind must be src:dst[:ro|rw]");
    }
    let read_only = parts.get(2).map(|mode| *mode == "ro").unwrap_or(false);
    if let Some(mode) = parts.get(2) {
        if !matches!(*mode, "ro" | "rw") {
            bail!("bind mode must be ro or rw");
        }
    }
    Ok(json!({"source":parts[0],"target":parts[1],"read_only":read_only}))
}

#[cfg(test)]
mod tests {
    use super::{clear_credential_environment, parser};

    #[test]
    fn enrollment_defaults_discover_containers_with_ubuntu_probes() {
        let matches = parser().try_get_matches_from([
            "rscape", "node", "enroll", "--pool", "p", "--node-token-out", "unused.json",
            "--controller-url", "http://127.0.0.1:1", "--token", "fixture",
        ]).unwrap();
        let args = matches.subcommand_matches("node").unwrap().subcommand_matches("enroll").unwrap();
        let (explicit, adapters) = super::runtime_selection(args);
        assert!(!explicit);
        assert_eq!(adapters, vec!["apptainer", "bubblewrap", "podman"]);
        for field in ["apptainer_probe_image", "podman_probe_image"] {
            assert_eq!(super::option(args, field).as_deref(), Some("docker://docker.io/library/ubuntu:24.04"));
        }
    }

    #[test]
    fn explicit_runtime_and_image_overrides_are_preserved() {
        let matches = parser().try_get_matches_from([
            "rscape", "node", "enroll", "--pool", "p", "--node-token-out", "unused.json",
            "--controller-url", "http://127.0.0.1:1", "--token", "fixture",
            "--runtime-adapter", "local-process", "--runtime-adapter", "local-process",
            "--apptainer-probe-image", "/local/probe.sif", "--podman-probe-image", "registry/probe:fixed",
        ]).unwrap();
        let args = matches.subcommand_matches("node").unwrap().subcommand_matches("enroll").unwrap();
        let (explicit, adapters) = super::runtime_selection(args);
        assert!(explicit);
        assert_eq!(adapters, vec!["local-process"]);
        assert_eq!(super::option(args, "apptainer_probe_image").as_deref(), Some("/local/probe.sif"));
        assert_eq!(super::option(args, "podman_probe_image").as_deref(), Some("registry/probe:fixed"));
    }

    fn token(argv: &[&str], sub: &str, id: &str) -> Option<String> {
        let matches = parser().try_get_matches_from(argv).unwrap();
        let mut args = matches.subcommand_matches(argv[1]).unwrap();
        if !sub.is_empty() {
            args = args.subcommand_matches(sub).unwrap();
        }
        args.get_one::<String>(id).cloned()
    }

    // One test owns CAPE_TOKEN / CAPE_JOIN_TOKEN so parallel tests never observe each other's values.
    #[test]
    fn credentials_fall_back_to_environment_and_retired_aliases_are_scrubbed() {
        std::env::set_var("CAPE_TOKEN", "env-credential");
        std::env::set_var("CAPE_JOIN_TOKEN", "env-join");
        assert_eq!(token(&["rscape", "status", "req-1"], "", "token").as_deref(), Some("env-credential"));
        assert_eq!(token(&["rscape", "status", "req-1", "--token", "flag"], "", "token").as_deref(), Some("flag"));
        let enroll = ["rscape", "node", "enroll", "--controller-url", "http://c", "--pool", "p", "--node-token-out", "agent.json"];
        assert_eq!(token(&enroll, "enroll", "token").as_deref(), Some("env-credential"));
        for action in ["status", "release", "check", "delete", "grow", "reclaim", "shrink", "recover"] {
            let args = ["rscape", "workspace", action, "workspace-id"];
            assert_eq!(token(&args, "", "token").as_deref(), Some("env-credential"));
            let mut explicit = args.to_vec();
            explicit.extend(["--token", "flag"]);
            assert_eq!(token(&explicit, "", "token").as_deref(), Some("flag"));
        }
        for name in ["get-workspace-config", "set-workspace-config"] {
            let args = ["rscape", "admin", name, "--pool", "p"];
            assert_eq!(token(&args, name, "token").as_deref(), Some("env-credential"));
            let mut explicit = args.to_vec();
            explicit.extend(["--token", "flag"]);
            assert_eq!(token(&explicit, name, "token").as_deref(), Some("flag"));
        }
        // Removed credential-writing commands stay rejected, even with credentials in the environment.
        let grant = ["rscape", "admin", "add-user-token", "--auth-file", "f", "--user", "u", "--pool", "p"];
        assert!(parser().try_get_matches_from(grant).is_err());
        let help = parser().find_subcommand_mut("status").unwrap().render_long_help().to_string();
        assert!(help.contains("CAPE_TOKEN") && !help.contains("env-credential"));
        let aliases = ["CAPE_ADMIN_TOKEN", "CAPE_PROBE_NODE_TOKEN", "CAPE_NODE_TOKEN", "CAPE_USER_TOKENS_INLINE",
            "CAPE_SUBMIT_TOKEN", "CAPE_SUBMIT_USER_TOKENS_INLINE", "CAPE_STATUS_TOKEN",
            "CAPE_RETIRE_TOKEN", "CAPE_CONCURRENCY_STATUS_TOKEN"];
        for name in aliases { std::env::set_var(name, "synthetic-secret"); }
        clear_credential_environment();
        for name in aliases.into_iter().chain(["CAPE_TOKEN", "CAPE_JOIN_TOKEN"]) {
            assert!(std::env::var_os(name).is_none());
        }
    }
}
