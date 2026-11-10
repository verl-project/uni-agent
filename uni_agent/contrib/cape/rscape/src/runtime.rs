// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod apptainer;
mod bubblewrap;
mod podman;
mod storage;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::time::Duration;
use tokio::process::Command;

pub use bubblewrap::OverlayMode;

pub fn workspace_transition(config: &RuntimeConfig, execution_session_id: &str, context: &Value, resume: bool) -> Result<()> {
    match context["runtime_adapter"].as_str().unwrap_or("") {
        "podman" => podman::workspace_transition(config,execution_session_id,context["session_spec"]["isolation_policy"].as_str().unwrap_or("default"),resume),
        "local-process"|"bubblewrap"|"apptainer" => Ok(()),
        _ => bail!("runtime cannot safely suspend workspace references"),
    }
}

#[derive(Clone, Debug, Default)]
pub struct NvidiaDiscovery {
    pub device_map: BTreeMap<String, usize>,
    pub binary: Option<PathBuf>,
    pub libraries: Vec<PathBuf>,
    pub configured: bool,
}

#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    pub execution_state_dir: Option<PathBuf>,
    pub execution_state_min_free_bytes: u64,
    pub execution_state_max_bytes: u64,
    pub adapters: Vec<String>,
    pub explicit_adapters: bool,
    pub apptainer_binary: String,
    pub bubblewrap_binary: String,
    pub podman_binary: String,
    pub apptainer_probe_image: Option<String>,
    pub apptainer_cache_dir: Option<PathBuf>,
    pub apptainer_tmp_dir: Option<PathBuf>,
    pub apptainer_runtime_dir: PathBuf,
    pub apptainer_overlay_mode: Option<String>,
    pub apptainer_no_contain: bool,
    pub apptainer_unsquash: bool,
    pub podman_probe_image: Option<String>,
    pub podman_graphroot: Option<PathBuf>,
    pub podman_runroot: Option<PathBuf>,
    pub podman_tmpdir: Option<PathBuf>,
    pub bubblewrap_runtime_dir: PathBuf,
    pub bubblewrap_system_paths: Vec<PathBuf>,
    pub bubblewrap_read_only_paths: Vec<PathBuf>,
    pub bubblewrap_user_namespace: Option<bool>,
    pub bubblewrap_overlay_mode: Option<OverlayMode>,
    pub bubblewrap_gpu_discovery: Arc<Mutex<Option<NvidiaDiscovery>>>,
    pub no_warm_sessions: bool,
    pub capabilities: BTreeMap<String, Value>,
    pub active_processes: Arc<Mutex<BTreeMap<String, u32>>>,
    pub cancel_commands: Arc<Mutex<BTreeMap<String, Vec<String>>>>,
    pub cancelled_commands: Arc<Mutex<BTreeSet<String>>>,
    pub log_dir: PathBuf,
}

#[cfg(test)]
impl RuntimeConfig {
    pub(crate) fn test_config(adapter: &str) -> Self {
        Self {
            execution_state_dir: None,
            execution_state_min_free_bytes: 512 * 1024 * 1024,
            execution_state_max_bytes: 0,
            adapters: vec![adapter.into()],
            explicit_adapters: true,
            apptainer_binary: "apptainer".into(),
            bubblewrap_binary: "bwrap".into(),
            podman_binary: "podman".into(),
            apptainer_probe_image: None,
            apptainer_cache_dir: None,
            apptainer_tmp_dir: None,
            apptainer_runtime_dir: PathBuf::from("/tmp/rscape-apptainer-test"),
            apptainer_overlay_mode: None,
            apptainer_no_contain: false,
            apptainer_unsquash: false,
            podman_probe_image: None,
            podman_graphroot: None,
            podman_runroot: None,
            podman_tmpdir: None,
            bubblewrap_runtime_dir: PathBuf::from("/tmp/rscape-bubblewrap-test"),
            bubblewrap_system_paths: vec![PathBuf::from("/usr"), PathBuf::from("/etc")],
            bubblewrap_read_only_paths: Vec::new(),
            bubblewrap_user_namespace: Some(false),
            bubblewrap_overlay_mode: None,
            bubblewrap_gpu_discovery: Arc::new(Mutex::new(None)),
            no_warm_sessions: false,
            capabilities: BTreeMap::new(),
            active_processes: Arc::new(Mutex::new(BTreeMap::new())),
            cancel_commands: Arc::new(Mutex::new(BTreeMap::new())),
            cancelled_commands: Arc::new(Mutex::new(BTreeSet::new())),
            log_dir: PathBuf::from("/tmp"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub failure_kind: String,
    pub argv: Vec<String>,
    pub adapter: String,
    pub isolation_policy: String,
}

pub async fn probe(config: &mut RuntimeConfig, gpu_uuids: &[String]) -> Result<()> {
    let requested = config.adapters.clone();
    let mut proven = Vec::new();
    let mut capabilities = BTreeMap::new();
    for adapter in requested {
        let capability = match adapter.as_str() {
            "local-process" => Ok(Some(local_process_capability())),
            "bubblewrap" => bubblewrap::probe(config, gpu_uuids).await,
            "apptainer" => apptainer::probe(config, gpu_uuids).await,
            "podman" => podman::probe(config, gpu_uuids).await,
            _ => Ok(None),
        };
        let capability = match capability {
            Ok(value) => value,
            Err(error) if config.explicit_adapters => return Err(error),
            Err(error) => {
                eprintln!("runtime probe skipped {adapter}: {error}");
                continue;
            }
        };
        match capability {
            Some(value) => {
                proven.push(adapter.clone());
                capabilities.insert(adapter, value);
            }
            None if config.explicit_adapters => {
                bail!("requested runtime adapter failed its execution probe: {adapter}")
            }
            None => eprintln!("runtime probe skipped {adapter}: unavailable or execution probe failed"),
        }
    }
    if proven.is_empty() {
        bail!("no runtime adapter passed its execution probe");
    }
    config.adapters = proven;
    config.capabilities = capabilities;
    Ok(())
}

pub fn aggregate_capabilities(capabilities: &BTreeMap<String, Value>) -> Value {
    if capabilities.len() == 1 {
        return capabilities
            .values()
            .next()
            .cloned()
            .unwrap_or_else(|| json!({}));
    }
    let policy_order = [
        "default",
        "none",
        "host-stateless",
        "host-stateful",
        "image-stateless",
        "image-stateful",
    ];
    let mut image_formats = BTreeSet::new();
    let mut isolation = BTreeSet::new();
    let mut gpu_isolation = BTreeSet::new();
    let mut reduced = BTreeSet::new();
    let mut nested = false;
    let mut binds = false;
    let mut selects_gpus = false;
    let mut cpu_memory = false;
    let mut cache = false;
    let mut warm = false;
    let mut privileged = false;
    let mut enforcement_rank = 0;
    for value in capabilities.values() {
        extend_strings(&mut image_formats, value.get("image_formats"));
        extend_strings(&mut isolation, value.get("isolation_policies"));
        extend_strings(&mut gpu_isolation, value.get("gpu_isolation_policies"));
        extend_strings(&mut reduced, value.get("reduced_isolation_policies"));
        nested |= bool_field(value, "nested_container");
        binds |= bool_field(value, "can_bind_mounts");
        selects_gpus |= bool_field(value, "can_select_gpus");
        cpu_memory |= bool_field(value, "can_enforce_cpu_memory");
        cache |= bool_field(value, "node_local_cache");
        warm |= bool_field(value, "supports_warm_sessions");
        privileged |= bool_field(value, "privileged_allowed");
        enforcement_rank =
            enforcement_rank.max(match value.get("gpu_enforcement").and_then(Value::as_str) {
                Some("hard") => 2,
                Some("advisory") => 1,
                _ => 0,
            });
    }
    let ordered = |values: &BTreeSet<String>| -> Vec<String> {
        policy_order
            .iter()
            .filter(|policy| values.contains(**policy))
            .map(|value| value.to_string())
            .collect()
    };
    json!({
        "adapter":"multi",
        "image_formats":image_formats,
        "isolation_policies":ordered(&isolation),
        "nested_container":nested,
        "can_bind_mounts":binds,
        "can_select_gpus":selects_gpus,
        "gpu_enforcement":match enforcement_rank {2=>"hard",1=>"advisory",_=>"none"},
        "can_enforce_cpu_memory":cpu_memory,
        "node_local_cache":cache,
        "supports_warm_sessions":warm,
        "privileged_allowed":privileged,
        "gpu_isolation_policies":ordered(&gpu_isolation),
        "reduced_isolation_policies":ordered(&reduced),
    })
}

fn leased_gpu_visibility(gpu_uuids: &[String], environment: &BTreeMap<String, String>) -> Result<String> {
    if gpu_uuids.is_empty() {
        return Ok(String::new());
    }
    let value = match environment.get("CUDA_VISIBLE_DEVICES") {
        Some(value) => value,
        None => return Ok(gpu_uuids.join(",")),
    };
    let leased: BTreeSet<&str> = gpu_uuids.iter().map(|uuid| uuid.trim()).collect();
    let invalid: BTreeSet<&str> = value.split(',').map(str::trim)
        .filter(|entry| !entry.is_empty() && !leased.contains(entry)).collect();
    if !invalid.is_empty() {
        bail!("CUDA_VISIBLE_DEVICES contains entries outside the GPU lease: {}; only leased UUIDs are allowed (ordinal indices are not supported)", invalid.into_iter().collect::<Vec<_>>().join(", "));
    }
    Ok(value.clone())
}

pub fn validate_assignment_visibility(assignment: &Value) -> Result<()> {
    let session = &assignment["session_spec"];
    let mut environment = string_map(session.get("env"));
    environment.extend(string_map(assignment["command_spec"].get("env")));
    leased_gpu_visibility(&string_list(session.get("gpu_uuids")), &environment)?;
    Ok(())
}

pub async fn execute(assignment: &Value, config: &RuntimeConfig) -> ExecutionResult {
    let session = assignment
        .get("session_spec")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let command_spec = assignment
        .get("command_spec")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let adapter = session
        .get("runtime_adapter")
        .and_then(Value::as_str)
        .unwrap_or("local-process")
        .to_string();
    let mut visibility_environment = string_map(session.get("env"));
    visibility_environment.extend(string_map(command_spec.get("env")));
    if let Err(error) = leased_gpu_visibility(&string_list(session.get("gpu_uuids")), &visibility_environment) {
        return ExecutionResult {
            exit_code: 127,
            stdout: String::new(),
            stderr: error.to_string(),
            failure_kind: "workload".into(),
            argv: Vec::new(),
            adapter,
            isolation_policy: session.get("isolation_policy").and_then(Value::as_str).unwrap_or("default").into(),
        };
    }
    if adapter == "bubblewrap" {
        return bubblewrap::run(assignment, config).await;
    }
    if adapter == "apptainer" {
        return apptainer::run(assignment, config).await;
    }
    if adapter == "podman" {
        return podman::run(assignment, config).await;
    }
    let policy = session
        .get("isolation_policy")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_string();
    let command: Vec<String> = command_spec
        .get("command")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let cwd = command_spec
        .get("cwd")
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut environment: BTreeMap<String, String> = string_map(session.get("env"));
    environment.extend(string_map(command_spec.get("env")));
    let gpu_uuids = string_list(session.get("gpu_uuids"));
    environment.insert("CUDA_VISIBLE_DEVICES".into(), leased_gpu_visibility(&gpu_uuids, &environment).expect("validated lease visibility"));
    let argv = match adapter.as_str() {
        _ => command.clone(),
    };
    if argv.is_empty() {
        return ExecutionResult {
            exit_code: 127,
            stdout: String::new(),
            stderr: "empty workload command".into(),
            failure_kind: "infrastructure".into(),
            argv,
            adapter,
            isolation_policy: policy,
        };
    }
    let command_id = assignment.get("command_id").and_then(Value::as_str);
    match run_process(&argv, &environment, cwd.as_deref(), config, command_id).await {
        Ok((exit_code, stdout, stderr)) => ExecutionResult {
            exit_code,
            stdout,
            stderr,
            failure_kind: "workload".into(),
            argv,
            adapter,
            isolation_policy: policy,
        },
        Err(error) => ExecutionResult {
            exit_code: 127,
            stdout: String::new(),
            stderr: error.to_string(),
            failure_kind: "infrastructure".into(),
            argv,
            adapter,
            isolation_policy: policy,
        },
    }
}

pub fn cleanup_session(config: &RuntimeConfig, execution_session_id: &str) -> Result<()> {
    bubblewrap::cleanup_session(config, execution_session_id)?;
    apptainer::cleanup_session(config, execution_session_id)?;
    podman::cleanup_session(config, execution_session_id)?;
    Ok(())
}

pub fn cleanup_session_for(config: &RuntimeConfig, execution_session_id: &str, adapter: &str) -> Result<()> {
    match adapter {
        "local-process" => Ok(()),
        "bubblewrap" => bubblewrap::cleanup_session(config, execution_session_id),
        "apptainer" => apptainer::cleanup_session(config, execution_session_id),
        "podman" => podman::cleanup_session(config, execution_session_id),
        _ => bail!("cannot clean an unknown runtime adapter: {adapter}"),
    }
}

pub fn retained_state_paths(config: &RuntimeConfig, id: &str, adapter: &str, policy: &str) -> Result<Vec<PathBuf>> {
    let policy = if policy == "default" && !config.no_warm_sessions {
        match adapter {
            "bubblewrap" if config.bubblewrap_overlay_mode.is_some() => "host-stateful",
            "apptainer" if config.apptainer_overlay_mode.is_some() => "image-stateful",
            _ => policy,
        }
    } else {policy};
    match (adapter, policy) {
        ("podman", "image-stateful") => bail!("Podman image-stateful sessions are not supported"),
        ("apptainer", "image-stateful") => Ok(vec![apptainer::overlay_path(config, id)]),
        ("bubblewrap", "host-stateful") => {
            let root = bubblewrap::session_root(config, id);
            let mut paths = vec![root.join("upper"), root.join("work")];
            for (index, lower) in config.bubblewrap_system_paths.iter().enumerate() {
                let (upper, work, _) = bubblewrap::layer_paths(config, id, index, lower)?;
                paths.extend([upper, work]);
            }
            Ok(paths)
        }
        ("local-process", "none" | "default") | ("bubblewrap", "default" | "host-stateless")
            | ("apptainer" | "podman", "default" | "image-stateless") => Ok(Vec::new()),
        _ => bail!("unsupported retained runtime policy: {adapter}/{policy}"),
    }
}

pub fn validate_retained_state(config: &RuntimeConfig, id: &str, adapter: &str, policy: &str) -> Result<()> {
    for path in retained_state_paths(config, id, adapter, policy)? {
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("retained state missing: {}; explicit rebuild required", path.display()))?;
        if metadata.file_type().is_symlink() { bail!("unsafe retained state path: {}", path.display()); }
        if !metadata.is_file() && !metadata.is_dir() { bail!("invalid retained state type"); }
    }
    Ok(())
}

pub fn ensure_retained_unmounted(config:&RuntimeConfig,id:&str,adapter:&str,policy:&str)->Result<()> {
    let mut roots=retained_state_paths(config,id,adapter,policy)?;
    if adapter=="bubblewrap"{roots.push(bubblewrap::session_root(config,id));}
    let mounts=std::fs::read_to_string("/proc/self/mountinfo")?;
    for root in roots {
        let root=match std::fs::canonicalize(&root){Ok(path)=>path,Err(error) if error.kind()==std::io::ErrorKind::NotFound=>continue,Err(error)=>return Err(error.into())};
        for line in mounts.lines(){
            if let Some(target)=line.split_whitespace().nth(4){
                let target=target.replace("\\040"," ").replace("\\011","\t").replace("\\012","\n").replace("\\134","\\");
                if Path::new(&target).starts_with(&root){bail!("execution layer is still mounted; cleanup refused");}
            }
        }
    }
    Ok(())
}

pub fn retention_fingerprint(config:&RuntimeConfig,spec:&Value)->Result<Value>{
    use std::os::unix::fs::MetadataExt;
    fn identity(path:&Path)->Result<Value>{
        let path=std::fs::canonicalize(path)?;let info=std::fs::metadata(&path)?;
        Ok(json!({"path":path,"device":info.dev(),"inode":info.ino(),"size":info.len(),
            "mtime_ns":(info.mtime() as i128*1_000_000_000+info.mtime_nsec() as i128).to_string()}))
    }
    let adapter=spec["runtime_adapter"].as_str().unwrap_or("local-process");
    if adapter=="local-process"{return Ok(Value::Null);}
    let mut policy=spec["isolation_policy"].as_str().unwrap_or("default");
    let mode=match adapter {
        "apptainer"=>config.apptainer_overlay_mode.clone(),
        "bubblewrap"=>config.bubblewrap_overlay_mode.as_ref().map(|mode|match mode {OverlayMode::Native=>"native".into(),OverlayMode::Fuse=>"fuse".into()}),
        _=>None,
    };
    if policy=="default"{
        policy=match adapter {
            "apptainer"=>if !config.no_warm_sessions&&mode.is_some(){"image-stateful"}else{"image-stateless"},
            "bubblewrap"=>if !config.no_warm_sessions&&mode.is_some(){"host-stateful"}else{"host-stateless"},
            _=>policy,
        };
    }
    let roots=if adapter=="bubblewrap" {bubblewrap::normalize_system_paths(&config.bubblewrap_system_paths)?
        .iter().map(|path|identity(path)).collect::<Result<Vec<_>>>()?}else{Vec::new()};
    let image_stat=match spec["image"].as_str(){Some(image) if Path::new(image).is_file()=>identity(Path::new(image))?,_=>Value::Null};
    Ok(json!({"adapter":adapter,"policy":policy,"image":spec["image"],"image_stat":image_stat,"system_roots":roots,"overlay_mode":mode}))
}

pub fn cancel_command(config: &RuntimeConfig, command_id: &str) -> bool {
    let mut cancelled = config.cancelled_commands.lock().unwrap();
    cancelled.insert(command_id.into());
    let pid = config
        .active_processes
        .lock()
        .unwrap()
        .get(command_id)
        .copied();
    let cancel_argv = config
        .cancel_commands
        .lock()
        .unwrap()
        .get(command_id)
        .cloned();
    drop(cancelled);
    if pid.is_none() && cancel_argv.is_none() {
        return false;
    }
    if let Some(argv) = cancel_argv.filter(|argv| !argv.is_empty()) {
        run_cancel_argv(&argv);
    }
    if let Some(pid) = pid {
        terminate_process_group_blocking(pid);
    }
    true
}

fn run_cancel_argv(argv: &[String]) {
    let Ok(mut child) = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return;
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while std::time::Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

pub fn terminate_all(config: &RuntimeConfig) {
    let command_ids: BTreeSet<String> = config
        .active_processes
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .chain(config.cancel_commands.lock().unwrap().keys().cloned())
        .collect();
    for command_id in command_ids {
        cancel_command(config, &command_id);
    }
}

pub async fn run_process(
    argv: &[String],
    environment: &BTreeMap<String, String>,
    cwd: Option<&str>,
    config: &RuntimeConfig,
    command_id: Option<&str>,
) -> Result<(i32, String, String)> {
    let mut process = Command::new(&argv[0]);
    process.args(&argv[1..]);
    process.envs(environment);
    process.stdout(Stdio::piped()).stderr(Stdio::piped());
    process.kill_on_drop(true);
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }
    unsafe {
        process.pre_exec(|| {
            nix::unistd::setsid()
                .map(|_| ())
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))
        });
    }
    let (child, pid) = {
        let cancelled = config.cancelled_commands.lock().unwrap();
        if command_id.map(|id| cancelled.contains(id)).unwrap_or(false) {
            return Ok((130, String::new(), "command cancelled before launch".into()));
        }
        let child = process.spawn()?;
        let pid = child.id().ok_or_else(|| anyhow::anyhow!("spawned process has no pid"))?;
        if let Some(command_id) = command_id {
            config.active_processes.lock().unwrap().insert(command_id.into(), pid);
        }
        (child, pid)
    };
    let output = child.wait_with_output().await;
    if let Some(command_id) = command_id {
        config.active_processes.lock().unwrap().remove(command_id);
    }
    let output = output?;
    let mut exit_code = output.status.code().unwrap_or(128);
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let mut stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if process_group_exists(pid) {
        terminate_process_group(pid).await;
        if exit_code == 0 {
            exit_code = 125;
        }
        if !stderr.is_empty() && !stderr.ends_with('\n') {
            stderr.push('\n');
        }
        stderr
            .push_str("CAPE reaped lingering workload process group after top-level command exit");
    }
    Ok((exit_code, stdout, stderr))
}

fn process_group_exists(pid: u32) -> bool {
    nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
}
fn signal_process_group(pid: u32, signal: nix::sys::signal::Signal) {
    let _ = nix::sys::signal::killpg(nix::unistd::Pid::from_raw(pid as i32), signal);
}
fn terminate_process_group_blocking(pid: u32) {
    signal_process_group(pid, nix::sys::signal::Signal::SIGTERM);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        if !process_group_exists(pid) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    signal_process_group(pid, nix::sys::signal::Signal::SIGKILL);
}
async fn terminate_process_group(pid: u32) {
    signal_process_group(pid, nix::sys::signal::Signal::SIGTERM);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    while tokio::time::Instant::now() < deadline {
        if !process_group_exists(pid) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    signal_process_group(pid, nix::sys::signal::Signal::SIGKILL);
}

pub fn write_log(path: &Path, assignment: &Value, result: &ExecutionResult) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut value = String::new();
    value.push_str(&format!(
        "request: {}\n",
        assignment
            .get("request_id")
            .and_then(Value::as_str)
            .unwrap_or("")
    ));
    value.push_str(&format!(
        "session: {}\n",
        assignment
            .get("execution_session_id")
            .and_then(Value::as_str)
            .unwrap_or("")
    ));
    value.push_str(&format!("runtime-adapter: {}\n", result.adapter));
    value.push_str(&format!("isolation-policy: {}\n", result.isolation_policy));
    value.push_str(&format!("exec: {}\n", result.argv.join(" ")));
    value.push_str(&format!(
        "argv-json: {}\n",
        serde_json::to_string(&result.argv).unwrap_or_else(|_| "[]".into())
    ));
    value.push_str(&format!("exit-code: {}\n", result.exit_code));
    value.push_str(&result.stdout);
    value.push_str(&result.stderr);
    std::fs::write(path, value)
}

fn local_process_capability() -> Value {
    json!({"adapter":"local-process","image_formats":["default"],"isolation_policies":["default","none"],"nested_container":true,"can_bind_mounts":true,"can_select_gpus":true,"gpu_enforcement":"none","can_enforce_cpu_memory":false,"node_local_cache":true,"supports_warm_sessions":true,"privileged_allowed":false,"gpu_isolation_policies":["default","none"],"reduced_isolation_policies":[]})
}

fn string_map(value: Option<&Value>) -> BTreeMap<String, String> {
    value
        .and_then(Value::as_object)
        .map(|items| {
            items
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}
fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}
fn extend_strings(target: &mut BTreeSet<String>, value: Option<&Value>) {
    if let Some(items) = value.and_then(Value::as_array) {
        target.extend(items.iter().filter_map(Value::as_str).map(str::to_string));
    }
}
fn bool_field(value: &Value, key: &str) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn automatic_probe_errors_do_not_hide_healthy_adapters() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("cape-auto-probe-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let binary = root.join("runtime-double");
        std::fs::write(&binary, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut config = super::RuntimeConfig::test_config("podman");
        config.adapters = vec!["apptainer".into(), "podman".into()];
        config.explicit_adapters = false;
        config.apptainer_binary = binary.to_string_lossy().into_owned();
        config.podman_binary = config.apptainer_binary.clone();
        config.apptainer_probe_image = Some("probe".into());
        config.podman_probe_image = Some("probe".into());
        config.apptainer_runtime_dir = binary.clone(); // Directory creation must fail.
        config.no_warm_sessions = true;
        super::probe(&mut config, &[]).await.unwrap();
        assert_eq!(config.adapters, vec!["podman"]);
        assert!(!config.capabilities.contains_key("local-process"));
        config.adapters = vec!["apptainer".into(), "podman".into()];
        config.explicit_adapters = true;
        assert!(super::probe(&mut config, &[]).await.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn no_automatic_runtime_does_not_enable_local_process() {
        let mut config = super::RuntimeConfig::test_config("apptainer");
        config.adapters = vec!["apptainer".into(), "bubblewrap".into(), "podman".into()];
        config.explicit_adapters = false;
        config.apptainer_binary = "/nonexistent/cape-apptainer".into();
        config.bubblewrap_binary = "/nonexistent/cape-bwrap".into();
        config.podman_binary = "/nonexistent/cape-podman".into();
        assert!(super::probe(&mut config, &[]).await.is_err());
        assert!(config.capabilities.is_empty());
    }

    #[test]
    fn visibility_accepts_only_leased_uuid_subsets() {
        let lease = vec!["GPU-a".to_string(), "GPU-b".to_string()];
        assert_eq!(super::leased_gpu_visibility(&lease, &Default::default()).unwrap(), "GPU-a,GPU-b");
        for value in ["", "GPU-b", "GPU-b,GPU-a", " GPU-b , GPU-b "] {
            let env = std::collections::BTreeMap::from([("CUDA_VISIBLE_DEVICES".into(), value.into())]);
            assert_eq!(super::leased_gpu_visibility(&lease, &env).unwrap(), value);
        }
        for value in ["GPU-other", "0", "GPU-a,1"] {
            let env = std::collections::BTreeMap::from([("CUDA_VISIBLE_DEVICES".into(), value.into())]);
            assert!(super::leased_gpu_visibility(&lease, &env).unwrap_err().to_string().contains("outside the GPU lease"));
            assert_eq!(super::leased_gpu_visibility(&[], &env).unwrap(), "");
        }
    }

    use super::*;

    fn normalize_argv(argv: Vec<String>, state: &Path) -> Vec<String> {
        let state = state.to_string_lossy();
        let uid_gid = format!("{}:{}", unsafe { nix::libc::geteuid() }, unsafe {
            nix::libc::getegid()
        });
        argv.into_iter()
            .map(|value| {
                value
                    .replace(state.as_ref(), "<STATE>")
                    .replace(&uid_gid, "<UIDGID>")
            })
            .collect()
    }

    fn assert_oracle(oracle: &Value, key: &str, actual: Vec<String>) {
        let expected: Vec<String> = serde_json::from_value(oracle[key].clone()).unwrap();
        assert_eq!(actual, expected, "runtime argv differs for {key}");
    }

    fn config(directory: &Path) -> RuntimeConfig {
        RuntimeConfig {
            execution_state_dir: None,
            execution_state_min_free_bytes: 0,
            execution_state_max_bytes: 0,
            adapters: vec!["local-process".into()],
            explicit_adapters: true,
            apptainer_binary: "apptainer".into(),
            bubblewrap_binary: "bwrap".into(),
            podman_binary: "podman".into(),
            apptainer_probe_image: None,
            apptainer_cache_dir: None,
            apptainer_tmp_dir: None,
            apptainer_runtime_dir: directory.join("apptainer"),
            apptainer_overlay_mode: None,
            apptainer_no_contain: false,
            apptainer_unsquash: false,
            podman_probe_image: None,
            podman_graphroot: None,
            podman_runroot: None,
            podman_tmpdir: None,
            bubblewrap_runtime_dir: directory.join("bubblewrap"),
            bubblewrap_system_paths: Vec::new(),
            bubblewrap_read_only_paths: Vec::new(),
            bubblewrap_user_namespace: None,
            bubblewrap_overlay_mode: None,
            bubblewrap_gpu_discovery: Arc::new(Mutex::new(None)),
            no_warm_sessions: false,
            capabilities: BTreeMap::new(),
            active_processes: Arc::new(Mutex::new(BTreeMap::new())),
            cancel_commands: Arc::new(Mutex::new(BTreeMap::new())),
            cancelled_commands: Arc::new(Mutex::new(BTreeSet::new())),
            log_dir: directory.into(),
        }
    }

    #[tokio::test]
    async fn local_process_returns_workload_exit_code() {
        let directory = tempfile::tempdir().unwrap();
        let config = config(directory.path());
        let assignment = json!({"session_spec":{"runtime_adapter":"local-process","isolation_policy":"none"},"command_spec":{"command":[std::env::var("PYTHON").unwrap_or_else(|_|"python3".into()),"-c","raise SystemExit(7)"],"env":{}}});
        let result = tokio::time::timeout(Duration::from_secs(5), execute(&assignment, &config))
            .await
            .expect("local process timed out");
        assert_eq!(result.exit_code, 7, "{}", result.stderr);
        assert_eq!(result.failure_kind, "workload");
    }

    #[tokio::test]
    async fn local_process_reaps_a_lingering_background_group() {
        let directory = tempfile::tempdir().unwrap();
        let config = config(directory.path());
        let assignment = json!({"command_id":"cmd-background","session_spec":{"runtime_adapter":"local-process","isolation_policy":"none"},"command_spec":{"command":["/bin/bash","-c","sleep 30 >/dev/null 2>&1 &"],"env":{}}});
        let result = tokio::time::timeout(Duration::from_secs(5), execute(&assignment, &config))
            .await
            .expect("background process reap timed out");
        assert_eq!(result.exit_code, 125, "{}", result.stderr);
        assert!(result
            .stderr
            .contains("reaped lingering workload process group"));
    }

    #[tokio::test]
    async fn cancellation_before_spawn_prevents_workspace_writes() {
        let directory = tempfile::tempdir().unwrap();
        let config = config(directory.path());
        cancel_command(&config, "cmd-before-spawn");
        let marker = directory.path().join("must-not-exist");
        let assignment = json!({"command_id":"cmd-before-spawn","session_spec":{"runtime_adapter":"local-process","isolation_policy":"none"},"command_spec":{"command":["/usr/bin/touch",marker],"env":{}}});
        let result = execute(&assignment, &config).await;
        assert_eq!(result.exit_code, 130);
        assert!(!marker.exists());
        assert!(config.active_processes.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancellation_terminates_the_recorded_process_group() {
        let directory = tempfile::tempdir().unwrap();
        let config = config(directory.path());
        let assignment = json!({"command_id":"cmd-cancel","session_spec":{"runtime_adapter":"local-process","isolation_policy":"none"},"command_spec":{"command":["/bin/bash","-c","sleep 30"],"env":{}}});
        let running_config = config.clone();
        let task = tokio::spawn(async move { execute(&assignment, &running_config).await });
        for _ in 0..100 {
            if config
                .active_processes
                .lock()
                .unwrap()
                .contains_key("cmd-cancel")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(cancel_command(&config, "cmd-cancel"));
        let result = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("cancelled process did not exit")
            .unwrap();
        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn aggregate_is_union_not_constant() {
        let capabilities = BTreeMap::from([(
            "bubblewrap".into(),
            json!({"adapter":"bubblewrap","image_formats":["host"],"isolation_policies":["default","host-stateless"],"nested_container":true,"can_bind_mounts":true,"can_select_gpus":false,"gpu_enforcement":"hard","can_enforce_cpu_memory":false,"node_local_cache":true,"supports_warm_sessions":false,"privileged_allowed":false,"gpu_isolation_policies":[],"reduced_isolation_policies":[]}),
        )]);
        let aggregate = aggregate_capabilities(&capabilities);
        assert_eq!(aggregate["adapter"], "bubblewrap");
        assert!(!aggregate["isolation_policies"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "image-stateless"));
    }

    #[test]
    fn runtime_argv_matches_pycape_oracle() {
        let Some(path) = std::env::var_os("RSCAPE_RUNTIME_ORACLE") else {
            return;
        };
        let oracle: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let state = tempfile::tempdir().unwrap();
        assert_oracle(&oracle, "none", vec!["/bin/true".into()]);

        let mut bubblewrap = RuntimeConfig::test_config("bubblewrap");
        bubblewrap.bubblewrap_binary = "/opt/bwrap".into();
        bubblewrap.bubblewrap_runtime_dir = state.path().into();
        bubblewrap.bubblewrap_system_paths = vec!["/usr".into(), "/etc".into(), "/opt".into()];
        bubblewrap.bubblewrap_user_namespace = Some(false);
        let environment = BTreeMap::from([
            ("COMMAND".into(), "yes".into()),
            ("SESSION".into(), "yes".into()),
        ]);
        for policy in ["host-stateless", "host-stateful"] {
            bubblewrap.bubblewrap_overlay_mode = if policy == "host-stateful" {
                Some(OverlayMode::Native)
            } else {
                None
            };
            let argv = bubblewrap::build_argv(
                &bubblewrap,
                &["/bin/true".into()],
                Some("/workspace"),
                &environment,
                &[],
                &[],
                policy,
                "session-a",
            )
            .unwrap();
            assert_oracle(&oracle, policy, normalize_argv(argv, state.path()));
        }

        let mut apptainer = RuntimeConfig::test_config("apptainer");
        apptainer.apptainer_binary = "/opt/apptainer".into();
        apptainer.apptainer_runtime_dir = state.path().into();
        apptainer.apptainer_no_contain = false;
        for policy in ["image-stateless", "image-stateful"] {
            apptainer.apptainer_overlay_mode = if policy == "image-stateful" {
                Some("directory".into())
            } else {
                None
            };
            let argv = apptainer::build_argv(
                &apptainer,
                "session-a",
                "docker://image",
                &["/bin/true".into()],
                Some("/workspace"),
                &BTreeMap::from([("COMMAND".into(), "yes".into())]),
                &[],
                &[],
                policy,
            )
            .unwrap();
            assert_oracle(
                &oracle,
                &format!("apptainer-{policy}"),
                normalize_argv(argv, state.path()),
            );
        }

        let mut podman = RuntimeConfig::test_config("podman");
        podman.podman_binary = "/opt/podman".into();
        podman.podman_graphroot = Some("/graph".into());
        podman.podman_runroot = Some("/runroot".into());
        podman.podman_tmpdir = Some("/tmpdir".into());
        let command_env = BTreeMap::from([("COMMAND".into(), "yes".into())]);
        let merged_env = BTreeMap::from([
            ("COMMAND".into(), "yes".into()),
            ("PYCAPE_PODMAN_ROOT".into(), "/graph".into()),
            ("PYCAPE_PODMAN_RUNROOT".into(), "/runroot".into()),
            ("SESSION".into(), "yes".into()),
            ("TMPDIR".into(), "/tmpdir".into()),
        ]);
        for policy in ["image-stateless", "image-stateful"] {
            let argv = podman::build_argv(
                &podman,
                "session-a",
                "image",
                &["/bin/true".into()],
                Some("/workspace"),
                &command_env,
                &merged_env,
                &[],
                policy,
            )
            .unwrap();
            assert_oracle(
                &oracle,
                &format!("podman-{policy}"),
                normalize_argv(argv, state.path()),
            );
        }
    }
}
