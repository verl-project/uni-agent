// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{ExecutionResult, NvidiaDiscovery, RuntimeConfig};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, Instant};
use tokio::process::Command;

const DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
const SYSTEM_ALIASES: &[&str] = &["/bin", "/sbin", "/lib", "/lib64", "/lib32"];
const VIRTUAL_ROOTS: &[&str] = &["/proc", "/sys", "/dev", "/run"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OverlayMode {
    Native,
    Fuse,
}

#[derive(Clone, Debug)]
struct FuseMount {
    child: u32,
    merged: PathBuf,
}

pub async fn probe(config: &mut RuntimeConfig, gpu_uuids: &[String]) -> Result<Option<Value>> {
    if !binary_works(&config.bubblewrap_binary, &["--version"]).await {
        return Ok(None);
    }
    config.bubblewrap_system_paths = normalize_system_paths(&config.bubblewrap_system_paths)?;
    fs::create_dir_all(&config.bubblewrap_runtime_dir)?;

    let original_mode = config.bubblewrap_user_namespace;
    let candidates: Vec<bool> = match original_mode {
        Some(mode) => vec![mode],
        None if unsafe { nix::libc::geteuid() } == 0 => vec![true, false],
        None => vec![true],
    };
    let mut functional = false;
    for mode in candidates {
        config.bubblewrap_user_namespace = Some(mode);
        let assignment = probe_assignment(
            "host-stateless",
            &[],
            &["/bin/sh", "-c", "printf cape-bubblewrap-probe"],
        );
        let result = timed_run(&assignment, config, 10).await;
        if result.exit_code == 0 && result.stdout == "cape-bubblewrap-probe" {
            functional = true;
            break;
        }
        eprintln!(
            "[rscape-runtime] Bubblewrap stateless probe failed mode={} exit={} stdout={:?} stderr={:?}",
            mode, result.exit_code, result.stdout, result.stderr
        );
    }
    if !functional {
        eprintln!("[rscape-runtime] Bubblewrap omitted: no stateless execution recipe passed");
        config.bubblewrap_user_namespace = original_mode;
        return Ok(None);
    }

    config.bubblewrap_overlay_mode = None;
    if !config.no_warm_sessions {
        if binary_help_has_native_overlay(&config.bubblewrap_binary).await {
            config.bubblewrap_overlay_mode = Some(OverlayMode::Native);
            if !probe_overlay_persistence(config).await {
                config.bubblewrap_overlay_mode = None;
            }
        }
        if config.bubblewrap_overlay_mode.is_none()
            && find_executable("fuse-overlayfs").is_some()
            && (find_executable("fusermount3").is_some() || find_executable("fusermount").is_some())
        {
            config.bubblewrap_overlay_mode = Some(OverlayMode::Fuse);
            if !probe_overlay_persistence(config).await {
                config.bubblewrap_overlay_mode = None;
            }
        }
    }

    let mut policies = vec!["default", "host-stateless"];
    if config.bubblewrap_overlay_mode.is_some() {
        policies.push("host-stateful");
    }
    let mut gpu_policies = Vec::new();
    if !gpu_uuids.is_empty() {
        for policy in ["host-stateless", "host-stateful"] {
            if policies.contains(&policy) && probe_gpu_policy(config, gpu_uuids, policy).await {
                gpu_policies.push(policy);
            }
        }
        let default_policy = if config.bubblewrap_overlay_mode.is_some() {
            "host-stateful"
        } else {
            "host-stateless"
        };
        if gpu_policies.contains(&default_policy) {
            gpu_policies.insert(0, "default");
        }
    }
    let reduced = if config.bubblewrap_user_namespace == Some(false) {
        policies.clone()
    } else {
        Vec::new()
    };
    Ok(Some(json!({
        "adapter": "bubblewrap",
        "image_formats": ["host"],
        "isolation_policies": policies,
        "nested_container": true,
        "can_bind_mounts": true,
        "can_select_gpus": !gpu_policies.is_empty(),
        "gpu_enforcement": "hard",
        "can_enforce_cpu_memory": false,
        "node_local_cache": true,
        "supports_warm_sessions": false,
        "privileged_allowed": false,
        "gpu_isolation_policies": gpu_policies,
        "reduced_isolation_policies": reduced,
    })))
}

pub async fn run(assignment: &Value, config: &RuntimeConfig) -> ExecutionResult {
    let adapter = "bubblewrap".to_string();
    let session = assignment
        .get("session_spec")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let command_spec = assignment
        .get("command_spec")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let policy = resolved_policy(
        session
            .get("isolation_policy")
            .and_then(Value::as_str)
            .unwrap_or("default"),
        config.bubblewrap_overlay_mode.as_ref(),
    );
    let policy = match policy {
        Ok(value) => value,
        Err(error) => return infrastructure(adapter, "default", error.to_string(), Vec::new()),
    };
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
    if command.is_empty() {
        return infrastructure(
            adapter,
            &policy,
            "empty workload command".into(),
            Vec::new(),
        );
    }
    let cwd = command_spec
        .get("cwd")
        .and_then(Value::as_str)
        .or_else(|| session.get("cwd").and_then(Value::as_str));
    let mut environment = string_map(session.get("env"));
    environment.extend(string_map(command_spec.get("env")));
    let gpu_uuids = string_list(session.get("gpu_uuids"));
    let mounts = session
        .get("mounts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let execution_session_id = assignment
        .get("execution_session_id")
        .and_then(Value::as_str)
        .or_else(|| session.get("execution_session_id").and_then(Value::as_str))
        .unwrap_or("runtime-session");

    let mut fuse_mounts = Vec::new();
    if policy == "host-stateful" && config.bubblewrap_overlay_mode == Some(OverlayMode::Fuse) {
        match mount_fuse_layers(config, execution_session_id) {
            Ok(value) => fuse_mounts = value,
            Err(error) => return infrastructure(adapter, &policy, error.to_string(), Vec::new()),
        }
    }
    let argv = match build_argv(
        config,
        &command,
        cwd,
        &environment,
        &mounts,
        &gpu_uuids,
        &policy,
        execution_session_id,
    ) {
        Ok(value) => value,
        Err(error) => {
            let _ = unmount_fuse_layers(&fuse_mounts);
            return infrastructure(adapter, &policy, error.to_string(), Vec::new());
        }
    };
    let output = run_argv(
        &argv,
        config,
        assignment.get("command_id").and_then(Value::as_str),
    )
    .await;
    let unmount_error = unmount_fuse_layers(&fuse_mounts).err();
    match output {
        Ok((exit_code, stdout, mut stderr)) => {
            if let Some(error) = unmount_error {
                if !stderr.is_empty() && !stderr.ends_with('\n') {
                    stderr.push('\n');
                }
                stderr.push_str(&error.to_string());
                return ExecutionResult {
                    exit_code: 127,
                    stdout,
                    stderr,
                    failure_kind: "infrastructure".into(),
                    argv,
                    adapter,
                    isolation_policy: policy,
                };
            }
            ExecutionResult {
                exit_code,
                stdout,
                stderr,
                failure_kind: "workload".into(),
                argv,
                adapter,
                isolation_policy: policy,
            }
        }
        Err(error) => infrastructure(adapter, &policy, error.to_string(), argv),
    }
}

pub fn cleanup_session(config: &RuntimeConfig, execution_session_id: &str) -> Result<()> {
    let root = session_root(config, execution_session_id);
    if root.exists() {
        fs::remove_dir_all(root)?;
    }
    Ok(())
}

pub fn build_argv(
    config: &RuntimeConfig,
    command: &[String],
    cwd: Option<&str>,
    environment: &BTreeMap<String, String>,
    mounts: &[Value],
    gpu_uuids: &[String],
    policy: &str,
    execution_session_id: &str,
) -> Result<Vec<String>> {
    if command.is_empty() {
        bail!("empty workload command");
    }
    let system_paths = normalize_system_paths(&config.bubblewrap_system_paths)?;
    let gpu_discovery = if gpu_uuids.is_empty() {
        None
    } else {
        Some(nvidia_discovery(config, gpu_uuids)?)
    };
    let mut argv = vec![
        config.bubblewrap_binary.clone(),
        "--die-with-parent".into(),
        "--new-session".into(),
    ];
    if config.bubblewrap_user_namespace != Some(false) {
        argv.extend(strings(&["--unshare-user", "--uid", "0", "--gid", "0"]));
    }
    argv.extend(strings(&[
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]));
    argv.extend(strings(&["--clearenv", "--tmpfs", "/"]));
    // The synthetic /tmp and /home come first so that a configured toolchain path below them stays visible.
    argv.extend(strings(&[
        "--tmpfs",
        "/tmp",
        "--dir",
        "/home",
        "--tmpfs",
        "/home",
        "--dir",
        "/home/cape",
    ]));
    argv.extend(system_mount_flags(
        config,
        &system_paths,
        policy,
        execution_session_id,
    )?);
    argv.extend(strings(&["--proc", "/proc", "--dev", "/dev"]));
    if !gpu_uuids.is_empty() {
        argv.extend(gpu_mount_flags(gpu_uuids, gpu_discovery.as_ref().unwrap())?);
    }
    for path in virtual_subpaths(&config.bubblewrap_read_only_paths)? {
        if path.exists() {
            let value = path.to_string_lossy().into_owned();
            argv.extend(["--ro-bind".into(), value.clone(), value]);
        }
    }
    for mount in mounts {
        let source = mount
            .get("source")
            .and_then(Value::as_str)
            .context("mount source is required")?;
        let target = mount
            .get("target")
            .and_then(Value::as_str)
            .context("mount target is required")?;
        if !Path::new(source).is_absolute() || !Path::new(target).is_absolute() {
            bail!("Bubblewrap mount paths must be absolute: {source}:{target}");
        }
        argv.push(
            if mount
                .get("read_only")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                "--ro-bind".into()
            } else {
                "--bind".into()
            },
        );
        argv.push(source.into());
        argv.push(target.into());
    }
    if let Some(cwd) = cwd {
        if !Path::new(cwd).is_absolute() {
            bail!("Bubblewrap cwd must be absolute: {cwd}");
        }
        argv.extend(["--chdir".into(), cwd.into()]);
    }
    let sandbox_environment =
        sandbox_environment(environment, cwd, gpu_uuids, gpu_discovery.as_ref())?;
    for (key, value) in sandbox_environment {
        argv.extend(["--setenv".into(), key, value]);
    }
    if config.bubblewrap_user_namespace == Some(false) {
        argv.extend(strings(&["--cap-drop", "ALL"]));
    }
    argv.push("--".into());
    argv.extend(command.iter().cloned());
    Ok(argv)
}

fn resolved_policy(policy: &str, overlay_mode: Option<&OverlayMode>) -> Result<String> {
    match policy {
        "default" => Ok(if overlay_mode.is_some() {
            "host-stateful"
        } else {
            "host-stateless"
        }
        .into()),
        "host-stateless" => Ok(policy.into()),
        "host-stateful" if overlay_mode.is_some() => Ok(policy.into()),
        "host-stateful" => {
            bail!("rootless Bubblewrap writable overlay was not proven on this node")
        }
        _ => bail!("bubblewrap cannot satisfy isolation policy: {policy}"),
    }
}

fn real_path_allow_missing(path: &Path) -> Result<PathBuf> {
    let mut pending: VecDeque<_> = path.iter().map(|part| part.to_os_string()).collect();
    let mut resolved = PathBuf::new();
    let mut links = 0;
    while let Some(part) = pending.pop_front() {
        if part == ".." {
            resolved.pop();
            continue;
        }
        if part == "." {
            continue;
        }
        let candidate = resolved.join(&part);
        match fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                links += 1;
                if links > 40 {
                    bail!("virtual path contains a symlink loop");
                }
                let target = fs::read_link(&candidate)?;
                if target.is_absolute() {
                    resolved = PathBuf::from("/");
                }
                for component in target.iter().rev() {
                    pending.push_front(component.to_os_string());
                }
            }
            Ok(_) => resolved = candidate,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound
                || error.raw_os_error() == Some(nix::libc::ENOTDIR) => resolved = candidate,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(resolved)
}

fn virtual_subpaths(configured: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    for path in configured {
        let mut normalized = PathBuf::new();
        for component in path.components() {
            match component {
                std::path::Component::ParentDir => { normalized.pop(); }
                std::path::Component::CurDir => {}
                value => normalized.push(value.as_os_str()),
            }
        }
        let root = ["/proc", "/sys", "/dev", "/run"].into_iter()
            .find(|root| normalized.starts_with(root) && normalized != Path::new(root))
            .context("read-only virtual path must be strictly under /proc, /sys, /dev, or /run")?;
        let resolved = real_path_allow_missing(&normalized)?;
        if !resolved.starts_with(root) || resolved == Path::new(root) {
            bail!("read-only virtual path escapes its virtual root: {}", normalized.display());
        }
        if !result.contains(&normalized) { result.push(normalized); }
    }
    Ok(result)
}

pub(super) fn normalize_system_paths(configured: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let candidates: Vec<PathBuf> = if configured.is_empty() {
        ["/usr", "/etc", "/opt"]
            .iter()
            .map(PathBuf::from)
            .filter(|path| path.is_dir())
            .collect()
    } else {
        configured.to_vec()
    };
    if candidates.is_empty() {
        bail!("Bubblewrap system-path allowlist must not be empty");
    }
    let mut normalized = Vec::<PathBuf>::new();
    for candidate in candidates {
        if !candidate.is_absolute() || candidate == Path::new("/") {
            bail!(
                "Bubblewrap system paths must be absolute and exclude host root: {}",
                candidate.display()
            );
        }
        let path = candidate.canonicalize().with_context(|| {
            format!(
                "Bubblewrap system path is unavailable: {}",
                candidate.display()
            )
        })?;
        if path == Path::new("/")
            || VIRTUAL_ROOTS
                .iter()
                .any(|root| path == Path::new(root) || path.starts_with(format!("{root}/")))
        {
            bail!(
                "Bubblewrap system path may not expose host root or a virtual kernel tree: {}",
                candidate.display()
            );
        }
        if normalized
            .iter()
            .any(|parent| path.starts_with(parent) && path != *parent)
        {
            continue;
        }
        normalized.retain(|child| !(child.starts_with(&path) && child != &path));
        if !normalized.contains(&path) {
            normalized.push(path);
        }
    }
    Ok(normalized)
}

fn system_mount_flags(
    config: &RuntimeConfig,
    system_paths: &[PathBuf],
    policy: &str,
    execution_session_id: &str,
) -> Result<Vec<String>> {
    let mut flags = Vec::new();
    if policy == "host-stateful" && !session_root(config,execution_session_id).exists() { super::storage::check(config)?; }
    for (index, path) in system_paths.iter().enumerate() {
        let text = path.to_string_lossy().to_string();
        if policy == "host-stateless" {
            flags.extend(["--ro-bind".into(), text.clone(), text]);
            continue;
        }
        let (upper, work, merged) = layer_paths(config, execution_session_id, index, path)?;
        fs::create_dir_all(&upper)?;
        fs::create_dir_all(&work)?;
        match config.bubblewrap_overlay_mode {
            Some(OverlayMode::Native) => flags.extend([
                "--overlay-src".into(),
                text.clone(),
                "--overlay".into(),
                upper.to_string_lossy().into_owned(),
                work.to_string_lossy().into_owned(),
                text,
            ]),
            Some(OverlayMode::Fuse) => {
                flags.extend(["--bind".into(), merged.to_string_lossy().into_owned(), text])
            }
            None => bail!("host-stateful requested without a proven overlay"),
        }
    }
    for alias in SYSTEM_ALIASES {
        let path = Path::new(alias);
        let metadata = match fs::symlink_metadata(path) {
            Ok(value) if value.file_type().is_symlink() => value,
            _ => continue,
        };
        let _ = metadata;
        let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if system_paths
            .iter()
            .any(|root| resolved == *root || resolved.starts_with(root))
        {
            let target = fs::read_link(path)?;
            flags.extend([
                "--symlink".into(),
                target.to_string_lossy().into_owned(),
                alias.to_string(),
            ]);
        }
    }
    Ok(flags)
}

fn sandbox_environment(
    environment: &BTreeMap<String, String>,
    cwd: Option<&str>,
    gpu_uuids: &[String],
    discovery: Option<&NvidiaDiscovery>,
) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::from([
        ("HOME".into(), "/home/cape".into()),
        ("PATH".into(), DEFAULT_PATH.into()),
    ]);
    result.extend(environment.clone());
    if let Some(cwd) = cwd {
        result.insert("PWD".into(), cwd.into());
    }
    result.insert("CUDA_VISIBLE_DEVICES".into(), super::leased_gpu_visibility(gpu_uuids, &result)?);
    if !gpu_uuids.is_empty() {
        let discovery = discovery.context("missing cached NVIDIA discovery")?;
        let binary = discovery
            .binary
            .as_ref()
            .context("leased GPU exposure requires a host nvidia-smi executable")?;
        append_paths(&mut result, "PATH", binary.parent().into_iter());
        append_paths(
            &mut result,
            "LD_LIBRARY_PATH",
            discovery.libraries.iter().filter_map(|path| path.parent()),
        );
    }
    Ok(result)
}

fn gpu_mount_flags(gpu_uuids: &[String], discovery: &NvidiaDiscovery) -> Result<Vec<String>> {
    let device_map = &discovery.device_map;
    let unresolved: Vec<&String> = gpu_uuids
        .iter()
        .filter(|uuid| !device_map.contains_key(*uuid))
        .collect();
    if !unresolved.is_empty() {
        bail!(
            "cannot resolve leased GPU UUID(s) to NVIDIA device minors: {}",
            unresolved
                .into_iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let mut paths: Vec<PathBuf> = gpu_uuids
        .iter()
        .map(|uuid| PathBuf::from(format!("/dev/nvidia{}", device_map[uuid])))
        .collect();
    for candidate in [
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
        "/dev/nvidia-caps",
    ] {
        if Path::new(candidate).exists() {
            paths.push(PathBuf::from(candidate));
        }
    }
    let mut flags = Vec::new();
    let mut seen = BTreeSet::new();
    for path in paths {
        if seen.insert(path.clone()) {
            let text = path.to_string_lossy().into_owned();
            flags.extend(["--dev-bind".into(), text.clone(), text]);
        }
    }
    let binary = discovery
        .binary
        .clone()
        .context("leased GPU exposure requires a host nvidia-smi executable")?;
    if discovery.libraries.is_empty() {
        bail!("leased GPU exposure requires host NVIDIA driver libraries");
    }
    for path in std::iter::once(binary).chain(discovery.libraries.clone()) {
        let path = path.canonicalize().unwrap_or(path);
        if seen.insert(path.clone()) {
            let text = path.to_string_lossy().into_owned();
            flags.extend(["--ro-bind".into(), text.clone(), text]);
        }
    }
    if Path::new("/proc/driver/nvidia").exists() {
        flags.extend(strings(&[
            "--ro-bind",
            "/proc/driver/nvidia",
            "/proc/driver/nvidia",
        ]));
    }
    Ok(flags)
}

fn nvidia_discovery(config: &RuntimeConfig, gpu_uuids: &[String]) -> Result<NvidiaDiscovery> {
    let mut cached = config.bubblewrap_gpu_discovery.lock().unwrap();
    let refresh = cached.as_ref().map_or(true, |discovery| {
        !discovery.configured
            && gpu_uuids
                .iter()
                .any(|uuid| !discovery.device_map.contains_key(uuid))
    });
    if refresh {
        *cached = Some(probe_nvidia_discovery()?);
    }
    let discovery = cached.as_ref().unwrap().clone();
    let unresolved: Vec<&str> = gpu_uuids
        .iter()
        .filter(|uuid| !discovery.device_map.contains_key(*uuid))
        .map(String::as_str)
        .collect();
    if !unresolved.is_empty() {
        bail!(
            "cannot resolve leased GPU UUID(s) to NVIDIA device minors: {}",
            unresolved.join(", ")
        );
    }
    Ok(discovery)
}

fn probe_nvidia_discovery() -> Result<NvidiaDiscovery> {
    let mut result = BTreeMap::new();
    let binary = find_executable("nvidia-smi")
        .context("leased GPU exposure requires a host nvidia-smi executable")?;
    if let Ok(entries) = fs::read_dir("/proc/driver/nvidia/gpus") {
        for entry in entries.flatten() {
            let information = entry.path().join("information");
            let Ok(text) = fs::read_to_string(information) else {
                continue;
            };
            let mut uuid = None;
            let mut minor = None;
            for line in text.lines() {
                let Some((key, value)) = line.split_once(':') else {
                    continue;
                };
                match key.trim().to_ascii_lowercase().as_str() {
                    "gpu uuid" => uuid = Some(value.trim().to_string()),
                    "device minor" => minor = value.trim().parse().ok(),
                    _ => {}
                }
            }
            if let (Some(uuid), Some(minor)) = (uuid, minor) {
                result.insert(uuid, minor);
            }
        }
    }
    if let Ok(output) = StdCommand::new(&binary)
        .args(["--query-gpu=uuid,index", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let values: Vec<&str> = line.split(',').map(str::trim).collect();
                if values.len() == 2 {
                    if let Ok(minor) = values[1].parse() {
                        result.entry(values[0].into()).or_insert(minor);
                    }
                }
            }
        }
    }
    let machine = std::env::consts::ARCH;
    let multiarch = match machine {
        "aarch64" => Some("aarch64-linux-gnu"),
        "x86_64" => Some("x86_64-linux-gnu"),
        _ => None,
    };
    let mut directories = vec![PathBuf::from("/usr/lib")];
    if let Some(multiarch) = multiarch {
        directories.push(PathBuf::from(format!("/usr/lib/{multiarch}")));
    }
    directories.extend([
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/lib64"),
        PathBuf::from("/usr/local/nvidia/lib"),
        PathBuf::from("/usr/local/nvidia/lib64"),
        PathBuf::from("/usr/local/cuda/compat/lib"),
        PathBuf::from("/usr/local/cuda/compat/lib.real"),
    ]);
    let mut libraries = Vec::new();
    let mut seen = BTreeSet::new();
    for directory in directories {
        let directory = directory.canonicalize().unwrap_or(directory);
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        let entries: Vec<_> = entries.flatten().collect();
        for prefix in ["libcuda.so", "libnvidia-"] {
            for entry in &entries {
                let name = entry.file_name().to_string_lossy().into_owned();
                if !name.starts_with(prefix) || (prefix == "libnvidia-" && !name.contains(".so")) {
                    continue;
                }
                let path = entry.path().canonicalize().unwrap_or_else(|_| entry.path());
                if path.is_file() && seen.insert(path.clone()) {
                    libraries.push(path);
                }
            }
        }
    }
    if libraries.is_empty() {
        bail!("leased GPU exposure requires host NVIDIA driver libraries");
    }
    Ok(NvidiaDiscovery {
        device_map: result,
        binary: Some(binary),
        libraries,
        configured: false,
    })
}

fn append_paths<'a>(
    environment: &mut BTreeMap<String, String>,
    key: &str,
    paths: impl Iterator<Item = &'a Path>,
) {
    let mut values = Vec::new();
    if let Some(current) = environment.get(key) {
        for value in current.split(':') {
            if !value.is_empty() && !values.iter().any(|item| item == value) {
                values.push(value.into());
            }
        }
    }
    for path in paths {
        let value = path.to_string_lossy().into_owned();
        if !value.is_empty() && !values.contains(&value) {
            values.push(value);
        }
    }
    environment.insert(key.into(), values.join(":"));
}

pub(super) fn layer_paths(
    config: &RuntimeConfig,
    execution_session_id: &str,
    index: usize,
    system_path: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let root = session_root(config, execution_session_id);
    let name = safe_name(system_path.to_string_lossy().trim_matches('/'));
    let key = format!("{index:02}-{name}");
    Ok((
        root.join("upper").join(&key),
        root.join("work").join(&key),
        root.join("merged").join(key),
    ))
}

pub(super) fn session_root(config: &RuntimeConfig, execution_session_id: &str) -> PathBuf {
    config
        .execution_state_dir.as_ref().unwrap_or(&config.bubblewrap_runtime_dir)
        .join("bubblewrap-overlays")
        .join(safe_name(execution_session_id))
}

fn safe_name(value: &str) -> String {
    let value: String = value
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .take(56).collect();
    if value.is_empty() {
        "root".into()
    } else {
        value
    }
}

fn mount_fuse_layers(config: &RuntimeConfig, execution_session_id: &str) -> Result<Vec<FuseMount>> {
    if !session_root(config,execution_session_id).exists() { super::storage::check(config)?; }
    let binary = find_executable("fuse-overlayfs").context("fuse-overlayfs is unavailable")?;
    let system_paths = normalize_system_paths(&config.bubblewrap_system_paths)?;
    let mut mounts = Vec::new();
    for (index, lower) in system_paths.iter().enumerate() {
        let (upper, work, merged) = layer_paths(config, execution_session_id, index, lower)?;
        fs::create_dir_all(&upper)?;
        fs::create_dir_all(&work)?;
        fs::create_dir_all(&merged)?;
        let options = format!(
            "lowerdir={},upperdir={},workdir={}",
            lower.display(),
            upper.display(),
            work.display()
        );
        let child = StdCommand::new(&binary)
            .args(["-f", "-o", &options])
            .arg(&merged)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to start fuse-overlayfs for {}", lower.display()))?;
        let pid = child.id();
        std::mem::forget(child);
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !is_mounted(&merged) {
            std::thread::sleep(Duration::from_millis(20));
        }
        if !is_mounted(&merged) {
            unsafe {
                nix::libc::kill(pid as i32, nix::libc::SIGKILL);
            }
            unmount_fuse_layers(&mounts).ok();
            bail!("timed out mounting fuse-overlayfs at {}", merged.display());
        }
        mounts.push(FuseMount { child: pid, merged });
    }
    Ok(mounts)
}

fn unmount_fuse_layers(mounts: &[FuseMount]) -> Result<()> {
    let binary = find_executable("fusermount3").or_else(|| find_executable("fusermount"));
    let mut errors = Vec::new();
    for mount in mounts.iter().rev() {
        if let Some(binary) = &binary {
            let status = StdCommand::new(binary)
                .arg("-u")
                .arg(&mount.merged)
                .status();
            if !matches!(status, Ok(value) if value.success()) {
                errors.push(format!("failed to unmount {}", mount.merged.display()));
            }
        } else {
            errors.push("fusermount is unavailable".into());
        }
        unsafe {
            nix::libc::kill(mount.child as i32, nix::libc::SIGTERM);
        }
        let _ = nix::sys::wait::waitpid(nix::unistd::Pid::from_raw(mount.child as i32), None);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        bail!(errors.join("; "))
    }
}

fn is_mounted(path: &Path) -> bool {
    let needle = path.to_string_lossy();
    fs::read_to_string("/proc/self/mountinfo")
        .map(|text| {
            text.lines()
                .any(|line| line.split_whitespace().nth(4) == Some(needle.as_ref()))
        })
        .unwrap_or(false)
}

async fn probe_overlay_persistence(config: &RuntimeConfig) -> bool {
    let session = format!("rscape-overlay-probe-{}", std::process::id());
    let target = config
        .bubblewrap_system_paths
        .last()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("/opt"))
        .join(format!(".rscape-overlay-probe-{}", std::process::id()));
    let write = format!(
        "mkdir -p '{}' && printf cape > '{}/value'",
        target.display(),
        target.display()
    );
    let first = timed_run(
        &probe_assignment_with_session(&session, "host-stateful", &[], &["/bin/sh", "-c", &write]),
        config,
        10,
    )
    .await;
    if first.exit_code != 0 {
        cleanup_session(config, &session).ok();
        return false;
    }
    let read = timed_run(
        &probe_assignment_with_session(
            &session,
            "host-stateful",
            &[],
            &["/bin/cat", &format!("{}/value", target.display())],
        ),
        config,
        10,
    )
    .await;
    cleanup_session(config, &session).ok();
    read.exit_code == 0 && read.stdout == "cape"
}

async fn probe_gpu_policy(config: &RuntimeConfig, gpu_uuids: &[String], policy: &str) -> bool {
    let session = format!("rscape-gpu-{policy}-probe-{}", std::process::id());
    let uuids: Vec<&str> = gpu_uuids.iter().map(String::as_str).collect();
    let visibility = timed_run(
        &probe_assignment_with_session(
            &session,
            policy,
            &uuids,
            &[
                "/usr/bin/nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
        ),
        config,
        30,
    )
    .await;
    if visibility.exit_code != 0 {
        cleanup_session(config, &session).ok();
        return false;
    }
    let visible: BTreeSet<String> = visibility
        .stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    if visible != gpu_uuids.iter().cloned().collect() {
        cleanup_session(config, &session).ok();
        return false;
    }
    let Some(python) = probe_python(&config.bubblewrap_system_paths) else {
        cleanup_session(config, &session).ok();
        return false;
    };
    let cuda = "import ctypes,sys; c=ctypes.CDLL('libcuda.so.1'); n=ctypes.c_int(); a=c.cuInit(0); b=c.cuDeviceGetCount(ctypes.byref(n)) if a==0 else a; print(a,b,n.value); sys.exit(0 if a==0 and b==0 and n.value>0 else 1)";
    let initialized = timed_run(
        &probe_assignment_with_session(&session, policy, &uuids, &[&python, "-c", cuda]),
        config,
        30,
    )
    .await;
    if initialized.exit_code != 0 {
        cleanup_session(config, &session).ok();
        return false;
    }
    let kernel = format!("import torch; assert torch.cuda.device_count()=={}; x=torch.arange(32,device='cuda'); assert int((x*x).sum().item())==10416; print('cuda-kernel-ok')", gpu_uuids.len());
    let result = timed_run(
        &probe_assignment_with_session(&session, policy, &uuids, &[&python, "-c", &kernel]),
        config,
        120,
    )
    .await;
    cleanup_session(config, &session).ok();
    result.exit_code == 0
}

fn probe_python(system_paths: &[PathBuf]) -> Option<String> {
    // An explicitly configured toolchain root is the workload's Python; prefer it over the host's, which may lack
    // the CUDA framework the GPU probe imports.
    let mut candidates = Vec::new();
    for root in system_paths {
        candidates.push(root.join("bin/python3"));
        candidates.push(root.join("bin/python"));
        candidates.push(root.join("local/bin/python3"));
        candidates.push(root.join("local/bin/python"));
    }
    candidates.push(PathBuf::from("/usr/bin/python3"));
    candidates.push(PathBuf::from("/usr/local/bin/python3"));
    candidates
        .into_iter()
        .filter_map(|path| path.canonicalize().ok())
        .find(|path| path.is_file() && system_paths.iter().any(|root| path.starts_with(root)))
        .map(|path| path.to_string_lossy().into_owned())
}

fn probe_assignment(policy: &str, gpu_uuids: &[&str], command: &[&str]) -> Value {
    probe_assignment_with_session("rscape-runtime-probe", policy, gpu_uuids, command)
}

fn probe_assignment_with_session(
    execution_session_id: &str,
    policy: &str,
    gpu_uuids: &[&str],
    command: &[&str],
) -> Value {
    json!({
        "execution_session_id": execution_session_id,
        "session_spec": {"runtime_adapter":"bubblewrap","isolation_policy":policy,"gpu_uuids":gpu_uuids,"env":{},"mounts":[]},
        "command_spec": {"command":command,"env":{}}
    })
}

async fn binary_works(binary: &str, args: &[&str]) -> bool {
    matches!(tokio::time::timeout(Duration::from_secs(5), Command::new(binary).args(args).output()).await, Ok(Ok(output)) if output.status.success())
}

async fn timed_run(assignment: &Value, config: &RuntimeConfig, seconds: u64) -> ExecutionResult {
    match tokio::time::timeout(Duration::from_secs(seconds), run(assignment, config)).await {
        Ok(result) => result,
        Err(_) => infrastructure(
            "bubblewrap".into(),
            assignment
                .get("session_spec")
                .and_then(|value| value.get("isolation_policy"))
                .and_then(Value::as_str)
                .unwrap_or("default"),
            format!("timed out after {seconds}s"),
            Vec::new(),
        ),
    }
}

async fn binary_help_has_native_overlay(binary: &str) -> bool {
    match tokio::time::timeout(
        Duration::from_secs(5),
        Command::new(binary).arg("--help").output(),
    )
    .await
    {
        Ok(Ok(output)) if output.status.success() => {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            text.contains("--overlay-src") && text.contains("--overlay")
        }
        _ => false,
    }
}

async fn run_argv(
    argv: &[String],
    config: &RuntimeConfig,
    command_id: Option<&str>,
) -> Result<(i32, String, String)> {
    super::run_process(argv, &BTreeMap::new(), None, config, command_id).await
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let path = Path::new(name);
    if path.components().count() > 1 {
        return path.is_file().then(|| path.to_path_buf());
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|root| root.join(name))
            .find(|candidate| candidate.is_file())
    })
}

fn string_map(value: Option<&Value>) -> BTreeMap<String, String> {
    value
        .and_then(Value::as_object)
        .map(|items| {
            items
                .iter()
                .filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.into())))
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

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

fn infrastructure(
    adapter: String,
    policy: &str,
    stderr: String,
    argv: Vec<String>,
) -> ExecutionResult {
    ExecutionResult {
        exit_code: 127,
        stdout: String::new(),
        stderr,
        failure_kind: "infrastructure".into(),
        argv,
        adapter,
        isolation_policy: policy.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_python_preserves_root_order_and_python_only_toolchains() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("first");
        let second = directory.path().join("second");
        for (root, name) in [(&first, "python"), (&second, "python3")] {
            fs::create_dir_all(root.join("bin")).unwrap();
            let path = root.join("bin").join(name);
            fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
            fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
        }
        assert_eq!(probe_python(&[first.clone(), second.clone()]),
            Some(first.join("bin/python").canonicalize().unwrap().to_string_lossy().into_owned()));
        assert_eq!(probe_python(&[second.clone(), first]),
            Some(second.join("bin/python3").canonicalize().unwrap().to_string_lossy().into_owned()));
    }

    fn config() -> RuntimeConfig {
        RuntimeConfig {
            execution_state_dir: None,
            execution_state_min_free_bytes: 0,
            execution_state_max_bytes: 0,
            adapters: vec!["bubblewrap".into()],
            explicit_adapters: true,
            apptainer_binary: "apptainer".into(),
            bubblewrap_binary: "bwrap".into(),
            podman_binary: "podman".into(),
            apptainer_probe_image: None,
            apptainer_cache_dir: None,
            apptainer_tmp_dir: None,
            apptainer_runtime_dir: PathBuf::from("/tmp/rscape-apptainer-test-runtime"),
            apptainer_overlay_mode: None,
            apptainer_no_contain: false,
            apptainer_unsquash: false,
            podman_probe_image: None,
            podman_graphroot: None,
            podman_runroot: None,
            podman_tmpdir: None,
            bubblewrap_runtime_dir: PathBuf::from("/tmp/rscape-bubblewrap-test-runtime"),
            bubblewrap_system_paths: vec![PathBuf::from("/usr"), PathBuf::from("/etc")],
            bubblewrap_read_only_paths: Vec::new(),
            bubblewrap_user_namespace: Some(false),
            bubblewrap_overlay_mode: None,
            bubblewrap_gpu_discovery: std::sync::Arc::new(std::sync::Mutex::new(None)),
            no_warm_sessions: false,
            capabilities: BTreeMap::new(),
            active_processes: std::sync::Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            cancel_commands: std::sync::Arc::new(std::sync::Mutex::new(BTreeMap::new())),
            cancelled_commands: std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::new())),
            log_dir: PathBuf::from("/tmp"),
        }
    }

    #[test]
    fn stateless_argv_assembles_root_aliases_namespaces_devices_and_environment() {
        let config = config();
        let argv = build_argv(
            &config,
            &["/usr/bin/true".into()],
            Some("/workspace"),
            &BTreeMap::from([("SENTINEL".into(), "explicit".into())]),
            &[json!({"source":"/tmp","target":"/workspace","read_only":false})],
            &[],
            "host-stateless",
            "s",
        )
        .unwrap();
        for expected in [
            "--clearenv",
            "--tmpfs",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--dev",
            "--cap-drop",
            "HOME",
            "/home/cape",
            "PATH",
        ] {
            assert!(
                argv.iter().any(|value| value == expected),
                "missing {expected}: {argv:?}"
            );
        }
        assert!(!argv
            .windows(3)
            .any(|values| values == ["--dev-bind", "/dev", "/dev"]));
        assert!(argv
            .windows(3)
            .any(|values| values[0] == "--symlink" && values[2] == "/bin"));
    }

    #[test]
    fn unresolved_gpu_fails_closed() {
        let config = config();
        *config.bubblewrap_gpu_discovery.lock().unwrap() = Some(NvidiaDiscovery {
            device_map: BTreeMap::new(),
            binary: Some(PathBuf::from("/bin/true")),
            libraries: vec![PathBuf::from("/bin/true")],
            configured: true,
        });
        let error = nvidia_discovery(&config, &["GPU-does-not-exist".into()])
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot resolve leased GPU UUID"));
    }

    #[test]
    fn aggregate_environment_does_not_inherit_agent_values() {
        std::env::set_var("CAPE_AGENT_ONLY_SENTINEL", "secret");
        let environment = sandbox_environment(&BTreeMap::new(), None, &[], None).unwrap();
        assert_eq!(environment.get("HOME").unwrap(), "/home/cape");
        assert_eq!(environment.get("PATH").unwrap(), DEFAULT_PATH);
        assert!(!environment.contains_key("CAPE_AGENT_ONLY_SENTINEL"));
    }

    #[test]
    fn optional_virtual_paths_are_read_only_and_deduplicated() {
        let path = PathBuf::from("/sys/devices/system/cpu");
        let mut configured = config();
        configured.bubblewrap_overlay_mode = Some(OverlayMode::Native);
        configured.bubblewrap_read_only_paths = vec![path.clone(), path];
        for policy in ["host-stateless", "host-stateful"] {
            let argv = build_argv(&configured, &["true".into()], None, &BTreeMap::new(),
                                 &[], &[], policy, "virtual").unwrap();
            assert_eq!(argv.windows(3).filter(|triple|
                triple[0] == "--ro-bind" && triple[1] == "/sys/devices/system/cpu"
                    && triple[2] == "/sys/devices/system/cpu").count(), 1);
            assert!(!argv.iter().any(|value| value == "/sys"));
        }
        for root in ["/sys", "/proc", "/dev", "/run", "/sys/../etc"] {
            assert!(virtual_subpaths(&[PathBuf::from(root)]).is_err());
        }
        assert!(virtual_subpaths(&[PathBuf::from("/sys/cape-nonexistent-path")]).is_ok());
        // /proc/self/root is a real virtual-root escape on Linux.
        assert!(virtual_subpaths(&[PathBuf::from("/proc/self/root/etc")]).is_err());
    }

    #[test]
    fn missing_symlink_targets_are_resolved_before_root_validation() {
        let directory = tempfile::tempdir().unwrap();
        let link = directory.path().join("link");
        let outside = directory.path().join("outside-missing");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        assert_eq!(real_path_allow_missing(&link.join("child")).unwrap(), outside.join("child"));
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink("link", &link).unwrap();
        assert!(real_path_allow_missing(&link).is_err());
    }

    #[test]
    fn driver_paths_follow_explicit_toolchain_and_cpu_visibility_is_empty() {
        let supplied = BTreeMap::from([
            ("PATH".into(), "/chosen/bin".into()),
            ("LD_LIBRARY_PATH".into(), "/chosen/lib".into()),
            ("CUDA_VISIBLE_DEVICES".into(), "caller-value".into()),
        ]);
        let cpu = sandbox_environment(&supplied, None, &[], None).unwrap();
        assert_eq!(cpu["CUDA_VISIBLE_DEVICES"], "");
        let discovery = NvidiaDiscovery {
            binary: Some(PathBuf::from("/usr/bin/nvidia-smi")),
            libraries: vec![PathBuf::from("/driver/libcuda.so")],
            ..NvidiaDiscovery::default()
        };
        assert!(sandbox_environment(&supplied, None, &["GPU-test".into()], Some(&discovery)).is_err());
        let mut gpu_supplied = supplied.clone();
        gpu_supplied.insert("CUDA_VISIBLE_DEVICES".into(), "GPU-test".into());
        let gpu = sandbox_environment(&gpu_supplied, None, &["GPU-test".into()], Some(&discovery)).unwrap();
        assert_eq!(gpu["PATH"], "/chosen/bin:/usr/bin");
        assert_eq!(gpu["LD_LIBRARY_PATH"], "/chosen/lib:/driver");
    }

    #[tokio::test]
    async fn probe_rejects_a_binary_that_ignores_the_sandbox_argv() {
        let mut config = config();
        config.bubblewrap_binary = "/bin/true".into();
        assert!(probe(&mut config, &[]).await.unwrap().is_none());
    }
}
