// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{ExecutionResult, RuntimeConfig};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;

pub async fn probe(config: &mut RuntimeConfig, gpu_uuids: &[String]) -> Result<Option<Value>> {
    let Some(image) = config.apptainer_probe_image.clone() else {
        return Ok(None);
    };
    if !binary_works(&config.apptainer_binary, &["--version"]).await {
        return Ok(None);
    }
    fs::create_dir_all(&config.apptainer_runtime_dir)?;
    for path in [&config.apptainer_cache_dir, &config.apptainer_tmp_dir].into_iter().flatten() {
        fs::create_dir_all(path)?;
    }
    config.apptainer_overlay_mode = None;
    let stateless = probe_assignment(
        "apptainer-stateless-probe",
        "image-stateless",
        &image,
        &[],
        &["/bin/true"],
    );
    let stateless_result = timed_run(&stateless, config, 300).await;
    if stateless_result.exit_code != 0 {
        eprintln!("Apptainer stateless readiness failed: {}", stateless_result.stderr);
        return Ok(None);
    }
    if !config.no_warm_sessions {
        for mode in ["directory", "ext3"] {
            config.apptainer_overlay_mode = Some(mode.into());
            if probe_overlay(config, &image).await {
                break;
            }
            config.apptainer_overlay_mode = None;
        }
    }
    let mut policies = vec!["default", "image-stateless"];
    if config.apptainer_overlay_mode.is_some() {
        policies.push("image-stateful");
    }
    let _ = gpu_uuids;
    let gpu_policies = policies.clone();
    Ok(Some(json!({
        "adapter":"apptainer",
        "image_formats":["default","docker","sif","sandbox","oci"],
        "isolation_policies":policies,
        "nested_container":true,
        "can_bind_mounts":true,
        "can_select_gpus":true,
        "gpu_enforcement":"advisory",
        "can_enforce_cpu_memory":false,
        "node_local_cache":true,
        "supports_warm_sessions":false,
        "privileged_allowed":false,
        "gpu_isolation_policies":gpu_policies,
        "reduced_isolation_policies":[],
    })))
}

pub async fn run(assignment: &Value, config: &RuntimeConfig) -> ExecutionResult {
    let session = assignment
        .get("session_spec")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let command_spec = assignment
        .get("command_spec")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let requested = session
        .get("isolation_policy")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let policy = match resolved_policy(requested, config.apptainer_overlay_mode.as_deref()) {
        Ok(value) => value,
        Err(error) => return infrastructure(requested, error.to_string(), Vec::new()),
    };
    let command = string_list(command_spec.get("command"));
    if command.is_empty() {
        return infrastructure(&policy, "empty workload command".into(), Vec::new());
    }
    let execution_session_id = assignment
        .get("execution_session_id")
        .and_then(Value::as_str)
        .unwrap_or("runtime-session");
    if policy == "image-stateful" {
        if let Err(error) = provision_overlay(config, execution_session_id).await {
            return infrastructure(&policy, error.to_string(), Vec::new());
        }
    }
    let image = session
        .get("image")
        .and_then(Value::as_str)
        .unwrap_or("docker://ubuntu:latest");
    let cwd = command_spec
        .get("cwd")
        .and_then(Value::as_str)
        .or_else(|| session.get("cwd").and_then(Value::as_str));
    let command_environment = string_map(command_spec.get("env"));
    let mut launch_environment = string_map(session.get("env"));
    launch_environment.extend(command_environment.clone());
    if let Some(path) = &config.apptainer_cache_dir {
        launch_environment.insert(
            "APPTAINER_CACHEDIR".into(),
            path.to_string_lossy().into_owned(),
        );
    }
    if let Some(path) = &config.apptainer_tmp_dir {
        launch_environment.insert(
            "APPTAINER_TMPDIR".into(),
            path.to_string_lossy().into_owned(),
        );
    }
    launch_environment.insert(
        "APPTAINER_LOCALCACHEDIR".into(),
        config.apptainer_runtime_dir.to_string_lossy().into_owned(),
    );
    let gpu_uuids = string_list(session.get("gpu_uuids"));
    let visibility = match super::leased_gpu_visibility(&gpu_uuids, &launch_environment) {
        Ok(value) => value,
        Err(error) => {
            let mut result = infrastructure(&policy, error.to_string(), Vec::new());
            result.failure_kind = "workload".into();
            return result;
        }
    };
    launch_environment.insert("CUDA_VISIBLE_DEVICES".into(), visibility);
    let mounts = session
        .get("mounts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let argv = match build_argv(
        config,
        execution_session_id,
        image,
        &command,
        cwd,
        &launch_environment,
        &mounts,
        &gpu_uuids,
        &policy,
    ) {
        Ok(value) => value,
        Err(error) => return infrastructure(&policy, error.to_string(), Vec::new()),
    };
    match super::run_process(
        &argv,
        &launch_environment,
        None,
        config,
        assignment.get("command_id").and_then(Value::as_str),
    )
    .await
    {
        Ok((exit_code, stdout, stderr)) => ExecutionResult {
            exit_code,
            stdout,
            stderr,
            failure_kind: "workload".into(),
            argv,
            adapter: "apptainer".into(),
            isolation_policy: policy,
        },
        Err(error) => infrastructure(&policy, error.to_string(), argv),
    }
}

pub fn build_argv(
    config: &RuntimeConfig,
    execution_session_id: &str,
    image: &str,
    command: &[String],
    cwd: Option<&str>,
    env: &BTreeMap<String, String>,
    mounts: &[Value],
    gpu_uuids: &[String],
    policy: &str,
) -> Result<Vec<String>> {
    let mut argv = vec![config.apptainer_binary.clone(), "exec".into()];
    if !config.apptainer_no_contain {
        argv.push("--contain".into());
    }
    if config.apptainer_unsquash {
        argv.push("--unsquash".into());
    }
    if policy == "image-stateful" {
        argv.extend([
            "--overlay".into(),
            overlay_path(config, execution_session_id)
                .to_string_lossy()
                .into_owned(),
        ]);
    }
    if !gpu_uuids.is_empty() {
        argv.push("--nv".into());
    }
    for mount in mounts {
        let source = mount
            .get("source")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("mount source is required"))?;
        let target = mount
            .get("target")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("mount target is required"))?;
        let mode = if mount
            .get("read_only")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            "ro"
        } else {
            "rw"
        };
        argv.extend(["--bind".into(), format!("{source}:{target}:{mode}")]);
    }
    if let Some(cwd) = cwd {
        argv.extend(["--pwd".into(), cwd.into()]);
    }
    let mut command_env = env.clone();
    command_env.insert("CUDA_VISIBLE_DEVICES".into(), super::leased_gpu_visibility(gpu_uuids, &command_env)?);
    for (key, value) in command_env {
        argv.extend(["--env".into(), format!("{key}={value}")]);
    }
    argv.push(image.into());
    argv.extend(command.iter().cloned());
    Ok(argv)
}

pub fn cleanup_session(config: &RuntimeConfig, execution_session_id: &str) -> Result<()> {
    let path = overlay_path(config, execution_session_id);
    if path.is_dir() {
        fs::remove_dir_all(path)?;
    } else if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

fn resolved_policy(policy: &str, overlay: Option<&str>) -> Result<String> {
    match policy {
        "default" => Ok(if overlay.is_some() {
            "image-stateful"
        } else {
            "image-stateless"
        }
        .into()),
        "image-stateless" => Ok(policy.into()),
        "image-stateful" if overlay.is_some() => Ok(policy.into()),
        "image-stateful" => {
            bail!("rootless Apptainer writable overlay was not proven on this node")
        }
        _ => bail!("apptainer cannot satisfy isolation policy: {policy}"),
    }
}
pub(super) fn overlay_path(config: &RuntimeConfig, execution_session_id: &str) -> PathBuf {
    let root = config.execution_state_dir.as_ref().unwrap_or(&config.apptainer_runtime_dir).join("apptainer-overlays");
    if config.apptainer_overlay_mode.as_deref() == Some("ext3") {
        root.join(format!("{}.img", safe_name(execution_session_id)))
    } else {
        root.join(safe_name(execution_session_id))
    }
}
async fn provision_overlay(config: &RuntimeConfig, execution_session_id: &str) -> Result<()> {
    let path = overlay_path(config, execution_session_id);
    if path.exists() {
        return Ok(());
    }
    super::storage::check(config)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if config.apptainer_overlay_mode.as_deref() == Some("ext3") {
        let output = Command::new(&config.apptainer_binary)
            .args(["overlay", "create", "--sparse", "--size", "64"])
            .arg(&path)
            .output()
            .await?;
        if !output.status.success() {
            bail!(
                "Apptainer ext3 overlay creation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        fs::create_dir_all(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o777))?;
        }
    }
    Ok(())
}
async fn probe_overlay(config: &RuntimeConfig, image: &str) -> bool {
    let session = format!("apptainer-overlay-probe-{}", std::process::id());
    let target = format!("/opt/.rscape-apptainer-probe-{}", std::process::id());
    let first = timed_run(
        &probe_assignment(
            &session,
            "image-stateful",
            image,
            &[],
            &[
                "/bin/sh",
                "-c",
                &format!("mkdir -p {target} && printf cape > {target}/value"),
            ],
        ),
        config,
        30,
    )
    .await;
    let second = if first.exit_code == 0 {
        timed_run(
            &probe_assignment(
                &session,
                "image-stateful",
                image,
                &[],
                &["/bin/cat", &format!("{target}/value")],
            ),
            config,
            30,
        )
        .await
    } else {
        infrastructure("image-stateful", String::new(), Vec::new())
    };
    cleanup_session(config, &session).ok();
    second.exit_code == 0 && second.stdout == "cape"
}
fn probe_assignment(
    session: &str,
    policy: &str,
    image: &str,
    gpus: &[&str],
    command: &[&str],
) -> Value {
    json!({"execution_session_id":session,"session_spec":{"runtime_adapter":"apptainer","isolation_policy":policy,"image":image,"gpu_uuids":gpus,"env":{},"mounts":[]},"command_spec":{"command":command,"env":{}}})
}
async fn binary_works(binary: &str, args: &[&str]) -> bool {
    matches!(tokio::time::timeout(Duration::from_secs(5),Command::new(binary).args(args).output()).await,Ok(Ok(output)) if output.status.success())
}
async fn timed_run(assignment: &Value, config: &RuntimeConfig, seconds: u64) -> ExecutionResult {
    match tokio::time::timeout(Duration::from_secs(seconds), run(assignment, config)).await {
        Ok(result) => result,
        Err(_) => infrastructure(
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
fn safe_name(value: &str) -> String {
    let value: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .take(56)
        .collect();
    if value.is_empty() {
        "session".into()
    } else {
        value
    }
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
fn infrastructure(policy: &str, stderr: String, argv: Vec<String>) -> ExecutionResult {
    ExecutionResult {
        exit_code: 127,
        stdout: String::new(),
        stderr,
        failure_kind: "infrastructure".into(),
        argv,
        adapter: "apptainer".into(),
        isolation_policy: policy.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stateless_argv_matches_contained_fresh_exec() {
        let config = RuntimeConfig::test_config("apptainer");
        let argv = build_argv(
            &config,
            "s",
            "docker://image",
            &["true".into()],
            Some("/work"),
            &BTreeMap::new(),
            &[json!({"source":"/src","target":"/dst","read_only":false})],
            &[],
            "image-stateless",
        )
        .unwrap();
        assert_eq!(&argv[..3], ["apptainer", "exec", "--contain"]);
        assert!(argv.windows(2).any(|v| v == ["--bind", "/src:/dst:rw"]));
        assert!(!argv.iter().any(|v| v == "--nv"));
    }
}
