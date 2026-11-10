// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{ExecutionResult, RuntimeConfig};
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::process::Command as StdCommand;
use std::time::Duration;
use tokio::process::Command;

pub async fn probe(config: &RuntimeConfig, gpu_uuids: &[String]) -> Result<Option<Value>> {
    let Some(image) = config.podman_probe_image.clone() else {
        return Ok(None);
    };
    if !binary_works(&config.podman_binary, &["--version"]).await {
        return Ok(None);
    }
    for path in [
        &config.podman_graphroot,
        &config.podman_runroot,
        &config.podman_tmpdir,
    ] {
        if let Some(path) = path {
            fs::create_dir_all(path)?;
        }
    }
    let stateless = probe_assignment(
        "podman-stateless-probe",
        "image-stateless",
        &image,
        &[],
        &["/bin/true"],
    );
    if timed_run(&stateless, config, 300).await.exit_code != 0 {
        return Ok(None);
    }
    let policies = vec!["default", "image-stateless"];
    let _ = gpu_uuids;
    let gpu_policies = policies.clone();
    Ok(Some(
        json!({"adapter":"podman","image_formats":["default","docker","oci"],"isolation_policies":policies,"nested_container":true,"can_bind_mounts":true,"can_select_gpus":true,"gpu_enforcement":"advisory","can_enforce_cpu_memory":false,"node_local_cache":config.podman_graphroot.is_some()||config.podman_runroot.is_some()||config.podman_tmpdir.is_some(),"supports_warm_sessions":policies.contains(&"image-stateful"),"privileged_allowed":false,"gpu_isolation_policies":gpu_policies,"reduced_isolation_policies":[]}),
    ))
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
    let policy = match resolved_policy(requested, config.no_warm_sessions) {
        Ok(value) => value,
        Err(error) => return infrastructure(requested, error.to_string(), Vec::new()),
    };
    let mut command = string_list(command_spec.get("command"));
    if command.is_empty() {
        return infrastructure(&policy, "empty workload command".into(), Vec::new());
    }
    let execution_session_id = assignment
        .get("execution_session_id")
        .and_then(Value::as_str)
        .unwrap_or("runtime-session");
    let command_id = assignment.get("command_id").and_then(Value::as_str);
    let image = session
        .get("image")
        .and_then(Value::as_str)
        .unwrap_or("docker.io/library/ubuntu:latest")
        .trim_start_matches("docker://");
    let cwd = command_spec
        .get("cwd")
        .and_then(Value::as_str)
        .or_else(|| session.get("cwd").and_then(Value::as_str));
    let mut session_env = string_map(session.get("env"));
    let mut command_env = string_map(command_spec.get("env"));
    session_env.extend(command_env.clone());
    if let Some(path) = &config.podman_graphroot {
        session_env.insert(
            "PYCAPE_PODMAN_ROOT".into(),
            path.to_string_lossy().into_owned(),
        );
    }
    if let Some(path) = &config.podman_runroot {
        session_env.insert(
            "PYCAPE_PODMAN_RUNROOT".into(),
            path.to_string_lossy().into_owned(),
        );
    }
    if let Some(path) = &config.podman_tmpdir {
        session_env.insert("TMPDIR".into(), path.to_string_lossy().into_owned());
    }
    let gpu_uuids = string_list(session.get("gpu_uuids"));
    let visibility = match super::leased_gpu_visibility(&gpu_uuids, &session_env) {
        Ok(value) => value,
        Err(error) => {
            let mut result = infrastructure(&policy, error.to_string(), Vec::new());
            result.failure_kind = "workload".into();
            return result;
        }
    };
    command_env.insert("CUDA_VISIBLE_DEVICES".into(), visibility.clone());
    session_env.insert("CUDA_VISIBLE_DEVICES".into(), visibility);
    let mounts = session
        .get("mounts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if policy == "image-stateful" {
        if let Err(error) = ensure_container(
            config,
            execution_session_id,
            image,
            session.get("cwd").and_then(Value::as_str),
            &session_env,
            &mounts,
        )
        .await
        {
            return infrastructure(&policy, error.to_string(), Vec::new());
        }
        if let Some(command_id) = command_id {
            command = wrapped_stateful_command(command_id, &command);
        }
    }
    let argv = match build_argv(
        config,
        execution_session_id,
        image,
        &command,
        cwd,
        &command_env,
        &session_env,
        &mounts,
        &policy,
    ) {
        Ok(value) => value,
        Err(error) => return infrastructure(&policy, error.to_string(), Vec::new()),
    };
    if let Some(command_id) = command_id {
        config.cancel_commands.lock().unwrap().insert(
            command_id.into(),
            cancel_argv(config, execution_session_id, &policy, command_id),
        );
    }
    let output = super::run_process(&argv, &BTreeMap::new(), None, config, command_id).await;
    if let Some(command_id) = command_id {
        config.cancel_commands.lock().unwrap().remove(command_id);
    }
    match output {
        Ok((exit_code, stdout, stderr)) => ExecutionResult {
            exit_code,
            stdout,
            stderr,
            failure_kind: "workload".into(),
            argv,
            adapter: "podman".into(),
            isolation_policy: policy,
        },
        Err(error) => infrastructure(&policy, error.to_string(), argv),
    }
}

fn wrapped_stateful_command(command_id: &str, command: &[String]) -> Vec<String> {
    let pidfile = command_pidfile(command_id);
    let mut wrapped = vec![
        "/bin/sh".into(),
        "-c".into(),
        "pidfile=\"$1\"; shift; echo \"$$\" > \"$pidfile\"; exec \"$@\"".into(),
        "cape-command".into(),
        pidfile,
    ];
    wrapped.extend(command.iter().cloned());
    wrapped
}

fn command_pidfile(command_id: &str) -> String {
    format!("/tmp/.cape-command-{}.pid", safe_name(command_id))
}

fn cancel_argv(
    config: &RuntimeConfig,
    execution_session_id: &str,
    policy: &str,
    command_id: &str,
) -> Vec<String> {
    let container = container_name(execution_session_id);
    let mut argv = base_args(config);
    if policy == "image-stateful" {
        let script = r#"i=0
while [ ! -s "$1" ] && [ "$i" -lt 100 ]; do i=$((i + 1)); sleep 0.02; done
[ -s "$1" ] || exit 0
root=$(cat "$1")
descend() {
  for child in $(cat "/proc/$1/task/$1/children" 2>/dev/null); do descend "$child"; done
  echo "$1"
}
pids=$(descend "$root")
for pid in $pids; do kill -TERM "$pid" 2>/dev/null || true; done
sleep 0.2
for pid in $pids; do kill -KILL "$pid" 2>/dev/null || true; done
rm -f "$1""#;
        argv.extend([
            "exec".into(),
            container,
            "/bin/sh".into(),
            "-c".into(),
            script.into(),
            "cape-cancel".into(),
            command_pidfile(command_id),
        ]);
    } else {
        argv.extend(["rm".into(), "-f".into(), container]);
    }
    argv
}

pub fn build_argv(
    config: &RuntimeConfig,
    execution_session_id: &str,
    image: &str,
    command: &[String],
    cwd: Option<&str>,
    command_env: &BTreeMap<String, String>,
    merged_env: &BTreeMap<String, String>,
    mounts: &[Value],
    policy: &str,
) -> Result<Vec<String>> {
    let mut argv = base_args(config);
    if policy == "image-stateful" {
        argv.push("exec".into());
        for (key, value) in command_env {
            argv.extend(["--env".into(), format!("{key}={value}")]);
        }
        if let Some(cwd) = cwd {
            argv.extend(["--workdir".into(), cwd.into()]);
        }
        argv.push(container_name(execution_session_id));
        argv.extend(command.iter().cloned());
        return Ok(argv);
    }
    argv.extend([
        "run".into(),
        "--rm".into(),
        "--name".into(),
        container_name(execution_session_id),
    ]);
    argv.extend(container_flags(cwd, merged_env, mounts)?);
    argv.push(image.trim_start_matches("docker://").into());
    argv.extend(command.iter().cloned());
    Ok(argv)
}

pub fn cleanup_session(config: &RuntimeConfig, execution_session_id: &str) -> Result<()> {
    if !config.adapters.iter().any(|adapter| adapter == "podman") { return Ok(()); }
    let mut argv = base_args(config);
    argv.extend(["rm".into(), "-f".into(), "--ignore".into(), container_name(execution_session_id)]);
    let output = StdCommand::new("timeout").args(["--signal=TERM", "--kill-after=5s", "15s"])
        .arg(&argv[0]).args(&argv[1..]).output()?;
    if !output.status.success() { bail!("Podman session cleanup failed: {}", String::from_utf8_lossy(&output.stderr)); }
    Ok(())
}

pub fn workspace_transition(config: &RuntimeConfig, execution_session_id: &str, policy: &str, resume: bool) -> Result<()> {
    if resolved_policy(policy,config.no_warm_sessions)? != "image-stateful" {return Ok(());}
    let mut argv=base_args(config);
    if resume {argv.extend(["start".into(),container_name(execution_session_id)]);}
    else {argv.extend(["stop".into(),"--time".into(),"5".into(),container_name(execution_session_id)]);}
    let output=StdCommand::new("timeout").args(["--signal=TERM","--kill-after=5s","30s"])
        .arg(&argv[0]).args(&argv[1..]).output()?;
    if !output.status.success(){bail!("Podman workspace transition failed: {}",String::from_utf8_lossy(&output.stderr));}
    Ok(())
}

async fn ensure_container(
    config: &RuntimeConfig,
    execution_session_id: &str,
    image: &str,
    cwd: Option<&str>,
    env: &BTreeMap<String, String>,
    mounts: &[Value],
) -> Result<()> {
    let name = container_name(execution_session_id);
    let mut inspect = base_args(config);
    inspect.extend(["container".into(), "exists".into(), name.clone()]);
    if Command::new(&inspect[0])
        .args(&inspect[1..])
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Ok(());
    }
    let mut argv = base_args(config);
    argv.extend(["run".into(), "-d".into(), "--name".into(), name]);
    argv.extend(container_flags(cwd, env, mounts)?);
    argv.push(image.trim_start_matches("docker://").into());
    argv.extend(["sleep".into(), "infinity".into()]);
    let output = Command::new(&argv[0]).args(&argv[1..]).output().await?;
    if !output.status.success() {
        bail!(
            "Podman stateful container start failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}
fn container_flags(
    cwd: Option<&str>,
    env: &BTreeMap<String, String>,
    mounts: &[Value],
) -> Result<Vec<String>> {
    let mut flags = vec![
        "--userns=keep-id".into(),
        "--user".into(),
        format!("{}:{}", unsafe { nix::libc::geteuid() }, unsafe {
            nix::libc::getegid()
        }),
    ];
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
        flags.extend(["-v".into(), format!("{source}:{target}:{mode}")]);
    }
    if let Some(cwd) = cwd {
        flags.extend(["--workdir".into(), cwd.into()]);
    }
    for (key, value) in env {
        flags.extend(["--env".into(), format!("{key}={value}")]);
    }
    Ok(flags)
}
fn base_args(config: &RuntimeConfig) -> Vec<String> {
    let mut args = vec![config.podman_binary.clone()];
    if let Some(path) = &config.podman_graphroot {
        args.extend(["--root".into(), path.to_string_lossy().into_owned()]);
    }
    if let Some(path) = &config.podman_runroot {
        args.extend(["--runroot".into(), path.to_string_lossy().into_owned()]);
    }
    if let Some(path) = &config.podman_tmpdir {
        args.extend(["--tmpdir".into(), path.to_string_lossy().into_owned()]);
    }
    args
}
fn resolved_policy(policy: &str, _no_warm: bool) -> Result<String> {
    match policy {
        "default" => Ok("image-stateless".into()),
        "image-stateless" => Ok(policy.into()),
        "image-stateful" => bail!("Podman image-stateful sessions are disabled"),
        _ => bail!("podman cannot satisfy isolation policy: {policy}"),
    }
}
fn container_name(session: &str) -> String {
    format!("pycape-{}", safe_name(session))
}
fn safe_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .take(56)
        .collect()
}
fn probe_assignment(
    session: &str,
    policy: &str,
    image: &str,
    gpus: &[&str],
    command: &[&str],
) -> Value {
    json!({"execution_session_id":session,"session_spec":{"runtime_adapter":"podman","isolation_policy":policy,"image":image,"gpu_uuids":gpus,"env":{},"mounts":[]},"command_spec":{"command":command,"env":{}}})
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
        adapter: "podman".into(),
        isolation_policy: policy.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stateless_and_stateful_argv_match_reference_shapes() {
        let config = RuntimeConfig::test_config("podman");
        let stateless = build_argv(
            &config,
            "s",
            "image",
            &["true".into()],
            None,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &[],
            "image-stateless",
        )
        .unwrap();
        assert!(stateless
            .windows(4)
            .any(|v| v == ["run", "--rm", "--name", "pycape-s"]));
        let stateful = build_argv(
            &config,
            "s",
            "image",
            &["true".into()],
            None,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &[],
            "image-stateful",
        )
        .unwrap();
        assert_eq!(&stateful[..3], ["podman", "exec", "pycape-s"]);
    }

    #[test]
    fn cancellation_targets_container_or_inner_pidfile_by_policy() {
        let config = RuntimeConfig::test_config("podman");
        let wrapped = wrapped_stateful_command("cmd-1", &["sleep".into(), "300".into()]);
        assert_eq!(
            &wrapped[..5],
            [
                "/bin/sh",
                "-c",
                "pidfile=\"$1\"; shift; echo \"$$\" > \"$pidfile\"; exec \"$@\"",
                "cape-command",
                "/tmp/.cape-command-cmd-1.pid",
            ]
        );
        let stateful = cancel_argv(&config, "session", "image-stateful", "cmd-1");
        assert_eq!(&stateful[..3], ["podman", "exec", "pycape-session"]);
        assert!(stateful
            .iter()
            .any(|value| value == "/tmp/.cape-command-cmd-1.pid"));
        let stateless = cancel_argv(&config, "session", "image-stateless", "cmd-1");
        assert_eq!(stateless, ["podman", "rm", "-f", "pycape-session"]);
    }
}
