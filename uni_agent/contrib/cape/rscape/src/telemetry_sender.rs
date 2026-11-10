// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One bounded invocation of the upstream serializer per batch; no persistent queue.

use anyhow::{ensure, Result};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use serde_json::json;

pub fn receiver_url(value: &str) -> Result<()> {
    ensure!(value.len() <= 4096 && !value.chars().any(|c| c < '!'), "invalid telemetry receiver URL");
    let url = url::Url::parse(value)?;
    ensure!(["http", "https"].contains(&url.scheme()) && url.host_str().is_some()
        && url.username().is_empty() && url.password().is_none() && url.query().is_none()
        && url.fragment().is_none() && url.path() == "/api/v1/write"
        && !value.split("://").nth(1).unwrap_or("").split('/').next().unwrap_or("").contains('@'),
        "invalid telemetry receiver URL");
    Ok(())
}

pub fn configuration(source: &Path, receiver: &str, timeout: f64) -> Result<String> {
    receiver_url(receiver)?;
    ensure!(timeout.is_finite() && (0.1..=30.0).contains(&timeout), "invalid sender timeout");
    Ok(format!(r#"[agent]
  omit_hostname = true
  metric_batch_size = 10000
  metric_buffer_limit = 10000
  skip_processors_after_aggregators = true
[[inputs.file]]
  files = [{}]
  data_format = "prometheus"
[[outputs.http]]
  url = {}
  timeout = "{}s"
  data_format = "prometheusremotewrite"
  [outputs.http.headers]
    Content-Type = "application/x-protobuf"
    Content-Encoding = "snappy"
    X-Prometheus-Remote-Write-Version = "0.1.0"
"#, json!(source.to_str().ok_or_else(|| anyhow::anyhow!("non-UTF8 sender path"))?), json!(receiver), timeout))
}

#[derive(Debug, PartialEq)]
pub enum SendOutcome { Empty, Sent, Failed, Timeout, Unavailable }

#[derive(Default)]
pub struct LatestBatch {
    blocks: std::collections::VecDeque<(String, usize)>,
    bytes: usize,
    rows: usize,
}

impl LatestBatch {
    pub fn push(&mut self, block: String) {
        let rows = block.lines().count();
        if block.len() > 4 * 1024 * 1024 || rows > 10000 { return; }
        self.bytes += block.len();
        self.rows += rows;
        self.blocks.push_back((block, rows));
        while self.bytes > 4 * 1024 * 1024 || self.rows > 10000 {
            if let Some((old, rows)) = self.blocks.pop_front() {
                self.bytes -= old.len();
                self.rows -= rows;
            }
        }
    }

    pub fn exposition(self) -> String {
        self.blocks.into_iter().map(|(block, _)| block).collect()
    }
}

pub fn send(receiver: &str, binary: &Path, exposition: &str, timeout: f64, runtime_dir: &Path) -> Result<SendOutcome> {
    use std::os::unix::fs::DirBuilderExt;
    receiver_url(receiver)?;
    ensure!(timeout.is_finite() && (0.1..=30.0).contains(&timeout), "invalid sender timeout");
    ensure!(exposition.len() <= 4 * 1024 * 1024 && exposition.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).count() <= 10000,
        "telemetry batch exceeds limit");
    if exposition.trim().is_empty() { return Ok(SendOutcome::Empty); }
    let root = runtime_dir.join(format!("cape-metric-batch-{}", uuid::Uuid::new_v4()));
    if std::fs::DirBuilder::new().mode(0o700).create(&root).is_err() { return Ok(SendOutcome::Unavailable); }
    // Only our exclusively-created private directory is removed by this guard.
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
    let _cleanup = Cleanup(root.clone());
    let source = root.join("snapshot.prom");
    if std::fs::write(&source, exposition).is_err() { return Ok(SendOutcome::Unavailable); }
    let config = configuration(&source, receiver, timeout)?;
    let config_path = root.join("sender.toml");
    if std::fs::write(&config_path, config).is_err() { return Ok(SendOutcome::Unavailable); }
    let mut command = Command::new(binary);
    command.arg("--once").arg("--config").arg(&config_path).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    for (key, _) in std::env::vars_os() {
        let name = key.to_string_lossy();
        if ["CAPE_", "INFLUX_", "TELEGRAF_"].iter().any(|prefix| name.starts_with(prefix)) { command.env_remove(key); }
    }
    let mut child = match command.spawn() { Ok(child) => child, Err(_) => return Ok(SendOutcome::Unavailable) };
    let deadline = Instant::now() + Duration::from_secs_f64(timeout + 1.0);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(if status.success() { SendOutcome::Sent } else { SendOutcome::Failed }),
            Err(_) => { let _ = child.kill(); let _ = child.wait(); return Ok(SendOutcome::Failed); }
            _ => (),
        }
        if Instant::now() >= deadline { let _ = child.kill(); let _ = child.wait(); return Ok(SendOutcome::Timeout); }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overflow_discards_oldest_complete_gpu_sample() {
        let mut batch=LatestBatch::default();
        for i in 0..5001 {batch.push(format!("value_{i} 1\nexpiry_{i} 2\n"));}
        let output=batch.exposition();
        assert_eq!(output.lines().count(),10000);
        assert!(!output.contains("value_0 "));
        assert!(output.contains("value_5000 1\nexpiry_5000 2\n"));
        let mut bytes=LatestBatch::default();
        bytes.push(format!("old {}\n", "x".repeat(3*1024*1024)));
        bytes.push(format!("new {}\n", "x".repeat(2*1024*1024)));
        let output=bytes.exposition();
        assert!(output.starts_with("new "));
        assert!(output.len()<=4*1024*1024);
    }
    #[test]
    fn no_credentials_or_arbitrary_paths() {
        for url in ["http://u:p@host/api/v1/write", "file:///api/v1/write", "http://host/metrics",
                    "http://host/api/v1/write?secret=x", "http://host/api/v1/write#x", "http://@host/api/v1/write"] {
            assert!(receiver_url(url).is_err(), "accepted {url}");
        }
        receiver_url("http://hostname:9090/api/v1/write").unwrap();
        let config = configuration(Path::new("/tmp/a\"b"), "http://host/api/v1/write", 2.0).unwrap();
        assert!(config.contains("prometheusremotewrite") && !config.contains("buffer_directory"));
    }

    #[test]
    fn missing_sender_discards_private_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(send("http://host/api/v1/write", Path::new("/missing/cape-telegraf"), "x 1\n", 1.0, dir.path()).unwrap(), SendOutcome::Unavailable);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        assert_eq!(send("http://host/api/v1/write", Path::new("/missing/cape-telegraf"), "", 1.0, dir.path()).unwrap(), SendOutcome::Empty);
    }

    #[test]
    fn hung_sender_is_reaped_and_snapshot_removed() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("hung-sender");
        std::fs::write(&binary, "#!/bin/sh\nexec sleep 30\n").unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        let started = Instant::now();
        assert_eq!(send("http://host/api/v1/write", &binary, "x 1\n", 0.1, dir.path()).unwrap(), SendOutcome::Timeout);
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1); // Only our test executable remains.
    }
}
