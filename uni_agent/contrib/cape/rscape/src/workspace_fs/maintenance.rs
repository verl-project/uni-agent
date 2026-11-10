// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct MaintenancePolicy {
    pub auto_grow: bool,
    pub grow_threshold_percent: u64,
    pub grow_target_percent: u64,
    pub grow_min_bytes: u64,
    pub grow_cooldown_seconds: u64,
    pub reclaim_on_release: bool,
    pub reclaim_min_allocated_bytes: u64,
    pub reclaim_cooldown_seconds: u64,
    pub reclaim_idle_seconds: u64,
    pub reclaim_start_interval_seconds: u64,
    pub reclaim_max_candidates: u64,
    pub reclaim_retry_backoff_seconds: u64,
    pub reclaim_low_yield_backoff_seconds: u64,
    pub reclaim_nice: u64,
    pub timeout_seconds: u64,
    pub backing_reserve_bytes: u64,
}

impl Default for MaintenancePolicy {
    fn default() -> Self {
        Self {
            auto_grow: false,
            grow_threshold_percent: 80,
            grow_target_percent: 60,
            grow_min_bytes: 1024 * 1024 * 1024,
            grow_cooldown_seconds: 60,
            reclaim_on_release: false,
            reclaim_min_allocated_bytes: 64 * 1024 * 1024,
            reclaim_cooldown_seconds: 300,
            reclaim_idle_seconds: 60,
            reclaim_start_interval_seconds: 30,
            reclaim_max_candidates: 32,
            reclaim_retry_backoff_seconds: 300,
            reclaim_low_yield_backoff_seconds: 3600,
            reclaim_nice: 10,
            timeout_seconds: 300,
            backing_reserve_bytes: 512 * 1024 * 1024,
        }
    }
}

impl MaintenancePolicy {
    pub fn from_value(value: Value) -> Result<Self> {
        let result: Self = serde_json::from_value(value)?;
        if !(1..100).contains(&result.grow_threshold_percent)
            || result.grow_target_percent == 0
            || result.grow_target_percent >= result.grow_threshold_percent
            || result.grow_min_bytes < 32 * 1024 * 1024
            || !(1..=3600).contains(&result.timeout_seconds)
            || !(1..=1024).contains(&result.reclaim_max_candidates)
            || result.reclaim_nice > 19
        {
            bail!("invalid workspace maintenance policy limits");
        }
        for value in [
            result.grow_min_bytes,
            result.grow_cooldown_seconds,
            result.reclaim_min_allocated_bytes,
            result.reclaim_cooldown_seconds,
            result.reclaim_idle_seconds,
            result.reclaim_start_interval_seconds,
            result.reclaim_retry_backoff_seconds,
            result.reclaim_low_yield_backoff_seconds,
            result.backing_reserve_bytes,
        ] {
            if value > i64::MAX as u64 {
                bail!("maintenance policy integer overflow");
            }
        }
        Ok(result)
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn checked_file(path: &Path, device: Option<u64>, inode: Option<u64>) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(path)?;
    let info = file.metadata()?;
    if !info.is_file()
        || info.uid() != uid()
        || info.nlink() != 1
        || info.mode() & 0o077 != 0
        || device.map(|v| v != info.dev()).unwrap_or(false)
        || inode.map(|v| v != info.ino()).unwrap_or(false)
    {
        bail!("unsafe maintenance image identity or ownership");
    }
    Ok(file)
}

pub(super) fn geometry(path: &Path, lease: &File, timeout: u64) -> Result<Value> {
    let (_, text, _) = run_tool(
        "dumpe2fs",
        &["-h".into(), path.display().to_string()],
        timeout,
        true,
        Some(lease.as_raw_fd()),
    )?;
    let fields: BTreeMap<&str, &str> = text
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(a, b)| (a, b.trim()))
        .collect();
    let mut result = json!({});
    for (key, field) in [
        ("block_size", "Block size"),
        ("blocks", "Block count"),
        ("free_blocks", "Free blocks"),
        ("inodes", "Inode count"),
        ("free_inodes", "Free inodes"),
        ("blocks_per_group", "Blocks per group"),
    ] {
        result[key] = json!(fields
            .get(field)
            .context("missing ext4 geometry")?
            .parse::<u64>()?);
    }
    let block = result["block_size"].as_u64().unwrap();
    if ![1024, 2048, 4096, 8192, 16384, 32768, 65536].contains(&block) {
        bail!("invalid ext4 block size");
    }
    result["size_bytes"] = json!(block
        .checked_mul(result["blocks"].as_u64().unwrap())
        .context("geometry overflow")?);
    Ok(result)
}

fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
fn publish(
    manager: &WorkspaceManager,
    directory: &Path,
    manifest: &mut WorkspaceManifest,
) -> Result<()> {
    if let Some(operation) = manifest.maintenance.as_mut() {
        let mut extra = 0_u64;
        for name in ["maintenance-candidate.img", "maintenance-backup.img"] {
            let path = directory.join(name);
            if exists(&path) {
                extra = extra
                    .saturating_add(checked_file(&path, None, None)?.metadata()?.blocks() * 512);
            }
        }
        operation["recovery_allocated_bytes"] = json!(extra);
    }
    write_manifest(directory, manifest)?;
    if let Some(observer) = &manager.maintenance_observer {
        observer(manifest);
    }
    Ok(())
}
fn exists(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok()
}

pub fn validate_request(action: &str, target: Option<u64>, maximum: u64) -> Result<()> {
    match action {
        "reclaim" if target.is_none() => Ok(()),
        "grow" | "shrink"
            if target
                .map(|n| n >= 32 * 1024 * 1024 && n <= maximum && n % (1024 * 1024) == 0)
                .unwrap_or(false) =>
        {
            Ok(())
        }
        _ => bail!("invalid maintenance action or target capacity"),
    }
}

impl WorkspaceManager {
    // Isolated capability fixture only; user images always use maintain/recover.
    pub(super) fn probe_disposable_tools(&self, id: &str) -> Result<()> {
        if !Path::new(&self.config.image_root)
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with(".cape-workspace-probe-"))
            .unwrap_or(false)
        {
            bail!("maintenance capability probe requires an isolated probe root");
        }
        let directory = self.directory(id, false)?;
        let lease = lease(&directory)?;
        let mut manifest = self.read_manifest(&directory, id, "__cape_probe__")?;
        if manifest.state != "retained"
            || mount_at(&directory)?.is_some()
            || self.handles.contains_key(id)
        {
            bail!("capability probe requires an offline disposable image");
        }
        image_file(&directory, &manifest)?;
        let image = directory.join("workspace.img");
        let candidate = directory.join("probe-maintenance.img");
        let run = |name: &str, args: Vec<String>| {
            run_tool(name, &args, 30, true, Some(lease.as_raw_fd()))
        };
        run("e2fsck", vec!["-fn".into(), image.display().to_string()])?;
        run(
            "cp",
            vec![
                "--reflink=auto".into(),
                "--sparse=always".into(),
                "--preserve=mode".into(),
                "--no-clobber".into(),
                image.display().to_string(),
                candidate.display().to_string(),
            ],
        )?;
        let info = checked_file(&candidate, None, None)?.metadata()?;
        if Some(info.dev()) == manifest.image_device && Some(info.ino()) == manifest.image_inode {
            bail!("probe candidate must be independent");
        }
        let (_, hashes, _) = run(
            "sha256sum",
            vec![image.display().to_string(), candidate.display().to_string()],
        )?;
        let hashes: Vec<_> = hashes
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .collect();
        if hashes.len() != 2 || hashes[0] != hashes[1] {
            bail!("probe candidate checksum mismatch");
        }
        run(
            "e2fsck",
            vec!["-f".into(), "-p".into(), candidate.display().to_string()],
        )?;
        OpenOptions::new()
            .write(true)
            .custom_flags(nix::libc::O_NOFOLLOW)
            .open(&candidate)?
            .set_len(64 * 1024 * 1024)?;
        run(
            "resize2fs",
            vec![candidate.display().to_string(), "65536K".into()],
        )?;
        if geometry(&candidate, &lease, 30)?["size_bytes"] != 64 * 1024 * 1024u64 {
            bail!("probe growth geometry mismatch");
        }
        run(
            "e2fsck",
            vec![
                "-f".into(),
                "-p".into(),
                "-E".into(),
                "discard".into(),
                candidate.display().to_string(),
            ],
        )?;
        run(
            "resize2fs",
            vec![candidate.display().to_string(), "32768K".into()],
        )?;
        let after = geometry(&candidate, &lease, 30)?;
        if after["size_bytes"] != 32 * 1024 * 1024u64 {
            bail!("probe shrink geometry mismatch");
        }
        let file = OpenOptions::new()
            .write(true)
            .custom_flags(nix::libc::O_NOFOLLOW)
            .open(&candidate)?;
        file.set_len(32 * 1024 * 1024)?;
        file.sync_all()?;
        drop(file);
        run(
            "e2fsck",
            vec!["-fn".into(), candidate.display().to_string()],
        )?;
        checked_file(&candidate, Some(info.dev()), Some(info.ino()))?;
        fs::rename(&candidate, &image)?;
        sync_dir(&directory)?;
        manifest.image_device = Some(info.dev());
        manifest.image_inode = Some(info.ino());
        manifest.geometry = Some(after);
        write_manifest(&directory, &mut manifest)?;
        unlock_completed_lease(&lease)?;
        Ok(())
    }

    pub fn maintain(
        &self,
        id: &str,
        user: &str,
        action: &str,
        target: Option<u64>,
        operation_id: &str,
        policy: &MaintenancePolicy,
    ) -> Result<Value> {
        self.maintain_with_limit(
            id,
            user,
            action,
            target,
            operation_id,
            policy,
            self.config.max_image_size_bytes,
        )
    }

    pub fn maintain_with_limit(
        &self,
        id: &str,
        user: &str,
        action: &str,
        target: Option<u64>,
        operation_id: &str,
        policy: &MaintenancePolicy,
        maximum: u64,
    ) -> Result<Value> {
        if !(32 * 1024 * 1024..=i64::MAX as u64).contains(&maximum) {
            bail!("invalid deployment capacity limit");
        }
        validate_request(action, target, maximum)?;
        if operation_id.len() != 32
            || !operation_id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            bail!("invalid maintenance operation id");
        }
        let directory = self.directory(id, false)?;
        let lease = lease(&directory)?;
        let mut manifest = self.read_manifest(&directory, id, user)?;
        if let Some(previous) = &manifest.maintenance {
            if previous["id"] == operation_id {
                if previous["action"] != action || previous["target_bytes"] != json!(target) {
                    bail!("operation id reused with different parameters");
                }
                if previous["state"] == "completed" {
                    unlock_completed_lease(&lease)?;
                    return Ok(previous.clone());
                }
                bail!("interrupted maintenance requires explicit recovery");
            }
        }
        let started = Instant::now();
        let candidate = directory.join("maintenance-candidate.img");
        let backup = directory.join("maintenance-backup.img");
        let image = directory.join("workspace.img");
        let timeout = policy.timeout_seconds;
        let outcome = (|| -> Result<Value> {
            if manifest.state != "retained"
                || self.handles.contains_key(id)
                || mount_at(&directory)?.is_some()
            {
                bail!("maintenance requires a clean unmounted exclusively owned workspace");
            }
            if exists(&candidate) || exists(&backup) {
                bail!("maintenance recovery files exist; reconcile before another operation");
            }
            let original = image_file(&directory, &manifest)?.metadata()?;
            let current = original.len();
            if (action == "grow" && target.unwrap() <= current)
                || (action == "shrink" && target.unwrap() >= current)
            {
                bail!("target must change capacity in the requested direction");
            }
            run_tool(
                "e2fsck",
                &["-fn".into(), image.display().to_string()],
                timeout,
                true,
                Some(lease.as_raw_fd()),
            )?;
            let before = geometry(&image, &lease, timeout)?;
            if before["size_bytes"] != current {
                bail!("image envelope differs from filesystem geometry");
            }
            let needed = original
                .blocks()
                .saturating_mul(512)
                .saturating_add(target.unwrap_or(current).saturating_sub(current))
                .saturating_add(policy.backing_reserve_bytes);
            let (_, storage, _) = run_tool(
                "stat",
                &[
                    "-f".into(),
                    "-c".into(),
                    "%S %a".into(),
                    directory.display().to_string(),
                ],
                5,
                true,
                Some(lease.as_raw_fd()),
            )?;
            let fields: Vec<u64> = storage
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()?;
            if fields.len() != 2 || fields[0].saturating_mul(fields[1]) < needed {
                bail!("insufficient backing space for recoverable maintenance");
            }
            let mut operation = json!({"id":operation_id,"action":action,"target_bytes":target,"state":"running","phase":"copying",
                "before":before,"started_at":now(),"original_device":original.dev(),"original_inode":original.ino(),
                "original_schema":manifest.schema_version,"before_allocated_bytes":original.blocks()*512});
            manifest.schema_version = 3;
            manifest.state = "maintenance".into();
            manifest.last_error = None;
            manifest.maintenance = Some(operation.clone());
            publish(self, &directory, &mut manifest)?;
            run_tool(
                "cp",
                &[
                    "--reflink=auto".into(),
                    "--sparse=always".into(),
                    "--preserve=mode".into(),
                    "--no-clobber".into(),
                    image.display().to_string(),
                    candidate.display().to_string(),
                ],
                timeout,
                true,
                Some(lease.as_raw_fd()),
            )?;
            let file = checked_file(&candidate, None, None)?;
            let copied = file.metadata()?;
            if copied.dev() == original.dev() && copied.ino() == original.ino() {
                bail!("candidate is not independent");
            }
            file.sync_all()?;
            drop(file);
            let (_, hashes, _) = run_tool(
                "sha256sum",
                &[image.display().to_string(), candidate.display().to_string()],
                timeout,
                true,
                Some(lease.as_raw_fd()),
            )?;
            let hashes: Vec<&str> = hashes
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .collect();
            if hashes.len() != 2 || hashes[0] != hashes[1] {
                bail!("maintenance copy verification failed");
            }
            operation["phase"] = json!("applying");
            operation["candidate_inode"] = json!(copied.ino());
            operation["candidate_device"] = json!(copied.dev());
            operation["original_sha256"] = json!(hashes[0]);
            manifest.maintenance = Some(operation.clone());
            publish(self, &directory, &mut manifest)?;
            run_tool(
                "e2fsck",
                &["-f".into(), "-p".into(), candidate.display().to_string()],
                timeout,
                true,
                Some(lease.as_raw_fd()),
            )?;
            match action {
                "grow" => {
                    let file = OpenOptions::new()
                        .write(true)
                        .custom_flags(nix::libc::O_NOFOLLOW)
                        .open(&candidate)?;
                    file.set_len(target.unwrap())?;
                    file.sync_all()?;
                    run_tool(
                        "resize2fs",
                        &[
                            candidate.display().to_string(),
                            format!("{}K", target.unwrap() / 1024),
                        ],
                        timeout,
                        true,
                        Some(lease.as_raw_fd()),
                    )?;
                }
                "shrink" => {
                    run_tool(
                        "resize2fs",
                        &[
                            candidate.display().to_string(),
                            format!("{}K", target.unwrap() / 1024),
                        ],
                        timeout,
                        true,
                        Some(lease.as_raw_fd()),
                    )?;
                }
                _ => {
                    run_tool(
                        "e2fsck",
                        &[
                            "-f".into(),
                            "-p".into(),
                            "-E".into(),
                            "discard".into(),
                            candidate.display().to_string(),
                        ],
                        timeout,
                        true,
                        Some(lease.as_raw_fd()),
                    )?;
                }
            }
            let after = geometry(&candidate, &lease, timeout)?;
            if after["size_bytes"] != target.unwrap_or(current) {
                bail!("resizer did not establish requested geometry");
            }
            if action == "shrink" {
                let file = OpenOptions::new()
                    .write(true)
                    .custom_flags(nix::libc::O_NOFOLLOW)
                    .open(&candidate)?;
                file.set_len(target.unwrap())?;
                file.sync_all()?;
            }
            run_tool(
                "e2fsck",
                &["-fn".into(), candidate.display().to_string()],
                timeout,
                true,
                Some(lease.as_raw_fd()),
            )?;
            let file = checked_file(&candidate, Some(copied.dev()), Some(copied.ino()))?;
            file.sync_all()?;
            let result = file.metadata()?;
            drop(file);
            operation["phase"] = json!("committing");
            operation["after"] = after.clone();
            operation["after_allocated_bytes"] = json!(result.blocks() * 512);
            manifest.maintenance = Some(operation.clone());
            publish(self, &directory, &mut manifest)?;
            image_file(&directory, &manifest)?;
            fs::rename(&image, &backup)?;
            sync_dir(&directory)?;
            operation["phase"] = json!("backup_saved");
            manifest.maintenance = Some(operation.clone());
            publish(self, &directory, &mut manifest)?;
            fs::rename(&candidate, &image)?;
            sync_dir(&directory)?;
            operation["phase"] = json!("swapped");
            manifest.maintenance = Some(operation.clone());
            publish(self, &directory, &mut manifest)?;
            manifest.image_device = Some(result.dev());
            manifest.image_inode = Some(result.ino());
            manifest.current_image_size_bytes = Some(target.unwrap_or(current));
            manifest.geometry = Some(after);
            manifest.generation = Some(
                manifest
                    .generation
                    .unwrap_or(0)
                    .checked_add(1)
                    .context("generation overflow")?,
            );
            manifest.state = "maintenance".into();
            manifest.last_error = None;
            operation["phase"] = json!("committed");
            let reference = crate::workspace_reclaim::sample(result.blocks() * 512,
                manifest.geometry.as_ref().unwrap(), manifest.generation.unwrap(), manifest.usage_generation)?;
            manifest.reclaim_reference = Some(reference.clone());
            manifest.reclaim_sample = Some(reference);
            operation["state"] = json!("completed");
            operation["completed_at"] = json!(now());
            operation["seconds"] = json!(started.elapsed().as_secs_f64());
            operation["reclaimed_bytes"] =
                json!((original.blocks() * 512).saturating_sub(result.blocks() * 512));
            manifest.maintenance = Some(operation.clone());
            publish(self, &directory, &mut manifest)?;
            unlock_completed_lease(&lease)?;
            Ok(manifest
                .maintenance
                .clone()
                .expect("published maintenance record"))
        })();
        if let Err(error) = &outcome {
            if manifest.state == "maintenance"
                && manifest
                    .maintenance
                    .as_ref()
                    .map(|op| op["id"] == operation_id && op["phase"] != "committed")
                    .unwrap_or(false)
            {
                manifest.state = "needs_recovery".into();
                manifest.last_error = Some(error.to_string());
                let op = manifest.maintenance.as_mut().unwrap();
                op["state"] = json!("failed");
                op["error"] = json!(error.to_string());
                publish(self, &directory, &mut manifest)?;
            }
        }
        outcome
    }

    pub fn recover_maintenance(&self, id: &str, user: &str) -> Result<Value> {
        let directory = self.directory(id, false)?;
        let lease = lease(&directory)?;
        let mut manifest = self.read_manifest(&directory, id, user)?;
        let mut operation = manifest
            .maintenance
            .clone()
            .context("recovery requires an offline maintenance record")?;
        if mount_at(&directory)?.is_some() || self.handles.contains_key(id) {
            bail!("recovery requires an offline workspace");
        }
        let image = directory.join("workspace.img");
        let backup = directory.join("maintenance-backup.img");
        let candidate = directory.join("maintenance-candidate.img");
        let committed = operation["phase"] == "committed" && operation["state"] == "completed";
        let source = if committed || !exists(&backup) {
            &image
        } else {
            &backup
        };
        let device = if committed {
            manifest.image_device
        } else {
            operation["original_device"].as_u64()
        };
        let inode = if committed {
            manifest.image_inode
        } else {
            operation["original_inode"].as_u64()
        };
        let info = checked_file(source, device, inode)?.metadata()?;
        run_tool(
            "e2fsck",
            &["-fn".into(), source.display().to_string()],
            300,
            true,
            Some(lease.as_raw_fd()),
        )?;
        let expected = if committed {
            manifest
                .current_image_size_bytes
                .unwrap_or(manifest.config.image_size_bytes)
        } else {
            operation["before"]["size_bytes"]
                .as_u64()
                .context("missing recovery geometry")?
        };
        if info.len() != expected || geometry(source, &lease, 300)?["size_bytes"] != expected {
            bail!("recovery source geometry mismatch");
        }
        if !committed {
            if let Some(hash) = operation["original_sha256"].as_str() {
                let (_, text, _) = run_tool(
                    "sha256sum",
                    &[source.display().to_string()],
                    300,
                    true,
                    Some(lease.as_raw_fd()),
                )?;
                if text.split_whitespace().next() != Some(hash) {
                    bail!("recovery source checksum mismatch");
                }
            }
        }
        if source == &backup {
            fs::rename(&backup, &image)?;
            sync_dir(&directory)?;
        }
        if !committed {
            manifest.image_device = Some(info.dev());
            manifest.image_inode = Some(info.ino());
            manifest.state = "maintenance".into();
            manifest.current_image_size_bytes = Some(expected);
            manifest.geometry = Some(operation["before"].clone());
            manifest.last_error = None;
            operation["state"] = json!("failed");
            operation["phase"] = json!("rolled_back");
            operation["recovered"] = json!(true);
            manifest.maintenance = Some(operation.clone());
            publish(self, &directory, &mut manifest)?;
        }
        for path in [&backup, &candidate] {
            if exists(path) {
                let (device, inode) = if path == &backup {
                    (
                        operation["original_device"].as_u64(),
                        operation["original_inode"].as_u64(),
                    )
                } else {
                    (
                        operation["candidate_device"].as_u64(),
                        operation["candidate_inode"].as_u64(),
                    )
                };
                checked_file(path, device, inode)?;
                fs::remove_file(path)?;
            }
        }
        sync_dir(&directory)?;
        operation["recovery_files_removed"] = json!(true);
        manifest.state = "retained".into();
        manifest.last_error = None;
        manifest.maintenance = Some(operation.clone());
        publish(self, &directory, &mut manifest)?;
        unlock_completed_lease(&lease)?;
        Ok(manifest
            .maintenance
            .clone()
            .expect("published maintenance record"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "subprocess fixture for offline_maintenance_crash_publication"]
    fn maintenance_transaction_child() {
        let config = WorkspaceConfig::from_value(
            serde_json::from_str(&std::env::var("CAPE_TEST_MAINTENANCE_CONFIG").unwrap()).unwrap(),
        )
        .unwrap();
        let mut manager = WorkspaceManager::new(config, "node".into(), "pool".into()).unwrap();
        let phase = std::env::var("CAPE_TEST_MAINTENANCE_PHASE").unwrap();
        manager.maintenance_observer = Some(std::sync::Arc::new(move |manifest| {
            if manifest
                .maintenance
                .as_ref()
                .map(|op| op["phase"] == phase)
                .unwrap_or(false)
            {
                std::process::exit(77);
            }
        }));
        let action = std::env::var("CAPE_TEST_MAINTENANCE_ACTION").unwrap();
        let target = match action.as_str() {
            "grow" => Some(128 * 1024 * 1024),
            "shrink" => Some(32 * 1024 * 1024),
            _ => None,
        };
        let policy = MaintenancePolicy {
            backing_reserve_bytes: 0,
            ..MaintenancePolicy::default()
        };
        manager
            .maintain(
                &std::env::var("CAPE_TEST_MAINTENANCE_ID").unwrap(),
                "user",
                &action,
                target,
                &Uuid::new_v4().simple().to_string(),
                &policy,
            )
            .unwrap();
        panic!("requested crash phase was not reached");
    }

    #[test]
    #[ignore = "requires real offline e2fsprogs tools; mandatory integration gate"]
    fn offline_maintenance_crash_publication() {
        for action in ["grow", "reclaim", "shrink"] {
            for phase in [
                "copying",
                "applying",
                "committing",
                "backup_saved",
                "swapped",
                "committed",
            ] {
                let root = std::env::temp_dir().join(format!(
                    "cape-maintenance-native-{}",
                    Uuid::new_v4().simple()
                ));
                let config = WorkspaceConfig {
                    image_root: root.display().to_string(),
                    image_size_bytes: 64 * 1024 * 1024,
                    ..WorkspaceConfig::default()
                };
                let manager =
                    WorkspaceManager::new(config.clone(), "node".into(), "pool".into()).unwrap();
                let id = new_workspace_id();
                let directory = manager.directory(&id, true).unwrap();
                let file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .open(directory.join("workspace.img"))
                    .unwrap();
                file.set_len(64 * 1024 * 1024).unwrap();
                file.sync_all().unwrap();
                let info = file.metadata().unwrap();
                drop(file);
                run_tool(
                    "mkfs.ext4",
                    &[
                        "-q".into(),
                        "-F".into(),
                        directory.join("workspace.img").display().to_string(),
                    ],
                    30,
                    true,
                    None,
                )
                .unwrap();
                let payload = directory.join("marker-data");
                fs::write(&payload, b"native-maintenance-crash-content\n").unwrap();
                let seed = Command::new("timeout")
                    .args(["10s", "debugfs", "-w", "-R"])
                    .arg(format!("write {} marker", payload.display()))
                    .arg(directory.join("workspace.img"))
                    .output()
                    .unwrap();
                assert!(
                    seed.status.success(),
                    "{}",
                    String::from_utf8_lossy(&seed.stderr)
                );
                let mut manifest = WorkspaceManifest {
                    schema_version: 3,
                    revision: 0,
                    workspace_id: id.clone(),
                    user_id: "user".into(),
                    pool: "pool".into(),
                    node_id: "node".into(),
                    execution_session_id: "session".into(),
                    config: config.clone(),
                    state: "retained".into(),
                    last_error: None,
                    image_device: Some(info.dev()),
                    image_inode: Some(info.ino()),
                    mount_id: None,
                    daemon_pid: None,
                    timings: BTreeMap::new(),
                    current_image_size_bytes: None,
                    geometry: None,
                    generation: None,
                    maintenance: None,
                    usage_generation: 0,
                    reclaim_reference: None,
                    reclaim_sample: None,
                    reclaim_sample_error: None,
                };
                write_manifest(&directory, &mut manifest).unwrap();
                let result = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--ignored",
                        "--exact",
                        "workspace_fs::maintenance::tests::maintenance_transaction_child",
                        "--nocapture",
                    ])
                    .env(
                        "CAPE_TEST_MAINTENANCE_CONFIG",
                        config.to_value().to_string(),
                    )
                    .env("CAPE_TEST_MAINTENANCE_ID", &id)
                    .env("CAPE_TEST_MAINTENANCE_PHASE", phase)
                    .env("CAPE_TEST_MAINTENANCE_ACTION", action)
                    .output()
                    .unwrap();
                assert_eq!(
                    result.status.code(),
                    Some(77),
                    "{action}/{phase}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                manager.recover_maintenance(&id, "user").unwrap();
                let restored = manager.read_manifest(&directory, &id, "user").unwrap();
                let expected = if phase == "committed" {
                    match action {
                        "grow" => 128 * 1024 * 1024,
                        "shrink" => 32 * 1024 * 1024,
                        _ => 64 * 1024 * 1024,
                    }
                } else {
                    64 * 1024 * 1024
                };
                assert_eq!(
                    image_file(&directory, &restored)
                        .unwrap()
                        .metadata()
                        .unwrap()
                        .len(),
                    expected
                );
                assert_eq!(restored.state, "retained");
                let content = Command::new("timeout")
                    .args(["10s", "debugfs", "-R", "cat marker"])
                    .arg(directory.join("workspace.img"))
                    .output()
                    .unwrap();
                assert!(content.status.success());
                assert_eq!(
                    content.stdout, b"native-maintenance-crash-content\n",
                    "{action}/{phase}"
                );
                assert!(
                    !exists(&directory.join("maintenance-candidate.img"))
                        && !exists(&directory.join("maintenance-backup.img"))
                );
                manager.recover_maintenance(&id, "user").unwrap();
                fs::remove_dir_all(&root).unwrap();
                println!("maintenance_crash action={action} phase={phase} recovery=passed content=passed");
            }
        }
    }

    #[test]
    fn policy_and_request_validation() {
        let manager =
            WorkspaceManager::new(WorkspaceConfig::default(), "node".into(), "pool".into())
                .unwrap();
        assert!(manager
            .probe_disposable_tools(&new_workspace_id())
            .unwrap_err()
            .to_string()
            .contains("isolated probe root"));
        assert!(!MaintenancePolicy::default().auto_grow);
        for value in [
            json!({"auto_grow":1}),
            json!({"unknown":1}),
            json!({"grow_target_percent":90}),
            json!({"timeout_seconds":0}),
        ] {
            assert!(MaintenancePolicy::from_value(value).is_err());
        }
        assert!(validate_request("grow", Some(64 * 1024 * 1024), 1024 * 1024 * 1024).is_ok());
        assert!(validate_request("reclaim", Some(64 * 1024 * 1024), 1024 * 1024 * 1024).is_err());
    }
}
