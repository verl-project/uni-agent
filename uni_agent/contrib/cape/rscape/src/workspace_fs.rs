// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::workspace::WorkspaceConfig;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use uuid::Uuid;

mod maintenance;
pub use maintenance::MaintenancePolicy;

thread_local! { static MAINTENANCE_NICE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }

/// Scope helper priority to this blocking worker, never the agent or user runtime.
pub struct MaintenancePriority(u64);
impl MaintenancePriority {
    pub fn enter(increment: u64) -> Result<Self> {
        anyhow::ensure!(increment <= 19, "invalid maintenance nice increment");
        Ok(Self(MAINTENANCE_NICE.with(|value| value.replace(increment))))
    }
}
impl Drop for MaintenancePriority {
    fn drop(&mut self) { MAINTENANCE_NICE.with(|value| value.set(self.0)); }
}

#[cfg(test)]
#[test]
fn maintenance_priority_is_scoped_to_helper_and_restored() {
    let before: i32 = run_tool("nice", &[], 5, true, None).unwrap().1.trim().parse().unwrap();
    {
        let _scope = MaintenancePriority::enter(10).unwrap();
        let child: i32 = run_tool("nice", &[], 5, true, None).unwrap().1.trim().parse().unwrap();
        assert_eq!(child, (before + 10).min(19));
    }
    let after: i32 = run_tool("nice", &[], 5, true, None).unwrap().1.trim().parse().unwrap();
    assert_eq!(before, after);
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkspaceManifest {
    pub schema_version: u32,
    #[serde(default)]
    pub revision: u64,
    pub workspace_id: String,
    pub user_id: String,
    pub pool: String,
    pub node_id: String,
    pub execution_session_id: String,
    pub config: WorkspaceConfig,
    pub state: String,
    pub last_error: Option<String>,
    pub image_device: Option<u64>,
    pub image_inode: Option<u64>,
    pub mount_id: Option<u64>,
    pub daemon_pid: Option<u32>,
    #[serde(default)]
    pub timings: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_image_size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maintenance: Option<Value>,
    #[serde(default)]
    pub usage_generation: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reclaim_reference: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reclaim_sample: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reclaim_sample_error: Option<String>,
}

#[derive(Clone, Debug)]
struct MountEntry {
    id: u64,
    source: String,
    filesystem: String,
}

struct WorkspaceHandle {
    directory: PathBuf,
    manifest: WorkspaceManifest,
    _lease: File,
    process: Child,
}

/// Use outside the controller loop. Runtime users must stop before release/check/delete.
pub struct WorkspaceManager {
    config: WorkspaceConfig,
    node_id: String,
    pool: String,
    handles: BTreeMap<String, WorkspaceHandle>,
    space_cache: std::sync::Mutex<BTreeMap<String, (Instant, u64, Value)>>,
    pub maintenance_observer: Option<std::sync::Arc<dyn Fn(&WorkspaceManifest) + Send + Sync>>,
}

fn uid() -> u32 {
    unsafe { nix::libc::geteuid() }
}
fn gid() -> u32 {
    unsafe { nix::libc::getegid() }
}

pub fn new_workspace_id() -> String {
    format!("ws-{}", Uuid::new_v4().simple())
}

pub fn probe_configuration(config: &WorkspaceConfig, node_id: &str, pool: &str) -> Result<Value> {
    DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&config.image_root)?;
    let root = Path::new(&config.image_root)
        .join(format!(".cape-workspace-probe-{}", Uuid::new_v4().simple()));
    DirBuilder::new().mode(0o700).create(&root)?;
    let mut probe = config.clone();
    probe.image_root = root
        .to_str()
        .context("workspace probe path is not UTF-8")?
        .into();
    probe.image_size_bytes = 32 * 1024 * 1024;
    probe.max_image_size_bytes = probe.max_image_size_bytes.max(64 * 1024 * 1024);
    let mut manager = WorkspaceManager::new(probe, node_id.into(), pool.into())?;
    let id = new_workspace_id();
    let outcome = (|| -> Result<Value> {
        let mount = manager.prepare(&id, "__cape_probe__", "__cape_probe__", true)?;
        let mut marker = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(mount.join("probe"))?;
        marker.write_all(b"cape-workspace-ready\n")?;
        marker.sync_all()?;
        drop(marker);
        manager.release(&id)?;
        manager.probe_disposable_tools(&id)?;
        let mount = manager.prepare(&id, "__cape_probe__", "__cape_probe__", false)?;
        if fs::read(mount.join("probe"))? != b"cape-workspace-ready\n" {
            bail!("workspace maintenance probe content mismatch");
        }
        manager.release(&id)?;
        manager.delete(&id, "__cape_probe__")?;
        let storage = manager.inspect(&id, "__cape_probe__")?["space"].clone();
        if !storage["error"].is_null() {
            bail!(
                "cannot inspect workspace backing storage: {}",
                storage["error"]
            );
        }
        fs::remove_dir_all(&root)?;
        Ok(
            json!({"state":"applied","applied_version":config.version,"error":null,"config":config.to_value(),"protocol":1,"maintenance_protocol":1,"storage":storage["backing"],"maintenance_host_key":crate::workspace_reclaim::maintenance_host_key()}),
        )
    })();
    if outcome.is_err() {
        let _ = manager.release(&id);
    }
    outcome.with_context(|| {
        format!(
            "workspace probe failed; diagnostic image retained at {}",
            root.display()
        )
    })
}

fn validate_id(id: &str) -> Result<()> {
    if id.len() != 35
        || !id.starts_with("ws-")
        || !id[3..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("invalid workspace_id");
    }
    Ok(())
}

fn private_directory(path: &Path, create: bool) -> Result<()> {
    if create {
        match DirBuilder::new().mode(0o700).create(path) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error.into()),
        }
    }
    let info = fs::symlink_metadata(path)?;
    if !info.file_type().is_dir() || info.uid() != uid() || info.mode() & 0o077 != 0 {
        bail!(
            "workspace directory must be private, owned, and not a symlink: {}",
            path.display()
        );
    }
    Ok(())
}

fn lease(directory: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(directory.join("writer.lock"))?;
    let info = file.metadata()?;
    if !info.is_file() || info.uid() != uid() || info.nlink() != 1 || info.mode() & 0o077 != 0 {
        bail!("unsafe workspace writer lock");
    }
    if unsafe { nix::libc::flock(file.as_raw_fd(), nix::libc::LOCK_EX | nix::libc::LOCK_NB) } != 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::WouldBlock {
            bail!("workspace already has a writer; takeover is forbidden");
        }
        return Err(error.into());
    }
    Ok(file)
}

fn unlock_completed_lease(file: &File) -> Result<()> {
    // Only after intended helpers have exited. An abnormal drop must leave the
    // inherited fence intact; a completed operation must not depend on unrelated
    // fork-before-exec copies of its close-on-exec descriptor disappearing.
    if unsafe { nix::libc::flock(file.as_raw_fd(), nix::libc::LOCK_UN) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

fn write_manifest(directory: &Path, manifest: &mut WorkspaceManifest) -> Result<()> {
    manifest.revision = manifest
        .revision
        .checked_add(1)
        .filter(|value| *value <= i64::MAX as u64)
        .context("workspace manifest revision overflow")?;
    let temporary = directory.join(format!(".manifest-{}", Uuid::new_v4().simple()));
    let outcome = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(&serde_json::to_vec(manifest)?)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temporary, directory.join("manifest.json"))?;
        File::open(directory)?.sync_all()?;
        Ok(())
    })();
    if temporary.exists() {
        let _ = fs::remove_file(&temporary);
    }
    outcome
}

fn image_file(directory: &Path, manifest: &WorkspaceManifest) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW)
        .open(directory.join("workspace.img"))?;
    let info = file.metadata()?;
    if !info.is_file()
        || info.uid() != uid()
        || info.nlink() != 1
        || info.mode() & 0o077 != 0
        || Some(info.dev()) != manifest.image_device
        || Some(info.ino()) != manifest.image_inode
        || info.len()
            != manifest
                .current_image_size_bytes
                .unwrap_or(manifest.config.image_size_bytes)
    {
        bail!("workspace backing image identity/size/ownership changed");
    }
    Ok(file)
}

fn decode_mount(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut result = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\'
            && index + 3 < bytes.len()
            && bytes[index + 1..index + 4]
                .iter()
                .all(|value| (b'0'..=b'7').contains(value))
        {
            let code = (bytes[index + 1] - b'0') as u16 * 64
                + (bytes[index + 2] - b'0') as u16 * 8
                + (bytes[index + 3] - b'0') as u16;
            result.push(code as u8);
            index += 4;
        } else {
            result.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&result).into_owned()
}

fn mount_at(directory: &Path) -> Result<Option<MountEntry>> {
    for _ in 0..5 {
        let mut file = File::open("/proc/self/mountinfo")?;
        let mut poll = nix::libc::pollfd {
            fd: file.as_raw_fd(),
            events: nix::libc::POLLPRI,
            revents: 0,
        };
        if unsafe { nix::libc::poll(&mut poll, 1, 0) } < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let changed = unsafe { nix::libc::poll(&mut poll, 1, 0) };
        if changed < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        if changed == 0 {
            return parse_mount_at(directory, &contents);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    bail!("mount table kept changing during workspace inspection")
}

fn parse_mount_at(directory: &Path, contents: &str) -> Result<Option<MountEntry>> {
    let target = directory.join("mnt");
    let mut result: Option<MountEntry> = None;
    for line in contents.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let fields: Vec<&str> = left.split_whitespace().collect();
        let filesystem: Vec<&str> = right.split_whitespace().collect();
        if fields.len() < 6 || filesystem.len() < 2 || Path::new(&decode_mount(fields[4])) != target
        {
            continue;
        }
        let entry = MountEntry {
            id: fields[0].parse()?,
            source: decode_mount(filesystem[1]),
            filesystem: filesystem[0].into(),
        };
        if let Some(previous) = &result {
            if previous.id == entry.id
                && previous.source == entry.source
                && previous.filesystem == entry.filesystem
            {
                continue;
            }
            bail!("stacked or inconsistent workspace mounts are not supported: {previous:?}, {entry:?}");
        }
        result = Some(entry);
    }
    Ok(result)
}

fn owned_mount(directory: &Path, entry: &MountEntry) -> bool {
    entry.filesystem.starts_with("fuse")
        && Path::new(&entry.source) == directory.join("workspace.img")
}

fn tool(name: &str) -> Result<PathBuf> {
    let suffix = match name {
        "fuse2fs" => "FUSE2FS",
        "mkfs.ext4" => "MKFS",
        "e2fsck" => "E2FSCK",
        "fusermount" => "FUSERMOUNT",
        "timeout" => "TIMEOUT",
        "stat" => "STAT",
        "resize2fs" => "RESIZE2FS",
        "dumpe2fs" => "DUMPE2FS",
        "cp" => "CP",
        "sha256sum" => "SHA256SUM",
        "nice" => "NICE",
        _ => bail!("unknown workspace tool"),
    };
    let configured = std::env::var(format!("CAPE_WORKSPACE_{suffix}_BIN"))
        .ok()
        .filter(|value| !value.is_empty());
    let choices = if let Some(value) = configured {
        vec![value]
    } else if name == "fusermount" {
        vec!["fusermount".into(), "fusermount3".into()]
    } else {
        vec![name.into()]
    };
    for choice in choices {
        let candidates: Vec<PathBuf> = if choice.contains('/') {
            vec![PathBuf::from(&choice)]
        } else {
            std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
                .map(|path| path.join(&choice))
                .collect()
        };
        for candidate in candidates {
            if candidate
                .metadata()
                .map(|info| info.is_file() && info.mode() & 0o111 != 0)
                .unwrap_or(false)
            {
                return Ok(candidate);
            }
        }
    }
    bail!("workspace tool is unavailable: {name}")
}

fn temporary_output() -> Result<File> {
    let path =
        std::env::temp_dir().join(format!("cape-workspace-tool-{}", Uuid::new_v4().simple()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)?;
    fs::remove_file(path)?;
    Ok(file)
}

fn wait_child(child: &mut Child, seconds: u64) -> Result<ExitStatus> {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            bail!(
                "workspace process {} did not terminate within {seconds}s",
                child.id()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn read_tail(file: &mut File) -> Result<String> {
    let length = file.metadata()?.len();
    file.seek(SeekFrom::Start(length.saturating_sub(65536)))?;
    let mut bytes = Vec::new();
    file.take(65536).read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn run_tool(
    name: &str,
    args: &[String],
    timeout: u64,
    check: bool,
    lease_fd: Option<i32>,
) -> Result<(i32, String, String)> {
    let mut stdout = temporary_output()?;
    let mut stderr = temporary_output()?;
    let priority = MAINTENANCE_NICE.with(|value| value.get());
    let mut command = if priority > 0 {
        let mut command = Command::new(tool("nice")?);
        command.args(["-n", &priority.to_string()]).arg(tool("timeout")?);
        command
    } else { Command::new(tool("timeout")?) };
    command
        .args(["--signal=TERM", "--kill-after=5s", &format!("{timeout}s")])
        .arg(tool(name)?)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout.try_clone()?))
        .stderr(Stdio::from(stderr.try_clone()?));
    unsafe {
        command.pre_exec(move || {
            if nix::libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if let Some(fd) = lease_fd {
                let flags = nix::libc::fcntl(fd, nix::libc::F_GETFD);
                if flags == -1
                    || nix::libc::fcntl(fd, nix::libc::F_SETFD, flags & !nix::libc::FD_CLOEXEC)
                        == -1
                {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut child = command.spawn()?;
    let status = match wait_child(&mut child, timeout + 10) {
        Ok(status) => status,
        Err(error) => {
            let _ = child.kill();
            let _ = wait_child(&mut child, 5);
            return Err(error);
        }
    };
    let result = (
        status
            .code()
            .unwrap_or_else(|| -status.signal().unwrap_or(127)),
        read_tail(&mut stdout)?,
        read_tail(&mut stderr)?,
    );
    if check && result.0 != 0 {
        bail!("{name} failed ({}): {}", result.0, result.2);
    }
    Ok(result)
}

impl WorkspaceManager {
    pub fn mounted_ids(&self) -> Vec<String> {
        self.handles.keys().cloned().collect()
    }

    pub fn new(config: WorkspaceConfig, node_id: String, pool: String) -> Result<Self> {
        Ok(Self {
            config: WorkspaceConfig::from_value(config.to_value())?,
            node_id,
            pool,
            handles: BTreeMap::new(),
            space_cache: std::sync::Mutex::new(BTreeMap::new()),
            maintenance_observer: None,
        })
    }

    fn directory(&self, id: &str, create: bool) -> Result<PathBuf> {
        validate_id(id)?;
        let root = Path::new(&self.config.image_root);
        if create {
            DirBuilder::new().recursive(true).mode(0o700).create(root)?;
        }
        let root = fs::canonicalize(root)?;
        WorkspaceConfig::from_value(
            json!({"image_root":root.to_str().context("workspace root is not UTF-8")?}),
        )?;
        let parent = root.join(format!("cape-workspaces-{}", uid()));
        private_directory(&parent, create)?;
        let directory = parent.join(id);
        private_directory(&directory, create)?;
        Ok(directory)
    }

    fn read_manifest(&self, directory: &Path, id: &str, user: &str) -> Result<WorkspaceManifest> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(nix::libc::O_NOFOLLOW)
            .open(directory.join("manifest.json"))?;
        let info = file.metadata()?;
        if !info.is_file() || info.uid() != uid() || info.nlink() != 1 || info.len() > 1024 * 1024 {
            bail!("unsafe workspace manifest");
        }
        let mut bytes = Vec::new();
        (&mut file).take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
        let manifest: WorkspaceManifest = serde_json::from_slice(&bytes)?;
        if manifest.schema_version != 3 || manifest.workspace_id != id {
            bail!("workspace manifest identity/schema mismatch");
        }
        if manifest.revision > i64::MAX as u64 {
            bail!("invalid workspace manifest revision");
        }
        if manifest.usage_generation > i64::MAX as u64 {
            bail!("invalid workspace usage generation");
        }
        if manifest.user_id != user || manifest.pool != self.pool {
            bail!("workspace belongs to another user or pool");
        }
        if manifest.node_id != self.node_id {
            bail!("workspace belongs to another node; migration is not supported");
        }
        if !matches!(
            manifest.state.as_str(),
            "creating"
                | "retained"
                | "mounting"
                | "mounted"
                | "needs_check"
                | "deleting"
                | "deleted"
                | "maintenance"
                | "needs_recovery"
        ) {
            bail!("invalid workspace lifecycle state");
        }
        WorkspaceConfig::from_value(manifest.config.to_value())?;
        if manifest.config != self.config {
            bail!("workspace creation configuration differs from its controller record");
        }
        Ok(manifest)
    }

    pub fn prepare(
        &mut self,
        id: &str,
        user: &str,
        session: &str,
        create: bool,
    ) -> Result<PathBuf> {
        validate_id(id)?;
        if let Some(handle) = self.handles.get(id) {
            if handle.manifest.user_id != user || handle.manifest.execution_session_id != session {
                bail!("workspace is in use by another session");
            }
            self.verify(id)?;
            return Ok(self.handles[id].directory.join("mnt"));
        }
        let directory = self.directory(id, create)?;
        let mut lease = Some(lease(&directory)?);
        let mut process: Option<Child> = None;
        let mut manifest: Option<WorkspaceManifest> = None;
        let mut changed = false;
        let outcome = (|| -> Result<WorkspaceHandle> {
            if directory.join("manifest.json").exists() {
                let existing = self.read_manifest(&directory, id, user)?;
                if matches!(existing.state.as_str(), "maintenance" | "needs_recovery") {
                    bail!("workspace maintenance requires explicit recovery");
                }
                if fs::symlink_metadata(directory.join("maintenance-backup.img")).is_ok()
                    || fs::symlink_metadata(directory.join("maintenance-candidate.img")).is_ok()
                {
                    bail!("maintenance recovery files require cleanup before reuse");
                }
                if existing.state == "deleted" {
                    bail!("workspace was deleted");
                }
                if existing.state == "deleting" {
                    bail!("workspace deletion is incomplete; retry deletion before reuse");
                }
                if create && existing.execution_session_id != session {
                    bail!("existing workspace requires explicit reuse");
                }
                manifest = Some(existing);
            } else if create {
                let create_started = Instant::now();
                let image = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(0o600)
                    .custom_flags(nix::libc::O_NOFOLLOW)
                    .open(directory.join("workspace.img"))?;
                changed = true;
                image.set_len(self.config.image_size_bytes)?;
                image.sync_all()?;
                let info = image.metadata()?;
                manifest = Some(WorkspaceManifest {
                    schema_version: 3,
                    revision: 0,
                    workspace_id: id.into(),
                    user_id: user.into(),
                    pool: self.pool.clone(),
                    node_id: self.node_id.clone(),
                    execution_session_id: session.into(),
                    config: self.config.clone(),
                    state: "creating".into(),
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
                });
                write_manifest(&directory, manifest.as_mut().unwrap())?;
                manifest.as_mut().unwrap().timings.insert(
                    "create_seconds".into(),
                    create_started.elapsed().as_secs_f64(),
                );
                let format_started = Instant::now();
                run_tool(
                    "mkfs.ext4",
                    &[
                        "-F".into(),
                        "-m".into(),
                        "0".into(),
                        "-i".into(),
                        self.config.bytes_per_inode.to_string(),
                        "-E".into(),
                        format!("root_owner={}:{}", uid(), gid()),
                        directory.join("workspace.img").display().to_string(),
                    ],
                    self.config.mount_timeout_seconds,
                    true,
                    Some(lease.as_ref().unwrap().as_raw_fd()),
                )?;
                manifest.as_mut().unwrap().timings.insert(
                    "format_seconds".into(),
                    format_started.elapsed().as_secs_f64(),
                );
                manifest.as_mut().unwrap().geometry = Some(maintenance::geometry(
                    &directory.join("workspace.img"),
                    lease.as_ref().unwrap(),
                    self.config.mount_timeout_seconds,
                )?);
                manifest.as_mut().unwrap().state = "retained".into();
                let created = manifest.as_mut().unwrap();
                created.reclaim_reference = Some(crate::workspace_reclaim::sample(
                    image_file(&directory, created)?.metadata()?.blocks() * 512,
                    created.geometry.as_ref().unwrap(), 0, 0,
                )?);
                write_manifest(&directory, manifest.as_mut().unwrap())?;
            } else {
                bail!("workspace does not exist");
            }
            if mount_at(&directory)?.is_some() {
                bail!("workspace has an existing mount; reconciliation required");
            }
            private_directory(&directory.join("mnt"), true)?;
            let manifest = manifest.as_mut().unwrap();
            image_file(&directory, manifest)?;
            if manifest.state != "retained" {
                let checked = run_tool(
                    "e2fsck",
                    &[
                        "-fn".into(),
                        directory.join("workspace.img").display().to_string(),
                    ],
                    manifest.config.mount_timeout_seconds,
                    false,
                    Some(lease.as_ref().unwrap().as_raw_fd()),
                )?;
                if checked.0 != 0 {
                    bail!("workspace needs offline maintenance; automatic repair is forbidden");
                }
            }
            if manifest.geometry.is_none() {
                manifest.geometry = Some(maintenance::geometry(
                    &directory.join("workspace.img"),
                    lease.as_ref().unwrap(),
                    manifest.config.mount_timeout_seconds,
                )?);
            }
            manifest.state = "mounting".into();
            manifest.usage_generation = manifest.usage_generation.checked_add(1)
                .filter(|value| *value <= i64::MAX as u64)
                .context("workspace usage generation exhausted")?;
            manifest.reclaim_sample = None;
            manifest.execution_session_id = session.into();
            manifest.last_error = None;
            changed = true;
            write_manifest(&directory, manifest)?;
            let mount_started = Instant::now();
            let log = OpenOptions::new()
                .write(true)
                .create(true)
                .mode(0o600)
                .custom_flags(nix::libc::O_NOFOLLOW)
                .open(directory.join("fuse2fs.log"))?;
            let info = log.metadata()?;
            if !info.is_file() || info.uid() != uid() || info.nlink() != 1 {
                bail!("unsafe workspace daemon log");
            }
            log.set_len(0)?;
            let mut command = Command::new(tool("fuse2fs")?);
            command
                .arg(directory.join("workspace.img"))
                .arg(directory.join("mnt"))
                .args(["-o", "rw", "-f"])
                .stdin(Stdio::null())
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            let fd = lease.as_ref().unwrap().as_raw_fd();
            unsafe {
                command.pre_exec(move || {
                    if nix::libc::setsid() == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                    let flags = nix::libc::fcntl(fd, nix::libc::F_GETFD);
                    if flags == -1
                        || nix::libc::fcntl(fd, nix::libc::F_SETFD, flags & !nix::libc::FD_CLOEXEC)
                            == -1
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            process = Some(command.spawn()?);
            let deadline =
                Instant::now() + Duration::from_secs(manifest.config.mount_timeout_seconds);
            let mut poll_delay = Duration::from_millis(5);
            while process.as_mut().unwrap().try_wait()?.is_none() {
                if let Some(entry) = mount_at(&directory)? {
                    if !owned_mount(&directory, &entry) {
                        bail!("unexpected filesystem mounted at workspace target");
                    }
                    manifest.state = "mounted".into();
                    manifest.mount_id = Some(entry.id);
                    manifest.daemon_pid = Some(process.as_ref().unwrap().id());
                    manifest.timings.insert(
                        "mount_seconds".into(),
                        mount_started.elapsed().as_secs_f64(),
                    );
                    write_manifest(&directory, manifest)?;
                    return Ok(WorkspaceHandle {
                        directory: directory.clone(),
                        manifest: manifest.clone(),
                        _lease: lease.take().unwrap(),
                        process: process.take().unwrap(),
                    });
                }
                if Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(poll_delay);
                poll_delay = (poll_delay * 2).min(Duration::from_millis(50));
            }
            bail!("fuse2fs failed to establish a verified workspace mount")
        })();
        match outcome {
            Ok(handle) => {
                let path = handle.directory.join("mnt");
                self.handles.insert(id.into(), handle);
                Ok(path)
            }
            Err(error) => {
                let cleanup = (|| -> Result<()> {
                    if let Some(child) = process.as_mut() {
                        if child.try_wait()?.is_none() {
                            child.kill()?;
                            wait_child(child, 5)?;
                        }
                        if let Some(entry) = mount_at(&directory)? {
                            if owned_mount(&directory, &entry) {
                                run_tool(
                                    "fusermount",
                                    &["-u".into(), directory.join("mnt").display().to_string()],
                                    10,
                                    true,
                                    lease.as_ref().map(AsRawFd::as_raw_fd),
                                )?;
                            }
                        }
                    }
                    Ok(())
                })();
                if changed {
                    if let Some(manifest) = manifest.as_mut() {
                        manifest.state = "needs_check".into();
                        manifest.last_error = Some(match cleanup {
                            Ok(()) => format!("{error:#}"),
                            Err(ref cleanup) => format!("{error:#}; cleanup: {cleanup:#}"),
                        });
                        write_manifest(&directory, manifest)?;
                    }
                }
                Err(error)
            }
        }
    }

    pub fn verify(&mut self, id: &str) -> Result<()> {
        let handle = self
            .handles
            .get_mut(id)
            .context("workspace is not mounted by this manager")?;
        let entry = mount_at(&handle.directory)?;
        if handle.process.try_wait()?.is_some()
            || !entry
                .as_ref()
                .map(|entry| {
                    owned_mount(&handle.directory, entry)
                        && Some(entry.id) == handle.manifest.mount_id
                })
                .unwrap_or(false)
        {
            if handle.manifest.state != "needs_check" {
                handle.manifest.state = "needs_check".into();
                handle.manifest.last_error = Some("workspace mount is no longer healthy; ordinary-directory fallback is forbidden".into());
                write_manifest(&handle.directory, &mut handle.manifest)?;
            }
            bail!("workspace mount is no longer healthy; ordinary-directory fallback is forbidden");
        }
        if let Err(error) = image_file(&handle.directory, &handle.manifest) {
            if handle.manifest.state != "needs_check" {
                handle.manifest.state = "needs_check".into();
                handle.manifest.last_error = Some(error.to_string());
                write_manifest(&handle.directory, &mut handle.manifest)?;
            }
            return Err(error);
        }
        Ok(())
    }

    pub fn release(&mut self, id: &str) -> Result<()> {
        validate_id(id)?;
        let Some(handle) = self.handles.get_mut(id) else {
            return Ok(());
        };
        let unmount_started = Instant::now();
        let entry = mount_at(&handle.directory)?;
        let timeout = handle.manifest.config.mount_timeout_seconds;
        if let Some(entry) = &entry {
            if !owned_mount(&handle.directory, entry) || Some(entry.id) != handle.manifest.mount_id
            {
                bail!("refusing to unmount an unexpected filesystem");
            }
            if let Err(error) = run_tool(
                "fusermount",
                &[
                    "-u".into(),
                    handle.directory.join("mnt").display().to_string(),
                ],
                timeout,
                true,
                Some(handle._lease.as_raw_fd()),
            ) {
                let message = error.to_string();
                if handle.manifest.last_error.as_deref() != Some(message.as_str()) {
                    handle.manifest.last_error = Some(message);
                    write_manifest(&handle.directory, &mut handle.manifest)?;
                }
                return Err(error);
            }
        }
        let status = wait_child(&mut handle.process, timeout)?;
        if mount_at(&handle.directory)?.is_some() {
            bail!("workspace remains mounted");
        }
        let mut handle = self.handles.remove(id).unwrap();
        let healthy = status.success() && entry.is_some();
        image_file(&handle.directory, &handle.manifest)?.sync_all()?;
        handle.manifest.state = if healthy { "retained" } else { "needs_check" }.into();
        handle.manifest.mount_id = None;
        handle.manifest.daemon_pid = None;
        handle.manifest.reclaim_sample = None;
        if healthy {
            let sampled = (|| -> Result<Value> {
                let geometry = maintenance::geometry(&handle.directory.join("workspace.img"), &handle._lease, timeout)?;
                crate::workspace_reclaim::sample(
                    image_file(&handle.directory, &handle.manifest)?.metadata()?.blocks() * 512,
                    &geometry, handle.manifest.generation.unwrap_or(0), handle.manifest.usage_generation,
                )
            })();
            match sampled {
                Ok(sample) => { handle.manifest.reclaim_sample = Some(sample); handle.manifest.reclaim_sample_error = None; }
                Err(error) => { handle.manifest.reclaim_sample_error = Some(error.to_string().chars().take(1000).collect()); }
            }
        }
        handle.manifest.last_error = if healthy {
            None
        } else {
            Some("workspace daemon or mount disappeared unexpectedly".into())
        };
        handle.manifest.timings.insert(
            "unmount_sync_seconds".into(),
            unmount_started.elapsed().as_secs_f64(),
        );
        write_manifest(&handle.directory, &mut handle.manifest)?;
        unlock_completed_lease(&handle._lease)?;
        if !healthy {
            bail!("workspace was not cleanly unmounted; offline check required");
        }
        Ok(())
    }

    pub fn inspect(&self, id: &str, user: &str) -> Result<Value> {
        let directory = self.directory(id, false)?;
        let manifest = self.read_manifest(&directory, id, user)?;
        let mut value = serde_json::to_value(&manifest)?;
        let entry = mount_at(&directory)?;
        let mounted = entry.is_some();
        let verified = entry
            .as_ref()
            .map(|entry| owned_mount(&directory, entry) && Some(entry.id) == manifest.mount_id)
            .unwrap_or(false);
        if mounted && !verified {
            value["state"] = json!("needs_check");
            value["last_error"] = json!("workspace has an unowned or unexpected mount");
        }
        value["path"] = json!(directory.join("mnt"));
        value["mounted"] = json!(mounted);
        if let Some((at, revision, space)) = self.space_cache.lock().unwrap().get(id) {
            if *revision == manifest.revision
                && at.elapsed() < Duration::from_secs(1)
                && (!mounted || verified)
            {
                value["space"] = space.clone();
                return Ok(value);
            }
        }
        let mut space =
            json!({"internal":null,"backing":null,"logical_bytes":null,"allocated_bytes":null});
        let result = (|| -> Result<()> {
            if manifest.state != "deleted" {
                let info = image_file(&directory, &manifest)?.metadata()?;
                space["logical_bytes"] = json!(info.len());
                space["allocated_bytes"] = json!(info.blocks() * 512);
            }
            for (name, path) in [
                ("backing", directory.clone()),
                ("internal", directory.join("mnt")),
            ] {
                if name == "internal" && !verified {
                    continue;
                }
                let (_, output, _) = run_tool(
                    "stat",
                    &[
                        "-f".into(),
                        "-c".into(),
                        "%S %b %a %c %d %T".into(),
                        path.display().to_string(),
                    ],
                    5,
                    true,
                    None,
                )?;
                let fields: Vec<&str> = output.split_whitespace().collect();
                if fields.len() != 6 {
                    bail!("invalid filesystem space response");
                }
                let unit: u64 = fields[0].parse()?;
                space[name] = json!({"total_bytes":unit * fields[1].parse::<u64>()?,"available_bytes":unit * fields[2].parse::<u64>()?,
                    "total_inodes":fields[3].parse::<u64>()?,"free_inodes":fields[4].parse::<u64>()?,"filesystem":fields[5]});
            }
            Ok(())
        })();
        if let Err(error) = result {
            space["error"] = json!(error.to_string());
        }
        self.space_cache.lock().unwrap().insert(
            id.into(),
            (Instant::now(), manifest.revision, space.clone()),
        );
        value["space"] = space;
        Ok(value)
    }

    pub fn check(&self, id: &str, user: &str) -> Result<Value> {
        let directory = self.directory(id, false)?;
        let _lease = lease(&directory)?;
        let mut manifest = self.read_manifest(&directory, id, user)?;
        if mount_at(&directory)?.is_some() || self.handles.contains_key(id) {
            bail!("offline check requires an unmounted workspace");
        }
        if matches!(
            manifest.state.as_str(),
            "deleting" | "deleted" | "maintenance" | "needs_recovery"
        ) {
            bail!("cannot check a deleted/deleting workspace");
        }
        image_file(&directory, &manifest)?;
        let (code, stdout, stderr) = run_tool(
            "e2fsck",
            &[
                "-fn".into(),
                directory.join("workspace.img").display().to_string(),
            ],
            manifest.config.mount_timeout_seconds,
            false,
            Some(_lease.as_raw_fd()),
        )?;
        manifest.state = if code == 0 { "retained" } else { "needs_check" }.into();
        manifest.mount_id = None;
        manifest.daemon_pid = None;
        manifest.last_error = if code == 0 {
            None
        } else {
            Some(stderr.clone())
        };
        write_manifest(&directory, &mut manifest)?;
        unlock_completed_lease(&_lease)?;
        Ok(
            json!({"workspace_id":id,"exit_code":code,"stdout":stdout,"stderr":stderr,"state":manifest.state}),
        )
    }

    pub fn reconcile_released(&self, id: &str, user: &str) -> Result<()> {
        let directory = match self.directory(id, false) {
            Ok(directory) => directory,
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .map(|error| error.kind() == std::io::ErrorKind::NotFound)
                    .unwrap_or(false) =>
            {
                return Ok(())
            }
            Err(error) => return Err(error),
        };
        let _lease = lease(&directory)?;
        let mut manifest = self.read_manifest(&directory, id, user)?;
        if mount_at(&directory)?.is_some() {
            bail!("workspace predecessor remains mounted; takeover is forbidden");
        }
        if !matches!(
            manifest.state.as_str(),
            "retained" | "deleted" | "needs_check" | "maintenance" | "needs_recovery"
        ) {
            manifest.state = "needs_check".into();
            manifest.mount_id = None;
            manifest.daemon_pid = None;
            manifest.last_error =
                Some("predecessor ownership ended without a verified clean unmount".into());
            write_manifest(&directory, &mut manifest)?;
        }
        unlock_completed_lease(&_lease)?;
        Ok(())
    }

    pub fn delete(&self, id: &str, user: &str) -> Result<Value> {
        let directory = self.directory(id, false)?;
        let _lease = lease(&directory)?;
        let mut manifest = self.read_manifest(&directory, id, user)?;
        if matches!(manifest.state.as_str(), "maintenance" | "needs_recovery")
            || fs::symlink_metadata(directory.join("maintenance-backup.img")).is_ok()
            || fs::symlink_metadata(directory.join("maintenance-candidate.img")).is_ok()
        {
            bail!("finalize maintenance recovery before deleting workspace data");
        }
        if manifest.state == "deleted" {
            unlock_completed_lease(&_lease)?;
            return Ok(serde_json::to_value(manifest)?);
        }
        if mount_at(&directory)?.is_some() || self.handles.contains_key(id) {
            bail!("refusing to delete a mounted workspace");
        }
        let image_exists = match fs::symlink_metadata(directory.join("workspace.img")) {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if manifest.state != "deleting" || image_exists {
            image_file(&directory, &manifest)?;
        }
        manifest.state = "deleting".into();
        write_manifest(&directory, &mut manifest)?;
        if image_exists {
            fs::remove_file(directory.join("workspace.img"))?;
        }
        manifest.state = "deleted".into();
        manifest.mount_id = None;
        manifest.daemon_pid = None;
        manifest.last_error = None;
        write_manifest(&directory, &mut manifest)?;
        unlock_completed_lease(&_lease)?;
        Ok(serde_json::to_value(manifest)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "subprocess fixture invoked by formatter_lease_survives_owner_crash"]
    fn formatter_lease_holder_child() {
        let directory = PathBuf::from(std::env::var("CAPE_TEST_LEASE_DIRECTORY").unwrap());
        let lease = lease(&directory).unwrap();
        run_tool("mkfs.ext4", &[], 5, true, Some(lease.as_raw_fd())).unwrap();
    }

    #[test]
    fn formatter_lease_survives_owner_crash() {
        let root = tempfile::tempdir().unwrap();
        let helper = root.path().join("formatter");
        let ready = root.path().join("ready");
        fs::write(
            &helper,
            "#!/bin/sh\nprintf ready > \"$CAPE_TEST_READY_FILE\"\nexec /bin/sleep 1\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "workspace_fs::tests::formatter_lease_holder_child",
            ])
            .env("CAPE_TEST_LEASE_DIRECTORY", root.path())
            .env("CAPE_TEST_READY_FILE", &ready)
            .env("CAPE_WORKSPACE_MKFS_BIN", &helper)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !ready.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(ready.exists());
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(
            lease(root.path()).is_err(),
            "orphan formatter lost its writer fence"
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if lease(root.path()).is_ok() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "formatter fence did not end after helper exit"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn completed_lease_releases_inherited_copies_without_unlocking_next_owner() {
        let root = tempfile::tempdir().unwrap();
        let first = lease(root.path()).unwrap();
        let inherited = first.try_clone().unwrap();
        unlock_completed_lease(&first).unwrap();
        let next = lease(root.path()).unwrap();
        drop(first);
        drop(inherited);
        assert!(lease(root.path()).is_err());
        unlock_completed_lease(&next).unwrap();
    }

    #[test]
    fn rejects_unowned_paths_and_excludes_duplicate_writers() {
        let root = tempfile::tempdir().unwrap();
        let config = WorkspaceConfig::from_value(
            json!({"image_root":root.path(),"image_size_bytes":64 * 1024 * 1024}),
        )
        .unwrap();
        let manager = WorkspaceManager::new(config, "node".into(), "pool".into()).unwrap();
        for id in ["", "../escape", "ws-../other", "/tmp/workspace"] {
            assert!(manager.directory(id, true).is_err());
        }
        let id = new_workspace_id();
        let directory = manager.directory(&id, true).unwrap();
        let _first = lease(&directory).unwrap();
        assert!(lease(&directory).is_err());
        assert!(manager.reconcile_released(&id, "user").is_err());
        assert_eq!(
            decode_mount("/tmp/work\\040space\\134name"),
            "/tmp/work space\\name"
        );
        let line = "42 1 0:1 / /tmp/example/mnt rw - fuse.ext4 /tmp/example/workspace.img rw\n";
        assert_eq!(
            parse_mount_at(Path::new("/tmp/example"), &line.repeat(2))
                .unwrap()
                .unwrap()
                .id,
            42
        );
        assert!(parse_mount_at(
            Path::new("/tmp/example"),
            &(line.to_string() + &line.replacen("42 ", "43 ", 1))
        )
        .is_err());
    }

    #[test]
    fn never_truncates_or_formats_an_orphan_image() {
        let root = tempfile::tempdir().unwrap();
        let config = WorkspaceConfig::from_value(
            json!({"image_root":root.path(),"image_size_bytes":64 * 1024 * 1024}),
        )
        .unwrap();
        let mut manager = WorkspaceManager::new(config, "node".into(), "pool".into()).unwrap();
        let id = new_workspace_id();
        let directory = manager.directory(&id, true).unwrap();
        fs::write(directory.join("workspace.img"), b"preserve-orphan-data").unwrap();
        assert!(manager.prepare(&id, "user", "session", true).is_err());
        assert_eq!(
            fs::read(directory.join("workspace.img")).unwrap(),
            b"preserve-orphan-data"
        );
        assert!(!directory.join("manifest.json").exists());
    }

    #[test]
    fn interrupted_delete_intent_finishes_without_recreating_image() {
        let root = tempfile::tempdir().unwrap();
        let config = WorkspaceConfig::from_value(
            json!({"image_root":root.path(),"image_size_bytes":64 * 1024 * 1024}),
        )
        .unwrap();
        let manager = WorkspaceManager::new(config.clone(), "node".into(), "pool".into()).unwrap();
        let id = new_workspace_id();
        let directory = manager.directory(&id, true).unwrap();
        let mut manifest = WorkspaceManifest {
            schema_version: 3,
            revision: 0,
            workspace_id: id.clone(),
            user_id: "user".into(),
            pool: "pool".into(),
            node_id: "node".into(),
            execution_session_id: "session".into(),
            config,
            state: "deleting".into(),
            last_error: None,
            image_device: Some(1),
            image_inode: Some(1),
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
        assert_eq!(manager.delete(&id, "user").unwrap()["state"], "deleted");
        assert!(!directory.join("workspace.img").exists());
        assert_eq!(manager.delete(&id, "user").unwrap()["state"], "deleted");
        assert!(!owned_mount(
            &directory,
            &MountEntry {
                id: 1,
                filesystem: "fuse.ext4".into(),
                source: "/other/image.img".into()
            }
        ));
    }

    #[test]
    #[ignore = "requires a verified fuse2fs-capable node"]
    fn real_mount_reuse_release_check_and_delete() {
        // Preserve the disposable image on panic; automatic TempDir cleanup must
        // never recurse into a live mount while diagnosing a lifecycle failure.
        let root = tempfile::Builder::new()
            .prefix("cape-task16-rust-manager-")
            .tempdir()
            .unwrap()
            .into_path();
        eprintln!("workspace_test_root={}", root.display());
        let config = WorkspaceConfig::from_value(
            json!({"image_root":root,"image_size_bytes":64 * 1024 * 1024}),
        )
        .unwrap();
        let mut manager =
            WorkspaceManager::new(config.clone(), "node".into(), "pool".into()).unwrap();
        let mut other = WorkspaceManager::new(config, "node".into(), "pool".into()).unwrap();
        let id = new_workspace_id();
        let mount = manager.prepare(&id, "user", "session", true).unwrap();
        fs::write(mount.join("data"), b"workspace-data").unwrap();
        assert_eq!(
            manager.prepare(&id, "user", "session", true).unwrap(),
            mount
        );
        assert!(other.prepare(&id, "user", "other-session", false).is_err());
        assert!(manager.check(&id, "user").is_err());
        assert!(manager.delete(&id, "user").is_err());
        assert!(manager
            .prepare(&id, "other-user", "session", false)
            .is_err());
        manager.release(&id).unwrap();
        assert_eq!(manager.inspect(&id, "user").unwrap()["state"], "retained");
        assert_eq!(manager.check(&id, "user").unwrap()["exit_code"], 0);
        assert!(other.prepare(&id, "user", "other-session", true).is_err());
        let mount = other.prepare(&id, "user", "other-session", false).unwrap();
        assert_eq!(fs::read(mount.join("data")).unwrap(), b"workspace-data");
        other.release(&id).unwrap();
        assert_eq!(other.delete(&id, "user").unwrap()["state"], "deleted");
        assert_eq!(other.delete(&id, "user").unwrap()["state"], "deleted");
        assert!(other.prepare(&id, "user", "third-session", false).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
