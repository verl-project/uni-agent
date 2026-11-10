// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use crate::credential_file::{open_child, private_parent};
use nix::fcntl::{flock, FlockArg, OFlag, renameat};
use nix::sys::stat::Mode;
use ring::digest::{digest, SHA256};
use serde_json::{json, Value};
use std::fs::{DirBuilder, File};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Path, PathBuf};

pub fn validate(value: &str) -> Result<()> {
    ensure!(!value.is_empty() && value.len() <= 256 && !value.chars().any(|c| c.is_whitespace() || c.is_control()),
        "session_id must be 1-256 UTF-8 bytes without whitespace/control characters");
    Ok(())
}

fn check(file: &File) -> Result<()> {
    let m = file.metadata()?;
    ensure!(m.is_file() && m.uid() == unsafe { nix::libc::getuid() } && m.mode() & 0o077 == 0
        && m.nlink() == 1 && m.len() <= 16384, "session cache must be an owner-only regular file");
    Ok(())
}

pub struct Selection {
    pub session_id: String,
    directory: File,
    _lock: File,
    filename: String,
    body: Value,
}

impl Selection {
    pub fn new(identity: &Value, explicit: Option<&str>, root: Option<&Path>, project: Option<&Path>) -> Result<Self> {
        let scheduler = identity["scheduler_id"].as_str().filter(|s| !s.is_empty()).context("controller identity missing")?;
        let user = identity["user_id"].as_str().filter(|s| !s.is_empty()).context("user identity missing")?;
        if let Some(id) = explicit { validate(id)?; }
        let project = std::fs::canonicalize(project.map(PathBuf::from).unwrap_or(std::env::current_dir()?))?;
        let project = project.to_str().context("non-UTF8 project path")?;
        let root = root.map(PathBuf::from).or_else(|| std::env::var_os("CAPE_SESSION_STATE_DIR").map(PathBuf::from))
            .unwrap_or_else(|| std::env::var_os("XDG_STATE_HOME").map(PathBuf::from)
                .unwrap_or_else(|| {
                    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
                    std::fs::canonicalize(&home).unwrap_or(home).join(".local/state")
                })
                .join("cape/sessions"));
        DirBuilder::new().recursive(true).mode(0o700).create(&root)?;
        let key = digest(&SHA256, format!("{scheduler}\0{user}\0{project}").as_bytes());
        let filename = format!("{}.json", key.as_ref().iter().map(|b|format!("{b:02x}")).collect::<String>());
        let path = root.join(&filename);
        let (directory, _) = private_parent(&path)?;
        let lock = open_child(&directory, Path::new(&format!("{filename}.lock")),
            OFlag::O_RDWR | OFlag::O_CREAT | OFlag::O_NONBLOCK, Mode::from_bits_truncate(0o600))?;
        check(&lock)?;
        flock(lock.as_raw_fd(), FlockArg::LockExclusive)?;
        let previous = match open_child(&directory, Path::new(&filename), OFlag::O_RDONLY | OFlag::O_NONBLOCK, Mode::empty()) {
            Ok(mut file) => {
                check(&file)?;
                let mut text = String::new(); file.read_to_string(&mut text)?;
                let body: Value = serde_json::from_str(&text)?;
                ensure!(body["version"] == 1 && body["scheduler_id"] == scheduler && body["user_id"] == user
                    && body["project"] == project, "session cache context mismatch");
                let id = body["session_id"].as_str().context("cached session_id missing")?;
                validate(id)?; Some(id.to_owned())
            }
            Err(error) if error.downcast_ref::<nix::errno::Errno>() == Some(&nix::errno::Errno::ENOENT) => None,
            Err(error) => return Err(error),
        };
        let session_id = explicit.map(str::to_owned).or(previous).unwrap_or_else(|| format!("session-{}",uuid::Uuid::new_v4().simple()));
        let body = json!({"version":1,"scheduler_id":scheduler,"user_id":user,"project":project,"session_id":session_id});
        Ok(Self {session_id, directory, _lock:lock, filename, body})
    }

    pub fn accepted(&self, response: &Value) -> Result<()> {
        ensure!(response["session_id"] == self.session_id && response["request_id"].as_str().map(|s|!s.is_empty()).unwrap_or(false),
            "submission response does not confirm selected logical session");
        let name = format!("{}.{}.tmp",self.filename,uuid::Uuid::new_v4().simple());
        let result = (|| -> Result<()> {
            let mut file = open_child(&self.directory, Path::new(&name), OFlag::O_WRONLY | OFlag::O_CREAT | OFlag::O_EXCL,
                Mode::from_bits_truncate(0o600))?;
            file.write_all(serde_json::to_string(&self.body)?.as_bytes())?;
            file.sync_all()?;
            renameat(Some(self.directory.as_raw_fd()), Path::new(&name), Some(self.directory.as_raw_fd()), Path::new(&self.filename))?;
            self.directory.sync_all()?; Ok(())
        })();
        let _ = nix::unistd::unlinkat(Some(self.directory.as_raw_fd()), Path::new(&name), nix::unistd::UnlinkatFlags::NoRemoveDir);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_session_commits_only_after_acceptance_and_is_scoped() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("cache");
        let identity = json!({"scheduler_id":"controller-A","user_id":"alice"});
        let first = Selection::new(&identity, Some("kept"), Some(&root), Some(temp.path()))?;
        first.accepted(&json!({"request_id":"r1","session_id":"kept"}))?;
        drop(first);
        drop(Selection::new(&identity, Some("failed"), Some(&root), Some(temp.path()))?);
        let next = Selection::new(&identity, None, Some(&root), Some(temp.path()))?;
        assert_eq!(next.session_id, "kept");
        drop(next);
        for other in [json!({"scheduler_id":"controller-B","user_id":"alice"}),
                      json!({"scheduler_id":"controller-A","user_id":"bob"})] {
            assert_ne!(Selection::new(&other, None, Some(&root), Some(temp.path()))?.session_id, "kept");
        }
        Ok(())
    }

    #[test]
    fn validates_logical_names() {
        for id in ["", "bad name", "bad\n"] { assert!(validate(id).is_err()); }
        assert!(validate("project/run-A").is_ok());
        assert!(validate(&"x".repeat(257)).is_err());
    }
}
