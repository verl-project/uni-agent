// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::identity::{name, verifier, IdentityStore};
use crate::journal::Journal;
use anyhow::{ensure, Context, Result};
use nix::fcntl::{flock, FlockArg};
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

pub struct Instance {
    pub store: IdentityStore,
    pub database: PathBuf,
    _lock: InstanceLock,
}

struct InstanceLock {
    file: File,
    owner_pid: nix::libc::pid_t,
}

impl Drop for InstanceLock {
    fn drop(&mut self) {
        // close alone can retain flock until a concurrently forked child execs.
        if self.owner_pid == unsafe { nix::libc::getpid() } {
            let _ = flock(self.file.as_raw_fd(), FlockArg::Unlock);
        }
    }
}

impl Instance {
    pub fn initialize(directory: &Path, admin_name: &str, secret: &str) -> Result<String> {
        name(admin_name)?;
        verifier(secret)?;
        DirBuilder::new().mode(0o700).create(directory)?;
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(directory.join("controller.lock"))?
            .sync_all()?;
        let database = directory.join("state.sqlite3");
        let store = IdentityStore::create(&database, admin_name, secret)?;
        let scheduler_id = store.snapshot()?["scheduler_id"]
            .as_str()
            .context("invalid instance identity")?
            .to_owned();
        drop(store);
        let mut journal = Journal::open(&database)?;
        journal.close()?;
        File::open(directory)?.sync_all()?;
        File::open(
            directory
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?
        .sync_all()?;
        Ok(scheduler_id)
    }

    pub fn open(directory: &Path, secret: &str, now: f64) -> Result<Self> {
        let info = fs::symlink_metadata(directory)?;
        ensure!(
            info.is_dir()
                && info.uid() == unsafe { nix::libc::getuid() }
                && info.mode() & 0o077 == 0,
            "instance must be an owner-only directory"
        );
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_NONBLOCK)
            .open(directory.join("controller.lock"))?;
        let info = lock.metadata()?;
        ensure!(
            info.is_file()
                && info.uid() == unsafe { nix::libc::getuid() }
                && info.mode() & 0o077 == 0
                && info.nlink() == 1,
            "invalid instance lock"
        );
        flock(lock.as_raw_fd(), FlockArg::LockExclusiveNonblock)?;
        let lock = InstanceLock { file: lock, owner_pid: unsafe { nix::libc::getpid() } };
        let database = directory.join("state.sqlite3");
        let mut store = IdentityStore::open(&database)?;
        store.require_admin(secret, now)?;
        let db = rusqlite::Connection::open_with_flags(
            &database,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        let count: i64 = db.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='events'",
            [],
            |row| row.get(0),
        )?;
        ensure!(count == 1, "incomplete instance: event journal is missing");
        store.mutate(|state| {
            let actor = crate::identity::authenticate(state, secret, now)?;
            ensure!(actor["is_admin"] == true, "admin credential required");
            let scheduler = state["scheduler_id"].as_str().unwrap().to_owned();
            crate::identity::audit(
                state,
                actor["user_id"].as_str().unwrap(),
                "controller.start",
                &scheduler,
                now,
            );
            Ok(())
        })?;
        Ok(Self {
            store,
            database,
            _lock: lock,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::new_secret;

    #[test]
    #[ignore = "cross-language fixture; invoked by tests.test_instance"]
    fn cross_language_fixture() -> Result<()> {
        let input: serde_json::Value =
            serde_json::from_slice(&fs::read(std::env::var("CAPE_INSTANCE_FIXTURE")?)?)?;
        let directory = Path::new(input["directory"].as_str().context("directory")?);
        let secret = input["secret"].as_str().context("secret")?;
        if input["create"] == true {
            Instance::initialize(directory, "admin", secret)?;
        }
        let opened = Instance::open(directory, secret, 100.);
        if input["locked"] == true {
            let error = opened.err().context("expected exclusive lock conflict")?;
            ensure!(
                error.downcast_ref::<nix::errno::Errno>() == Some(&nix::errno::Errno::EWOULDBLOCK),
                "not a lock conflict"
            );
        } else {
            opened?;
        }
        Ok(())
    }

    #[test]
    fn strict_init_start_restart_and_lock() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let directory = temp.path().join("instance");
        let secret = new_secret()?;
        assert!(Instance::open(&directory, &secret, 100.).is_err());
        assert!(!directory.exists());
        let scheduler = Instance::initialize(&directory, "admin", &secret)?;
        assert!(Instance::initialize(&directory, "other", &secret).is_err());
        assert!(Instance::open(&directory, &new_secret()?, 100.).is_err());
        let instance = Instance::open(&directory, &secret, 100.)?;
        assert_eq!(instance.store.snapshot()?["scheduler_id"], scheduler);
        assert!(Instance::open(&directory, &secret, 100.).is_err());
        let inherited = instance._lock.file.try_clone()?;
        drop(instance);
        let reopened = Instance::open(&directory, &secret, 100.)?;
        drop(inherited);
        assert_eq!(reopened.store.snapshot()?["scheduler_id"], scheduler);
        assert_eq!(reopened.database, directory.join("state.sqlite3"));
        Ok(())
    }
}
