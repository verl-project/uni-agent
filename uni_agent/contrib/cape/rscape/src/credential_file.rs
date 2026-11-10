// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::identity::{name as validate_name, new_secret, verifier};
use anyhow::{ensure, Context, Result};
use nix::fcntl::{flock, openat, FlockArg, OFlag};
use nix::sys::stat::Mode;
use serde_json::{json, Value};
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path};
use uuid::Uuid;

pub(crate) fn open_child(directory: &File, name: &Path, flags: OFlag, mode: Mode) -> Result<File> {
    let fd = openat(
        directory.as_raw_fd(),
        name,
        flags | OFlag::O_NOFOLLOW | OFlag::O_CLOEXEC,
        mode,
    )?;
    // openat returned a new descriptor owned exclusively by this File.
    Ok(unsafe { File::from_raw_fd(fd) })
}

pub(crate) fn private_parent(path: &Path) -> Result<(File, &Path)> {
    let filename = path.file_name().context("credential filename required")?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let parent = absolute.parent().context("credential parent required")?;
    let mut directory = File::open("/")?;
    for component in parent.components() {
        match component {
            Component::RootDir | Component::CurDir => (),
            Component::Normal(name) => {
                directory = open_child(
                    &directory,
                    Path::new(name),
                    OFlag::O_RDONLY | OFlag::O_DIRECTORY,
                    Mode::empty(),
                )?;
            }
            _ => anyhow::bail!("credential path must not contain parent traversal"),
        }
    }
    let info = directory.metadata()?;
    ensure!(
        info.uid() == unsafe { nix::libc::getuid() } && info.mode() & 0o077 == 0,
        "credential parent must be an owner-only directory"
    );
    Ok((directory, Path::new(filename)))
}

pub fn prepare_registration(
    path: &Path,
    controller: &str,
    username: Option<&str>,
    name: &str,
    expiry: Option<f64>,
) -> Result<Value> {
    prepare_registration_for(path, controller, username, name, expiry, None)
}

pub fn prepare_registration_for(
    path: &Path,
    controller: &str,
    username: Option<&str>,
    name: &str,
    expiry: Option<f64>,
    target_user_id: Option<&str>,
) -> Result<Value> {
    validate_name(name)?;
    if let Some(username) = username {
        validate_name(username)?;
    }
    ensure!(
        !controller.is_empty() && controller.len() <= 8192,
        "invalid controller URL"
    );
    ensure!(expiry.map(f64::is_finite).unwrap_or(true), "invalid expiry");
    let mut parameters = json!({"username":username,"name":name,"expires_at":expiry});
    if let Some(target) = target_user_id {
        ensure!(
            Uuid::parse_str(target)?.to_string() == target,
            "invalid target identity"
        );
        parameters["user_id"] = json!(target);
    }
    prepare_operation(path, controller, parameters)
}

pub fn prepare_operation(path: &Path, controller: &str, mut parameters: Value) -> Result<Value> {
    ensure!(
        !controller.is_empty() && controller.len() <= 8192 && parameters.is_object(),
        "invalid credential parameters"
    );
    let secret = new_secret()?;
    parameters["operation_id"] = json!(Uuid::new_v4().to_string());
    parameters["verifier"] = json!(verifier(&secret)?);
    let body =
        json!({"version":1,"secret":secret,"controller_url":controller,"registration":parameters});
    ensure!(
        body.to_string().len() + 1 <= 16384,
        "credential envelope too large"
    );
    let (directory, filename) = private_parent(path)?;
    let mut file = open_child(
        &directory,
        filename,
        OFlag::O_WRONLY | OFlag::O_CREAT | OFlag::O_EXCL,
        Mode::from_bits_truncate(0o600),
    )?;
    writeln!(file, "{body}")?;
    file.sync_all()?;
    directory.sync_all()?;
    Ok(body)
}

pub fn read_registration(path: &Path) -> Result<Value> {
    let (directory, filename) = private_parent(path)?;
    let file = open_child(
        &directory,
        filename,
        OFlag::O_RDONLY | OFlag::O_NONBLOCK,
        Mode::empty(),
    )?;
    let info = file.metadata()?;
    ensure!(
        info.is_file()
            && info.uid() == unsafe { nix::libc::getuid() }
            && info.mode() & 0o077 == 0
            && info.nlink() == 1
            && info.len() <= 16384,
        "credential must be an owner-only, unlinked regular file"
    );
    let mut bytes = Vec::new();
    (&file).take(16385).read_to_end(&mut bytes)?;
    let body: Value = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("invalid credential envelope"))?;
    let secret = body["secret"]
        .as_str()
        .context("invalid credential envelope")?;
    ensure!(
        body["version"] == 1
            && body["controller_url"].is_string()
            && body["registration"]["verifier"] == verifier(secret)?,
        "invalid credential envelope"
    );
    file.sync_all()?;
    directory.sync_all()?;
    Ok(body)
}

pub fn lock_registration(path: &Path) -> Result<File> {
    let (directory, filename) = private_parent(path)?;
    let file = open_child(
        &directory,
        filename,
        OFlag::O_RDWR | OFlag::O_NONBLOCK,
        Mode::empty(),
    )?;
    let info = file.metadata()?;
    ensure!(
        info.is_file()
            && info.uid() == unsafe { nix::libc::getuid() }
            && info.mode() & 0o077 == 0
            && info.nlink() == 1,
        "invalid enrollment credential file"
    );
    flock(file.as_raw_fd(), FlockArg::LockExclusiveNonblock)
        .context("another agent may already use this enrollment")?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{symlink, PermissionsExt};

    #[test]
    fn immutable_private_recoverable_envelope() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))?;
        let path = dir.path().join("token.json");
        let body = prepare_registration(
            &path,
            "https://controller.example",
            Some("alice"),
            "laptop",
            None,
        )?;
        assert_eq!(read_registration(&path)?, body);
        let guard = lock_registration(&path)?;
        assert!(lock_registration(&path).is_err());
        drop(guard);
        drop(lock_registration(&path)?);
        assert_eq!(std::fs::metadata(&path)?.mode() & 0o777, 0o600);
        assert!(prepare_registration(
            &path,
            "https://controller.example",
            Some("bob"),
            "laptop",
            None
        )
        .is_err());
        let link = dir.path().join("link");
        symlink(&path, &link)?;
        assert!(read_registration(&link).is_err());
        std::fs::hard_link(&path, dir.path().join("hardlink"))?;
        assert!(read_registration(&path).is_err());
        Ok(())
    }

    #[test]
    fn rejects_public_and_symlinked_parent() -> Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755))?;
        assert!(prepare_registration(
            &dir.path().join("token"),
            "https://controller.example",
            None,
            "laptop",
            None
        )
        .is_err());
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))?;
        symlink(dir.path(), dir.path().join("link"))?;
        assert!(prepare_registration(
            &dir.path().join("link/token"),
            "https://controller.example",
            None,
            "laptop",
            None
        )
        .is_err());
        Ok(())
    }
}
