// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::RuntimeConfig;
use anyhow::{ensure, Result};
use std::collections::HashSet;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};

pub(super) fn check(config: &RuntimeConfig) -> Result<()> {
    let Some(root) = &config.execution_state_dir else {return Ok(());};
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(root)?;
    let path = root.join(".state-owner");
    let _ = crate::credential_file::private_parent(&path)?;
    let free = nix::sys::statvfs::statvfs(root)?;
    ensure!(free.blocks_available().saturating_mul(free.fragment_size()) >= config.execution_state_min_free_bytes,
        "execution state storage reserve exhausted; existing state preserved");
    if config.execution_state_max_bytes > 0 {
        let device=std::fs::symlink_metadata(root)?.dev();
        let mut pending=vec![root.clone()];let mut seen=HashSet::new();let mut total=0_u64;
        while let Some(path)=pending.pop(){
            let metadata=std::fs::symlink_metadata(&path)?;
            if metadata.dev()!=device{continue;}
            if !seen.insert((metadata.dev(),metadata.ino())){continue;}
            total=total.saturating_add(metadata.blocks().saturating_mul(512));
            ensure!(total<config.execution_state_max_bytes,"execution state storage quota exhausted; explicit purge required");
            if metadata.is_dir(){for entry in std::fs::read_dir(path)?{pending.push(entry?.path());}}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quota_and_reserve_errors_never_delete_state() -> Result<()> {
        let temporary=tempfile::tempdir()?;
        let root=temporary.path().join("retained");
        std::fs::DirBuilder::new().mode(0o700).create(&root)?;
        let marker=root.join("data");std::fs::write(&marker,vec![1_u8;8192])?;
        let mut config=RuntimeConfig::test_config("apptainer");
        config.execution_state_dir=Some(root);
        config.execution_state_min_free_bytes=0;config.execution_state_max_bytes=1;
        assert!(check(&config).unwrap_err().to_string().contains("quota"));
        config.execution_state_max_bytes=0;config.execution_state_min_free_bytes=u64::MAX;
        assert!(check(&config).unwrap_err().to_string().contains("reserve"));
        assert_eq!(std::fs::read(marker)?.len(),8192);
        Ok(())
    }
}
