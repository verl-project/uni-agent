# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline candidate-image transactions; callers must first release runtime users."""

import copy
import os
import re
import stat
import time

from .workspace import MIN_IMAGE_SIZE_BYTES
from .workspace_fs import WorkspaceError


from .workspace_policy import MaintenancePolicy


def publish(manager, directory, manifest):
    if manifest.get("maintenance"):
        extra = 0
        for name in ("maintenance-candidate.img", "maintenance-backup.img"):
            path = directory / name
            if os.path.lexists(path):
                fd, info = checked_file(path)
                os.close(fd)
                extra += info.st_blocks * 512
        manifest["maintenance"]["recovery_allocated_bytes"] = extra
    manager._write(directory, manifest)
    observer = getattr(manager, "maintenance_observer", None)
    if observer is not None:
        observer(copy.deepcopy(manifest))


def geometry(manager, path, lease, timeout):
    text = manager._run("dumpe2fs", ["-h", path], timeout, lease_fd=lease).stdout
    values = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
    try:
        result = {"block_size": int(values["Block size"]), "blocks": int(values["Block count"]),
                  "free_blocks": int(values["Free blocks"]), "inodes": int(values["Inode count"]),
                  "free_inodes": int(values["Free inodes"]), "blocks_per_group": int(values["Blocks per group"])}
        result["size_bytes"] = result["blocks"] * result["block_size"]
        if result["block_size"] not in (1024, 2048, 4096, 8192, 16384, 32768, 65536) or result["blocks"] <= 0:
            raise ValueError("invalid geometry")
        return result
    except (KeyError, ValueError):
        raise WorkspaceError("invalid ext4 geometry returned by dumpe2fs")


def sync_directory(directory):
    fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def probe_disposable_tools(manager, workspace_id):
    """Capability fixture only: never a transaction path for user images."""
    if not os.path.basename(manager.config.image_root).startswith(".cape-workspace-probe-"):
        raise WorkspaceError("maintenance capability probe requires an isolated probe root")
    with manager._guard(workspace_id):
        directory = manager._directory(workspace_id, False)
        lease = manager._lease(directory)
        try:
            manifest = manager._read(directory, workspace_id, "__cape_probe__")
            if manifest["state"] != "retained" or manager._mount_at(directory) is not None or workspace_id in manager.handles:
                raise WorkspaceError("capability probe requires an offline disposable image")
            fd = manager._image_fd(directory, manifest)
            os.close(fd)
            image, candidate = directory / "workspace.img", directory / "probe-maintenance.img"
            manager._run("e2fsck", ["-fn", image], 30, lease_fd=lease)
            manager._run("cp", ["--reflink=auto", "--sparse=always", "--preserve=mode", "--no-clobber", image, candidate], 30, lease_fd=lease)
            fd, info = checked_file(candidate)
            os.close(fd)
            if info.st_ino == manifest["image_inode"] and info.st_dev == manifest["image_device"]:
                raise WorkspaceError("probe candidate must be independent")
            hashes = manager._run("sha256sum", [image, candidate], 30, lease_fd=lease).stdout.splitlines()
            if len(hashes) != 2 or hashes[0].split()[0] != hashes[1].split()[0]:
                raise WorkspaceError("probe candidate checksum mismatch")
            manager._run("e2fsck", ["-f", "-p", candidate], 30, lease_fd=lease)
            fd = os.open(candidate, os.O_WRONLY | os.O_NOFOLLOW)
            try:
                os.ftruncate(fd, 64 * 1024**2)
            finally:
                os.close(fd)
            manager._run("resize2fs", [candidate, "65536K"], 30, lease_fd=lease)
            if geometry(manager, candidate, lease, 30)["size_bytes"] != 64 * 1024**2:
                raise WorkspaceError("probe growth geometry mismatch")
            manager._run("e2fsck", ["-f", "-p", "-E", "discard", candidate], 30, lease_fd=lease)
            manager._run("resize2fs", [candidate, "32768K"], 30, lease_fd=lease)
            after = geometry(manager, candidate, lease, 30)
            if after["size_bytes"] != 32 * 1024**2:
                raise WorkspaceError("probe shrink geometry mismatch")
            fd = os.open(candidate, os.O_WRONLY | os.O_NOFOLLOW)
            try:
                os.ftruncate(fd, after["size_bytes"])
                os.fsync(fd)
            finally:
                os.close(fd)
            manager._run("e2fsck", ["-fn", candidate], 30, lease_fd=lease)
            fd, info = checked_file(candidate, info.st_dev, info.st_ino)
            os.close(fd)
            os.replace(candidate, image)
            sync_directory(directory)
            manifest.update(image_device=info.st_dev, image_inode=info.st_ino, geometry=after)
            manager._write(directory, manifest)
            manager._unlock_completed_lease(lease)
        finally:
            os.close(lease)


def checked_file(path, device=None, inode=None):
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1
            or info.st_mode & 0o077 or (device is not None and info.st_dev != device)
            or (inode is not None and info.st_ino != inode)):
        os.close(fd)
        raise WorkspaceError("unsafe maintenance image identity or ownership")
    return fd, info


def validate_request(action, target, maximum):
    if action not in ("grow", "reclaim", "shrink"):
        raise WorkspaceError("unknown image maintenance action")
    if action == "reclaim":
        if target is not None:
            raise WorkspaceError("reclaim does not accept a target capacity")
    elif type(target) is not int or not MIN_IMAGE_SIZE_BYTES <= target <= maximum or target % (1024 ** 2):
        raise WorkspaceError("target capacity must be whole MiB within deployment limits")


def execute(manager, workspace_id, user_id, action, target, operation_id, policy=None, maximum_bytes=None):
    policy = MaintenancePolicy(policy).to_dict()
    maximum = manager.config.max_image_size_bytes if maximum_bytes is None else maximum_bytes
    if type(maximum) is not int or not MIN_IMAGE_SIZE_BYTES <= maximum < 2 ** 63:
        raise WorkspaceError("invalid deployment capacity limit")
    validate_request(action, target, maximum)
    if not isinstance(operation_id, str) or re.fullmatch(r"[0-9a-f]{32}", operation_id) is None:
        raise WorkspaceError("invalid maintenance operation id")
    timeout = policy["timeout_seconds"]
    with manager._guard(workspace_id):
        directory = manager._directory(workspace_id, False)
        lease = manager._lease(directory)
        manifest = None
        started = time.monotonic()
        try:
            manifest = manager._read(directory, workspace_id, user_id)
            previous = manifest.get("maintenance", {})
            if previous.get("id") == operation_id:
                if previous.get("action") != action or previous.get("target_bytes") != target:
                    raise WorkspaceError("operation id was reused with different parameters")
                if previous.get("state") == "completed":
                    manager._unlock_completed_lease(lease)
                    return copy.deepcopy(previous)
                raise WorkspaceError("interrupted maintenance requires explicit recovery")
            if manifest["state"] != "retained" or workspace_id in manager.handles or manager._mount_at(directory) is not None:
                raise WorkspaceError("maintenance requires a clean unmounted exclusively owned workspace")
            candidate = directory / "maintenance-candidate.img"
            backup = directory / "maintenance-backup.img"
            if os.path.lexists(candidate) or os.path.lexists(backup):
                raise WorkspaceError("maintenance recovery files exist; reconcile before another operation")
            image = directory / "workspace.img"
            fd = manager._image_fd(directory, manifest)
            try:
                original = os.fstat(fd)
            finally:
                os.close(fd)
            current = original.st_size
            if (action == "grow" and target <= current) or (action == "shrink" and target >= current):
                raise WorkspaceError("target must change capacity in the requested direction")
            manager._run("e2fsck", ["-fn", image], timeout, lease_fd=lease)
            before = geometry(manager, image, lease, timeout)
            if before["size_bytes"] != current:
                raise WorkspaceError("image envelope differs from filesystem geometry")
            # Conservative reservation: a full sparse-copy allocation plus possible growth.
            needed = original.st_blocks * 512 + max(0, (target or current) - current) + policy["backing_reserve_bytes"]
            storage = os.statvfs(directory)
            if storage.f_bavail * storage.f_frsize < needed:
                raise WorkspaceError("insufficient backing space for recoverable maintenance")
            operation = {"id": operation_id, "action": action, "target_bytes": target, "state": "running",
                         "phase": "copying", "before": before, "started_at": time.time(),
                         "original_device": original.st_dev, "original_inode": original.st_ino,
                         "original_schema": manifest["schema_version"], "before_allocated_bytes": original.st_blocks * 512}
            manifest.update(schema_version=3, state="maintenance", maintenance=operation, last_error=None)
            publish(manager, directory, manifest)
            manager._run("cp", ["--reflink=auto", "--sparse=always", "--preserve=mode", "--no-clobber", image, candidate], timeout, lease_fd=lease)
            fd, info = checked_file(candidate)
            try:
                if info.st_ino == original.st_ino and info.st_dev == original.st_dev:
                    raise WorkspaceError("maintenance candidate is not an independent file")
                os.fsync(fd)
            finally:
                os.close(fd)
            hashes = manager._run("sha256sum", [image, candidate], timeout, lease_fd=lease).stdout.splitlines()
            if len(hashes) != 2 or hashes[0].split()[0] != hashes[1].split()[0]:
                raise WorkspaceError("maintenance copy verification failed")
            operation.update(phase="applying", candidate_inode=info.st_ino, candidate_device=info.st_dev,
                             original_sha256=hashes[0].split()[0])
            publish(manager, directory, manifest)
            # Read-only checking does not refresh the clean-check metadata required
            # by resize2fs after a later mount. Only the independent candidate is writable.
            manager._run("e2fsck", ["-f", "-p", candidate], timeout, lease_fd=lease)
            if action == "grow":
                fd = os.open(str(candidate), os.O_WRONLY | os.O_NOFOLLOW)
                try:
                    os.ftruncate(fd, target)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                manager._run("resize2fs", [candidate, str(target // 1024) + "K"], timeout, lease_fd=lease)
            elif action == "shrink":
                manager._run("resize2fs", [candidate, str(target // 1024) + "K"], timeout, lease_fd=lease)
            else:
                manager._run("e2fsck", ["-f", "-p", "-E", "discard", candidate], timeout, lease_fd=lease)
            after = geometry(manager, candidate, lease, timeout)
            if after["size_bytes"] != (current if action == "reclaim" else target):
                raise WorkspaceError("resizer did not establish the requested geometry")
            if action == "shrink":
                fd = os.open(str(candidate), os.O_WRONLY | os.O_NOFOLLOW)
                try:
                    os.ftruncate(fd, after["size_bytes"])
                    os.fsync(fd)
                finally:
                    os.close(fd)
            manager._run("e2fsck", ["-fn", candidate], timeout, lease_fd=lease)
            fd, info = checked_file(candidate, operation["candidate_device"], operation["candidate_inode"])
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            operation.update(phase="committing", after=after, after_allocated_bytes=info.st_blocks * 512)
            publish(manager, directory, manifest)
            # Revalidate both sources immediately before the two-step recoverable swap.
            fd = manager._image_fd(directory, manifest)
            os.close(fd)
            os.rename(image, backup)
            sync_directory(directory)
            operation["phase"] = "backup_saved"
            publish(manager, directory, manifest)
            os.rename(candidate, image)
            sync_directory(directory)
            operation["phase"] = "swapped"
            publish(manager, directory, manifest)
            manifest.update(image_inode=info.st_ino, image_device=info.st_dev,
                            current_image_size_bytes=after["size_bytes"], geometry=after,
                            generation=manifest.get("generation", 0) + 1, state="maintenance", last_error=None)
            from .workspace_reclaim import reclaim_sample
            manifest["reclaim_reference"] = reclaim_sample(info.st_blocks * 512, after,
                                                          manifest["generation"], manifest.get("usage_generation", 0))
            manifest["reclaim_sample"] = copy.deepcopy(manifest["reclaim_reference"])
            operation.update(phase="committed", state="completed", completed_at=time.time(),
                             seconds=time.monotonic() - started,
                             reclaimed_bytes=max(0, original.st_blocks * 512 - info.st_blocks * 512))
            publish(manager, directory, manifest)
            # Leave the recovery image until explicit cleanup/recovery finalization.
            manager._unlock_completed_lease(lease)
            return copy.deepcopy(operation)
        except Exception as error:
            if manifest is not None and manifest.get("maintenance", {}).get("id") == operation_id and manifest["state"] == "maintenance" and manifest["maintenance"].get("phase") != "committed":
                manifest.update(state="needs_recovery", last_error=str(error)[-4000:])
                manifest["maintenance"].update(state="failed", error=str(error)[-4000:])
                publish(manager, directory, manifest)
            raise
        finally:
            os.close(lease)


def recover(manager, workspace_id, user_id):
    """Commit cleanup or restore the old image; never adopt an unfinished candidate."""
    with manager._guard(workspace_id):
        directory = manager._directory(workspace_id, False)
        lease = manager._lease(directory)
        try:
            manifest = manager._read(directory, workspace_id, user_id)
            operation = manifest.get("maintenance")
            if not operation or manager._mount_at(directory) is not None or workspace_id in manager.handles:
                raise WorkspaceError("recovery requires an offline maintenance record")
            image, backup, candidate = (directory / name for name in ("workspace.img", "maintenance-backup.img", "maintenance-candidate.img"))
            committed = operation.get("phase") == "committed" and operation.get("state") == "completed"
            timeout = 300
            source = image if committed or not os.path.lexists(backup) else backup
            device = manifest["image_device"] if committed else operation["original_device"]
            inode = manifest["image_inode"] if committed else operation["original_inode"]
            fd, info = checked_file(source, device, inode)
            os.close(fd)
            manager._run("e2fsck", ["-fn", source], timeout, lease_fd=lease)
            expected = manifest.get("current_image_size_bytes", manifest["config"]["image_size_bytes"]) if committed else operation["before"]["size_bytes"]
            if info.st_size != expected or geometry(manager, source, lease, timeout)["size_bytes"] != expected:
                raise WorkspaceError("recovery source geometry mismatch")
            if not committed and operation.get("original_sha256"):
                actual = manager._run("sha256sum", [source], timeout, lease_fd=lease).stdout.split()[0]
                if actual != operation["original_sha256"]:
                    raise WorkspaceError("recovery source checksum mismatch")
            if source == backup:
                os.replace(backup, image)
                sync_directory(directory)
            if not committed:
                manifest.update(image_device=info.st_dev, image_inode=info.st_ino, state="maintenance",
                                current_image_size_bytes=expected, geometry=operation["before"], last_error=None)
                operation.update(state="failed", phase="rolled_back", recovered=True)
                publish(manager, directory, manifest)
            for path in (backup, candidate):
                if os.path.lexists(path):
                    fd, _ = checked_file(path, operation["original_device"] if path == backup else operation.get("candidate_device"),
                                         operation["original_inode"] if path == backup else operation.get("candidate_inode"))
                    os.close(fd)
                    os.unlink(path)
            sync_directory(directory)
            operation["recovery_files_removed"] = True
            manifest.update(state="retained", last_error=None)
            publish(manager, directory, manifest)
            manager._unlock_completed_lease(lease)
            return copy.deepcopy(operation)
        finally:
            os.close(lease)
