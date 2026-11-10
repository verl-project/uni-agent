# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from pycape.workspace import WorkspaceConfig
from pycape.workspace_fs import WorkspaceError, WorkspaceManager, mount_entries, new_workspace_id


@contextmanager
def disposable_live_root():
    root = tempfile.mkdtemp(prefix="cape-task16-python-manager-")
    try:
        yield root
    except BaseException:
        print("retained_workspace_test_root=" + root, flush=True)
        raise
    else:
        if any(entry["target"].startswith(root + "/") for entry in mount_entries()):
            raise AssertionError("refusing to remove mounted test root: " + root)
        shutil.rmtree(root)


class WorkspaceManagerSafetyTests(unittest.TestCase):
    def test_completed_lease_releases_inherited_copies_but_not_the_next_owner(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            first = manager._lease(Path(root))
            inherited = os.dup(first)
            manager._unlock_completed_lease(first)
            successor = manager._lease(Path(root))
            try:
                os.close(first)
                os.close(inherited)
                with self.assertRaises(WorkspaceError):
                    manager._lease(Path(root))
            finally:
                os.close(successor)

    def test_formatter_lease_survives_owner_crash(self):
        with tempfile.TemporaryDirectory() as root:
            helper = Path(root) / "formatter"
            ready = Path(root) / "ready"
            helper.write_text('#!/bin/sh\nprintf ready > "$CAPE_TEST_READY_FILE"\nexec /bin/sleep 1\n')
            helper.chmod(0o700)
            environment = dict(os.environ, CAPE_WORKSPACE_MKFS_BIN=str(helper), CAPE_TEST_READY_FILE=str(ready))
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            code = "from pathlib import Path; import sys; from pycape.workspace import WorkspaceConfig; from pycape.workspace_fs import WorkspaceManager; m=WorkspaceManager(WorkspaceConfig(image_root=sys.argv[1]), 'node','pool'); fd=m._lease(Path(sys.argv[1])); m._run('mkfs.ext4',[],5,lease_fd=fd)"
            child = subprocess.Popen([sys.executable, "-c", code, root], env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                child.kill(); child.wait(timeout=5)
                with self.assertRaisesRegex(WorkspaceError, "already has a writer"):
                    self.manager(root)._lease(Path(root))
                deadline = time.monotonic() + 5
                while True:
                    try:
                        lease = self.manager(root)._lease(Path(root))
                        os.close(lease)
                        break
                    except WorkspaceError:
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(0.01)
            finally:
                if child.poll() is None:
                    child.kill(); child.wait(timeout=5)

    def test_duplicate_observation_is_not_a_second_mount_but_stacked_ids_are_rejected(self):
        entry = {"mount_id": 42, "target": "/tmp/example/mnt", "source": "/tmp/example/workspace.img", "fstype": "fuse.ext4"}
        with mock.patch("pycape.workspace_fs.mount_entries", return_value=[entry, dict(entry)]):
            self.assertEqual(WorkspaceManager._mount_at(Path("/tmp/example")), entry)
        with mock.patch("pycape.workspace_fs.mount_entries", return_value=[entry, dict(entry, mount_id=43)]):
            with self.assertRaisesRegex(WorkspaceError, "stacked"):
                WorkspaceManager._mount_at(Path("/tmp/example"))

    def test_interrupted_delete_finishes_without_recreating_image(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            identity = new_workspace_id()
            directory = manager._directory(identity, True)
            manager._write(directory, {"schema_version": 3, "workspace_id": identity, "user_id": "user", "node_id": "node",
                "pool": "pool", "execution_session_id": "session", "config": manager.config.to_dict(), "state": "deleting",
                "revision": 1, "image_device": 1, "image_inode": 1, "mount_id": None, "daemon_pid": None, "last_error": None})
            self.assertEqual(manager.delete(identity, "user")["state"], "deleted")
            self.assertEqual(manager.delete(identity, "user")["state"], "deleted")
            self.assertFalse((directory / "workspace.img").exists())
            self.assertFalse(manager._is_image_mount(directory, {"fstype": "fuse.ext4", "source": "/other/image.img"}))

    def manager(self, root):
        return WorkspaceManager(WorkspaceConfig(image_root=root, image_size_bytes=64 * 1024 ** 2), "node", "pool")

    def test_rejects_paths_and_ids_outside_owned_private_namespace(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            for value in ("../escape", "", "ws-../other", "/tmp/workspace"):
                with self.assertRaises(WorkspaceError):
                    manager.prepare(value, "user", "session", create=True)
            parent = Path(root) / ("cape-workspaces-" + str(os.geteuid()))
            parent.symlink_to(root, target_is_directory=True)
            with self.assertRaises(WorkspaceError):
                manager.prepare(new_workspace_id(), "user", "session", create=True)

    def test_writer_lock_excludes_other_managers(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            identity = new_workspace_id()
            directory = manager._directory(identity, True)
            lease = manager._lease(directory)
            try:
                with self.assertRaisesRegex(WorkspaceError, "already has a writer"):
                    self.manager(root).prepare(identity, "user", "session", create=True)
                with self.assertRaisesRegex(WorkspaceError, "already has a writer"):
                    self.manager(root).reconcile_released(identity, "user")
            finally:
                os.close(lease)

    def test_orphan_image_is_never_truncated_or_formatted(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            identity = new_workspace_id()
            directory = manager._directory(identity, True)
            image = directory / "workspace.img"
            image.write_bytes(b"preserve-orphan-data")
            with mock.patch.object(manager, "_run") as run:
                with self.assertRaises(FileExistsError):
                    manager.prepare(identity, "user", "session", create=True)
            run.assert_not_called()
            self.assertEqual(image.read_bytes(), b"preserve-orphan-data")
            self.assertFalse((directory / "manifest.json").exists())

    def test_failed_format_retains_image_and_diagnostic_without_retrying_format(self):
        with tempfile.TemporaryDirectory() as root:
            manager = self.manager(root)
            identity = new_workspace_id()
            with mock.patch.object(manager, "_run", side_effect=WorkspaceError("formatter failed")) as run:
                with self.assertRaisesRegex(WorkspaceError, "formatter failed"):
                    manager.prepare(identity, "user", "session", create=True)
            self.assertEqual(run.call_args.args[0], "mkfs.ext4")
            status = manager.inspect(identity, "user")
            self.assertEqual(status["state"], "needs_check")
            self.assertIn("formatter failed", status["last_error"])
            with self.assertRaises(WorkspaceError):
                manager.inspect(identity, "other-user")
            with mock.patch.object(manager, "_run", side_effect=WorkspaceError("offline check refused")) as run:
                with self.assertRaisesRegex(WorkspaceError, "offline check refused"):
                    manager.prepare(identity, "user", "session", create=True)
            self.assertEqual(run.call_args.args[0], "e2fsck")

    @unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_LIVE_TEST") == "1", "requires verified fuse2fs node")
    def test_real_mount_reuse_release_and_new_manager_remount(self):
        with disposable_live_root() as root:
            manager = self.manager(root)
            identity = new_workspace_id()
            try:
                handle = manager.prepare(identity, "user", "session", create=True)
                mount = Path(handle.path)
                (mount / "data").write_bytes(b"workspace-data")
                self.assertIs(manager.prepare(identity, "user", "session", create=True), handle)
                for operation in (manager.check, manager.delete):
                    with self.assertRaisesRegex(WorkspaceError, "already has a writer"):
                        operation(identity, "user")
                with self.assertRaisesRegex(WorkspaceError, "already has a writer"):
                    self.manager(root).prepare(identity, "user", "other-session")
                with self.assertRaises(WorkspaceError):
                    manager.prepare(identity, "other-user", "session")
                manager.release(identity)
                self.assertEqual(manager.inspect(identity, "user")["state"], "retained")
                self.assertEqual(manager.check(identity, "user")["exit_code"], 0)
                other = self.manager(root)
                try:
                    with self.assertRaisesRegex(WorkspaceError, "explicit reuse"):
                        other.prepare(identity, "user", "other-session", create=True)
                    self.assertEqual(other.inspect(identity, "user")["state"], "retained")
                    reopened = other.prepare(identity, "user", "other-session")
                    self.assertEqual((Path(reopened.path) / "data").read_bytes(), b"workspace-data")
                finally:
                    other.release(identity)
                self.assertEqual(other.delete(identity, "user")["state"], "deleted")
                self.assertEqual(other.delete(identity, "user")["state"], "deleted")
                with self.assertRaisesRegex(WorkspaceError, "was deleted"):
                    other.prepare(identity, "user", "third-session")
            finally:
                manager.release(identity)
