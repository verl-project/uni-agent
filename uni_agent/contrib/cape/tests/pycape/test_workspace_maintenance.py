# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid

from pycape.workspace import WorkspaceConfig, WorkspaceConfigStore
from pycape.workspace_fs import WorkspaceError, WorkspaceManager, new_workspace_id
from pycape.workspace_maintenance import MaintenancePolicy
from pycape.workspace_policy import automatic_grow_target
from pycape.agent import NodeAgentClient
from pycape import CAPEController, FakeRuntimeAdapter, NodeResources, RuntimeCapabilities, WorkloadSpec, ResourceRequest
from tests.workspace_fixtures import install_embedded_workspace_double


class MaintenancePolicyTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        self.assertFalse(MaintenancePolicy().to_dict()["auto_grow"])
        for values in ({"auto_grow": 1}, {"unknown": 1}, {"grow_target_percent": 90},
                       {"timeout_seconds": 0}, {"grow_min_bytes": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                MaintenancePolicy(values)

    def test_nested_policy_updates_and_node_precedence(self):
        store = WorkspaceConfigStore()
        store.update("pool", {"maintenance": {"auto_grow": True, "grow_threshold_percent": 85}})
        store.update("pool", {"maintenance": {"grow_cooldown_seconds": 7}})
        store.update("pool", {"maintenance": {"reclaim_on_release": True}}, "node")
        policy = store.get("pool", "node").maintenance
        self.assertTrue(policy["auto_grow"])
        self.assertTrue(policy["reclaim_on_release"])
        self.assertEqual(policy["grow_threshold_percent"], 85)
        self.assertEqual(policy["grow_cooldown_seconds"], 7)
        with self.assertRaises(ValueError):
            store.update("pool", {"maintenance": None})

    def test_inode_growth_crosses_group_and_respects_limit(self):
        config = WorkspaceConfig(image_size_bytes=64*1024**2,
                                 maintenance={"auto_grow": True, "grow_min_bytes": 32*1024**2})
        record = {"state": "mounted", "config": config.to_dict(),
                  "space": {"internal": {"total_bytes": 64*1024**2, "available_bytes": 50*1024**2,
                                          "total_inodes": 128, "free_inodes": 0}},
                  "geometry": {"block_size": 4096, "blocks_per_group": 32768}}
        result = automatic_grow_target(record, config, 100)
        self.assertEqual(result["reason"], "inodes")
        self.assertEqual(result["target_bytes"], 256*1024**2)
        record["last_maintenance_at"] = 99
        self.assertIsNone(automatic_grow_target(record, config, 100))
        record.pop("last_maintenance_at")
        config.max_image_size_bytes = 128*1024**2
        with self.assertRaisesRegex(ValueError, "exceeds"):
            automatic_grow_target(record, config, 100)

    def test_old_ack_cannot_drop_new_same_revision_telemetry(self):
        agent = object.__new__(NodeAgentClient)
        agent._state_lock = threading.RLock()
        agent.workspace_updates = {"ws": {"execution_session_id": "session", "revision": 8, "telemetry_id": "new"}}
        agent.workspace_acknowledged = {}
        agent._save_state = lambda: None
        response = {"workspace_acks": {"ws": {"execution_session_id": "session", "revision": 8, "telemetry_id": "old"}}}
        agent._workspace_response(response)
        self.assertIn("ws", agent.workspace_updates)
        response["workspace_acks"]["ws"]["telemetry_id"] = "new"
        agent._workspace_response(response)
        self.assertNotIn("ws", agent.workspace_updates)

    def test_completed_operation_receipt_fences_delayed_delivery(self):
        agent = object.__new__(NodeAgentClient)
        agent._state_lock = threading.RLock()
        agent.workspace_updates = {}
        agent.workspace_acknowledged = {}
        agent._save_state = lambda: None
        agent.workspace_operation_receipts = {"ws": {"sequence": 2, "id": "latest", "update": {"revision": 10}}}
        agent._workspace_response({"workspace_operations": [{"id": "ws", "operation": {"id": "old", "sequence": 1}}]})
        self.assertFalse(agent.workspace_updates)
        agent._workspace_response({"workspace_operations": [{"id": "ws", "operation": {"id": "latest", "sequence": 2}}]})
        self.assertEqual(agent.workspace_updates["ws"], {"revision": 10})

    def test_failed_boundary_fails_only_waiting_request_and_preserves_session(self):
        install_embedded_workspace_double(self)
        controller = CAPEController(runtime_adapters={"fake": FakeRuntimeAdapter()})
        controller.register_node("node", "pool", NodeResources(cpu_cores=4, memory_gb=8), RuntimeCapabilities(adapter="fake"))
        def spec():
            return WorkloadSpec(pool="pool", user_id="user", session_id="task", command=["echo", "hello"],
                                resources=ResourceRequest(gpus=0))
        first = controller.submit_workload(spec())
        controller.schedule_once()
        self.assertEqual(first.state.value, "COMPLETED")
        session = controller.sessions[first.execution_session_id]
        record = controller.workspaces[session.workspace["id"]]
        record["operation"] = {"id": "failed", "boundary": True, "state": "failed", "error": "injected"}
        second = controller.submit_workload(spec())
        controller.schedule_once()
        self.assertEqual(second.reason, "workspace_maintenance_failed")
        self.assertEqual(second.failure_kind, "infrastructure")
        self.assertIsNone(second.started_at)
        self.assertEqual(first.state.value, "COMPLETED")
        self.assertEqual(session.state.value, "CACHED_IDLE")


@unittest.skipUnless(all(shutil.which(tool) for tool in ("mkfs.ext4", "e2fsck", "resize2fs", "dumpe2fs", "cp", "sha256sum", "timeout")),
                     "requires offline ext4 utilities, not FUSE")
class OfflineMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cape-maintenance-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.manager = WorkspaceManager(WorkspaceConfig(image_root=self.temporary.name,
                                        image_size_bytes=64*1024**2), "node", "pool")
        self.identity = new_workspace_id()
        self.directory = self.manager._directory(self.identity, True)
        self.image = self.directory / "workspace.img"
        fd = os.open(self.image, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(fd, 64*1024**2)
        info = os.fstat(fd)
        os.close(fd)
        self.manager._run("mkfs.ext4", ["-q", "-F", self.image], 30)
        self.manifest = dict(schema_version=3, workspace_id=self.identity, user_id="user", pool="pool",
                             node_id="node", execution_session_id="session", config=self.manager.config.to_dict(),
                             state="retained", last_error=None, image_device=info.st_dev,
                             image_inode=info.st_ino, mount_id=None, daemon_pid=None)
        self.manager._write(self.directory, self.manifest)

    def operate(self, action, target=None, operation_id=None):
        return self.manager.maintain(self.identity, "user", action, target, operation_id or uuid.uuid4().hex,
                                     {"backing_reserve_bytes": 0})

    def test_grow_reclaim_shrink_and_creation_config(self):
        for action, target in (("grow", 128*1024**2), ("reclaim", None), ("shrink", 64*1024**2)):
            result = self.operate(action, target)
            self.assertEqual(result["state"], "completed")
            current = json.loads((self.directory / "manifest.json").read_text())
            self.assertEqual(current["config"], self.manifest["config"])
            self.assertEqual(current["schema_version"], 3)
            self.assertEqual(self.image.stat().st_size, current["current_image_size_bytes"])
            self.assertEqual(current["reclaim_reference"]["image_generation"], current["generation"])
            self.assertEqual(current["reclaim_sample"], current["reclaim_reference"])
            self.assertTrue((self.directory / "maintenance-backup.img").exists())
            self.manager.recover_maintenance(self.identity, "user")
            self.assertFalse((self.directory / "maintenance-backup.img").exists())

    def test_older_optional_policy_fields_do_not_change_image_identity(self):
        config = dict(self.manifest["config"], maintenance={"reclaim_on_release": True})
        self.manifest["config"] = config
        self.manager._write(self.directory, self.manifest)
        reader = WorkspaceManager(WorkspaceConfig.from_dict(config), "node", "pool")
        before = self.image.stat()
        observed = reader.inspect(self.identity, "user")
        self.assertNotIn("reclaim_reference", observed)
        self.assertEqual(self.image.stat().st_ino, before.st_ino)
        self.assertEqual(self.image.stat().st_size, before.st_size)

    def test_capability_fixture_refuses_user_image_root(self):
        from pycape.workspace_maintenance import probe_disposable_tools
        original_inode = self.image.stat().st_ino
        with self.assertRaisesRegex(WorkspaceError, "isolated probe root"):
            probe_disposable_tools(self.manager, self.identity)
        self.assertEqual(self.image.stat().st_ino, original_inode)
        self.assertEqual(self.image.stat().st_size, 64*1024**2)

    def test_failure_fences_reuse_and_restores_original(self):
        original_inode = self.image.stat().st_ino
        real = self.manager._run
        def fail(name, *args, **kwargs):
            if name == "resize2fs":
                raise WorkspaceError("injected resizer failure")
            return real(name, *args, **kwargs)
        with patch.object(self.manager, "_run", side_effect=fail), self.assertRaisesRegex(WorkspaceError, "injected"):
            self.operate("grow", 128*1024**2)
        self.assertEqual(self.image.stat().st_ino, original_inode)
        with self.assertRaisesRegex(WorkspaceError, "recovery"):
            self.manager.prepare(self.identity, "user", "next")
        self.manager.recover_maintenance(self.identity, "user")
        self.assertEqual(self.manager.inspect(self.identity, "user")["state"], "retained")
        self.assertEqual(self.image.stat().st_size, 64*1024**2)

    def test_idempotence_and_conflicting_id(self):
        op = uuid.uuid4().hex
        first = self.operate("grow", 128*1024**2, op)
        self.assertEqual(first, self.operate("grow", 128*1024**2, op))
        with self.assertRaises(WorkspaceError):
            self.operate("grow", 256*1024**2, op)

    def test_rejects_wrong_direction_and_unsafe_candidate(self):
        with self.assertRaises(WorkspaceError):
            self.operate("grow", 32*1024**2)
        os.symlink(self.image, self.directory / "maintenance-candidate.img")
        with self.assertRaisesRegex(WorkspaceError, "recovery files"):
            self.operate("grow", 128*1024**2)

    def test_interrupted_swap_restores_verified_original(self):
        original_inode = self.image.stat().st_ino
        rename = os.rename
        def interrupted(source, target):
            if Path(source).name == "maintenance-candidate.img":
                raise OSError("injected crash window after backup rename")
            return rename(source, target)
        with patch("pycape.workspace_maintenance.os.rename", side_effect=interrupted), self.assertRaisesRegex(OSError, "crash window"):
            self.operate("grow", 128*1024**2)
        self.assertFalse(self.image.exists())
        self.assertTrue((self.directory / "maintenance-backup.img").exists())
        result = self.manager.recover_maintenance(self.identity, "user")
        self.assertTrue(result["recovered"])
        self.assertEqual(self.image.stat().st_ino, original_inode)
        self.assertEqual(self.image.stat().st_size, 64*1024**2)

    def test_committed_metadata_recovers_without_rolling_back_capacity(self):
        real_write = self.manager._write
        interrupted = []
        def write(directory, manifest):
            real_write(directory, manifest)
            if manifest.get("maintenance", {}).get("phase") == "committed" and not interrupted:
                interrupted.append(True)
                raise OSError("injected lost acknowledgement after durable commit")
        with patch.object(self.manager, "_write", side_effect=write), self.assertRaisesRegex(OSError, "acknowledgement"):
            self.operate("grow", 128*1024**2)
        result = self.manager.recover_maintenance(self.identity, "user")
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.image.stat().st_size, 128*1024**2)
        self.assertEqual(json.loads((self.directory / "manifest.json").read_text())["generation"], 1)
