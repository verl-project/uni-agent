# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from pycape.workspace import WorkspaceConfig, WorkspaceConfigStore, workspace_bind_conflict
from pycape import CAPEController, MemoryJournal, Role, TokenAuthority


class WorkspaceConfigTests(unittest.TestCase):
    def test_workspace_bind_conflicts_include_parent_and_lexical_aliases(self):
        for target in ("/", "/workspace", "/workspace/", "/workspace/./", "/workspace/../workspace"):
            self.assertTrue(workspace_bind_conflict(target, "/workspace"), target)
        self.assertTrue(workspace_bind_conflict("/work", "/work/session"))
        self.assertFalse(workspace_bind_conflict("/workspace/external", "/workspace"))
        self.assertFalse(workspace_bind_conflict("/workspaces", "/workspace"))

    def test_controller_configuration_replays_and_requires_pool_admin(self):
        authority = TokenAuthority()
        authority.add_user_token("admin", "admin", "pool", (Role.ADMIN,))
        authority.add_user_token("reader", "reader", "pool", (Role.VIEWER,))
        authority.add_user_token("other", "other", "other", (Role.ADMIN,))
        journal = MemoryJournal()
        controller = CAPEController(auth=authority, journal=journal)
        for token in (None, "reader", "other"):
            with self.assertRaises(PermissionError):
                controller.get_workspace_config("pool", token=token)
            with self.assertRaises(PermissionError):
                controller.set_workspace_config("pool", {"image_root": "/tmp/new"}, token=token)
        result = controller.set_workspace_config("pool", {"image_root": "/tmp/new"}, token="admin")
        self.assertEqual(result["desired_config"]["version"], 1)
        restored = CAPEController.reconcile_from_journal(journal, auth=authority)
        self.assertEqual(restored.get_workspace_config("pool", token="admin"), result)

    def test_pool_node_precedence_replay_and_atomic_validation(self):
        store = WorkspaceConfigStore()
        self.assertEqual(store.get("pool", "node").version, 0)
        store.update("pool", {"image_root": "/scratch/pool"})
        snapshot = store.update("pool", {"image_root": "/scratch/node", "image_size_bytes": 20 * 1024 ** 3}, "node")
        self.assertEqual(store.get("pool", "node").image_root, "/scratch/node")
        self.assertEqual(store.get("pool", "other").image_root, "/scratch/pool")
        self.assertEqual(store.get("pool", "node").version, 2)
        with self.assertRaises(ValueError):
            store.update("pool", {"max_image_size_bytes": 15 * 1024 ** 3})
        self.assertEqual(store.record("pool"), snapshot)
        restored = WorkspaceConfigStore()
        restored.restore("pool", snapshot)
        self.assertEqual(restored.get("pool", "node").to_dict(), store.get("pool", "node").to_dict())
        snapshot["nodes"]["node"]["image_root"] = "/changed"
        self.assertEqual(store.get("pool", "node").image_root, "/scratch/node")
        self.assertEqual(restored.get("pool", "node").image_root, "/scratch/node")

    def test_registry_rejects_overriding_version_or_unknown_fields(self):
        store = WorkspaceConfigStore()
        for change in ({"version": 10}, {"bad": 1}, {}):
            with self.assertRaises(ValueError):
                store.update("pool", change)
        with self.assertRaises(ValueError):
            store.update("pool", {"image_root": "/tmp"}, "")
        with self.assertRaises(ValueError):
            store.restore("pool", {"version": 1, "defaults": {}, "nodes": {"n": {"version": 5}}})

    def test_bootstrap_defaults_are_explicit_and_roundtrip(self):
        config = WorkspaceConfig()
        self.assertEqual(config.image_root, "/tmp")
        self.assertEqual(config.image_size_bytes, 10 * 1024 ** 3)
        self.assertEqual(config.version, 0)
        self.assertEqual(WorkspaceConfig.from_dict(config.to_dict()).to_dict(), config.to_dict())

    def test_update_is_validated_versioned_and_does_not_mutate_prior_snapshot(self):
        original = WorkspaceConfig()
        updated = original.updated({"image_root": "/scratch//work/./", "image_size_bytes": 20 * 1024 ** 3})
        self.assertEqual(updated.image_root, "/scratch/work")
        self.assertEqual(updated.version, 1)
        self.assertEqual(original.image_root, "/tmp")
        self.assertEqual(original.image_size_bytes, 10 * 1024 ** 3)
        self.assertEqual(updated.bytes_per_inode, original.bytes_per_inode)

    def test_invalid_sizes_types_limits_and_unknown_fields(self):
        for changes in ({"image_size_bytes": True}, {"image_size_bytes": 1},
                        {"image_size_bytes": 2 ** 63}, {"image_size_bytes": 20 * 1024 ** 4},
                        {"image_size_bytes": 1.5}, {"bytes_per_inode": 0},
                        {"mount_timeout_seconds": 301}, {"unknown": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                WorkspaceConfig.from_dict(changes)
        for changes in ({}, {"version": 2}, None):
            with self.assertRaises(ValueError):
                WorkspaceConfig().updated(changes)

    def test_paths_are_node_side_lexical_and_reject_unsafe_roots(self):
        for path in ("relative", "", "/", "/tmp/../etc", "/sys", "/proc/self", "/dev", "/run", "/tmp/bad\npath", "/tmp/bad\x7fpath"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                WorkspaceConfig(image_root=path)
        self.assertEqual(WorkspaceConfig(image_root="//tmp//cape").image_root, "/tmp/cape")
        self.assertEqual(WorkspaceConfig(image_root="/dev/shm/cape").image_root, "/dev/shm/cape")
        self.assertEqual(WorkspaceConfig(image_root="/not-present-on-submitter/work space").image_root,
                         "/not-present-on-submitter/work space")
