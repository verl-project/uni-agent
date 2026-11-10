# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace
import unittest

from pycape import CAPEController
from tests.pycape import test_workspace_reclaim_scheduling as scheduling


class WorkspaceAutoReclaimTests(unittest.TestCase):
    setUp = scheduling.WorkspaceReclaimSchedulingTests.setUp
    workspace = scheduling.WorkspaceReclaimSchedulingTests.workspace

    def set_policy(self, value, identity="w1"):
        return self.controller.workspace_operation(identity, "set", auto_reclaim=value)

    def test_active_command_can_set_without_release_or_operation(self):
        record = self.workspace()
        record["state"] = "mounted"
        session = SimpleNamespace(active_request_id="running")
        self.controller.sessions["w1"] = session
        result = self.set_policy("off")
        self.assertEqual(result["state"], "mounted")
        self.assertEqual(session.active_request_id, "running")
        self.assertEqual(result["auto_reclaim"], "off")
        self.assertFalse(result["auto_reclaim_enabled"])
        self.assertNotIn("operation", result)
        with self.assertRaises(ValueError):
            self.controller.workspace_operation("w1", "reclaim")

    def test_set_before_grant_clears_candidate_and_survives_replay(self):
        self.workspace()
        self.now += 11
        self.assertIsNotNone(self.controller.workspaces["w1"]["reclaim_candidate"])
        self.set_policy("off")
        for _ in range(2):
            self.assertEqual(self.controller.workspace_operations("a", True), [])
        restored = CAPEController.reconcile_from_journal(self.controller.journal, clock=lambda: self.now)
        restored.nodes = copy.deepcopy(self.controller.nodes)
        restored.workspace_configs = copy.deepcopy(self.controller.workspace_configs)
        self.assertEqual(restored.workspace_status("w1")["auto_reclaim"], "off")
        self.assertEqual(restored.workspace_operations("a", True), [])

    def test_grant_before_set_is_preserved_and_redelivered(self):
        self.workspace()
        self.now += 11
        operation = self.controller.workspace_operations("a", True)[0]["operation"]
        result = self.set_policy("off")
        self.assertEqual(result["operation"], operation)
        self.assertEqual(self.controller.workspace_operations("a", True)[0]["operation"], operation)
        restored = CAPEController.reconcile_from_journal(self.controller.journal, clock=lambda: self.now)
        restored.nodes = copy.deepcopy(self.controller.nodes)
        self.assertEqual(restored.workspace_operations("a", True)[0]["operation"], operation)
        self.assertEqual(restored.workspaces["w1"]["auto_reclaim"], "off")

    def test_inherit_rechecks_current_config_and_backoff(self):
        record = self.workspace()
        self.assertTrue(self.controller.workspace_status("w1")["auto_reclaim_enabled"])
        record["reclaim_retry_after"] = self.now + 100
        self.set_policy("off")
        self.set_policy("inherit")
        self.now += 11
        self.assertEqual(self.controller.workspace_operations("a", True), [])
        self.now += 100
        self.assertEqual(len(self.controller.workspace_operations("a", True)), 1)

    def test_policy_is_independent_per_workspace_and_explicit_reclaim_still_works(self):
        self.workspace()
        self.workspace("w2", "b")
        self.set_policy("off")
        self.assertEqual(self.controller.workspace_status("w2")["auto_reclaim"], "inherit")
        result = self.controller.workspace_operation("w1", "reclaim")
        self.assertEqual(result["operation"]["source"], "explicit")
        self.assertEqual(result["auto_reclaim"], "off")
        operation = copy.deepcopy(result["operation"])
        self.assertEqual(self.set_policy("off")["operation"], operation)
        self.assertEqual(len(self.controller.workspace_operations("a", True)), 1)

    def test_agent_cannot_overwrite_policy(self):
        record = self.workspace()
        record["revision"] = 0
        self.set_policy("off")
        update = dict(record, workspace_id="w1", revision=1, auto_reclaim="inherit")
        self.controller.apply_workspace_updates("a", [update], agent_epoch="epoch")
        self.assertEqual(self.controller.workspaces["w1"]["auto_reclaim"], "off")
        self.now += 11
        self.assertEqual(self.controller.workspace_operations("a", True), [])

    def test_invalid_policy_and_maintenance_parameters_leave_state_unchanged(self):
        self.workspace()
        before = copy.deepcopy(self.controller.workspaces)
        for value in (None, "on", "", True, {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.set_policy(value)
        for kwargs in ({"target_bytes": 1024}, {"operation_id": "id"}):
            with self.assertRaises(ValueError):
                self.controller.workspace_operation("w1", "set", auto_reclaim="off", **kwargs)
        with self.assertRaises(KeyError):
            self.set_policy("off", "missing")
        self.assertEqual(before, self.controller.workspaces)

    def test_inherit_tracks_config_without_persisting_derived_status(self):
        self.workspace()
        self.set_policy("inherit")
        self.assertNotIn("auto_reclaim_enabled", self.controller.workspaces["w1"])
        self.controller.workspace_configs.update("pool", {"maintenance": {"reclaim_on_release": False}})
        self.assertFalse(self.controller.workspace_status("w1")["auto_reclaim_enabled"])


if __name__ == "__main__":
    unittest.main()
