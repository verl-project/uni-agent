# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest
import os
import shutil
import tempfile

from pycape.workspace_policy import MaintenancePolicy
from pycape.workspace_reclaim import reclaim_candidate, reclaim_sample, reclaim_signal


class WorkspaceReclaimPolicyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("nice") and shutil.which("timeout"), "requires coreutils")
    def test_helper_priority_does_not_change_parent_priority(self):
        from pycape.workspace import WorkspaceConfig
        from pycape.workspace_fs import WorkspaceManager
        with tempfile.TemporaryDirectory() as root:
            manager = WorkspaceManager(WorkspaceConfig(image_root=root), "node", "pool")
            before = os.getpriority(os.PRIO_PROCESS, 0)
            manager.maintenance_nice = 10
            child = int(manager._run("nice", [], 5).stdout.strip())
            self.assertEqual(child, min(19, before + 10))
            self.assertEqual(os.getpriority(os.PRIO_PROCESS, 0), before)

    def setUp(self):
        self.unit = 1024 ** 2
        base = {"allocated_bytes": 260 * self.unit, "used_bytes": 260 * self.unit,
                "logical_bytes": 32 * 1024 ** 3, "image_generation": 0, "usage_generation": 0}
        self.record = {"state": "retained", "generation": 0, "usage_generation": 1,
                       "config": {"image_size_bytes": base["logical_bytes"]},
                       "reclaim_reference": base, "reclaim_sample": dict(base, usage_generation=1),
                       "reclaim_release_observed_at": 1000.0}
        self.policy = MaintenancePolicy({"reclaim_on_release": True}).to_dict()

    def test_metadata_and_live_data_do_not_look_like_deleted_blocks(self):
        self.assertEqual(reclaim_signal(self.record), (0, None))
        for field in ("allocated_bytes", "used_bytes"):
            self.record["reclaim_sample"][field] += 1024 * self.unit
        self.assertEqual(reclaim_signal(self.record), (0, None))
        self.assertIsNone(reclaim_candidate(self.record, self.policy, 2000)[0])

    def test_write_delete_and_delete_without_allocation_growth(self):
        self.record["reclaim_sample"]["allocated_bytes"] += 1024 * self.unit
        self.assertEqual(reclaim_signal(self.record), (1024 * self.unit, None))
        reference = self.record["reclaim_reference"]
        reference["allocated_bytes"] += 1024 * self.unit
        reference["used_bytes"] += 1024 * self.unit
        self.assertEqual(reclaim_signal(self.record), (1024 * self.unit, None))

    def test_missing_or_mismatched_measurements_defer_without_guessing(self):
        for field, value in (("reclaim_reference", None), ("usage_generation", 2), ("generation", 1),
                             ("generation", None), ("current_image_size_bytes", True), ("current_image_size_bytes", None)):
            with self.subTest(field=field):
                record = copy.deepcopy(self.record)
                record[field] = value
                self.assertIsNone(reclaim_signal(record)[0])
        self.record["reclaim_sample"]["allocated_bytes"] = True
        self.assertIsNone(reclaim_signal(self.record)[0])

    def test_success_dedup_is_per_usage_not_execution_lifetime(self):
        self.record["reclaim_sample"]["allocated_bytes"] += 1024 * self.unit
        self.record["reclaim_success_generation"] = 1
        self.assertIsNone(reclaim_candidate(self.record, self.policy, 2000)[0])
        self.record["usage_generation"] = self.record["reclaim_sample"]["usage_generation"] = 2
        self.assertEqual(reclaim_candidate(self.record, self.policy, 2000)[1], "ready")

    def test_idle_retry_and_low_yield_are_candidate_delays_not_operations(self):
        self.record["reclaim_sample"]["allocated_bytes"] += 1024 * self.unit
        candidate, reason = reclaim_candidate(self.record, self.policy, 1001)
        self.assertEqual(reason, "waiting_idle_or_backoff")
        self.assertEqual(candidate["not_before"], 1060)
        self.record["reclaim_retry_after"] = 3000
        self.record["reclaim_low_yield_until"] = 4000
        self.assertEqual(reclaim_candidate(self.record, self.policy, 2000)[0]["not_before"], 4000)
        self.assertNotIn("operation", self.record)

    def test_sample_geometry_and_policy_validation(self):
        geometry = {"block_size": 4096, "blocks": 8192, "free_blocks": 4096, "size_bytes": 32 * self.unit}
        self.assertEqual(reclaim_sample(20 * self.unit, geometry, 0, 1)["used_bytes"], 16 * self.unit)
        with self.assertRaises(ValueError):
            reclaim_sample(0, dict(geometry, free_blocks=9999), 0, 1)
        for changes in ({"reclaim_nice": 20}, {"reclaim_max_candidates": 0}):
            with self.assertRaises(ValueError):
                MaintenancePolicy(changes)


if __name__ == "__main__":
    unittest.main()
