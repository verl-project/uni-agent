# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression for metadata-only images admitted to automatic reclaim.

The empty-image allocation is the issue reporter's measurement, not a filesystem
measurement performed by this fixture. No maintenance subprocess is executed.
"""

from types import SimpleNamespace
import copy
import unittest

from pycape import CAPEController


class WorkspaceReclaimAdmissionTests(unittest.TestCase):
    EMPTY_ALLOCATED_BYTES = 273055744
    IMAGE_BYTES = 32 * 1024 ** 3

    def setUp(self):
        self.now = 1000.0
        self.controller = CAPEController(clock=lambda: self.now)
        self.configure(True)

    def configure(self, enabled):
        self.controller.workspace_configs.update("pool", {
            "image_size_bytes": self.IMAGE_BYTES,
            "maintenance": {"reclaim_on_release": enabled},
        })
        self.config = self.controller.workspace_configs.get("pool", "node")
        self.node = SimpleNamespace(workspace_status={
            "maintenance_protocol": 1, "state": "applied",
            "applied_version": self.config.version,
        })

    def new_workspace(self, index=0, allocated=None):
        record = {
            "id": "workspace-{}".format(index), "execution_session_id": "session-{}".format(index),
            "user_id": "alice", "pool": "pool", "node_id": "node",
            "state": "mounted", "config": self.config.to_dict(),
            "space": {"allocated_bytes": self.EMPTY_ALLOCATED_BYTES if allocated is None else allocated},
            "usage_generation": 1, "generation": 0,
        }
        record["reclaim_reference"] = {"allocated_bytes": record["space"]["allocated_bytes"],
            "used_bytes": self.EMPTY_ALLOCATED_BYTES, "logical_bytes": self.IMAGE_BYTES,
            "image_generation": 0, "usage_generation": 0}
        self.controller.workspaces[record["id"]] = record
        self.controller._automatic_workspace_operation(record, self.node)
        self.assertNotIn("operation", record, "auto-grow is disabled; mounted image must not reclaim")
        return record

    def release(self, record):
        record["state"] = "retained"
        record["reclaim_sample"] = dict(record["reclaim_reference"],
            allocated_bytes=record["space"]["allocated_bytes"], usage_generation=record["usage_generation"])
        self.controller._automatic_workspace_operation(record, self.node)

    def test_empty_metadata_above_threshold_does_not_trigger_reclaim(self):
        self.assertEqual(self.config.maintenance["reclaim_min_allocated_bytes"], 64 * 1024 ** 2)
        self.assertGreater(self.EMPTY_ALLOCATED_BYTES, self.config.maintenance["reclaim_min_allocated_bytes"])
        record = self.new_workspace()
        self.release(record)  # No writes or allocation growth between creation and release.
        self.assertNotIn("operation", record,
                         "unchanged empty-image metadata must not enqueue automatic reclaim")

    def test_fresh_empty_workspaces_do_not_flood_maintenance(self):
        records = []
        for index in range(8):
            record = self.new_workspace(index)
            self.release(record)
            records.append(record)
            self.now += 1  # Entire burst fits within the default 300-second cooldown.
        queued = [r["id"] for r in records if r.get("operation", {}).get("action") == "reclaim"]
        self.assertEqual(queued, [], "per-workspace cooldown does not filter fresh empty images")

    def test_reclaim_disabled_does_not_enqueue(self):
        self.configure(False)
        record = self.new_workspace()
        self.release(record)
        self.assertNotIn("operation", record)

    def test_small_empty_image_does_not_enqueue(self):
        record = self.new_workspace(allocated=4 * 1024 ** 2)
        self.release(record)
        self.assertNotIn("operation", record)

    def test_large_allocation_growth_remains_eligible_and_deduplicated(self):
        record = self.new_workspace()
        # Model backing blocks left after temporary data was written and deleted.
        record["space"]["allocated_bytes"] += 2 * 1024 ** 3
        self.release(record)
        candidate = copy.deepcopy(record["reclaim_candidate"])
        self.assertEqual(candidate["estimated_bytes"], 2 * 1024 ** 3)
        self.assertEqual(candidate["usage_generation"], 1)
        self.assertNotIn("operation", record, "candidate must not block new commands")
        self.controller._automatic_workspace_operation(record, self.node)
        self.assertEqual(record["reclaim_candidate"], candidate)


if __name__ == "__main__":
    unittest.main()
