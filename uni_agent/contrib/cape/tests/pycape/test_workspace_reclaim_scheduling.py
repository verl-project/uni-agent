# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from types import SimpleNamespace
import unittest

from pycape import CAPEController
from pycape.models import NodeState, RequestState
from pycape.workspace_admission import complete


class WorkspaceReclaimSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.
        self.controller = CAPEController(clock=lambda: self.now)
        self.controller.workspace_configs.update("pool", {"maintenance": {
            "reclaim_on_release": True, "reclaim_idle_seconds": 10}})
        self.config = self.controller.workspace_configs.get("pool")
        for node_id in ("a", "b"):
            self.controller.nodes[node_id] = SimpleNamespace(pool="pool", state=NodeState.ACTIVE,
                agent_epoch="epoch", workspace_status={"maintenance_protocol": 1, "state": "applied",
                "applied_version": self.config.version, "maintenance_host_key": "a"*64})

    def workspace(self, identity="w1", node="a"):
        reference = {"allocated_bytes": 256*1024**2, "used_bytes": 256*1024**2,
            "logical_bytes": self.config.image_size_bytes, "image_generation": 0, "usage_generation": 0}
        record = {"id": identity, "node_id": node, "pool": "pool", "user_id": "alice",
            "execution_session_id": identity, "state": "retained", "usage_generation": 1, "generation": 0,
            "config": self.config.to_dict(), "reclaim_reference": reference,
            "reclaim_sample": dict(reference, allocated_bytes=1024**3, usage_generation=1)}
        self.controller.workspaces[identity] = record
        self.controller._automatic_workspace_operation(record, self.controller.nodes[node])
        return record

    def test_candidate_is_nonblocking_until_idle_worker_claim(self):
        record = self.workspace()
        self.assertNotIn("operation", record)
        self.assertEqual(self.controller.workspace_operations("a", True), [])
        self.now += 11
        self.assertEqual(self.controller.workspace_operations("a"), [])
        grant = self.controller.workspace_operations("a", True)
        self.assertEqual(len(grant), 1)
        self.assertTrue(grant[0]["operation"]["dispatch_granted"])
        self.assertEqual(self.controller.workspace_operations("a", True), grant)

    def test_host_slot_serializes_agents_and_start_rate_survives_completion(self):
        first, second = self.workspace(), self.workspace("w2", "b")
        self.now += 11
        self.assertEqual(len(self.controller.workspace_operations("a", True)), 1)
        self.assertEqual(self.controller.workspace_operations("b", True), [])
        first["operation"].update(state="completed", reclaimed_bytes=100*1024**2)
        complete(first, self.now)
        self.assertEqual(self.controller.workspace_operations("b", True), [])
        self.now += 31
        self.assertEqual(len(self.controller.workspace_operations("b", True)), 1)
        self.assertIsNone(second["reclaim_candidate"])

    def test_unknown_host_serializes_globally_but_known_distinct_hosts_do_not(self):
        self.workspace()
        self.workspace("w2", "b")
        self.now += 11
        self.controller.workspace_operations("a", True)
        self.controller.nodes["b"].workspace_status.pop("maintenance_host_key")
        self.assertEqual(self.controller.workspace_operations("b", True), [])
        self.controller.nodes["b"].workspace_status["maintenance_host_key"] = "b"*64
        self.assertEqual(len(self.controller.workspace_operations("b", True)), 1)

    def test_new_user_work_and_changed_policy_preempt_unstarted_candidate(self):
        record = self.workspace()
        self.now += 11
        self.controller.requests["request"] = SimpleNamespace(spec=SimpleNamespace(pool="pool"), state=RequestState.QUEUED)
        self.assertEqual(self.controller.workspace_operations("a", True), [])
        self.assertNotIn("operation", record)
        self.controller.requests.clear()
        self.controller.workspace_configs.update("pool", {"maintenance": {"reclaim_on_release": False}})
        self.assertEqual(self.controller.workspace_operations("a", True), [])
        self.assertNotIn("operation", record)

    def test_failure_and_low_yield_backoff_and_later_same_execution(self):
        record = self.workspace()
        record["operation"] = {"action": "reclaim", "state": "failed", "policy": self.config.maintenance}
        complete(record, self.now)
        self.assertEqual(record["reclaim_retry_after"], self.now+300)
        record["operation"].update(state="completed", reclaimed_bytes=0)
        complete(record, self.now)
        self.assertEqual(record["reclaim_success_generation"], 1)
        self.assertEqual(record["reclaim_low_yield_until"], self.now+3600)
        record["usage_generation"] = 2
        record["reclaim_sample"]["usage_generation"] = 2
        self.controller._automatic_workspace_operation(record, self.controller.nodes["a"])
        self.assertEqual(record["reclaim_candidate"]["usage_generation"], 2)
        self.now += 3601
        self.assertEqual(len(self.controller.workspace_operations("a", True)), 1)

    def test_journal_preserves_grant_for_redelivery(self):
        self.workspace()
        self.now += 11
        grant = self.controller.workspace_operations("a", True)
        restored = CAPEController.reconcile_from_journal(self.controller.journal, clock=lambda: self.now)
        restored.nodes = copy.deepcopy(self.controller.nodes)
        self.assertEqual(restored.workspace_operations("a", True), grant)

    def test_unavailable_ungranted_owner_does_not_block_other_agent(self):
        unavailable = self.workspace()
        self.workspace("w2", "b")
        unavailable["operation"] = {"id": "explicit", "state": "pending", "action": "reclaim",
                                    "dispatch_granted": False, "requested_at": 1}
        self.controller.nodes["a"].state = NodeState.UNHEALTHY
        self.now += 11
        self.assertEqual(len(self.controller.workspace_operations("b", True)), 1)
        self.assertFalse(unavailable["operation"]["dispatch_granted"])

    def test_uncertain_granted_owner_still_blocks_after_health_or_epoch_change(self):
        self.workspace()
        self.workspace("w2", "b")
        self.now += 11
        self.controller.workspace_operations("a", True)
        self.controller.nodes["a"].state = NodeState.UNHEALTHY
        self.controller.nodes["a"].agent_epoch = "new-epoch"
        self.controller.nodes["b"].workspace_status["maintenance_host_key"] = "b"*64
        self.assertEqual(self.controller.workspace_operations("b", True), [])

    def test_requests_in_a_detached_pool_do_not_block_unrelated_host(self):
        self.workspace()
        removed = self.controller.nodes["b"]
        removed.state = NodeState.REMOVED
        removed.pool = "detached"
        removed.workspace_status.pop("maintenance_host_key")
        self.controller.requests["request"] = SimpleNamespace(spec=SimpleNamespace(pool="detached"), state=RequestState.QUEUED)
        self.now += 11
        self.assertEqual(len(self.controller.workspace_operations("a", True)), 1)


if __name__ == "__main__":
    unittest.main()
