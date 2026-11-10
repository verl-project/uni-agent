# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from pycape import CAPEController
from pycape.identity import new_secret, verifier
from pycape.identity_nodes import authenticate_agent, enroll
from pycape.managed_auth import ManagedAuthority
from pycape.models import NodeState
from pycape.node_purge import purge
from pycape.workspace import WorkspaceConfig
from tests import test_managed_identity as managed


class NodePurgeTests(unittest.TestCase):
    setUp = managed.ManagedIdentityTests.setUp
    call = managed.ManagedIdentityTests.call
    cli = managed.ManagedIdentityTests.cli
    user = managed.ManagedIdentityTests.user
    pool = managed.ManagedIdentityTests.pool

    def node(self):
        owner, _, _ = self.user("contributor")
        pool = self.pool()
        agent = new_secret()
        body = {"node_id": "node-a", "pool": pool, "agent_epoch": "epoch", "allocation_id": "allocation",
                "operation_id": str(uuid.uuid4()), "verifier": verifier(agent),
                "resources": {"cpu_cores": 1, "memory_gb": 1,
                              "gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000001"]}}
        self.call("POST", "/v1/identity/nodes/enroll", owner, body)
        self.call("POST", "/v1/agent/announce", agent, body)
        return owner, agent, pool

    def test_only_lost_and_admin_no_force_bypass(self):
        owner, agent, _ = self.node()
        for state in NodeState:
            if state == NodeState.LOST:
                continue
            self.controller.nodes["node-a"].state = state
            with self.subTest(state=state), self.assertRaises(ValueError):
                purge(self.controller, self.admin, "node-a", "reason")
        self.controller.nodes["node-a"].state = NodeState.LOST
        for token in (owner, agent, "invalid"):
            with self.assertRaises(PermissionError):
                purge(self.controller, token, "node-a", "reason")
        result = purge(self.controller, self.admin, "node-a", "reason")
        self.assertFalse(result["cleanup_confirmed"])
        self.assertEqual(purge(self.controller, self.admin, "node-a", "different reason"), result)
        from pycape.managed_lifecycle import reconcile
        with patch("pycape.node_purge.project") as repeated:
            reconcile(self.controller)
            reconcile(self.controller)
            repeated.assert_not_called()
        intent = self.instance.store.snapshot()["nodes"]["node-a"]
        body = {k: intent[k] for k in ("operation_id", "verifier", "node_id", "pool", "agent_epoch", "allocation_id", "resources")}
        self.assertEqual(enroll(self.instance.store, owner, body)["status"], "enrolled")
        with self.assertRaises(PermissionError):
            authenticate_agent(self.instance.store, agent)
        self.assertEqual(self.instance.store.snapshot()["nodes"]["node-a"]["desired"], "removed")

    def test_committed_intent_survives_failed_projection_and_replay(self):
        _, agent, pool = self.node()
        node = self.controller.nodes["node-a"]
        node.state = NodeState.LOST
        image = Path(self.temp.name) / "retained.img"
        image.write_bytes(b"untouched filesystem image placeholder")
        operation = {"id": "uncertain", "state": "pending", "action": "reclaim", "dispatch_granted": True}
        workspace = {"id": "ws-retained", "node_id": node.node_id, "pool": pool,
                     "user_id": "owner", "execution_session_id": "execution", "state": "retained",
                     "config": WorkspaceConfig().to_dict(), "revision": 1, "path": str(image), "operation": operation}
        self.controller.workspaces[workspace["id"]] = copy.deepcopy(workspace)
        self.controller._emit("workspace_state", workspace["id"], "retained", None, workspace)
        self.journal.flush()
        with patch("pycape.node_purge.project", side_effect=RuntimeError("projection interrupted")):
            with self.assertRaises(RuntimeError):
                purge(self.controller, self.admin, node.node_id, "allocation ended")
        with self.assertRaises(PermissionError):
            authenticate_agent(self.instance.store, agent)
        restored = CAPEController.reconcile_from_journal(self.journal, auth=ManagedAuthority(self.instance.store))
        self.assertEqual(restored.nodes[node.node_id].state, NodeState.REMOVED)
        self.assertEqual(restored.node_purge_quarantine, set(node.resources.gpu_uuids))
        self.assertTrue(restored.workspaces[workspace["id"]]["node_retired"])
        self.assertEqual(restored.workspaces[workspace["id"]]["operation"], operation)
        self.assertEqual(image.read_bytes(), b"untouched filesystem image placeholder")
        self.assertFalse(restored._gpu_admission_allowed(node, node.resources.gpu_uuids[0]))
        self.assertEqual(purge(restored, self.admin, node.node_id, "retry")["reason"], "allocation ended")

    def test_corrupt_tombstone_cannot_drop_gpu_fence(self):
        self.node()
        self.controller.nodes["node-a"].state = NodeState.LOST
        purge(self.controller, self.admin, "node-a", "ended")
        with self.instance.store._transaction() as state:
            state["nodes"]["node-a"]["purge"]["quarantined_gpu_uuids"] = []
        with self.assertRaises(ValueError):
            self.instance.store.snapshot()

    def test_previously_relinquished_gpu_is_not_quarantined(self):
        self.node()
        node = self.controller.nodes["node-a"]
        node.state = NodeState.LOST
        with self.instance.store._transaction() as state:
            state["nodes"]["node-a"]["resources"]["gpu_uuids"].append("GPU-00000000-0000-0000-0000-000000000002")
        result = purge(self.controller, self.admin, "node-a", "ended after a verified resource reduction")
        self.assertEqual(result["quarantined_gpu_uuids"], list(node.resources.gpu_uuids))
