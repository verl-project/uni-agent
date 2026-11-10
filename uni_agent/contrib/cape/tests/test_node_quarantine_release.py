# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from pycape import CAPEController
from pycape.identity import new_secret, verifier
from pycape.identity_nodes import authenticate_agent
from pycape.managed_auth import ManagedAuthority
from pycape.models import NodeState
from pycape.node_purge import purge, release_quarantine
from pycape.workspace import WorkspaceConfig
from tests import test_managed_identity as managed


GPU1 = "GPU-00000000-0000-0000-0000-000000000001"
GPU2 = "GPU-00000000-0000-0000-0000-000000000002"


class QuarantineReleaseTests(unittest.TestCase):
    setUp = managed.ManagedIdentityTests.setUp
    call = managed.ManagedIdentityTests.call
    cli = managed.ManagedIdentityTests.cli
    user = managed.ManagedIdentityTests.user
    pool = managed.ManagedIdentityTests.pool

    def nodes(self):
        owner, _, _ = self.user("contributor")
        pool = self.pool()
        agents = {}
        for node_id, gpus in (("a", [GPU1, GPU2]), ("b", [GPU2]), ("new", [GPU1, GPU2])):
            agent = new_secret()
            agents[node_id] = agent
            body = {"operation_id": str(uuid.uuid4()), "verifier": verifier(agent), "node_id": node_id,
                    "pool": pool, "agent_epoch": "epoch-" + node_id, "allocation_id": "allocation-" + node_id,
                    "resources": {"cpu_cores": 2, "memory_gb": 2, "gpu_uuids": gpus}}
            self.call("POST", "/v1/identity/nodes/enroll", owner, body)
            self.call("POST", "/v1/agent/announce", agent, body)
        for node_id in ("a", "b"):
            self.controller.nodes[node_id].state = NodeState.LOST
            purge(self.controller, self.admin, node_id, "lost")
        return owner, pool, agents

    def release(self, node="a", **kwargs):
        return release_quarantine(self.controller, kwargs.get("token", self.admin), node,
                                  kwargs.get("reason", "old processes stopped"), kwargs.get("confirm", True))

    def test_overlap_receipt_idempotency_and_existing_leases(self):
        _, _, agents = self.nodes()
        first = self.release()
        self.assertEqual(first["released_source_gpu_uuids"], [GPU1, GPU2])
        self.assertEqual(first["unquarantined_gpu_uuids"], [GPU1])
        self.assertEqual(first["still_quarantined_gpu_uuids"], [GPU2])
        self.assertEqual(self.release(reason="retry"), first)
        self.assertFalse(first["cleanup_confirmed"])
        node = self.controller.nodes["new"]
        self.assertTrue(self.controller._gpu_admission_allowed(node, GPU1))
        self.assertFalse(self.controller._gpu_admission_allowed(node, GPU2))
        ledger = self.controller.gpu_ledger[GPU1]
        ledger.exclusive_execution_session_id = "ordinary-existing-lease"
        second = self.release("b")
        self.assertEqual(second["unquarantined_gpu_uuids"], [GPU2])
        self.assertFalse(self.controller._gpu_admission_allowed(node, GPU1))
        self.assertEqual(ledger.exclusive_execution_session_id, "ordinary-existing-lease")
        refreshed = self.release()
        self.assertEqual(refreshed["release_receipt"], first["release_receipt"])
        self.assertEqual(refreshed["unquarantined_gpu_uuids"], [GPU1, GPU2])
        self.assertEqual(refreshed["still_quarantined_gpu_uuids"], [])
        purge(self.controller, self.admin, "a", "retry purge")
        self.assertEqual(self.controller.node_purge_quarantine, set())
        for node_id in ("a", "b"):
            self.assertEqual(self.controller.nodes[node_id].state, NodeState.REMOVED)
            with self.assertRaises(PermissionError):
                authenticate_agent(self.instance.store, agents[node_id])
        audits = self.instance.store.audit_events(self.admin, 1000)
        self.assertEqual(sum(e["action"] == "node.release-quarantine" for e in audits), 2)

    def test_permission_confirmation_reason_and_nonpurged_rejection(self):
        owner, _, agents = self.nodes()
        before = self.instance.store.snapshot()
        for token in (owner, agents["a"], "invalid"):
            with self.assertRaises(PermissionError):
                self.release(token=token)
        for confirm in (False, None, 1, "true"):
            with self.assertRaises(ValueError):
                self.release(confirm=confirm)
        for reason in (None, "", "   ", "界" * 342):
            with self.assertRaises(ValueError):
                self.release(reason=reason)
        for node in ("missing", "new"):
            with self.assertRaises(ValueError):
                self.release(node)
        self.assertEqual(before, self.instance.store.snapshot())

    def test_durable_release_survives_cache_update_failure_without_changing_images(self):
        _, pool, agents = self.nodes()
        image = Path(self.temp.name) / "retained.img"
        image.write_bytes(b"preserved image bytes")
        workspace = {"id": "ws", "node_id": "a", "pool": pool, "user_id": "owner", "execution_session_id": "old",
                     "state": "retained", "revision": 1, "path": str(image), "config": WorkspaceConfig().to_dict(),
                     "node_retired": True, "cleanup_confirmed": False,
                     "operation": {"id": "uncertain-helper", "state": "pending", "dispatch_granted": True}}
        self.controller.workspaces["ws"] = copy.deepcopy(workspace)
        self.controller._emit("workspace_state", "ws", "retained", None, workspace)
        self.journal.flush()
        with patch("pycape.managed_lifecycle.reconcile", side_effect=RuntimeError("interrupted after commit")):
            with self.assertRaises(RuntimeError):
                self.release()
        self.assertIn("quarantine_release", self.instance.store.snapshot()["nodes"]["a"])
        restored = CAPEController.reconcile_from_journal(self.journal, auth=ManagedAuthority(self.instance.store))
        self.assertEqual(restored.node_purge_quarantine, {GPU2})
        self.assertEqual(restored.workspaces["ws"], workspace)
        self.assertEqual(image.read_bytes(), b"preserved image bytes")
        with self.assertRaises(PermissionError):
            authenticate_agent(self.instance.store, agents["a"])
        retry = release_quarantine(restored, self.admin, "a", "retry", True)
        self.assertEqual(retry["release_receipt"]["reason"], "old processes stopped")

    def test_malformed_release_cannot_silently_remove_fence(self):
        self.nodes()
        with self.instance.store._transaction() as state:
            state["nodes"]["a"]["quarantine_release"] = {"confirmed_stopped": False}
        with self.assertRaises(ValueError):
            self.instance.store.snapshot()
