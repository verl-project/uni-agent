# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from pycape.cli import CapeHTTPError
from pycape.identity import new_secret, verifier
from tests import test_managed_identity as managed


class IdentityNodeTests(managed.ManagedIdentityTests):
    def enrollment(self):
        owner, _, _ = self.user("alice")
        pool = self.pool()
        agent = new_secret()
        body = {"operation_id": str(uuid.uuid4()), "verifier": verifier(agent), "node_id": "node-a", "pool": pool,
                "allocation_id": "allocation-a", "agent_epoch": "epoch-a", "resources": {"cpu_cores": 1, "memory_gb": 1, "gpu_uuids": []}}
        result = self.call("POST", "/v1/identity/nodes/enroll", owner, body)
        return owner, agent, body, result

    def test_enroll_has_no_join_or_approval_and_is_idempotent(self):
        owner, agent, body, result = self.enrollment()
        self.assertEqual(self.call("POST", "/v1/identity/nodes/enroll", owner, body), result)
        announcement = {key: body[key] for key in ("node_id", "pool", "allocation_id", "agent_epoch", "resources")}
        response = self.call("POST", "/v1/agent/announce", agent, announcement)
        self.assertEqual(response["state"], "REGISTERING")
        with self.assertRaises(CapeHTTPError):
            self.call("GET", "/v1/identity/me", agent)
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/agent/announce", owner, announcement)
        for key, value in (("agent_epoch", "other-epoch"), ("allocation_id", "other-allocation"), ("node_id", "other-node")):
            with self.subTest(key=key), self.assertRaises(CapeHTTPError):
                self.call("POST", "/v1/agent/announce", agent, dict(announcement, **{key: value}))
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/agent/heartbeat", agent, {"node_id": "node-a"})
        self.call("POST", "/v1/agent/heartbeat", agent, {"node_id": "node-a", "agent_epoch": "epoch-a", "status": {"authoritative": False}})

    def test_node_owner_and_scope_are_not_transferable(self):
        owner, agent, body, _ = self.enrollment()
        bob, _, _ = self.user("bob")
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/nodes/enroll", bob, body)
        announcement = {key: body[key] for key in ("node_id", "pool", "allocation_id", "agent_epoch", "resources")}
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/agent/announce", agent, dict(announcement, resources={"cpu_cores": 2, "memory_gb": 1, "gpu_uuids": []}))
        self.call("POST", "/v1/agent/announce", agent, announcement)
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/admin/drain-node", bob, {"node_id": "node-a"})
        self.call("POST", "/v1/admin/drain-node", owner, {"node_id": "node-a"})
        # Personal account disable must not cut off its agent's cleanup channel.
        actor = self.call("GET", "/v1/identity/me", owner)
        self.call("POST", "/v1/identity/admin", self.admin, {"action": "user-edit", "user_id": actor["user_id"], "disabled": True})
        self.call("POST", "/v1/agent/heartbeat", agent, {"node_id": "node-a", "agent_epoch": "epoch-a", "status": {"authoritative": False}})

    def test_uncommitted_enrollment_cannot_change_target_owner(self):
        alice, identity, _ = self.user("alice")
        bob, _, _ = self.user("bob")
        body = {"operation_id": str(uuid.uuid4()), "verifier": verifier(new_secret()), "node_id": "node-a", "pool": self.pool(),
                "allocation_id": "allocation-a", "agent_epoch": "epoch-a", "owner_user_id": identity["user_id"],
                "resources": {"cpu_cores": 1, "memory_gb": 1, "gpu_uuids": []}}
        before = self.instance.store.snapshot()
        with self.assertRaises(CapeHTTPError) as denied:
            self.call("POST", "/v1/identity/nodes/enroll", bob, body)
        self.assertEqual(denied.exception.status, 403)
        self.assertEqual(before, self.instance.store.snapshot())
        self.call("POST", "/v1/identity/nodes/enroll", alice, body)
