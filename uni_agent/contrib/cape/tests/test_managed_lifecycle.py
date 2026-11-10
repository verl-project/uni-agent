# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from pycape.cli import CapeHTTPError
from pycape.models import CommandState, NodeState, RequestState, SessionState
from tests import test_identity_nodes as nodes


class ManagedLifecycleTests(nodes.IdentityNodeTests):
    def announced(self):
        owner, agent, body, result = self.enrollment()
        self.call("POST", "/v1/agent/announce", agent,
                  {key: body[key] for key in ("node_id", "pool", "allocation_id", "agent_epoch", "resources")})
        return owner, agent, body, result

    def pulse(self, agent):
        return self.call("POST", "/v1/agent/heartbeat", agent,
                         {"node_id": "node-a", "agent_epoch": "epoch-a", "status": {"authoritative": True, "sessions": []}})

    def test_drain_waits_pending_assignment_and_new_cleanup_heartbeat(self):
        owner, agent, body, _ = self.announced()
        clock = [100.]
        self.controller.clock = lambda: clock[0]
        self.pulse(agent)
        assignment = SimpleNamespace(node_id="node-a", state=CommandState.PENDING, created_at=100., updated_at=100.)
        self.controller.command_assignments["command"] = assignment
        result = self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "drain"})
        self.assertEqual(result["state"], "DRAINING")
        clock[0] = 101.
        self.pulse(agent)
        self.controller.schedule_once()
        self.assertEqual(self.controller.nodes["node-a"].state, NodeState.DRAINING)
        assignment.state = CommandState.COMPLETED
        assignment.updated_at = 102.
        clock[0] = 103.
        self.controller.schedule_once()
        self.assertEqual(self.controller.nodes["node-a"].state, NodeState.DRAINING)
        self.pulse(agent)
        self.controller.schedule_once()
        self.assertEqual(self.controller.nodes["node-a"].state, NodeState.REMOVED)
        self.assertEqual(self.pulse(agent)["agent_control"]["action"], "shutdown")

    def test_force_and_lost_node_do_not_invent_cleanup(self):
        owner, agent, _, _ = self.announced()
        clock = [100.]
        self.controller.clock = lambda: clock[0]
        self.pulse(agent)
        self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "drain", "force": True})
        clock[0] = 200.
        self.controller.nodes["node-a"].state = NodeState.LOST
        self.controller.schedule_once()
        self.assertEqual(self.controller.nodes["node-a"].state, NodeState.LOST)
        self.pulse(agent)
        self.controller.schedule_once()
        self.assertEqual(self.controller.nodes["node-a"].state, NodeState.REMOVED)

    def test_exit_requires_verified_idle_and_pool_drain_reopens_empty(self):
        owner, agent, body, _ = self.announced()
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "exit"})
        result = self.call("POST", "/v1/identity/admin", self.admin, {"action": "pool-drain", "pool": body["pool"]})
        self.assertEqual(result["state"], "draining")
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/submit", owner, {"pool": body["pool"], "session_id": "t", "command": ["true"]})
        self.pulse(agent)
        self.controller.schedule_once()
        pools = self.call("GET", "/v1/identity/pools", owner)
        self.assertEqual(pools[0]["state"], "unavailable")
        request = self.call("POST", "/v1/submit", owner, {"pool": body["pool"], "session_id": "t", "command": ["true"]})
        self.assertEqual(request["state"], "QUEUED")
        self.call("POST", "/v1/identity/admin", self.admin, {"action": "pool-remove", "pool": body["pool"]})
        self.assertEqual(self.call("GET", "/v1/requests/" + request["request_id"], owner)["reason"], "pool_draining")

    def test_pool_remove_cannot_discard_live_node(self):
        _, _, body, _ = self.announced()
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/admin", self.admin, {"action": "pool-remove", "pool": body["pool"]})

    def test_announced_but_missing_node_is_not_treated_as_never_used(self):
        owner, _, body, _ = self.announced()
        del self.controller.nodes["node-a"]
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "exit"})
        result = self.call("POST", "/v1/identity/admin", self.admin, {"action": "pool-drain", "pool": body["pool"], "force": True})
        self.assertEqual(result["state"], "draining")
        self.assertEqual(self.instance.store.snapshot()["nodes"]["node-a"]["desired"], "draining")

    def test_owner_force_can_cancel_but_not_inspect_other_user_request(self):
        owner, _, body, _ = self.announced()
        bob, _, _ = self.user("bob")
        request = self.call("POST", "/v1/submit", bob, {"pool": body["pool"], "session_id": "t", "command": ["true"]})
        record = self.controller.requests[request["request_id"]]
        record.node_id = "node-a"
        record.state = RequestState.RUNNING
        with self.assertRaises(CapeHTTPError):
            self.call("GET", "/v1/requests/" + record.request_id, owner)
        result = self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "drain", "force": True})
        self.assertEqual(set(result), {"node_id", "state"})
        self.assertEqual(self.call("GET", "/v1/requests/" + record.request_id, bob)["state"], "CANCELLED")

    def test_node_edit_reduces_only_after_cleanup_and_keeps_identity(self):
        owner, agent, body, _ = self.announced()
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "edit", "changes": {"cpu_cores": 2}})
        result = self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "edit", "changes": {"cpu_cores": 0.5}})
        self.assertEqual(result["desired"], "reconfigure")
        self.assertEqual(self.controller.nodes["node-a"].resources.cpu_cores, 1)
        self.pulse(agent)
        self.controller.schedule_once()
        self.assertEqual(self.controller.nodes["node-a"].resources.cpu_cores, 0.5)
        self.assertEqual(self.controller.nodes["node-a"].state, NodeState.ACTIVE)
        # The original enrollment envelope remains the credential's immutable
        # verified ceiling; re-registration cannot restore the reduced capacity.
        self.call("POST", "/v1/agent/announce", agent,
                  {key: body[key] for key in ("node_id", "pool", "allocation_id", "agent_epoch", "resources")})
        self.assertEqual(self.controller.nodes["node-a"].resources.cpu_cores, 0.5)

    def test_terminal_report_during_reconciliation_stops_draining_session(self):
        owner, agent, body, _ = self.announced()
        registration = {key: body[key] for key in ("node_id", "pool", "allocation_id", "agent_epoch", "resources")}
        desired = self.controller.workspace_configs.get(body["pool"], "node-a")
        registration.update(runtime_capabilities={"local-process": {"adapter": "local-process", "nested_container": True, "can_bind_mounts": True,
                            "image_formats": ["default"], "isolation_policies": ["default", "none"]}},
                            workspace_config={"protocol": 1, "state": "applied", "applied_version": desired.version, "config": desired.to_dict()})
        self.call("POST", "/v1/agent/register", agent, registration)
        request = self.call("POST", "/v1/submit", owner, {"pool": body["pool"], "session_id": "t", "command": ["true"],
                            "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}})
        self.controller.schedule_once()
        record = self.controller.requests[request["request_id"]]
        self.assertEqual(record.state, RequestState.RUNNING, record.reason)
        record.state = RequestState.RECONCILING
        session = self.controller.sessions[record.execution_session_id]
        session.state = SessionState.RECONCILING
        self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "drain"})
        result = self.call("POST", "/v1/agent/report-command", agent,
                           {"node_id": "node-a", "agent_epoch": "epoch-a", "command_id": self.controller.request_commands[record.request_id],
                            "exit_code": 0, "accept_commands": True})
        self.assertEqual(result["state"], "COMPLETED")
        self.assertEqual(session.state, SessionState.STOPPED)
        self.assertFalse(self.controller.nodes["node-a"].active_sessions)

    def test_late_cached_reconciliation_is_swept_without_stopping_inflight_work(self):
        owner, agent, body, _ = self.announced()
        desired = self.controller.workspace_configs.get(body["pool"], "node-a")
        registration = {key: body[key] for key in ("node_id", "pool", "allocation_id", "agent_epoch", "resources")}
        registration.update(runtime_capabilities={"local-process": {"adapter": "local-process", "nested_container": True, "can_bind_mounts": True,
                            "image_formats": ["default"], "isolation_policies": ["default", "none"]}},
                            workspace_config={"protocol": 1, "state": "applied", "applied_version": desired.version, "config": desired.to_dict()})
        self.call("POST", "/v1/agent/register", agent, registration)
        response = self.call("POST", "/v1/submit", owner, {"pool": body["pool"], "session_id": "t", "command": ["true"],
                             "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}})
        self.controller.schedule_once()
        record = self.controller.requests[response["request_id"]]
        session = self.controller.sessions[record.execution_session_id]
        assignment = self.controller.command_assignments[self.controller.request_commands[record.request_id]]
        self.call("POST", "/v1/identity/nodes/node-a/action", owner, {"action": "drain"})
        session.state = SessionState.CACHED_IDLE
        self.controller.schedule_once()
        self.assertEqual(session.state, SessionState.CACHED_IDLE, "pending assignment remains the original node's responsibility")
        assignment.state = CommandState.COMPLETED
        record.state = RequestState.COMPLETED
        session.active_request_id = None
        self.controller.schedule_once()
        self.assertEqual(session.state, SessionState.STOPPED)
        self.assertFalse(self.controller.nodes["node-a"].active_sessions)

    def test_pool_limit_policy_and_deleted_id_do_not_expand_permissions(self):
        owner, _, body, _ = self.enrollment()
        self.call("POST", "/v1/identity/admin", self.admin,
                  {"action": "pool-edit", "pool": body["pool"], "changes": {"max_nodes": 1}})
        import uuid
        from pycape.identity import new_secret, verifier
        extra = dict(body, operation_id=str(uuid.uuid4()), verifier=verifier(new_secret()), node_id="node-b", allocation_id="allocation-b")
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/nodes/enroll", owner, extra)
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/admin", self.admin,
                      {"action": "pool-edit", "pool": body["pool"], "changes": {"config": {"pool_mode": "shared", "gpu_slots": 2, "placement": "pack"}}})
        self.call("POST", "/v1/identity/admin", self.admin, {"action": "group-set", "name": "default", "pools": [body["pool"]]})
        self.call("POST", "/v1/identity/admin", self.admin, {"action": "pool-drain", "pool": body["pool"]})
        self.call("POST", "/v1/identity/admin", self.admin, {"action": "pool-remove", "pool": body["pool"]})
        recreated = self.pool()
        self.assertNotEqual(recreated, body["pool"])
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/submit", owner, {"pool": recreated, "session_id": "t", "command": ["true"]})
