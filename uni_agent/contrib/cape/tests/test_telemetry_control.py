# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
import uuid

from pycape.cli import CapeHTTPError
from pycape.identity import new_secret, verifier
from pycape.telemetry_control import get_config, poll_control, set_config
from tests import test_managed_identity as managed


class TelemetryControlTests(unittest.TestCase):
    setUp = managed.ManagedIdentityTests.setUp
    call = managed.ManagedIdentityTests.call
    cli = managed.ManagedIdentityTests.cli
    user = managed.ManagedIdentityTests.user
    pool = managed.ManagedIdentityTests.pool

    def enroll(self, owner, pool, node):
        secret = new_secret()
        self.call("POST", "/v1/identity/nodes/enroll", owner, {
            "operation_id": str(uuid.uuid4()), "verifier": verifier(secret), "node_id": node,
            "pool": pool, "agent_epoch": "epoch-" + node, "allocation_id": "allocation-" + node,
            "resources": {"cpu_cores": 1, "memory_gb": 1, "gpu_uuids": []}})
        return secret

    def control(self, secret, node, service="host-service", **extra):
        return self.call("POST", "/v1/agent/metrics/control", secret,
                         dict(node_id=node, agent_epoch="epoch-" + node, service_id=service, **extra))

    def test_authorization_versions_scopes_and_cli(self):
        owner, _, owner_path = self.user("contributor")
        pool = self.pool()
        a, b, c = (self.enroll(owner, pool, node) for node in ("a", "b", "c"))
        self.assertFalse(self.control(a, "a")["config"]["enabled"])
        self.control(b, "b")
        self.control(c, "c", "other-host")
        body = {"action": "metric-set", "changes": {"interval_seconds": 2}, "node_id": "a"}
        for token in (None, owner, a):
            with self.assertRaises(CapeHTTPError) as error:
                self.call("POST", "/v1/identity/admin", token, body)
            self.assertEqual(error.exception.status, 403)
        changed = self.call("POST", "/v1/identity/admin", self.admin, body)
        self.assertEqual(changed["version"], 1)
        self.assertEqual(self.control(b, "b")["config"]["interval_seconds"], 2)
        self.assertEqual(self.control(c, "c", "other-host")["config"]["interval_seconds"], 5)
        self.assertEqual(self.call("POST", "/v1/identity/admin", self.admin, body)["version"], 1)
        self.control(a, "a", applied_version=1, state="applied")
        self.control(a, "a", applied_version=0, state="pending")
        status = self.call("GET", "/v1/metrics/status?node_id=a", owner)
        self.assertEqual(status["receipts"]["a"]["version"], 1)
        self.assertEqual(status["receipts"]["a"]["state"], "applied")
        code, out, err = self.cli("metric", "status", "--controller-url", self.service.url, "--token-file", owner_path)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["version"], 1)
        code, out, err = self.cli("admin", "metric", "set", "--controller-url", self.service.url,
                                  "--token", self.admin, "--expected-version", "1", "--profiling", "on")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["version"], 2)
        self.assertNotIn(self.admin, out + err)
        audits = self.instance.store.audit_events(self.admin, 100)
        self.assertEqual(sum(e["action"] == "metric.configure" for e in audits), 2)

    def test_invalid_inputs_leave_store_unchanged(self):
        owner, _, _ = self.user("contributor")
        secret = self.enroll(owner, self.pool(), "a")
        self.control(secret, "a")
        before = self.instance.store.snapshot()
        for changes in ({}, [], {"enabled": 1}, {"interval_seconds": 0}, {"interval_seconds": 10 ** 400}, {"token": "x"}):
            with self.assertRaises(ValueError):
                set_config(self.instance.store, self.admin, changes)
        for extra in ({"agent_epoch": "old"}, {"service_id": "other"}, {"applied_version": 1},
                      {"applied_version": True}, {"state": "bad"}, {"config": {"enabled": True}}):
            body = dict(node_id="a", agent_epoch="epoch-a", service_id="host-service")
            body.update(extra)
            with self.assertRaises((ValueError, PermissionError)):
                poll_control(self.instance.store, secret, body)
        with self.assertRaises(ValueError):
            set_config(self.instance.store, self.admin, {"enabled": True}, expected_version=7)
        self.assertEqual(self.instance.store.snapshot(), before)

    def test_retired_agent_cannot_fetch_control(self):
        owner, _, _ = self.user("contributor")
        secret = self.enroll(owner, self.pool(), "a")
        self.control(secret, "a")
        with self.instance.store._transaction() as state:
            state["nodes"]["a"]["desired"] = "removed"
        with self.assertRaises(PermissionError):
            poll_control(self.instance.store, secret, dict(node_id="a", agent_epoch="epoch-a", service_id="host-service"))
        self.assertEqual(get_config(self.instance.store, owner)["version"], 0)


if __name__ == "__main__":
    unittest.main()
