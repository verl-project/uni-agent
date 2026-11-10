# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import tempfile
import unittest
import urllib.error

from tests.rscape import test_functional_parity as parity


class TelemetryControlParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_both_clients_servers_and_cross_backend_recovery(self):
        binaries = (parity.PYCAPE_BIN, parity.RSCAPE_BIN)
        for server in binaries:
            with self.subTest(server=server), tempfile.TemporaryDirectory() as root:
                helper = parity.FunctionalParityTests()
                process, url, _ = helper.start_controller(server, root)
                fixture = helper.fixtures[url]
                try:
                    for node in ("a", "b", "c"):
                        helper.register(url, node, "pool")
                    def poll(node, service="shared-service", **extra):
                        return helper.request("POST", url, "/v1/agent/metrics/control", dict(node_id=node, service_id=service, **extra))
                    for node, service in (("a", "shared-service"), ("b", "shared-service"), ("c", "other-service")):
                        self.assertEqual(poll(node, service)["version"], 0)
                    heartbeat = helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "a", "max_commands": 0,
                        "status": {"authoritative": False, "metrics": {"service_id": "shared-service"}}})
                    self.assertEqual(heartbeat["metrics_control"]["version"], 0)
                    invalid_metrics = helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "a", "max_commands": 0,
                        "status": {"authoritative": False, "metrics": {"service_id": []}}})
                    self.assertEqual(invalid_metrics["metrics_control"]["reason"], "invalid_metrics_control")
                    regular = fixture.user_token("reader")
                    for token in (regular, helper.agent_credentials[(str(fixture.state), "a")][0]):
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            fixture.request("POST", "/v1/identity/admin", {"action": "metric-set", "changes": {"enabled": True}}, token)
                        self.assertEqual(error.exception.code, 403)
                    for client in binaries:
                        command = [client, "admin", "metric", "set", "--controller-url", url,
                                   "--token-file", str(fixture.admin_file), "--enabled", "on", "--interval-seconds", "2"]
                        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(json.loads(result.stdout)["version"], 1)
                        query = subprocess.run([client, "metric", "latest", "--metric", "sm_active_ratio",
                                                "--controller-url", url, "--token-file", str(fixture.admin_file)],
                                               capture_output=True, text=True, timeout=10)
                        self.assertEqual(query.returncode, 0, query.stderr)
                        self.assertEqual(json.loads(query.stdout)["state"], "unavailable")
                    override = fixture.request("POST", "/v1/identity/admin", {"action": "metric-set", "node_id": "a",
                                               "expected_version": 1, "changes": {"profiling": True}})
                    self.assertEqual(override["version"], 2)
                    self.assertTrue(poll("b")["config"]["profiling"])
                    self.assertFalse(poll("c", "other-service")["config"]["profiling"])
                    poll("a", applied_version=2, state="applied")
                    poll("a", applied_version=0)
                    status = fixture.request("GET", "/v1/metrics/status?node_id=a", token=regular)
                    self.assertEqual(status["receipts"]["a"]["state"], "applied")
                    for bad in ({"changes": {"enabled": 1}}, {"changes": {"interval_seconds": 0}},
                                {"changes": {"unknown": 1}}, {"changes": []}, {"changes": {"basic": False}, "expected_version": True}):
                        with self.assertRaises(urllib.error.HTTPError) as error:
                            fixture.request("POST", "/v1/identity/admin", dict(action="metric-set", **bad))
                        self.assertEqual(error.exception.code, 400)
                    helper.stop_controller(process)
                    other = binaries[1] if server == binaries[0] else binaries[0]
                    process, url, _ = helper.start_controller(other, root)
                    fixture = helper.fixtures[url]
                    recovered = fixture.request("GET", "/v1/metrics/status?node_id=a")
                    self.assertEqual(recovered, status)
                    unchanged = fixture.request("POST", "/v1/identity/admin", {"action": "metric-set", "changes": {"interval_seconds": 2}})
                    self.assertEqual(unchanged["version"], 2)
                    self.assertTrue(poll("a")["config"]["profiling"])
                finally:
                    helper.stop_controller(process)


if __name__ == "__main__":
    unittest.main()
