# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real CLI/HTTP persistence matrix with protocol agents; no remote process kills."""

import json
import subprocess
import tempfile
import time
import unittest
import urllib.error
import uuid

from pycape.identity import new_secret, verifier
from tests.rscape import test_functional_parity as parity


class NodePurgeParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_lost_purge_cli_auth_history_fencing_and_cross_backend_restart(self):
        binaries = (parity.PYCAPE_BIN, parity.RSCAPE_BIN)
        gpu = "GPU-00000000-0000-0000-0000-000000000001"
        for server in binaries:
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                helper = parity.FunctionalParityTests()
                process, url, _ = helper.start_controller(server, directory,
                    "--heartbeat-timeout-seconds", "3", "--lost-timeout-seconds", "5")
                fixture = helper.fixtures[url]
                try:
                    helper.register(url, "retiring", "pool", gpus=(gpu,))
                    pool = fixture.pool("pool")
                    agent, epoch = helper.agent_credentials[(str(fixture.state), "retiring")]
                    owner = fixture.user_token("contributor")
                    regular = new_secret()
                    admin_id = fixture.request("GET", "/v1/identity/me")["user_id"]
                    fixture.request("POST", "/v1/identity/tokens", {"operation_id": str(uuid.uuid4()),
                        "verifier": verifier(regular), "name": "regular", "username": None,
                        "expires_at": None, "user_id": admin_id})
                    payload = {"action": "node-purge", "node_id": "retiring", "reason": "allocation ended"}
                    with self.assertRaises(urllib.error.HTTPError) as active:
                        fixture.request("POST", "/v1/identity/admin", payload)
                    self.assertEqual(active.exception.code, 400)
                    for credential in (owner, regular, agent, "invalid"):
                        with self.subTest(credential_kind=credential == regular), self.assertRaises(urllib.error.HTTPError) as denied:
                            fixture.request("POST", "/v1/identity/admin", payload, credential)
                        self.assertEqual(denied.exception.code, 403)
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "retiring", "status": {"authoritative": False}})
                    completed = helper.submit(url, "pool", "alice", "complete", gpus=0)
                    helper.wait_state(url, completed["request_id"], "RUNNING")
                    helper.finish(url, completed["request_id"])
                    completed_before = helper.status(url, completed["request_id"])
                    request = helper.submit(url, "pool", "alice", "lost", gpus=1)
                    running = helper.wait_state(url, request["request_id"], "RUNNING")
                    workspace = running["workspace"]["id"]
                    lost = helper.wait_state(url, request["request_id"], "LOST", timeout=10)
                    self.assertEqual(lost["reason"], "node_lost")
                    workspace_before = fixture.request("GET", "/v1/workspaces/" + workspace)
                    for changes in ({"reason": " "}, {"reason": "x"*1025}, {"reason": None},
                                    {"force": True}, {"node_id": "missing"}):
                        with self.assertRaises(urllib.error.HTTPError) as invalid:
                            fixture.request("POST", "/v1/identity/admin", dict(payload, **changes))
                        self.assertEqual(invalid.exception.code, 400)
                    first = None
                    for client in binaries:
                        result = subprocess.run([client, "admin", "node", "purge", "retiring", "--reason", "allocation ended",
                            "--controller-url", url, "--token-file", str(fixture.admin_file)], capture_output=True, text=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        record = json.loads(result.stdout)
                        self.assertEqual(record["state"], "REMOVED")
                        self.assertFalse(record["cleanup_confirmed"])
                        self.assertEqual(record["quarantined_gpu_uuids"], [gpu])
                        if first is not None:
                            self.assertEqual(record, first)
                        first = record
                    self.assertEqual(fixture.request("GET", "/v1/identity/nodes"), [])
                    utilization = fixture.request("GET", "/v1/pools/" + pool + "/utilization")
                    self.assertEqual(utilization["nodes"], 0)
                    self.assertEqual(utilization["gpus_total"], 0)
                    self.assertEqual(utilization["physical"]["gpus"], [])
                    for identity, before in ((request["request_id"], lost), (completed["request_id"], completed_before)):
                        after_status = helper.status(url, identity)
                        for field in ("state", "reason", "failure_kind", "exit_code", "ended_at", "updated_at"):
                            self.assertEqual(after_status[field], before[field])
                    after = fixture.request("GET", "/v1/workspaces/" + workspace)
                    self.assertEqual(after["path"], workspace_before["path"])
                    self.assertTrue(after["node_retired"])
                    self.assertFalse(after["cleanup_confirmed"])
                    unavailable = helper.request("POST", url, "/v1/submit", {"pool": "pool", "user_id": "alice",
                        "session_id": "unavailable", "command": ["true"], "resources": {"gpus": 0},
                        "workspace": {"id": workspace}})
                    failed = helper.wait_state(url, unavailable["request_id"], "FAILED")
                    self.assertEqual(failed["reason"], "workspace_node_unavailable")
                    for route, body in (("heartbeat", {"node_id": "retiring", "agent_epoch": epoch, "status": {}}),
                                        ("report-command", {"node_id": "retiring", "agent_epoch": epoch,
                                                            "command_id": running["command_id"], "exit_code": 0}),
                                        ("register", {"node_id": "retiring", "agent_epoch": epoch, "pool": pool,
                                                      "allocation_id": "allocation-retiring",
                                                      "resources": {"cpu_cores": 16, "memory_gb": 128, "gpu_uuids": [gpu]}})):
                        with self.assertRaises(urllib.error.HTTPError) as rejected:
                            fixture.request("POST", "/v1/agent/" + route, body, agent)
                        self.assertEqual(rejected.exception.code, 403)
                    # A fresh enrollment in another pool must not bypass the physical fence.
                    helper.register(url, "replacement", "other", gpus=(gpu,))
                    blocked = helper.submit(url, "other", "alice", "blocked", gpus=1)
                    self.assertEqual(helper.status(url, blocked["request_id"])["state"], "QUEUED")
                    other = fixture.pool("other")
                    util = fixture.request("GET", "/v1/pools/" + other + "/utilization")
                    self.assertEqual(util["gpus_available"], 0)
                    self.assertEqual(util["gpu_slots_available"], 0)
                    fixture.request("POST", "/v1/identity/admin", {"action": "pool-remove", "pool": pool})
                    process.kill()
                    process.wait(timeout=5)
                finally:
                    helper.stop_controller(process)
                for reader in reversed(binaries):
                    process, url, _ = helper.start_controller(reader, directory)
                    try:
                        self.assertEqual(fixture.request("POST", "/v1/identity/admin", payload), first)
                        self.assertFalse(any(n["node_id"] == "retiring" for n in fixture.request("GET", "/v1/identity/nodes")))
                        with self.assertRaises(urllib.error.HTTPError):
                            fixture.request("POST", "/v1/agent/heartbeat", {"node_id": "retiring", "agent_epoch": epoch}, agent)
                        helper.register(url, "replacement", "other", gpus=(gpu,))
                        self.assertEqual(helper.status(url, blocked["request_id"])["state"], "QUEUED")
                        util = fixture.request("GET", "/v1/pools/" + other + "/utilization")
                        self.assertEqual(util["gpus_available"], 0)
                        audit = fixture.request("POST", "/v1/identity/admin", {"action": "audit", "limit": 1000})
                        self.assertEqual(sum(e.get("action") == "node.purge" for e in audit), 1)
                    finally:
                        helper.stop_controller(process)
