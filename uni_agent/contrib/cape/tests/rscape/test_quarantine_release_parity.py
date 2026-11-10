# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both CLIs/controllers, durable multi-source release and ordinary lease exclusion."""

import json
import subprocess
import tempfile
import time
import unittest
import urllib.error
import uuid

from pycape.identity import new_secret, verifier
from tests.rscape import test_functional_parity as parity


class QuarantineReleaseParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_sources_release_without_reenrollment_and_old_agents_stay_revoked(self):
        binaries = (parity.PYCAPE_BIN, parity.RSCAPE_BIN)
        gpu = "GPU-00000000-0000-0000-0000-000000000001"
        for server in binaries:
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                helper = parity.FunctionalParityTests()
                process, url, _ = helper.start_controller(server, directory,
                    "--heartbeat-timeout-seconds", "3", "--lost-timeout-seconds", "5")
                fixture = helper.fixtures[url]
                receipt_a = None

                def release_body(node, reason="confirmed old processes stopped"):
                    return {"action": "node-release-quarantine", "node_id": node,
                            "reason": reason, "confirm_stopped": True}

                def pulse():
                    helper.request("POST", url, "/v1/agent/heartbeat", {
                        "node_id": "replacement", "status": {"authoritative": False}, "max_commands": 0})

                def reject_old_agents():
                    for node in ("old-a", "old-b"):
                        secret, epoch = helper.agent_credentials[(str(fixture.state), node)]
                        with self.assertRaises(urllib.error.HTTPError) as rejected:
                            fixture.request("POST", "/v1/agent/heartbeat", {"node_id": node, "agent_epoch": epoch}, secret)
                        self.assertEqual(rejected.exception.code, 403)

                def release_cli(client, node, confirmation=True):
                    args = [client, "admin", "node", "release-quarantine", node, "--reason", "CLI confirmation",
                            "--controller-url", url, "--token-file", str(fixture.admin_file)]
                    if confirmation:
                        args.append("--confirm-stopped")
                    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
                    if not confirmation:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("confirm-stopped", result.stderr)
                        return None
                    self.assertEqual(result.returncode, 0, result.stderr)
                    return json.loads(result.stdout)

                try:
                    for node in ("old-a", "old-b"):
                        helper.register(url, node, "pool", gpus=(gpu,))
                    with self.assertRaises(urllib.error.HTTPError) as not_purged:
                        fixture.request("POST", "/v1/identity/admin", release_body("old-a"))
                    self.assertEqual(not_purged.exception.code, 400)
                    deadline = time.monotonic() + 12
                    while time.monotonic() < deadline:
                        if all(n["state"] == "LOST" for n in fixture.request("GET", "/v1/identity/nodes")):
                            break
                        time.sleep(.05)
                    else:
                        self.fail("protocol agents did not become LOST")
                    original_purges = {}
                    for node in ("old-a", "old-b"):
                        original_purges[node] = fixture.request("POST", "/v1/identity/admin", {
                            "action": "node-purge", "node_id": node, "reason": "allocation ended"})
                    fixture.request("POST", "/v1/identity/admin", {"action": "pool-remove", "pool": fixture.pool("pool")})
                    owner = fixture.user_token("contributor")
                    regular_admin = new_secret()
                    actor = fixture.request("GET", "/v1/identity/me")
                    fixture.request("POST", "/v1/identity/tokens", {"operation_id": str(uuid.uuid4()),
                        "verifier": verifier(regular_admin), "name": "regular", "username": None,
                        "expires_at": None, "user_id": actor["user_id"]})
                    old_agent = helper.agent_credentials[(str(fixture.state), "old-a")][0]
                    for token in (owner, regular_admin, old_agent, "invalid"):
                        with self.assertRaises(urllib.error.HTTPError) as denied:
                            fixture.request("POST", "/v1/identity/admin", release_body("old-a"), token)
                        self.assertEqual(denied.exception.code, 403)
                    for changes in ({"confirm_stopped": False}, {"confirm_stopped": "true"}, {"confirm_stopped": 1},
                                    {"reason": " "}, {"reason": "界"*342}, {"force": True}, {"node_id": "missing"}):
                        with self.assertRaises(urllib.error.HTTPError) as invalid:
                            fixture.request("POST", "/v1/identity/admin", dict(release_body("old-a"), **changes))
                        self.assertEqual(invalid.exception.code, 400)
                    missing_flag = release_body("old-a")
                    del missing_flag["confirm_stopped"]
                    with self.assertRaises(urllib.error.HTTPError) as invalid:
                        fixture.request("POST", "/v1/identity/admin", missing_flag)
                    self.assertEqual(invalid.exception.code, 400)
                    helper.register(url, "replacement", "replacement-pool", gpus=(gpu,))
                    pending = helper.submit(url, "replacement-pool", "alice", "pending", gpus=1)
                    self.assertEqual(helper.status(url, pending["request_id"])["state"], "QUEUED")
                    for client in binaries:
                        pulse()
                        release_cli(client, "old-a", confirmation=False)
                        result = release_cli(client, "old-a")
                        self.assertEqual(result["released_source_gpu_uuids"], [gpu])
                        self.assertEqual(result["still_quarantined_gpu_uuids"], [gpu])
                        self.assertEqual(result["unquarantined_gpu_uuids"], [])
                        self.assertFalse(result["cleanup_confirmed"])
                        if receipt_a is not None:
                            self.assertEqual(result["release_receipt"], receipt_a)
                        receipt_a = result["release_receipt"]
                        self.assertEqual(helper.status(url, pending["request_id"])["state"], "QUEUED")
                    reject_old_agents()
                    process.kill()
                    process.wait(timeout=5)
                finally:
                    helper.stop_controller(process)

                for reader in reversed(binaries):
                    process, url, _ = helper.start_controller(reader, directory)
                    try:
                        helper.register(url, "replacement", "replacement-pool", gpus=(gpu,))
                        result = fixture.request("POST", "/v1/identity/admin", release_body("old-a", "retry"))
                        self.assertEqual(result["release_receipt"], receipt_a)
                        self.assertEqual(result["still_quarantined_gpu_uuids"], [gpu])
                        self.assertEqual(helper.status(url, pending["request_id"])["state"], "QUEUED")
                        reject_old_agents()
                    finally:
                        helper.stop_controller(process)

                process, url, _ = helper.start_controller(server, directory)
                try:
                    helper.register(url, "replacement", "replacement-pool", gpus=(gpu,))
                    receipt_b = None
                    for client in binaries:
                        result = release_cli(client, "old-b")
                        self.assertEqual(result["still_quarantined_gpu_uuids"], [])
                        self.assertEqual(result["unquarantined_gpu_uuids"], [gpu])
                        if receipt_b is not None:
                            self.assertEqual(result["release_receipt"], receipt_b)
                        receipt_b = result["release_receipt"]
                    running = helper.wait_state(url, pending["request_id"], "RUNNING")
                    self.assertEqual(running["node_id"], "replacement")
                    second = helper.submit(url, "replacement-pool", "alice", "second", gpus=1)
                    self.assertEqual(helper.status(url, second["request_id"])["state"], "QUEUED")
                    result = fixture.request("POST", "/v1/identity/admin", release_body("old-a", "repeat"))
                    self.assertEqual(result["unquarantined_gpu_uuids"], [gpu])
                    self.assertEqual(result["release_receipt"], receipt_a)
                    self.assertEqual(helper.status(url, second["request_id"])["state"], "QUEUED")
                    helper.finish(url, pending["request_id"])
                    helper.wait_state(url, second["request_id"], "RUNNING")
                    helper.finish(url, second["request_id"])
                    for node in ("old-a", "old-b"):
                        self.assertEqual(fixture.request("POST", "/v1/identity/admin", {
                            "action": "node-purge", "node_id": node, "reason": "retry"}), original_purges[node])
                    reject_old_agents()
                    audit = fixture.request("POST", "/v1/identity/admin", {"action": "audit", "limit": 1000})
                    self.assertEqual(sum(e["action"] == "node.release-quarantine" for e in audit), 2)
                    self.assertEqual(sum(e["action"] == "node.enroll" for e in audit), 3)
                finally:
                    helper.stop_controller(process)
                reader = binaries[1] if server == binaries[0] else binaries[0]
                process, url, _ = helper.start_controller(reader, directory)
                try:
                    helper.register(url, "replacement", "replacement-pool", gpus=(gpu,))
                    third = helper.submit(url, "replacement-pool", "alice", "after-restart", gpus=1)
                    helper.wait_state(url, third["request_id"], "RUNNING")
                    helper.finish(url, third["request_id"])
                    reject_old_agents()
                finally:
                    helper.stop_controller(process)
