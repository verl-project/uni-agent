# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import uuid

from pycape.cli import _request, CapeHTTPError
from pycape.credential_file import read_registration
from pycape.identity import new_secret, verifier


class IdentityInteropTests(unittest.TestCase):
    def test_persistent_cli_http_matrix(self):
        python = os.environ.get("PYCAPE_BIN", "pycape")
        rust = os.environ.get("RSCAPE_BIN", "rscape")
        for server in (python, rust):
            for client in (python, rust):
                with self.subTest(server=server, client=client), tempfile.TemporaryDirectory() as directory:
                    self.scenario(server, client, directory)

    def command(self, binary, *args):
        env = dict(os.environ)
        for key in ("CAPE_TOKEN", "CAPE_ADMIN_TOKEN", "CAPE_JOIN_TOKEN"):
            env.pop(key, None)
        result = subprocess.run([binary, *args], text=True, capture_output=True, timeout=20, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def scenario(self, server, client, directory):
        state = os.path.join(directory, "state")
        admin_file = os.path.join(directory, "admin.json")
        initial = self.command(client, "controller", "init", "--state-dir", state, "--user", "admin", "--token-out", admin_file)
        admin = read_registration(admin_file)["secret"]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = "http://127.0.0.1:{}".format(port)
        process = subprocess.Popen([server, "controller", "start", "--state-dir", state, "--token-file", admin_file,
                                    "--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    self.fail("controller exited: " + process.communicate()[1])
                try:
                    _request("GET", url, "/health", timeout=1)
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        self.fail("controller did not become healthy")
                    time.sleep(0.05)
            pool = self.command(client, "admin", "pool", "create", "pool", "--token-file", admin_file,
                                "--controller-url", url)["pool_id"]
            file = os.path.join(directory, "alice.json")
            registered = self.command(client, "user", "token", "create", "--user", "alice", "--token-out", file, "--controller-url", url)
            recovered = self.command(client, "user", "token", "resume", "--token-out", file)
            self.assertEqual(registered, recovered)
            actor = self.command(client, "user", "whoami", "--token-file", file, "--controller-url", url)
            self.assertEqual(actor["scheduler_id"], initial["scheduler_id"])
            targeted = {"operation_id": str(uuid.uuid4()), "verifier": verifier(new_secret()), "name": "bound", "username": None,
                        "expires_at": None, "user_id": actor["user_id"]}
            with self.assertRaises(CapeHTTPError) as mismatch:
                _request("POST", url, "/v1/identity/tokens", token=admin, payload=targeted)
            self.assertEqual(mismatch.exception.status, 403)
            self.assertEqual(_request("POST", url, "/v1/identity/tokens", token=read_registration(file)["secret"], payload=targeted)["user_id"], actor["user_id"])
            pools = self.command(client, "user", "pools", "--token-file", file, "--controller-url", url)
            self.assertEqual(pools, [{"pool_id": pool, "name": "pool", "state": "unavailable",
                                     "config": {"pool_mode": "exclusive", "gpu_slots": 1, "placement": "pack"}, "max_nodes": None}])
            token = read_registration(file)["secret"]
            request = _request("POST", url, "/v1/submit", token=token,
                               payload={"pool": pool, "session_id": "t", "command": ["true"]})
            self.command(client, "admin", "pool", "group", "set", "default", "--token-file", admin_file, "--controller-url", url)
            for _ in range(100):
                status = _request("GET", url, "/v1/requests/" + request["request_id"], token=token)
                if status["state"] == "FAILED":
                    break
                time.sleep(0.02)
            self.assertEqual(status["reason"], "permission_revoked")
            self.assertEqual(status["state"], "FAILED")
            self.command(client, "admin", "pool", "group", "set", "access", "--pool", "pool", "--token-file", admin_file, "--controller-url", url)
            self.command(client, "admin", "user", "edit", actor["user_id"], "--group", "access", "--token-file", admin_file, "--controller-url", url)
            permission = self.command(client, "user", "permissions", "--token-file", file, "--controller-url", url)
            self.assertEqual(permission["groups"], ["access"])
            self.command(client, "admin", "pool", "group", "remove", "access", "--token-file", admin_file, "--controller-url", url)
            permission = self.command(client, "user", "permissions", "--token-file", file, "--controller-url", url)
            self.assertEqual(permission["groups"], [])
            self.command(client, "admin", "pool", "edit", "pool", "--max-nodes", "1", "--token-file", admin_file, "--controller-url", url)
            agent = new_secret()
            enrollment = {"operation_id": str(uuid.uuid4()), "verifier": verifier(agent), "node_id": "node-a", "pool": pool,
                          "allocation_id": "allocation-a", "agent_epoch": "epoch-a", "owner_user_id": actor["user_id"],
                          "resources": {"cpu_cores": 1, "memory_gb": 1, "gpu_uuids": []}}
            with self.assertRaises(CapeHTTPError) as mismatch:
                _request("POST", url, "/v1/identity/nodes/enroll", token=admin, payload=enrollment)
            self.assertEqual(mismatch.exception.status, 403)
            _request("POST", url, "/v1/identity/nodes/enroll", token=token, payload=enrollment)
            self.command(client, "node", "edit", "node-a", "--cpu-cores", "0.5", "--token-file", file, "--controller-url", url)
            announcement = {key: enrollment[key] for key in ("node_id", "pool", "allocation_id", "agent_epoch", "resources")}
            announced = _request("POST", url, "/v1/agent/announce", token=agent, payload=announcement)
            self.assertEqual(announced["resources"]["cpu_cores"], 0.5)
            public_nodes = self.command(client, "user", "nodes", "--pool", "pool", "--token-file", file, "--controller-url", url)
            self.assertEqual(set(public_nodes[0]), {"node_id", "pool_id", "resources", "state", "gpu_metrics"})
            utilization = self.command(client, "utilization", "--pool", "pool", "--token-file", file, "--controller-url", url)
            self.assertNotIn("active_sessions", utilization["node_states"][0])
            drained = self.command(client, "node", "drain", "node-a", "--token-file", file, "--controller-url", url)
            self.assertEqual(drained["state"], "DRAINING")
            pulse = {"node_id": "node-a", "agent_epoch": "epoch-a", "status": {"authoritative": True, "sessions": []}}
            for _ in range(100):
                response = _request("POST", url, "/v1/agent/heartbeat", token=agent, payload=pulse)
                if response.get("agent_control", {}).get("action") == "shutdown":
                    break
                time.sleep(0.02)
            self.assertEqual(response.get("agent_control", {}).get("action"), "shutdown")
            events = self.command(client, "admin", "audit", "--token-file", admin_file, "--controller-url", url)
            self.assertIn("node.retired", [event["action"] for event in events])
            for secret in (admin, token, agent, verifier(agent)):
                self.assertNotIn(secret, json.dumps(events))
            second_file = os.path.join(directory, "alice-second.json")
            second = self.command(client, "user", "token", "create", "--token-file", file, "--token-out", second_file, "--controller-url", url)
            self.assertEqual(second["user_id"], actor["user_id"])
            self.command(client, "user", "token", "revoke", registered["token_id"], "--token-file", second_file, "--controller-url", url)
            denied = subprocess.run([client, "user", "whoami", "--token-file", file, "--controller-url", url], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(denied.returncode, 0)
            self.assertNotIn(token, denied.stdout + denied.stderr)
            self.command(client, "admin", "pool", "remove", "pool", "--token-file", admin_file, "--controller-url", url)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
