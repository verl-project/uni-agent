# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public logical session and interoperable last-session cache, without GPU/FUSE."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import urllib.error
import time

from tests.rscape import test_functional_parity as parity


class LogicalSessionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_cli_cache_owner_scope_and_controller_restart(self):
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                helper = parity.FunctionalParityTests()
                process, url, _ = helper.start_controller(server, directory)
                try:
                    fixture = helper.fixtures[url]
                    fixture.pool("one")
                    fixture.pool("two")
                    alice = fixture.user_file("alice")
                    bob = fixture.user_file("bob")
                    environment = dict(os.environ, CAPE_SESSION_STATE_DIR=str(Path(directory) / "client-state"))
                    environment.pop("CAPE_TOKEN", None)

                    def run(client, credential, pool, explicit=None):
                        args = [client, "run", "--controller-url", url, "--token-file", str(credential), "--pool", pool]
                        if explicit is not None:
                            args += ["--session-id", explicit]
                        result = subprocess.run(args + ["--", "true"], cwd=directory, env=environment,
                                                capture_output=True, text=True, timeout=20)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        token = fixture.user_token("alice" if credential == alice else "bob")
                        return fixture.request("GET", "/v1/requests/" + result.stdout.strip(), token=token)

                    first = run(parity.PYCAPE_BIN, alice, "one", "sequence")
                    second = run(parity.RSCAPE_BIN, alice, "two")
                    self.assertEqual(first["session_id"], second["session_id"])
                    self.assertEqual(first["session_id"], "sequence")
                    other = run(parity.RSCAPE_BIN, bob, "one")
                    self.assertNotEqual(other["session_id"], "sequence")
                    result = subprocess.run([parity.RSCAPE_BIN, "session", "list", "--controller-url", url,
                                             "--token-file", str(alice)], env=environment,
                                            capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual([s["session_id"] for s in json.loads(result.stdout)["sessions"]], ["sequence"])
                    helper.stop_controller(process)
                    process = fixture.start(server)
                    after = run(parity.PYCAPE_BIN, alice, "one")
                    self.assertEqual(after["session_id"], "sequence")
                    for flag in ("--task-id", "--session-key"):
                        for client in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
                            result = subprocess.run([client, "run", "--pool", "one", flag, "obsolete", "--", "true"],
                                                    env=environment, capture_output=True, text=True, timeout=10)
                            self.assertNotEqual(result.returncode, 0)
                finally:
                    helper.stop_controller(process)

    def test_force_suspend_before_agent_dispatch_does_not_replay_cancelled_command(self):
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                helper = parity.FunctionalParityTests()
                process, url, _ = helper.start_controller(server, directory)
                try:
                    fixture = helper.fixtures[url]
                    helper.register(url, "node-a", "one", gpus=())
                    owner = fixture.user_token("alice")

                    def submit(command):
                        submitted = helper.request("POST", url, "/v1/submit", {
                            "pool": "one", "user_id": "alice", "session_id": "sequence",
                            "command": [command], "resources": {"gpus": 0,
                            "runtime_adapter": "local-process", "isolation_policy": "none"}})
                        return helper.wait_state(url, submitted["request_id"], "RUNNING", timeout=5)

                    first = submit("old-command-must-not-run")
                    execution = first["execution_session_id"]
                    suspended = fixture.request("POST", "/v1/executions/" + execution + "/suspend",
                                                {"force": True}, owner)
                    self.assertEqual(suspended["state"], "SUSPENDED")
                    helper.wait_state(url, first["request_id"], "CANCELLED", timeout=5)
                    second = submit("new-command")
                    self.assertEqual(second["execution_session_id"], execution)
                    self.assertEqual(second["workspace"]["id"], first["workspace"]["id"])
                    response = helper.request("POST", url, "/v1/agent/heartbeat", {
                        "node_id": "node-a", "status": {"authoritative": True, "sessions": []}})
                    encoded = json.dumps(response)
                    self.assertIn("new-command", encoded)
                    self.assertNotIn("old-command-must-not-run", encoded)
                finally:
                    helper.stop_controller(process)

    def test_independent_pools_and_owner_scoped_suspend_resume_purge(self):
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                helper = parity.FunctionalParityTests()
                process, url, _ = helper.start_controller(server, directory)
                try:
                    fixture = helper.fixtures[url]
                    helper.register(url, "node-a", "one", gpus=())
                    helper.register(url, "node-b", "two", gpus=())
                    alice, bob = fixture.user_token("alice"), fixture.user_token("bob")

                    def submit(pool):
                        value = helper.request("POST", url, "/v1/submit", {
                            "pool": pool, "user_id": "alice", "session_id": "sequence", "command": ["true"],
                            "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}})
                        return helper.wait_state(url, value["request_id"], "RUNNING", timeout=5)

                    first, second = submit("one"), submit("two")
                    execution = first["execution_session_id"]
                    self.assertNotEqual(execution, second["execution_session_id"])
                    self.assertEqual(first["session_id"], second["session_id"])
                    workspace = first["workspace"]
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a", "status": {"authoritative": True, "sessions": []}})
                    helper.request("POST", url, "/v1/agent/report-command", {"node_id": "node-a", "command_id": first["command_id"], "exit_code": 0})
                    with self.assertRaises(urllib.error.HTTPError) as denied:
                        fixture.request("POST", "/v1/executions/" + execution + "/suspend", {}, bob)
                    self.assertEqual(denied.exception.code, 403)
                    pending = fixture.request("POST", "/v1/executions/" + execution + "/suspend", {}, alice)
                    self.assertEqual(pending["state"], "SUSPENDING")
                    version = pending["suspend_version"]
                    update = {"workspace_id": workspace["id"], "execution_session_id": execution,
                              "user_id": workspace["user_id"], "config": workspace["config"],
                              "revision": 1, "state": "retained", "path": "/tmp/simulated-workspace"}
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a", "status": {
                        "authoritative": True, "sessions": [], "command_ids": [],
                        "suspended_sessions": {execution: version}, "workspace_updates": [update]}})

                    def execution_state():
                        result = fixture.request("GET", "/v1/sessions?session_id=sequence", token=alice)
                        return next(s for s in result["sessions"][0]["executions"] if s["execution_session_id"] == execution)

                    self.assertEqual(execution_state()["state"], "SUSPENDED")
                    third = submit("one")
                    self.assertEqual(third["execution_session_id"], execution)
                    self.assertEqual(third["workspace"]["id"], workspace["id"])
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a", "status": {"authoritative": True, "sessions": []}})
                    helper.request("POST", url, "/v1/agent/report-command", {"node_id": "node-a", "command_id": third["command_id"], "exit_code": 0})
                    pending = fixture.request("POST", "/v1/executions/" + execution + "/suspend", {}, alice)
                    update["revision"] = 2
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a", "status": {
                        "authoritative": True, "sessions": [], "suspended_sessions": {execution: pending["suspend_version"]},
                        "workspace_updates": [update]}})
                    fixture.request("POST", "/v1/executions/" + execution + "/purge", {}, alice)
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a", "status": {
                        "authoritative": True, "sessions": [], "purged_executions": [execution]}})
                    self.assertEqual(execution_state()["state"], "PURGED")
                    self.assertEqual(fixture.request("GET", "/v1/workspaces/" + workspace["id"], token=alice)["state"], "retained")
                finally:
                    helper.stop_controller(process)

    def test_retained_resume_obeys_pool_and_gpu_limits(self):
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            for flag, gpus, reason in (("--max-active-sessions-per-pool", 0, "pool_session_limit"),
                                      ("--max-active-gpu-sessions-per-user", 1, "quota_exceeded")):
                with self.subTest(server=server, limit=flag), tempfile.TemporaryDirectory() as directory:
                    helper = parity.FunctionalParityTests()
                    process, url, _ = helper.start_controller(server, directory, flag, "1")
                    try:
                        fixture = helper.fixtures[url]
                        helper.register(url, "node-a", "one", gpus=(
                            "GPU-00000000-0000-0000-0000-000000000001",
                            "GPU-00000000-0000-0000-0000-000000000002"))
                        owner = fixture.user_token("alice")

                        def submit(sequence):
                            return helper.request("POST", url, "/v1/submit", {
                                "pool": "one", "user_id": "alice", "session_id": sequence,
                                "command": ["true"], "resources": {"gpus": gpus,
                                "runtime_adapter": "local-process", "isolation_policy": "none"}})["request_id"]

                        first = helper.wait_state(url, submit("retained"), "RUNNING", timeout=5)
                        fixture.request("POST", "/v1/executions/" + first["execution_session_id"] + "/suspend",
                                        {"force": True}, owner)
                        helper.wait_state(url, submit("occupant"), "RUNNING", timeout=5)
                        resumed = submit("retained")
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            status = fixture.request("GET", "/v1/requests/" + resumed, token=owner)
                            if status["reason"] == reason:
                                break
                            time.sleep(.02)
                        self.assertEqual(status["state"], "QUEUED")
                        self.assertEqual(status["reason"], reason)
                    finally:
                        helper.stop_controller(process)

    def test_retained_resume_reclaims_idle_capacity_after_acknowledgment(self):
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                helper = parity.FunctionalParityTests()
                process, url, _ = helper.start_controller(server, directory)
                try:
                    fixture = helper.fixtures[url]
                    helper.register(url, "node-a", "one", gpus=())
                    owner = fixture.user_token("alice")

                    def submit(sequence):
                        return helper.request("POST", url, "/v1/submit", {
                            "pool": "one", "user_id": "alice", "session_id": sequence,
                            "command": ["true"], "resources": {"gpus": 0, "cpu_cores": 16,
                            "runtime_adapter": "local-process", "isolation_policy": "none"}})["request_id"]

                    first = helper.wait_state(url, submit("retained"), "RUNNING", timeout=5)
                    execution = first["execution_session_id"]
                    fixture.request("POST", "/v1/executions/" + execution + "/suspend", {"force": True}, owner)
                    other = helper.wait_state(url, submit("occupant"), "RUNNING", timeout=5)
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a"})
                    helper.request("POST", url, "/v1/agent/report-command", {
                        "node_id": "node-a", "command_id": other["command_id"], "exit_code": 0})
                    resumed = submit("retained")
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        status = fixture.request("GET", "/v1/sessions?session_id=occupant", token=owner)
                        occupant = status["sessions"][0]["executions"][0]
                        if occupant["state"] == "SUSPENDING":
                            break
                        time.sleep(.02)
                    self.assertEqual(occupant["state"], "SUSPENDING")
                    self.assertEqual(fixture.request("GET", "/v1/requests/" + resumed, token=owner)["state"], "QUEUED")
                    workspace = other["workspace"]
                    helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a", "status": {
                        "authoritative": True, "sessions": [], "command_ids": [],
                        "suspended_sessions": {other["execution_session_id"]: occupant["suspend_version"]},
                        "workspace_updates": [{"workspace_id": workspace["id"],
                            "execution_session_id": other["execution_session_id"], "user_id": workspace["user_id"],
                            "config": workspace["config"], "revision": 1, "state": "retained", "path": "/tmp/test-retained"}]}})
                    result = helper.wait_state(url, resumed, "RUNNING", timeout=5)
                    self.assertEqual(result["execution_session_id"], execution)
                finally:
                    helper.stop_controller(process)


if __name__ == "__main__":
    unittest.main()
