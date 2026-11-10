# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import socket
import sqlite3
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from pycape import CAPEController, CAPEControllerService
from pycape.workspace import WorkspaceConfig
from pycape.identity import new_secret, verifier
from tests.managed_fixture import ManagedFixture, simulated_gpu_environment


PYCAPE_BIN = os.environ.get("PYCAPE_BIN", "pycape")
RSCAPE_BIN = os.environ.get("RSCAPE_BIN", "rscape")


class FunctionalParityTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            if not (Path(binary).is_file() and os.access(binary, os.X_OK)):
                raise unittest.SkipTest("functional parity binary is unavailable: " + binary)

    def free_port(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def request(self, method, url, path, payload=None, token=None):
        fixture = getattr(self, "fixtures", {}).get(url)
        if fixture is not None:
            if path == "/v1/submit":
                fixture.pool(payload["pool"])
                token = token or fixture.user_token(payload.get("user_id") or "alice")
            elif path == "/v1/admin/workspace-config" and method == "POST":
                fixture.pool(payload["pool"])
            elif path.startswith("/v1/agent/"):
                payload = dict(payload)
                credential, epoch = self.agent_credentials[(str(fixture.state), payload["node_id"])]
                token = token or credential
                payload.setdefault("agent_epoch", epoch)
            return fixture.request(method, path, payload, token)
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    def start_controller(self, binary, directory, *extra, journal_name="journal.sqlite"):
        root = Path(directory) / (journal_name + ".instance")
        fixture = next((f for f in getattr(self, "fixtures", {}).values() if f.root == root), None)
        if fixture is None:
            fixture = ManagedFixture(binary, root)
        process = fixture.start(binary, "--schedule-interval", "0.01", *extra)
        if not hasattr(self, "fixtures"):
            self.fixtures = {}
            self.agent_credentials = {}
        self.fixtures[fixture.url] = fixture
        return process, fixture.url, fixture.state / "state.sqlite3"

    def node_credentials(self, url, node, pool):
        return self.fixtures[url].node_credentials(node, pool)

    def stop_controller(self, process):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def marker_pids(self, marker):
        matches = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes()
            except (OSError, IOError):
                continue
            if marker.encode() in command:
                matches.append(int(entry.name))
        return matches

    def register(self, url, node_id, pool, mode="exclusive", slots=1, gpus=("GPU-00000000-0000-0000-0000-000000000001",)):
        fixture = self.fixtures[url]
        pool_id = fixture.pool(pool, mode, slots)
        key = (str(fixture.state), node_id)
        if key not in self.agent_credentials:
            credential, epoch = new_secret(), uuid.uuid4().hex
            fixture.request("POST", "/v1/identity/nodes/enroll", {
                "operation_id": str(uuid.uuid4()), "verifier": verifier(credential),
                "node_id": node_id, "pool": pool_id, "allocation_id": "allocation-" + node_id,
                "agent_epoch": epoch, "resources": {"cpu_cores": 16, "memory_gb": 128, "gpu_uuids": list(gpus)},
            }, fixture.user_token("contributor"))
            self.agent_credentials[key] = (credential, epoch)
        return self.request(
            "POST",
            url,
            "/v1/agent/register",
            {
                "node_id": node_id,
                "pool": pool_id,
                "pool_mode": mode,
                "gpu_slots": slots,
                "placement": "pack",
                "resources": {
                    "cpu_cores": 16,
                    "memory_gb": 128,
                    "gpu_uuids": list(gpus),
                },
                "capabilities": {
                    "adapter": "local-process",
                    "isolation_policies": ["default", "none"],
                },
                "runtime_capabilities": {
                    "local-process": {
                        "adapter": "local-process",
                        "image_formats": ["default"],
                        "isolation_policies": ["default", "none"],
                        "gpu_isolation_policies": ["default", "none"],
                        "can_select_gpus": True,
                        "gpu_enforcement": "none",
                    }
                },
                "allocation_id": "allocation-" + node_id,
                "runtime_cache_config": "default",
                "workspace_config": {"state": "applied", "applied_version": 0, "config": WorkspaceConfig().to_dict()},
            },
        )

    def submit(self, url, pool, user, task, profile=None, gpus=None):
        payload = {
            "pool": pool,
            "user_id": user,
            "session_id": task,
            "command": ["false"],
            "resources": {"gpus": (1 if profile is None else 0) if gpus is None else gpus},
        }
        if profile is not None:
            payload["profile"] = profile
        return self.request("POST", url, "/v1/submit", payload)

    def status(self, url, request_id):
        return self.request("GET", url, "/v1/requests/" + request_id)

    def wait_state(self, url, request_id, expected, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.status(url, request_id)
            if status["state"] == expected:
                return status
            time.sleep(0.01)
        self.fail("request {} did not reach {}: {}".format(request_id, expected, status))

    def finish(self, url, request_id):
        status = self.status(url, request_id)
        self.assertIsNotNone(
            status.get("command_id"),
            {"status": status, "utilization": self.request("GET", url, "/v1/pools/{}/utilization".format(status["pool"]))},
        )
        self.assertIsNotNone(status.get("node_id"), status)
        self.request(
            "POST",
            url,
            "/v1/agent/heartbeat",
            {
                "node_id": status["node_id"],
                "status": {"authoritative": True, "sessions": []},
                "max_commands": 8,
            },
        )
        return self.request(
            "POST",
            url,
            "/v1/agent/report-command",
            {
                "node_id": status["node_id"],
                "command_id": status["command_id"],
                "exit_code": 0,
                "failure_kind": "workload",
            },
        )

    def test_explicit_no_reuse_gets_fresh_session(self):
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(binary, tmp)
                try:
                    self.register(url, "node-a", "pool-a")
                    payload = {
                        "pool": "pool-a", "user_id": "alice", "session_id": "reuse",
                        "reuse": True,
                        "command": ["true"], "resources": {"gpus": 1},
                    }
                    first = self.request("POST", url, "/v1/submit", payload)
                    original = self.wait_state(url, first["request_id"], "RUNNING")
                    self.finish(url, first["request_id"])
                    self.wait_state(url, first["request_id"], "COMPLETED")
                    warm = self.request("POST", url, "/v1/submit", payload)
                    reused = self.wait_state(url, warm["request_id"], "RUNNING")
                    self.assertEqual(reused["execution_session_id"], original["execution_session_id"])
                    self.finish(url, warm["request_id"])
                    self.wait_state(url, warm["request_id"], "COMPLETED")
                    payload["reuse"] = False
                    cold = self.request("POST", url, "/v1/submit", payload)
                    fresh = self.wait_state(url, cold["request_id"], "RUNNING")
                    self.assertNotEqual(fresh["execution_session_id"], original["execution_session_id"])
                    self.finish(url, cold["request_id"])
                    self.wait_state(url, cold["request_id"], "COMPLETED")
                finally:
                    self.stop_controller(process)

    def test_undeliverable_report_releases_lease_and_agent_serves_successor(self):
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(binary, tmp)
                with open(Path(tmp) / "node.log", "w") as output:
                    node = subprocess.Popen([
                        binary, "node", "enroll", *self.node_credentials(url, "report-node", "report-pool"), "--controller-url", url,
                        "--node-id", "report-node", "--pool", "report-pool",
                        "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000003", "--cpu-cores", "4",
                        "--memory-gb", "8", "--runtime-adapter", "local-process",
                        "--heartbeat-interval", "0.05", "--state-path", str(Path(tmp) / "state.json"),
                        "--log-dir", str(Path(tmp) / "logs"),
                    ], stdout=output, stderr=output, env=simulated_gpu_environment(tmp, ["GPU-00000000-0000-0000-0000-000000000003"]))
                    try:
                        failed_id = None
                        for index, size in enumerate((100 * 1024, 2 * 1024 * 1024, 32)):
                            submitted = self.request("POST", url, "/v1/submit", {
                                "pool": "report-pool", "user_id": "alice", "session_id": "output-" + str(index),
                                "reuse": False, "command": ["/bin/sh", "-c", "head -c {} /dev/zero | tr '\\0' x".format(size)],
                                "resources": {"gpus": 1, "cpu_cores": 1, "memory_gb": 1,
                                              "max_duration_seconds": 120, "runtime_adapter": "local-process",
                                              "isolation_policy": "none"},
                            })
                            request_id = submitted["request_id"]
                            status = self.wait_state(url, request_id, "FAILED" if index == 1 else "COMPLETED", timeout=15)
                            logs = self.request("GET", url, "/v1/requests/" + request_id + "/logs")
                            if index == 1:
                                failed_id = request_id
                                self.assertEqual(status["failure_kind"], "infrastructure")
                                self.assertNotEqual(status["exit_code"], 0)
                                self.assertEqual(logs.get("stdout"), "")
                                self.assertEqual(json.loads(logs["stderr"])["original_exit_code"], 0)
                                self.assertTrue(logs.get("log_path"), logs)
                            else:
                                self.assertEqual(status["exit_code"], 0)
                                self.assertEqual(logs["stdout"], "x" * size)
                            self.assertIsNone(node.poll(), Path(tmp, "node.log").read_text())
                            utilization = self.request("GET", url, "/v1/pools/report-pool/utilization")
                            self.assertEqual(utilization["gpus_leased"], 0)
                        # A one-GPU successor completed, proving the failed
                        # request released the only lease on this same agent.
                    finally:
                        node.terminate()
                        try:
                            node.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            node.kill()
                            node.wait(timeout=3)
                        self.stop_controller(process)
                process, url, _ = self.start_controller(binary, tmp)
                try:
                    restored = self.status(url, failed_id)
                    self.assertEqual(restored["state"], "FAILED")
                    self.assertEqual(restored["failure_kind"], "infrastructure")
                    logs = self.request("GET", url, "/v1/requests/" + failed_id + "/logs")
                    self.assertEqual(json.loads(logs["stderr"])["original_exit_code"], 0)
                finally:
                    self.stop_controller(process)

    def policy_outcome(self, binary, policy, override=None):
        with tempfile.TemporaryDirectory() as tmp:
            flags = [
                "--gpu-contention-policy",
                policy,
                "--gpu-shared-ratio",
                "1",
            ]
            if override:
                flags += ["--gpu-share-override", override]
            process, url, _ = self.start_controller(binary, tmp, *flags)
            try:
                self.register(url, "exclusive-node", "exclusive")
                blocker = self.submit(url, "exclusive", "blocker", "blocker")
                self.wait_state(url, blocker["request_id"], "RUNNING")
                self.register(url, "shared-node", "shared", "shared", 2)
                shared_a = self.submit(url, "shared", "shared-a", "a")
                shared_b = self.submit(url, "shared", "shared-b", "b")
                exclusive = self.submit(url, "exclusive", "exclusive", "next")
                self.finish(url, blocker["request_id"])
                states = {
                    "shared_a": self.status(url, shared_a["request_id"])["state"],
                    "shared_b": self.status(url, shared_b["request_id"])["state"],
                    "exclusive": self.status(url, exclusive["request_id"])["state"],
                }
                return states
            finally:
                self.stop_controller(process)

    def test_contention_policy_and_override_outcomes_match(self):
        expected = {
            "barrier-ratio": {"shared_a": "RUNNING", "shared_b": "QUEUED", "exclusive": "QUEUED"},
            "ratio-only": {"shared_a": "RUNNING", "shared_b": "RUNNING", "exclusive": "QUEUED"},
            "exclusive-first": {"shared_a": "QUEUED", "shared_b": "QUEUED", "exclusive": "RUNNING"},
        }
        for policy, outcome in expected.items():
            observed = [self.policy_outcome(binary, policy) for binary in (PYCAPE_BIN, RSCAPE_BIN)]
            self.assertEqual(observed, [outcome, outcome], policy)
        overridden = [
            self.policy_outcome(binary, "barrier-ratio", "GPU-00000000-0000-0000-0000-000000000001=exclusive-first:1")
            for binary in (PYCAPE_BIN, RSCAPE_BIN)
        ]
        self.assertEqual(overridden, [expected["exclusive-first"]] * 2)

    def test_fairness_prefers_user_without_gpu_holdings(self):
        outcomes = []
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(binary, tmp)
                try:
                    self.register(url, "node-a", "pool-a", gpus=("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"))
                    first_holding = self.submit(url, "pool-a", "holder", "first")
                    self.wait_state(url, first_holding["request_id"], "RUNNING")
                    blocker = self.submit(url, "pool-a", "releaser", "first")
                    self.wait_state(url, blocker["request_id"], "RUNNING")
                    holder = self.submit(url, "pool-a", "holder", "second")
                    newcomer = self.submit(url, "pool-a", "newcomer", "first")
                    self.finish(url, blocker["request_id"])
                    outcomes.append((
                        self.status(url, holder["request_id"])["state"],
                        self.status(url, newcomer["request_id"])["state"],
                    ))
                finally:
                    self.stop_controller(process)
        self.assertEqual(outcomes, [("QUEUED", "RUNNING")] * 2)

    def test_profiles_and_queued_quota_are_reachable_from_both_clis(self):
        outcomes = []
        with tempfile.TemporaryDirectory() as tmp:
            profiles = Path(tmp) / "profiles.json"
            profiles.write_text(json.dumps({"profiles": {"gpu": {
                "image": "docker://example/profile:1",
                "runtime_preference": "local-process",
                "gpu_required": True,
                "env": {"FROM_PROFILE": "1"},
                "default_mounts": [{"source": "/data", "target": "/data", "read_only": True}],
            }}}))
            for index, binary in enumerate((PYCAPE_BIN, RSCAPE_BIN)):
                directory = Path(tmp) / str(index)
                directory.mkdir()
                process, url, _ = self.start_controller(
                    binary,
                    directory,
                    "--profiles-file",
                    str(profiles),
                    "--max-queued-requests-per-user",
                    "0",
                )
                try:
                    value = self.submit(url, "pool-a", "user-a", "profile", profile="gpu")
                    outcomes.append({key: value[key] for key in ("state", "reason", "image", "profile")})
                finally:
                    self.stop_controller(process)
        expected = {"state": "INFEASIBLE", "reason": "quota_exceeded", "image": "docker://example/profile:1", "profile": "gpu"}
        self.assertEqual(outcomes, [expected, expected])

    def test_unknown_profile_is_rejected_by_both_clis(self):
        outcomes = []
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(binary, tmp)
                try:
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        self.submit(url, "pool-a", "user-a", "unknown", profile="missing")
                    outcomes.append((caught.exception.code, json.load(caught.exception)["error"]))
                finally:
                    self.stop_controller(process)
        self.assertEqual([status for status, _ in outcomes], [404, 404])
        for _, error in outcomes:
            self.assertIn("unknown environment profile: missing", error)

    def test_active_user_and_pool_quota_flags_match(self):
        user_outcomes = []
        pool_outcomes = []
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(
                    binary, tmp, "--max-active-gpu-sessions-per-user", "1"
                )
                try:
                    self.register(url, "node-a", "pool-a", gpus=("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"))
                    first = self.submit(url, "pool-a", "user-a", "first")
                    self.wait_state(url, first["request_id"], "RUNNING")
                    limited = self.submit(url, "pool-a", "user-a", "second")
                    other = self.submit(url, "pool-a", "user-b", "first")
                    self.wait_state(url, other["request_id"], "RUNNING")
                    user_outcomes.append((
                        self.status(url, limited["request_id"])["state"],
                        self.status(url, limited["request_id"])["reason"],
                    ))
                finally:
                    self.stop_controller(process)
            with tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(
                    binary, tmp, "--max-active-sessions-per-pool", "1"
                )
                try:
                    self.register(url, "node-a", "pool-a", gpus=("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"))
                    first = self.submit(url, "pool-a", "user-a", "first")
                    self.wait_state(url, first["request_id"], "RUNNING")
                    limited = self.submit(url, "pool-a", "user-b", "second")
                    time.sleep(0.05)
                    status = self.status(url, limited["request_id"])
                    pool_outcomes.append((status["state"], status["reason"]))
                finally:
                    self.stop_controller(process)
        self.assertEqual(user_outcomes, [("QUEUED", "quota_exceeded")] * 2)
        self.assertEqual(pool_outcomes, [("QUEUED", "pool_session_limit")] * 2)

    def test_reconcile_transition_and_missing_grace_match(self):
        outcomes = []
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as tmp:
                flags = [
                    "--reconcile-missing-grace-seconds",
                    "0",
                    "--reconcile-missing-heartbeats",
                    "2",
                ]
                process, url, journal = self.start_controller(binary, tmp, *flags)
                self.register(url, "node-a", "pool-a")
                request = self.submit(url, "pool-a", "user-a", "running")
                request_id = request["request_id"]
                self.wait_state(url, request_id, "RUNNING")
                self.stop_controller(process)
                process, url, _ = self.start_controller(binary, tmp, *flags)
                try:
                    initial = self.status(url, request_id)["state"]
                    empty = {"node_id": "node-a", "status": {"authoritative": True, "sessions": []}}
                    self.request("POST", url, "/v1/agent/heartbeat", empty)
                    first = self.status(url, request_id)["state"]
                    self.request("POST", url, "/v1/agent/heartbeat", empty)
                    terminal = self.status(url, request_id)
                    outcomes.append((initial, first, terminal["state"], terminal["reason"]))
                finally:
                    self.stop_controller(process)
                connection = sqlite3.connect(journal)
                try:
                    reasons = [row[0] for row in connection.execute(
                        "SELECT reason FROM events WHERE entity_id=? ORDER BY id", (request_id,)
                    )]
                finally:
                    connection.close()
                self.assertIn("reconcile_missing", reasons)
        self.assertEqual(outcomes, [("RECONCILING", "RECONCILING", "FAILED", "reconcile_missing")] * 2)

    def test_reconcile_present_running_session_recovers_and_completes(self):
        outcomes = []
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(binary, tmp)
                self.register(url, "node-a", "pool-a")
                request = self.submit(url, "pool-a", "user-a", "running")
                request_id = request["request_id"]
                running = self.wait_state(url, request_id, "RUNNING")
                execution_session_id = running["execution_session_id"]
                self.stop_controller(process)
                process, url, _ = self.start_controller(binary, tmp)
                try:
                    initial = self.status(url, request_id)["state"]
                    self.request(
                        "POST",
                        url,
                        "/v1/agent/heartbeat",
                        {"node_id": "node-a", "status": {"authoritative": True, "sessions": [{"execution_session_id": execution_session_id, "state": "RUNNING_COMMAND", "active_request_id": request_id}]}},
                    )
                    recovered = self.status(url, request_id)["state"]
                    self.finish(url, request_id)
                    terminal = self.status(url, request_id)["state"]
                    outcomes.append((initial, recovered, terminal))
                finally:
                    self.stop_controller(process)
        self.assertEqual(outcomes, [("RECONCILING", "RUNNING", "COMPLETED")] * 2)

    def test_idle_cached_session_expires_in_both(self):
        outcomes = []
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(
                    binary, tmp, "--idle-session-timeout-seconds", "0.5"
                )
                try:
                    self.register(url, "node-a", "pool-a")
                    request = self.submit(url, "pool-a", "user-a", "idle")
                    request_id = request["request_id"]
                    self.wait_state(url, request_id, "RUNNING")
                    execution_session_id = self.status(url, request_id)["execution_session_id"]
                    self.finish(url, request_id)
                    cached_status = {"authoritative": True, "sessions": [{"execution_session_id": execution_session_id, "state": "CACHED_IDLE"}]}
                    young = self.request(
                        "POST", url, "/v1/agent/heartbeat",
                        {"node_id": "node-a", "status": cached_status},
                    )
                    self.assertNotIn("stop_session", [command.get("action") for command in young["commands"]])
                    time.sleep(0.6)
                    response = self.request(
                        "POST",
                        url,
                        "/v1/agent/heartbeat",
                        {"node_id": "node-a", "status": cached_status},
                    )
                    outcomes.append([command.get("action") for command in response["commands"]])
                finally:
                    self.stop_controller(process)
        self.assertEqual(outcomes, [["suspend_session"], ["suspend_session"]])

    def cancellation_outcome(self, binary, delivery, reason="cancelled"):
        with tempfile.TemporaryDirectory() as tmp:
            process, url, journal = self.start_controller(binary, tmp)
            try:
                self.register(url, "node-a", "pool-a", gpus=("GPU-00000000-0000-0000-0000-000000000001",))
                submitted = self.submit(url, "pool-a", "user-a", "cancel", gpus=0)
                request_id = submitted["request_id"]
                self.wait_state(url, request_id, "RUNNING")
                pull_response = self.request(
                    "POST",
                    url,
                    "/v1/agent/heartbeat",
                    {"node_id": "node-a", "max_commands": 8},
                )
                self.assertTrue(
                    pull_response["commands"],
                    (binary, delivery, self.status(url, request_id), pull_response),
                )
                pulled = pull_response["commands"][0]
                command_id = pulled["command_id"]
                self.request(
                    "POST",
                    url,
                    "/v1/requests/{}/cancel".format(request_id),
                    {"reason": reason},
                )
                if delivery == "heartbeat":
                    response = self.request(
                        "POST",
                        url,
                        "/v1/agent/heartbeat",
                        {"node_id": "node-a", "max_commands": 8},
                    )
                else:
                    response = self.request(
                        "POST",
                        url,
                        "/v1/agent/report-command",
                        {
                            "node_id": "node-a",
                            "command_id": command_id,
                            "exit_code": 0,
                            "failure_kind": "workload",
                            "accept_commands": True,
                            "max_commands": 8,
                        },
                    )
                status = self.status(url, request_id)
                actions = [item.get("action") for item in response["commands"]]
            finally:
                self.stop_controller(process)
            connection = sqlite3.connect(journal)
            try:
                command_events = list(connection.execute(
                    "SELECT state, reason FROM events WHERE event_type='command_state' AND entity_id=? ORDER BY id",
                    (command_id,),
                ))
            finally:
                connection.close()
            return {
                "request": (status["state"], status["reason"]),
                "actions": actions,
                "cancel_control_state": response["commands"][0].get("state"),
                "command_terminal": command_events[-1],
            }

    def test_running_cancel_delivery_and_report_race_match(self):
        for delivery in ("heartbeat", "piggyback"):
            for reason in ("cancelled", "wait_timeout"):
                outcomes = [
                    self.cancellation_outcome(binary, delivery, reason=reason)
                    for binary in (PYCAPE_BIN, RSCAPE_BIN)
                ]
                self.assertEqual(outcomes[0], outcomes[1], (delivery, reason))
                self.assertEqual(outcomes[0]["request"], ("CANCELLED", reason))
                self.assertEqual(outcomes[0]["actions"][:2], ["cancel", "stop_session"])
                self.assertEqual(outcomes[0]["cancel_control_state"], "CANCELLED")
                self.assertEqual(outcomes[0]["command_terminal"], ("CANCELLED", reason))

    def test_queued_cancel_has_no_started_at_or_node_command(self):
        outcomes = []
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as tmp:
                process, url, _ = self.start_controller(binary, tmp)
                try:
                    submitted = self.submit(url, "pool-a", "user-a", "queued", gpus=0)
                    request_id = submitted["request_id"]
                    cancelled = self.request(
                        "POST",
                        url,
                        "/v1/requests/{}/cancel".format(request_id),
                        {"reason": "cancelled"},
                    )
                    self.register(url, "node-a", "pool-a", gpus=("GPU-00000000-0000-0000-0000-000000000001",))
                    response = self.request(
                        "POST",
                        url,
                        "/v1/agent/heartbeat",
                        {"node_id": "node-a", "max_commands": 8},
                    )
                    outcomes.append((
                        cancelled["state"],
                        cancelled["reason"],
                        cancelled.get("started_at"),
                        response["commands"],
                    ))
                finally:
                    self.stop_controller(process)
        self.assertEqual(outcomes, [("CANCELLED", "cancelled", None, [])] * 2)

    def test_running_cancel_kills_local_process_in_both(self):
        outcomes = []
        for index, binary in enumerate((PYCAPE_BIN, RSCAPE_BIN)):
            with tempfile.TemporaryDirectory() as tmp:
                process, url, journal = self.start_controller(binary, tmp)
                node_stdout = open(Path(tmp) / "node.stdout", "w")
                node_stderr = open(Path(tmp) / "node.stderr", "w")
                node = subprocess.Popen(
                    [
                        binary,
                        "node", "enroll", *self.node_credentials(url, "cancel-node", "cancel-pool"),
                        "--controller-url",
                        url,
                        "--node-id", "cancel-node", "--pool", "cancel-pool",
                        "--allocation-id",
                        "cancel-allocation",
                        "--runtime-adapter",
                        "local-process",
                        "--heartbeat-interval",
                        "0.01",
                        "--state-path",
                        str(Path(tmp) / "agent-state.json"),
                        "--log-dir",
                        str(Path(tmp) / "agent-logs"),
                    ],
                    stdout=node_stdout,
                    stderr=node_stderr,
                )
                marker = "cape-cancel-live-{}-{}".format(index, os.getpid())
                try:
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        try:
                            utilization = self.request(
                                "GET", url, "/v1/pools/cancel-pool/utilization"
                            )
                            if utilization.get("active_nodes") == 1:
                                break
                            time.sleep(0.02)
                        except urllib.error.HTTPError:
                            time.sleep(0.02)
                    else:
                        self.fail("node did not register: " + Path(tmp, "node.stderr").read_text())
                    submitted = self.request(
                        "POST",
                        url,
                        "/v1/submit",
                        {
                            "pool": "cancel-pool",
                            "user_id": "user-a",
                            "session_id": "cancel-live",
                            "command": [
                                "bash",
                                "-c",
                                "exec -a {} sleep 300".format(marker),
                            ],
                            "resources": {
                                "gpus": 0,
                                "runtime_adapter": "local-process",
                                "isolation_policy": "none",
                            },
                        },
                    )
                    request_id = submitted["request_id"]
                    self.wait_state(url, request_id, "RUNNING", timeout=5.0)
                    deadline = time.monotonic() + 5.0
                    while not self.marker_pids(marker) and time.monotonic() < deadline:
                        time.sleep(0.02)
                    before = self.marker_pids(marker)
                    self.assertTrue(before, (binary, "marker never started"))
                    self.request(
                        "POST",
                        url,
                        "/v1/requests/{}/cancel".format(request_id),
                        {"reason": "cancelled"},
                    )
                    deadline = time.monotonic() + 5.0
                    while self.marker_pids(marker) and time.monotonic() < deadline:
                        time.sleep(0.02)
                    after = self.marker_pids(marker)
                    status = self.status(url, request_id)
                    outcomes.append((status["state"], status["reason"], bool(before), bool(after)))
                    self.assertEqual(after, [], (binary, after))
                finally:
                    if node.poll() is None:
                        node.send_signal(signal.SIGTERM)
                        try:
                            node.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            node.kill()
                            node.wait(timeout=5)
                    node_stdout.close()
                    node_stderr.close()
                    self.stop_controller(process)
        self.assertEqual(outcomes, [("CANCELLED", "cancelled", True, False)] * 2)

    @unittest.skipUnless(
        os.environ.get("CAPE_GPU_METRICS_TEST_GPU_UUID"),
        "set CAPE_GPU_METRICS_TEST_GPU_UUID on a real GPU node",
    )
    def test_rscape_and_pycape_agents_send_same_gpu_metric_fields(self):
        gpu_uuid = os.environ["CAPE_GPU_METRICS_TEST_GPU_UUID"]
        authority_root = tempfile.TemporaryDirectory()
        fixture = ManagedFixture(PYCAPE_BIN, authority_root.name)
        service = fixture.start_embedded(schedule_interval=0.01)
        controller = fixture.controller
        field_sets = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for name, binary in (("py", PYCAPE_BIN), ("rs", RSCAPE_BIN)):
                    result = subprocess.run(
                        [
                            binary,
                            "node",
                            "enroll", *fixture.node_credentials(name + "-metrics-node", "metrics-pool"),
                            "--controller-url",
                            service.url,
                            "--node-id",
                            name + "-metrics-node",
                            "--pool",
                            "metrics-pool",
                            "--gpu-uuid",
                            gpu_uuid,
                            "--runtime-adapter",
                            "local-process",
                            "--heartbeat-interval",
                            "0.01",
                            "--once",
                            "--state-path",
                            str(Path(tmp) / (name + "-state.json")),
                            "--log-dir",
                            str(Path(tmp) / (name + "-logs")),
                        ],
                        text=True,
                        capture_output=True,
                        timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    metrics = controller.nodes[name + "-metrics-node"].last_status.get(
                        "gpu_metrics", []
                    )
                    self.assertTrue(metrics, (name, controller.nodes[name + "-metrics-node"].last_status))
                    metric = next(item for item in metrics if item["uuid"] == gpu_uuid)
                    field_sets.append(set(metric))
        finally:
            fixture.close_embedded()
            authority_root.cleanup()
        self.assertEqual(field_sets[0], field_sets[1])
        self.assertEqual(
            field_sets[0],
            {
                "index", "uuid", "name", "timestamp",
                "gpu_utilization_percent", "memory_utilization_percent",
                "memory_total_bytes", "memory_used_bytes", "memory_free_bytes",
                "temperature_c",
            },
        )


if __name__ == "__main__":
    unittest.main()
