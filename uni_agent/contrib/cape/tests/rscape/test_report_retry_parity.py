# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests.rscape import test_functional_parity as parity
from pycape.workspace import WorkspaceConfig
from tests.managed_fixture import EnrollmentHandlerMixin, simulated_gpu_environment


BINARIES = (os.environ.get("PYCAPE_BIN", "pycape"), os.environ.get("RSCAPE_BIN", "rscape"))


class ReportRetryParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for binary in BINARIES:
            if not Path(binary).is_file():
                raise unittest.SkipTest("set PYCAPE_BIN and RSCAPE_BIN to both built CLIs")

    def exercise(self, binary, mode, once=False):
        state = {"issued": 0, "reports": [], "heartbeats": 0, "stopping": False}
        lock = threading.Lock()

        class Handler(EnrollmentHandlerMixin, BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.enrollment_response(payload):
                    return
                status, result = 200, {}
                with lock:
                    if self.path == "/v1/agent/announce":
                        result = {"workspace_config": workspace_config}
                    elif self.path == "/v1/admin/drain-node":
                        state["stopping"] = True
                    elif self.path == "/v1/agent/heartbeat" and payload.get("state") != "REGISTERING":
                        state["heartbeats"] += 1
                        count = state["issued"]
                        first_reports = [row for row in state["reports"] if row["command_id"] == "unknown-command"]
                        issue = not state["stopping"] and (count == 0 or (count == 1 and first_reports and mode != "exhausted"))
                        result = {"commands": []}
                        if state["stopping"]:
                            result["commands"] = [{"action": "stop_session", "execution_session_id": "session-" + str(index)} for index in range(count)]
                        if issue:
                            command_id = "unknown-command" if count == 0 else "later-command"
                            execution_session_id = "session-" + str(count)
                            result["commands"] = [{
                                "action": "run", "command_id": command_id,
                                "execution_session_id": execution_session_id, "request_id": "request-" + str(count),
                                "session_spec": {"execution_session_id": execution_session_id, "runtime_adapter": "local-process",
                                                 "workspace": {"id": "ws-" + str(count).zfill(32), "user_id": "alice", "pool": "fixture-pool", "node_id": "retry-node",
                                                               "execution_session_id": execution_session_id, "config": workspace_config, "create": True, "target": "/workspace"},
                                                 "isolation_policy": "none", "gpu_uuids": []},
                                "command_spec": {"command": ["/bin/sh", "-c", "printf later"], "env": {}},
                            }]
                            state["issued"] += 1
                    elif self.path == "/v1/agent/report-command":
                        state["reports"].append(payload)
                        if payload["command_id"] == "unknown-command":
                            attempts = sum(row["command_id"] == "unknown-command" for row in state["reports"])
                            if mode == "permanent":
                                status, result = 400, {"error": "unknown command_id"}
                            elif mode == "exhausted" or attempts < 3:
                                status, result = 503, {"error": "controller restarting"}
                body = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="cape-report-retry-") as directory:
                root = Path(directory)
                workspace_config = WorkspaceConfig(image_root=str(root / "images"), image_size_bytes=64 * 1024 ** 2).to_dict()
                with open(root / "agent.log", "w") as stream:
                    process = subprocess.Popen([
                        binary, "node", "enroll", "--token", "fixture-personal", "--node-token-out", str(root / "agent.json"), "--controller-url", "http://127.0.0.1:{}".format(server.server_port),
                        "--pool", "pool", "--node-id", "retry-node", "--runtime-adapter", "local-process",
                        "--heartbeat-interval", "0.05", "--state-path", str(root / "state.json"),
                        "--log-dir", str(root / "logs"),
                    ] + (["--once"] if once else []), stdout=stream, stderr=stream)
                    try:
                        if mode == "exhausted":
                            self.assertNotEqual(process.wait(timeout=12), 0)
                        else:
                            deadline = time.monotonic() + 10
                            while time.monotonic() < deadline:
                                with lock:
                                    reports = list(state["reports"])
                                expected = 1 if mode == "permanent" else 3
                                if (sum(row["command_id"] == "unknown-command" for row in reports) >= expected
                                        and any(row["command_id"] == "later-command" for row in reports)):
                                    break
                                self.assertIsNone(process.poll(), (root / "agent.log").read_text())
                                time.sleep(0.05)
                            else:
                                self.fail("report/successor did not finish: " + (root / "agent.log").read_text())
                            # Old unbounded retries add attempts during this observation.
                            time.sleep(2.2)
                            self.assertIsNone(process.poll())
                        with lock:
                            reports = list(state["reports"])
                        first = [row for row in reports if row["command_id"] == "unknown-command"]
                        self.assertEqual(len(first), {"permanent": 1, "recover": 3, "exhausted": 4}[mode])
                        if mode == "permanent":
                            self.assertIn("reason=permanent_http", (root / "agent.log").read_text())
                        elif mode == "recover":
                            self.assertEqual(first[-1]["failure_kind"], "infrastructure")
                            self.assertEqual(json.loads(first[-1]["stderr"])["original_exit_code"], 0)
                        else:
                            self.assertIn("retry_exhausted_agent_stopping", (root / "agent.log").read_text())
                        if mode != "exhausted":
                            later = next(row for row in reports if row["command_id"] == "later-command")
                            self.assertEqual((later["exit_code"], later["stdout"]), (0, "later"))
                        print(json.dumps({"binary": binary, "mode": mode, "attempts": len(first),
                                          "agent_exited": process.poll() is not None}), flush=True)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_permanent_rejection_finishes_and_same_agent_serves_later_command(self):
        for binary in BINARIES:
            with self.subTest(binary=binary):
                self.exercise(binary, "permanent")

    def test_transient_failure_recovers(self):
        for binary in BINARIES:
            with self.subTest(binary=binary):
                self.exercise(binary, "recover")

    def test_transient_exhaustion_stops_agent_instead_of_stranding_work(self):
        for binary in BINARIES:
            with self.subTest(binary=binary):
                self.exercise(binary, "exhausted")

    def test_one_shot_agent_also_exits_on_exhaustion(self):
        for binary in BINARIES:
            with self.subTest(binary=binary):
                self.exercise(binary, "exhausted", once=True)

    def test_report_only_outage_ends_in_node_loss_and_releases_real_controller_lease(self):
        helper = parity.FunctionalParityTests()
        for binary in BINARIES:
            with self.subTest(binary=binary), tempfile.TemporaryDirectory(prefix="cape-report-outage-") as directory:
                root = Path(directory)
                controller, url, _ = helper.start_controller(
                    binary, directory, "--heartbeat-timeout-seconds", "1", "--lost-timeout-seconds", "2"
                )
                reports = []

                class Proxy(BaseHTTPRequestHandler):
                    def log_message(self, *_args):
                        pass

                    def do_GET(self):
                        self.do_POST()

                    def do_POST(self):
                        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))) if self.command == "POST" else None
                        if self.path == "/v1/agent/report-command":
                            reports.append(json.loads(body))
                            status, data = 503, b'{"error":"report endpoint unavailable"}'
                        else:
                            request = urllib.request.Request(url + self.path, data=body, headers={"Content-Type": "application/json", "Authorization": self.headers.get("Authorization", "")}, method=self.command)
                            try:
                                with urllib.request.urlopen(request, timeout=5) as response:
                                    status, data = response.status, response.read()
                            except urllib.error.HTTPError as error:
                                status, data = error.code, error.read()
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)

                server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
                server.daemon_threads = True
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                with open(root / "node.log", "w") as stream:
                    node = subprocess.Popen([
                        binary, "node", "enroll", *helper.node_credentials(url, "outage-node", "outage"), "--controller-url", "http://127.0.0.1:{}".format(server.server_port),
                        "--pool", "outage", "--node-id", "outage-node", "--runtime-adapter", "local-process",
                        "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000003", "--heartbeat-interval", "0.05",
                        "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs"),
                    ], stdout=stream, stderr=stream, env=simulated_gpu_environment(root, ["GPU-00000000-0000-0000-0000-000000000003"]))
                    try:
                        submitted = helper.request("POST", url, "/v1/submit", {
                            "pool": "outage", "user_id": "alice", "session_id": "outage",
                            "reuse": False, "command": ["/bin/sh", "-c", "sleep 0.2; printf done"],
                            "resources": {"gpus": 1, "runtime_adapter": "local-process",
                                          "isolation_policy": "none", "max_duration_seconds": 120},
                        })
                        helper.wait_state(url, submitted["request_id"], "RUNNING", timeout=5)
                        self.assertEqual(helper.request("GET", url, "/v1/pools/outage/utilization")["gpus_leased"], 1)
                        self.assertNotEqual(node.wait(timeout=12), 0)
                        self.assertEqual(len(reports), 4)
                        status = helper.wait_state(url, submitted["request_id"], "LOST", timeout=8)
                        self.assertEqual(status["failure_kind"], "infrastructure")
                        self.assertEqual(helper.request("GET", url, "/v1/pools/outage/utilization")["gpus_leased"], 0)
                        print(json.dumps({"binary": binary, "report_only_outage": "LOST", "leases": 0}), flush=True)
                    finally:
                        if node.poll() is None:
                            node.kill()
                            node.wait(timeout=5)
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=5)
                        helper.stop_controller(controller)
