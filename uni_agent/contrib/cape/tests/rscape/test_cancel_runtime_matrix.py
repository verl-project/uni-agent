# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import signal
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tests.managed_fixture import ManagedFixture
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ENABLED = os.environ.get("CAPE_CANCEL_RUNTIME_MATRIX") == "1"
PYCAPE_BIN = os.environ.get("PYCAPE_BIN", "pycape")
RSCAPE_BIN = os.environ.get("RSCAPE_BIN", "rscape")
IMAGE = os.environ.get("CAPE_CANCEL_TEST_IMAGE", "docker://ubuntu:24.04")
GPU_UUID = os.environ.get("CAPE_CANCEL_GPU_UUID")
PODMAN_BIN = os.environ.get("CAPE_CANCEL_PODMAN_BINARY", "/usr/bin/podman")

POLICIES = (
    ("local-process", "none"),
    ("bubblewrap", "host-stateless"),
    ("bubblewrap", "host-stateful"),
    ("apptainer", "image-stateless"),
    ("apptainer", "image-stateful"),
    ("podman", "image-stateless"),
    ("podman", "image-stateful"),
)


class DeferredSessionStopProxy:
    """Separate inner-command cancellation from controller-requested teardown.

    Both old and new controllers cancel a command and queue stop_session. The
    native-container assertion must run before that distinct teardown phase;
    otherwise it races successful removal of the container it is inspecting.
    """
    def __init__(self, upstream):
        self.lock = threading.Lock()
        self.pending = {}
        self.allowed = set()
        self.all_allowed = False
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.do_POST()

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0"))) if self.command == "POST" else None
                headers = {"Content-Type": "application/json"}
                if self.headers.get("Authorization"):
                    headers["Authorization"] = self.headers["Authorization"]
                request = urllib.request.Request(upstream + self.path, data=body, headers=headers, method=self.command)
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        status, data = response.status, response.read()
                except urllib.error.HTTPError as error:
                    status, data = error.code, error.read()
                value = json.loads(data)
                if status < 300 and isinstance(value, dict) and "commands" in value:
                    with proxy.lock:
                        commands = []
                        for command in value["commands"]:
                            identity = command.get("execution_session_id")
                            if command.get("action") == "stop_session":
                                if not proxy.all_allowed and identity not in proxy.allowed:
                                    proxy.pending[identity] = command
                                    continue
                                proxy.pending.pop(identity, None)
                            commands.append(command)
                        for identity in list(proxy.pending):
                            if proxy.all_allowed or identity in proxy.allowed:
                                commands.append(proxy.pending.pop(identity))
                        value["commands"] = commands
                    data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def allow(self, execution_session_id=None):
        with self.lock:
            if execution_session_id is None:
                self.all_allowed = True
            else:
                self.allowed.add(execution_session_id)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@unittest.skipUnless(ENABLED, "set CAPE_CANCEL_RUNTIME_MATRIX=1 on the runtime node")
class CancelRuntimeMatrixTests(unittest.TestCase):
    maxDiff = None

    def free_port(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def request(self, method, url, path, payload=None):
        token = self.fixture.user_token("user-a") if path == "/v1/submit" else self.fixture.admin
        return self.fixture.request(method, path, payload, token)

    def wait_status(self, url, request_id, state, timeout=15):
        deadline = time.monotonic() + timeout
        status = None
        while time.monotonic() < deadline:
            status = self.request("GET", url, "/v1/requests/" + request_id)
            if status["state"] == state:
                return status
            time.sleep(0.02)
        self.fail("request did not reach {}: {}".format(state, status))

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

    def gpu_compute_pids(self, gpu_uuid):
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout
        matches = set()
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",", 1)]
            if len(fields) == 2 and fields[1] == gpu_uuid and fields[0].isdigit():
                matches.add(int(fields[0]))
        return matches

    def wait_marker(self, marker, present, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pids = self.marker_pids(marker)
            if bool(pids) == present:
                return pids
            time.sleep(0.02)
        self.fail("marker {} present={} pids={}".format(marker, present, self.marker_pids(marker)))

    def stop(self, process, timeout=10):
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def archive_instance(self, root):
        # Keep live WAL state node-local; only the stopped snapshot is archived.
        destination = root / "journal.sqlite"
        with sqlite3.connect(str(self.fixture.state / "state.sqlite3")) as source, sqlite3.connect(str(destination)) as target:
            source.backup(target)
        destination.chmod(0o600)
        shutil.rmtree(self.fixture.root)

    def node_args(self, binary, url, adapter, root):
        runtime_root = Path("/tmp/task8-r40-runtime-matrix") / root.name
        runtime_root.mkdir(parents=True, exist_ok=True)
        args = [
            binary, "node", "enroll", *self.fixture.node_credentials("cancel-node", "cancel-pool"), "--controller-url", url,
            "--node-id", "cancel-node", "--pool", "cancel-pool",
            "--allocation-id", "cancel-allocation", "--cpu-cores", "16",
            "--memory-gb", "64", "--runtime-adapter", adapter,
            "--heartbeat-interval", "0.01", "--state-path", str(root / "agent-state.json"),
            "--log-dir", str(root / "agent-logs"),
        ]
        if adapter == "bubblewrap":
            args += ["--bubblewrap-binary", "/usr/bin/bwrap"]
            for path in ("/usr", "/etc", "/opt"):
                args += ["--bubblewrap-system-path", path]
            args += ["--bubblewrap-runtime-dir", str(runtime_root / "bubblewrap")]
        elif adapter == "apptainer":
            args += [
                "--apptainer-binary", "/usr/bin/apptainer",
                "--apptainer-probe-image", IMAGE,
                "--apptainer-cache", "/tmp/task8-r40-runtime-matrix/shared-apptainer-cache",
                "--apptainer-tmp", str(runtime_root / "apptainer-tmp"),
                "--apptainer-runtime-dir", str(runtime_root / "apptainer-runtime"),
            ]
        elif adapter == "podman":
            args += [
                "--podman-binary", PODMAN_BIN,
                "--podman-probe-image", IMAGE,
                "--podman-graphroot", "/tmp/task8-r40-runtime-matrix/shared-podman-graph",
                "--podman-runroot", "/tmp/task8-r40-runtime-matrix/shared-podman-run",
                "--podman-tmpdir", str(runtime_root / "podman-tmp"),
            ]
        return args

    def wait_node(self, process, url, stderr_path, timeout=320):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail("node probe failed: " + stderr_path.read_text()[-6000:])
            try:
                value = self.request("GET", url, "/v1/pools/cancel-pool/utilization")
                if value.get("active_nodes") == 1:
                    return
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            time.sleep(0.1)
        self.fail("node did not become ACTIVE: " + stderr_path.read_text()[-6000:])

    def command(self, marker):
        return ["bash", "-c", "exec -a {} sleep 300".format(marker)]

    def submit(self, url, adapter, policy, marker):
        return self.request("POST", url, "/v1/submit", {
            "pool": "cancel-pool", "user_id": "user-a", "session_id": marker,
            "command": self.command(marker),
            "image": IMAGE if adapter in ("apptainer", "podman") else None,
            "resources": {"gpus": 0, "runtime_adapter": adapter, "isolation_policy": policy},
        })

    def assert_agent_empty(self, state_path):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                state = json.loads(state_path.read_text())
            except (OSError, ValueError):
                time.sleep(0.02)
                continue
            if not state.get("running_commands"):
                return
            time.sleep(0.02)
        self.fail("agent retains running command: " + state_path.read_text())

    def command_terminal(self, journal, command_id):
        with sqlite3.connect(str(journal)) as connection:
            rows = list(connection.execute(
                "SELECT state, reason FROM events WHERE event_type='command_state' AND entity_id=? ORDER BY id",
                (command_id,),
            ))
        self.assertTrue(rows, command_id)
        return rows[-1]

    def assert_runtime_native_state(self, adapter, policy, root, execution_session_id):
        if adapter != "podman":
            return "inner-process-gone"
        runtime_root = Path("/tmp/task8-r40-runtime-matrix") / root.name
        name = "pycape-" + "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in execution_session_id
        )[:56]
        exists = subprocess.run([
            PODMAN_BIN, "--root", "/tmp/task8-r40-runtime-matrix/shared-podman-graph",
            "--runroot", "/tmp/task8-r40-runtime-matrix/shared-podman-run", "--tmpdir", str(runtime_root / "podman-tmp"),
            "container", "exists", name,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if policy == "image-stateful":
            self.assertTrue(exists, name)
            reusable = subprocess.run([
                PODMAN_BIN, "--root", "/tmp/task8-r40-runtime-matrix/shared-podman-graph",
                "--runroot", "/tmp/task8-r40-runtime-matrix/shared-podman-run",
                "--tmpdir", str(runtime_root / "podman-tmp"),
                "exec", name, "true",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            self.assertTrue(reusable, name)
            subprocess.run([
                PODMAN_BIN, "--root", "/tmp/task8-r40-runtime-matrix/shared-podman-graph",
                "--runroot", "/tmp/task8-r40-runtime-matrix/shared-podman-run",
                "--tmpdir", str(runtime_root / "podman-tmp"),
                "rm", "-f", name,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "reusable-then-cleaned"
        else:
            self.assertFalse(exists, name)
            return "removed"

    def run_cell(self, implementation, binary, adapter, policy, output_root):
        label = "{}-{}-{}".format(implementation, adapter, policy)
        root = output_root / label
        root.mkdir(parents=True, exist_ok=True)
        self.fixture = ManagedFixture(binary, tempfile.mkdtemp(prefix="cape-cancel-state-"))
        journal = self.fixture.state / "state.sqlite3"
        controller_out = open(root / "controller.log", "w")
        node_out = open(root / "node.stdout", "w")
        node_err_path = root / "node.stderr"
        node_err = open(node_err_path, "w")
        controller = self.fixture.start(binary, "--schedule-interval", "0.01", output=controller_out)
        url = self.fixture.url
        node = None
        proxy = None
        markers = []
        try:
            self.fixture.pool("cancel-pool")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    self.request("GET", url, "/health")
                    break
                except Exception:
                    if controller.poll() is not None:
                        self.fail("controller exited")
                    time.sleep(0.02)
            proxy = DeferredSessionStopProxy(url)
            node = subprocess.Popen(
                self.node_args(binary, proxy.url, adapter, root),
                stdout=node_out, stderr=node_err,
            )
            self.wait_node(node, url, node_err_path)

            marker = "cape-cancel-{}-manual".format(label)
            markers.append(marker)
            submitted = self.submit(url, adapter, policy, marker)
            running = self.wait_status(url, submitted["request_id"], "RUNNING")
            before = self.wait_marker(marker, True)
            cancelled = subprocess.run([
                binary, "cancel", submitted["request_id"], "--controller-url", url, "--token-file", str(self.fixture.user_file("user-a")),
            ], text=True, capture_output=True, timeout=20)
            self.assertEqual(cancelled.returncode, 0, cancelled.stderr)
            self.wait_marker(marker, False, timeout=30)
            self.assert_agent_empty(root / "agent-state.json")
            status = self.wait_status(url, submitted["request_id"], "CANCELLED")
            manual_native = self.assert_runtime_native_state(
                adapter, policy, root, running["execution_session_id"]
            )
            proxy.allow(running["execution_session_id"])
            manual = {
                "state": status["state"], "reason": status["reason"],
                "before": bool(before), "after": False,
                "command_terminal": self.command_terminal(journal, running["command_id"]),
                "runtime_native": manual_native,
            }

            marker = "cape-cancel-{}-timeout".format(label)
            markers.append(marker)
            run_args = [
                binary, "run", "--controller-url", url, "--pool", "cancel-pool", "--token-file", str(self.fixture.user_file("user-a")),
                "--user", "user-a", "--session-id", marker, "--gpus", "0",
                "--runtime-adapter", adapter, "--isolation-policy", policy,
                "--wait", "--timeout", "30", "--running-timeout", "2",
                "--cancel-on-timeout", "--cancel-reason", "wait_timeout",
            ]
            if adapter in ("apptainer", "podman"):
                run_args += ["--image", IMAGE]
            run_args += ["--"] + self.command(marker)
            timed = subprocess.Popen(run_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            timeout_before = self.wait_marker(marker, True, timeout=15)
            timed_stdout, timed_stderr = timed.communicate(timeout=20)
            self.assertEqual(timed.returncode, 0, timed_stderr)
            timed_status = json.loads(timed_stdout)
            self.assertEqual((timed_status["state"], timed_status["reason"]), ("CANCELLED", "wait_timeout"))
            self.assertTrue(timed_status.get("wait_timed_out"), timed_status)
            self.wait_marker(marker, False, timeout=30)
            self.assert_agent_empty(root / "agent-state.json")
            timeout_native = self.assert_runtime_native_state(
                adapter, policy, root, timed_status["execution_session_id"]
            )
            proxy.allow(timed_status["execution_session_id"])
            timeout_result = {
                "state": timed_status["state"], "reason": timed_status["reason"],
                "before": bool(timeout_before), "after": False,
                "command_terminal": self.command_terminal(journal, timed_status["command_id"]),
                "runtime_native": timeout_native,
            }
            return {"implementation": implementation, "adapter": adapter, "policy": policy,
                    "manual": manual, "timeout": timeout_result}
        finally:
            if proxy is not None:
                proxy.allow()
            for marker in markers:
                for pid in self.marker_pids(marker):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
            if node is not None:
                self.stop(node)
            self.stop(controller)
            self.archive_instance(root)
            if proxy is not None:
                proxy.close()
            node_out.close()
            node_err.close()
            controller_out.close()

    def test_all_runtime_cells_cancel_inner_work(self):
        for binary in (PYCAPE_BIN, RSCAPE_BIN):
            self.assertTrue(Path(binary).is_file() and os.access(binary, os.X_OK), binary)
        output = Path(os.environ.get("CAPE_CANCEL_MATRIX_OUTPUT", tempfile.mkdtemp()))
        output.mkdir(parents=True, exist_ok=True)
        results = []
        for implementation, binary in (("pycape", PYCAPE_BIN), ("rscape", RSCAPE_BIN)):
            for adapter, policy in POLICIES:
                with self.subTest(implementation=implementation, adapter=adapter, policy=policy):
                    results.append(self.run_cell(implementation, binary, adapter, policy, output))
        normalized = {}
        for result in results:
            key = (result["adapter"], result["policy"])
            value = {"manual": result["manual"], "timeout": result["timeout"]}
            if key in normalized:
                self.assertEqual(normalized[key], value)
            else:
                normalized[key] = value
        (output / "matrix.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    @unittest.skipUnless(GPU_UUID, "set CAPE_CANCEL_GPU_UUID on the allocated GPU node")
    def test_cancelled_gpu_process_releases_real_context(self):
        output = Path(os.environ.get("CAPE_CANCEL_MATRIX_OUTPUT", tempfile.mkdtemp()))
        for implementation, binary in (("pycape", PYCAPE_BIN), ("rscape", RSCAPE_BIN)):
            root = output / (implementation + "-gpu")
            root.mkdir(parents=True, exist_ok=True)
            self.fixture = ManagedFixture(binary, tempfile.mkdtemp(prefix="cape-cancel-state-"))
            controller_log = open(root / "controller.log", "w")
            node_log = open(root / "node.log", "w")
            controller = self.fixture.start(binary, "--schedule-interval", "0.01", output=controller_log)
            url = self.fixture.url
            node = None
            marker = "cape-cancel-{}-gpu".format(implementation)
            try:
                self.fixture.pool("cancel-pool")
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        self.request("GET", url, "/health")
                        break
                    except Exception:
                        time.sleep(0.02)
                node_args = self.node_args(binary, url, "local-process", root)
                node_args += ["--gpu-uuid", GPU_UUID]
                node = subprocess.Popen(node_args, stdout=node_log, stderr=node_log)
                self.wait_node(node, url, root / "node.log")
                code = (
                    "import torch,time; "
                    "value=torch.ones(1,device='cuda'); "
                    "torch.cuda.synchronize(); print(value.item(),flush=True); time.sleep(300)"
                )
                submitted = self.request("POST", url, "/v1/submit", {
                    "pool": "cancel-pool", "user_id": "user-a", "session_id": marker,
                    "command": [
                        "bash", "-c", 'exec -a "$1" python3 -c "$2"',
                        "cape-gpu-wrapper", marker, code,
                    ],
                    "resources": {
                        "gpus": 1, "runtime_adapter": "local-process", "isolation_policy": "none",
                    },
                })
                self.wait_status(url, submitted["request_id"], "RUNNING")
                marker_pids = set(self.wait_marker(marker, True, timeout=15))
                deadline = time.monotonic() + 15
                context_pids = set()
                while time.monotonic() < deadline:
                    context_pids = self.gpu_compute_pids(GPU_UUID) & marker_pids
                    if context_pids:
                        break
                    time.sleep(0.05)
                self.assertTrue(context_pids, (implementation, marker_pids, self.gpu_compute_pids(GPU_UUID)))
                self.request(
                    "POST", url, "/v1/requests/{}/cancel".format(submitted["request_id"]),
                    {"reason": "cancelled"},
                )
                self.wait_marker(marker, False)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and (self.gpu_compute_pids(GPU_UUID) & marker_pids):
                    time.sleep(0.05)
                self.assertFalse(self.gpu_compute_pids(GPU_UUID) & marker_pids)
                utilization = self.request("GET", url, "/v1/pools/cancel-pool/utilization")
                self.assertEqual(utilization["gpus_leased"], 0)
                self.assertEqual(utilization["gpus_available"], 1)
            finally:
                for pid in self.marker_pids(marker):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
                if node is not None:
                    self.stop(node)
                self.stop(controller)
                self.archive_instance(root)
                node_log.close()
                controller_log.close()
