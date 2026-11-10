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
from pathlib import Path
from tests.managed_fixture import ManagedFixture, simulated_gpu_environment


CAPE_BIN = os.environ.get("CAPE_CONFORMANCE_BIN")
RESTART_BIN = os.environ.get("CAPE_CONFORMANCE_RESTART_BIN")


def _port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _run(binary, *arguments, timeout=15):
    completed = subprocess.run(
        [binary, *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "command failed ({}): {}\nstdout={}\nstderr={}".format(
                completed.returncode,
                " ".join([binary, *arguments]),
                completed.stdout,
                completed.stderr,
            )
        )
    return completed.stdout.strip()


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
        except (OSError, subprocess.SubprocessError, ValueError):
            time.sleep(0.05)
            continue
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError("condition did not become true; last={!r}".format(last))


class Stack:
    def __init__(self, binary, root, label):
        self.binary = binary
        self.root = Path(root)
        self.label = label
        self.port = _port()
        self.url = "http://127.0.0.1:{}".format(self.port)
        self.fixture = ManagedFixture(binary, self.root / (label + "-instance"))
        self.journal = self.fixture.state / "state.sqlite3"
        self.controller = None
        self.node = None

    def start_controller(self, controller_binary=None):
        controller_binary = controller_binary or self.binary
        self.controller = self.fixture.start(controller_binary, "--schedule-interval", "0.02",
                                             "--heartbeat-timeout-seconds", "0.5", "--lost-timeout-seconds", "3")
        self.url = self.fixture.url
        self.port = int(self.url.rsplit(":", 1)[1])
        self.fixture.pool("p")
        self.fixture.request("POST", "/v1/admin/workspace-config", {"pool": "p", "changes": {
            "image_root": str(self.root / (self.label + "-images")), "image_size_bytes": 64 * 1024 ** 2}})

    def _health(self):
        return bool(self.fixture.request("GET", "/health"))

    def start_node(self, allocation, deadline=None):
        arguments = [
                self.binary,
                "node",
                "enroll", *self.fixture.node_credentials("node-a", "p"),
                "--controller-url",
                self.url,
                "--node-id",
                "node-a",
                "--pool",
                "p",
                "--allocation-id",
                allocation,
                "--gpu-uuid",
                "GPU-00000000-0000-0000-0000-000000000001",
                "--cpu-cores",
                "4",
                "--memory-gb",
                "8",
                "--runtime-adapter",
                "local-process",
                "--heartbeat-interval",
                "0.05",
                "--log-dir",
                str(self.root / (self.label + "-logs")),
                "--state-path",
                str(self.root / (self.label + "-state.json")),
            ]
        if deadline is not None:
            arguments.extend(["--allocation-deadline", str(deadline)])
        self.node = subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=simulated_gpu_environment(self.root, ["GPU-00000000-0000-0000-0000-000000000001"]),
        )
        def active_utilization():
            if self.node.poll() is not None:
                return None
            value = self.utilization()
            return value if value.get("active_nodes") == 1 else None

        return _wait_for(active_utilization, timeout=8)

    def utilization(self):
        return json.loads(
            _run(
                self.binary,
                "utilization", "--token-file", str(self.fixture.admin_file),
                "--controller-url",
                self.url,
                "--pool",
                "p",
            )
        )

    def submit(self, task, seconds, max_duration=120, command_timeout=5):
        arguments = [
            self.binary,
            "run", "--token-file", str(self.fixture.user_file("u")),
            "--controller-url",
            self.url,
            "--pool",
            "p",
            "--user",
            "u",
            "--session-id",
            task,
            "--gpus",
            "1",
            "--max-duration-seconds",
            str(max_duration),
        ]
        if command_timeout is not None:
            arguments.extend(["--command-timeout-seconds", str(command_timeout)])
        arguments.extend([
            "--runtime-adapter",
            "local-process",
            "--isolation-policy",
            "none",
            "--",
            "sleep",
            str(seconds),
        ])
        return _run(*arguments).splitlines()[-1]

    def status(self, request_id):
        return json.loads(
            _run(
                self.binary,
                "status",
                request_id, "--token-file", str(self.fixture.user_file("u")),
                "--controller-url",
                self.url,
            )
        )

    def wait_state(self, request_id, state, timeout=8):
        def matching_status():
            value = self.status(request_id)
            return value if value.get("state") == state else None

        return _wait_for(matching_status, timeout=timeout)

    def stop_controller(self):
        if self.controller is not None and self.controller.poll() is None:
            os.killpg(self.controller.pid, signal.SIGTERM)
            self.controller.wait(timeout=5)
        if self.controller is not None:
            for stream in (self.controller.stdout, self.controller.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

    def cleanup(self):
        if self.node is not None and self.node.poll() is None:
            os.killpg(self.node.pid, signal.SIGKILL)
            self.node.wait(timeout=5)
        self.stop_controller()
        for process in (self.node, self.controller):
            if process is None:
                continue
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def _status_projection(value):
    keys = (
        "state",
        "reason",
        "failure_kind",
        "exit_code",
        "gpu_uuids",
        "runtime_adapter",
        "isolation_policy",
        "status_message",
    )
    projection = {key: value.get(key) for key in keys}
    projection["field_names"] = sorted(value)
    projection["null_fields"] = sorted(
        key for key, field_value in value.items() if field_value is None
    )
    return projection


def _journal_projection(path):
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT event_type,state,reason,details FROM events ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    projection = []
    for event_type, state, reason, details_json in rows:
        if event_type not in {"request_state", "session_state", "node_state"}:
            continue
        details = json.loads(details_json)
        projection.append(
            {
                "event_type": event_type,
                "state": state,
                "reason": reason,
                "failure_kind": details.get("failure_kind"),
                "detail_fields": sorted(details),
            }
        )
    return projection


@unittest.skipUnless(CAPE_BIN, "set CAPE_CONFORMANCE_BIN to rscape or pycape")
class BehavioralConformanceTests(unittest.TestCase):
    def test_sigkill_loss_sigterm_drain_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            loss = Stack(CAPE_BIN, directory, "loss")
            try:
                loss.start_controller()
                utilization = loss.start_node("loss-a")
                self.assertEqual(
                    utilization["node_states"][0].get("allocation_deadline"),
                    None,
                )
                request_id = loss.submit("loss", 2)
                loss.wait_state(request_id, "RUNNING")
                os.killpg(loss.node.pid, signal.SIGKILL)
                loss.node.wait(timeout=5)
                lost = loss.wait_state(request_id, "LOST", timeout=5)
                self.assertEqual(lost["reason"], "node_lost")
                self.assertEqual(lost["failure_kind"], "infrastructure")
                self.assertIsNone(lost["exit_code"])
                utilization = loss.utilization()
                self.assertEqual(utilization["gpus_leased"], 0)
                self.assertEqual(utilization["node_states"][0]["state"], "LOST")
                loss.stop_controller()
                loss.start_controller(RESTART_BIN or CAPE_BIN)
                restarted = loss.status(request_id)
                self.assertEqual(_status_projection(restarted), _status_projection(lost))
                self.assertEqual(loss.utilization()["gpus_leased"], 0)
                loss_projection = _journal_projection(loss.journal)
            finally:
                loss.cleanup()

            drain = Stack(CAPE_BIN, directory, "drain")
            try:
                drain.start_controller()
                drain.start_node("drain-a")
                request_id = drain.submit("drain", 2)
                drain.wait_state(request_id, "RUNNING")
                os.killpg(drain.node.pid, signal.SIGTERM)
                completed = drain.wait_state(request_id, "COMPLETED", timeout=8)
                drain.node.wait(timeout=8)
                self.assertEqual(drain.node.returncode, 0)
                utilization = drain.utilization()
                self.assertEqual(utilization["running_requests"], 0)
                self.assertEqual(utilization["gpus_leased"], 0)
                self.assertFalse(any(node["node_id"] == "node-a" for node in utilization["node_states"]))
            finally:
                drain.cleanup()

            admission = Stack(CAPE_BIN, directory, "admission")
            try:
                admission.start_controller()
                admission.start_node("admission-a", deadline=time.time() + 900)
                short_id = admission.submit(
                    "short-command", 0.1, max_duration=1200, command_timeout=10
                )
                short = admission.wait_state(short_id, "COMPLETED")
                long_id = admission.submit(
                    "long-command", 0.1, max_duration=1200, command_timeout=1200
                )
                long_status = admission.wait_state(long_id, "QUEUED")
                self.assertEqual(long_status["reason"], "allocation_expiring")
                legacy_id = admission.submit(
                    "legacy-bound", 0.1, max_duration=1200, command_timeout=None
                )
                legacy = admission.wait_state(legacy_id, "QUEUED")
                self.assertEqual(legacy["reason"], "allocation_expiring")
            finally:
                admission.cleanup()

            output = os.environ.get("CAPE_CONFORMANCE_OUTPUT")
            if output:
                Path(output).write_text(
                    json.dumps(
                        {
                            "loss_status": _status_projection(lost),
                            "restart_status": _status_projection(restarted),
                            "completed_status": _status_projection(completed),
                            "short_admission_status": _status_projection(short),
                            "long_admission_status": _status_projection(long_status),
                            "legacy_admission_status": _status_projection(legacy),
                            "loss_journal": loss_projection,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )


if __name__ == "__main__":
    unittest.main()
