# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from tests.managed_fixture import ManagedFixture
from tests.rscape import test_functional_parity as parity
from pycape.managed_auth import ManagedAuthority
from pycape.identity import IdentityStore

from pycape import (
    CAPEController,
    NodeResources,
    ResourceRequest,
    RuntimeCapabilities,
    SQLiteJournal,
    WorkloadSpec,
)


PYCAPE_BIN = os.environ.get("PYCAPE_BIN", "pycape")
RSCAPE_BIN = os.environ.get("RSCAPE_BIN", "rscape")


class JournalInteropTests(unittest.TestCase):
    maxDiff = None
    def free_port(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def start_controller(self, binary, journal):
        process = self.fixture.start(binary)
        self.fixture.pool("pool-a")
        return process, self.fixture.url

    def test_rscape_reconciles_pycape_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = parity.FunctionalParityTests()
            source, url, journal_path = helper.start_controller(PYCAPE_BIN, tmp)
            self.fixture = helper.fixtures[url]
            try:
                helper.register(url, "node-a", "pool-a", gpus=())
                submitted = helper.submit(url, "pool-a", "alice", "journal-interop", gpus=0)
                request_id = submitted["request_id"]
                running = helper.wait_state(url, request_id, "RUNNING")
                helper.request("POST", url, "/v1/agent/heartbeat", {"node_id": "node-a", "max_commands": 8})
                helper.request("POST", url, "/v1/agent/report-command", {
                    "node_id": "node-a", "command_id": running["command_id"], "exit_code": 7, "failure_kind": "workload"})
                helper.wait_state(url, request_id, "FAILED")
            finally:
                helper.stop_controller(source)

            oracle_journal = SQLiteJournal(journal_path)
            store = IdentityStore(str(journal_path))
            oracle = CAPEController.reconcile_from_journal(oracle_journal, auth=ManagedAuthority(store))
            from pycape.managed_lifecycle import reconcile
            reconcile(oracle)
            expected = oracle.request_status(request_id, token=self.fixture.admin)
            expected_utilization = oracle.pool_utilization(oracle.auth.pool_id("pool-a"))
            for node in expected_utilization["node_states"]:
                # Public inventory intentionally omits private session identities.
                node.pop("active_sessions", None)
            store.close()
            oracle_journal.close()

            process, url = self.start_controller(RSCAPE_BIN, journal_path)
            try:
                status = subprocess.run(
                    [
                        RSCAPE_BIN,
                        "status",
                        request_id, "--token-file", str(self.fixture.admin_file),
                        "--controller-url",
                        url,
                    ],
                    text=True,
                    capture_output=True,
                    check=True,
                )
                payload = json.loads(status.stdout)
                self.assertEqual(payload["state"], "FAILED")
                self.assertEqual(payload["failure_kind"], "workload")
                self.assertEqual(payload["execution_session_id"], running["execution_session_id"])
                self.assertEqual(payload, expected)
                utilization = subprocess.run(
                    [
                        RSCAPE_BIN,
                        "utilization", "--token-file", str(self.fixture.admin_file),
                        "--controller-url",
                        url,
                        "--pool",
                        "pool-a",
                    ],
                    text=True,
                    capture_output=True,
                    check=True,
                )
                actual_utilization = json.loads(utilization.stdout)
                for expected_node, actual_node in zip(
                    expected_utilization["node_states"],
                    actual_utilization["node_states"],
                ):
                    expected_node["last_heartbeat_at"] = actual_node[
                        "last_heartbeat_at"
                    ]
                self.assertEqual(actual_utilization, expected_utilization)
            finally:
                process.terminate()
                process.communicate(timeout=5)

    def test_pycape_reconciles_rscape_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture = ManagedFixture(RSCAPE_BIN, tmp)
            journal_path = self.fixture.state / "state.sqlite3"
            controller, url = self.start_controller(RSCAPE_BIN, journal_path)
            agent = subprocess.Popen(
                [
                    RSCAPE_BIN,
                    "node",
                    "enroll", *self.fixture.node_credentials("node-a", "pool-a"),
                    "--controller-url",
                    url,
                    "--node-id",
                    "node-a",
                    "--pool",
                    "pool-a",
                    "--allocation-id",
                    "allocation-a",
                    "--cpu-cores",
                    "4",
                    "--memory-gb",
                    "8",
                    "--runtime-adapter",
                    "local-process",
                    "--heartbeat-interval",
                    "0.02",
                    "--log-dir",
                    str(Path(tmp) / "logs"),
                    "--state-path",
                    str(Path(tmp) / "state.json"),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                for _ in range(200):
                    util = subprocess.run(
                        [
                            RSCAPE_BIN,
                            "utilization", "--token-file", str(self.fixture.admin_file),
                            "--controller-url",
                            url,
                            "--pool",
                            "pool-a",
                        ],
                        text=True,
                        capture_output=True,
                    )
                    if util.returncode == 0 and json.loads(util.stdout)["active_nodes"] == 1:
                        break
                    time.sleep(0.01)
                failed = subprocess.run(
                    [
                        RSCAPE_BIN,
                        "run", "--token-file", str(self.fixture.user_file("alice")),
                        "--controller-url",
                        url,
                        "--pool",
                        "pool-a",
                        "--user",
                        "alice",
                        "--session-id",
                        "journal-interop",
                        "--runtime-adapter",
                        "local-process",
                        "--isolation-policy",
                        "none",
                        "--wait",
                        "--timeout",
                        "10",
                        "--",
                        sys.executable,
                        "-c",
                        "raise SystemExit(7)",
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(failed.returncode, 7, failed.stderr)
                request_id = json.loads(failed.stdout)["request_id"]
            finally:
                agent.terminate()
                agent.communicate(timeout=5)
                controller.terminate()
                controller.communicate(timeout=5)

            journal = SQLiteJournal(journal_path)
            replayed = CAPEController.reconcile_from_journal(journal)
            status = replayed.request_status(request_id)
            self.assertEqual(status["state"], "FAILED")
            self.assertEqual(status["failure_kind"], "workload")
            replayed.assert_invariants()
            journal.close()


if __name__ == "__main__":
    unittest.main()
