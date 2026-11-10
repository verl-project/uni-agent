# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import urllib.error

from tests.rscape import test_functional_parity as parity
from pycape import EventRecord, SQLiteJournal
from pycape.workspace import WorkspaceConfig


class WorkspaceConfigParityTests(unittest.TestCase):
    def test_reclaim_policy_cli_fields_match_both_controllers(self):
        helper = parity.FunctionalParityTests()
        expected = {"reclaim_idle_seconds": 7, "reclaim_start_interval_seconds": 9,
                    "reclaim_max_candidates": 12, "reclaim_retry_backoff_seconds": 17,
                    "reclaim_low_yield_backoff_seconds": 90, "reclaim_nice": 6}
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as root:
                process, url, _ = helper.start_controller(server, root)
                fixture = helper.fixtures[url]
                fixture.pool("pool")
                try:
                    for client in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
                        arguments = [client, "admin", "set-workspace-config", "--pool", "pool",
                                     "--controller-url", url, "--token-file", str(fixture.admin_file)]
                        for field, value in expected.items():
                            arguments.extend(["--" + field.replace("_", "-"), str(value)])
                        result = subprocess.run(arguments, capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        status = helper.request("GET", url, "/v1/admin/workspace-config?pool=pool")
                        for field, value in expected.items():
                            self.assertEqual(status["desired_config"]["maintenance"][field], value)
                finally:
                    helper.stop_controller(process)

    def test_corrupt_agent_state_is_preserved_and_fails_closed(self):
        helper = parity.FunctionalParityTests()
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "agent-state.json"
                state.write_text('{"local_sessions": [')
                before = state.read_bytes()
                process, url, _ = helper.start_controller(binary, tmp)
                try:
                    node = subprocess.run([binary, "node", "enroll", *helper.node_credentials(url, "node", "pool"), "--controller-url", url, "--pool", "pool", "--node-id", "node",
                        "--runtime-adapter", "local-process", "--once", "--state-path", str(state), "--log-dir", str(Path(tmp) / "logs")],
                        capture_output=True, text=True, timeout=15)
                    self.assertNotEqual(node.returncode, 0)
                    self.assertIn("agent state", node.stderr)
                    self.assertEqual(state.read_bytes(), before)
                finally:
                    helper.stop_controller(process)

    def test_interrupted_workspace_commands_fence_agent_restart(self):
        helper = parity.FunctionalParityTests()
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "agent-state.json"
                state.write_text(json.dumps({"version": 2, "local_sessions": ["old-session"], "running_commands": ["old-command"],
                    "session_contexts": {"old-session": {"runtime_adapter": "local-process", "session_spec": {
                        "workspace": {"id": "ws-" + "b" * 32, "user_id": "alice", "pool": "pool", "node_id": "node",
                                      "config": WorkspaceConfig(image_root=tmp).to_dict()}}}}}))
                before = state.read_bytes()
                process, url, _ = helper.start_controller(binary, tmp)
                try:
                    node = subprocess.run([binary, "node", "enroll", *helper.node_credentials(url, "node", "pool"), "--controller-url", url, "--pool", "pool", "--node-id", "node",
                        "--runtime-adapter", "local-process", "--once", "--state-path", str(state), "--log-dir", str(Path(tmp) / "logs")],
                        capture_output=True, text=True, timeout=15)
                    self.assertNotEqual(node.returncode, 0)
                    self.assertIn("fence predecessor process groups", node.stderr)
                    self.assertEqual(state.read_bytes(), before)
                finally:
                    helper.stop_controller(process)

    def test_workspace_mutation_requires_owner_or_admin_and_replays_pending_operation(self):
        helper = parity.FunctionalParityTests()
        identity = "ws-" + "a" * 32
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as tmp:
                process, url, journal_path = helper.start_controller(server, tmp)
                fixture = helper.fixtures[url]
                pool_id = fixture.pool("pool")
                alice_token = fixture.user_token("alice")
                bob_token = fixture.user_token("bob")
                viewer_token = fixture.user_token("viewer")
                alice_id = fixture.request("GET", "/v1/identity/me", token=alice_token)["user_id"]
                helper.stop_controller(process)
                journal = SQLiteJournal(str(journal_path))
                journal.append(EventRecord(0, "workspace_state", identity, "retained", None, {
                    "id": identity, "user_id": alice_id, "pool": pool_id, "node_id": "node", "execution_session_id": "old-session",
                    "state": "retained", "revision": 5, "config": WorkspaceConfig().to_dict(), "path": None, "last_error": None}))
                journal.close()
                process, url, _ = helper.start_controller(server, tmp)
                operation = None
                try:
                    for client in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
                        for token in (bob_token, viewer_token, "unknown-token"):
                            refused = subprocess.run([client, "workspace", "delete", identity, "--controller-url", url,
                                "--token", token], capture_output=True, text=True, timeout=15)
                            self.assertNotEqual(refused.returncode, 0, refused.stdout)
                            for action in ("grow", "reclaim", "shrink", "recover"):
                                args = [client, "workspace", action, identity, "--controller-url", url, "--token", token]
                                if action in ("grow", "shrink"):
                                    args += ["--size-gib", "1"]
                                refused = subprocess.run(args, capture_output=True, text=True, timeout=15)
                                self.assertNotEqual(refused.returncode, 0, refused.stdout)
                                self.assertIn("403", refused.stderr, "authorization must fail before capability checks")
                        accepted = subprocess.run([client, "workspace", "check", identity, "--controller-url", url,
                            "--token", alice_token], capture_output=True, text=True, timeout=15)
                        self.assertEqual(accepted.returncode, 0, accepted.stderr)
                        current = json.loads(accepted.stdout)["operation"]
                        if operation is not None:
                            self.assertEqual(current, operation, "repeated pending operation must be idempotent")
                        operation = current
                    process.kill(); process.wait(timeout=5)
                finally:
                    helper.stop_controller(process)
                process, url, _ = helper.start_controller(server, tmp)
                try:
                    result = subprocess.run([server, "workspace", "status", identity, "--controller-url", url,
                        "--token", fixture.admin], capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout)["operation"], operation)
                finally:
                    helper.stop_controller(process)

    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def cli(self, binary, url, action, *args, success=True):
        if "--token" not in args:
            args = (*args, "--token-file", str(self.fixture.admin_file))
        result = subprocess.run([binary, "admin", action, "--controller-url", url,
                                 "--pool", "pool", *args], capture_output=True, text=True, timeout=15)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_both_clients_and_servers_share_versioned_defaults_and_durable_replay(self):
        helper = parity.FunctionalParityTests()
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            for client in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
                with self.subTest(server=server, client=client), tempfile.TemporaryDirectory() as tmp:
                    process, url, _ = helper.start_controller(server, tmp)
                    self.fixture = helper.fixtures[url]
                    self.fixture.pool("pool")
                    try:
                        initial = self.cli(client, url, "get-workspace-config")
                        self.assertEqual(initial["desired_config"]["version"], 0)
                        self.assertEqual(initial["desired_config"]["image_root"], "/tmp")
                        pool = self.cli(client, url, "set-workspace-config", "--image-size-gib", "12",
                                        "--image-root", "/not-on-submitter/work space")
                        self.assertEqual(pool["desired_config"]["image_size_bytes"], 12 * 1024 ** 3)
                        self.assertEqual(pool["desired_config"]["version"], 1)
                        node = self.cli(client, url, "set-workspace-config", "--node-id", "node",
                                        "--image-root", "/node/local", "--bytes-per-inode", "4096")
                        self.assertEqual(node["desired_config"]["version"], 2)
                        self.assertEqual(node["desired_config"]["image_size_bytes"], 12 * 1024 ** 3)
                        self.assertEqual(self.cli(client, url, "get-workspace-config", "--node-id", "other")["desired_config"]["image_root"], "/not-on-submitter/work space")
                        self.cli(client, url, "set-workspace-config", "--image-root", "/pool/new")
                        expected = self.cli(client, url, "get-workspace-config", "--node-id", "node")
                        self.assertEqual(expected["desired_config"]["image_root"], "/node/local")
                        self.assertEqual(expected["desired_config"]["version"], 3)
                        self.cli(client, url, "set-workspace-config", "--image-size-gib", "0", success=False)
                        self.cli(client, url, "set-workspace-config", "--image-root", "/sys", success=False)
                        for changes in ({"version": 8}, {"unknown": 1}, {"max_image_size_bytes": 1}):
                            with self.assertRaises(urllib.error.HTTPError) as error:
                                helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "pool", "changes": changes})
                            self.assertEqual(error.exception.code, 400)
                        self.assertEqual(self.cli(client, url, "get-workspace-config", "--node-id", "node"), expected)
                        # Successful update acknowledgement must survive abrupt controller exit.
                        process.kill()
                        process.wait(timeout=5)
                    finally:
                        helper.stop_controller(process)
                    process, url, _ = helper.start_controller(server, tmp)
                    try:
                        self.assertEqual(self.cli(client, url, "get-workspace-config", "--node-id", "node"), expected)
                    finally:
                        helper.stop_controller(process)

    def test_read_and_update_require_admin_credential(self):
        helper = parity.FunctionalParityTests()
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as tmp:
                process, url, _ = helper.start_controller(server, tmp)
                fixture = helper.fixtures[url]
                fixture.pool("pool")
                fixture.pool("other")
                regular = fixture.user_token("viewer")
                other = fixture.user_token("other")
                try:
                    for client in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
                        for token in (regular, other, "fixture-missing"):
                            self.cli(client, url, "get-workspace-config", "--token", token, success=False)
                            self.cli(client, url, "set-workspace-config", "--image-size-gib", "11", "--token", token, success=False)
                        result = self.cli(client, url, "get-workspace-config", "--token", fixture.admin)
                        self.assertEqual(result["desired_config"]["version"], 0)
                finally:
                    helper.stop_controller(process)
