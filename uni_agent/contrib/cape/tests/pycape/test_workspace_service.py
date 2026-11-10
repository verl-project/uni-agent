# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest

from pycape.workspace_fs import mount_entries
from tests.rscape import test_functional_parity as parity


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_LIVE_TEST") == "1", "requires the verified fuse2fs test environment")
class WorkspaceServiceTests(unittest.TestCase):
    def test_one_shot_agents_acknowledge_clean_retention_before_exit(self):
        helper = parity.FunctionalParityTests()
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                root = Path(tempfile.mkdtemp(prefix="cape-task16-once-"))
                controller = None
                passed = False
                try:
                    controller, url, journal = helper.start_controller(binary, str(root))
                    helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "once", "changes": {
                        "image_root": str(root / "images"), "image_size_bytes": 64 * 1024 ** 2}})
                    node_args = [binary, "node", "enroll", *helper.node_credentials(url, "once-node", "once"), "--controller-url", url, "--pool", "once", "--node-id", "once-node",
                        "--runtime-adapter", "local-process", "--once", "--cpu-cores", "2", "--memory-gb", "4",
                        "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs")]
                    enrolled = subprocess.run(node_args, capture_output=True, text=True, timeout=45)
                    self.assertEqual(enrolled.returncode, 0, enrolled.stderr)
                    request = helper.request("POST", url, "/v1/submit", {"pool": "once", "user_id": "alice", "session_id": "once", "reuse": True,
                        "command": ["/bin/sh", "-c", "printf saved > marker"],
                        "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}})
                    # A one-shot heartbeat need not wait for the next scheduler tick.
                    # Queue the assignment before testing its execution/retention cleanup.
                    helper.wait_state(url, request["request_id"], "RUNNING", timeout=5)
                    node = subprocess.run(node_args + ["--resume"], capture_output=True, text=True, timeout=45)
                    self.assertEqual(node.returncode, 0, node.stderr)
                    status = helper.wait_state(url, request["request_id"], "COMPLETED", timeout=5)
                    workspace = helper.request("GET", url, "/v1/workspaces/" + status["workspace"]["id"])
                    self.assertEqual(workspace["state"], "retained", workspace)
                    with sqlite3.connect(str(journal)) as connection:
                        state = connection.execute("SELECT state FROM events WHERE event_type='session_state' AND entity_id=? ORDER BY id DESC LIMIT 1", (status["execution_session_id"],)).fetchone()
                    self.assertEqual(state[0], "STOPPED")
                    self.assertFalse(any(entry["target"].startswith(str(root) + "/") for entry in mount_entries()))
                    passed = True
                finally:
                    if controller is not None:
                        helper.stop_controller(controller)
                    mounted = any(entry["target"].startswith(str(root) + "/") for entry in mount_entries())
                    if passed and not mounted:
                        shutil.rmtree(root)
                    else:
                        print("retained_once_root=" + str(root), flush=True)
                    self.assertFalse(mounted)

    def test_default_workspace_warm_reuse_retention_and_live_configuration(self):
        binary = os.environ.get("CAPE_WORKSPACE_TEST_BIN", os.environ["PYCAPE_BIN"])
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-task16-service-"))
        controller = node = None
        passed = False
        try:
            controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "1")
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "workspace", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 64 * 1024 ** 2}})
            with (root / "node.log").open("w") as output:
                node = subprocess.Popen([binary, "node", "enroll", *helper.node_credentials(url, "workspace-node", "workspace"), "--controller-url", url, "--pool", "workspace",
                    "--node-id", "workspace-node", "--allocation-id", "workspace-allocation", "--runtime-adapter", "local-process",
                    "--cpu-cores", "4", "--memory-gb", "8", "--heartbeat-interval", "0.05",
                    "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs")], stdout=output, stderr=output)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    self.assertIsNone(node.poll(), (root / "node.log").read_text())
                    state = helper.request("GET", url, "/v1/admin/workspace-config?pool=workspace")
                    if state["nodes"] and state["nodes"][0]["state"] == "applied":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("workspace configuration never applied: " + (root / "node.log").read_text())

                def command(task, code, workspace=None, reuse=True):
                    body = {"pool": "workspace", "user_id": "alice", "session_id": task, "reuse": reuse,
                            "command": ["python3", "-c", code], "resources": {"gpus": 0, "isolation_policy": "none", "runtime_adapter": "local-process"}}
                    if workspace is not None:
                        body["workspace"] = {"id": workspace}
                    request = helper.request("POST", url, "/v1/submit", body)
                    try:
                        status = helper.wait_state(url, request["request_id"], "COMPLETED", timeout=30)
                    except AssertionError:
                        print(json.dumps({"logs": helper.request("GET", url, "/v1/requests/" + request["request_id"] + "/logs"),
                                          "node_log": (root / "node.log").read_text()}), flush=True)
                        raise
                    return status

                first = command("warm", "import os; from pathlib import Path; p=Path(os.environ['CAPE_WORKSPACE_DIR']); assert str(p)==os.getcwd(); (p/'marker').write_text('hello'); print('first')")
                identity = first["workspace"]["id"]
                second = command("warm", "from pathlib import Path; assert Path('marker').read_text()=='hello'; Path('marker').write_text('second')")
                self.assertEqual(second["execution_session_id"], first["execution_session_id"])
                self.assertEqual(second["workspace"]["id"], identity)
                helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "workspace", "changes": {
                    "image_root": str(root / "new-images"), "image_size_bytes": 128 * 1024 ** 2}})
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    retained = helper.request("GET", url, "/v1/workspaces/" + identity)
                    if retained["state"] == "retained":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("workspace did not become retained: " + (root / "node.log").read_text())
                self.assertEqual(retained["config"]["image_size_bytes"], 64 * 1024 ** 2)
                self.assertEqual(retained["config"]["image_root"], str(root / "images"))
                reopened = command("reopened", "from pathlib import Path; assert Path('marker').read_text()=='second'; print('reopened')", identity, reuse=False)
                self.assertNotEqual(reopened["execution_session_id"], first["execution_session_id"])
                self.assertEqual(reopened["workspace"]["id"], identity)
                fresh = command("fresh", "from pathlib import Path; assert not Path('marker').exists(); print('fresh')", reuse=False)
                self.assertNotEqual(fresh["workspace"]["id"], identity)
                self.assertEqual(fresh["workspace"]["config"]["image_size_bytes"], 128 * 1024 ** 2)
                # The old suspended execution still references the shared image.
                # Retention is not deletion authorization: explicitly purge it first.
                refused = subprocess.run([binary, "workspace", "delete", identity, "--controller-url", url,
                    "--token-file", str(helper.fixtures[url].user_file("alice"))], capture_output=True, text=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn("referenced", refused.stderr)
                owner = helper.fixtures[url].user_token("alice")
                helper.request("POST", url, "/v1/executions/" + first["execution_session_id"] + "/purge", {}, token=owner)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    logical = helper.request("GET", url, "/v1/sessions?session_id=warm", token=owner)
                    execution = next(item for item in logical["sessions"][0]["executions"]
                                     if item["execution_session_id"] == first["execution_session_id"])
                    if execution["state"] == "PURGED":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("referencing execution was not purged: " + repr(execution))
                preserved = command("reopened", "from pathlib import Path; assert Path('marker').read_text()=='second'")
                self.assertEqual(preserved["workspace"]["id"], identity)
                for identity in (identity, fresh["workspace"]["id"]):
                    subprocess.run([binary, "workspace", "release", identity, "--controller-url", url, "--token-file", str(helper.fixtures[url].user_file("alice"))], check=True, capture_output=True, text=True)
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        retained = helper.request("GET", url, "/v1/workspaces/" + identity)
                        if retained["state"] == "retained":
                            break
                        time.sleep(0.05)
                    else:
                        self.fail("explicit release did not retain workspace: " + repr(retained))
                    for action in ("check", "delete"):
                        submitted = subprocess.run([binary, "workspace", action, identity, "--controller-url", url, "--token-file", str(helper.fixtures[url].user_file("alice"))], check=True, capture_output=True, text=True)
                        operation = json.loads(submitted.stdout)["operation"]["id"]
                        deadline = time.monotonic() + 30
                        while time.monotonic() < deadline:
                            observed = helper.request("GET", url, "/v1/workspaces/" + identity)
                            if observed.get("operation", {}).get("id") == operation and observed["operation"]["state"] != "pending":
                                break
                            time.sleep(0.05)
                        self.assertEqual(observed["operation"]["state"], "completed", observed)
                        self.assertEqual(observed["state"], "deleted" if action == "delete" else "retained")
                print(json.dumps({"default_creation": "passed", "warm_reuse": "passed", "retained_reopen": "passed",
                                  "live_defaults_new_images_only": "passed", "maintenance": "passed", "binary": binary}), flush=True)
                passed = True
        finally:
            if node is not None and node.poll() is None:
                node.terminate()
                try:
                    node.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    node.kill()
                    node.wait(timeout=10)
                    passed = False
            if controller is not None:
                helper.stop_controller(controller)
            mounted = any(entry["target"].startswith(str(root) + "/") for entry in mount_entries())
            if passed and not mounted:
                shutil.rmtree(root)
            else:
                print("retained_workspace_service_root=" + str(root), flush=True)
            self.assertFalse(mounted, "test-owned workspace mounts remain at " + str(root))
