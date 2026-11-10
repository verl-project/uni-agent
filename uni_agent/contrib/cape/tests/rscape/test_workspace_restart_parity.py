# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

from pycape.workspace_fs import mount_entries
from tests.rscape import test_functional_parity as parity


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_LIVE_TEST") == "1", "requires real filesystem images")
class WorkspaceRestartParityTests(unittest.TestCase):
    def test_live_predecessor_is_fenced_until_explicit_safe_unmount(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                self.exercise(binary)

    def exercise(self, binary):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-task16-restart-"))
        controller = node = None
        mount = None
        passed = False
        try:
            controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "300")
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "restart", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 64 * 1024 ** 2}})
            args = [binary, "node", "enroll", "--controller-url", url, "--pool", "restart", "--node-id", "restart-node",
                    "--cpu-cores", "4", "--memory-gb", "8", "--runtime-adapter", "local-process", "--heartbeat-interval", "0.05",
                    "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs")]

            def launch():
                with (root / "node.log").open("a") as stream:
                    return subprocess.Popen(args + helper.node_credentials(url, "restart-node", "restart"), stdout=stream, stderr=stream)

            def wait_ready(previous_epoch=None):
                deadline = time.monotonic() + 60
                applied_heartbeat = None
                while time.monotonic() < deadline:
                    self.assertIsNone(node.poll(), (root / "node.log").read_text())
                    config = helper.request("GET", url, "/v1/admin/workspace-config?pool=restart")
                    entries = config["nodes"]
                    if entries and entries[0]["state"] == "applied":
                        if previous_epoch is not None:
                            self.assertEqual(entries[0]["agent_epoch"], previous_epoch, "resume retains the enrolled epoch")
                        if applied_heartbeat is None:
                            applied_heartbeat = entries[0]["last_heartbeat_at"]
                        elif entries[0]["last_heartbeat_at"] > applied_heartbeat:
                            return entries[0]["agent_epoch"]
                    time.sleep(0.05)
                self.fail("restarted agent did not finish registration/reconciliation")

            def wait_record(identity, state):
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    record = helper.request("GET", url, "/v1/workspaces/" + identity)
                    if record["state"] == state:
                        return record
                    time.sleep(0.05)
                self.fail(repr(record) + "\n" + (root / "node.log").read_text()[-5000:])

            def run(task, code, expected="COMPLETED", identity=None):
                body = {"pool": "restart", "user_id": "alice", "session_id": task, "reuse": identity is None,
                        "command": ["python3", "-c", code], "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}}
                if identity:
                    body["workspace"] = {"id": identity}
                request = helper.request("POST", url, "/v1/submit", body)
                return helper.wait_state(url, request["request_id"], expected, timeout=45)

            node = launch()
            epoch = wait_ready()
            first = run("original", "from pathlib import Path; Path('marker').write_text('retained-content')")
            identity = first["workspace"]["id"]
            record = helper.request("GET", url, "/v1/workspaces/" + identity)
            mount = Path(record["path"])
            manifest = json.loads((mount.parent / "manifest.json").read_text())
            old_pid = manifest["daemon_pid"]
            node.kill(); node.wait(timeout=10)
            self.assertIn(str(mount.parent / "workspace.img").encode(), Path("/proc/{}/cmdline".format(old_pid)).read_bytes())
            node = launch()
            wait_ready(epoch)
            failed = run("original", "from pathlib import Path; Path('must-not-run').write_text('unsafe')", "FAILED")
            self.assertEqual(failed["failure_kind"], "infrastructure")
            self.assertFalse((mount / "must-not-run").exists())
            self.assertEqual((mount / "marker").read_text(), "retained-content")
            self.assertIsNone(node.poll(), (root / "node.log").read_text())
            # Explicit operator-equivalent reconciliation of only this test-owned
            # mount, after the old command worker and both user commands have ended.
            subprocess.run([os.environ.get("CAPE_WORKSPACE_FUSERMOUNT_BIN", "fusermount3"), "-u", str(mount)], check=True, capture_output=True, text=True, timeout=10)
            wait_record(identity, "needs_check")
            helper.request("POST", url, "/v1/workspaces/" + identity + "/check", {})
            wait_record(identity, "retained")
            resumed = run("resumed", "from pathlib import Path; assert Path('marker').read_text()=='retained-content'", identity=identity)
            self.assertEqual(resumed["workspace"]["id"], identity)
            helper.request("POST", url, "/v1/workspaces/" + identity + "/release", {})
            wait_record(identity, "retained")
            passed = True
            print(json.dumps({"binary": binary, "agent_sigkill": "injected", "live_predecessor_writer_refused": "passed",
                              "explicit_owned_unmount_and_offline_check": "passed", "retained_reopen": "passed"}), flush=True)
        finally:
            if node is not None and node.poll() is None:
                node.terminate()
                try:
                    node.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    node.kill(); node.wait(timeout=5)
            if controller is not None:
                helper.stop_controller(controller)
            matches = [entry for entry in mount_entries() if entry["target"].startswith(str(root) + "/")]
            if passed and not matches:
                shutil.rmtree(root)
            else:
                print(json.dumps({"retained_restart_root": str(root), "mounts": matches,
                                  "node_log": (root / "node.log").read_text()[-5000:]}), flush=True)
            self.assertFalse(matches, "test-owned mounts remain")
