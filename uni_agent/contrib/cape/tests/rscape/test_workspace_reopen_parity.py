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
class WorkspaceReopenParityTests(unittest.TestCase):
    def test_retained_images_reopen_in_both_language_directions(self):
        for binaries in ((parity.PYCAPE_BIN, parity.RSCAPE_BIN), (parity.RSCAPE_BIN, parity.PYCAPE_BIN)):
            with self.subTest(binaries=binaries):
                self.exercise(binaries)

    def exercise(self, binaries):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-task16-cross-language-"))
        passed = False
        controller = node = None
        identity = None
        try:
            for index, binary in enumerate(binaries):
                controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "120")
                if index == 0:
                    helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "reopen", "changes": {
                        "image_root": str(root / "images"), "image_size_bytes": 64 * 1024 ** 2}})
                with (root / ("node-" + str(index) + ".log")).open("w") as stream:
                    node = subprocess.Popen([binary, "node", "enroll", *helper.node_credentials(url, "reopen-node", "reopen"), "--controller-url", url, "--pool", "reopen",
                        "--node-id", "reopen-node", "--runtime-adapter", "local-process", "--heartbeat-interval", "0.05",
                        "--cpu-cores", "4", "--memory-gb", "8", "--state-path", str(root / "state.json"),
                        "--log-dir", str(root / "logs")], stdout=stream, stderr=stream)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    self.assertIsNone(node.poll(), (root / ("node-" + str(index) + ".log")).read_text())
                    config = helper.request("GET", url, "/v1/admin/workspace-config?pool=reopen")
                    if config["nodes"] and config["nodes"][0]["state"] == "applied":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("workspace readiness timed out")
                shell = "printf cross-language > marker; chmod 750 marker; ln marker hard; ln -s marker link" if index == 0 else (
                    'test "$(cat marker)" = cross-language; test "$(stat -c %a marker)" = 750; '
                    'test "$(stat -c %i marker)" = "$(stat -c %i hard)"; test "$(readlink link)" = marker')
                body = {"pool": "reopen", "user_id": "alice", "session_id": "phase-" + str(index), "reuse": False,
                        "command": ["/bin/sh", "-ec", shell], "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}}
                if identity:
                    body["workspace"] = {"id": identity}
                submitted = helper.request("POST", url, "/v1/submit", body)
                complete = helper.wait_state(url, submitted["request_id"], "COMPLETED", timeout=30)
                if identity:
                    self.assertEqual(identity, complete["workspace"]["id"])
                identity = complete["workspace"]["id"]
                helper.request("POST", url, "/v1/workspaces/" + identity + "/release", {})
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    saved = helper.request("GET", url, "/v1/workspaces/" + identity)
                    if saved["state"] == "retained":
                        break
                    time.sleep(0.05)
                self.assertEqual(saved["state"], "retained", saved)
                self.assertFalse(any(entry["target"].startswith(str(root) + "/") for entry in mount_entries()))
                if index == 0:
                    # SIGTERM now retires the enrollment. Restart the idle process
                    # without a retirement intent, after verified clean retention.
                    node.kill()
                else:
                    node.terminate()
                node.wait(timeout=30); node = None
                helper.stop_controller(controller); controller = None
                self.assertFalse(any(entry["target"].startswith(str(root) + "/") for entry in mount_entries()))
            print(json.dumps({"writer": binaries[0], "reader": binaries[1], "content_links_modes": "passed", "clean_release": "passed"}), flush=True)
            passed = True
        finally:
            if node is not None and node.poll() is None:
                node.terminate()
                try:
                    node.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    node.kill(); node.wait(timeout=5)
            if controller is not None:
                helper.stop_controller(controller)
            mounted = any(entry["target"].startswith(str(root) + "/") for entry in mount_entries())
            if passed and not mounted:
                shutil.rmtree(root)
            else:
                print("retained_cross_language_root=" + str(root), flush=True)
            self.assertFalse(mounted)
