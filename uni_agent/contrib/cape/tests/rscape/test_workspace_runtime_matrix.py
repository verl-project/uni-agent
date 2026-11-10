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


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_RUNTIME_MATRIX") == "1", "requires real GB200 runtime environment")
class WorkspaceRuntimeMatrixTests(unittest.TestCase):
    def test_native_workspace_lifecycle_all_runtime_policies(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            for adapter, policy in (("local-process", "none"), ("bubblewrap", "host-stateless"),
                                    ("bubblewrap", "host-stateful"), ("apptainer", "image-stateless"),
                                    ("apptainer", "image-stateful"), ("podman", "image-stateless")):
                with self.subTest(binary=binary, adapter=adapter, policy=policy):
                    self.run_cell(binary, adapter, policy)

    def run_cell(self, binary, adapter, policy):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-task16-matrix-"))
        controller = node = None
        passed = False
        image = os.environ.get("CAPE_WORKSPACE_NESTED_IMAGE", "docker://ubuntu:24.04")
        try:
            controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "120")
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "matrix", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 128 * 1024 ** 2}})
            args = [binary, "node", "enroll", *helper.node_credentials(url, "matrix-node", "matrix"), "--controller-url", url, "--pool", "matrix", "--node-id", "matrix-node",
                    "--cpu-cores", "4", "--memory-gb", "8", "--heartbeat-interval", "0.05", "--runtime-adapter", adapter,
                    "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs"), "--runtime-dir", str(root / "runtime")]
            if adapter == "apptainer":
                args += ["--apptainer-binary", os.environ.get("CAPE_ENVIRONMENT_APPTAINER", "/usr/bin/apptainer"),
                         "--apptainer-probe-image", image, "--apptainer-cache", "/opt/cape-task16/matrix-apptainer-cache",
                         "--apptainer-tmp", str(root / "apptainer-tmp")]
            if adapter == "podman":
                args += ["--podman-binary", os.environ.get("CAPE_ENVIRONMENT_PODMAN", "/usr/bin/podman"),
                         "--podman-probe-image", image, "--podman-graphroot", "/opt/cape-task16/matrix-podman-graph",
                         "--podman-runroot", "/opt/cape-task16/matrix-podman-run", "--podman-tmpdir", str(root / "podman-tmp")]
            with (root / "node.log").open("w") as stream:
                node = subprocess.Popen(args, stdout=stream, stderr=stream)
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                self.assertIsNone(node.poll(), (root / "node.log").read_text())
                status = helper.request("GET", url, "/v1/admin/workspace-config?pool=matrix")
                if status["nodes"] and status["nodes"][0]["state"] == "applied":
                    break
                time.sleep(0.1)
            else:
                self.fail("workspace readiness failed: " + (root / "node.log").read_text())

            def run(task, shell, identity=None, cwd=None):
                body = {"pool": "matrix", "user_id": "alice", "session_id": task, "reuse": True,
                        "command": ["/bin/sh", "-ec", shell], "resources": {"gpus": 0, "runtime_adapter": adapter,
                        "isolation_policy": policy}, "image": image if adapter in ("apptainer", "podman") else None}
                if identity:
                    body["workspace"] = {"id": identity}
                if cwd:
                    body["cwd"] = cwd
                request = helper.request("POST", url, "/v1/submit", body)
                try:
                    return helper.wait_state(url, request["request_id"], "COMPLETED", timeout=60)
                except AssertionError:
                    print(json.dumps({"root": str(root), "adapter": adapter, "policy": policy,
                                      "logs": helper.request("GET", url, "/v1/requests/" + request["request_id"] + "/logs"),
                                      "node_log": (root / "node.log").read_text()}), flush=True)
                    raise

            first = run("warm", 'test "$PWD" = "$CAPE_WORKSPACE_DIR"; mkdir data; printf hello > data/file; '
                        'ln data/file data/hard; ln -s file data/link; chmod 751 data/file; mv data/file data/renamed; '
                        'printf world >> data/hard; test "$(cat data/renamed)" = helloworld; printf "#!/bin/sh\\necho executable\\n" > script; chmod +x script; ./script')
            identity = first["workspace"]["id"]
            initial = helper.request("GET", url, "/v1/workspaces/" + identity)
            second = run("warm", 'test "$(cat data/hard)" = helloworld; test "$(stat -c %a data/renamed)" = 751; '
                         'test "$(readlink data/link)" = file; test "$(stat -c %i data/hard)" = "$(stat -c %i data/renamed)"; ./script')
            self.assertEqual(first["execution_session_id"], second["execution_session_id"])
            self.assertEqual(initial["revision"], helper.request("GET", url, "/v1/workspaces/" + identity)["revision"], "warm execution remounted or rewrote lifecycle")
            run("warm", 'test "$PWD" = /tmp; test "$(cat "$CAPE_WORKSPACE_DIR/data/hard")" = helloworld', cwd="/tmp")

            def release():
                helper.request("POST", url, "/v1/workspaces/" + identity + "/release", {})
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    record = helper.request("GET", url, "/v1/workspaces/" + identity)
                    if record["state"] == "retained":
                        return record
                    time.sleep(0.05)
                self.fail("workspace release failed: " + repr(record))

            release()
            reopened = run("reopened", 'test "$PWD" = "$CAPE_WORKSPACE_DIR"; test "$(cat data/renamed)" = helloworld; '
                           'test "$(stat -c %a data/renamed)" = 751; test "$(readlink data/link)" = file; ./script', identity)
            self.assertNotEqual(first["execution_session_id"], reopened["execution_session_id"])
            release()
            self.assertFalse(any(entry["target"].startswith(str(root) + "/images/") for entry in mount_entries()))
            print(json.dumps({"binary": binary, "adapter": adapter, "policy": policy, "nested_image": image,
                              "default_cwd": "passed", "explicit_cwd": "passed", "warm_reuse": "passed",
                              "retained_reopen": "passed", "content_links_modes": "passed", "unmount": "passed"}), flush=True)
            passed = True
        finally:
            if node is not None and node.poll() is None:
                node.terminate()
                try:
                    node.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    node.kill(); node.wait(timeout=10); passed = False
            if controller is not None:
                helper.stop_controller(controller)
            mounted = any(entry["target"].startswith(str(root) + "/") for entry in mount_entries())
            if passed and not mounted:
                shutil.rmtree(root)
            else:
                print("retained_workspace_matrix_root=" + str(root), flush=True)
            self.assertFalse(mounted, "test-owned mounts remain at " + str(root))
