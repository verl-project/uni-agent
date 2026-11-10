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

from tests.rscape import test_functional_parity as parity
from pycape.workspace_fs import mount_entries
from tests.workspace_fixtures import copy_workspace_evidence


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_MAINTENANCE_TEST") == "1", "requires real workspace maintenance environment")
class WorkspaceMaintenanceParityTests(unittest.TestCase):
    def test_command_boundary_growth_and_retained_maintenance(self):
        cells = [("local-process", "none")]
        if os.environ.get("CAPE_WORKSPACE_MAINTENANCE_MATRIX") == "1":
            cells += [("bubblewrap", "host-stateless"), ("bubblewrap", "host-stateful"),
                      ("apptainer", "image-stateless"), ("apptainer", "image-stateful"),
                      ("podman", "image-stateless"), ("podman", "image-stateful")]
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            for adapter, policy in cells:
                with self.subTest(binary=binary, adapter=adapter, policy=policy):
                    self.run_cell(binary, adapter, policy)

    def run_cell(self, binary, adapter, policy):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-maintenance-parity-"))
        controller = node = None
        passed = False
        try:
            controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "180")
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "maint", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 64*1024**2,
                "maintenance": {"auto_grow": True, "grow_threshold_percent": 60, "grow_target_percent": 40,
                                "grow_min_bytes": 64*1024**2, "backing_reserve_bytes": 0}}})
            image = os.environ.get("CAPE_WORKSPACE_NESTED_IMAGE", "docker://ubuntu:24.04")
            args = [binary, "node", "enroll", *helper.node_credentials(url, "node", "maint"), "--controller-url", url, "--pool", "maint", "--node-id", "node",
                    "--runtime-adapter", adapter, "--cpu-cores", "4", "--memory-gb", "8", "--heartbeat-interval", "0.05",
                    "--state-path", str(root / "state.json"), "--runtime-dir", str(root / "runtime"), "--log-dir", str(root / "logs")]
            if adapter == "apptainer":
                args += ["--apptainer-binary", os.environ.get("CAPE_ENVIRONMENT_APPTAINER", "/usr/bin/apptainer"),
                         "--apptainer-probe-image", image, "--apptainer-cache", str(root / "apptainer-cache"), "--apptainer-tmp", str(root / "apptainer-tmp")]
            if adapter == "podman":
                args += ["--podman-binary", os.environ.get("CAPE_ENVIRONMENT_PODMAN", "/usr/bin/podman"),
                         "--podman-probe-image", image, "--podman-graphroot", str(root / "podman-graph"),
                         "--podman-runroot", str(root / "podman-run"), "--podman-tmpdir", str(root / "podman-tmp")]
            with (root / "node.log").open("w") as stream:
                node = subprocess.Popen(args, stdout=stream, stderr=stream)
            deadline = time.monotonic()+180
            while time.monotonic()<deadline:
                self.assertIsNone(node.poll(), (root / "node.log").read_text())
                if helper.request("GET", url, "/v1/pools/maint/utilization")["active_nodes"] == 1:
                    break
                time.sleep(.05)
            else:
                self.fail("node readiness timeout: " + (root / "node.log").read_text())

            def submit(command, identity=None):
                body = {"pool": "maint", "user_id": "alice", "session_id": "same", "reuse": True,
                        "command": ["/bin/sh", "-ec", command],
                        "resources": {"gpus": 0, "runtime_adapter": adapter, "isolation_policy": policy}}
                if adapter in ("apptainer", "podman"):
                    body["image"] = image
                if identity:
                    body["workspace"] = {"id": identity}
                return helper.request("POST", url, "/v1/submit", body)

            stateful = policy in ("host-stateful", "image-stateful")
            persistent = "mkdir -p /opt/cape-maintenance-test; printf preserved > /opt/cape-maintenance-test/marker; " if stateful else ""
            first = submit(persistent + 'dd if=/dev/urandom of=payload bs=1M count=40 status=none; '
                           'sha256sum payload > checksum; chmod 751 payload; ln payload hard; ln -s payload link; '
                           'printf first-started; sleep 2; printf first-finished')
            running = helper.wait_state(url, first["request_id"], "RUNNING", timeout=180)
            identity = running["workspace"]["id"]
            second = submit(('test "$(cat /opt/cape-maintenance-test/marker)" = preserved; ' if stateful else '') +
                            'sha256sum -c checksum; test "$(stat -c %a payload)" = 751; '
                            'test "$(stat -c %i payload)" = "$(stat -c %i hard)"; test "$(readlink link)" = payload; '
                            'dd if=/dev/zero of=new-capacity bs=1M count=64 status=none; printf second-finished', identity)
            pending_while_running = False
            observations = []
            deadline = time.monotonic()+10
            while time.monotonic()<deadline:
                observed = helper.request("GET", url, "/v1/workspaces/"+identity)
                first_state = helper.request("GET", url, "/v1/requests/"+first["request_id"])
                sample = {"request": first_state["state"], "workspace": observed["state"],
                          "operation": observed.get("operation", {}).get("state"),
                          "space": (observed.get("space") or {}).get("internal")}
                if not observations or sample != observations[-1]:
                    observations.append(sample)
                if first_state["state"] != "RUNNING":
                    break
                if observed.get("operation", {}).get("state") == "pending":
                    pending_while_running = True
                    self.assertEqual(observed["state"], "mounted")
                    self.assertEqual(observed.get("current_image_size_bytes", 64*1024**2), 64*1024**2)
                time.sleep(.05)
            self.assertTrue(pending_while_running, "automatic threshold must request maintenance while command 1 runs: " + json.dumps(observations))
            first_done = helper.wait_state(url, first["request_id"], "COMPLETED", timeout=180)
            second_done = helper.wait_state(url, second["request_id"], "COMPLETED", timeout=180)
            self.assertEqual(first_done["execution_session_id"], second_done["execution_session_id"])
            state = helper.request("GET", url, "/v1/workspaces/"+identity)
            self.assertGreater(state["current_image_size_bytes"], 64*1024**2)
            self.assertEqual(state["config"]["image_size_bytes"], 64*1024**2)
            self.assertEqual(state["operation"]["state"], "completed")
            self.assertGreaterEqual(second_done["started_at"], state["operation"]["completed_at"])
            self.assertGreaterEqual(second_done["started_at"], first_done["ended_at"])
            helper.request("POST", url, "/v1/workspaces/"+identity+"/release", {})
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                state=helper.request("GET",url,"/v1/workspaces/"+identity)
                if state["state"]=="retained":break
                time.sleep(.05)
            self.assertEqual(state["state"],"retained")
            for action,target in (("grow",256*1024**2),("reclaim",None),("shrink",128*1024**2)):
                operation=helper.request("POST",url,"/v1/workspaces/"+identity+"/"+action,{"target_bytes":target})
                deadline=time.monotonic()+120
                while time.monotonic()<deadline:
                    state=helper.request("GET",url,"/v1/workspaces/"+identity)
                    if state["operation"]["id"]==operation["operation"]["id"] and state["operation"]["state"]!="pending":break
                    time.sleep(.05)
                self.assertEqual(state["operation"]["state"],"completed",state)
            self.assertEqual(state["current_image_size_bytes"],128*1024**2)
            passed=True
            print(json.dumps({"binary":binary,"adapter":adapter,"policy":policy,"boundary_growth":"passed",
                              "retained_grow_reclaim_shrink":"passed","persistent_runtime_files":"passed" if stateful else "not-applicable",
                              "workspace_id_preserved":True,"creation_config_preserved":True}),flush=True)
        finally:
            for process in (node,controller):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                if process is not None:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None:stream.close()
            output = os.environ.get("CAPE_WORKSPACE_MAINTENANCE_OUTPUT")
            if output:
                destination = Path(output) / (Path(binary).name + "-" + adapter + "-" + policy)
                copy_workspace_evidence(root, destination)
            mounts=[r for r in mount_entries() if r["target"].startswith(str(root)+"/")]
            if mounts:
                raise AssertionError("test workspace mounts remain: "+str(root))
            if passed:shutil.rmtree(root)
            else:print("maintenance diagnostic root="+str(root),flush=True)
