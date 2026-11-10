# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real CPU/FUSE state retention; never exercises a GPU kernel."""

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


@unittest.skipUnless(os.environ.get("CAPE_EXECUTION_RETENTION_LIVE") == "1", "requires real FUSE and stateful runtimes")
class ExecutionRetentionLiveTests(unittest.TestCase):
    def test_stateful_layer_and_workspace_survive_suspend_agent_restart_and_purge(self):
        for writer, reader in ((parity.PYCAPE_BIN, parity.RSCAPE_BIN), (parity.RSCAPE_BIN, parity.PYCAPE_BIN)):
            for adapter, policy in (("bubblewrap", "host-stateful"), ("apptainer", "image-stateful")):
                with self.subTest(writer=writer, reader=reader, adapter=adapter):
                    self.run_cell(writer, reader, adapter, policy)

    def run_cell(self, writer, reader, adapter, policy):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-retention-live-"))
        controller = node = None
        passed = False
        image = os.environ.get("CAPE_EXECUTION_RETENTION_IMAGE", "docker://ubuntu:24.04")
        logs = []
        agents = []
        try:
            controller, url, _ = helper.start_controller(writer, str(root), "--idle-session-timeout-seconds", "120")
            fixture = helper.fixtures[url]
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "retention", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 64 * 1024 ** 2}})
            owner = fixture.user_token("alice")

            def start_agent(binary, phase):
                args = [binary, "node", "enroll", *helper.node_credentials(url, "retained-node", "retention"),
                    "--controller-url", url, "--pool", "retention", "--node-id", "retained-node",
                    "--cpu-cores", "4", "--memory-gb", "8", "--heartbeat-interval", "0.05",
                    "--runtime-adapter", adapter, "--state-path", str(root / "agent.json"),
                    "--log-dir", str(root / "command-logs"), "--runtime-dir", str(root / "runtime"),
                    "--execution-state-dir", str(root / "retained")]
                if adapter == "apptainer":
                    args += ["--apptainer-probe-image", image, "--apptainer-cache",
                             os.environ.get("CAPE_EXECUTION_RETENTION_CACHE", str(root / "image-cache")),
                             "--apptainer-tmp", str(root / "apptainer-tmp")]
                    if os.environ.get("CAPE_APPTAINER_UNSQUASH") == "1":
                        args.append("--apptainer-unsquash")
                path = root / ("node-" + phase + ".log")
                logs.append(path)
                with path.open("w") as stream:
                    process = subprocess.Popen(args, stdout=stream, stderr=stream, start_new_session=True)
                agents.append(process)
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    self.assertIsNone(process.poll(), path.read_text())
                    state = helper.request("GET", url, "/v1/admin/workspace-config?pool=retention")
                    if state["nodes"] and state["nodes"][0]["state"] == "applied":
                        return process
                    time.sleep(.1)
                process.terminate()
                process.wait(timeout=30)
                self.fail("runtime readiness timed out: " + path.read_text())

            node = start_agent(writer, "writer")

            def run(sequence, shell, selected_policy=policy, workspace=None):
                body = {"pool": "retention", "user_id": "alice", "session_id": sequence,
                        "command": ["/bin/sh", "-ec", shell], "resources": {"gpus": 0,
                        "runtime_adapter": adapter, "isolation_policy": selected_policy}}
                if adapter == "apptainer":
                    body["image"] = image
                if workspace:
                    body["workspace"] = {"id": workspace}
                request = helper.request("POST", url, "/v1/submit", body)
                try:
                    deadline = time.monotonic() + 300
                    while time.monotonic() < deadline:
                        status = fixture.request("GET", "/v1/requests/" + request["request_id"], token=owner)
                        if status["state"] == "COMPLETED":
                            return status
                        self.assertNotIn(status["state"], ("FAILED", "CANCELLED", "TIMED_OUT"), repr(status))
                        time.sleep(.05)
                    self.fail("command completion timed out: " + repr(status))
                except AssertionError:
                    print(json.dumps(fixture.request("GET", "/v1/requests/" + request["request_id"] + "/logs", token=owner)), flush=True)
                    raise

            def wait_execution(identity, desired):
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    state = fixture.request("GET", "/v1/sessions", token=owner)
                    execution = next(e for s in state["sessions"] for e in s["executions"] if e["execution_session_id"] == identity)
                    if execution["state"] == desired:
                        return execution
                    self.assertIsNone(node.poll(), logs[-1].read_text())
                    time.sleep(.05)
                self.fail("execution did not reach " + desired + ": " + repr(execution))

            first = run("sequence", 'printf workspace > "$CAPE_WORKSPACE_DIR/value"; '
                        'mkdir -p /opt/cape-retained-demo; printf layer > /opt/cape-retained-demo/value')
            execution, workspace = first["execution_session_id"], first["workspace"]["id"]
            fixture.request("POST", "/v1/executions/" + execution + "/suspend", {}, owner)
            wait_execution(execution, "SUSPENDED")
            self.assertFalse(any(m["target"].startswith(str(root) + "/") for m in mount_entries()))
            self.assertEqual(fixture.request("GET", "/v1/workspaces/" + workspace, token=owner)["state"], "retained")
            checked = fixture.request("POST", "/v1/workspaces/" + workspace + "/check", {}, owner)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                record = fixture.request("GET", "/v1/workspaces/" + workspace, token=owner)
                if record.get("operation", {}).get("state") == "completed":
                    break
                self.assertNotEqual(record.get("operation", {}).get("state"), "failed", repr(record))
                time.sleep(.05)
            self.assertEqual(record["operation"]["id"], checked["operation"]["id"])
            self.assertEqual(record["operation"]["state"], "completed")
            self.assertEqual(record["state"], "retained")
            # Crash only this test-owned, already-suspended agent. SIGTERM would
            # retire the allocation identity, which is not an agent restart test.
            node.kill()
            node.wait(timeout=10)
            node = start_agent(reader, "reader")
            second = run("sequence", 'test "$(cat "$CAPE_WORKSPACE_DIR/value")" = workspace; '
                         'test "$(cat /opt/cape-retained-demo/value)" = layer; echo retained-ok')
            self.assertEqual(second["execution_session_id"], execution)
            self.assertEqual(second["workspace"]["id"], workspace)
            fixture.request("POST", "/v1/executions/" + execution + "/suspend", {}, owner)
            wait_execution(execution, "SUSPENDED")
            fixture.request("POST", "/v1/executions/" + execution + "/purge", {}, owner)
            wait_execution(execution, "PURGED")
            self.assertEqual(fixture.request("GET", "/v1/workspaces/" + workspace, token=owner)["state"], "retained")
            plain = "host-stateless" if adapter == "bubblewrap" else "image-stateless"
            third = run("fresh", 'test "$(cat "$CAPE_WORKSPACE_DIR/value")" = workspace; '
                        'test ! -e /opt/cape-retained-demo/value', plain, workspace)
            self.assertNotEqual(third["execution_session_id"], execution)
            fixture.request("POST", "/v1/executions/" + third["execution_session_id"] + "/suspend", {}, owner)
            wait_execution(third["execution_session_id"], "SUSPENDED")
            fixture.request("POST", "/v1/executions/" + third["execution_session_id"] + "/purge", {}, owner)
            wait_execution(third["execution_session_id"], "PURGED")
            self.assertFalse(any(m["target"].startswith(str(root) + "/") for m in mount_entries()))
            print(json.dumps({"writer": writer, "reader": reader, "adapter": adapter,
                "layer_retention": "passed", "workspace_retention": "passed", "agent_restart": "passed",
                "explicit_purge": "passed", "offline_check_while_suspended": "passed", "gpus": 0}), flush=True)
            passed = True
        finally:
            for process in reversed(agents):
                if process.poll() is not None:
                    continue
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                    passed = False
            if controller is not None:
                helper.stop_controller(controller)
            mounted = any(m["target"].startswith(str(root) + "/") for m in mount_entries())
            destination = os.environ.get("CAPE_EXECUTION_RETENTION_EVIDENCE")
            if destination:
                output = Path(destination) / root.name
                output.mkdir(parents=True, exist_ok=True)
                for path in logs:
                    if path.exists():
                        shutil.copyfile(path, output / path.name)
            if passed and not mounted:
                shutil.rmtree(root)
            else:
                print("retained_execution_test_root=" + str(root), flush=True)
            self.assertFalse(mounted, "test-owned mounts remain at " + str(root))


if __name__ == "__main__":
    unittest.main()
