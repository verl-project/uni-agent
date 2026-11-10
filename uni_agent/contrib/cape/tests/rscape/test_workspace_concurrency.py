# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
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


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_LIVE_TEST") == "1", "requires real mounted workspaces")
class WorkspaceConcurrencyTests(unittest.TestCase):
    def test_sixteen_sessions_keep_mount_identity_and_release_all_writers(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                self.exercise(binary)

    def exercise(self, binary):
        helper = parity.FunctionalParityTests()
        root = Path(tempfile.mkdtemp(prefix="cape-task16-concurrency-"))
        controller = node = None
        passed = False
        try:
            controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "120")
            helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "parallel", "changes": {
                "image_root": str(root / "images"), "image_size_bytes": 64 * 1024 ** 2}})
            with (root / "node.log").open("w") as stream:
                node = subprocess.Popen([binary, "node", "enroll", *helper.node_credentials(url, "parallel-node", "parallel"), "--controller-url", url, "--pool", "parallel", "--node-id", "parallel-node",
                    "--cpu-cores", "8", "--memory-gb", "16", "--runtime-adapter", "local-process", "--heartbeat-interval", "0.02",
                    "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs")], stdout=stream, stderr=stream)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                self.assertIsNone(node.poll(), (root / "node.log").read_text())
                config = helper.request("GET", url, "/v1/admin/workspace-config?pool=parallel")
                if config["nodes"] and config["nodes"][0]["state"] == "applied":
                    break
                time.sleep(0.05)
            else:
                self.fail("workspace readiness timed out")

            def command(index, warm):
                code = ("from pathlib import Path; import sys; assert Path('marker').read_text()==sys.argv[1]" if warm else
                        "from pathlib import Path; import sys,time; Path('marker').write_text(sys.argv[1]); time.sleep(0.1)")
                start = time.perf_counter()
                request = helper.request("POST", url, "/v1/submit", {"pool": "parallel", "user_id": "alice", "session_id": str(index), "reuse": True,
                    "command": ["python3", "-c", code, str(index)], "resources": {"gpus": 0, "cpu_cores": 0.5, "memory_gb": 0.1,
                        "runtime_adapter": "local-process", "isolation_policy": "none"}})
                result = helper.wait_state(url, request["request_id"], "COMPLETED", timeout=45)
                return result, time.perf_counter() - start

            health = []
            waves = []
            snapshots = []
            # Provision identity before concurrent submission and latency measurement.
            helper.fixtures[url].user_file("alice")
            for warm in (False, True):
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = [executor.submit(command, index, warm) for index in range(16)]
                    while not all(future.done() for future in futures):
                        start = time.perf_counter()
                        helper.request("GET", url, "/health")
                        health.append(time.perf_counter() - start)
                        time.sleep(0.01)
                    waves.append([future.result() for future in futures])
                entries = {}
                for status, _ in waves[-1]:
                    identity = status["workspace"]["id"]
                    directory = root / "images" / ("cape-workspaces-" + str(os.geteuid())) / identity
                    entries[identity] = json.loads((directory / "manifest.json").read_text())
                    self.assertLessEqual(len(list(directory.iterdir())) + 3, 12)
                snapshots.append(entries)
            self.assertEqual(set(snapshots[0]), set(snapshots[1]))
            for identity, before in snapshots[0].items():
                after = snapshots[1][identity]
                self.assertEqual((before["mount_id"], before["daemon_pid"]), (after["mount_id"], after["daemon_pid"]))
            self.assertLessEqual(max(health), 0.5)
            self.assertEqual(len([entry for entry in mount_entries() if entry["target"].startswith(str(root / "images") + "/")]), 16)
            rss = []
            cpu = []
            for manifest in snapshots[1].values():
                pid = manifest["daemon_pid"]
                memory = Path("/proc/{}/smaps_rollup".format(pid)).read_text()
                rss.append(next(int(line.split()[1]) * 1024 for line in memory.splitlines() if line.startswith("Rss:")))
                fields = Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()
                cpu.append((int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"))
            self.assertLessEqual(max(rss), 256 * 1024 ** 2)
            descriptors_before = len(list(Path("/proc/{}/fd".format(node.pid)).iterdir()))
            for identity in snapshots[1]:
                helper.request("POST", url, "/v1/workspaces/" + identity + "/release", {})
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if all(helper.request("GET", url, "/v1/workspaces/" + identity)["state"] == "retained" for identity in snapshots[1]):
                    break
                time.sleep(0.05)
            self.assertFalse(any(entry["target"].startswith(str(root / "images") + "/") for entry in mount_entries()))
            descriptors_after = list(Path("/proc/{}/fd".format(node.pid)).iterdir())
            for descriptor in descriptors_after:
                try:
                    self.assertNotIn(str(root / "images"), os.readlink(descriptor))
                except FileNotFoundError:
                    pass
            print(json.dumps({"binary": binary, "sessions": 16, "commands_per_session": 2, "stable_mount_identity": "passed",
                "cold_envelope_seconds": [elapsed for _, elapsed in waves[0]], "warm_envelope_seconds": [elapsed for _, elapsed in waves[1]],
                "controller_health_max_seconds": max(health), "fuse_rss_bytes": rss, "fuse_cpu_seconds": cpu,
                "agent_descriptors_before_release": descriptors_before, "agent_descriptors_after_release": len(descriptors_after),
                "image_descriptors_after_release": 0, "workspace_mounts_after_release": 0}), flush=True)
            passed = True
        finally:
            if node is not None and node.poll() is None:
                node.terminate()
                try:
                    node.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    node.kill(); node.wait(timeout=10)
            if controller is not None:
                helper.stop_controller(controller)
            mounts = [entry for entry in mount_entries() if entry["target"].startswith(str(root) + "/")]
            if passed and not mounts:
                shutil.rmtree(root)
            else:
                print(json.dumps({"retained_concurrency_root": str(root), "mounts": mounts,
                                  "node_log": (root / "node.log").read_text()[-10000:]}), flush=True)
            self.assertFalse(mounts)
