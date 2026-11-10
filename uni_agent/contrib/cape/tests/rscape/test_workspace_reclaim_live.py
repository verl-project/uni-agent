# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real FUSE gates; no GPU kernels and no filesystem doubles."""

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest

from tests.rscape import test_functional_parity as parity
from pycape.workspace_fs import mount_entries
from pycape.workspace_reclaim import maintenance_host_key, reclaim_signal


@unittest.skipUnless(os.environ.get("CAPE_WORKSPACE_RECLAIM_LIVE") == "1", "requires real FUSE reclaim environment")
class WorkspaceReclaimLiveTests(unittest.TestCase):
    def start(self, binary, size, enabled, threshold=32*1024**2):
        root = Path(tempfile.mkdtemp(prefix="cape-reclaim-live-"))
        def capture():
            output = os.environ.get("CAPE_WORKSPACE_RECLAIM_OUTPUT")
            if output:
                target = Path(output) / root.name
                target.mkdir(parents=True, exist_ok=True)
                if (root / "node.log").exists():
                    shutil.copyfile(root / "node.log", target / "node.log")
                if (root / "logs").is_dir():
                    shutil.copytree(root / "logs", target / "commands", dirs_exist_ok=True)
        self.addCleanup(capture)
        helper = parity.FunctionalParityTests()
        controller, url, _ = helper.start_controller(binary, str(root), "--idle-session-timeout-seconds", "180")
        self.addCleanup(helper.stop_controller, controller)
        helper.request("POST", url, "/v1/admin/workspace-config", {"pool": "reclaim", "changes": {
            "image_root": str(root / "images"), "image_size_bytes": size, "maintenance": {
                "reclaim_on_release": enabled, "reclaim_idle_seconds": 1, "reclaim_cooldown_seconds": 0,
                "reclaim_start_interval_seconds": 1, "reclaim_min_allocated_bytes": threshold,
                "backing_reserve_bytes": 0, "reclaim_low_yield_backoff_seconds": 0}}})
        stream = (root / "node.log").open("w")
        self.addCleanup(stream.close)
        node = subprocess.Popen([binary, "node", "enroll", *helper.node_credentials(url, "node", "reclaim"),
            "--controller-url", url, "--pool", "reclaim", "--node-id", "node", "--runtime-adapter", "local-process",
            "--cpu-cores", "4", "--memory-gb", "8", "--heartbeat-interval", "0.05",
            "--state-path", str(root / "agent.json"), "--log-dir", str(root / "logs")], stdout=stream, stderr=stream)
        self.addCleanup(helper.stop_controller, node)
        self.poll(lambda: helper.request("GET", url, "/v1/admin/workspace-config?pool=reclaim")["nodes"],
                  lambda rows: rows and rows[0]["state"] == "applied", 120)
        return root, helper, url

    def poll(self, read, done, timeout=120):
        deadline = time.monotonic() + timeout
        value = None
        while time.monotonic() < deadline:
            value = read()
            if done(value):
                return value
            time.sleep(.05)
        self.fail("timeout: " + repr(value))

    def command(self, helper, url, session, code):
        request = helper.request("POST", url, "/v1/submit", {"pool": "reclaim", "user_id": "alice",
            "session_id": session, "command": ["/bin/sh", "-ec", code],
            "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}})
        result = self.poll(lambda: helper.request("GET", url, "/v1/requests/" + request["request_id"]),
                           lambda r: r["state"] in ("COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "REJECTED", "LOST"))
        self.assertEqual(result["state"], "COMPLETED", result)
        return result

    def release(self, helper, url, identity):
        helper.request("POST", url, "/v1/workspaces/" + identity + "/release", {})
        return self.poll(lambda: helper.request("GET", url, "/v1/workspaces/" + identity),
                         lambda r: r["state"] == "retained")

    def suspend(self, helper, url, identity, execution):
        helper.request("POST", url, "/v1/executions/" + execution + "/suspend", {})
        return self.poll(lambda: helper.request("GET", url, "/v1/workspaces/" + identity),
                         lambda r: r["state"] == "retained")

    def test_workspace_off_during_command_allows_explicit_then_inherited_reclaim(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                root, helper, url = self.start(binary, 128*1024**2, True)
                request = helper.request("POST", url, "/v1/submit", {"pool": "reclaim", "user_id": "alice",
                    "session_id": "controlled", "command": ["/bin/sh", "-ec",
                        "sleep 3; printf survivor > keep; dd if=/dev/urandom of=discard bs=1M count=64 status=none; rm discard"],
                    "resources": {"gpus": 0, "runtime_adapter": "local-process", "isolation_policy": "none"}})
                running = self.poll(lambda: helper.request("GET", url, "/v1/requests/" + request["request_id"]),
                                    lambda r: r["state"] == "RUNNING")
                identity = running["workspace"]["id"]
                path = "/v1/workspaces/" + identity
                changed = helper.request("POST", url, path + "/set", {"auto_reclaim": "off"})
                self.assertEqual(changed["auto_reclaim"], "off")
                self.assertFalse(changed["auto_reclaim_enabled"])
                result = helper.wait_state(url, request["request_id"], "COMPLETED", timeout=45)
                self.suspend(helper, url, identity, result["execution_session_id"])
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    status = helper.request("GET", url, path)
                    self.assertEqual(status["auto_reclaim"], "off")
                    self.assertNotIn("operation", status)
                    self.assertIsNone(status.get("reclaim_candidate"))
                    time.sleep(.1)
                requested = helper.request("POST", url, path + "/reclaim", {})
                done = self.poll(lambda: helper.request("GET", url, path),
                                 lambda r: r.get("operation", {}).get("state") in ("completed", "failed"))
                self.assertEqual(done["operation"]["state"], "completed", done)
                self.assertEqual(done["operation"]["id"], requested["operation"]["id"])
                self.assertEqual(done["operation"]["source"], "explicit")
                self.assertGreater(done["operation"]["reclaimed_bytes"], 32*1024**2)
                self.assertEqual(done["auto_reclaim"], "off")
                self.assertEqual(done["current_image_size_bytes"], 128*1024**2)
                resumed = self.command(helper, url, "controlled", "test $(cat keep) = survivor; "
                    "dd if=/dev/urandom of=discard bs=1M count=64 status=none; rm discard")
                self.assertEqual(resumed["workspace"]["id"], identity)
                helper.request("POST", url, path + "/set", {"auto_reclaim": "inherit"})
                self.suspend(helper, url, identity, resumed["execution_session_id"])
                done = self.poll(lambda: helper.request("GET", url, path),
                    lambda r: r.get("operation", {}).get("state") == "completed"
                    and r["operation"]["id"] != requested["operation"]["id"])
                self.assertEqual(done["operation"]["source"], "automatic")
                self.assertGreater(done["operation"]["reclaimed_bytes"], 32*1024**2)
                self.command(helper, url, "controlled", "test $(cat keep) = survivor; test ! -e discard")
                self.suspend(helper, url, identity, resumed["execution_session_id"])
                print(json.dumps({"binary": binary, "workspace_off_explicit_inherit": True}), flush=True)
                self.doCleanups()
                self.assertFalse(any(m["target"].startswith(str(root)+"/") for m in mount_entries()))
                shutil.rmtree(root)

    def test_same_execution_reclaims_again_after_later_write_delete(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                root, helper, url = self.start(binary, 128*1024**2, True)
                identity = execution = previous = None
                for cycle in range(2):
                    result = self.command(helper, url, "sequence", "printf survivor > keep; "
                        "dd if=/dev/urandom of=discard bs=1M count=64 status=none; rm discard; test $(cat keep) = survivor")
                    if identity is not None:
                        self.assertEqual(result["workspace"]["id"], identity)
                        self.assertEqual(result["execution_session_id"], execution)
                    identity, execution = result["workspace"]["id"], result["execution_session_id"]
                    released = self.suspend(helper, url, identity, execution)
                    done = self.poll(lambda: helper.request("GET", url, "/v1/workspaces/" + identity),
                        lambda r: r.get("operation", {}).get("state") == "completed" and r["operation"]["id"] != previous)
                    self.assertEqual(done["operation"]["action"], "reclaim")
                    self.assertIsNotNone(maintenance_host_key())
                    self.assertEqual(done["operation"]["grant_host_key"], maintenance_host_key())
                    self.assertGreater(done["operation"]["reclaimed_bytes"], 32*1024**2)
                    self.assertEqual(done["current_image_size_bytes"], 128*1024**2)
                    self.assertEqual(done["reclaim_success_generation"], released["usage_generation"])
                    previous = done["operation"]["id"]
                self.command(helper, url, "sequence", "test $(cat keep) = survivor; test ! -e discard")
                self.suspend(helper, url, identity, execution)
                print(json.dumps({"binary": binary, "same_execution_two_reclaims": True, "root": str(root)}), flush=True)
                self.doCleanups()
                self.assertFalse(any(m["target"].startswith(str(root)+"/") for m in mount_entries()))
                shutil.rmtree(root)

    def test_empty_large_image_burst_auto_off_on_cpu_comparison(self):
        # The actual metadata floor varies with formatter/backing filesystem.
        # Select a threshold below it, so the former absolute rule would admit it.
        # CPU duration is measured, not asserted as a portable hard latency quota.
        benchmark = "import hashlib,time; b=b'x'*(4*1024**2); t=time.perf_counter(); " \
                    "[hashlib.sha256(b).digest() for _ in range(512)]; print(time.perf_counter()-t)"
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            measurements = {}
            allocations = {}
            threshold = 1024**2
            for enabled in (False, True):
                with self.subTest(binary=binary, auto=enabled):
                    try:
                        root, helper, url = self.start(binary, 32*1024**3, enabled, threshold)
                        identities, durations, allocated = [], [], []
                        for index in range(3):
                            cpu = subprocess.Popen([sys.executable, "-c", benchmark], stdout=subprocess.PIPE, text=True)
                            try:
                                result = self.command(helper, url, "empty-" + str(index), "true")
                                identity = result["workspace"]["id"]
                                self.release(helper, url, identity)
                                identities.append(identity)
                                output, _ = cpu.communicate(timeout=60)
                                self.assertEqual(cpu.returncode, 0)
                                durations.append(float(output))
                            finally:
                                if cpu.poll() is None:
                                    cpu.kill(); cpu.wait(timeout=10)
                        time.sleep(2)
                        for identity in identities:
                            record = helper.request("GET", url, "/v1/workspaces/" + identity)
                            self.assertNotIn("operation", record, record)
                            self.assertIsNone(record.get("reclaim_candidate"))
                            self.assertEqual(record["state"], "retained")
                            allocated.append(record["reclaim_sample"]["allocated_bytes"])
                            self.assertGreater(allocated[-1], threshold, "old absolute rule must admit this control")
                            signal, error = reclaim_signal(record)
                            self.assertIsNone(error, record)
                            self.assertLess(signal, threshold)
                        measurements[str(enabled)] = durations
                        allocations[str(enabled)] = allocated
                    finally:
                        self.doCleanups()
                    self.assertFalse(any(m["target"].startswith(str(root)+"/") for m in mount_entries()))
                    shutil.rmtree(root)
            if set(measurements) == {"True", "False"}:
                print(json.dumps({"binary": binary, "empty_32gib_burst": "passed", "cpu_seconds": measurements,
                    "allocated_bytes": allocations, "threshold_bytes": threshold,
                    "median_auto_on_over_off": statistics.median(measurements["True"])/statistics.median(measurements["False"])}), flush=True)
