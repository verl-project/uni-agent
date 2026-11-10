# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UUID-scoped DCGM reads through actual node CLIs; CPU commands only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from urllib.parse import urlencode
from urllib.request import urlopen
from unittest.mock import patch

from pycape.telemetry_supervisor import process_record
from tests.managed_fixture import ManagedFixture, command
from tests.rscape import test_functional_parity as parity


@unittest.skipUnless(os.environ.get("CAPE_DCGM_LIVE") == "1", "requires approved GPU UUIDs and working FUSE/DCGM")
class TelemetryDcgmLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_public_agents_share_real_dcgm(self):
        gpus = os.environ["CAPE_TELEMETRY_GPU_UUIDS"].split(",")
        self.assertTrue(gpus and all(gpu.startswith(("GPU-", "MIG-")) for gpu in gpus))
        output = Path(os.environ["CAPE_TELEMETRY_TEST_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory(prefix="cape-dcgm-live-") as directory:
                root = Path(directory)
                runtime = root / "runtime"
                runtime.mkdir()
                with patch.dict(os.environ, CAPE_METRICS_RUNTIME_DIR=str(runtime),
                                CAPE_METRIC_ADVERTISE_HOST="127.0.0.1", CAPE_METRIC_HELPER_BIN=parity.RSCAPE_BIN):
                    fixture = ManagedFixture(server, root / "instance")
                    controller = fixture.start(server, "--metrics")
                    nodes, streams = [], []
                    owner = None
                    try:
                        fixture.pool("shared", "shared", 4)
                        owner_file = fixture.user_file("contributor")
                        owner = fixture.user_token("contributor")
                        fixture.request("POST", "/v1/admin/workspace-config",
                                        {"pool": "shared", "changes": {"image_root": str(root / "images"), "image_size_bytes": 128 * 1024 ** 2}})
                        fixture.request("POST", "/v1/identity/admin",
                                        {"action": "metric-set", "changes": {"interval_seconds": 2, "profiling": True}})
                        for name, binary in (("python-node", parity.PYCAPE_BIN), ("rust-node", parity.RSCAPE_BIN)):
                            stream = (output / (Path(server).name + "-" + name + ".log")).open("w")
                            streams.append(stream)
                            args = [binary, "node", "enroll", "--metrics", "--controller-url", fixture.url,
                                    "--pool", "shared", "--node-id", name, "--token-file", str(owner_file),
                                    "--node-token-out", str(root / (name + ".json")), "--cpu-cores", "2", "--memory-gb", "4",
                                    "--runtime-adapter", "local-process", "--heartbeat-interval", "0.2",
                                    "--state-path", str(root / (name + "-state.json")),
                                    "--log-dir", str(root / (name + "-logs")), "--runtime-dir", str(root / (name + "-runtime"))]
                            for gpu in gpus:
                                args += ["--gpu-uuid", gpu]
                            nodes.append((name, subprocess.Popen(args, stdout=stream, stderr=stream,
                                env=dict(os.environ, CAPE_NODE_TOKEN="test-only-secret-must-not-reach-shared-service"))))

                        def wait(predicate, seconds=90):
                            deadline = time.monotonic() + seconds
                            while time.monotonic() < deadline:
                                for name, proc in nodes:
                                    self.assertIsNone(proc.poll(), "agent exited: " + name)
                                value = predicate()
                                if value:
                                    return value
                                time.sleep(.2)
                            self.fail("DCGM convergence timeout; evidence=" + str(output))

                        wait(lambda: len([n for n in fixture.request("GET", "/v1/identity/nodes") if n["state"] == "ACTIVE"]) == 2)
                        status = wait(lambda: (s if len(s["receipts"]) == 2 else None)
                                      if (s := fixture.request("GET", "/v1/metrics/status")) else None)
                        services = {r["service_id"] for r in status["receipts"].values()}
                        self.assertEqual(len(services), 1)
                        public = json.load(urlopen(fixture.url + "/v1/metrics/service-config?" + urlencode({"service_id": next(iter(services))})))
                        address = public["location"]["url"]

                        def local():
                            snapshot = json.load(urlopen(address + "/snapshot", timeout=3))
                            (output / (Path(server).name + "-latest.json")).write_text(json.dumps(snapshot, indent=2))
                            return snapshot

                        sample = wait(lambda: (s if len(s["samples"]) == len(gpus) and
                                               all(v["values"].get("memory_total_bytes", 0) for v in s["samples"]) else None)
                                      if (s := local()) else None)
                        self.assertEqual({s["gpu_uuid"] for s in sample["samples"]}, set(gpus))
                        service_environment = Path("/proc", str(sample["pid"]), "environ").read_bytes().split(b"\0")
                        self.assertFalse(any(item.startswith(b"CAPE_NODE_TOKEN=") for item in service_environment))
                        self.assertTrue(all("sm_active_ratio" in s["values"] for s in sample["samples"]))
                        (output / (Path(server).name + "-samples.json")).write_text(json.dumps(sample, indent=2))
                        wait(lambda: fixture.request("GET", "/v1/metrics/query?metric=memory_total_bytes")["series"])
                        result = command(server, "run", "--controller-url", fixture.url, "--token-file", owner_file,
                                         "--pool", "shared", "--session-id", "metrics-cpu", "--gpus", "0",
                                         "--runtime-adapter", "local-process", "--isolation-policy", "none",
                                         "--wait", "--timeout", "30", "--", "/bin/echo", "cpu-only")
                        self.assertEqual(result["state"], "COMPLETED")
                        manifest = json.loads((fixture.root / "prometheus/process.json").read_text())
                        record = process_record(manifest["pid"])
                        self.assertIn("--storage.tsdb.path=" + str(fixture.root / "prometheus/data"), record["argv"])
                        os.kill(manifest["pid"], signal.SIGTERM)
                        wait(lambda: local()["last_send"] == "dropped", seconds=15)
                        outage_result = command(server, "run", "--controller-url", fixture.url,
                            "--token-file", owner_file, "--pool", "shared", "--session-id", "metrics-cpu",
                            "--gpus", "0", "--runtime-adapter", "local-process", "--isolation-policy", "none",
                            "--wait", "--timeout", "30", "--", "/bin/echo", "receiver-down-cpu-only")
                        self.assertEqual(outage_result["state"], "COMPLETED")
                        time.sleep(7)
                        gap = time.time()
                        controller.terminate()
                        controller.communicate(timeout=10)
                        controller = fixture.start(server)
                        wait(lambda: fixture.request("GET", "/v1/metrics/query?metric=memory_total_bytes")["series"])
                        query = urlencode({"metric": "memory_total_bytes", "mode": "history", "start": gap, "end": gap, "step": 1})
                        self.assertEqual(fixture.request("GET", "/v1/metrics/query?" + query)["series"], [])
                    finally:
                        for name, proc in nodes:
                            if proc.poll() is None:
                                try:
                                    fixture.request("POST", "/v1/identity/nodes/" + name + "/action", {"action": "drain"}, owner)
                                    proc.wait(timeout=30)
                                except Exception:
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=15)
                                    except subprocess.TimeoutExpired:
                                        proc.kill()
                                        proc.wait(timeout=5)
                        for record_file in runtime.glob("cape-metrics-*/*.json"):
                            record = json.loads(record_file.read_text())
                            proc = process_record(record["pid"])
                            if proc:
                                self.assertIn("__metric-service", proc["argv"])
                                os.kill(record["pid"], signal.SIGTERM)
                                deadline = time.monotonic() + 5
                                while process_record(record["pid"]) and time.monotonic() < deadline:
                                    time.sleep(.05)
                                self.assertIsNone(process_record(record["pid"]))
                        if controller.poll() is None:
                            controller.terminate()
                        controller.communicate(timeout=10)
                        for stream in streams:
                            stream.close()


if __name__ == "__main__":
    unittest.main()
