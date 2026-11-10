# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Prometheus init/start handoff across implementations; no GPU workloads."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

from pycape.identity import new_secret, verifier
from pycape.telemetry_sender import BatchSender
from pycape.telemetry_supervisor import PrometheusSupervisor, process_record
from tests.managed_fixture import ManagedFixture
from tests.rscape import test_functional_parity as parity


@unittest.skipUnless(os.environ.get("CAPE_TELEMETRY_LIVE") == "1", "requires real Prometheus/Telegraf")
class TelemetrySupervisorLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_init_start_reuse_query_and_normal_shutdown_cross_backend(self):
        binaries = (parity.PYCAPE_BIN, parity.RSCAPE_BIN)
        for initializer, starter in ((binaries[0], binaries[1]), (binaries[1], binaries[0])):
            with self.subTest(initializer=initializer), tempfile.TemporaryDirectory(prefix="cape-metric-supervisor-") as directory:
                root = Path(directory)
                (root / "secrets").mkdir(mode=0o700)
                state = root / "controller-state"
                admin_file = root / "secrets/admin.json"
                initialized = subprocess.run([initializer, "controller", "init", "--state-dir", str(state),
                    "--user", "admin", "--token-out", str(admin_file), "--metrics"], text=True, capture_output=True, timeout=20)
                self.assertEqual(initialized.returncode, 0, initialized.stderr)
                info = json.loads(initialized.stdout)
                supervisor = PrometheusSupervisor(state, info["scheduler_id"], os.environ["CAPE_PROMETHEUS_BIN"])
                process = None
                try:
                    self.assertEqual(info["metrics"]["state"], "ready", info)
                    original = json.loads((root / "prometheus/process.json").read_text())
                    (root / "prometheus/process.json").unlink()  # Recover the spawn/publication crash window.
                    fixture = ManagedFixture(initializer, root)
                    process = fixture.start(starter)
                    current = json.loads((root / "prometheus/process.json").read_text())
                    self.assertEqual((current["pid"], current["start_ticks"]), (original["pid"], original["start_ticks"]))
                    owner = fixture.user_token("contributor")
                    pool = fixture.pool("pool")
                    secret = new_secret()
                    gpu = "GPU-00000000-0000-0000-0000-000000000001"
                    fixture.request("POST", "/v1/identity/nodes/enroll", {"operation_id": str(uuid.uuid4()),
                        "verifier": verifier(secret), "node_id": "n", "pool": pool, "agent_epoch": "e", "allocation_id": "a",
                        "resources": {"cpu_cores": 1, "memory_gb": 1, "gpu_uuids": [gpu]}}, owner)
                    control = fixture.request("POST", "/v1/agent/metrics/control", {"node_id": "n", "agent_epoch": "e", "service_id": "s"}, secret)
                    self.assertTrue(control["receiver_url"].endswith("/api/v1/write"))
                    sender = BatchSender(current["query_url"] + "/api/v1/write", os.environ["CAPE_TELEGRAF_BIN"])
                    exposition = 'cape_gpu_sm_active_ratio{scheduler_id="%s",service_id="s",gpu_uuid="%s"} 0.25\n' % (info["scheduler_id"], gpu)
                    exposition += 'cape_gpu_sample_expires_at_seconds{scheduler_id="%s",service_id="s",gpu_uuid="%s"} %s\n' % (info["scheduler_id"], gpu, time.time() + 60)
                    self.assertEqual(sender.send(exposition)["outcome"], "sent")
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        result = fixture.request("GET", "/v1/metrics/query?metric=sm_active_ratio&node_id=n", token=owner)
                        if result["state"] == "ready":
                            break
                        time.sleep(0.05)
                    self.assertEqual(result["series"][0]["samples"][0][1], 0.25, result)
                    process.kill()  # Abrupt controller death must not duplicate the surviving TSDB writer.
                    process.communicate(timeout=10)
                    self.assertIsNotNone(process_record(original["pid"]))
                    process = fixture.start(starter)
                    recovered = json.loads((root / "prometheus/process.json").read_text())
                    self.assertEqual((recovered["pid"], recovered["start_ticks"]), (original["pid"], original["start_ticks"]))
                    process.terminate()  # SIGTERM must also run the Python cleanup path.
                    process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0)
                    self.assertIsNone(process_record(original["pid"]))
                    self.assertTrue((root / "prometheus/data").is_dir())
                    process = fixture.start(starter)
                    retained = fixture.request("GET", "/v1/metrics/query?metric=sm_active_ratio&node_id=n", token=owner)
                    self.assertEqual(retained["series"][0]["samples"][0][1], 0.25, retained)
                    restarted = json.loads((root / "prometheus/process.json").read_text())
                    process.terminate()
                    process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0)
                    self.assertIsNone(process_record(restarted["pid"]))
                finally:
                    if process is not None:
                        if process.poll() is None:
                            process.terminate()
                        process.communicate(timeout=10)
                    supervisor.stop()

    def test_missing_binary_does_not_destroy_initialized_instance(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, CAPE_PROMETHEUS_BIN="/missing/prometheus"):
                result = subprocess.run([binary, "controller", "init", "--state-dir", directory + "/state",
                    "--user", "admin", "--token-out", directory + "/admin.json", "--metrics"], text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["metrics"]["state"], "unavailable")
                self.assertTrue(Path(directory, "state/state.sqlite3").is_file())


if __name__ == "__main__":
    unittest.main()
