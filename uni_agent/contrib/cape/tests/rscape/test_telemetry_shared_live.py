# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import uuid
from unittest.mock import patch

from pycape.identity import new_secret, verifier
from pycape.telemetry_agent import MetricsLink
from pycape.telemetry_supervisor import process_record
from tests.managed_fixture import ManagedFixture
from tests.rscape import test_functional_parity as parity


@unittest.skipUnless(os.environ.get("CAPE_TELEMETRY_LIVE") == "1", "requires monitoring binaries")
class TelemetrySharedLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_simultaneous_scheduler_domains_are_independent(self):
        with tempfile.TemporaryDirectory(prefix="cape-metric-domains-") as directory:
            root = Path(directory)
            fake = root / "dcgmi"
            fake.write_text("#!" + sys.executable + "\nprint('GPU 0 25 10 100 20 40 100')\n")
            fake.chmod(0o755)
            runtime = root / "runtime"
            runtime.mkdir()
            fixtures, processes, links, pids = [], [], [], []
            with patch.dict(os.environ, CAPE_DCGMI_BIN=str(fake), CAPE_METRIC_HELPER_BIN=parity.RSCAPE_BIN,
                            CAPE_METRICS_RUNTIME_DIR=str(runtime), CAPE_METRIC_ADVERTISE_HOST="127.0.0.1"):
                try:
                    for index, binary in enumerate((parity.PYCAPE_BIN, parity.RSCAPE_BIN)):
                        fixture = ManagedFixture(binary, root / str(index))
                        fixtures.append(fixture)
                        processes.append(fixture.start(binary, "--metrics"))
                        pool = fixture.pool("shared", "shared", 2)
                        token = new_secret()
                        gpu = "GPU-00000000-0000-0000-0000-000000000001"
                        fixture.request("POST", "/v1/identity/nodes/enroll", {
                            "operation_id": str(uuid.uuid4()), "verifier": verifier(token),
                            "node_id": "same-node", "pool": pool, "agent_epoch": "same-epoch",
                            "allocation_id": "same-allocation", "resources": {"cpu_cores": 1,
                            "memory_gb": 1, "gpu_uuids": [gpu]}}, fixture.user_token("owner"))
                        links.append(MetricsLink(fixture.url, "same-node", "same-epoch", pool, [gpu], token))
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if all(link.status().get("url") for link in links):
                            if all(f.request("GET", "/v1/metrics/query?metric=memory_total_bytes")["series"] for f in fixtures):
                                break
                        time.sleep(.1)
                    else:
                        self.fail("independent collectors did not converge")
                    locations = [link.status() for link in links]
                    self.assertNotEqual(locations[0]["service_id"], locations[1]["service_id"])
                    self.assertNotEqual(locations[0]["url"], locations[1]["url"])
                    for location in locations:
                        pids.append(json.load(urlopen(location["url"] + "/snapshot"))["pid"])
                    fixtures[0].request("POST", "/v1/identity/admin", {"action": "metric-set", "changes": {"enabled": False}})
                    self.assertEqual(fixtures[0].request("GET", "/v1/metrics/query?metric=memory_total_bytes")["state"], "disabled")
                    self.assertTrue(fixtures[1].request("GET", "/v1/metrics/query?metric=memory_total_bytes")["series"])
                finally:
                    for link in links:
                        link.close()
                    for record in runtime.glob("cape-metrics-*/*.json"):
                        pids.append(json.loads(record.read_text())["pid"])
                    for pid in set(pids):
                        record = process_record(pid)
                        if record:
                            self.assertIn("__metric-service", record["argv"])
                            os.kill(pid, signal.SIGTERM)
                    for process in processes:
                        process.terminate()
                        process.communicate(timeout=10)

    def test_shared_native_service_registration_control_and_history_gaps(self):
        for server in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(server=server), tempfile.TemporaryDirectory(prefix="cape-metric-shared-") as directory:
                root = Path(directory)
                cross_uid = os.environ.get("CAPE_METRIC_CROSS_UID")
                if cross_uid and os.geteuid() != 0:
                    self.skipTest("cross-UID test requires an explicitly authorized root runner")
                if cross_uid:
                    root.chmod(0o755)
                fake = root / "dcgmi"
                fake.write_text("#!" + os.path.realpath(sys.executable) + "\nimport sys\n"
                                "fields=sys.argv[sys.argv.index('-e')+1].split(',')\n"
                                "values={'203':'25','204':'10','250':'100','252':'20','150':'40','155':'100'}\n"
                                "print('GPU 0 '+' '.join(values.get(f,'0.5') for f in fields))\n")
                fake.chmod(0o755)
                runtime = root / "runtime"
                runtime.mkdir()
                if cross_uid:
                    runtime.chmod(0o755)
                env = dict(CAPE_DCGMI_BIN=str(fake), CAPE_METRIC_HELPER_BIN=parity.RSCAPE_BIN,
                           CAPE_METRICS_RUNTIME_DIR=str(runtime), CAPE_METRIC_ADVERTISE_HOST="127.0.0.1")
                if cross_uid:
                    for name, source in (("rscape", parity.RSCAPE_BIN), ("telegraf", os.environ["CAPE_TELEGRAF_BIN"])):
                        shutil.copyfile(source, root / name)
                        (root / name).chmod(0o755)
                    env.update(CAPE_METRIC_HELPER_BIN=str(root / "rscape"), CAPE_TELEGRAF_BIN=str(root / "telegraf"))
                with patch.dict(os.environ, env):
                    fixture = ManagedFixture(server, root / "instance")
                    process = fixture.start(server, "--metrics")
                    links = []
                    service_pid = None
                    try:
                        pool = fixture.pool("shared", "shared", 2)
                        owner = fixture.user_token("owner")
                        gpu = "GPU-00000000-0000-0000-0000-000000000001"
                        fixture.request("POST", "/v1/identity/admin", {"action": "metric-set", "changes": {"interval_seconds": 1, "profiling": True}})
                        for node in ("a", "b"):
                            secret = new_secret()
                            fixture.request("POST", "/v1/identity/nodes/enroll", {"operation_id": str(uuid.uuid4()),
                                "verifier": verifier(secret), "node_id": node, "pool": pool, "agent_epoch": "e-" + node,
                                "allocation_id": "alloc-" + node, "resources": {"cpu_cores": 1, "memory_gb": 1, "gpu_uuids": [gpu]}}, owner)
                            if cross_uid and node == "b":
                                original_popen = subprocess.Popen
                                def launch(*args, **kwargs):
                                    def drop():
                                        os.setgroups([])
                                        os.setgid(int(cross_uid))
                                        os.setuid(int(cross_uid))
                                    return original_popen(*args, preexec_fn=drop, **kwargs)
                                with patch("subprocess.Popen", side_effect=launch):
                                    links.append(MetricsLink(fixture.url, node, "e-" + node, pool, [gpu], secret))
                                self.assertEqual(Path("/proc", str(links[-1]._process.pid)).stat().st_uid, int(cross_uid))
                            else:
                                links.append(MetricsLink(fixture.url, node, "e-" + node, pool, [gpu], secret))
                            if cross_uid and node == "a":
                                deadline = time.monotonic() + 20
                                while not links[0].status().get("url") and time.monotonic() < deadline:
                                    time.sleep(.1)
                                self.assertTrue(links[0].status().get("url"))
                        def wait(predicate, seconds=20):
                            deadline = time.monotonic() + seconds
                            while time.monotonic() < deadline:
                                result = predicate()
                                if result:
                                    return result
                                time.sleep(.1)
                            self.fail("metrics convergence timeout: " + repr([link.status() for link in links]))
                        wait(lambda: all(link.status().get("url") for link in links))
                        first, second = [link.status() for link in links]
                        self.assertEqual((first["service_id"], first["url"]), (second["service_id"], second["url"]))
                        def snapshot():
                            with urlopen(first["url"] + "/snapshot", timeout=2) as response:
                                return json.load(response)
                        service_pid = snapshot()["pid"]
                        sample = wait(lambda: (s if s["samples"] and s["samples"][0]["state"] == "ready" else None) if (s := snapshot()) else None)
                        self.assertEqual(len(sample["samples"]), 1)
                        self.assertEqual(sample["samples"][0]["values"]["memory_total_bytes"], 100 * 1024 ** 2)
                        wait(lambda: fixture.request("GET", "/v1/metrics/query?metric=sm_active_ratio")["series"])
                        public = json.load(urlopen(fixture.url + "/v1/metrics/service-config?" + urlencode({"service_id": first["service_id"]})))
                        self.assertNotIn("token", json.dumps(public))
                        with self.assertRaises(urllib.error.HTTPError):
                            urlopen(Request(first["url"] + "/register", data=json.dumps({"node_id": "a", "agent_epoch": "e-a", "config": {"enabled": False}}).encode(), headers={"Content-Type": "application/json"}))
                        if os.environ.get("CAPE_METRIC_RECOVERY") == "1":
                            old_incarnation = first["incarnation"]
                            os.kill(service_pid, signal.SIGKILL)
                            wait(lambda: all(link.status().get("url") and
                                             link.status().get("incarnation") != old_incarnation
                                             for link in links), seconds=50)
                            first, second = [link.status() for link in links]
                            self.assertEqual(first["url"], second["url"])
                            service_pid = snapshot()["pid"]
                            wait(lambda: snapshot()["samples"])
                            wait(lambda: fixture.request("GET", "/v1/metrics/query?metric=sm_active_ratio")["series"])
                        links[0].close()
                        self.assertEqual(len(snapshot()["samples"]), 1)
                        links[1].close()
                        wait(lambda: not snapshot()["samples"])
                        time.sleep(4)
                        gap_start = time.time()
                        fixture.request("POST", "/v1/identity/admin", {"action": "metric-set", "changes": {"interval_seconds": 60}})
                        query = urlencode({"metric": "sm_active_ratio", "mode": "history", "start": gap_start, "end": gap_start, "step": 1})
                        self.assertEqual(fixture.request("GET", "/v1/metrics/query?" + query)["series"], [])
                    finally:
                        for link in links:
                            link.close()
                        if service_pid is None:
                            for record in runtime.glob("cape-metrics-*/*.json"):
                                try:
                                    service_pid = json.loads(record.read_text())["pid"]
                                except (OSError, ValueError, KeyError):
                                    continue
                        if service_pid and process_record(service_pid):
                            record = process_record(service_pid)
                            self.assertIn("__metric-service", record["argv"])
                            os.kill(service_pid, signal.SIGTERM)
                            deadline = time.monotonic() + 5
                            while process_record(service_pid) and time.monotonic() < deadline:
                                time.sleep(.05)
                            self.assertIsNone(process_record(service_pid))
                        process.terminate()
                        process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
