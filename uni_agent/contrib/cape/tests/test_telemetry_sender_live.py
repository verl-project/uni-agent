# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Telegraf -> Prometheus test; synthetic metrics, no GPU use."""

import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.parse import urlencode
from urllib.request import urlopen

from pycape.telemetry_sender import BatchSender


@unittest.skipUnless(os.environ.get("CAPE_TELEMETRY_LIVE") == "1", "requires real telemetry binaries")
class TelemetrySenderLiveTests(unittest.TestCase):
    def test_remote_write_and_outage_batch_is_not_replayed(self):
        prometheus = os.environ["CAPE_PROMETHEUS_BIN"]
        telegraf = os.environ["CAPE_TELEGRAF_BIN"]
        output = Path(os.environ["CAPE_TELEMETRY_TEST_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cape-metric-live-") as root:
            root = Path(root)
            config = root / "prometheus.yml"
            config.write_text("global:\n  scrape_interval: 5s\nscrape_configs: []\n", encoding="utf-8")
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            base = "http://127.0.0.1:{}".format(port)
            sender = BatchSender(base + "/api/v1/write", telegraf, timeout_seconds=2, runtime_dir=root)
            process = None
            def start(log):
                proc = subprocess.Popen([prometheus, "--config.file=" + str(config),
                                         "--storage.tsdb.path=" + str(root / "data"),
                                         "--web.listen-address=127.0.0.1:" + str(port),
                                         "--web.enable-remote-write-receiver"], stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        try:
                            with urlopen(base + "/-/ready", timeout=0.5) as response:
                                if response.status == 200:
                                    return proc
                        except OSError:
                            pass
                        if proc.poll() is not None:
                            self.fail("Prometheus exited during startup")
                        time.sleep(0.05)
                    self.fail("Prometheus readiness timeout")
                except BaseException:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=5)
                    raise
            def stop(proc):
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
            def query(name):
                with urlopen(base + "/api/v1/query?" + urlencode({"query": name}), timeout=2) as response:
                    return json.load(response)["data"]["result"]
            def await_value(name):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    result = query(name)
                    if result:
                        return result[0]["value"][1]
                    time.sleep(0.05)
                self.fail("successful write did not become queryable: " + name)
            with (output / "prometheus.log").open("w") as log:
                try:
                    process = start(log)
                    self.assertEqual(sender.send('cape_live_first{gpu_uuid="GPU-test"} 0.25\n')["outcome"], "sent")
                    self.assertEqual(await_value("cape_live_first"), "0.25")
                    stop(process)
                    self.assertEqual(sender.send('cape_live_dropped{gpu_uuid="GPU-test"} 1\n')["outcome"], "dropped")
                    self.assertFalse(list(root.glob("cape-metric-batch-*")))
                    process = start(log)
                    self.assertEqual(sender.send('cape_live_after{gpu_uuid="GPU-test"} 0.5\n')["outcome"], "sent")
                    self.assertEqual(await_value("cape_live_after"), "0.5")
                    self.assertEqual(query("cape_live_dropped"), [])
                    self.assertEqual(query("cape_live_first")[0]["value"][1], "0.25")
                finally:
                    if process is not None:
                        stop(process)
            self.assertFalse(list(root.glob("cape-metric-batch-*")))


if __name__ == "__main__":
    unittest.main()
