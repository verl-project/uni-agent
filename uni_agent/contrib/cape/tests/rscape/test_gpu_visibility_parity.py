# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.rscape import test_functional_parity as parity
from tests.managed_fixture import simulated_gpu_environment


class GPUVisibilityParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_visibility_and_terminal_refusal_release_leases(self):
        helper = parity.FunctionalParityTests()
        cases = [
            (2, None, "GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002", None),
            (2, "GPU-00000000-0000-0000-0000-000000000002", "GPU-00000000-0000-0000-0000-000000000002", None),
            (2, "", "", None),
            (2, " GPU-00000000-0000-0000-0000-000000000002 , GPU-00000000-0000-0000-0000-000000000002 ", " GPU-00000000-0000-0000-0000-000000000002 , GPU-00000000-0000-0000-0000-000000000002 ", None),
            (2, "GPU-00000000-0000-0000-0000-000000000001,GPU-other", None, "GPU-other"),
            (2, "0", None, "0"),
            (0, "GPU-other", "", None),
            (2, "GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000001", None),
        ]
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                controller, url, _ = helper.start_controller(binary, tmp)
                with open(root / "node.log", "w") as stream:
                    node = subprocess.Popen([
                        binary, "node", "enroll", *helper.node_credentials(url, "visibility-node", "visibility"), "--controller-url", url,
                        "--node-id", "visibility-node", "--pool", "visibility",
                        "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000001", "--gpu-uuid", "GPU-00000000-0000-0000-0000-000000000002",
                        "--cpu-cores", "4", "--memory-gb", "8",
                        "--runtime-adapter", "local-process", "--heartbeat-interval", "0.05",
                        "--state-path", str(root / "state.json"), "--log-dir", str(root / "logs"),
                    ], stdout=stream, stderr=stream, env=simulated_gpu_environment(root, ["GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"]))
                    try:
                        for index, (count, mask, expected, invalid) in enumerate(cases):
                            submitted = helper.request("POST", url, "/v1/submit", {
                                "pool": "visibility", "user_id": "alice", "session_id": str(index),
                                "reuse": False,
                                "command": ["/bin/sh", "-c", 'printf "%s" "$CUDA_VISIBLE_DEVICES"'],
                                "env": {} if mask is None else {"CUDA_VISIBLE_DEVICES": mask},
                                "resources": {"gpus": count, "cpu_cores": 1, "memory_gb": 1,
                                              "runtime_adapter": "local-process", "isolation_policy": "none"},
                            })
                            request_id = submitted["request_id"]
                            status = helper.wait_state(url, request_id, "FAILED" if invalid else "COMPLETED", timeout=15)
                            logs = helper.request("GET", url, "/v1/requests/" + request_id + "/logs")
                            if invalid:
                                self.assertEqual(status["failure_kind"], "workload")
                                self.assertIn("outside the GPU lease", logs["stderr"])
                                self.assertIn(invalid, logs["stderr"])
                                self.assertEqual(logs["stdout"], "")
                            else:
                                self.assertEqual(logs["stdout"], expected)
                            self.assertEqual(helper.request("GET", url, "/v1/pools/visibility/utilization")["gpus_leased"], 0)
                            self.assertIsNone(node.poll(), (root / "node.log").read_text())
                    finally:
                        node.terminate()
                        try:
                            node.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            node.kill()
                            node.wait(timeout=5)
                        helper.stop_controller(controller)
