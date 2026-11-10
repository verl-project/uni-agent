# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import itertools
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tests.rscape import test_functional_parity as parity


@unittest.skipUnless(os.environ.get("CAPE_SANDBOX_ENVIRONMENT_TEST") == "1",
                     "set CAPE_SANDBOX_ENVIRONMENT_TEST=1 on a runtime-capable node")
class SandboxEnvironmentParityTests(unittest.TestCase):
    def test_virtual_roots_and_symlink_escapes_are_rejected_by_both_clis(self):
        helper = parity.FunctionalParityTests()
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with tempfile.TemporaryDirectory(prefix="cape-path-validation-") as tmp:
                controller, url, _ = helper.start_controller(binary, tmp)
                try:
                    for index, path in enumerate(("/sys", "/proc", "/dev", "/run", "/proc/self/root/etc")):
                        result = subprocess.run([
                            binary, "node", "enroll", *helper.node_credentials(url, "invalid-" + str(index), "paths"), "--controller-url", url,
                            "--pool", "paths", "--node-id", "invalid-" + str(index),
                            "--runtime-adapter", "bubblewrap", "--once",
                            "--bubblewrap-read-only-path", path,
                            "--log-dir", str(Path(tmp) / str(index)),
                        ], capture_output=True, text=True, timeout=30)
                        self.assertNotEqual(result.returncode, 0, (binary, path))
                        self.assertIn("virtual", result.stderr.lower(), (binary, path, result.stderr))
                finally:
                    helper.stop_controller(controller)

    def run_case(self, binary, adapter, policies, gpu_uuid=None, gpu_uuids=(), visibility_cases=None):
        helper = parity.FunctionalParityTests()
        selected_gpus = tuple(gpu_uuids) or ((gpu_uuid,) if gpu_uuid else ())
        cases = visibility_cases if visibility_cases is not None else [(gpu_uuid or "caller-value", gpu_uuid or "", None)]
        with tempfile.TemporaryDirectory(prefix="cape-environment-") as tmp:
            root = Path(tmp)
            controller, url, _ = helper.start_controller(binary, tmp)
            image = os.environ.get("CAPE_ENVIRONMENT_IMAGE", "docker://ubuntu:24.04")
            if adapter == "apptainer":
                image = os.environ.get("CAPE_ENVIRONMENT_APPTAINER_IMAGE", image)
            args = [
                binary, "node", "enroll", *helper.node_credentials(url, "environment-node", "environment"), "--controller-url", url,
                "--pool", "environment", "--node-id", "environment-node",
                "--cpu-cores", "4", "--memory-gb", "8",
                "--runtime-adapter", adapter, "--heartbeat-interval", "0.05",
                "--state-path", str(root / "state.json"),
                "--log-dir", str(root / "logs"), "--runtime-dir", str(root / "runtime"),
            ]
            if adapter == "bubblewrap":
                args += ["--bubblewrap-binary", os.environ.get("CAPE_ENVIRONMENT_BWRAP", "/usr/bin/bwrap"),
                         "--bubblewrap-read-only-path", "/sys/devices/system/cpu",
                         "--bubblewrap-read-only-path", "/sys/devices/system/cpu",
                         "--bubblewrap-read-only-path", "/sys/cape-nonexistent-path"]
            elif adapter == "apptainer":
                args += ["--apptainer-binary", os.environ.get("CAPE_ENVIRONMENT_APPTAINER", "apptainer"),
                         "--apptainer-probe-image", image,
                         "--apptainer-cache", str(root / "cache"), "--apptainer-tmp", str(root / "tmp")]
            elif adapter == "podman":
                args += ["--podman-binary", os.environ.get("CAPE_ENVIRONMENT_PODMAN", "podman"),
                         "--podman-probe-image", image, "--podman-graphroot", str(root / "graph"),
                         "--podman-runroot", str(root / "run"), "--podman-tmpdir", str(root / "tmp")]
            for selected_gpu in selected_gpus:
                args += ["--gpu-uuid", selected_gpu]
            with open(root / "node.log", "w") as stream:
                node = subprocess.Popen(args, stdout=stream, stderr=stream)
                try:
                    deadline = time.monotonic() + 300
                    while time.monotonic() < deadline:
                        self.assertIsNone(node.poll(), (root / "node.log").read_text()[-6000:])
                        capacity = helper.request("GET", url, "/v1/pools/environment/utilization")
                        if capacity.get("active_nodes") == 1:
                            break
                        time.sleep(0.1)
                    else:
                        self.fail("runtime node never became active")
                    for policy, (mask, expected, invalid) in itertools.product(policies, cases):
                        shell = (
                            'printf "visibility=%s\\npath=%s\\n" "$CUDA_VISIBLE_DEVICES" "$PATH"; '
                            'if test -r /sys/devices/system/cpu/online; then echo topology=readable; else echo topology=absent; fi; '
                            'if test -e /sys/class; then echo sysroot=exposed; else echo sysroot=absent; fi'
                        )
                        if gpu_uuid and not invalid:
                            expected_count = len({entry.strip() for entry in expected.split(",") if entry.strip()})
                            cuda_probe = "import torch; assert torch.cuda.device_count()=={}; ".format(expected_count)
                            if expected_count:
                                cuda_probe += "x=torch.arange(32,device='cuda',dtype=torch.bfloat16); assert x.float().sum().item()==496; "
                            shell += '; /usr/bin/python3 -c "' + cuda_probe + "print('cuda=passed')\""
                        request = helper.request("POST", url, "/v1/submit", {
                            "pool": "environment", "user_id": "alice", "session_id": policy,
                            "reuse": True,                             "image": image if adapter in ("apptainer", "podman") else None,
                            "env": dict({"PATH": "/chosen/toolchain:/bin"}, **({} if mask is None else {"CUDA_VISIBLE_DEVICES": mask})),
                            "command": ["/bin/sh", "-c", shell],
                            "resources": {"gpus": len(selected_gpus), "runtime_adapter": adapter,
                                          "isolation_policy": policy, "max_duration_seconds": 120},
                        })
                        try:
                            status = helper.wait_state(url, request["request_id"], "FAILED" if invalid else "COMPLETED", timeout=60)
                        except AssertionError:
                            print(json.dumps({"failed_request": request["request_id"],
                                              "logs": helper.request("GET", url, "/v1/requests/" + request["request_id"] + "/logs"),
                                              "node_log": (root / "node.log").read_text()[-12000:]}), flush=True)
                            raise
                        logs = helper.request("GET", url, "/v1/requests/" + request["request_id"] + "/logs")
                        if invalid:
                            self.assertEqual(status["failure_kind"], "workload")
                            self.assertIn(invalid, logs["stderr"])
                            self.assertIn("outside the GPU lease", logs["stderr"])
                            self.assertEqual(helper.request("GET", url, "/v1/pools/environment/utilization")["gpus_leased"], 0)
                            print(json.dumps({"binary": binary, "adapter": adapter, "policy": policy,
                                              "mask": mask, "state": status["state"], "failure_kind": status["failure_kind"]}), flush=True)
                            continue
                        fields = dict(line.split("=", 1) for line in logs["stdout"].splitlines())
                        self.assertEqual(fields["visibility"], expected)
                        self.assertEqual(fields["path"].split(":" )[0], "/chosen/toolchain")
                        if adapter == "bubblewrap":
                            self.assertEqual(fields["topology"], "readable")
                            self.assertEqual(fields["sysroot"], "absent")
                            if gpu_uuid and expected:
                                self.assertIn("/usr/bin", fields["path"].split(":"))
                                self.assertEqual(fields["cuda"], "passed")
                        print(json.dumps({"binary": binary, "adapter": adapter, "policy": policy,
                                          "gpu": gpu_uuid, "observed": fields, "state": status["state"]}), flush=True)
                finally:
                    node.terminate()
                    try:
                        node.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        node.kill()
                        node.wait(timeout=5)
                    helper.stop_controller(controller)
                    if adapter == "podman":
                        # This graph/runroot belongs only to this test case.
                        # Tear down warm containers before TemporaryDirectory
                        # encounters their mounted shared-memory directories.
                        cleanup = subprocess.run([
                            os.environ.get("CAPE_ENVIRONMENT_PODMAN", "podman"),
                            "--root", str(root / "graph"), "--runroot", str(root / "run"),
                            "--tmpdir", str(root / "tmp"), "rm", "--force", "--all",
                        ], capture_output=True, text=True, timeout=60)
                        self.assertEqual(cleanup.returncode, 0, cleanup.stderr)

    def test_cpu_visibility_and_virtual_tree_audit(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            for adapter, policies in (
                ("local-process", ("none",)),
                ("bubblewrap", ("host-stateless", "host-stateful")),
                ("apptainer", ("image-stateless", "image-stateful")),
                ("podman", ("image-stateless", "image-stateful")),
            ):
                with self.subTest(binary=binary, adapter=adapter):
                    self.run_case(binary, adapter, policies)

    def test_gpu_driver_paths_preserve_toolchain(self):
        gpu_uuid = os.environ["CAPE_ENVIRONMENT_GPU_UUID"]
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary):
                self.run_case(binary, "bubblewrap", ("host-stateless", "host-stateful"), gpu_uuid)

    def test_gpu_lease_narrowing_all_adapters(self):
        gpu_uuids = tuple(os.environ["CAPE_ENVIRONMENT_GPU_UUIDS"].split(","))
        self.assertEqual(len(gpu_uuids), 2, "provide exactly the two authorized GPU UUIDs")
        cases = [(None, ",".join(gpu_uuids), None), (gpu_uuids[1], gpu_uuids[1], None),
                 ("", "", None), ("GPU-unleased", None, "GPU-unleased"), ("0", None, "0"),
                 (gpu_uuids[0], gpu_uuids[0], None)]
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            for adapter, policies in (
                ("local-process", ("none",)),
                ("bubblewrap", ("host-stateless", "host-stateful")),
                ("apptainer", ("image-stateless", "image-stateful")),
                ("podman", ("image-stateless", "image-stateful")),
            ):
                with self.subTest(binary=binary, adapter=adapter):
                    self.run_case(binary, adapter, policies, gpu_uuids=gpu_uuids,
                                  gpu_uuid=gpu_uuids[0] if adapter in ("bubblewrap", "local-process") else None,
                                  visibility_cases=cases)


if __name__ == "__main__":
    unittest.main()
