# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
from unittest import mock

from pycape.runtime import GPUVisibilityError, leased_gpu_visibility
from pycape import ApptainerRuntimeAdapter, PodmanRuntimeAdapter, BubbleWrapRuntimeAdapter, ContainerSessionSpec, ContainerCommandSpec
from pycape import NodeAgentClient, NodeResources, RuntimeResult
from tests.workspace_fixtures import install_synthetic_assignment_workspaces


class GPUVisibilityTests(unittest.TestCase):
    def setUp(self):
        install_synthetic_assignment_workspaces(self)

    def test_dispatch_command_mask_overrides_session_mask_before_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = NodeAgentClient("http://unused", "node", "pool", NodeResources(1, 1), log_dir=directory)
            runtime = mock.Mock(spec=["start_session", "run_command"])
            runtime.start_session.return_value = RuntimeResult(0)
            runtime.run_command.return_value = RuntimeResult(0)
            agent.runtime_adapters["local-process"] = runtime
            assignment = {
                "command_id": "override", "execution_session_id": "session", "request_id": "request",
                "session_spec": {"execution_session_id": "session", "runtime_adapter": "local-process",
                                 "gpu_uuids": ["GPU-a", "GPU-b"], "env": {"CUDA_VISIBLE_DEVICES": "GPU-unleased"}},
                "command_spec": {"command": ["true"], "env": {"CUDA_VISIBLE_DEVICES": "GPU-b"}},
            }
            with mock.patch.object(agent, "_report") as report:
                agent.execute_assignment(assignment)
            self.assertEqual(report.call_args.args[1], 0, report.call_args)
            self.assertEqual(runtime.start_session.call_args.args[0].env["CUDA_VISIBLE_DEVICES"], "GPU-b")

    def test_adapter_masks_and_command_precedence(self):
        for adapter, policy in (
            (ApptainerRuntimeAdapter(), "image-stateless"),
            (ApptainerRuntimeAdapter(), "image-stateful"),
            (PodmanRuntimeAdapter(), "image-stateless"),
            (BubbleWrapRuntimeAdapter(), "host-stateless"),
            (BubbleWrapRuntimeAdapter(), "host-stateful"),
        ):
            session = ContainerSessionSpec("subset", image="test-image", isolation_policy=policy,
                                           gpu_uuids=("GPU-a", "GPU-b"), env={"CUDA_VISIBLE_DEVICES": "GPU-a"},
                                           overlay_path="/tmp/cape-test-overlay")
            for mask in (None, "GPU-b", ""):
                command = ContainerCommandSpec(("true",), env={} if mask is None else {"CUDA_VISIBLE_DEVICES": mask})
                expected = "GPU-a" if mask is None else mask
                with self.subTest(adapter=adapter.name, policy=policy, mask=mask):
                    self.assertEqual(adapter._merged_env(session, command)["CUDA_VISIBLE_DEVICES"], expected)
                    if isinstance(adapter, BubbleWrapRuntimeAdapter):
                        with mock.patch.object(adapter, "_gpu_search_path_environment", return_value={}):
                            self.assertEqual(adapter._sandbox_environment(session, command, None)["CUDA_VISIBLE_DEVICES"], expected)
                    else:
                        launch = adapter.build_run_command(session, command)
                        self.assertIn("CUDA_VISIBLE_DEVICES=" + expected, launch.argv)
            with self.assertRaises(GPUVisibilityError):
                adapter._merged_env(session, ContainerCommandSpec(("true",), env={"CUDA_VISIBLE_DEVICES": "0"}))

        with self.assertRaisesRegex(ValueError, "not supported"):
            PodmanRuntimeAdapter().build_run_command(
                ContainerSessionSpec("unsupported", isolation_policy="image-stateful", gpu_uuids=("GPU-a",)),
                ContainerCommandSpec(("true",)))

    def test_defaults_and_cpu_clearing(self):
        self.assertEqual(leased_gpu_visibility(["GPU-a", "GPU-b"], {}), "GPU-a,GPU-b")
        for value in ("GPU-other", "0", "", "GPU-a"):
            self.assertEqual(leased_gpu_visibility([], {"CUDA_VISIBLE_DEVICES": value}), "")

    def test_valid_subsets_preserve_caller_text(self):
        for value in ("", "GPU-b", "GPU-b,GPU-a", " GPU-b , GPU-b "):
            self.assertEqual(
                leased_gpu_visibility(["GPU-a", "GPU-b"], {"CUDA_VISIBLE_DEVICES": value}), value
            )

    def test_invalid_entries_are_named(self):
        for value, invalid in (("GPU-other", "GPU-other"), ("0", "0"), ("GPU-a,1", "1")):
            with self.assertRaisesRegex(GPUVisibilityError, "outside the GPU lease: " + invalid):
                leased_gpu_visibility(["GPU-a"], {"CUDA_VISIBLE_DEVICES": value})
