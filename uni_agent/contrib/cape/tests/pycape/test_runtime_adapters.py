# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import (
    ApptainerRuntimeAdapter,
    BubbleWrapRuntimeAdapter,
    ContainerCommandSpec,
    ContainerSessionSpec,
    EnforcementMode,
    LocalProcessRuntimeAdapter,
    MountSpec,
    PodmanRuntimeAdapter,
    RuntimeExecution,
    RuntimeLaunch,
    RuntimeResult,
)


class ContainerRuntimeCommandTests(unittest.TestCase):
    def test_toolchain_mounts_follow_synthetic_parents(self):
        for policy in ("host-stateless", "host-stateful"):
            for root in ("/home/toolchain", "/tmp/toolchain"):
                with mock.patch("os.path.isdir", return_value=True):
                    adapter = BubbleWrapRuntimeAdapter(system_paths=(root, "/usr"), overlay_supported=True)
                session = ContainerSessionSpec("toolchain", isolation_policy=policy,
                                               overlay_upper="/tmp/upper", overlay_work="/tmp/work")
                with mock.patch("os.path.exists", return_value=True):
                    argv = adapter.build_run_command(session, ContainerCommandSpec(("true",))).argv
                parent = root.rsplit("/", 1)[0]
                parent_mount = next(i for i in range(len(argv) - 1)
                                    if argv[i:i + 2] == ["--tmpfs", parent])
                self.assertLess(parent_mount, argv.index(root))

    def test_probe_interpreter_follows_configured_root_order(self):
        for roots in (("/home/toolchain", "/usr"), ("/usr", "/home/toolchain")):
            with mock.patch("os.path.isdir", return_value=True):
                adapter = BubbleWrapRuntimeAdapter(system_paths=roots)
            with mock.patch("os.path.realpath", side_effect=lambda p: p), \
                    mock.patch("os.path.isfile", return_value=True), \
                    mock.patch("os.access", return_value=True):
                self.assertEqual(adapter._probe_python_executable(), roots[0] + "/bin/python3")

    def test_probe_interpreter_preserves_python_only_toolchains(self):
        with tempfile.TemporaryDirectory() as root:
            executable = Path(root) / "bin/python"
            executable.parent.mkdir()
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            adapter = BubbleWrapRuntimeAdapter(system_paths=(root, "/usr"))
            self.assertEqual(adapter._probe_python_executable(), str(executable.resolve()))

    def test_virtual_subpaths_are_optional_read_only_and_never_overlaid(self):
        for policy in ("host-stateless", "host-stateful"):
            adapter = BubbleWrapRuntimeAdapter(
                overlay_supported=True,
                read_only_paths=("/sys/devices/system/cpu", "/sys/devices/system/cpu"),
            )
            session = ContainerSessionSpec("virtual", isolation_policy=policy,
                                           overlay_upper="/tmp/cape-upper", overlay_work="/tmp/cape-work")
            argv = adapter.build_run_command(session, ContainerCommandSpec(("true",))).argv
            path = "/sys/devices/system/cpu"
            index = argv.index(path)
            self.assertEqual(argv[index - 1:index + 2], ["--ro-bind", path, path])
            self.assertEqual(argv.count(path), 2)
            self.assertNotIn("/sys", argv)
        for path in ("/sys", "/proc", "/dev", "/run", "/sys/../usr", "/usr/lib"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                BubbleWrapRuntimeAdapter(read_only_paths=(path,))
        with mock.patch("pycape.runtime.os.path.realpath", return_value="/etc"):
            with self.assertRaises(ValueError):
                BubbleWrapRuntimeAdapter(read_only_paths=("/sys/escape",))

    def test_empty_and_absent_virtual_subpaths_preserve_launch(self):
        session = ContainerSessionSpec("virtual", isolation_policy="host-stateless")
        command = ContainerCommandSpec(("true",))
        base = BubbleWrapRuntimeAdapter().build_run_command(session, command).argv
        missing = BubbleWrapRuntimeAdapter(read_only_paths=("/sys/cape-nonexistent-path",))
        self.assertEqual(base, missing.build_run_command(session, command).argv)

    def test_cpu_sandbox_clears_caller_gpu_visibility(self):
        for adapter, policy in (
            (ApptainerRuntimeAdapter(), "image-stateless"),
            (ApptainerRuntimeAdapter(), "image-stateful"),
            (PodmanRuntimeAdapter(), "image-stateless"),
            (PodmanRuntimeAdapter(), "image-stateful"),
            (BubbleWrapRuntimeAdapter(), "host-stateless"),
            (BubbleWrapRuntimeAdapter(), "host-stateful"),
        ):
            with self.subTest(adapter=adapter.name, policy=policy):
                session = ContainerSessionSpec(
                    "cpu", image="test-image", isolation_policy=policy,
                    env={"CUDA_VISIBLE_DEVICES": "GPU-parent"},
                )
                command = ContainerCommandSpec(
                    ("true",), env={"CUDA_VISIBLE_DEVICES": "GPU-caller"}
                )
                self.assertEqual(adapter._merged_env(session, command)["CUDA_VISIBLE_DEVICES"], "")
                if isinstance(adapter, BubbleWrapRuntimeAdapter):
                    self.assertEqual(adapter._sandbox_environment(session, command, None)["CUDA_VISIBLE_DEVICES"], "")
                if isinstance(adapter, ApptainerRuntimeAdapter):
                    self.assertIn("CUDA_VISIBLE_DEVICES=", adapter._env_flags(command.env, ()))

    def test_driver_search_paths_preserve_caller_precedence(self):
        adapter = BubbleWrapRuntimeAdapter()
        with mock.patch.object(adapter, "_nvidia_smi_binary", return_value="/usr/bin/nvidia-smi"), mock.patch.object(adapter, "_nvidia_driver_paths", return_value=("/driver/lib/libcuda.so",)):
            env = adapter._gpu_search_path_environment({"PATH": "/task/bin", "LD_LIBRARY_PATH": "/task/lib"})
        self.assertEqual(env["PATH"], "/task/bin:/usr/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/task/lib:/driver/lib")

    def test_runtime_result_retains_the_exact_executed_launch(self):
        adapter = ApptainerRuntimeAdapter(binary="/opt/apptainer")
        process = mock.Mock()
        process.communicate.return_value = ("payload\n", "")
        process.returncode = 0
        launch = RuntimeLaunch(
            ["/opt/apptainer", "exec", "--env", "MODE=test", "/image.sif", "true"],
            env={"APPTAINER_CACHEDIR": "/cache"},
            cwd="/host/work",
        )

        with mock.patch("pycape.runtime.subprocess.Popen", return_value=process):
            result = adapter._run_launch(launch)

        self.assertEqual(result.argv, launch.argv)
        self.assertEqual(result.cwd, "/host/work")
        self.assertEqual(result.env, {"APPTAINER_CACHEDIR": "/cache"})
        self.assertEqual(len(result.executions), 1)
        self.assertEqual(result.executions[0].phase, "command")
        self.assertEqual(result.executions[0].argv, launch.argv)
        self.assertEqual(result.executions[0].exit_code, 0)

    def test_none_policy_launches_the_workload_directly(self):
        adapter = LocalProcessRuntimeAdapter()
        process = mock.Mock()
        with mock.patch("pycape.runtime.subprocess.Popen", return_value=process) as popen:
            adapter.start_process(("python", "workload.py"), cwd="/workspace")

        self.assertEqual(popen.call_args.args[0], ["python", "workload.py"])
        self.assertEqual(popen.call_args.kwargs["cwd"], "/workspace")
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_apptainer_exec_command_exposes_selected_gpu_and_binds_paths(self):
        adapter = ApptainerRuntimeAdapter(
            binary="/opt/apptainer",
            warm_sessions=False,
            default_image="docker://ubuntu:latest",
            gpu_enforcement=EnforcementMode.ADVISORY,
        )
        session = ContainerSessionSpec(
            execution_session_id="sess:one",
            image="/images/train.sif",
            mounts=(
                MountSpec("/host/workspace", "/workspace"),
                MountSpec("/host/output", "/output", read_only=True),
            ),
            cwd="/workspace",
            env={"SESSION_ENV": "1"},
            gpu_uuids=("GPU-abc",),
            cache_dir="/local/cache",
            tmp_dir="/local/tmp",
        )
        command = ContainerCommandSpec(("python", "train.py"), env={"RUN_ENV": "2"})
        launch = adapter.build_run_command(session, command)

        self.assertEqual(launch.argv[:2], ["/opt/apptainer", "exec"])
        self.assertIn("--nv", launch.argv)
        self.assertIn("/host/workspace:/workspace:rw", launch.argv)
        self.assertIn("/host/output:/output:ro", launch.argv)
        self.assertIn("--pwd", launch.argv)
        self.assertIn("/workspace", launch.argv)
        self.assertIn("/images/train.sif", launch.argv)
        self.assertEqual(launch.argv[-2:], ["python", "train.py"])
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-abc", launch.argv)
        self.assertEqual(launch.env["CUDA_VISIBLE_DEVICES"], "GPU-abc")
        self.assertEqual(launch.env["APPTAINER_CACHEDIR"], "/local/cache")
        self.assertEqual(launch.env["APPTAINER_TMPDIR"], "/local/tmp")
        self.assertEqual(launch.enforcement_mode, EnforcementMode.ADVISORY)
        self.assertIn("advisory", launch.enforcement_message)

    def test_apptainer_policies_use_fresh_exec_and_only_stateful_adds_overlay(self):
        adapter = ApptainerRuntimeAdapter(
            binary="apptainer", overlay_supported=True, unsquash=True
        )
        stateless = ContainerSessionSpec(
            "sess/stateless",
            image="/images/base.sif",
            cwd="/work",
            isolation_policy="image-stateless",
        )
        stateful = ContainerSessionSpec(
            "sess/stateful",
            image="/images/base.sif",
            cwd="/work",
            isolation_policy="image-stateful",
            overlay_path="/state/session-overlay",
        )

        stateless_run = adapter.build_run_command(
            stateless, ContainerCommandSpec(("echo", "stateless"))
        )
        stateful_run = adapter.build_run_command(
            stateful, ContainerCommandSpec(("echo", "stateful"))
        )

        self.assertEqual(stateless_run.argv[:2], ["apptainer", "exec"])
        self.assertNotIn("--overlay", stateless_run.argv)
        self.assertIn("--unsquash", stateless_run.argv)
        self.assertEqual(stateful_run.argv[:2], ["apptainer", "exec"])
        overlay_index = stateful_run.argv.index("--overlay")
        self.assertEqual(
            stateful_run.argv[overlay_index + 1], "/state/session-overlay"
        )
        for launch in (stateless_run, stateful_run):
            self.assertNotIn("instance", launch.argv)
            self.assertFalse(any("instance://" in part for part in launch.argv))

    def test_apptainer_stateful_overlay_lifecycle_is_session_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = ApptainerRuntimeAdapter(
                binary="apptainer",
                runtime_dir=tmp,
                overlay_supported=True,
            )
            session = ContainerSessionSpec(
                "sess/layer",
                image="/images/base.sif",
                isolation_policy="image-stateful",
                state_layer_id="layer-identity",
            )

            first_start = adapter.start_session(session)
            self.assertEqual(first_start.exit_code, 0, first_start.stderr)
            self.assertTrue(os.path.exists(session.overlay_path))
            first_path = session.overlay_path
            second_start = adapter.start_session(session)
            self.assertEqual(second_start.exit_code, 0, second_start.stderr)
            self.assertEqual(session.overlay_path, first_path)
            run = adapter.build_run_command(
                session, ContainerCommandSpec(("true",))
            )
            self.assertEqual(run.argv[run.argv.index("--overlay") + 1], first_path)

            stopped = adapter.stop_session(session.execution_session_id)
            self.assertEqual(stopped.exit_code, 0, stopped.stderr)
            self.assertTrue(os.path.exists(first_path))
            self.assertEqual(adapter.resume_session(session).exit_code, 0)
            self.assertEqual(adapter.purge_session(session.execution_session_id).exit_code, 0)
            self.assertFalse(os.path.exists(first_path))

    def test_container_runtime_creates_resolved_node_local_directories_before_launch(self):
        class RecordingApptainerRuntime(ApptainerRuntimeAdapter):
            def __init__(self, **kwargs):
                ApptainerRuntimeAdapter.__init__(self, **kwargs)
                self.launches = []

            def _run_launch(self, launch, timeout=None, reap_process_group=False):
                self.launches.append(launch)
                return RuntimeResult(0, "", "")

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "cache", "node-0")
            scratch = os.path.join(tmp, "tmp", "node-0")
            runtime = os.path.join(tmp, "runtime", "node-0")
            adapter = RecordingApptainerRuntime(
                binary="apptainer",
                cache_dir=cache,
                tmp_dir=scratch,
                runtime_dir=runtime,
            )
            session = ContainerSessionSpec("sess/dirs", image="docker://ubuntu:latest")

            self.assertFalse(os.path.exists(cache))
            self.assertFalse(os.path.exists(scratch))
            self.assertFalse(os.path.exists(runtime))
            result = adapter.start_session(session)

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.isdir(cache))
            self.assertTrue(os.path.isdir(scratch))
            self.assertTrue(os.path.isdir(runtime))
            self.assertEqual(adapter.sessions[session.execution_session_id].cache_dir, None)
            run = adapter.build_run_command(
                session, ContainerCommandSpec(("true",))
            )
            self.assertEqual(run.env["APPTAINER_CACHEDIR"], cache)
            self.assertEqual(run.env["APPTAINER_TMPDIR"], scratch)
            self.assertEqual(run.env["APPTAINER_LOCALCACHEDIR"], runtime)

    def test_podman_run_command_uses_node_local_storage_and_cdi_gpu(self):
        adapter = PodmanRuntimeAdapter(
            binary="podman",
            warm_sessions=False,
            graphroot="/local/graph",
            runroot="/local/run",
            tmpdir="/local/tmp",
            gpu_device_mode="cdi",
            gpu_enforcement=EnforcementMode.HARD,
        )
        session = ContainerSessionSpec(
            execution_session_id="sess-podman",
            image="ubuntu:latest",
            mounts=(MountSpec("/host/ws", "/workspace"),),
            cwd="/workspace",
            gpu_uuids=("GPU-0", "GPU-1"),
        )
        launch = adapter.build_run_command(session, ContainerCommandSpec(("bash", "-lc", "echo ok")))

        self.assertEqual(launch.argv[:7], ["podman", "--root", "/local/graph", "--runroot", "/local/run", "--tmpdir", "/local/tmp"])
        self.assertIn("run", launch.argv)
        self.assertIn("--rm", launch.argv)
        self.assertIn("/host/ws:/workspace:rw", launch.argv)
        self.assertIn("--userns=keep-id", launch.argv)
        self.assertIn("CUDA_VISIBLE_DEVICES=GPU-0,GPU-1", launch.argv)
        self.assertIn("nvidia.com/gpu=GPU-0", launch.argv)
        self.assertIn("nvidia.com/gpu=GPU-1", launch.argv)
        self.assertEqual(launch.enforcement_mode, EnforcementMode.HARD)

    def test_podman_stateless_runs_and_stateful_is_rejected(self):
        adapter = PodmanRuntimeAdapter(binary="podman", warm_sessions=True)
        stateless = ContainerSessionSpec(
            "podman-stateless",
            image="ubuntu:latest",
            isolation_policy="image-stateless",
        )
        stateful = ContainerSessionSpec(
            "podman-stateful",
            image="ubuntu:latest",
            isolation_policy="image-stateful",
        )

        stateless_run = adapter.build_run_command(
            stateless, ContainerCommandSpec(("true",))
        )
        self.assertIn("run", stateless_run.argv)
        self.assertIn("--rm", stateless_run.argv)
        with self.assertRaisesRegex(ValueError, "not supported"):
            adapter.build_start_session_command(stateful)
        with self.assertRaisesRegex(ValueError, "not supported"):
            adapter.build_run_command(stateful, ContainerCommandSpec(("true",)))

    def test_bubblewrap_policies_use_assembled_allowlisted_roots(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="/opt/bwrap",
            overlay_supported=True,
            system_paths=("/usr", "/etc", "/opt"),
        )
        stateless = ContainerSessionSpec(
            "host/stateless",
            isolation_policy="host-stateless",
            mounts=(MountSpec("/host/workspace", "/workspace"),),
            cwd="/workspace",
        )
        stateful = ContainerSessionSpec(
            "host/stateful",
            isolation_policy="host-stateful",
            overlay_upper="/state/upper",
            overlay_work="/state/work",
        )

        stateless_run = adapter.build_run_command(
            stateless, ContainerCommandSpec(("sh", "-c", "true"))
        )
        stateful_run = adapter.build_run_command(
            stateful, ContainerCommandSpec(("true",))
        )

        triples = [
            stateless_run.argv[index : index + 3]
            for index in range(len(stateless_run.argv) - 2)
        ]
        self.assertIn(["--tmpfs", "/"], [stateless_run.argv[i:i + 2] for i in range(len(stateless_run.argv) - 1)])
        self.assertNotIn(["--ro-bind", "/", "/"], triples)
        for path in ("/usr", "/etc", "/opt"):
            self.assertIn(["--ro-bind", path, path], triples)
        self.assertIn("--die-with-parent", stateless_run.argv)
        self.assertIn("--new-session", stateless_run.argv)
        self.assertIn("--clearenv", stateless_run.argv)
        self.assertIn("--unshare-user", stateless_run.argv)
        self.assertIn("--unshare-pid", stateless_run.argv)
        self.assertIn("--proc", stateless_run.argv)
        self.assertIn("--tmpfs", stateless_run.argv)
        self.assertIn("--bind", stateless_run.argv)
        self.assertNotIn("--overlay", stateless_run.argv)
        stateful_triples = [stateful_run.argv[i:i + 3] for i in range(len(stateful_run.argv) - 2)]
        self.assertNotIn(["--ro-bind", "/", "/"], stateful_triples)
        self.assertNotIn(["--ro-bind", "/sys", "/sys"], stateful_triples)
        overlay_destinations = []
        for index, value in enumerate(stateful_run.argv):
            if value == "--overlay":
                overlay_destinations.append(stateful_run.argv[index + 3])
                self.assertTrue(stateful_run.argv[index + 1].startswith("/state/upper/"))
                self.assertTrue(stateful_run.argv[index + 2].startswith("/state/work/"))
        self.assertEqual(overlay_destinations, ["/usr", "/etc", "/opt"])
        self.assertEqual(stateful_run.argv.count("--overlay-src"), 3)

    def test_bubblewrap_rejects_root_and_relative_system_paths(self):
        for paths in (
            ("/",),
            ("relative",),
            ("/proc",),
            ("/sys",),
            ("/dev",),
            ("/run",),
            (),
        ):
            with self.subTest(paths=paths):
                with self.assertRaises(ValueError):
                    BubbleWrapRuntimeAdapter(system_paths=paths)

    def test_bubblewrap_extra_args_cannot_bypass_the_filesystem_contract(self):
        with self.assertRaisesRegex(ValueError, "extra_args"):
            BubbleWrapRuntimeAdapter(
                extra_args=("--ro-bind", "/", "/")
            )

    def test_bubblewrap_clearenv_rebuilds_only_runtime_and_explicit_environment(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap", system_paths=("/usr",), overlay_supported=False
        )
        session = ContainerSessionSpec(
            "host/env",
            isolation_policy="host-stateless",
            cwd="/workspace",
            env={"SESSION_ONLY": "one"},
        )
        launch = adapter.build_run_command(
            session,
            ContainerCommandSpec(("true",), env={"COMMAND_ONLY": "two"}),
        )

        self.assertIn("--clearenv", launch.argv)
        set_environment = {
            launch.argv[index + 1]: launch.argv[index + 2]
            for index, value in enumerate(launch.argv)
            if value == "--setenv"
        }
        self.assertEqual(set_environment["HOME"], "/home/cape")
        self.assertEqual(set_environment["PWD"], "/workspace")
        self.assertIn("/usr/bin", set_environment["PATH"])
        self.assertEqual(set_environment["SESSION_ONLY"], "one")
        self.assertEqual(set_environment["COMMAND_ONLY"], "two")
        self.assertNotIn("USER", set_environment)
        self.assertNotIn("SSH_AUTH_SOCK", set_environment)

    def test_bubblewrap_reports_hard_gpu_enforcement_for_exact_device_binding(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap", system_paths=("/usr",), overlay_supported=False
        )
        capabilities = adapter.probe(execute=False)
        self.assertEqual(capabilities.gpu_enforcement, EnforcementMode.HARD)

    def test_bubblewrap_binds_only_the_leased_gpu_minor_and_shared_devices(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="/opt/bwrap",
            overlay_supported=True,
            gpu_device_map={"GPU-A": 3, "GPU-B": 7},
            gpu_shared_device_paths=(
                "/dev/nvidiactl",
                "/dev/nvidia-uvm",
                "/dev/nvidia-uvm-tools",
                "/dev/nvidia-caps",
            ),
            nvidia_smi_binary="/usr/bin/nvidia-smi",
            nvidia_driver_paths=(
                "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
                "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
            ),
        )
        session = ContainerSessionSpec(
            "host/gpu",
            isolation_policy="host-stateless",
            gpu_uuids=("GPU-A",),
        )

        launch = adapter.build_run_command(
            session, ContainerCommandSpec(("nvidia-smi", "-L"))
        )
        device_sources = [
            launch.argv[index + 1]
            for index, value in enumerate(launch.argv)
            if value == "--dev-bind"
        ]

        self.assertIn("/dev/nvidia3", device_sources)
        self.assertNotIn("/dev/nvidia7", device_sources)
        self.assertNotIn("/dev/nvidia0", device_sources)
        self.assertNotIn("/dev", device_sources)
        self.assertIn("/dev/nvidiactl", device_sources)
        self.assertIn("/dev/nvidia-uvm", device_sources)
        self.assertIn("/dev/nvidia-uvm-tools", device_sources)
        self.assertIn("/dev/nvidia-caps", device_sources)
        self.assertIn("/usr/bin/nvidia-smi", launch.argv)
        self.assertIn("/usr/lib/aarch64-linux-gnu/libcuda.so.1", launch.argv)
        self.assertIn("/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1", launch.argv)
        self.assertNotIn("/sys", launch.argv)
        self.assertEqual(launch.env["CUDA_VISIBLE_DEVICES"], "GPU-A")

    def test_bubblewrap_two_gpu_lease_binds_exactly_those_two_minors(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            overlay_supported=False,
            gpu_device_map={"GPU-A": 1, "GPU-B": 6, "GPU-C": 8},
            gpu_shared_device_paths=("/dev/nvidiactl",),
            nvidia_smi_binary="/usr/bin/nvidia-smi",
            nvidia_driver_paths=("/usr/lib/libcuda.so.1",),
        )
        session = ContainerSessionSpec(
            "host/two-gpus",
            isolation_policy="host-stateless",
            gpu_uuids=("GPU-A", "GPU-C"),
        )

        argv = adapter.build_run_command(
            session, ContainerCommandSpec(("true",))
        ).argv
        per_gpu_devices = {
            argv[index + 1]
            for index, value in enumerate(argv)
            if value == "--dev-bind"
            and argv[index + 1].startswith("/dev/nvidia")
            and argv[index + 1][len("/dev/nvidia") :].isdigit()
        }

        self.assertEqual(per_gpu_devices, {"/dev/nvidia1", "/dev/nvidia8"})

    def test_bubblewrap_gpu_launch_fails_closed_when_uuid_cannot_be_resolved(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            overlay_supported=False,
            gpu_device_map={"GPU-A": 1},
            gpu_shared_device_paths=("/dev/nvidiactl",),
            nvidia_smi_binary="/usr/bin/nvidia-smi",
            nvidia_driver_paths=("/usr/lib/libcuda.so.1",),
        )
        session = ContainerSessionSpec(
            "host/unresolved",
            isolation_policy="host-stateless",
            gpu_uuids=("GPU-UNKNOWN",),
        )

        with self.assertRaisesRegex(ValueError, "GPU-UNKNOWN"):
            adapter.build_run_command(session, ContainerCommandSpec(("true",)))

    def test_bubblewrap_supplements_partial_proc_gpu_map_with_nvidia_smi(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            nvidia_smi_binary="/usr/bin/nvidia-smi",
        )
        with tempfile.TemporaryDirectory() as temporary:
            information = os.path.join(temporary, "information")
            with open(information, "w") as output:
                output.write("GPU UUID: GPU-A\nDevice Minor: 0\n")
            failed_minor_query = subprocess.CalledProcessError(
                1, ["nvidia-smi"]
            )
            with mock.patch(
                "pycape.runtime.glob.glob", return_value=[information]
            ), mock.patch(
                "pycape.runtime.subprocess.check_output",
                side_effect=(failed_minor_query, "GPU-A, 0\nGPU-B, 7\n"),
            ):
                resolved = adapter._gpu_device_map()

        self.assertEqual(resolved, {"GPU-A": 0, "GPU-B": 7})

    def test_bubblewrap_canonicalizes_symlinked_driver_assets_for_binds(self):
        adapter = BubbleWrapRuntimeAdapter(binary="bwrap")
        with tempfile.TemporaryDirectory() as tmp:
            real_directory = os.path.join(tmp, "lib.real")
            alias_directory = os.path.join(tmp, "lib")
            os.mkdir(real_directory)
            os.symlink(real_directory, alias_directory)
            target = os.path.join(real_directory, "libcuda.so.610")
            with open(target, "w") as fh:
                fh.write("driver")
            alias = os.path.join(alias_directory, "libcuda.so")
            os.symlink("libcuda.so.610", alias)

            with mock.patch(
                "pycape.runtime.glob.glob", return_value=[alias]
            ):
                paths = adapter._nvidia_driver_paths()

        self.assertEqual(paths, (target,))

    def test_bubblewrap_both_host_policies_use_fresh_proc_and_explicit_gpu_assets(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            overlay_supported=True,
            gpu_device_map={"GPU-A": 4},
            gpu_shared_device_paths=("/dev/nvidiactl",),
            nvidia_smi_binary="/usr/bin/nvidia-smi",
            nvidia_driver_paths=("/usr/lib/libcuda.so.1",),
        )
        sessions = (
            ContainerSessionSpec(
                "host/stateless-gpu",
                isolation_policy="host-stateless",
                gpu_uuids=("GPU-A",),
            ),
            ContainerSessionSpec(
                "host/stateful-gpu",
                isolation_policy="host-stateful",
                gpu_uuids=("GPU-A",),
                overlay_upper="/state/upper",
                overlay_work="/state/work",
            ),
        )

        for session in sessions:
            argv = adapter.build_run_command(
                session, ContainerCommandSpec(("true",))
            ).argv
            self.assertIn("--unshare-pid", argv)
            self.assertIn("--proc", argv)
            self.assertNotIn("/sys", argv)
            self.assertIn("/dev/nvidia4", argv)
            self.assertIn("/usr/lib/libcuda.so.1", argv)
            self.assertFalse(
                any(
                    argv[index : index + 3] == ["--bind", "/proc", "/proc"]
                    for index in range(len(argv) - 2)
                )
            )

    def test_bubblewrap_rejects_the_host_proc_view_configuration(self):
        with self.assertRaisesRegex(ValueError, "host.*proc|host.view"):
            BubbleWrapRuntimeAdapter(host_view_mode=True)

    def test_bubblewrap_gpu_probe_runs_nvidia_smi_before_cuda_initialization(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            overlay_supported=False,
            node_gpu_uuids=("GPU-A",),
            gpu_device_map={"GPU-A": 3},
            gpu_shared_device_paths=("/dev/nvidiactl",),
            nvidia_smi_binary="/usr/bin/nvidia-smi",
            nvidia_driver_paths=("/usr/lib/libcuda.so.1",),
        )
        results = (
            RuntimeResult(0, "GPU-A\n", ""),
            RuntimeResult(0, "1\n", ""),
            RuntimeResult(0, "cuda-kernel-ok\n", ""),
        )
        with mock.patch.object(
            adapter, "start_session", return_value=RuntimeResult(0, "", "")
        ), mock.patch.object(
            adapter, "run_command", side_effect=results
        ) as run_command, mock.patch.object(adapter, "stop_session"):
            proven = adapter._probe_gpu_policy("host-stateless")

        self.assertTrue(proven)
        first = run_command.call_args_list[0].args[1]
        second = run_command.call_args_list[1].args[1]
        third = run_command.call_args_list[2].args[1]
        probe_python = adapter._probe_python_executable()
        self.assertEqual(first[0], "/usr/bin/nvidia-smi")
        self.assertIn("--query-gpu=uuid", first)
        self.assertEqual(second[0], probe_python)
        self.assertIn("cuInit", second[-1])
        self.assertIn("print(rc,rc2,count.value)", second[-1])
        self.assertEqual(third[0], probe_python)
        self.assertIn("import torch", third[-1])
        self.assertIn("cuda-kernel-ok", third[-1])

    def test_bubblewrap_gpu_probe_never_uses_an_interpreter_outside_the_allowlist(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            system_paths=("/usr",),
            overlay_supported=False,
            node_gpu_uuids=("GPU-A",),
            gpu_device_map={"GPU-A": 3},
            gpu_shared_device_paths=("/dev/nvidiactl",),
            nvidia_smi_binary="/usr/bin/nvidia-smi",
            nvidia_driver_paths=("/usr/lib/libcuda.so.1",),
        )
        results = (
            RuntimeResult(0, "GPU-A\n", ""),
            RuntimeResult(0, "0 0 1\n", ""),
            RuntimeResult(0, "cuda-kernel-ok\n", ""),
        )
        with mock.patch("pycape.runtime.sys.executable", "/var/tmp/cape/venv/bin/python"), \
                mock.patch("pycape.runtime.shutil.which", return_value="/var/tmp/cape/venv/bin/python"), \
                mock.patch.object(
                    adapter, "start_session", return_value=RuntimeResult(0, "", "")
                ), mock.patch.object(
                    adapter, "run_command", side_effect=results
                ) as run_command, mock.patch.object(adapter, "stop_session"):
            proven = adapter._probe_gpu_policy("host-stateless")

        self.assertTrue(proven)
        probe_python = run_command.call_args_list[1].args[1][0]
        self.assertEqual(probe_python, run_command.call_args_list[2].args[1][0])
        self.assertTrue(
            probe_python == "/usr"
            or probe_python.startswith("/usr/"),
            probe_python,
        )

    def test_bubblewrap_gpu_probe_preserves_the_fail_closed_reason(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            overlay_supported=False,
            node_gpu_uuids=("GPU-A",),
            gpu_device_map={"GPU-A": 3},
            gpu_shared_device_paths=("/dev/nvidiactl",),
            nvidia_smi_binary="/usr/bin/nvidia-smi",
            nvidia_driver_paths=("/usr/lib/libcuda.so.1",),
        )
        with mock.patch.object(
            adapter, "start_session", return_value=RuntimeResult(0, "", "")
        ), mock.patch.object(
            adapter,
            "run_command",
            side_effect=(
                RuntimeResult(0, "GPU-A\n", ""),
                RuntimeResult(1, "304 304 0\n", ""),
            ),
        ), mock.patch.object(adapter, "stop_session"):
            proven = adapter._probe_gpu_policy("host-stateless")

        self.assertFalse(proven)
        self.assertIn(
            "304", adapter.gpu_probe_failures["host-stateless"]
        )

    def test_bubblewrap_reports_gpu_support_and_rootful_reduced_isolation_per_policy(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            overlay_supported=True,
            user_namespace_mode=False,
            node_gpu_uuids=("GPU-A",),
        )
        with mock.patch.object(adapter, "binary_available", return_value=True), mock.patch.object(
            adapter, "version", return_value=RuntimeResult(0, "bubblewrap 0.9", "")
        ), mock.patch.object(adapter, "_probe_stateless", return_value=True), mock.patch.object(
            adapter, "_probe_gpu_policy", side_effect=(True, False)
        ) as gpu_probe:
            capabilities = adapter.probe(execute=True)

        self.assertEqual(
            gpu_probe.call_args_list,
            [mock.call("host-stateless"), mock.call("host-stateful")],
        )
        self.assertIn("host-stateless", capabilities.gpu_isolation_policies)
        self.assertNotIn("host-stateful", capabilities.gpu_isolation_policies)
        self.assertEqual(
            set(capabilities.reduced_isolation_policies),
            set(capabilities.isolation_policies),
        )

    def test_bubblewrap_probe_never_falls_back_to_a_host_proc_view(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="/opt/bwrap", overlay_supported=False
        )
        denied_proc = RuntimeResult(
            1, "", "bwrap: Can't mount proc on /newroot/proc: Operation not permitted"
        )
        with mock.patch.object(adapter, "binary_available", return_value=True), \
                mock.patch("pycape.runtime.os.geteuid", return_value=1000), \
                mock.patch.object(
                    adapter,
                    "version",
                    return_value=RuntimeResult(0, "bubblewrap 0.9.0", ""),
                ), \
                mock.patch.object(
                    adapter,
                    "_run_launch",
                    return_value=denied_proc,
                ) as run_launch:
            capabilities = adapter.probe(execute=True)

        self.assertEqual(run_launch.call_count, 1)
        isolated = run_launch.call_args_list[0].args[0].argv
        self.assertIn("--unshare-pid", isolated)
        self.assertIn("--proc", isolated)
        self.assertNotIn(
            ["--ro-bind", "/proc", "/proc"],
            [isolated[index : index + 3] for index in range(len(isolated) - 2)],
        )
        self.assertIsNone(adapter.user_namespace_mode)
        self.assertFalse(capabilities.nested_container)
        self.assertEqual(capabilities.isolation_policies, ())

    def test_bubblewrap_rootful_fresh_proc_fallback_drops_all_capabilities(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="/opt/bwrap", overlay_supported=False
        )
        denied_proc = RuntimeResult(
            1, "", "bwrap: Can't mount proc on /newroot/proc: Operation not permitted"
        )
        with mock.patch.object(adapter, "binary_available", return_value=True), \
                mock.patch("pycape.runtime.os.geteuid", return_value=0), \
                mock.patch.object(
                    adapter,
                    "version",
                    return_value=RuntimeResult(0, "bubblewrap 0.9.0", ""),
                ), \
                mock.patch.object(
                    adapter,
                    "_run_launch",
                    side_effect=(
                        denied_proc,
                        RuntimeResult(0, "cape-bubblewrap-probe", ""),
                    ),
                ) as run_launch:
            capabilities = adapter.probe(execute=True)

        self.assertEqual(run_launch.call_count, 2)
        preferred = run_launch.call_args_list[0].args[0].argv
        rootful = run_launch.call_args_list[1].args[0].argv
        self.assertIn("--unshare-user", preferred)
        self.assertNotIn("--unshare-user", rootful)
        self.assertIn("--unshare-pid", rootful)
        self.assertIn("--proc", rootful)
        cap_drop = rootful.index("--cap-drop")
        self.assertEqual(rootful[cap_drop : cap_drop + 2], ["--cap-drop", "ALL"])
        self.assertFalse(adapter.host_view_mode)
        self.assertFalse(adapter.user_namespace_mode)
        self.assertIn(
            "host-stateless", capabilities.reduced_isolation_policies
        )

    def test_bubblewrap_probe_rejects_a_binary_that_ignores_the_sandbox_argv(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="/bin/true", overlay_supported=False
        )
        capabilities = adapter.probe(execute=True)
        self.assertFalse(capabilities.nested_container)
        self.assertEqual(capabilities.isolation_policies, ())

    def test_bubblewrap_failed_fresh_proc_probe_advertises_no_policy(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="/opt/bwrap", overlay_supported=False
        )
        denied = RuntimeResult(1, "", "Operation not permitted")
        with mock.patch.object(adapter, "binary_available", return_value=True), \
                mock.patch("pycape.runtime.os.geteuid", return_value=1000), \
                mock.patch.object(
                    adapter,
                    "version",
                    return_value=RuntimeResult(0, "bubblewrap 0.9.0", ""),
                ), \
                mock.patch.object(
                    adapter, "_run_launch", return_value=denied
                ) as run_launch:
            capabilities = adapter.probe(execute=True)

        self.assertEqual(run_launch.call_count, 1)
        self.assertIsNone(adapter.user_namespace_mode)
        self.assertFalse(capabilities.nested_container)
        self.assertEqual(capabilities.isolation_policies, ())

    def test_bubblewrap_stateful_layer_is_created_once_and_removed_on_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = BubbleWrapRuntimeAdapter(
                binary="bwrap", runtime_dir=tmp, overlay_supported=True
            )
            session = ContainerSessionSpec(
                "host/layer",
                isolation_policy="host-stateful",
                state_layer_id="host-layer-identity",
            )
            self.assertEqual(adapter.start_session(session).exit_code, 0)
            self.assertTrue(os.path.isdir(session.overlay_upper))
            self.assertTrue(os.path.isdir(session.overlay_work))
            first_upper = session.overlay_upper
            self.assertEqual(adapter.start_session(session).exit_code, 0)
            self.assertEqual(session.overlay_upper, first_upper)
            self.assertEqual(adapter.stop_session(session.execution_session_id).exit_code, 0)
            self.assertTrue(os.path.exists(os.path.dirname(first_upper)))
            self.assertEqual(adapter.resume_session(session).exit_code, 0)
            self.assertEqual(adapter.purge_session(session.execution_session_id).exit_code, 0)
            self.assertFalse(os.path.exists(os.path.dirname(first_upper)))

    def test_bubblewrap_fuse_fallback_binds_a_per_command_merged_root(self):
        adapter = BubbleWrapRuntimeAdapter(
            binary="bwrap",
            overlay_supported=True,
            fuse_overlayfs_binary="fuse-overlayfs",
            fusermount_binary="fusermount3",
        )
        adapter.overlay_mode = "fuse"
        session = ContainerSessionSpec(
            "host-fuse",
            isolation_policy="host-stateful",
            overlay_upper="/state/layer/upper",
            overlay_work="/state/layer/work",
        )

        launch = adapter.build_run_command(
            session, ContainerCommandSpec(("true",))
        )

        bind_triples = [launch.argv[index:index + 3] for index in range(len(launch.argv) - 2)]
        self.assertIn(["--bind", "/state/layer/merged/00-usr", "/usr"], bind_triples)
        self.assertIn(["--bind", "/state/layer/merged/01-etc", "/etc"], bind_triples)
        self.assertIn(["--bind", "/state/layer/merged/02-opt", "/opt"], bind_triples)
        self.assertNotIn(["--bind", "/state/layer/merged", "/"], bind_triples)
        self.assertNotIn("--overlay-src", launch.argv)

    def test_capability_probe_does_not_advertise_unproven_stateful_policy(self):
        unavailable = ApptainerRuntimeAdapter(
            binary=sys.executable, overlay_supported=False
        ).probe(execute=False)
        available = ApptainerRuntimeAdapter(
            binary=sys.executable, overlay_supported=True
        ).probe(execute=False)

        self.assertIn("image-stateless", unavailable.isolation_policies)
        self.assertNotIn("image-stateful", unavailable.isolation_policies)
        self.assertIn("image-stateful", available.isolation_policies)
        self.assertFalse(available.supports_warm_sessions)

    def test_container_adapter_exposes_a_cancellable_command_handle(self):
        adapter = ApptainerRuntimeAdapter(binary="/opt/apptainer")
        session = ContainerSessionSpec(
            "session",
            image="/image.sif",
            isolation_policy="image-stateless",
        )
        adapter.sessions[session.execution_session_id] = session
        process = mock.Mock(pid=1234)
        with mock.patch("pycape.runtime.subprocess.Popen", return_value=process):
            handle = adapter.start_command_process(
                session.execution_session_id,
                ("sleep", "300"),
                command_id="cmd-1",
            )
        self.assertEqual(handle.launch.argv[-2:], ["sleep", "300"])
        self.assertIs(handle.process, process)

    def test_podman_cancel_targets_stateless_container_and_rejects_stateful(self):
        process = mock.Mock(pid=1234)
        for policy, expected_action in (
            ("image-stateless", "rm"),
            ("default", "rm"),
        ):
            adapter = PodmanRuntimeAdapter(binary="podman", warm_sessions=True)
            session = ContainerSessionSpec(
                "session-" + policy,
                image="ubuntu:latest",
                isolation_policy=policy,
            )
            adapter.sessions[session.execution_session_id] = session
            with mock.patch("pycape.runtime.subprocess.Popen", return_value=process), \
                    mock.patch("pycape.runtime.subprocess.call") as call, \
                    mock.patch("pycape.runtime.terminate_process_tree") as terminate:
                handle = adapter.start_command_process(
                    session.execution_session_id,
                    ("sleep", "300"),
                    command_id="cmd-1",
                )
                handle.terminate_tree()
            self.assertEqual(call.call_args.args[0][1], expected_action)
            terminate.assert_called_once_with(process, 2.0)
        adapter.sessions["stateful"] = ContainerSessionSpec("stateful", isolation_policy="image-stateful")
        with mock.patch("pycape.runtime.subprocess.Popen") as start:
            with self.assertRaisesRegex(ValueError, "not supported"):
                adapter.start_command_process("stateful", ("true",))
            start.assert_not_called()

    def test_probe_inspection_reports_executable_binary_and_cache_capability(self):
        adapter = ApptainerRuntimeAdapter(binary=sys.executable, cache_dir="/local/cache")
        caps = adapter.probe(execute=False)
        self.assertEqual(caps.adapter, "apptainer")
        self.assertTrue(caps.nested_container)
        self.assertTrue(caps.node_local_cache)
        self.assertEqual(caps.gpu_enforcement, EnforcementMode.ADVISORY)

    def test_apptainer_join_probe_fails_closed_without_a_smoke_image(self):
        adapter = ApptainerRuntimeAdapter(binary=sys.executable)

        caps = adapter.probe(execute=True)

        self.assertFalse(caps.nested_container)
        self.assertEqual(caps.isolation_policies, ())

    def test_podman_join_probe_advertises_only_executed_policies(self):
        adapter = PodmanRuntimeAdapter(
            binary="/runtime/podman",
            probe_image="docker://example/probe:latest",
            warm_sessions=True,
        )
        with mock.patch.object(adapter, "binary_available", return_value=True), \
                mock.patch.object(
                    adapter,
                    "version",
                    return_value=RuntimeResult(0, "podman version", ""),
                ), \
                mock.patch.object(
                    adapter,
                    "_probe_policy",
                    side_effect=(True, False),
                ) as policy_probe:
            caps = adapter.probe(execute=True)

        self.assertEqual(
            policy_probe.call_args_list,
            [mock.call("image-stateless")],
        )
        self.assertTrue(caps.nested_container)
        self.assertIn("image-stateless", caps.isolation_policies)
        self.assertNotIn("image-stateful", caps.isolation_policies)
        self.assertFalse(caps.supports_warm_sessions)

    def test_podman_join_probe_fails_closed_without_a_smoke_image(self):
        adapter = PodmanRuntimeAdapter(
            binary="/runtime/podman",
            probe_image=None,
        )
        with mock.patch.object(adapter, "binary_available", return_value=True), \
                mock.patch.object(
                    adapter,
                    "version",
                    return_value=RuntimeResult(0, "podman version", ""),
                ):
            caps = adapter.probe(execute=True)

        self.assertFalse(caps.nested_container)
        self.assertEqual(caps.isolation_policies, ())

    def test_local_process_timeout_reaps_process_group(self):
        runtime = LocalProcessRuntimeAdapter()
        started = time.time()
        result = runtime.run_command(
            "local",
            (sys.executable, "-c", "import time; time.sleep(10)"),
            timeout=0.2,
        )
        elapsed = time.time() - started
        self.assertNotEqual(result.exit_code, 0)
        self.assertLess(elapsed, 3.0)

    def test_local_process_reaps_lingering_background_process_after_parent_exit(self):
        runtime = LocalProcessRuntimeAdapter()
        code = (
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "print('parent-done')"
        )
        started = time.time()
        result = runtime.run_command("local", (sys.executable, "-c", code))
        elapsed = time.time() - started
        self.assertEqual(result.stdout.strip(), "parent-done")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("lingering workload process group", result.stderr)
        self.assertLess(elapsed, 4.0)


class ContainerRuntimeAcceptanceTests(unittest.TestCase):
    def test_apptainer_runs_bound_cpu_command_when_configured(self):
        image = os.environ.get("PYCAPE_APPTAINER_TEST_IMAGE")
        if not image:
            self.skipTest("PYCAPE_APPTAINER_TEST_IMAGE is not configured")
        binary = os.environ.get("PYCAPE_APPTAINER_BINARY", "apptainer")
        adapter = ApptainerRuntimeAdapter(binary=binary, warm_sessions=False)
        probe = adapter.version(timeout=10)
        if probe.exit_code != 0:
            self.fail("apptainer probe failed: {}".format(probe.stderr.strip()))
        with tempfile.TemporaryDirectory() as tmp:
            workspace = os.path.join(tmp, "workspace")
            cache = os.path.join(tmp, "cache")
            scratch = os.path.join(tmp, "tmp")
            os.makedirs(workspace)
            os.makedirs(cache)
            os.makedirs(scratch)
            session = ContainerSessionSpec(
                execution_session_id="acceptance",
                image=image,
                mounts=(MountSpec(workspace, "/workspace"),),
                cwd="/workspace",
                cache_dir=cache,
                tmp_dir=scratch,
            )
            adapter.start_session(session)
            result = adapter.run_command(
                "acceptance",
                ("sh", "-lc", "echo pycape-acceptance > /workspace/out.txt && cat /workspace/out.txt"),
                timeout=60,
            )
            self.assertEqual(result.exit_code, 0, result.stderr)
            self.assertIn("pycape-acceptance", result.stdout)
            with open(os.path.join(workspace, "out.txt")) as fh:
                self.assertEqual(fh.read().strip(), "pycape-acceptance")
            adapter.cleanup_session("acceptance")


    def test_apptainer_stateful_overlay_persists_across_fresh_execs(self):
        image = os.environ.get("PYCAPE_APPTAINER_TEST_IMAGE")
        if not image:
            self.skipTest("PYCAPE_APPTAINER_TEST_IMAGE is not configured")
        binary = os.environ.get("PYCAPE_APPTAINER_BINARY", "apptainer")
        adapter = ApptainerRuntimeAdapter(
            binary=binary,
            probe_image=image,
            fakeroot=os.environ.get("PYCAPE_APPTAINER_TEST_FAKEROOT") == "1",
        )
        capabilities = adapter.probe(execute=True)
        self.assertIn(
            "image-stateful",
            capabilities.isolation_policies,
            "Apptainer writable overlay probe did not find a usable format",
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = os.path.join(tmp, "workspace")
            cache = os.path.join(tmp, "cache")
            scratch = os.path.join(tmp, "tmp")
            os.makedirs(workspace)
            os.makedirs(cache)
            os.makedirs(scratch)
            runtime = os.path.join(tmp, "runtime")
            os.makedirs(runtime)
            execution_session_id = "overlay-acceptance-{}".format(os.getpid())
            session = ContainerSessionSpec(
                execution_session_id=execution_session_id,
                image=image,
                mounts=(MountSpec(workspace, "/workspace"),),
                cwd="/workspace",
                cache_dir=cache,
                tmp_dir=scratch,
                runtime_dir=runtime,
                isolation_policy="image-stateful",
                state_layer_id="real-overlay-acceptance",
            )
            start = adapter.start_session(session)
            try:
                self.assertEqual(start.exit_code, 0, start.stderr)
                first = adapter.run_command(
                    execution_session_id,
                    (
                        "sh",
                        "-lc",
                        "mkdir -p /opt/pycape-overlay && "
                        "echo pycape-overlay > /opt/pycape-overlay/state.txt",
                    ),
                    timeout=60,
                )
                self.assertEqual(first.exit_code, 0, first.stderr)
                second = adapter.run_command(
                    execution_session_id,
                    ("cat", "/opt/pycape-overlay/state.txt"),
                    timeout=60,
                )
                self.assertEqual(second.exit_code, 0, second.stderr)
                self.assertEqual(second.stdout.strip(), "pycape-overlay")
                self.assertFalse(any("instance://" in part for part in adapter.build_run_command(
                    session, ContainerCommandSpec(("true",))
                ).argv))
            finally:
                adapter.cleanup_session(execution_session_id)


    def test_apptainer_reports_selected_gpu_visibility_when_configured(self):
        image = os.environ.get("PYCAPE_APPTAINER_TEST_IMAGE")
        gpu_uuid = os.environ.get("PYCAPE_APPTAINER_TEST_GPU_UUID")
        if not image or not gpu_uuid:
            self.skipTest("PYCAPE_APPTAINER_TEST_IMAGE and PYCAPE_APPTAINER_TEST_GPU_UUID are not configured")
        binary = os.environ.get("PYCAPE_APPTAINER_BINARY", "apptainer")
        adapter = ApptainerRuntimeAdapter(binary=binary, warm_sessions=False)
        probe = adapter.version(timeout=10)
        if probe.exit_code != 0:
            self.fail("apptainer probe failed: {}".format(probe.stderr.strip()))
        session = ContainerSessionSpec(
            execution_session_id="gpu-acceptance",
            image=image,
            gpu_uuids=(gpu_uuid,),
            cache_dir=os.environ.get("PYCAPE_APPTAINER_CACHE_DIR"),
            tmp_dir=os.environ.get("PYCAPE_APPTAINER_TMP_DIR"),
        )
        adapter.start_session(session)
        result = adapter.run_command(
            "gpu-acceptance",
            ("sh", "-lc", "printf '%s' \"$CUDA_VISIBLE_DEVICES\"; test -e /dev/nvidiactl"),
            timeout=60,
        )
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertEqual(result.stdout, gpu_uuid)
        launch = adapter.build_run_command(session, ContainerCommandSpec(("true",)))
        self.assertEqual(launch.enforcement_mode, EnforcementMode.ADVISORY)
        self.assertIn("advisory", launch.enforcement_message)
        adapter.cleanup_session("gpu-acceptance")

    def _assert_real_bubblewrap_gpu_policy(self, isolation_policy):
        gpu_uuid = os.environ.get("PYCAPE_BUBBLEWRAP_TEST_GPU_UUID")
        if not gpu_uuid:
            self.skipTest("PYCAPE_BUBBLEWRAP_TEST_GPU_UUID is not configured")
        binary = os.environ.get("PYCAPE_BUBBLEWRAP_BINARY", "bwrap")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = os.path.join(tmp, "granted-workspace")
            cache_source = os.path.join(tmp, "cache-source")
            os.mkdir(workspace)
            os.mkdir(cache_source)
            Path(workspace, "granted.txt").write_text("granted")
            Path(cache_source, "value").write_text("new-target")
            sibling_canary = os.path.join(tmp, "ground-truth.txt")
            Path(sibling_canary).write_text("secret")
            unique = "{}-{}".format(os.getpid(), isolation_policy)
            home_canary = os.path.join(
                os.path.expanduser("~"), ".cape-home-canary-" + unique
            )
            root_canary = os.path.join(
                "/root", ".cape-root-canary-" + unique
            )
            Path(home_canary).write_text("secret")
            Path(root_canary).write_text("secret")
            marker_token = "cape-foreign-marker-" + unique
            Path(workspace, "marker-token.txt").write_text(marker_token)
            probe_source = r'''import errno
import os
import sys

for path in sys.argv[1:4]:
    try:
        os.stat(path)
    except OSError as exc:
        assert exc.errno == errno.ENOENT, (path, exc.errno)
    else:
        raise AssertionError("host canary is visible: " + path)

marker_pid = sys.argv[4]
assert not os.path.exists("/proc/" + marker_pid)
token = open("/workspace/marker-token.txt").read().encode()
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    try:
        cmdline = open("/proc/" + name + "/cmdline", "rb").read()
    except OSError:
        continue
    assert token not in cmdline
assert open("/workspace/granted.txt").read() == "granted"
assert open("/cache/value").read() == "new-target"
assert os.environ["CAPE_EXPLICIT"] == "yes"
assert "CAPE_BWRAP_OUTER_SENTINEL" not in os.environ
print("confinement-ok")
'''
            Path(workspace, "confinement_probe.py").write_text(probe_source)
            marker = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(300)",
                    marker_token,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            adapter = BubbleWrapRuntimeAdapter(
                binary=binary,
                runtime_dir=tmp,
                node_gpu_uuids=(gpu_uuid,),
                stateful_sessions=True,
            )
            try:
                capabilities = adapter.probe(execute=True)
                self.assertIn(
                    isolation_policy,
                    capabilities.isolation_policies,
                    "Bubblewrap did not prove {}".format(isolation_policy),
                )
                self.assertIn(
                    isolation_policy,
                    capabilities.gpu_isolation_policies,
                    "Bubblewrap did not prove GPU access for {}: {}".format(
                        isolation_policy,
                        adapter.gpu_probe_failures.get(
                            isolation_policy, "no probe reason"
                        ),
                    ),
                )
                probe_python = adapter._probe_python_executable()
                self.assertIsNotNone(probe_python)
                execution_session_id = "real-{}-{}".format(
                    isolation_policy, os.getpid()
                )
                session = ContainerSessionSpec(
                    execution_session_id,
                    runtime_dir=tmp,
                    isolation_policy=isolation_policy,
                    state_layer_id=execution_session_id,
                    gpu_uuids=(gpu_uuid,),
                    mounts=(
                        MountSpec(workspace, "/workspace"),
                        MountSpec(cache_source, "/cache"),
                    ),
                    cwd="/workspace",
                )
                started = adapter.start_session(session)
                self.assertEqual(started.exit_code, 0, started.stderr)
                try:
                    visible = adapter.run_command(
                        execution_session_id,
                        (
                            adapter._nvidia_smi_binary(),
                            "--query-gpu=uuid",
                            "--format=csv,noheader",
                        ),
                        timeout=60,
                    )
                    self.assertEqual(visible.exit_code, 0, visible.stderr)
                    self.assertEqual(
                        tuple(
                            line.strip()
                            for line in visible.stdout.splitlines()
                            if line.strip()
                        ),
                        (gpu_uuid,),
                    )

                    minor = adapter._gpu_device_map()[gpu_uuid]
                    devices = adapter.run_command(
                        execution_session_id,
                        (
                            "sh",
                            "-c",
                            "for path in /dev/nvidia[0-9]*; do "
                            "test -e \"$path\" && basename \"$path\"; done",
                        ),
                        timeout=60,
                    )
                    self.assertEqual(devices.exit_code, 0, devices.stderr)
                    self.assertEqual(
                        devices.stdout.split(), ["nvidia{}".format(minor)]
                    )

                    cuda = adapter.run_command(
                        execution_session_id,
                        (
                            probe_python,
                            "-c",
                            "import torch; "
                            "assert torch.cuda.device_count() == 1; "
                            "x=torch.arange(32,device='cuda'); "
                            "assert int((x*x).sum().item()) == 10416; "
                            "print('cuda-kernel-ok')",
                        ),
                        timeout=120,
                    )
                    self.assertEqual(cuda.exit_code, 0, cuda.stderr)
                    self.assertIn("cuda-kernel-ok", cuda.stdout)

                    launch = adapter.build_run_command(
                        session, ContainerCommandSpec(("true",))
                    )
                    triples = [
                        launch.argv[index : index + 3]
                        for index in range(len(launch.argv) - 2)
                    ]
                    self.assertNotIn(["--bind", "/proc", "/proc"], triples)
                    self.assertNotIn(["--ro-bind", "/", "/"], triples)
                    self.assertNotIn(["--ro-bind", "/sys", "/sys"], triples)

                    proc_security = adapter.run_command(
                        execution_session_id,
                        (
                            "sh",
                            "-c",
                            "printf cape >/proc/self/comm && "
                            "awk '/^(CapEff|NoNewPrivs):/ "
                            "{print $1 $2}' /proc/self/status",
                        ),
                        timeout=60,
                    )
                    self.assertEqual(
                        proc_security.exit_code, 0, proc_security.stderr
                    )
                    if adapter.user_namespace_mode is False:
                        self.assertIn(
                            "CapEff:0000000000000000", proc_security.stdout
                        )
                        self.assertIn("NoNewPrivs:1", proc_security.stdout)

                    with mock.patch.dict(
                        os.environ,
                        {"CAPE_BWRAP_OUTER_SENTINEL": "must-not-cross"},
                    ):
                        confined = adapter.run_command(
                            execution_session_id,
                            (
                                probe_python,
                                "/workspace/confinement_probe.py",
                                sibling_canary,
                                home_canary,
                                root_canary,
                                str(marker.pid),
                            ),
                            env={"CAPE_EXPLICIT": "yes"},
                            timeout=60,
                        )
                    self.assertEqual(
                        confined.exit_code,
                        0,
                        confined.stderr + confined.stdout,
                    )
                    self.assertIn("confinement-ok", confined.stdout)

                    if isolation_policy == "host-stateful":
                        written = adapter.run_command(
                            execution_session_id,
                            (
                                "sh",
                                "-c",
                                "mkdir -p /opt/cape-real-gpu && "
                                "printf persisted > /opt/cape-real-gpu/value",
                            ),
                            timeout=60,
                        )
                        self.assertEqual(written.exit_code, 0, written.stderr)
                        persisted = adapter.run_command(
                            execution_session_id,
                            ("cat", "/opt/cape-real-gpu/value"),
                            timeout=60,
                        )
                        self.assertEqual(
                            persisted.exit_code, 0, persisted.stderr
                        )
                        self.assertEqual(persisted.stdout, "persisted")
                finally:
                    adapter.cleanup_session(execution_session_id)

                if isolation_policy == "host-stateful":
                    fresh_id = execution_session_id + "-fresh"
                    fresh = ContainerSessionSpec(
                        fresh_id,
                        runtime_dir=tmp,
                        isolation_policy=isolation_policy,
                        state_layer_id=fresh_id,
                    )
                    self.assertEqual(adapter.start_session(fresh).exit_code, 0)
                    try:
                        absent = adapter.run_command(
                            fresh_id,
                            (
                                "sh",
                                "-c",
                                "test ! -e /opt/cape-real-gpu/value",
                            ),
                            timeout=60,
                        )
                        self.assertEqual(absent.exit_code, 0, absent.stderr)
                    finally:
                        adapter.cleanup_session(fresh_id)
            finally:
                marker.terminate()
                try:
                    marker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    marker.kill()
                    marker.wait()
                for path in (home_canary, root_canary):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def test_real_bubblewrap_host_stateless_exposes_only_leased_gpu_and_runs_cuda(self):
        self._assert_real_bubblewrap_gpu_policy("host-stateless")

    def test_real_bubblewrap_host_stateful_exposes_only_leased_gpu_runs_cuda_and_persists(self):
        self._assert_real_bubblewrap_gpu_policy("host-stateful")


if __name__ == "__main__":
    unittest.main()
