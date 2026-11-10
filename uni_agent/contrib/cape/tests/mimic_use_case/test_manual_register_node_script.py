# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MANUAL_DIR = Path(__file__).resolve().parent / "manual"
REGISTER_NODE_SCRIPT = MANUAL_DIR / "register_node.sh"
MANUAL_README = MANUAL_DIR / "README.md"
PUBLIC_README = Path(__file__).resolve().parents[2] / "README.md"


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        pass


class ManualRegisterNodeScriptTests(unittest.TestCase):
    def test_default_evidence_is_beside_manual_script(self):
        script = REGISTER_NODE_SCRIPT.read_text()
        self.assertIn('CAPE_NODE_BUNDLE="${CAPE_NODE_BUNDLE:-$SCRIPT_DIR/evidence}"', script)
        makefile = (MANUAL_DIR / "Makefile").read_text()
        self.assertIn('CAPE_NODE_BUNDLE ?= $(abspath $(CUR_DIR)/evidence)', makefile)
        self.assertIn('export CAPE_NODE_BUNDLE', makefile)

    def test_explicit_regular_credential_enrolls_same_gpu_in_both_pools(self):
        credential = self.tmp_path / "regular.json"
        credential.write_text('{"secret":"regular-fixture"}')
        credential.chmod(0o600)
        node_ids = []
        for mode, slots in (("exclusive", "1"), ("shared", "4")):
            env = dict(self.env, CAPE_URL="http://controller.example:18892",
                       CAPE_POOL=mode + "-pool", CAPE_POOL_MODE=mode, CAPE_GPU_SLOTS=slots)
            env.pop("CAPE_NODE_ID")
            result = subprocess.run(
                ["bash", str(MANUAL_DIR / "credential_exec.sh"), str(credential),
                 "bash", str(REGISTER_NODE_SCRIPT)], env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            args = json.loads(self.args_file.read_text())
            self.assertEqual(args[:2], ["node", "enroll"])
            self.assertEqual(args[args.index("--pool") + 1], mode + "-pool")
            self.assertEqual(args[args.index("--gpu-uuid") + 1], "GPU-local-node")
            node_ids.append(args[args.index("--node-id") + 1])
            self.assertNotIn("--join-token", args)
        self.assertNotEqual(*node_ids)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

        self.node_repo = self.tmp_path / "node-host" / "repo"
        self.node_project = self.node_repo / "cape"
        self.node_project.mkdir(parents=True)
        (self.node_project / "pyproject.toml").write_text(
            "[project]\nname='fake-cape'\nversion='0'\n",
            encoding="utf-8",
        )

        self.node_venv = self.tmp_path / "node-host" / "venv" / "x86_64-python3.12"
        (self.node_venv / "bin").mkdir(parents=True)
        fake_python = self.node_venv / "bin" / "python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *sys.version_info* ]]; then\n"
            "  printf '3.12\\n'\n"
            "elif [[ \"$*\" == *hashlib* ]]; then\n"
            "  printf 'node-install-fingerprint\\n'\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        self.args_file = self.tmp_path / "node-host" / "cape-args.json"
        fake_cape = self.node_venv / "bin" / "pycape"
        fake_cape.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['FAKE_CAPE_ARGS'], 'w') as stream:\n"
            "    json.dump(sys.argv[1:], stream)\n",
            encoding="utf-8",
        )
        fake_nvidia_smi = self.tmp_path / "node-host" / "nvidia-smi"
        fake_nvidia_smi.write_text(
            "#!/usr/bin/env bash\nprintf 'GPU-local-node\\n'\n",
            encoding="utf-8",
        )
        for executable in (fake_python, fake_cape, fake_nvidia_smi):
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

        self.controller_env = self.tmp_path / "controller.env"
        self.controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export REPO_ROOT=/foreign/controller/repo\n"
            "export CAPE_PROJECT=/foreign/controller/repo/cape\n"
            "export CAPE_VENV=/foreign/controller/venv/aarch64-python3.12\n"
            "export CAPE_BIN=/foreign/controller/venv/aarch64-python3.12/bin/cape\n"
            "export PATH=/foreign/controller/venv/aarch64-python3.12/bin:$PATH\n"
            "export BUNDLE=/foreign/controller/results/run-a\n"
            "export CAPE_NODE_BIN=/foreign/controller/bin/cape\n"
            "export CAPE_NODE_BUNDLE=/foreign/controller/node-results\n"
            "export CAPE_RUNTIME_ADAPTER=apptainer\n"
            "export CAPE_APPTAINER_BIN=/foreign/controller/bin/apptainer\n"
            "export CAPE_NVIDIA_SMI=/foreign/controller/bin/nvidia-smi\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_NODE_TOKEN=join-token\n"
            "export CAPE_ADMIN_TOKEN=admin-token\n"
            "export RUN_ID=run-a\n"
            "export CASE=manual\n",
            encoding="utf-8",
        )

        self.node_bundle = self.tmp_path / "node-host" / "results" / "run-a"
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CAPE_")
            and key not in ("REPO_ROOT", "BUNDLE", "RUN_ID", "CASE")
        }
        self.env.update({
            "CAPE_NODE_REPO_ROOT": str(self.node_repo),
            "CAPE_NODE_PROJECT": str(self.node_project),
            "CAPE_NODE_BUNDLE": str(self.node_bundle),
            "CAPE_NODE_VENV_ROOT": str(self.node_venv.parent),
            "CAPE_NODE_VENV": str(self.node_venv),
            "CAPE_NODE_BIN": str(fake_cape),
            "CAPE_NVIDIA_SMI": str(fake_nvidia_smi),
            "CAPE_NODE_CPU_CORES": "1",
            "CAPE_NODE_MEMORY_GB": "1",
            "CAPE_NODE_ID": "node-local",
            "FAKE_CAPE_ARGS": str(self.args_file),
        })

    def test_handoff_cannot_select_controller_host_paths(self):
        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=self.env,
            check=True,
            text=True,
            capture_output=True,
        )
        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertEqual(arguments[:2], ["node", "enroll"])
        self.assertEqual(
            arguments[arguments.index("--controller-url") + 1],
            "http://controller.example:18892",
        )
        state_path = arguments[arguments.index("--state-path") + 1]
        self.assertTrue(state_path.startswith(str(self.node_bundle)))
        self.assertNotIn("/foreign/controller", "\n".join(arguments))
        self.assertNotIn("--runtime-adapter", arguments)
        for flag in ("--apptainer-probe-image", "--podman-probe-image"):
            self.assertEqual(arguments[arguments.index(flag) + 1], "docker://docker.io/library/ubuntu:24.04")
        self.assertIn("runtime_filter=auto:apptainer,bubblewrap,podman", result.stdout)
        self.assertNotIn("--allocation-deadline", arguments)
        self.assertIn(str(self.node_venv / "bin" / "pycape"), result.stdout)
        self.assertNotIn("/foreign/controller", result.stdout + result.stderr)

    def test_default_rscape_skips_node_venv_and_records_version(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        binary_dir = self.tmp_path / "rscape-bin"
        binary_dir.mkdir()
        rscape = binary_dir / "rscape"
        rscape.write_text(
            "#!/usr/bin/env bash\n"
            "if [ \"${1:-}\" = --version ]; then "
            "printf 'rscape 0.1.0\\n'; exit 0; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        rscape.chmod(rscape.stat().st_mode | stat.S_IXUSR)
        controller_env = self.tmp_path / "default-rscape-controller.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://127.0.0.1:{}\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_NODE_TOKEN=join-token\n"
            "export CAPE_ADMIN_TOKEN=admin-token\n"
            "export RUN_ID=default-rscape\n"
            "export CASE=manual\n".format(server.server_address[1]),
            encoding="utf-8",
        )
        venv_root = self.tmp_path / "must-not-create-node-venv"
        bundle = self.tmp_path / "default-rscape-node-evidence"
        env = dict(self.env)
        env.pop("CAPE_NODE_BIN")
        env.update({
            "PATH": str(binary_dir) + os.pathsep + env.get("PATH", ""),
            "CAPE_NODE_VENV_ROOT": str(venv_root),
            "CAPE_NODE_VENV": str(venv_root / "node"),
            "CAPE_NODE_BUNDLE": str(bundle),
        })

        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(controller_env)],
            env=env, check=True, text=True, capture_output=True,
        )

        self.assertFalse(venv_root.exists())
        evidence = bundle / "pool" / "node-local" / "implementation.txt"
        self.assertEqual(
            evidence.read_text(encoding="utf-8").splitlines(),
            [
                "implementation=rscape",
                "version=0.1.0",
                "executable=" + str(rscape),
            ],
        )
        self.assertIn("cape_implementation=rscape", result.stderr)

    def test_forwards_current_allocation_and_cache_registration_fields(self):
        env = dict(
            self.env,
            CAPE_ALLOCATION_DEADLINE="2000000000",
            CAPE_NODE_DRAIN_BEFORE_SEC="180",
            CAPE_NODE_CLEANUP_BUFFER_SEC="45",
            CAPE_NODE_RUNTIME_CACHE_CONFIG="local-process-x86",
            CAPE_NODE_CACHED_IMAGES="docker://example/a:latest\n/shared/image-b.sif",
            CAPE_NO_WARM_SESSIONS="1",
            CAPE_NODE_LOG_REDACT_ENV_KEYS="PRIVATE_TOKEN,LICENSE_KEY PRIVATE_TOKEN",
        )
        subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertEqual(
            arguments[arguments.index("--allocation-deadline") + 1],
            "2000000000",
        )
        self.assertEqual(
            arguments[arguments.index("--drain-before-sec") + 1], "180"
        )
        self.assertEqual(
            arguments[arguments.index("--cleanup-buffer-sec") + 1], "45"
        )
        self.assertEqual(
            arguments[arguments.index("--runtime-cache-config") + 1],
            "local-process-x86",
        )
        self.assertEqual(
            [
                arguments[index + 1]
                for index, value in enumerate(arguments)
                if value == "--cached-image"
            ],
            ["docker://example/a:latest", "/shared/image-b.sif"],
        )
        self.assertIn("--no-warm-sessions", arguments)
        self.assertEqual(
            [
                arguments[index + 1]
                for index, value in enumerate(arguments)
                if value == "--log-redact-env-key"
            ],
            ["PRIVATE_TOKEN", "LICENSE_KEY"],
        )

    def test_forwards_join_declared_pool_mode_slots_and_placement(self):
        env = dict(
            self.env,
            CAPE_POOL_MODE="shared",
            CAPE_GPU_SLOTS="4",
            CAPE_POOL_PLACEMENT="spread",
        )
        subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertEqual(
            arguments[arguments.index("--pool-mode") + 1], "shared"
        )
        self.assertEqual(
            arguments[arguments.index("--gpu-slots") + 1], "4"
        )
        self.assertEqual(
            arguments[arguments.index("--pool-placement") + 1], "spread"
        )

    def test_shared_pool_requires_explicit_slots_but_explicit_k1_is_legal(self):
        missing = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(self.env, CAPE_POOL_MODE="shared"),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn(
            "shared pools require explicit CAPE_GPU_SLOTS",
            missing.stderr,
        )
        self.assertFalse(self.args_file.exists())

        explicit = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(
                self.env,
                CAPE_POOL_MODE="shared",
                CAPE_GPU_SLOTS="1",
            ),
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertEqual(
            arguments[arguments.index("--gpu-slots") + 1], "1"
        )

    def test_default_node_id_is_pool_qualified_and_explicit_override_wins(self):
        default_env = dict(self.env)
        default_env.pop("CAPE_NODE_ID")
        subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=default_env,
            check=True,
            text=True,
            capture_output=True,
        )
        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        default_id = arguments[arguments.index("--node-id") + 1]
        self.assertTrue(default_id.startswith("pool-a-"), default_id)

        subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(self.env, CAPE_NODE_ID="explicit-node"),
            check=True,
            text=True,
            capture_output=True,
        )
        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertEqual(
            arguments[arguments.index("--node-id") + 1], "explicit-node"
        )

    def test_default_allocation_id_is_fresh_for_every_invocation(self):
        allocation_ids = []
        for _ in range(2):
            subprocess.run(
                [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
                env=self.env,
                check=True,
                text=True,
                capture_output=True,
            )
            arguments = json.loads(
                self.args_file.read_text(encoding="utf-8")
            )
            allocation_ids.append(
                arguments[arguments.index("--allocation-id") + 1]
            )

        self.assertNotEqual(allocation_ids[0], allocation_ids[1])
        for allocation_id in allocation_ids:
            self.assertLessEqual(len(allocation_id), 64)
            self.assertRegex(
                allocation_id,
                r"^manual-run-a-node-local-[A-Za-z0-9_.-]+$",
            )

    def test_generated_ids_fit_identity_limits_with_long_run_name(self):
        env = dict(self.env, CAPE_URL="http://controller.example:18892", CAPE_TOKEN="fixture",
                   CAPE_POOL="pool-" + "a" * 59, RUN_ID="r" * 80)
        env.pop("CAPE_NODE_ID")
        result = subprocess.run([str(REGISTER_NODE_SCRIPT)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        arguments = json.loads(self.args_file.read_text())
        for option in ("--node-id", "--allocation-id"):
            value = arguments[arguments.index(option) + 1]
            self.assertRegex(value, r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

    def test_explicit_allocation_id_overrides_remain_verbatim(self):
        cases = (
            (
                {"ALLOCATION_ID": "operator-allocation.alias"},
                "operator-allocation.alias",
            ),
            (
                {
                    "ALLOCATION_ID": "ignored-alias",
                    "CAPE_ALLOCATION_ID": "cape-managed_allocation",
                },
                "cape-managed_allocation",
            ),
        )
        for extra_env, expected in cases:
            with self.subTest(extra_env=extra_env):
                observed = []
                for _ in range(2):
                    subprocess.run(
                        [
                            str(REGISTER_NODE_SCRIPT),
                            str(self.controller_env),
                        ],
                        env=dict(self.env, **extra_env),
                        check=True,
                        text=True,
                        capture_output=True,
                    )
                    arguments = json.loads(
                        self.args_file.read_text(encoding="utf-8")
                    )
                    observed.append(
                        arguments[arguments.index("--allocation-id") + 1]
                    )
                self.assertEqual(observed, [expected, expected])

    def test_chosen_allocation_id_is_written_to_the_agent_log(self):
        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(
                self.env,
                CAPE_ALLOCATION_ID="operator-visible-allocation",
            ),
            check=True,
            text=True,
            capture_output=True,
        )

        agent_log = (
            self.node_bundle
            / "pool"
            / "node-local"
            / "agent.log"
        )
        self.assertIn("allocation=operator-visible-allocation", result.stdout)
        self.assertIn(
            "allocation=operator-visible-allocation",
            agent_log.read_text(encoding="utf-8"),
        )

    def test_self_exit_is_default_and_stay_after_removal_is_explicit(self):
        subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=self.env,
            check=True,
            text=True,
            capture_output=True,
        )
        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertNotIn("--stay-after-removal", arguments)

        subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(self.env, CAPE_STAY_AFTER_REMOVAL="1"),
            check=True,
            text=True,
            capture_output=True,
        )
        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertEqual(arguments.count("--stay-after-removal"), 1)

    def test_rejects_invalid_stay_after_removal_boolean(self):
        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(self.env, CAPE_STAY_AFTER_REMOVAL="sometimes"),
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("CAPE_STAY_AFTER_REMOVAL", result.stderr)
        self.assertFalse(self.args_file.exists())

    def test_help_documents_default_self_exit_and_opt_out(self):
        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), "--help"],
            env=self.env,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("CAPE_STAY_AFTER_REMOVAL", result.stdout)
        self.assertIn("exits", result.stdout)
        self.assertIn("authoritative", result.stdout)
        self.assertIn("fresh", result.stdout)
        self.assertIn("per invocation", result.stdout)
        self.assertIn("ALLOCATION_ID", result.stdout)
        self.assertIn("explicit", result.stdout)
        self.assertIn("belongs to the administrator", result.stdout)
        self.assertIn("admin pool edit", result.stdout)

    def test_readmes_document_fresh_allocation_incarnations(self):
        for readme_path in (MANUAL_README, PUBLIC_README):
            with self.subTest(readme=readme_path):
                readme = readme_path.read_text(encoding="utf-8").lower()
                self.assertIn("fresh unique allocation id", readme)
                self.assertIn("per invocation", readme)
                self.assertIn("cape_allocation_id", readme)
                self.assertIn("authoritative shutdown", readme)

    def test_quick_example_drain_does_not_wait_for_cache_expiry(self):
        readme = MANUAL_README.read_text(encoding="utf-8")
        quick_example = readme.split("## Quick Example", 1)[1].split(
            "## Deployment Model", 1
        )[0]

        self.assertIn("retire_node.sh", quick_example)
        self.assertIn("--timeout 300", quick_example)
        self.assertIn("cached", quick_example)
        self.assertIn("stops", quick_example)
        self.assertNotIn("--timeout 3700", quick_example)
        self.assertIn("never cancels a", quick_example)
        self.assertIn("exits", quick_example)
        self.assertNotIn("stop register_node.sh with Ctrl-C", quick_example)
        self.assertIn("CAPE_STAY_AFTER_REMOVAL=1", readme)

    def test_readme_separates_saturated_handoff_from_heartbeat_liveness(self):
        readme = MANUAL_README.read_text(encoding="utf-8")

        self.assertIn("CAPE_HEARTBEAT_INTERVAL", readme)
        self.assertIn("completion-report response", readme)
        self.assertIn("saturated handoff latency", readme)
        self.assertIn("liveness", readme)
        self.assertIn("mixed-version", readme)

    def test_readmes_document_shared_dual_membership_and_pool_only_routing(self):
        manual = MANUAL_README.read_text(encoding="utf-8")
        for text in (
            "CAPE_ADDITIONAL_POOLS",
            "controller-shared-pool.env",
            "CAPE_POOL_MODE=exclusive",
            "CAPE_POOL_MODE=shared",
            "CAPE_GPU_SLOTS=4",
            "same physical GPU UUID",
            "advisory",
            "gpu_cotenant_count",
            "retire_node.sh",
        ):
            with self.subTest(manual_text=text):
                self.assertIn(text, manual)

        public = PUBLIC_README.read_text(encoding="utf-8")
        run_section = public.split(
            "### `rscape run` — submit a workload (async)", 1
        )[1].split("### `rscape status", 1)[0]
        self.assertNotIn("--gpu-mode", run_section)
        for text in (
            "--pool-mode",
            "--gpu-slots",
            "--pool-placement",
            "--gpu-contention-policy",
            "barrier-ratio",
            "gpu_cotenant_count",
            "deduplicated physical",
        ):
            with self.subTest(public_text=text):
                self.assertIn(text, public)

    def test_manual_readme_requires_explicit_empty_pool_configuration(self):
        readme = MANUAL_README.read_text(encoding="utf-8")
        for text in (
            "reconfigurable-empty",
            "non-terminal requests",
            "cached",
            "pycape admin pool edit",
            "pool_config",
            "CAPE_GPU_SLOTS",
            "explicit",
        ):
            with self.subTest(text=text):
                self.assertIn(text, readme)

    def test_rejects_invalid_custom_log_redaction_key(self):
        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(
                self.env,
                CAPE_NODE_LOG_REDACT_ENV_KEYS="VALID not-a-variable",
            ),
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("environment-variable names", result.stderr)
        self.assertFalse(self.args_file.exists())

    def test_repeatable_runtime_filters_and_adapter_local_paths_are_forwarded(self):
        runtime_root = self.tmp_path / "node-host" / "runtime"
        env = dict(
            self.env,
            CAPE_RUNTIME_ADAPTERS="apptainer,bubblewrap podman",
            CAPE_APPTAINER_BIN="/bin/true",
            CAPE_BUBBLEWRAP_BIN="/bin/true",
            CAPE_BUBBLEWRAP_SYSTEM_PATHS="/usr,/etc\n/opt",
            CAPE_BUBBLEWRAP_READ_ONLY_PATHS="/sys/devices/system/cpu,/sys/cape-absent",
            CAPE_PODMAN_BIN="/bin/true",
            CAPE_RUNTIME_ROOT=str(runtime_root),
            CAPE_APPTAINER_PROBE_IMAGE="docker://example/apptainer-probe:latest",
            CAPE_PODMAN_PROBE_IMAGE="docker://example/podman-probe:latest",
        )
        subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        arguments = json.loads(self.args_file.read_text(encoding="utf-8"))
        self.assertEqual(
            [
                arguments[index + 1]
                for index, value in enumerate(arguments)
                if value == "--runtime-adapter"
            ],
            ["apptainer", "bubblewrap", "podman"],
        )
        expected_options = {
            "--apptainer-cache": runtime_root / "apptainer" / "cache",
            "--apptainer-tmp": runtime_root / "apptainer" / "tmp",
            "--apptainer-runtime-dir": runtime_root / "apptainer" / "runtime",
            "--bubblewrap-cache": runtime_root / "bubblewrap" / "cache",
            "--bubblewrap-tmp": runtime_root / "bubblewrap" / "tmp",
            "--bubblewrap-runtime-dir": runtime_root / "bubblewrap" / "runtime",
            "--podman-graphroot": runtime_root / "podman" / "graphroot",
            "--podman-tmpdir": runtime_root / "podman" / "tmp",
            "--podman-runroot": runtime_root / "podman" / "runroot",
        }
        self.assertEqual(
            [arguments[index + 1] for index, value in enumerate(arguments)
             if value == "--bubblewrap-read-only-path"],
            ["/sys/devices/system/cpu", "/sys/cape-absent"],
        )
        for option, expected in expected_options.items():
            self.assertEqual(arguments[arguments.index(option) + 1], str(expected))
        self.assertEqual(
            [
                arguments[index + 1]
                for index, value in enumerate(arguments)
                if value == "--bubblewrap-system-path"
            ],
            ["/usr", "/etc", "/opt"],
        )
        self.assertEqual(
            arguments[arguments.index("--apptainer-probe-image") + 1],
            "docker://example/apptainer-probe:latest",
        )
        self.assertEqual(
            arguments[arguments.index("--podman-probe-image") + 1],
            "docker://example/podman-probe:latest",
        )

    def test_removed_singular_runtime_environment_is_rejected(self):
        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
            env=dict(self.env, CAPE_RUNTIME_ADAPTER="apptainer"),
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("CAPE_RUNTIME_ADAPTERS", result.stderr)

    def test_bubblewrap_system_paths_reject_unconfined_roots_before_join(self):
        for path in ("/", "/proc", "/sys", "/dev", "/run"):
            with self.subTest(path=path):
                result = subprocess.run(
                    [str(REGISTER_NODE_SCRIPT), str(self.controller_env)],
                    env=dict(self.env, CAPE_BUBBLEWRAP_SYSTEM_PATHS=path),
                    check=False,
                    text=True,
                    capture_output=True,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn("must not expose", result.stderr)
                self.assertFalse(self.args_file.exists())

    def test_passed_handoff_clears_stale_portable_values_it_omits(self):
        incomplete_env = self.tmp_path / "controller-incomplete.env"
        incomplete_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_NODE_TOKEN=join-token\n"
            "export RUN_ID=run-a\n"
            "export CASE=manual\n",
            encoding="utf-8",
        )
        env = dict(
            self.env,
            CAPE_URL="http://stale-controller.example:18892",
            CAPE_NODE_TOKEN="stale-node-token",
        )
        result = subprocess.run(
            [str(REGISTER_NODE_SCRIPT), str(incomplete_env)],
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("CAPE_URL is required", result.stderr)
        self.assertFalse(self.args_file.exists())

    def test_old_node_script_name_is_not_retained(self):
        self.assertTrue(REGISTER_NODE_SCRIPT.is_file())
        self.assertFalse((MANUAL_DIR / "node.sh").exists())

    def test_retired_handoff_versions_fail_before_starting_an_agent(self):
        handoff = self.tmp_path / "retired.env"
        for version in ("1", "2", None):
            with self.subTest(version=version):
                handoff.write_text("" if version is None else "export CAPE_CONTROLLER_ENV_VERSION=" + version + "\n")
                result = subprocess.run([str(REGISTER_NODE_SCRIPT), str(handoff)], env=self.env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("unsupported CAPE_CONTROLLER_ENV_VERSION", result.stderr)
                self.assertFalse(self.args_file.exists())


if __name__ == "__main__":
    unittest.main()
