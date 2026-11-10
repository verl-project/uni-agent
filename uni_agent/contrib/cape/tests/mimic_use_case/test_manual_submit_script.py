# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


SUBMIT_SCRIPT = Path(__file__).resolve().parent / "manual" / "submit.sh"


class ManualSubmitScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.args_path = self.tmp_path / "cape-args.json"
        self.fake_cape = self.tmp_path / "cape"
        self.fake_cape.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys, time\n"
            "time.sleep(float(os.environ.get('FAKE_CAPE_SLEEP', '0')))\n"
            "with open(os.environ['FAKE_CAPE_ARGS'], 'w') as stream:\n"
            "    json.dump(sys.argv[1:], stream)\n"
            "print(os.environ.get('FAKE_CAPE_OUTPUT', 'req-submit-test'))\n",
            encoding="utf-8",
        )
        self.fake_cape.chmod(self.fake_cape.stat().st_mode | stat.S_IXUSR)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CAPE_")
            and key not in ("REPO_ROOT", "BUNDLE", "RUN_ID")
        }
        self.env.update({
            "CAPE_SUBMIT_BIN": str(self.fake_cape),
            "FAKE_CAPE_ARGS": str(self.args_path),
        })

    def run_submit(self, *arguments, env=None, check=True):
        return subprocess.run(
            [str(SUBMIT_SCRIPT), *arguments],
            env=env or self.env,
            check=check,
            text=True,
            capture_output=True,
        )

    def cape_args(self):
        return json.loads(self.args_path.read_text(encoding="utf-8"))

    def option_value(self, arguments, option):
        index = arguments.index(option)
        return arguments[index + 1]

    def test_resolve_client_prints_the_executable_and_submits_nothing(self):
        result = self.run_submit("--resolve-client")
        self.assertEqual(result.stdout.strip(), str(self.fake_cape))
        self.assertFalse(
            self.args_path.exists(),
            "resolve-client must not invoke cape run",
        )

    def test_submit_phase_file_accounts_for_the_binary_round_trip(self):
        timing = self.tmp_path / "evidence" / "submit-timing.env"
        env = dict(
            self.env,
            CAPE_SUBMIT_TIMING_FILE=str(timing),
            FAKE_CAPE_SLEEP="0.05",
            CAPE_ADMIN_TOKEN="must-not-appear",
        )
        result = self.run_submit(
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a", "--user", "alice", "--token", "secret-token",
            "--gpus", "0", "--", "true", env=env,
        )
        self.assertEqual(result.returncode, 0)
        fields = dict(
            line.split("=", 1)
            for line in timing.read_text(encoding="utf-8").splitlines()
        )
        stamps = [
            float(fields[name]) for name in (
                "entry", "handoff_loaded", "client_ready", "pre_binary",
                "binary_return", "exit",
            )
        ]
        self.assertEqual(stamps, sorted(stamps))
        self.assertGreaterEqual(
            float(fields["binary_return"]) - float(fields["pre_binary"]),
            0.04,
        )
        self.assertEqual(fields["exit_code"], "0")
        contents = timing.read_text(encoding="utf-8")
        self.assertNotIn("secret-token", contents)
        self.assertNotIn("must-not-appear", contents)

    def test_default_rscape_resolves_without_submit_venv(self):
        binary_dir = self.tmp_path / "rscape-bin"
        binary_dir.mkdir()
        rscape = binary_dir / "rscape"
        rscape.write_text(
            "#!/usr/bin/env bash\n"
            "[ \"${1:-}\" = --version ] && "
            "{ printf 'rscape 0.1.0\\n'; exit 0; }\n"
            "exit 2\n",
            encoding="utf-8",
        )
        rscape.chmod(rscape.stat().st_mode | stat.S_IXUSR)
        venv_root = self.tmp_path / "must-not-create-submit-venv"
        bundle = self.tmp_path / "default-rscape-evidence"
        env = dict(self.env)
        env.pop("CAPE_SUBMIT_BIN")
        env.update({
            "PATH": str(binary_dir) + os.pathsep + env.get("PATH", ""),
            "CAPE_SUBMIT_VENV_ROOT": str(venv_root),
            "CAPE_SUBMIT_VENV": str(venv_root / "submit"),
            "CAPE_SUBMIT_BUNDLE": str(bundle),
        })

        result = self.run_submit("--resolve-client", env=env)

        self.assertEqual(result.stdout.strip(), str(rscape))
        self.assertFalse(venv_root.exists())
        self.assertEqual(
            (bundle / "submit-client-implementation.txt")
            .read_text(encoding="utf-8").splitlines(),
            [
                "implementation=rscape",
                "version=0.1.0",
                "executable=" + str(rscape),
            ],
        )

    def test_forwards_generic_run_surface_and_records_submission(self):
        record = self.tmp_path / "evidence" / "submits.csv"
        result = self.run_submit(
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "alice",
            "--token", "alice-token",
            "--session-id", "alice-task",
            "--image", "docker://example/train:latest",
            "--runtime-adapter", "apptainer",
            "--gpus", "2",
            "--cpu-cores", "4",
            "--memory-gb", "16",
            "--max-duration-seconds", "120.5",
            "--command-timeout-seconds", "45",
            "--hard-gpu-enforcement",
            "--port", "8080",
            "--port", "9000",
            "--locality", "rack-a",
            "--isolation-policy", "image-stateful",
            "--bind", "/shared/alice:/workspace:rw",
            "--bind", "/datasets:/datasets:ro",
            "--env", "RUN_NAME=alice-a",
            "--cwd", "/workspace",
            "--log-path", "/output/runtime.log",
            "--artifact-path", "/output/model.pt",
            "--record", str(record),
            "--label", "train",
            "--", "python", "train.py", "--epochs", "2",
        )

        self.assertEqual(result.stdout, "req-submit-test\n")
        self.assertIn("request=req-submit-test", result.stderr)
        arguments = self.cape_args()
        self.assertEqual(arguments[0], "run")
        self.assertEqual(self.option_value(arguments, "--controller-url"), "http://controller.example:18892")
        self.assertEqual(self.option_value(arguments, "--pool"), "pool-a")
        self.assertEqual(self.option_value(arguments, "--user"), "alice")
        self.assertEqual(self.option_value(arguments, "--token"), "alice-token")
        self.assertEqual(self.option_value(arguments, "--session-id"), "alice-task")
        self.assertNotIn("--workspace", arguments)
        self.assertEqual(self.option_value(arguments, "--gpus"), "2")
        self.assertEqual(self.option_value(arguments, "--cpu-cores"), "4")
        self.assertEqual(self.option_value(arguments, "--memory-gb"), "16")
        self.assertEqual(
            self.option_value(arguments, "--command-timeout-seconds"),
            "45",
        )
        self.assertIn("--hard-gpu-enforcement", arguments)
        self.assertEqual(
            [arguments[index + 1] for index, value in enumerate(arguments) if value == "--port"],
            ["8080", "9000"],
        )
        separator = arguments.index("--")
        self.assertEqual(arguments[separator + 1:], ["python", "train.py", "--epochs", "2"])
        self.assertNotIn("--wait", arguments)
        self.assertNotIn("--gpu-mode", arguments)

        rows = record.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows[0], "uid,request,jobid,t_start,t_return,latency,session_id")
        fields = rows[1].split(",")
        self.assertEqual(fields[:3], ["alice", "train", "req-submit-test"])
        self.assertEqual(fields[-1], "alice-task")
        self.assertGreaterEqual(float(fields[5]), 0.0)

    def test_selects_user_token_from_controller_environment(self):
        controller_env = self.tmp_path / "controller.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export REPO_ROOT=/foreign/controller/repo\n"
            "export CAPE_PROJECT=/foreign/controller/repo/cape\n"
            "export CAPE_VENV=/foreign/controller/venv/aarch64-python3.12\n"
            "export CAPE_BIN=/foreign/controller/venv/aarch64-python3.12/bin/cape\n"
            "export PATH=/foreign/controller/venv/aarch64-python3.12/bin:$PATH\n"
            "export CAPE_SUBMIT_BIN=/foreign/controller/bin/cape\n"
            "export CAPE_SUBMIT_BUNDLE=/foreign/controller/submit-results\n"
            "export CAPE_SUBMIT_PYTHON_312=/foreign/controller/bin/python3.12\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-from-env\n"
            "export CAPE_IMAGE=/foreign/controller/images/aarch64-only.sif\n"
            "export CAPE_USER_TOKENS=/foreign/controller/user_tokens.csv\n"
            "export CAPE_USER_TOKENS_INLINE=$'alice,alice-token\\nbob,bob-token'\n",
            encoding="utf-8",
        )

        self.run_submit(
            "--controller-env", str(controller_env),
            "--user", "bob",
            "--", "echo", "hello",
        )
        arguments = self.cape_args()
        self.assertEqual(self.option_value(arguments, "--token"), "bob-token")
        self.assertEqual(self.option_value(arguments, "--pool"), "pool-from-env")
        self.assertNotIn("--image", arguments)
        self.assertNotIn("/foreign/controller", "\n".join(arguments))

    def test_suspend_on_failure_is_explicit_opt_out(self):
        common = (
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "alice",
        )
        self.run_submit(*common, "--", "false")
        self.assertNotIn(
            "--suspend-on-failure", self.cape_args()
        )

        self.run_submit(
            *common,
            "--suspend-on-failure",
            "--",
            "false",
        )
        self.assertIn(
            "--suspend-on-failure", self.cape_args()
        )

    def test_accepts_bubblewrap_and_rejects_invalid_current_run_contracts(self):
        common = (
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "alice",
        )
        result = self.run_submit(
            *common,
            "--runtime-adapter", "bubblewrap",
            "--isolation-policy", "host-stateless",
            "--", "true",
        )
        self.assertEqual(result.stdout, "req-submit-test\n")
        self.assertEqual(
            self.option_value(self.cape_args(), "--runtime-adapter"), "bubblewrap"
        )

        invalid_policy = self.run_submit(
            *common,
            "--isolation-policy", "strict",
            "--", "true",
            check=False,
        )
        self.assertEqual(invalid_policy.returncode, 2)
        self.assertIn("isolation-policy", invalid_policy.stderr)

        mismatch = self.run_submit(
            *common,
            "--runtime-adapter", "apptainer",
            "--isolation-policy", "host-stateful",
            "--", "true",
            check=False,
        )
        self.assertEqual(mismatch.returncode, 2)
        self.assertIn("cannot satisfy", mismatch.stderr)

    def test_rejects_removed_gpu_mode_option_in_both_forms(self):
        common = (
            "--controller-url", "http://controller.example:18892",
            "--pool", "shared-pool",
            "--user", "alice",
        )
        for option in ("--gpu-mode", "--gpu-mode=whole"):
            with self.subTest(option=option):
                arguments = list(common) + [option]
                if option == "--gpu-mode":
                    arguments.append("whole")
                arguments.extend(["--", "true"])
                result = self.run_submit(*arguments, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("unknown option", result.stderr)
                self.assertFalse(self.args_path.exists())

        image_and_profile = self.run_submit(
            *common,
            "--image", "docker://example/image:latest",
            "--profile", "profile-a",
            "--", "true",
            check=False,
        )
        self.assertEqual(image_and_profile.returncode, 2)
        self.assertIn("mutually exclusive", image_and_profile.stderr)

    def test_passed_handoff_does_not_reuse_an_omitted_stale_admin_token(self):
        controller_env = self.tmp_path / "controller-without-token.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n",
            encoding="utf-8",
        )
        env = dict(self.env, CAPE_ADMIN_TOKEN="stale-admin-token")
        self.run_submit(
            "--controller-env", str(controller_env),
            "--user", "alice",
            "--", "true",
            env=env,
        )
        self.assertNotIn("--token", self.cape_args())

    def test_mimic_uses_only_the_submit_role_bundle(self):
        controller_env = self.tmp_path / "legacy-controller.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-from-env\n"
            "export BUNDLE=/foreign/controller/results/run-a\n",
            encoding="utf-8",
        )
        submit_bundle = self.tmp_path / "submit-role" / "results" / "run-a"
        env = dict(self.env, CAPE_SUBMIT_BUNDLE=str(submit_bundle))

        self.run_submit(
            "--controller-env", str(controller_env),
            "--user", "alice",
            "--token", "alice-token",
            "--mimic", "A",
            env=env,
        )
        arguments = self.cape_args()
        binds = [
            arguments[index + 1]
            for index, value in enumerate(arguments)
            if value == "--bind"
        ]
        self.assertIn(
            str(submit_bundle / "users" / "alice" / "workspace")
            + ":/workspace:rw",
            binds,
        )
        self.assertTrue(all("/foreign/controller" not in bind for bind in binds))

    def test_can_force_unauthenticated_submission(self):
        env = dict(self.env, CAPE_ADMIN_TOKEN="admin-token")
        self.run_submit(
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "alice",
            "--no-token",
            "--", "true",
            env=env,
        )
        self.assertNotIn("--token", self.cape_args())

    def test_builds_isolated_mimic_a_and_b_requests(self):
        user_root = self.tmp_path / "shared" / "users" / "user-000"
        base = (
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "user-000",
            "--token", "user-token",
            "--user-root", str(user_root),
        )
        env = dict(self.env, RUN_ID="run-test")

        first = self.run_submit(*base, "--mimic", "A", env=env)
        self.assertEqual(first.stdout, "req-submit-test\n")
        arguments = self.cape_args()
        binds = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--bind"]
        self.assertIn(str(user_root / "workspace") + ":/workspace:rw", binds)
        self.assertIn(str(user_root / "output") + ":/output:rw", binds)
        self.assertTrue(any(value.endswith(":/cape-mimic:ro") for value in binds))
        self.assertNotIn("--runtime-adapter", arguments)
        self.assertEqual(self.option_value(arguments, "--workspace-target"), "/cape-workspace")
        self.assertEqual(self.option_value(arguments, "--session-id"), "user-000-task")
        self.assertIn("CAPE_RUN_TAG=user-000@run-test", arguments)
        separator = arguments.index("--")
        self.assertEqual(arguments[separator + 1:], ["python", "/cape-mimic/gemm_silu.py"])
        self.assertTrue((user_root / "workspace").is_dir())
        self.assertTrue((user_root / "output").is_dir())

        second = self.run_submit(*base, "--mimic", "B", env=env)
        self.assertEqual(second.stdout, "req-submit-test\n")
        arguments = self.cape_args()
        separator = arguments.index("--")
        self.assertEqual(arguments[separator + 1:separator + 3], ["bash", "-lc"])
        self.assertIn("cutedsl_add.py", arguments[separator + 3])

    def test_managed_workspace_flags_preserve_explicit_cwd_and_command(self):
        identity = "ws-" + "a" * 32
        self.run_submit("--controller-url", "http://controller.example:18892", "--pool", "pool-a", "--user", "alice",
                        "--workspace-id", identity, "--workspace-target", "/managed", "--cwd", "/source", "--", "echo", "literal")
        arguments = self.cape_args()
        self.assertEqual(self.option_value(arguments, "--workspace-id"), identity)
        self.assertEqual(self.option_value(arguments, "--workspace-target"), "/managed")
        self.assertEqual(self.option_value(arguments, "--cwd"), "/source")
        self.assertEqual(arguments[arguments.index("--") + 1:], ["echo", "literal"])

    def test_rejects_wait_and_malformed_cape_output(self):
        common = (
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "alice",
        )
        result = self.run_submit(*common, "--wait", "--", "true", check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("use status.sh wait", result.stderr)

        env = dict(self.env, FAKE_CAPE_OUTPUT="not-a-request")
        result = self.run_submit(*common, "--", "true", env=env, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unexpected CAPE run output", result.stderr)

    def test_removed_workspace_identity_option_is_rejected(self):
        result = self.run_submit(
            "--controller-url", "http://controller.example:18892",
            "--pool", "pool-a",
            "--user", "alice",
            "--workspace", "old-name",
            "--", "true",
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("--session-id", result.stderr)


if __name__ == "__main__":
    unittest.main()
