# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import stat
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path


CONTROLLER_SCRIPT = (
    Path(__file__).resolve().parent / "manual" / "controller.sh"
)


class ManualControllerScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)

        self.repo_root = self.tmp_path / "controller-host" / "repo"
        self.cape_project = self.repo_root / "cape"
        self.cape_project.mkdir(parents=True)
        (self.cape_project / "pyproject.toml").write_text(
            "[project]\nname='fake-cape'\nversion='0'\n",
            encoding="utf-8",
        )

        self.venv = self.tmp_path / "controller-host" / "venv" / "aarch64-python3.12"
        (self.venv / "bin").mkdir(parents=True)
        fake_python = self.venv / "bin" / "python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "if [ \"${1:-}\" = -m ] && [ \"${2:-}\" = pip ] && "
            "[ \"${3:-}\" = install ] && "
            "[ -n \"${FAKE_PYTHON_CALLS:-}\" ]; then printf 'pip-install\\n' >> \"$FAKE_PYTHON_CALLS\"; fi\n"
            "if [[ \"$*\" == *sys.version_info* ]]; then printf '3.12\\n'; fi\n"
            "if [[ \"$*\" == *hashlib* ]]; then printf 'fake-install-fingerprint\\n'; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )

        self.fake_cape = self.venv / "bin" / "pycape"
        self.fake_cape.write_text("#!/usr/bin/env python3\n" + (Path(__file__).with_name("controller_cli_double.py")).read_text())
        for executable in (fake_python, self.fake_cape):
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

        self.bundle = self.tmp_path / "controller-host" / "results" / "run-a"
        # Default per-pool handoff naming: controller-<pool>.env.
        self.controller_env = self.bundle / "pool" / "controller-exclusive-pool.env"
        self.calls_file = self.tmp_path / "controller-host" / "cape-calls.txt"
        self.python_calls_file = (
            self.tmp_path / "controller-host" / "python-calls.txt"
        )
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CAPE_")
            and key not in ("REPO_ROOT", "BUNDLE", "RUN_ID", "CASE")
        }
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.env.update({
            "REPO_ROOT": str(self.repo_root),
            "CAPE_PROJECT": str(self.cape_project),
            "CAPE_VENV_ROOT": str(self.venv.parent),
            "CAPE_VENV": str(self.venv),
            "CAPE_BIN": str(self.fake_cape),
            "BUNDLE": str(self.bundle),
            "RUN_ID": "run-a",
            "CASE": "manual",
            "CONTROLLER_HOST": "controller.example",
            "CAPE_USER_COUNT": "2",
            "CAPE_PORT": str(port),
            "FAKE_CAPE_CALLS": str(self.calls_file),
            "FAKE_PYTHON_CALLS": str(self.python_calls_file),
        })

    def test_controller_creates_only_admin_and_no_handoff(self):
        result = subprocess.run([str(CONTROLLER_SCRIPT)], env=self.env,
                                check=True, text=True, capture_output=True)
        self.assertIn("fake controller stopped", result.stdout)
        self.assertFalse(list(self.bundle.rglob("*.env")))
        secrets = list((self.bundle / "pool" / "secrets").glob("*.json"))
        self.assertEqual([p.name for p in secrets], ["admin.json"])
        self.assertEqual(stat.S_IMODE(secrets[0].stat().st_mode), 0o600)
        self.assertNotIn("user token create", self.calls_file.read_text())
        self.assertNotIn("fixture-cape-admin", result.stdout)

    def test_default_run_creates_exclusive_and_shared_pools(self):
        subprocess.run([str(CONTROLLER_SCRIPT)], env=self.env, check=True, capture_output=True)
        calls = self.calls_file.read_text()
        self.assertIn("admin pool create exclusive-pool ", calls)
        self.assertIn("--pool-mode exclusive --gpu-slots 1", calls)
        self.assertIn("admin pool create shared-pool ", calls)
        self.assertIn("--pool-mode shared --gpu-slots 4", calls)

    def test_explicit_empty_additional_pools_is_single_pool(self):
        subprocess.run([str(CONTROLLER_SCRIPT)],
                       env=dict(self.env, CAPE_ADDITIONAL_POOLS=""),
                       check=True, capture_output=True)
        calls = self.calls_file.read_text()
        self.assertIn("admin pool create exclusive-pool ", calls)
        self.assertNotIn("admin pool create shared-pool ", calls)

    def test_removed_join_token_restrictions_are_rejected(self):
        result = subprocess.run([str(CONTROLLER_SCRIPT)],
                                env=dict(self.env, CAPE_JOIN_ALLOWED_NODE_PREFIXES="gpu-node-"),
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Join Token restrictions are removed", result.stderr)

    def test_forwards_gpu_contention_policy_ratio_and_per_uuid_overrides(self):
        env = dict(
            self.env,
            CAPE_GPU_CONTENTION_POLICY="ratio-only",
            CAPE_GPU_SHARED_RATIO="7",
            CAPE_GPU_SHARE_OVERRIDES=(
                "GPU-a=exclusive-first:1\n"
                "GPU-b=barrier-ratio:3"
            ),
        )
        subprocess.run(
            [str(CONTROLLER_SCRIPT)],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )

        calls = self.calls_file.read_text(encoding="utf-8").splitlines()
        start_call = next(
            line for line in calls if line.startswith("controller start ")
        )
        self.assertIn(
            "--gpu-contention-policy ratio-only", start_call
        )
        self.assertIn("--gpu-shared-ratio 7", start_call)
        self.assertIn(
            "--gpu-share-override GPU-a=exclusive-first:1", start_call
        )
        self.assertIn(
            "--gpu-share-override GPU-b=barrier-ratio:3", start_call
        )

    def test_forwards_profiles_quotas_idle_and_reconciliation_policy(self):
        profiles = self.tmp_path / "profiles.json"
        profiles.write_text('{"profiles":{"gpu":{"gpu_required":true}}}\n')
        env = dict(
            self.env,
            CAPE_PROFILES_FILE=str(profiles),
            CAPE_MAX_ACTIVE_GPU_SESSIONS_PER_USER="2",
            CAPE_MAX_QUEUED_REQUESTS_PER_USER="5",
            CAPE_MAX_ACTIVE_SESSIONS_PER_POOL="8",
            CAPE_IDLE_SESSION_TIMEOUT_SECONDS="45.5",
            CAPE_RECONCILE_MISSING_GRACE_SECONDS="1.5",
            CAPE_RECONCILE_MISSING_HEARTBEATS="3",
        )
        subprocess.run(
            [str(CONTROLLER_SCRIPT)], env=env, check=True, text=True, capture_output=True
        )
        start_call = next(
            line for line in self.calls_file.read_text().splitlines()
            if line.startswith("controller start ")
        )
        for expected in (
            "--profiles-file " + str(profiles),
            "--max-active-gpu-sessions-per-user 2",
            "--max-queued-requests-per-user 5",
            "--max-active-sessions-per-pool 8",
            "--idle-session-timeout-seconds 45.5",
            "--reconcile-missing-grace-seconds 1.5",
            "--reconcile-missing-heartbeats 3",
        ):
            self.assertIn(expected, start_call)

    def test_additional_pools_do_not_create_credentials(self):
        subprocess.run([str(CONTROLLER_SCRIPT)],
                       env=dict(self.env, CAPE_ADDITIONAL_POOLS="shared-pool\nbatch.pool"),
                       check=True, capture_output=True)
        calls = self.calls_file.read_text()
        self.assertEqual(calls.count("admin pool create "), 3)
        self.assertNotIn("user token create", calls)
        self.assertFalse(list(self.bundle.rglob("*.env")))

    def test_additional_pool_names_must_be_unique(self):
        for additional in (
            "exclusive-pool",
            "batch-pool\nbatch-pool",
        ):
            with self.subTest(additional=additional):
                result = subprocess.run(
                    [str(CONTROLLER_SCRIPT)],
                    env=dict(
                        self.env,
                        CAPE_ADDITIONAL_POOLS=additional,
                    ),
                    check=False,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "CAPE_ADDITIONAL_POOLS contains duplicate pool",
                    result.stderr,
                )

    def test_reuses_an_unchanged_controller_installation(self):
        install_env = dict(self.env)
        install_env["CAPE_BIN"] = "pycape"
        first = subprocess.run(
            [str(CONTROLLER_SCRIPT)],
            env=install_env,
            check=True,
            text=True,
            capture_output=True,
        )
        second = subprocess.run(
            [str(CONTROLLER_SCRIPT)],
            env=dict(install_env, CAPE_REPLACE_EXISTING="1"),
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("installing CAPE", first.stdout)
        self.assertIn("reusing CAPE installation", second.stdout)
        self.assertEqual(
            self.python_calls_file.read_text(encoding="utf-8").splitlines(),
            ["pip-install"],
        )

    def test_default_rscape_skips_controller_venv_and_records_version(self):
        binary_dir = self.tmp_path / "rscape-bin"
        binary_dir.mkdir()
        rscape = binary_dir / "rscape"
        rscape.write_text(
            self.fake_cape.read_text(encoding="utf-8").replace(
                "pycape 0.1.0", "rscape 0.1.0"
            ),
            encoding="utf-8",
        )
        rscape.chmod(rscape.stat().st_mode | stat.S_IXUSR)
        venv_root = self.tmp_path / "must-not-create-controller-venv"
        bundle = self.tmp_path / "controller-host" / "results" / "default-rscape"
        env = dict(self.env)
        env.pop("CAPE_BIN")
        env.update({
            "PATH": str(binary_dir) + os.pathsep + env.get("PATH", ""),
            "CAPE_VENV_ROOT": str(venv_root),
            "CAPE_VENV": str(venv_root / "controller"),
            "CAPE_ADDITIONAL_POOLS": "",
            "BUNDLE": str(bundle),
            "RUN_ID": "default-rscape",
        })

        result = subprocess.run(
            [str(CONTROLLER_SCRIPT)], env=env, check=True,
            text=True, capture_output=True,
        )

        self.assertFalse(venv_root.exists())
        evidence = bundle / "pool" / "controller" / "implementation.txt"
        self.assertEqual(
            evidence.read_text(encoding="utf-8").splitlines(),
            [
                "implementation=rscape",
                "version=0.1.0",
                "executable=" + str(rscape),
            ],
        )
        self.assertIn("cape_implementation=rscape", result.stderr)


if __name__ == "__main__":
    unittest.main()
