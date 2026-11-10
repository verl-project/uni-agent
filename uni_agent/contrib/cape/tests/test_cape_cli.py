# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tests.managed_fixture import ManagedFixture


def cape_binary():
    selected = os.environ.get("CAPE_BIN", "pycape")
    if os.sep in selected:
        path = Path(selected)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    else:
        found = shutil.which(selected)
        if found:
            return str(Path(found).resolve())
    raise RuntimeError("CAPE_BIN is not executable: " + selected)


class SharedCapeCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = cape_binary()
        selected = os.environ.get("CAPE_CONTROLLER_BIN")
        if selected is None:
            cls.controller_binary = cls.binary
        elif os.sep in selected:
            cls.controller_binary = str(Path(selected).resolve())
        else:
            cls.controller_binary = shutil.which(selected)
        if not cls.controller_binary:
            raise RuntimeError("CAPE_CONTROLLER_BIN is not executable")

    def command(self, *args, check=False):
        if getattr(self, "credential", None) and "--help" not in args:
            args = (args[0], "--token-file", str(self.credential), *args[1:])
        return subprocess.run(
            [self.binary, *args],
            check=check,
            text=True,
            capture_output=True,
        )

    def test_version_surface_identifies_selected_implementation(self):
        result = self.command("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        implementation = Path(self.binary).name
        self.assertRegex(
            result.stdout.strip(),
            r"^{} [0-9]+[.][0-9]+[.][0-9]+$".format(implementation),
        )

    def test_complete_help_surface(self):
        commands = (
            ("run",),
            ("status",),
            ("logs",),
            ("wait",),
            ("cancel",),
            ("utilization",),
            ("node", "enroll"),
            ("node", "drain"),
            ("node", "exit"),
            ("controller", "init"),
            ("controller", "start"),
            ("admin", "pool", "create"),
            ("admin", "pool", "drain"),
            ("admin", "pool", "edit"),
            ("admin", "user", "edit"),
            ("user", "token", "create"),
            ("user", "token", "resume"),
        )
        for command in commands:
            with self.subTest(command=command):
                result = self.command(*command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_malformed_run_has_parser_exit_and_no_stdout(self):
        result = self.command("run", "--user", "alice", "--", "true")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("--pool", result.stderr)

    def test_removed_interfaces_are_rejected_without_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in (
                ("controller", "start"),
                ("controller", "start", "--state-dir", tmp, "--journal-path", tmp + "/j"),
                ("controller", "start", "--state-dir", tmp, "--auth-file", tmp + "/auth"),
                ("node", "join", "--pool", "p"),
                ("node", "enroll", "--pool", "p", "--node-token-out", tmp + "/agent", "--join-token", "retired"),
                ("node", "enroll", "--pool", "p", "--node-token-out", tmp + "/agent", "--runtime-probe-image", "retired"),
                *(("admin", verb) for verb in ("add-user-token", "add-join-token", "drain-allocation", "remove-allocation", "set-pool-config")),
            ):
                with self.subTest(args=args):
                    result = self.command(*args)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_default_enrollment_attempts_all_containers_without_local_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ManagedFixture(self.controller_binary, tmp)
            controller = fixture.start()
            try:
                fixture.pool("auto-probe")
                credential = fixture.user_file("contributor")
                result = subprocess.run([
                    self.binary, "node", "enroll", "--controller-url", fixture.url,
                    "--pool", "auto-probe", "--node-id", "auto-missing",
                    "--token-file", str(credential), "--node-token-out", tmp + "/node.json",
                    "--apptainer-binary", tmp + "/missing-apptainer",
                    "--bubblewrap-binary", tmp + "/missing-bwrap",
                    "--podman-binary", tmp + "/missing-podman", "--once",
                    "--state-path", tmp + "/agent-state.json", "--log-dir", tmp + "/logs",
                ], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                for adapter in ("apptainer", "bubblewrap", "podman"):
                    self.assertIn("runtime probe skipped " + adapter, result.stderr)
                self.assertIn("no runtime adapter passed", result.stderr)
                state = fixture.request("GET", "/v1/pools/auto-probe/utilization")
                self.assertEqual(state["active_nodes"], 0)
                self.assertEqual(state["gpus_schedulable"], 0)
            finally:
                controller.terminate()
                controller.communicate(timeout=10)

    def test_async_submit_status_cancel_wait_and_utilization(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ManagedFixture(self.controller_binary, tmp)
            controller = fixture.start()
            url = fixture.url
            try:
                fixture.pool("pool-a")
                self.credential = fixture.user_file("alice")
                for _ in range(100):
                    probe = self.command(
                        "utilization",
                        "--controller-url",
                        url,
                        "--pool",
                        "pool-a",
                    )
                    if probe.returncode == 0:
                        break
                    time.sleep(0.02)
                self.assertEqual(probe.returncode, 0, probe.stderr)
                utilization = json.loads(probe.stdout)
                self.assertEqual(utilization["active_nodes"], 0)

                submitted = self.command(
                    "run",
                    "--controller-url",
                    url,
                    "--pool",
                    "pool-a",
                    "--user",
                    "alice",
                    "--session-id",
                    "shared-cli",
                    "--",
                    "true",
                )
                self.assertEqual(submitted.returncode, 0, submitted.stderr)
                request_id = submitted.stdout.strip()
                self.assertRegex(request_id, r"^req-[A-Za-z0-9._-]+$")

                status = self.command(
                    "status", request_id, "--controller-url", url
                )
                self.assertEqual(status.returncode, 0, status.stderr)
                self.assertEqual(json.loads(status.stdout)["state"], "QUEUED")

                cancelled = self.command(
                    "cancel", request_id, "--controller-url", url
                )
                self.assertEqual(cancelled.returncode, 0, cancelled.stderr)
                self.assertEqual(
                    json.loads(cancelled.stdout)["state"], "CANCELLED"
                )

                waited = self.command(
                    "wait",
                    request_id,
                    "--controller-url",
                    url,
                    "--timeout",
                    "1",
                )
                self.assertEqual(waited.returncode, 0, waited.stderr)
                self.assertEqual(json.loads(waited.stdout)["state"], "CANCELLED")

                logs = self.command(
                    "logs", request_id, "--controller-url", url
                )
                self.assertEqual(logs.returncode, 0, logs.stderr)
                self.assertEqual(logs.stdout, "")
            finally:
                self.credential = None
                controller.terminate()
                controller.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
