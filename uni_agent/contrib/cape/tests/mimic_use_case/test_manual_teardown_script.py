# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


MANUAL = Path(__file__).resolve().parent / "manual"
LAUNCH = MANUAL / "launch_role.sh"
SHUTDOWN = MANUAL / "shutdown.sh"
CONTROLLER = MANUAL / "controller.sh"
REGISTER = MANUAL / "register_node.sh"


class ManualTeardownScriptTests(unittest.TestCase):
    def test_shutdown_retires_only_structurally_recorded_memberships(self):
        shutdown = self.root / "shutdown.sh"
        shutil.copy2(SHUTDOWN, shutdown)
        shutil.copy2(MANUAL / "role_record.py", self.root / "role_record.py")
        calls = self.root / "retire-calls.txt"
        self.executable("retire_node.sh", 'printf "%s\\n" "$*" >> "$RETIRE_CALLS"\n')
        binary = self.executable("rscape", "trap 'exit 0' TERM INT\nwhile :; do sleep 1; done\n")
        logs = self.root / "logs"
        logs.mkdir()
        for role in ("exclusive", "shared", "foreign"):
            process = subprocess.Popen(["setsid", str(binary), "node", "enroll"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process_groups.append(process)
            for _ in range(100):
                if os.getsid(process.pid) == process.pid:
                    break
                time.sleep(0.01)
            if role == "foreign":
                foreign = process
                continue
            (logs / (role + "_node.pid")).write_text(f"{process.pid} {process.pid} {role}-node\n")
            start = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            (logs / (role + "_node.json")).write_text(json.dumps({
                "version": 1, "pid": process.pid, "sid": process.pid, "start": start,
                "controller_url": "http://scheduler:18892", "pool": role + "-pool",
                "node_id": role + "-owned", "allocation_id": role + "-allocation"}))
        result = subprocess.run([str(shutdown), "--log-dir", str(logs), "--settle", "0"],
                                env=dict(os.environ, CAPE_URL="http://scheduler:18892",
                                         CAPE_USER_TOKEN_FILE=str(self.root / "user.json"), RETIRE_CALLS=str(calls)),
                                capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        recorded = calls.read_text().splitlines()
        self.assertEqual(len(recorded), 2)
        for role, line in zip(("exclusive", "shared"), recorded):
            self.assertIn("--node-id " + role + "-owned", line)
            self.assertIn("--pool " + role + "-pool", line)
        self.assertIsNone(foreign.poll())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.process_groups = []

    def tearDown(self):
        for process in self.process_groups:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)

    def executable(self, name, body):
        path = self.root / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    @staticmethod
    def process_exists(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def test_long_lived_wrappers_redirect_then_exec_the_binary(self):
        controller = CONTROLLER.read_text(encoding="utf-8")
        register = REGISTER.read_text(encoding="utf-8")
        self.assertIn('"$CAPE_BIN" "${controller_args[@]}" > >(tee "$CAPE_CONTROLLER_LOG") 2>&1 &', controller)
        self.assertIn('trap cleanup_controller EXIT INT TERM', controller)
        self.assertIn('wait "$controller_pid"', controller)
        self.assertIn('exec > >(tee -a "$CAPE_AGENT_LOG") 2>&1', register)
        self.assertTrue(register.rstrip().endswith('exec "$CAPE_NODE_BIN" "${join_args[@]}"'))
        self.assertNotIn('| tee "$CAPE_CONTROLLER_LOG"', controller)
        self.assertNotIn('| tee -a "$CAPE_AGENT_LOG"', register)

    def test_launch_records_the_execed_session_leader_and_shutdown_is_idempotent(self):
        binary = self.executable(
            "rscape",
            "trap 'exit 0' TERM INT\nwhile :; do sleep 1; done\n",
        )
        wrapper = self.executable(
            "controller-wrapper.sh",
            "sid=$(python3 -c 'import os; print(os.getsid(0))')\n"
            "printf '%s %s %s\\n' \"$$\" \"$sid\" \"$CAPE_ROLE_LABEL\" > \"$CAPE_ROLE_PID_FILE\"\n"
            'exec "$1" controller start\n',
        )
        log_dir = self.root / "logs"
        pid_file = log_dir / "controller.pid"
        launched = subprocess.run(
            [
                str(LAUNCH),
                "--pid-file",
                str(pid_file),
                "--log-file",
                str(log_dir / "controller.out"),
                "--label",
                "controller",
                "--timeout",
                "10",
                "--",
                str(wrapper),
                str(binary),
            ],
            text=True,
            capture_output=True,
        )
        launch_log = (log_dir / "controller.out").read_text() if (log_dir / "controller.out").exists() else ""
        self.assertEqual(
            launched.returncode, 0, launched.stdout + launched.stderr + launch_log
        )
        pid, execution_session_id, label = pid_file.read_text().split()
        pid = int(pid)
        self.assertEqual(int(execution_session_id), pid)
        self.assertEqual(label, "controller")
        self.assertIn("[launch] controller", launched.stdout)
        command = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)],
            text=True,
            capture_output=True,
        ).stdout
        if command:
            self.assertIn("rscape controller start", command)
            self.assertNotIn("controller-wrapper.sh", command)

        first = subprocess.run(
            [str(SHUTDOWN), "--log-dir", str(log_dir), "--settle", "0"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        for _ in range(50):
            if not self.process_exists(pid):
                break
            time.sleep(0.02)
        self.assertFalse(self.process_exists(pid))
        second = subprocess.run(
            [str(SHUTDOWN), "--log-dir", str(log_dir), "--settle", "0"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("complete", second.stdout)

    def test_unrecorded_role_is_reported_but_never_signalled(self):
        binary = self.executable(
            "rscape",
            "trap 'exit 0' TERM INT\nwhile :; do sleep 1; done\n",
        )
        process = subprocess.Popen(
            ["setsid", str(binary), "node", "enroll"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.process_groups.append(process)
        for _ in range(50):
            if process.poll() is not None:
                self.fail("unrecorded fake role exited")
            command = subprocess.run(
                ["ps", "-o", "args=", "-p", str(process.pid)],
                text=True,
                capture_output=True,
            ).stdout
            if "rscape node enroll" in command:
                break
            time.sleep(0.02)
        log_dir = self.root / "empty-logs"
        log_dir.mkdir()
        scoped = subprocess.run(
            [str(SHUTDOWN), "--log-dir", str(log_dir), "--settle", "0"],
            text=True, capture_output=True,
        )
        self.assertEqual(scoped.returncode, 0, scoped.stdout + scoped.stderr)
        self.assertIsNone(process.poll())
        result = subprocess.run(
            [str(SHUTDOWN), "--log-dir", str(log_dir), "--settle", "0", "--strict-host-audit"],
            text=True,
            capture_output=True,
        )
        process_table = subprocess.check_output(
            ["ps", "-u", str(os.getuid()), "-o", "pid=,pgid=,sid=,args="],
            text=True,
        )
        self.assertNotEqual(
            result.returncode,
            0,
            result.stdout + result.stderr + process_table,
        )
        self.assertIsNone(process.poll())
        self.assertIn(
            "unmanaged CAPE role remains",
            result.stderr,
            result.stdout + result.stderr + process_table,
        )
        self.assertRegex(result.stderr, r"stop explicitly with: kill -TERM -- -[0-9]+")


if __name__ == "__main__":
    unittest.main()
