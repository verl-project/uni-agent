# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RECORD = Path(__file__).with_name("manual") / "role_record.py"


class RoleRecordTests(unittest.TestCase):
    def test_record_has_no_secret_and_rejects_mismatched_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "shared_node.pid"
            command = [sys.executable, str(RECORD), "write", "--pid-file", str(pidfile),
                       "--controller-url", "http://scheduler:18892", "--node-id", "own-node",
                       "--pool", "shared-pool", "--allocation-id", "own-allocation"]
            subprocess.run(command, check=True, capture_output=True)
            pidfile.write_text(f"{os.getpid()} {os.getsid(0)} shared-node\n")
            record = pidfile.with_suffix(".json")
            value = json.loads(record.read_text())
            self.assertEqual(set(value), {"version", "pid", "sid", "start", "controller_url",
                                          "node_id", "pool", "allocation_id"})
            command = [sys.executable, str(RECORD), "read", "--pid-file", str(pidfile),
                       "--controller-url", "http://scheduler:18892"]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.splitlines(), ["shared-pool", "own-node"])
            wrong_url = subprocess.run(command[:-1] + ["http://other:18892"], capture_output=True)
            self.assertNotEqual(wrong_url.returncode, 0)
            value["start"] = "not-this-process"
            record.write_text(json.dumps(value))
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
