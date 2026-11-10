# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest

from pycape.telemetry_supervisor import PrometheusSupervisor


class TelemetrySupervisorTests(unittest.TestCase):
    def test_no_symlink_or_shared_directory_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir()
            (root / "other").mkdir()
            (root / "prometheus").symlink_to(root / "other", target_is_directory=True)
            supervisor = PrometheusSupervisor(root / "state", "scheduler")
            with self.assertRaises(PermissionError):
                supervisor.stop()
            (root / "prometheus").unlink()
            (root / "prometheus").mkdir(mode=0o755)
            with self.assertRaises(PermissionError):
                supervisor.stop()

    def test_other_scheduler_is_not_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir()
            (root / "prometheus").mkdir(mode=0o700)
            (root / "prometheus/instance.json").write_text(json.dumps({"scheduler_id": "other"}))
            with self.assertRaises(ValueError):
                PrometheusSupervisor(root / "state", "scheduler").stop()


if __name__ == "__main__":
    unittest.main()
