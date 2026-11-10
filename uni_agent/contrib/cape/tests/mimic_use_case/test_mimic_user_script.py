# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


MIMIC_DIR = Path(__file__).resolve().parent
USER_SCRIPT = MIMIC_DIR / "user.sh"


class MimicUserScriptTests(unittest.TestCase):
    def run_user(self, direct_paths):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            call_log = root / "cape-calls.txt"
            fake_cape = fake_bin / "pycape"
            fake_cape.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$CAPE_TEST_LOG\"\n"
                "count=$(wc -l < \"$CAPE_TEST_LOG\")\n"
                "printf 'req-%06d\\n' \"$count\"\n",
                encoding="utf-8",
            )
            fake_cape.chmod(0o755)
            submits = root / "submits.csv"
            tokens = root / "tokens.csv"
            tokens.write_text("user-000,fixture-user-credential\n")
            tokens.chmod(0o600)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
                    "CAPE_TEST_LOG": str(call_log),
                    "CAPE_BIN": str(fake_cape),
                    "CAPE_DIRECT_PATHS": direct_paths,
                    "MIMIC_DIR": str(MIMIC_DIR),
                    "USER_ROOT": str(root / "users"),
                    "SUBMITS_CSV": str(submits),
                    "RUN_NONCE": "test-run",
                    "AUTH": "on",
                    "CAPE_USER_TOKENS": str(tokens),
                }
            )
            subprocess.run(
                ["bash", str(USER_SCRIPT), "user-000", "0"],
                check=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return call_log.read_text(encoding="utf-8").splitlines()

    def test_direct_paths_select_local_process_none(self):
        calls = self.run_user("on")
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn("--runtime-adapter local-process", call)
            self.assertIn("--isolation-policy none", call)
            self.assertNotIn(" --bind ", call)

    def test_bound_paths_preserve_image_stateful_default(self):
        calls = self.run_user("off")
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertNotIn("--runtime-adapter", call)
            self.assertIn("--isolation-policy image-stateful", call)
            self.assertIn(" --bind ", call)


if __name__ == "__main__":
    unittest.main()
