# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANUAL = ROOT / "manual"


class DefaultImplementationTests(unittest.TestCase):
    def test_repository_and_python_project_roots_are_distinct(self):
        helper = ROOT / "project_root.sh"
        probe = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; cape_project_root "$2"; pycape_project_root "$2"',
                "bash",
                str(helper),
                str(ROOT),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(
            probe.stdout.splitlines(),
            [str(ROOT.parents[1]), str(ROOT.parents[1] / "pycape")],
        )

    def test_every_role_defaults_to_rscape_and_keeps_one_pycape_branch(self):
        expectations = {
            MANUAL / "controller.sh": 'CAPE_BIN_SELECTION="${CAPE_BIN:-rscape}"',
            MANUAL / "register_node.sh": 'CAPE_NODE_BIN_SELECTION="${CAPE_NODE_BIN:-rscape}"',
            MANUAL / "submit.sh": 'CAPE_SUBMIT_BIN_SELECTION="${CAPE_SUBMIT_BIN:-rscape}"',
            MANUAL / "status.sh": 'CAPE_STATUS_BIN_SELECTION="${CAPE_STATUS_BIN:-rscape}"',
            MANUAL / "retire_node.sh": 'CAPE_RETIRE_BIN_SELECTION="${CAPE_RETIRE_BIN:-rscape}"',
            ROOT / "start.sh": 'CAPE_BIN_SELECTION="${CAPE_BIN:-rscape}"',
        }
        for path, expected in expectations.items():
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn(expected, text)
                self.assertIn("cape_inspect_executable", text)
                self.assertIn("cape_record_executable", text)

        manual_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in expectations
            if path.parent == MANUAL
        )
        self.assertEqual(manual_text.count("/bin/" + "pycape"), 5)

    def test_every_python_role_defaults_to_the_pycape_sibling_project(self):
        expectations = {
            MANUAL / "controller.sh": 'CAPE_PROJECT="${CAPE_PROJECT:-$REPO_ROOT/pycape}"',
            MANUAL / "register_node.sh": 'CAPE_PROJECT="${CAPE_NODE_PROJECT:-$REPO_ROOT/pycape}"',
            MANUAL / "submit.sh": 'CAPE_PROJECT="${CAPE_SUBMIT_PROJECT:-$REPO_ROOT/pycape}"',
            MANUAL / "status.sh": 'CAPE_PROJECT="${CAPE_STATUS_PROJECT:-$REPO_ROOT/pycape}"',
            MANUAL / "retire_node.sh": 'CAPE_PROJECT="${CAPE_RETIRE_PROJECT:-$REPO_ROOT/pycape}"',
        }
        for path, expected in expectations.items():
            with self.subTest(path=path.name):
                self.assertIn(expected, path.read_text(encoding="utf-8"))

    def test_rscape_resolution_records_version_without_creating_a_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            rscape = binary_dir / "rscape"
            rscape.write_text(
                "#!/usr/bin/env bash\n"
                "if [ \"${1:-}\" = --version ]; then "
                "printf 'rscape 0.1.0\\n'; exit 0; fi\n"
                "exit 2\n",
                encoding="utf-8",
            )
            rscape.chmod(rscape.stat().st_mode | stat.S_IXUSR)
            venv = root / "must-not-exist"
            evidence = root / "evidence" / "implementation.txt"
            script = "\n".join((
                "set -euo pipefail",
                'source "$HELPER"',
                'resolved="$(cape_resolve_executable rscape CAPE_BIN)"',
                'cape_inspect_executable rscape "$resolved"',
                'cape_record_executable focused "$resolved" "$EVIDENCE"',
                '[ ! -e "$VENV" ]',
            ))
            env = dict(os.environ)
            env.update({
                "PATH": str(binary_dir) + os.pathsep + env.get("PATH", ""),
                "HELPER": str(ROOT / "project_root.sh"),
                "EVIDENCE": str(evidence),
                "VENV": str(venv),
            })
            result = subprocess.run(
                ["bash", "-c", script], env=env, check=True,
                text=True, capture_output=True,
            )
            fields = dict(
                line.split("=", 1)
                for line in evidence.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(fields["implementation"], "rscape")
            self.assertEqual(fields["version"], "0.1.0")
            self.assertEqual(fields["executable"], str(rscape))
            self.assertIn("cape_implementation=rscape", result.stderr)
            self.assertFalse(venv.exists())

    def test_missing_default_rscape_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env.update({
                "PATH": "/usr/bin:/bin",
                "HELPER": str(ROOT / "project_root.sh"),
            })
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$HELPER"; cape_resolve_executable '
                    'definitely-missing-rscape CAPE_BIN',
                ],
                env=env, check=False, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "CAPE_BIN was not found: definitely-missing-rscape",
                result.stderr,
            )
            self.assertNotIn("pycape", result.stderr)


if __name__ == "__main__":
    unittest.main()
