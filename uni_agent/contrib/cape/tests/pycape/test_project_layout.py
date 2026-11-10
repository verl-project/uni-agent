# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_PROJECT = ROOT / "pycape"


class PythonProjectLayoutTests(unittest.TestCase):
    @staticmethod
    def pip_python():
        candidates = [
            sys.executable,
            getattr(sys, "_base_executable", None),
            shutil.which("python3.12"),
            shutil.which("python3.11"),
            shutil.which("python3"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            probe = subprocess.run(
                [candidate, "-m", "pip", "wheel", "--help"],
                text=True,
                capture_output=True,
            )
            backend = subprocess.run(
                [candidate, "-c", "import setuptools"],
                text=True,
                capture_output=True,
            )
            if (
                probe.returncode == 0
                and backend.returncode == 0
                and "--no-build-isolation" in probe.stdout
            ):
                return candidate
        return None

    def test_only_python_implementation_owns_python_metadata(self):
        self.assertFalse((ROOT / "pyproject.toml").exists())
        metadata = PYTHON_PROJECT / "pyproject.toml"
        self.assertTrue(metadata.is_file())
        text = metadata.read_text(encoding="utf-8")
        self.assertIn('packages = ["pycape"]', text)
        self.assertIn('pycape = "."', text)
        self.assertIn('pycape = "pycape.cli:cape_main"', text)
        self.assertTrue((ROOT / "rscape" / "Cargo.toml").is_file())

    def test_isolated_python_directory_builds_and_installs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "pycape-project"
            shutil.copytree(
                PYTHON_PROJECT,
                project,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", "build", "*.egg-info"
                ),
            )
            wheel_dir = root / "wheel"
            wheel_dir.mkdir()
            pip_python = self.pip_python()
            if pip_python is None:
                self.skipTest("packaging test requires modern pip/build tooling")
            built = subprocess.run(
                [
                    pip_python,
                    "-m",
                    "pip",
                    "wheel",
                    "--disable-pip-version-check",
                    "--no-build-isolation",
                    "--no-deps",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(project),
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            wheels = list(wheel_dir.glob("pycape-*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as archive:
                names = set(archive.namelist())
            self.assertIn("pycape/__init__.py", names)
            self.assertIn("pycape/cli.py", names)

            target = root / "installed"
            installed = subprocess.run(
                [
                    pip_python,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--target",
                    str(target),
                    str(wheels[0]),
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                installed.returncode,
                0,
                installed.stdout + installed.stderr,
            )
            env = dict(os.environ, PYTHONPATH=str(target))
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.metadata, pathlib, pycape; "
                        "pathlib.Path(pycape.__file__).resolve().relative_to(pathlib.Path(%r).resolve()); "
                        "dist=importlib.metadata.distribution('pycape'); "
                        "assert any(ep.name=='pycape' and ep.value=='pycape.cli:cape_main' "
                        "for ep in dist.entry_points)"
                    )
                    % str(target),
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)


if __name__ == "__main__":
    unittest.main()
