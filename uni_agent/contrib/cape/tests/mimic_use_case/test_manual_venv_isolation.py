# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path


MANUAL_DIR = Path(__file__).resolve().parent / "manual"


class ManualVenvIsolationContractTests(unittest.TestCase):
    CASES = (
        (
            "controller.sh",
            "controller",
            "CAPE_VENV_HOST_ID",
            "CAPE_VENV_ROOT",
        ),
        (
            "register_node.sh",
            "node",
            "CAPE_NODE_VENV_HOST_ID",
            "CAPE_NODE_VENV_ROOT",
        ),
        (
            "retire_node.sh",
            "retire",
            "CAPE_RETIRE_VENV_HOST_ID",
            "CAPE_RETIRE_VENV_ROOT",
        ),
        (
            "submit.sh",
            "submit",
            "CAPE_SUBMIT_VENV_HOST_ID",
            "CAPE_SUBMIT_VENV_ROOT",
        ),
        (
            "status.sh",
            "status",
            "CAPE_STATUS_VENV_HOST_ID",
            "CAPE_STATUS_VENV_ROOT",
        ),
    )

    def test_explicit_pycape_venvs_include_role_host_architecture_and_python(self):
        for script_name, role, host_variable, root_variable in self.CASES:
            with self.subTest(script=script_name):
                source = (MANUAL_DIR / script_name).read_text(encoding="utf-8")
                self.assertIn(
                    '{}="${{{}:-$(hostname -s'.format(
                        host_variable, host_variable
                    ),
                    source,
                )
                self.assertIn(
                    "${{{}}}/{}-${{{}}}-$(uname -m)-python3.12".format(
                        root_variable, role, host_variable
                    ),
                    source,
                )
                self.assertTrue(
                    "!= pycape" in source or
                    '_BIN_SELECTION" = pycape' in source
                )

    def test_install_locks_are_scoped_to_the_same_role_and_host(self):
        for script_name, role, host_variable, _root_variable in self.CASES:
            with self.subTest(script=script_name):
                source = (MANUAL_DIR / script_name).read_text(encoding="utf-8")
                self.assertIn(
                    ".install-{}-${{{}}}-$(uname -m)-python3.12.lock".format(
                        role, host_variable
                    ),
                    source,
                )

    def test_automatic_python_requires_a_pip_bootstrap_path(self):
        for script_name, _role, _host_variable, _root_variable in self.CASES:
            with self.subTest(script=script_name):
                source = (MANUAL_DIR / script_name).read_text(encoding="utf-8")
                self.assertIn("python312_can_bootstrap_venv() {", source)
                self.assertIn(
                    'getattr(ensurepip, "_WHEEL_PKG_DIR", "")', source
                )
                self.assertIn(
                    "python312_can_bootstrap_venv python3.12", source
                )
                self.assertIn('"$@" -m pip --help', source)
                self.assertIn("external-pip", source)
                self.assertIn("--without-pip", source)
                self.assertIn('--python "$target/bin/python"', source)
                self.assertIn("cannot bootstrap a pip-enabled venv", source)
                self.assertIn(
                    '"$VENV_PYTHON" -m pip --version', source
                )

    def test_operator_readme_explains_equal_architecture_collision(self):
        readme = (MANUAL_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("role-and-host-qualified", readme)
        self.assertIn("same CPU architecture", readme)
        self.assertIn("must not share one default venv", readme)
        self.assertIn("ensurepip", readme)
        self.assertIn("pip wheel", readme)
        self.assertIn("--without-pip", readme)
        self.assertIn("--python", readme)


if __name__ == "__main__":
    unittest.main()
