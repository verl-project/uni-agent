# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run: python -m unittest tests.rscape.test_queued_running_timeout -v."""
import unittest
from tests.running_timeout_repro import RunningTimeoutReproMixin


class RscapeRunningTimeoutTests(RunningTimeoutReproMixin, unittest.TestCase):
    binary_variable = 'RSCAPE_BIN'
    binary_default = 'rscape'


if __name__ == '__main__':
    unittest.main()
