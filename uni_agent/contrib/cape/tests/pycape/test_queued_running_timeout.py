# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run: python -m unittest tests.pycape.test_queued_running_timeout -v."""
import unittest
from tests.running_timeout_repro import RunningTimeoutReproMixin


class PycapeRunningTimeoutTests(RunningTimeoutReproMixin, unittest.TestCase):
    binary_variable = 'PYCAPE_BIN'
    binary_default = 'pycape'


if __name__ == '__main__':
    unittest.main()
