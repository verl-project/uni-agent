# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest


POOL_SCRIPT = Path(__file__).resolve().parent / "pool.sh"



class PoolScriptTests(unittest.TestCase):
    def test_teardown_keeps_controller_alive_during_agent_grace(self) -> None:
        source = POOL_SCRIPT.read_text(encoding="utf-8")

        remove = source.index('node drain', source.index('  down)'))
        signal_agent = source.index('kill "$p"', remove)
        poll_agent = source.index('kill -0 "$p"', signal_agent)
        force_agent = source.index('kill -KILL "$p"', poll_agent)
        stop_controller = source.index('kill "$controller_pid"', force_agent)
        poll_controller = source.index('kill -0 "$controller_pid"', stop_controller)
        force_controller = source.index('kill -KILL "$controller_pid"', poll_controller)

        self.assertLess(remove, signal_agent)
        self.assertLess(signal_agent, poll_agent)
        self.assertLess(poll_agent, force_agent)
        self.assertLess(force_agent, stop_controller)
        self.assertLess(stop_controller, poll_controller)
        self.assertLess(poll_controller, force_controller)
        self.assertIn('seq 1 50', source)


if __name__ == "__main__":
    unittest.main()
