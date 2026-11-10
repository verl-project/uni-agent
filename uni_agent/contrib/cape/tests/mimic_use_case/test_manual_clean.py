# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('manual_clean', Path(__file__).parent / 'manual/clean.py')
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)


class ManualCleanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'Makefile').touch()
        self.logs = self.root / 'logs'
        self.reports = self.root / 'report'
        self.bundle = self.root / 'state'
        for path in (self.logs, self.reports, self.bundle):
            path.mkdir()
        self.state = self.bundle / 'controller'

    def run_clean(self):
        cleaner.clean(self.root, self.logs, self.reports, self.bundle, self.state)

    def spawn(self, state):
        script = self.root / 'rscape'
        script.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(120)\n')
        script.chmod(0o700)
        process = subprocess.Popen([str(script), 'controller', 'start', '--state-dir', str(state)],
                                   start_new_session=True)
        self.addCleanup(lambda: process.poll() is None and (process.terminate(), process.wait(timeout=5)))
        return process

    def test_orphan_exact_state_recovered_and_foreign_controller_preserved(self):
        own = self.spawn(self.state)
        foreign = self.spawn(self.root / 'foreign-state')
        original = cleaner.processes
        def selected():
            return (row for row in original() if int(row[0].name) in (own.pid, foreign.pid))
        with patch.object(cleaner, 'processes', selected):
            self.run_clean()
            self.run_clean()
        self.assertIsNotNone(own.poll())
        self.assertIsNone(foreign.poll())
        self.assertFalse(self.bundle.exists())

    def test_open_holder_blocks_all_deletion(self):
        (self.bundle / 'credential').write_text('preserve')
        with patch.object(cleaner, 'recover_controller'), patch.object(cleaner, 'open_holders', return_value=[123]), patch.object(cleaner.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'files preserved'):
                self.run_clean()
        self.assertEqual((self.bundle / 'credential').read_text(), 'preserve')
        self.assertTrue(self.logs.exists())

    def test_remote_nfs_placeholder_blocks_deletion(self):
        (self.logs / '.nfs123').touch()
        with patch.object(cleaner, 'recover_controller'), patch.object(cleaner, 'open_holders', return_value=[]), patch.object(cleaner.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'NFS files=1'):
                self.run_clean()
        self.assertTrue(self.bundle.exists())

    def test_unsafe_target_rejected_before_signalling(self):
        self.logs = self.root
        with patch.object(cleaner, 'recover_controller') as recover:
            with self.assertRaisesRegex(RuntimeError, 'refusing cleanup'):
                self.run_clean()
            recover.assert_not_called()

    def test_controller_parser_does_not_match_arbitrary_command_text(self):
        self.assertIsNone(cleaner.controller_state(['bash', '-c', 'rscape controller start --state-dir /tmp/x']))
        self.assertEqual(cleaner.controller_state(['/bin/rscape', 'controller', 'start', '--state-dir=/tmp/x']), '/tmp/x')

    def test_output_symlink_cannot_escape(self):
        (self.root / 'unsafe.out').symlink_to(self.root.parent)
        with patch.object(cleaner, 'recover_controller'):
            with self.assertRaisesRegex(RuntimeError, 'symlink escapes'):
                self.run_clean()
        self.assertTrue(self.bundle.exists())

    def test_mismatched_proc_namespace_fails_closed(self):
        with patch.object(cleaner.os, 'readlink', return_value=str(os.getpid() + 1)):
            with self.assertRaisesRegex(RuntimeError, 'PID namespace'):
                self.run_clean()
        self.assertTrue(self.bundle.exists())


if __name__ == '__main__':
    unittest.main()
