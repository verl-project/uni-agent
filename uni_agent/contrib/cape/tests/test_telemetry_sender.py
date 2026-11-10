# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from pycape.telemetry_sender import BatchSender, batch_configuration, receiver_url


class TelemetrySenderTests(unittest.TestCase):
    def test_destination_is_not_a_credential_or_arbitrary_path(self):
        for value in ("file:///etc/passwd", "http://u:p@host/api/v1/write", "http://host/metrics",
                      "http://host/api/v1/write?token=x", "http://host/api/v1/write#x",
                      "http://host:99999/api/v1/write", "http://host\n/api/v1/write"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                receiver_url(value)
        self.assertEqual(receiver_url("http://hostname:9090/api/v1/write"), "http://hostname:9090/api/v1/write")

    def test_configuration_escapes_path_and_has_no_wal(self):
        config = batch_configuration('/tmp/a"b/snapshot', "http://host/api/v1/write", 2)
        self.assertIn('a\\"b', config)
        self.assertIn('data_format = "prometheusremotewrite"', config)
        self.assertNotIn("buffer_directory", config)

    def test_send_cleanup_and_does_not_pass_cape_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            sender = BatchSender("http://host/api/v1/write", runtime_dir=root)
            def run(argv, **kwargs):
                self.assertEqual(argv[1], "--once")
                self.assertNotIn("CAPE_TOKEN", kwargs["env"])
                self.assertEqual(len(os.listdir(root)), 1)
                return subprocess.CompletedProcess(argv, 0)
            with patch.dict(os.environ, CAPE_TOKEN="do-not-forward"), patch("subprocess.run", side_effect=run):
                self.assertEqual(sender.send("gauge 1\n")["outcome"], "sent")
            self.assertEqual(os.listdir(root), [])

    def test_timeout_and_failure_drop_without_queue(self):
        with tempfile.TemporaryDirectory() as root:
            sender = BatchSender("http://host/api/v1/write", runtime_dir=root)
            for error in (FileNotFoundError(), subprocess.TimeoutExpired("telegraf", 3)):
                with patch("subprocess.run", side_effect=error):
                    self.assertEqual(sender.send("gauge 1\n")["outcome"], "dropped")
                self.assertEqual(os.listdir(root), [])
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1)):
                self.assertEqual(sender.send("gauge 1\n")["outcome"], "dropped")
            self.assertEqual(os.listdir(root), [])

    def test_bounded_batch(self):
        sender = BatchSender("http://host/api/v1/write")
        for data in ("x" * (4 * 1024 * 1024 + 1), "x 1\n" * 10001, b"x"):
            with self.assertRaises(ValueError):
                sender.send(data)
        self.assertEqual(sender.send("")["outcome"], "empty")


if __name__ == "__main__":
    unittest.main()
