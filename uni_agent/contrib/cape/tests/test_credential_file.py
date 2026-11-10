# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from unittest.mock import patch

from pycape.credential_file import prepare_registration, read_registration, lock_registration


class CredentialFileTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.path = os.path.join(self.root.name, "token.json")

    def test_immutable_envelope_reuses_secret_and_operation(self):
        body = prepare_registration(self.path, "https://controller.example", "alice")
        self.assertEqual(read_registration(self.path), body)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            prepare_registration(self.path, "https://controller.example", "bob")
        self.assertEqual(read_registration(self.path), body)

    def test_symlink_ancestors_and_destination_rejected(self):
        directory = os.path.join(self.root.name, "private")
        os.mkdir(directory, 0o700)
        link = os.path.join(self.root.name, "link")
        os.symlink(directory, link)
        with self.assertRaises(OSError):
            prepare_registration(os.path.join(link, "token"), "https://controller.example")
        os.symlink(os.path.join(directory, "secret"), self.path)
        with self.assertRaises(FileExistsError):
            prepare_registration(self.path, "https://controller.example")
        self.assertFalse(os.path.exists(os.path.join(directory, "secret")))

    def test_parent_permissions_and_hardlinks_rejected(self):
        os.chmod(self.root.name, 0o755)
        with self.assertRaises(PermissionError):
            prepare_registration(self.path, "https://controller.example")
        os.chmod(self.root.name, 0o700)
        prepare_registration(self.path, "https://controller.example")
        os.link(self.path, os.path.join(self.root.name, "duplicate"))
        with self.assertRaises(PermissionError):
            read_registration(self.path)

    def test_fsync_failure_is_not_success_and_file_is_not_overwritten(self):
        with patch("pycape.credential_file.os.fsync", side_effect=OSError("injected sync failure")):
            with self.assertRaises(OSError):
                prepare_registration(self.path, "https://controller.example")
        # An uncertain local write is not permission to generate another secret
        # under the same path; recovery inspects the preserved envelope.
        with self.assertRaises(FileExistsError):
            prepare_registration(self.path, "https://controller.example")

    def test_nonregular_file_does_not_block(self):
        os.mkfifo(self.path, 0o600)
        with self.assertRaises(PermissionError):
            read_registration(self.path)

    def test_invalid_parameters_do_not_create_secret_file(self):
        for expiry in (float("nan"), float("inf"), True, "forever"):
            with self.assertRaises(ValueError):
                prepare_registration(self.path, "https://controller.example", expires_at=expiry)
            self.assertFalse(os.path.exists(self.path))
        with self.assertRaises(ValueError):
            prepare_registration(os.path.join(self.root.name, "unused", "..", "token"), "https://controller.example")

    def test_enrollment_file_lock_excludes_duplicate_agents(self):
        original = prepare_registration(self.path, "https://controller.example")
        guard = lock_registration(self.path)
        try:
            with self.assertRaisesRegex(ValueError, "another agent"):
                lock_registration(self.path)
            self.assertEqual(read_registration(self.path), original)
        finally:
            guard.close()
        lock_registration(self.path).close()

    def test_invalid_envelope_does_not_echo_secret(self):
        body = prepare_registration(self.path, "https://controller.example")
        with open(self.path, "w") as stream:
            stream.write(body["secret"])
        with self.assertRaisesRegex(ValueError, "^invalid credential envelope$"):
            read_registration(self.path)


if __name__ == "__main__":
    unittest.main()
