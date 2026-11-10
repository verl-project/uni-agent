# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import subprocess
import tempfile
import unittest
import uuid

from pycape.identity import IdentityStore, new_secret, verifier
from pycape.instance import Instance
from pycape.journal import EventRecord, SQLiteJournal


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = os.path.join(self.temp.name, "instance")
        self.admin = new_secret()

    def test_start_does_not_initialize_and_init_never_overwrites(self):
        with self.assertRaises(FileNotFoundError):
            Instance.open(self.directory, self.admin)
        self.assertFalse(os.path.exists(self.directory))
        scheduler = Instance.initialize(self.directory, "admin", self.admin)
        with self.assertRaises(FileExistsError):
            Instance.initialize(self.directory, "admin", new_secret())
        instance = Instance.open(self.directory, self.admin)
        self.addCleanup(instance.close)
        self.assertEqual(instance.store.snapshot()["scheduler_id"], scheduler)

    def test_restart_retains_identity_and_events(self):
        scheduler = Instance.initialize(self.directory, "admin", self.admin)
        instance = Instance.open(self.directory, self.admin)
        database = instance.database
        journal = SQLiteJournal(database)
        journal.append(EventRecord(100, "test", "entity", details={"marker": "persisted"}))
        journal.close()
        instance.close()
        restarted = Instance.open(self.directory, self.admin)
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.store.snapshot()["scheduler_id"], scheduler)
        journal = SQLiteJournal(database)
        try:
            self.assertEqual(journal.list_events()[0].details["marker"], "persisted")
        finally:
            journal.close()

    def test_admin_credential_and_exclusive_lock(self):
        Instance.initialize(self.directory, "admin", self.admin)
        with self.assertRaises(PermissionError):
            Instance.open(self.directory, new_secret())
        instance = Instance.open(self.directory, self.admin)
        secret = new_secret()
        instance.store.register_token(str(uuid.uuid4()), verifier(secret), "regular", bearer=self.admin)
        with self.assertRaises(BlockingIOError):
            Instance.open(self.directory, self.admin)
        instance.close()
        with self.assertRaises(PermissionError):
            Instance.open(self.directory, secret)
        # Failed authentication releases its lock for a valid subsequent start.
        Instance.open(self.directory, self.admin).close()

    def test_incomplete_init_is_not_repaired_by_start(self):
        os.mkdir(self.directory, 0o700)
        path = os.path.join(self.directory, "controller.lock")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        store = IdentityStore.create(os.path.join(self.directory, "state.sqlite3"), "admin", self.admin)
        store.close()
        with self.assertRaisesRegex(ValueError, "journal is missing"):
            Instance.open(self.directory, self.admin)

    def test_close_unlocks_even_if_a_fork_inherited_the_description(self):
        Instance.initialize(self.directory, "admin", self.admin)
        instance = Instance.open(self.directory, self.admin)
        inherited = os.dup(instance.lock.fileno())
        try:
            instance.close()
            Instance.open(self.directory, self.admin).close()
        finally:
            os.close(inherited)

    @unittest.skipUnless(os.environ.get("CAPE_IDENTITY_RUST_TEST_BIN"), "requires native Rust test binary")
    def test_native_instance_interop_and_cross_process_lock(self):
        fixture = os.path.join(self.temp.name, "fixture.json")

        def native(directory, create=False, locked=False):
            with open(fixture, "w") as stream:
                os.chmod(fixture, 0o600)
                json.dump({"directory": directory, "secret": self.admin, "create": create, "locked": locked}, stream)
            env = dict(os.environ, CAPE_INSTANCE_FIXTURE=fixture)
            subprocess.run([env["CAPE_IDENTITY_RUST_TEST_BIN"], "--ignored", "--exact",
                            "instance::tests::cross_language_fixture"], env=env, check=True, timeout=30,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        Instance.initialize(self.directory, "admin", self.admin)
        instance = Instance.open(self.directory, self.admin)
        try:
            native(self.directory, locked=True)
        finally:
            instance.close()
        native(self.directory)
        native_directory = os.path.join(self.temp.name, "native-instance")
        native(native_directory, create=True)
        Instance.open(native_directory, self.admin).close()


if __name__ == "__main__":
    unittest.main()
