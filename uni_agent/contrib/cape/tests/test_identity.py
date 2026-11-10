# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
import uuid

from pycape.identity import IdentityStore, new_secret, verifier
from pycape.credential_file import prepare_registration, read_registration


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = os.path.join(self.temp.name, "identity.sqlite3")
        self.admin = new_secret()
        self.store = IdentityStore.create(self.path, "admin", self.admin, clock=lambda: 100)
        self.addCleanup(self.store.close)

    def register(self, username="alice", bearer=None, expires_at=None):
        secret, operation = new_secret(), str(uuid.uuid4())
        args = (operation, verifier(secret), "laptop", username)
        result = self.store.register_token(*args, bearer=bearer, expires_at=expires_at)
        return secret, args, result

    def test_initial_identity_and_no_plaintext(self):
        actor = self.store.require_admin(self.admin)
        self.assertEqual(actor["name"], "admin")
        self.assertEqual(self.store.snapshot()["groups"]["default"], {"all_pools": True, "pools": []})
        with open(self.path, "rb") as stream:
            self.assertNotIn(self.admin.encode(), stream.read())
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_open_never_initializes_or_overwrites(self):
        missing = os.path.join(self.temp.name, "missing")
        with self.assertRaises(FileNotFoundError):
            IdentityStore(missing)
        self.assertFalse(os.path.exists(missing))
        with self.assertRaises(FileExistsError):
            IdentityStore.create(self.path, "other", new_secret())
        self.assertEqual(self.store.require_admin(self.admin)["name"], "admin")

    def test_symlink_and_permissions_rejected(self):
        link = os.path.join(self.temp.name, "link")
        os.symlink(self.path, link)
        with self.assertRaises(PermissionError):
            IdentityStore(link)
        with self.assertRaises(FileExistsError):
            IdentityStore.create(link, "admin", self.admin)
        os.chmod(self.path, 0o644)
        with self.assertRaises(PermissionError):
            IdentityStore(self.path)

    def test_regular_registration_and_authenticated_recovery(self):
        secret, args, result = self.register()
        self.assertEqual(self.store.authenticate(secret)["user_id"], result["user_id"])
        self.assertEqual(self.store.register_token(*args, bearer=secret), result)
        self.assertEqual(len(self.store.snapshot()["tokens"]), 2)
        with self.assertRaises(PermissionError):
            self.store.register_token(*args)
        with self.assertRaises(PermissionError):
            self.store.register_token(*args, bearer=args[1])
        with self.assertRaises(ValueError):
            self.store.register_token(args[0], verifier(new_secret()), args[2], args[3], bearer=secret)

    def test_duplicate_identity_and_cross_user_recovery(self):
        secret, args, result = self.register()
        bob, _, _ = self.register("bob")
        with self.assertRaises(ValueError):
            self.register()
        with self.assertRaises(PermissionError):
            self.store.register_token(*args, bearer=bob)
        with self.assertRaises(PermissionError):
            self.register("alice", bearer=bob)
        self.assertEqual(len(self.store.snapshot()["users"]), 3)

    def test_additional_tokens_keep_identity_and_groups(self):
        secret, _, result = self.register()
        self.store.edit_user(self.admin, result["user_id"], groups=[])
        second, _, other = self.register(None, bearer=secret)
        self.assertEqual(result["user_id"], other["user_id"])
        self.assertEqual(self.store.authenticate(second)["groups"], [])

    def test_regular_admin_is_not_privileged(self):
        regular, _, _ = self.register(None, bearer=self.admin)
        self.assertFalse(self.store.authenticate(regular)["is_admin"])
        with self.assertRaises(PermissionError):
            self.store.create_pool(regular, "forbidden")
        self.assertTrue(self.store.require_admin(self.admin)["is_admin"])
        self.assertEqual([t["kind"] for t in self.store.list_tokens(regular)], ["regular"])
        admin_id = self.store.authenticate(self.admin)["token_id"]
        with self.assertRaises(PermissionError):
            self.store.revoke_token(regular, admin_id)

    def test_union_default_changes_and_delete_without_fallback(self):
        secret, _, result = self.register()
        a = self.store.create_pool(self.admin, "a")
        b = self.store.create_pool(self.admin, "b")
        self.assertTrue(self.store.can_submit(secret, a))
        self.store.set_group(self.admin, "default", False, [])
        self.assertFalse(self.store.can_submit(secret, a))
        self.store.set_group(self.admin, "access-a", False, [a])
        self.store.set_group(self.admin, "access-b", False, [b])
        self.store.edit_user(self.admin, result["user_id"], groups=["access-a", "access-b"])
        self.assertTrue(self.store.can_submit(secret, a))
        self.assertTrue(self.store.can_submit(secret, b))
        self.store.delete_group(self.admin, "access-a")
        self.assertFalse(self.store.can_submit(secret, a))
        self.store.delete_group(self.admin, "access-b")
        self.assertEqual(self.store.authenticate(secret)["groups"], [])
        with self.assertRaises(ValueError):
            self.store.delete_group(self.admin, "default")
        self.store.require_owner(secret, result["user_id"])

    def test_owner_not_confused_with_submission_permission(self):
        secret, _, result = self.register()
        bob, _, bob_result = self.register("bob")
        self.store.edit_user(self.admin, result["user_id"], groups=[])
        self.store.require_owner(secret, result["user_id"])
        with self.assertRaises(PermissionError):
            self.store.require_owner(secret, bob_result["user_id"])
        self.store.require_owner(self.admin, bob_result["user_id"])

    def test_revocation_and_disable_cannot_be_replayed_away(self):
        secret, args, result = self.register()
        second, _, _ = self.register(None, bearer=secret)
        self.store.revoke_token(second, result["token_id"])
        with self.assertRaises(PermissionError):
            self.store.register_token(*args, bearer=secret)
        # Another still-valid owner credential may inspect the historical result,
        # but doing so cannot reactivate the revoked credential.
        self.assertEqual(self.store.register_token(*args, bearer=second), result)
        with self.assertRaises(PermissionError):
            self.store.authenticate(secret)
        self.store.edit_user(self.admin, result["user_id"], disabled=True)
        with self.assertRaises(PermissionError):
            self.store.register_token(*args, bearer=second)

    def test_expiry_and_validation(self):
        for expires in (99, float("nan"), float("inf"), True, "forever"):
            with self.assertRaises(ValueError):
                self.register(expires_at=expires)
        secret, _, _ = self.register(expires_at=101)
        self.store.clock = lambda: 101
        with self.assertRaises(PermissionError):
            self.store.authenticate(secret)
        for name in ("", "../alice", "alice/bob", "a" * 65):
            with self.assertRaises(ValueError):
                self.register(name)

    def test_transaction_failure_rolls_back_user_and_token(self):
        original = self.store.snapshot()
        self.store.db.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON authority BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.register()
        self.assertEqual(self.store.snapshot(), original)

    def test_reopen_and_other_connection_see_latest_permissions(self):
        secret, args, result = self.register()
        pool = self.store.create_pool(self.admin, "pool")
        other = IdentityStore(self.path, clock=lambda: 100)
        self.addCleanup(other.close)
        self.assertEqual(other.register_token(*args, bearer=secret), result)
        self.assertTrue(other.can_submit(secret, pool))
        self.store.set_group(self.admin, "default", False, [])
        self.assertFalse(other.can_submit(secret, pool))
        self.assertEqual(other.snapshot()["scheduler_id"], self.store.snapshot()["scheduler_id"])

    def test_two_connections_cannot_register_same_identity(self):
        barrier = threading.Barrier(2)
        outcomes = []

        def register():
            store = IdentityStore(self.path, clock=lambda: 100)
            try:
                secret = new_secret()
                barrier.wait(timeout=5)
                try:
                    store.register_token(str(uuid.uuid4()), verifier(secret), "laptop", "alice")
                    outcomes.append("registered")
                except ValueError:
                    outcomes.append("rejected")
            finally:
                store.close()

        threads = [threading.Thread(target=register) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        self.assertCountEqual(outcomes, ["registered", "rejected"])
        state = self.store.snapshot()
        self.assertEqual(len(state["users"]), 2)
        self.assertEqual(len(state["tokens"]), 2)
        self.assertEqual(len(state["operations"]), 1)

    def test_corrupt_version_fails_closed(self):
        state = self.store.snapshot()
        state["version"] = 999
        self.store.db.execute("UPDATE authority SET body=?", (json.dumps(state),))
        with self.assertRaises(ValueError):
            IdentityStore(self.path)

    def test_corrupt_boolean_policy_fails_closed(self):
        state = self.store.snapshot()
        state["groups"]["default"]["all_pools"] = "false"
        self.store.db.execute("UPDATE authority SET body=?", (json.dumps(state),))
        with self.assertRaises(ValueError):
            IdentityStore(self.path)

    def test_audit_is_atomic_private_and_secret_free(self):
        secret, args, result = self.register()
        self.store.set_group(self.admin, "default", False, [])
        events = self.store.audit_events(self.admin)
        self.assertTrue({"controller.init", "token.create", "group.set"}.issubset({e["action"] for e in events}))
        encoded = json.dumps(events)
        self.assertNotIn(secret, encoded)
        self.assertNotIn(self.admin, encoded)
        self.assertNotIn(verifier(secret), encoded)
        with self.assertRaises(PermissionError):
            self.store.audit_events(secret)
        count = self.store.db.execute("SELECT count(*) FROM audit").fetchone()[0]
        self.store.db.execute("CREATE TRIGGER reject_audited_update BEFORE UPDATE ON authority BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.register("bob")
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM audit").fetchone()[0], count)

    @unittest.skipUnless(os.environ.get("CAPE_IDENTITY_RUST_TEST_BIN"), "requires native Rust test binary")
    def test_native_cross_language_persistence(self):
        for rust_creates in (False, True):
            with self.subTest(rust_creates=rust_creates):
                database = os.path.join(self.temp.name, "rust-created.sqlite3") if rust_creates else self.path
                secret = new_secret()
                operation = str(uuid.uuid4())
                if not rust_creates:
                    self.store.register_token(operation, verifier(secret), "laptop", "rust-user", expires_at=200)
                fixture = os.path.join(self.temp.name, "fixture.json")
                client_input = os.path.join(self.temp.name, "python-client-{}.json".format(rust_creates))
                client_output = os.path.join(self.temp.name, "rust-client-{}.json".format(rust_creates))
                prepare_registration(client_input, "https://controller.example", "python-client", expires_at=200)
                with open(fixture, "w") as stream:
                    os.chmod(fixture, 0o600)
                    json.dump({"database": database, "admin_secret": self.admin, "regular_secret": secret,
                               "operation_id": operation, "create": rust_creates, "retry": not rust_creates,
                               "credential_input": client_input, "credential_output": client_output}, stream)
                env = dict(os.environ, CAPE_IDENTITY_FIXTURE=fixture)
                subprocess.run([env["CAPE_IDENTITY_RUST_TEST_BIN"], "--ignored", "--exact",
                                "identity::tests::cross_language_fixture"], env=env, check=True, timeout=30,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(read_registration(client_output)["registration"]["username"], "rust-client")
                with_store = IdentityStore(database, clock=lambda: 100)
                try:
                    actor = with_store.authenticate(secret)
                    result = with_store.register_token(operation, verifier(secret), "laptop", "rust-user",
                                                       bearer=secret, expires_at=200.0)
                    self.assertEqual(result["user_id"], actor["user_id"])
                    pool = next(iter(with_store.snapshot()["pools"]))
                    self.assertTrue(with_store.can_submit(secret, pool))
                    with_store.set_group(self.admin, "default", False, [])
                    self.assertFalse(with_store.can_submit(secret, pool))
                finally:
                    with_store.close()


if __name__ == "__main__":
    unittest.main()
