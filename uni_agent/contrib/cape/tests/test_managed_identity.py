# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from pycape.cli import cape_main, _request, CapeHTTPError
from pycape.controller import CAPEController
from pycape.credential_file import read_registration
from pycape.identity import new_secret
from pycape.instance import Instance
from pycape.journal import SQLiteJournal
from pycape.managed_auth import ManagedAuthority
from pycape.service import CAPEControllerService


class ManagedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.admin = new_secret()
        directory = os.path.join(self.temp.name, "state")
        Instance.initialize(directory, "admin", self.admin)
        self.instance = Instance.open(directory, self.admin)
        self.addCleanup(self.instance.close)
        self.journal = SQLiteJournal(self.instance.database)
        self.addCleanup(self.journal.close)
        self.controller = CAPEController.reconcile_from_journal(self.journal, auth=ManagedAuthority(self.instance.store))
        self.service = CAPEControllerService(self.controller, schedule_interval=3600).start()
        self.addCleanup(self.service.stop)

    def call(self, method, path, token=None, payload=None):
        return _request(method, self.service.url, path, token=token, payload=payload)

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        code = cape_main(list(args), stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    def user(self, name):
        path = os.path.join(self.temp.name, name + ".json")
        code, out, err = self.cli("user", "token", "create", "--user", name, "--token-out", path,
                                  "--controller-url", self.service.url)
        self.assertEqual(code, 0, err)
        return read_registration(path)["secret"], json.loads(out), path

    def pool(self):
        return self.call("POST", "/v1/identity/admin", self.admin,
                         {"action": "pool-create", "name": "pool"})["pool_id"]

    def test_cli_registration_recovery_and_no_secret_output(self):
        token, result, path = self.user("alice")
        code, out, err = self.cli("user", "token", "resume", "--token-out", path)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["token_id"], result["token_id"])
        self.assertNotIn(token, out + err)
        actor = self.call("GET", "/v1/identity/me", token)
        self.assertEqual(actor["user_id"], result["user_id"])
        self.assertFalse(actor["is_admin"])

    def test_committed_response_loss_recovers_same_token(self):
        path = os.path.join(self.temp.name, "alice.json")
        original = _request

        def lost_response(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("response lost")

        with patch("pycape.cli._request", side_effect=lost_response):
            code, out, err = self.cli("user", "token", "create", "--user", "alice", "--token-out", path,
                                      "--controller-url", self.service.url)
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(err)["outcome"], "unknown")
        before = self.instance.store.snapshot()
        code, out, err = self.cli("user", "token", "resume", "--token-out", path)
        self.assertEqual(code, 0, err)
        self.assertEqual(before, self.instance.store.snapshot())

    def test_uncommitted_additional_token_cannot_change_target_identity(self):
        alice, identity, alice_file = self.user("alice")
        bob, _, bob_file = self.user("bob")
        path = os.path.join(self.temp.name, "additional.json")
        original = _request
        def disconnected(method, url, route, **kwargs):
            if method == "POST" and route == "/v1/identity/tokens":
                raise ConnectionError("not submitted")
            return original(method, url, route, **kwargs)
        with patch("pycape.cli._request", side_effect=disconnected):
            code, _, error = self.cli("user", "token", "create", "--token-file", alice_file,
                                      "--token-out", path, "--controller-url", self.service.url)
        self.assertEqual(code, 3, error)
        body = read_registration(path)["registration"]
        self.assertEqual(body["user_id"], identity["user_id"])
        before = self.instance.store.snapshot()
        with self.assertRaises(CapeHTTPError) as denied:
            self.call("POST", "/v1/identity/tokens", bob, body)
        self.assertEqual(denied.exception.status, 403)
        code, _, _ = self.cli("user", "token", "resume", "--token-out", path, "--token-file", bob_file)
        self.assertEqual(code, 3)
        self.assertEqual(before, self.instance.store.snapshot())
        code, out, error = self.cli("user", "token", "resume", "--token-out", path, "--token-file", alice_file)
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(out)["user_id"], identity["user_id"])

    def test_owner_access_survives_group_loss_and_queue_rechecks(self):
        token, user, _ = self.user("alice")
        bob, _, _ = self.user("bob")
        pool = self.pool()
        request = self.call("POST", "/v1/submit", token, {"pool": pool, "session_id": "t", "command": ["true"]})
        path = "/v1/requests/" + request["request_id"]
        with self.assertRaises(CapeHTTPError) as denied:
            self.call("GET", path, bob)
        self.assertEqual(denied.exception.status, 403)
        self.call("POST", "/v1/identity/admin", self.admin, {"action": "group-set", "name": "default", "all_pools": False})
        self.assertEqual(self.call("GET", path, token)["state"], "QUEUED")
        self.controller.schedule_once()
        self.assertEqual(self.call("GET", path, token)["reason"], "permission_revoked")
        self.journal.flush()
        restored = CAPEController.reconcile_from_journal(self.journal, auth=ManagedAuthority(self.instance.store))
        self.assertEqual(restored.get_request(request["request_id"], token).admission_token_id, user["token_id"])

    def test_regular_admin_credential_cannot_manage_or_see_admin_token(self):
        path = os.path.join(self.temp.name, "regular.json")
        code, out, err = self.cli("user", "token", "create", "--token", self.admin, "--token-out", path,
                                  "--controller-url", self.service.url)
        self.assertEqual(code, 0, err)
        token = read_registration(path)["secret"]
        self.assertFalse(self.call("GET", "/v1/identity/me", token)["is_admin"])
        self.assertEqual([t["kind"] for t in self.call("GET", "/v1/identity/tokens", token)], ["regular"])
        with self.assertRaises(CapeHTTPError):
            self.call("POST", "/v1/identity/admin", token, {"action": "pool-create", "name": "forbidden"})
        self.pool()

    def test_public_inventory_requires_identity(self):
        pool = self.pool()
        with self.assertRaises(CapeHTTPError) as denied:
            self.call("GET", "/v1/pools/{}/utilization".format(pool))
        self.assertEqual(denied.exception.status, 403)
        token, _, _ = self.user("alice")
        self.call("GET", "/v1/pools/{}/utilization".format(pool), token)

    def test_anonymous_registration_limit_does_not_block_authenticated_recovery(self):
        token, result, path = self.user("alice")
        body = read_registration(path)["registration"]
        for _ in range(63):
            with self.assertRaises(CapeHTTPError) as denied:
                self.call("POST", "/v1/identity/tokens", payload=body)
            self.assertEqual(denied.exception.status, 403)
        with self.assertRaises(CapeHTTPError) as limited:
            self.call("POST", "/v1/identity/tokens", payload=body)
        self.assertEqual(limited.exception.status, 429)
        recovered = self.call("POST", "/v1/identity/tokens", token, body)
        self.assertEqual(recovered["token_id"], result["token_id"])


if __name__ == "__main__":
    unittest.main()
