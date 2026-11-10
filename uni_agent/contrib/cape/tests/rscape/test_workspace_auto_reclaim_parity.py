# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real CLI/HTTP/journal tests with seeded workspace records, no FUSE claims."""

import json
import subprocess
import tempfile
import unittest
import urllib.error

from pycape import EventRecord, SQLiteJournal
from pycape.workspace import WorkspaceConfig
from tests.rscape import test_functional_parity as parity


class WorkspaceAutoReclaimParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def test_cli_permissions_validation_and_cross_backend_restart(self):
        helper = parity.FunctionalParityTests()
        binaries = (parity.PYCAPE_BIN, parity.RSCAPE_BIN)
        for server in binaries:
            with self.subTest(server=server), tempfile.TemporaryDirectory() as tmp:
                process, url, journal_path = helper.start_controller(server, tmp)
                fixture = helper.fixtures[url]
                pool = fixture.pool("pool")
                owner = fixture.user_token("alice")
                other = fixture.user_token("bob")
                user_id = fixture.request("GET", "/v1/identity/me", token=owner)["user_id"]
                helper.stop_controller(process)
                journal = SQLiteJournal(str(journal_path))
                for identity in ("ws-active", "ws-granted"):
                    record = {"id": identity, "user_id": user_id, "pool": pool, "node_id": "node",
                        "execution_session_id": "execution", "state": "mounted", "revision": 1,
                        "config": WorkspaceConfig().to_dict()}
                    if identity == "ws-granted":
                        record["operation"] = {"id": "already-granted", "state": "pending", "action": "reclaim",
                            "dispatch_granted": True, "source": "automatic", "sequence": 1}
                    journal.append(EventRecord(0, "workspace_state", identity, "mounted", None, record))
                journal.close()
                process, url, _ = helper.start_controller(server, tmp)
                try:
                    for client in binaries:
                        def cli(action, identity="ws-active", *extra, token=owner, success=True):
                            result = subprocess.run([client, "workspace", action, identity, "--controller-url", url,
                                "--token", token, *extra], capture_output=True, text=True, timeout=20)
                            if success:
                                self.assertEqual(result.returncode, 0, result.stderr)
                                return json.loads(result.stdout)
                            self.assertNotEqual(result.returncode, 0, result.stdout)

                        for bad_token in (other, "invalid"):
                            cli("set", "ws-active", "--auto-reclaim", "off", token=bad_token, success=False)
                        for args in ((), ("--auto-reclaim", "on"), ("--auto-reclaim", "off", "--size-gib", "1"),
                                     ("--auto-reclaim", "off", "--operation-id", "existing")):
                            cli("set", "ws-active", *args, success=False)
                        cli("status", "ws-active", "--auto-reclaim", "off", success=False)
                        cli("set", "missing", "--auto-reclaim", "off", success=False)
                        for body in ({}, {"auto_reclaim": True}, {"auto_reclaim": "off", "operation_id": None}):
                            with self.assertRaises(urllib.error.HTTPError) as error:
                                fixture.request("POST", "/v1/workspaces/ws-active/set", body, owner)
                            self.assertEqual(error.exception.code, 400)
                        for token in (owner, fixture.admin):
                            for policy in ("off", "off", "inherit", "off"):
                                result = cli("set", "ws-active", "--auto-reclaim", policy, token=token)
                                self.assertEqual(result["auto_reclaim"], policy)
                                self.assertFalse(result["auto_reclaim_enabled"])
                                self.assertEqual(result["state"], "mounted")
                                self.assertNotIn("operation", result)
                        granted = cli("set", "ws-granted", "--auto-reclaim", "off")
                        self.assertEqual(granted["operation"]["id"], "already-granted")
                        self.assertTrue(granted["operation"]["dispatch_granted"])
                    process.kill()
                    process.wait(timeout=5)
                finally:
                    helper.stop_controller(process)
                for reader in reversed(binaries):
                    process, url, _ = helper.start_controller(reader, tmp)
                    try:
                        for identity in ("ws-active", "ws-granted"):
                            result = helper.request("GET", url, "/v1/workspaces/" + identity, token=owner)
                            self.assertEqual(result["auto_reclaim"], "off")
                            self.assertFalse(result["auto_reclaim_enabled"])
                            if identity == "ws-granted":
                                self.assertEqual(result["operation"]["id"], "already-granted")
                    finally:
                        helper.stop_controller(process)


if __name__ == "__main__":
    unittest.main()
