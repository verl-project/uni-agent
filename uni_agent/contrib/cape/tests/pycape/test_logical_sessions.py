# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pycape import CAPEController, MemoryJournal, WorkloadSpec
from pycape.session_identity import SessionSelection, validate_session_id
from pycape.service import workload_spec_from_dict


class LogicalSessionTests(unittest.TestCase):
    def test_owner_scope_and_replay(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        for owner, pool in (("alice", "one"), ("alice", "two"), ("bob", "one")):
            request = controller.submit_workload(WorkloadSpec(pool, owner, "same", ("true",)))
            self.assertEqual(controller._request_status_record(request)["session_id"], "same")
        self.assertEqual(set(controller.logical_sessions), {("alice", "same"), ("bob", "same")})
        restored = CAPEController.reconcile_from_journal(journal)
        self.assertEqual(restored.logical_sessions, controller.logical_sessions)

    def test_obsolete_wire_fields_are_rejected(self):
        for field in ("task_id", "session_key", "workspace_id"):
            body = dict(pool="pool", user_id="alice", session_id="logical", command=["true"])
            body[field] = "old"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "obsolete"):
                workload_spec_from_dict(body)

    def test_id_validation(self):
        for value in (None, "", "space here", "bad\n", "bad\u2003name", "bad\x85name", "x" * 257):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_session_id(value)
        self.assertEqual(validate_session_id("project/run-A"), "project/run-A")

    def test_durable_submission_alone_restores_logical_identity(self):
        controller = CAPEController()
        controller.submit_workload(WorkloadSpec("pool", "alice", "sequence", ("true",)))
        truncated = MemoryJournal()
        truncated.append(next(event for event in controller.journal.list_events() if event.event_type == "request_submitted"))
        restored = CAPEController.reconcile_from_journal(truncated)
        self.assertIn(("alice", "sequence"), restored.logical_sessions)


class LastSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        os.chmod(self.root, 0o700)
        self.project = self.root / "project"
        self.project.mkdir()
        self.identity = {"scheduler_id": "controller-A", "user_id": "alice"}

    def select(self, explicit=None, identity=None, project=None):
        return SessionSelection(identity or self.identity, explicit, self.root / "cache", project or self.project)

    def test_only_accepted_submission_updates_cache(self):
        with self.select() as first:
            original = first.session_id
        with self.select() as second:
            self.assertNotEqual(second.session_id, original)
            second.accepted({"request_id": "req-1", "session_id": second.session_id})
            accepted = second.session_id
        with self.select("failed-explicit"):
            pass
        with self.select() as again:
            self.assertEqual(again.session_id, accepted)
        with self.select("chosen") as explicit:
            explicit.accepted({"request_id": "req-2", "session_id": "chosen"})
        with self.select() as again:
            self.assertEqual(again.session_id, "chosen")

    def test_controller_owner_project_isolation_and_no_token_key(self):
        with self.select("kept") as selection:
            selection.accepted({"request_id": "req-1", "session_id": "kept"})
        other_project = self.root / "other"
        other_project.mkdir()
        for identity, project in ((dict(self.identity, user_id="bob"), self.project),
                                  (dict(self.identity, scheduler_id="controller-B"), self.project),
                                  (self.identity, other_project)):
            with self.select(identity=identity, project=project) as selection:
                self.assertNotEqual(selection.session_id, "kept")
        with self.select(identity=dict(self.identity, token_id="new-token")) as selection:
            self.assertEqual(selection.session_id, "kept")

    def test_invalid_receipt_and_symlink_fail_closed(self):
        with self.select("chosen") as selection:
            with self.assertRaisesRegex(ValueError, "does not confirm"):
                selection.accepted({"request_id": "req-1", "session_id": "different"})
            path = self.root / "cache" / selection.filename
        target = self.root / "target"
        target.write_text("do not overwrite")
        path.symlink_to(target)
        with self.assertRaises(OSError):
            with self.select():
                pass
        self.assertEqual(target.read_text(), "do not overwrite")

    def test_cli_omission_reuses_accepted_session_and_emits_identity(self):
        from pycape.cli import cape_main
        submitted = []

        def request(method, url, path, payload=None, **kwargs):
            if method == "GET" and path == "/v1/identity/me":
                return self.identity
            self.assertEqual((method, path), ("POST", "/v1/submit"))
            submitted.append(payload)
            return {"request_id": "req-{}".format(len(submitted)), "session_id": payload["session_id"]}

        common = ["run", "--controller-url", "http://controller.invalid", "--pool", "pool", "--token", "test"]
        with patch.dict(os.environ, {"CAPE_SESSION_STATE_DIR": str(self.root / "cache")}), \
                patch("pycape.cli._request", side_effect=request), patch("sys.stderr", io.StringIO()) as stderr:
            for args in ([], [], ["--session-id", "chosen"], []):
                self.assertEqual(cape_main(common + args + ["--", "true"], stdout=io.StringIO()), 0)
            self.assertIn("session_id=chosen", stderr.getvalue())
        self.assertEqual(submitted[0]["session_id"], submitted[1]["session_id"])
        self.assertEqual(submitted[2]["session_id"], "chosen")
        self.assertEqual(submitted[3]["session_id"], "chosen")
        self.assertTrue(all("task_id" not in body and "session_key" not in body for body in submitted))


if __name__ == "__main__":
    unittest.main()
