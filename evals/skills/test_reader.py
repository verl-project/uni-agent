"""Boundary checks for the allowlisted evidence adapter. No model calls."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from adapters.reader import Reader


class ReaderTests(unittest.TestCase):
    def test_read_and_helper_are_allowlisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work = base / "work"
            (work / "scene").mkdir(parents=True)
            (work / "scene/note.txt").write_text("two badges", encoding="utf-8")
            (base / "private.txt").write_text("grader secret")
            (work / "package/scripts").mkdir(parents=True)
            helper = work / "package/scripts/check.py"
            helper.write_text("print('synthetic smoke helper')", encoding="utf-8")
            sha = hashlib.sha256(helper.read_bytes()).hexdigest()
            r = Reader(work, "check.py", sha)
            self.assertEqual(r.call("read", {"path": "scene/note.txt", "offset": 0}), "two badges")
            self.assertIn("scene/note.txt:1", r.call("search", {"literal": "badges"}))
            self.assertNotIn("private", r.call("inventory", {}))
            self.assertRaises(KeyError, r.call, "read", {"path": "../private.txt", "offset": 0})
            self.assertRaises(ValueError, Reader, work, "check.py", "wrong-pin")
            self.assertIn("smoke helper", r.call("check", {}))
            with self.assertRaisesRegex(ValueError, "No approved helper installed"):
                Reader(work).call("check", {})
