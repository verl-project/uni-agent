"""Boundary checks for the allowlisted evidence adapter. No model calls."""

import hashlib
import json
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

    def test_match_preserves_one_literal_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "scene").mkdir()
            scripts = work / "package/scripts"
            scripts.mkdir(parents=True)
            h = scripts / "check.py"
            h.write_text("import json,sys; print(json.dumps(sys.argv[1:]))", encoding="utf-8")
            r = Reader(work, h.name, hashlib.sha256(h.read_bytes()).hexdigest())
            marker = "--help two words; $x"
            self.assertEqual(
                json.loads(r.call("check", {"literal": marker})), [str(work / "scene"), "--match=" + marker]
            )
            for bad in (
                {"literal": ""},
                {"literal": 2},
                {"literal": None},
                {"literal": "a" * 1001},
                {"argv": "--help"},
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    r.call("check", bad)
