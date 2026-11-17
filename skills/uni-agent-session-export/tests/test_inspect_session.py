"""Synthetic parser fixtures; these are format tests, not incident or agent evaluations."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_session.py"


class ExportIndexTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def export(self, directory, declared, rows, **extra):
        folder = self.root / directory
        folder.mkdir(parents=True)
        meta = {"session_id": "reused", "trajectories_count": declared, **extra}
        (folder / "session_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        if rows is not None:
            (folder / "trajectories.jsonl").write_bytes(rows)
        return folder

    def run_index(self, *args):
        output = subprocess.check_output([sys.executable, "-B", SCRIPT, self.root, *args], text=True)
        return json.loads(output)

    def test_same_id_and_marker_report_all_candidates(self):
        for group in ("first", "second"):
            self.export(group + "/reused", 1, b'{"session_id":"reused","request":"shared-marker private-detail"}\n')
        report = self.run_index("--match", "shared-marker")
        self.assertEqual([e["directory"] for e in report["exports"]], ["first/reused", "second/reused"])
        self.assertEqual(
            [m["file"] for m in report["matches"]], [f"{g}/reused/trajectories.jsonl" for g in ("first", "second")]
        )
        self.assertTrue(all(m["line"] == 1 and m["fields"] == ["$.request"] for m in report["matches"]))
        self.assertNotIn("private-detail", json.dumps(report))

    def test_missing_empty_and_snapshot_are_distinct(self):
        self.export("missing", 2, None, trajectories_path=str(self.root / "empty" / "trajectories.jsonl"))
        folder = self.export("empty", 0, b"\n  \n")
        (folder / "debug_snapshot.json").write_text('{"session_state":{"num_trajectories": 4}}', encoding="utf-8")
        empty, missing = self.run_index()["exports"]
        self.assertEqual(
            (empty["jsonl"]["state"], empty["jsonl"]["valid"], empty["snapshot"]["snapshot_count"]), ("empty", 0, 4)
        )
        self.assertEqual(
            (missing["jsonl"]["state"], missing["jsonl"]["valid"], missing["comparison"]), ("missing", None, "unknown")
        )

    def test_corrupt_rows_and_mixed_identifiers(self):
        self.export("one", 3, b'{"session_id":"reused"}\n\nnull\n\xff\n{}\n{"session_id":"other"}')
        exp = self.run_index()["exports"][0]
        self.assertEqual(exp["jsonl"]["invalid_line_examples"], [3, 4])
        self.assertEqual(exp["jsonl"]["valid"], 3)
        self.assertEqual(exp["jsonl"]["without_id"], 1)
        self.assertTrue(exp["id_conflict"])
        self.assertEqual(exp["comparison"], "unknown")

    def test_count_disagreement_and_invalid_declaration(self):
        self.export("invalid", True, b"{}")
        self.export("different", 3, b"{}\n{}")
        different, invalid = self.run_index()["exports"]
        self.assertEqual(different["comparison"], "different")
        self.assertEqual(invalid["metadata"]["state"], "count_missing_or_invalid")
        self.assertIsNone(invalid["metadata"]["declared"])


if __name__ == "__main__":
    unittest.main()
