"""Smoke tests for the evaluation harness. All scenes and answers here are invented."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import evaluate as ev


class HarnessTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.suite = self.base / "suite"
        self.case = self.suite / "cases" / "twins"
        (self.case / "scene").mkdir(parents=True)
        (self.case / "scene/a").write_text("original A")
        (self.case / "scene/b").write_text("original B")
        (self.case / "prompt.txt").write_text("Which artifact produced this event?")
        self.spec = {
            "kind": "smoke",
            "provenance": "invented unit test",
            "disposition": "ambiguous",
            "candidates": ["a", "b"],
            "facts": {"declared": None},
            "evidence": ["a", "b"],
            "next_step_rubric": "2: request event id; 1: ask generally; 0: guess.",
        }
        ev.save(self.case / "rubric.json", self.spec)
        ev.save(
            self.case / "answer-schema.json",
            {"fact_types": {"declared": "integer"}, "candidate_format": "scene-relative path"},
        )
        (self.suite / "package/scripts").mkdir(parents=True)
        (self.suite / "package/SKILL.md").write_text("test guide")
        (self.suite / "package/scripts/check.py").write_text("pass")
        (self.suite / "protocol.md").write_text("Smoke only; not effectiveness evidence.")
        self.lock = ev.frozen(self.suite)
        self.out = self.base / "runs"
        self.answer = {
            "disposition": "ambiguous",
            "candidates": ["b", "a"],
            "facts": {"declared": None},
            "evidence": ["b", "a"],
            "next_step": "Request the event id.",
        }
        self.adapter = self.base / "adapter.py"
        self.adapter.write_text("import json\nprint(" + repr(json.dumps(self.answer)) + ")")
        self.cfg = {
            "command": [sys.executable, str(self.adapter)],
            "rounds": 1,
            "timeout_seconds": 5,
            "model": "smoke",
            "seed": 5,
        }

    def test_timestamped_findings_use_actual_reader_and_frozen_oracle(self):
        reader = Path(ev.__file__).parent / "adapters/reader.py"
        self.cfg["measure_evidence"] = True
        # Use the real MCP reader protocol with a wrong finding followed by a correct one.
        ticket = {**self.answer, "candidates": ["third"]}
        self.adapter.write_text(
            "import json, os, sys, subprocess\n"
            + f"answers = {[ticket, self.answer]!r}\n"
            + "calls = [dict(jsonrpc='2.0', id=i,method='tools/call',"
            + "params=dict(name='submit_evidence',arguments=dict(finding=a))) for i,a in enumerate(answers)]\n"
            + f"p = subprocess.run([sys.executable, {str(reader)!r}, '.', '--journal', "
            + "os.environ['SKILL_EVAL_JOURNAL'], '--start-ns', os.environ['SKILL_EVAL_START_NS']],"
            + "input='\\n'.join(map(json.dumps,calls)), capture_output=True, text=True, check=True)\n"
            + "assert p.stdout.count('Recorded. No correctness feedback.') == 2\n"
            + "print(json.dumps(answers[1]))\n"
        )
        ev.run(self.suite, self.lock, self.out, self.cfg)
        rows = ev.replay(self.suite, self.out)
        for i, row in enumerate(rows):
            path = self.out / f"{i:04}" / "findings.jsonl"
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(row["seconds_first_correct_evidence_submission"], entries[1]["ns"] / 1e9)
        self.assertEqual(ev.summarize(rows)[0]["trials_with_correct_evidence_submission"], 1)
        path.write_text("")
        with self.assertRaisesRegex(ValueError, "checkpoint journal changed"):
            ev.replay(self.suite, self.out)

    def test_replay_three_arms_and_blind_grading(self):
        ev.run(self.suite, self.lock, self.out, self.cfg)
        results = ev.replay(self.suite, self.out)
        self.assertEqual(results, ev.load(self.out / "results.json"))
        self.assertTrue(all(r["fact_pass"] for r in results))
        for i, row in enumerate(results):
            paths = ev.load(self.out / f"{i:04d}" / "input.json")["files"]
            self.assertEqual("package/scripts/check.py" in paths, row["arm"] != "baseline")
            self.assertEqual("package/SKILL.md" in paths, row["arm"] == "skill")
            self.assertFalse(any("rubric" in file for file in paths))
        packet = self.base / "review"
        ev.review(self.suite, self.out, packet)
        questions = ev.load(packet / "questions.json")
        self.assertTrue(all("arm" not in q for q in questions))
        votes = self.base / "votes.json"
        ev.save(votes, {q["id"]: {"score": 2, "reason": "asks for discriminating event id"} for q in questions})
        self.assertTrue(all(r["next_step_score"] == 2 for r in ev.ratings(self.suite, self.out, votes)))
        (self.case / "scene/a").write_text("changed")
        self.assertRaises(ValueError, ev.replay, self.suite, self.out)

    def test_failure_and_incomplete_run_not_dropped(self):
        self.adapter.write_text("raise SystemExit(2)")
        ev.run(self.suite, self.lock, self.out, self.cfg)
        (self.out / "0000/receipt.json").unlink()
        rows = ev.replay(self.suite, self.out)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["error"] and not r["fact_pass"] for r in rows))
        self.assertFalse(any(r["wrong_identification"] for r in rows))

    def test_public_contract_and_fabricated_evidence(self):
        self.spec["facts"]["event"] = "private-value"
        ev.save(self.case / "rubric.json", self.spec)
        ev.save(
            self.case / "answer-schema.json",
            {"fact_types": {"declared": "integer", "event": "string"}, "candidate_format": "scene-relative path"},
        )
        ev.run(self.suite, ev.frozen(self.suite), self.out, self.cfg)
        for i in range(3):
            prompt = ev.load(self.out / f"{i:04d}" / "input.json")["prompt"]
            self.assertIn('"event": "string"', prompt)
            self.assertIn("scene-relative path", prompt)
            self.assertNotIn("private-value", prompt)
        full = dict(self.answer, facts={"declared": None, "event": "private-value"})
        bogus = dict(full, evidence=["a", "b", "missing"])
        self.assertFalse(ev.score(self.spec, bogus, {"a", "b"})["fact_pass"])
        extra = dict(full, facts={**full["facts"], "imagined": True})
        self.assertFalse(ev.score(self.spec, extra, {"a", "b"})["fact_pass"])

    def test_artifact_two_arms_verify_saved_bytes_not_self_claim(self):
        (self.suite / "package/scripts/check.py").unlink()
        (self.suite / "package/scripts").rmdir()
        (self.case / "answer-schema.json").unlink()
        ev.save(
            self.case / "rubric.json",
            {
                "profile": "artifact",
                "kind": "smoke",
                "provenance": "synthetic",
                "outputs": {"config.json": {"retries": 7}},
            },
        )
        (self.case / "prompt.txt").write_text("Create config.json using the retry count in the inputs.")
        self.adapter.write_text(
            "import pathlib\nif pathlib.Path('package/SKILL.md').exists():\n"
            "    pathlib.Path('output/config.json').write_text('{\"retries\":7}')\n"
            "print('{\"success\":true}')"
        )
        lock = ev.frozen(self.suite)
        self.assertEqual(lock["arms"], ["baseline", "skill"])
        ev.run(self.suite, lock, self.out, self.cfg)
        reports = ev.replay(self.suite, self.out)
        self.assertEqual({r["arm"]: r["task_pass"] for r in reports}, {"baseline": False, "skill": True})
        self.assertNotIn("identified", ev.load(self.out / "0000/input.json")["prompt"])
        self.assertTrue(all(r["wrong_identifications"] is None for r in ev.summarize(reports)))
        archived = next(self.out.glob("*/artifacts/config.json"))
        archived.write_text("{}")
        self.assertRaisesRegex(ValueError, "Archived output changed", ev.replay, self.suite, self.out)

    def test_wrong_id_null_and_ambiguity(self):
        self.assertTrue(ev.score(self.spec, self.answer, {"a", "b"})["fact_pass"])
        wrong = dict(self.answer, disposition="identified", candidates=["a"])
        self.assertTrue(ev.score(self.spec, wrong, {"a", "b"})["wrong_identification"])
        self.assertFalse(ev.score(self.spec, wrong, {"a", "b"})["fact_pass"])
        self.assertFalse(ev.score(self.spec, dict(self.answer, facts={}), {"a", "b"})["fact_pass"])
        self.assertFalse(ev.score(self.spec, dict(self.answer, evidence=["a"]), {"a", "b"})["fact_pass"])


if __name__ == "__main__":
    unittest.main()
