"""CPU tests for strict OpenClaw trajectory evidence (stdlib runner)."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[3] / "uni_agent/agents/openclaw/trajectory.py"
spec = importlib.util.spec_from_file_location("openclaw_trajectory", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = sqlite3.connect(Path(self.tmp.name) / "agent.sqlite")
        self.addCleanup(self.db.close)
        self.db.executescript("""
        CREATE TABLE session_windows(session_id TEXT, previous_session_id TEXT, reason TEXT);
        INSERT INTO session_windows VALUES ('s',NULL,NULL);
        CREATE TABLE session_nodes(current_session_id TEXT, parent_session_key TEXT,
          spawned_by TEXT, fork_source_session_id TEXT, fork_source_session_key TEXT);
        INSERT INTO session_nodes VALUES ('s',NULL,NULL,NULL,NULL);
        CREATE TABLE transcript_events(session_id TEXT, seq INTEGER, event_json TEXT);
        """)
        self.events = [
            {"type": "session", "id": "s"},
            {"type": "message", "id": "u", "parentId": None,
             "message": {"role": "user", "content": "solve"}},
            {"type": "message", "id": "a", "parentId": "u", "message": {
                "role": "assistant", "provider": "vllm", "model": "policy",
                "stopReason": "stop", "content": [{"type": "text", "text": "answer"}],
                "__openclaw": {"runId": "r"}}},
        ]

    def audit(self):
        self.db.execute("DELETE FROM transcript_events")
        self.db.executemany("INSERT INTO transcript_events VALUES ('s',?,?)",
                            [(i, json.dumps(e)) for i, e in enumerate(self.events)])
        self.db.commit()
        return module.audit_state(self.tmp.name, "s", "policy")

    def test_complete_chain(self):
        self.assertTrue(self.audit()["verified"])

    def test_empty_evidence(self):
        self.events = []
        self.assertFalse(self.audit()["verified"])

    def test_parent_branch(self):
        self.events[-1]["parentId"] = None
        self.assertFalse(self.audit()["verified"])

    def test_compaction(self):
        self.events.insert(2, {"type": "compaction", "id": "c", "parentId": "u"})
        self.events[-1]["parentId"] = "c"
        self.assertFalse(self.audit()["verified"])

    def test_second_window(self):
        self.db.execute("INSERT INTO session_windows VALUES ('other','s','recovery')")
        self.assertFalse(self.audit()["verified"])

    def test_model_fallback(self):
        self.events[-1]["message"]["model"] = "other"
        self.assertFalse(self.audit()["verified"])

    def test_missing_tool_result(self):
        self.events[-1]["message"]["content"] = [{"type": "toolCall", "id": "t", "name": "exec"}]
        self.assertFalse(self.audit()["verified"])

    def test_second_run(self):
        second = copy.deepcopy(self.events[-1])
        second.update(id="b", parentId="a")
        second["message"]["__openclaw"]["runId"] = "other"
        self.events.append(second)
        self.assertFalse(self.audit()["verified"])

    def test_truncated_completion(self):
        self.events[-1]["message"]["stopReason"] = "length"
        self.assertFalse(self.audit()["verified"])

    def test_orphan_result(self):
        self.events[-1]["message"] = {"role": "toolResult", "toolCallId": "unknown"}
        self.assertFalse(self.audit()["verified"])


if __name__ == "__main__":
    unittest.main()
