"""Fail-closed audit of OpenClaw 2026.9.2 per-episode SQLite transcripts.

Run after the CLI exits, before sandbox teardown. This proves recorded trajectory
structure, not task correctness or absence of unrecorded transport retries.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3


def audit_state(state_dir: str, session_id: str, model: str, export_path: str | None = None) -> dict:
    errors = []
    databases = []
    windows, nodes, records = [], [], []
    try:
        for path in sorted(Path(state_dir).rglob("*.sqlite")):
            # ``sqlite3.Connection``'s context manager commits/rolls back but
            # does not close the connection.  Explicitly close it so the
            # temporary SQLite file can be removed on Windows after auditing.
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                db.row_factory = sqlite3.Row
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "transcript_events" not in tables:
                    continue
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    errors.append("database_integrity")
                databases.append(str(path))
                windows.extend(dict(r) for r in db.execute("SELECT * FROM session_windows"))
                nodes.extend(dict(r) for r in db.execute("SELECT * FROM session_nodes"))
                records.extend(dict(r) for r in db.execute("SELECT * FROM transcript_events ORDER BY seq"))
        if len(databases) != 1 or len(windows) != 1 or len(nodes) != 1:
            errors.append("expected_one_database_window_node")
        for window in windows:
            if window["session_id"] != session_id or window["previous_session_id"] is not None:
                errors.append("session_replaced")
            if window["reason"] not in (None, "initial"):
                errors.append("session_recovery_or_branch")
        for node in nodes:
            if node["current_session_id"] != session_id:
                errors.append("node_session_mismatch")
            if any(node.get(k) for k in ("parent_session_key", "spawned_by", "fork_source_session_id", "fork_source_session_key")):
                errors.append("spawn_or_fork")
        if [r["seq"] for r in records] != list(range(len(records))):
            errors.append("sequence_gap")
        if any(r["session_id"] != session_id for r in records):
            errors.append("multiple_sessions")
        events = [json.loads(r["event_json"]) for r in records]
        if not events or events[0].get("type") != "session" or events[0].get("id") != session_id:
            errors.append("missing_session_header")
        parent = None
        ids = set()
        messages = []
        pending = set()
        calls = set()
        runs = set()
        for event in events[1:]:
            event_id = event.get("id")
            if not event_id or event_id in ids or event.get("parentId") != parent:
                errors.append("broken_or_branched_parent_chain")
            ids.add(event_id)
            parent = event_id
            kind = event.get("type")
            if kind not in {"model_change", "thinking_level_change", "custom", "message"}:
                errors.append("unsupported_or_recovery_event:" + str(kind))
            if kind == "custom" and event.get("customType") != "model-snapshot":
                errors.append("unknown_custom_event")
            if kind == "model_change" or (kind == "custom" and event.get("customType") == "model-snapshot"):
                source = event if kind == "model_change" else event["data"]
                if source.get("provider") != "vllm" or source.get("modelId") != model:
                    errors.append("model_changed")
            if kind != "message":
                continue
            message = event["message"]
            messages.append(message)
            role = message.get("role")
            run = message.get("__openclaw", {}).get("runId")
            if run:
                runs.add(run)
            if role == "assistant":
                if pending:
                    errors.append("missing_tool_result")
                if message.get("provider") != "vllm" or message.get("model") != model:
                    errors.append("assistant_model_mismatch")
                if message.get("stopReason") not in {"stop", "toolUse"}:
                    errors.append("assistant_error_or_truncation")
                for block in message.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "toolCall":
                        call_id = block.get("id")
                        if not call_id or call_id in calls or block.get("name") != "exec":
                            errors.append("duplicate_or_disallowed_tool")
                        calls.add(call_id)
                        pending.add(call_id)
            elif role == "toolResult":
                if message.get("toolCallId") not in pending:
                    errors.append("orphan_tool_result")
                pending.discard(message.get("toolCallId"))
            elif role != "user":
                errors.append("unexpected_message_role")
        if len(runs) != 1:
            errors.append("expected_one_run")
        if sum(m.get("role") == "user" for m in messages) != 1:
            errors.append("expected_one_user_task")
        if not messages or messages[0].get("role") != "user":
            errors.append("missing_initial_user")
        if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("stopReason") != "stop" or pending:
            errors.append("unfinished_trajectory")
        if export_path is not None:
            with Path(export_path).open("x", encoding="utf-8") as stream:
                json.dump({"session_id": session_id, "events": events}, stream, ensure_ascii=False)
        digest = hashlib.sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"verified": not errors, "errors": sorted(set(errors)), "session_id": session_id,
                "event_count": len(events), "message_count": len(messages), "tool_calls": len(calls),
                "run_ids": sorted(runs), "events_sha256": digest, "databases": databases}
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        return {"verified": False, "errors": ["audit_failed:" + type(exc).__name__], "session_id": session_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dir")
    parser.add_argument("session_id")
    parser.add_argument("model")
    parser.add_argument("--export")
    args = parser.parse_args()
    result = audit_state(args.state_dir, args.session_id, args.model, args.export)
    print(json.dumps(result))
    raise SystemExit(0 if result["verified"] else 1)
