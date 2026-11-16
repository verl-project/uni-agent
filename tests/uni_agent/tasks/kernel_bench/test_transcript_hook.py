from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uni_agent.tasks.kernel_bench.track_verify_snapshot import post, pre

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def hook_payload(workspace: Path, transcript: Path) -> dict:
    return {
        "tool_name": "Bash",
        "tool_input": {"command": "bash tools/verify_once.sh smoke"},
        "tool_use_id": "toolu_test",
        "cwd": str(workspace),
        "transcript_path": str(transcript),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_hook_annotates_only_a_changed_best_pair(tmp_path: Path) -> None:
    write_json(tmp_path / "TASK_METADATA.json", {"op_name": "smoke"})
    transcript = tmp_path / "transcript.jsonl"
    events = [
        {"type": "assistant", "message": {"id": "a1", "content": []}},
        {"type": "assistant", "message": {"id": "a1", "content": []}},
        {"type": "assistant", "message": {"id": "a2", "content": []}},
    ]
    transcript.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    best_metrics = tmp_path / "metrics_best.json"
    best_impl = tmp_path / "src/smoke_triton_ascend_impl_best.py"
    write_json(best_metrics, {"correctness_ok": True, "speedup": 1.0})
    best_impl.parent.mkdir(parents=True)
    best_impl.write_text("# old\n", encoding="utf-8")
    payload = hook_payload(tmp_path, transcript)

    pre(payload)
    write_json(best_metrics, {"correctness_ok": True, "speedup": 1.2})
    best_impl.write_text("# new\n", encoding="utf-8")
    post(payload)

    annotated = json.loads(best_metrics.read_text(encoding="utf-8"))
    assert annotated["assistant_index"] == 1
    assert annotated["assistant_messages_seen"] == 2
    assert annotated["assistant_index_source"] == "claude_code_transcript_hook"
    assert len(annotated["assistant_snapshot_impl_sha256"]) == 64


@pytest.mark.parametrize("change", ["neither", "metrics", "implementation"])
def test_hook_does_not_relabel_without_both_changes(
    tmp_path: Path,
    change: str,
) -> None:
    write_json(tmp_path / "TASK_METADATA.json", {"op_name": "smoke"})
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "assistant", "message": {"id": "a1", "content": []}}) + "\n",
        encoding="utf-8",
    )
    best_metrics = tmp_path / "metrics_best.json"
    best_impl = tmp_path / "src/smoke_triton_ascend_impl_best.py"
    write_json(best_metrics, {"speedup": 1.0})
    best_impl.parent.mkdir(parents=True)
    best_impl.write_text("# old\n", encoding="utf-8")
    payload = hook_payload(tmp_path, transcript)

    pre(payload)
    if change == "metrics":
        write_json(best_metrics, {"speedup": 1.1})
    elif change == "implementation":
        best_impl.write_text("# new\n", encoding="utf-8")
    post(payload)

    assert "assistant_index" not in json.loads(best_metrics.read_text(encoding="utf-8"))


def test_correctness_patience_respects_minimum_reward_and_resets(tmp_path: Path) -> None:
    write_json(tmp_path / "TASK_METADATA.json", {"op_name": "smoke"})
    write_json(
        tmp_path / ".claude/hooks/triton_verify_policy.json",
        {"correctness_patience": 2, "correctness_min_reward": 0.15, "latency_patience": 2},
    )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    best_metrics = tmp_path / "metrics_best.json"
    best_impl = tmp_path / "src/smoke_triton_ascend_impl_best.py"
    write_json(
        best_metrics, {"correctness_ok": False, "total_cases": 2, "passed_cases": 1, "failed_cases": 1, "reward": 0.1}
    )
    best_impl.parent.mkdir(parents=True)
    best_impl.write_text("# implementation\n", encoding="utf-8")
    payload = hook_payload(tmp_path, transcript)

    for _ in range(2):
        pre(payload)
        assert post(payload) is None

    pre(payload)
    write_json(
        best_metrics, {"correctness_ok": False, "total_cases": 2, "passed_cases": 1, "failed_cases": 1, "reward": 0.2}
    )
    assert post(payload) is None
    pre(payload)
    assert post(payload) is None
    pre(payload)
    assert post(payload) == {"continue": False, "stopReason": "no_verify_improvement"}


def test_latency_patience_preserves_same_impl_best_hint(tmp_path: Path) -> None:
    write_json(tmp_path / "TASK_METADATA.json", {"op_name": "smoke"})
    write_json(
        tmp_path / ".claude/hooks/triton_verify_policy.json",
        {"correctness_patience": 2, "correctness_min_reward": 0.15, "latency_patience": 2},
    )
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("", encoding="utf-8")
    best_metrics = tmp_path / "metrics_best.json"
    best_impl = tmp_path / "src/smoke_triton_ascend_impl_best.py"
    best_impl.parent.mkdir(parents=True)
    best_impl.write_text("# implementation\n", encoding="utf-8")
    digest = hashlib.sha256(best_impl.read_bytes()).hexdigest()
    hint = {
        "assistant_index": 2,
        "assistant_messages_seen": 3,
        "assistant_index_source": "claude_code_transcript_hook",
        "assistant_snapshot_impl_sha256": digest,
    }
    write_json(
        best_metrics,
        {"correctness_ok": True, "total_cases": 2, "passed_cases": 2, "failed_cases": 0, "reward": 0.7, **hint},
    )
    payload = hook_payload(tmp_path, transcript)

    pre(payload)
    write_json(
        best_metrics,
        {"correctness_ok": True, "total_cases": 2, "passed_cases": 2, "failed_cases": 0, "reward": 0.8},
    )
    assert post(payload) is None
    assert {key: json.loads(best_metrics.read_text(encoding="utf-8"))[key] for key in hint} == hint

    pre(payload)
    assert post(payload) is None
    pre(payload)
    assert post(payload) == {"continue": False, "stopReason": "no_latency_improvement"}
