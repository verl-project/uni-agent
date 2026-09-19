"""KernelBench Claude Code hook: bind best snapshots and stop stale searches.

The command receives Claude Code's hook JSON on stdin. ``pre`` records the
current transcript assistant count and best snapshot; ``post`` binds an updated
snapshot to its implementation's assistant turn and stops Claude after the
configured number of verifier calls without a best-metric improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

_STATE = ".triton_verify_assistant_pending.json"
_PATIENCE_STATE = ".triton_verify_patience.json"
_STOP = ".triton_verify_stop.json"
_POLICY = ".claude/hooks/triton_verify_policy.json"
_ASSISTANT_FIELDS = (
    "assistant_index",
    "assistant_messages_seen",
    "assistant_index_source",
    "assistant_snapshot_impl_sha256",
)
_VERIFY_COMMAND = re.compile(r"(?:^|[;&|()\s])(?:bash\s+)?(?:\./)?tools/verify_once\.sh(?:\s|$)")


def _payload() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_verifier(payload: dict[str, Any]) -> bool:
    if str(payload.get("tool_name", "")).lower() != "bash":
        return False
    tool_input = payload.get("tool_input")
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    return bool(_VERIFY_COMMAND.search(str(command)))


def _workspace(payload: dict[str, Any]) -> Path:
    # The installed hook owns a fixed workspace even when Bash changed cwd.
    hook_dir = Path(__file__).resolve().parent
    if hook_dir.name == "hooks" and hook_dir.parent.name == ".claude":
        return hook_dir.parent.parent
    cwd = payload.get("cwd") or os.getcwd()
    return Path(str(cwd)).resolve()


def _state_path(workspace: Path, payload: dict[str, Any]) -> Path:
    tool_use_id = payload.get("tool_use_id")
    if not tool_use_id:
        return workspace / _STATE
    suffix = hashlib.sha256(str(tool_use_id).encode()).hexdigest()[:16]
    return workspace / f"{_STATE[:-5]}.{suffix}.json"


def _operator_name(workspace: Path) -> str | None:
    try:
        metadata = json.loads((workspace / "TASK_METADATA.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = metadata.get("op_name") if isinstance(metadata, dict) else None
    return str(value) if value else None


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _fully_correct(metrics: dict[str, Any]) -> bool:
    total = _as_int(metrics.get("total_cases"))
    passed = _as_int(metrics.get("passed_cases"))
    failed = _as_int(metrics.get("failed_cases"))
    return (
        bool(metrics.get("correctness_ok") or metrics.get("success")) and total > 0 and passed == total and failed == 0
    )


def _assistant_count(transcript_path: Any) -> int:
    if not transcript_path:
        return 0
    seen_ids: set[str] = set()
    count = 0
    try:
        lines = Path(str(transcript_path)).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        message_id = str(message.get("id") or "")
        if message_id:
            if message_id in seen_ids:
                continue
            seen_ids.add(message_id)
        count += 1
    return count


def _active_stop(workspace: Path) -> dict[str, Any] | None:
    stop = _read_json(workspace / _STOP)
    if stop and stop.get("reason") in {"no_verify_improvement", "no_latency_improvement"}:
        return stop
    return None


def _stop_response(stop: dict[str, Any], *, pre_tool: bool = False) -> dict[str, Any]:
    response = {"continue": False, "stopReason": str(stop["reason"])}
    if pre_tool:
        response["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": str(stop["reason"]),
        }
    return response


def pre(payload: dict[str, Any]) -> dict[str, Any] | None:
    workspace = _workspace(payload)
    stop = _active_stop(workspace)
    if stop is not None:
        return _stop_response(stop, pre_tool=True)
    if not _is_verifier(payload):
        return None
    op_name = _operator_name(workspace)
    if not op_name:
        return
    messages_seen = _assistant_count(payload.get("transcript_path"))
    best_metrics = _read_json(workspace / "metrics_best.json") or {}
    state = {
        "assistant_messages_seen": messages_seen,
        "assistant_index": messages_seen - 1 if messages_seen > 0 else None,
        "metrics_digest": _digest(workspace / "metrics_best.json"),
        "implementation_digest": _digest(workspace / "src" / f"{op_name}_triton_ascend_impl_best.py"),
        "best_fully_correct": _fully_correct(best_metrics),
        "best_assistant": {key: best_metrics.get(key) for key in _ASSISTANT_FIELDS},
    }
    _atomic_json(_state_path(workspace, payload), state)


def post(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_verifier(payload):
        return None
    workspace = _workspace(payload)
    state_path = _state_path(workspace, payload)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        stop = _active_stop(workspace)
        return _stop_response(stop) if stop is not None else None
    try:
        state_path.unlink(missing_ok=True)
    except OSError:
        pass
    op_name = _operator_name(workspace)
    if not op_name or not isinstance(state, dict):
        return None
    metrics_path = workspace / "metrics_best.json"
    implementation_path = workspace / "src" / f"{op_name}_triton_ascend_impl_best.py"
    current_metrics = _digest(metrics_path)
    current_impl = _digest(implementation_path)
    pair_exists = current_metrics is not None and current_impl is not None
    improved = pair_exists and current_metrics != state.get("metrics_digest")
    if improved:
        assistant_index = state.get("assistant_index")
        if (
            current_impl != state.get("implementation_digest")
            and isinstance(assistant_index, int)
            and assistant_index >= 0
        ):
            annotation = {
                "assistant_index": assistant_index,
                "assistant_messages_seen": assistant_index + 1,
                "assistant_index_source": "claude_code_transcript_hook",
                "assistant_snapshot_impl_sha256": current_impl,
            }
        else:
            previous = state.get("best_assistant")
            annotation = (
                previous
                if isinstance(previous, dict) and previous.get("assistant_snapshot_impl_sha256") == current_impl
                else None
            )
        metrics = _read_json(metrics_path)
        if metrics is not None and annotation is not None:
            metrics.update(annotation)
            _atomic_json(metrics_path, metrics)

    stop = _update_patience(
        workspace,
        improved=improved,
        had_fully_correct_best=state.get("best_fully_correct") is True,
    )
    if stop is None:
        return None
    return _stop_response(stop)


def _update_patience(
    workspace: Path,
    *,
    improved: bool,
    had_fully_correct_best: bool,
) -> dict[str, Any] | None:
    # A decision is terminal for this fresh-workspace attempt. Late completions
    # may bind their snapshot, but cannot resume the search or rewrite patience.
    stopped = _active_stop(workspace)
    if stopped is not None:
        return stopped
    policy = _read_json(workspace / _POLICY) or {}
    state_path = workspace / _PATIENCE_STATE
    state = _read_json(state_path) or {}
    best = _read_json(workspace / "metrics_best.json") or {}
    mode = "latency" if _fully_correct(best) else "correctness"
    stale_key = f"{mode}_stale_verify_count"
    reset = improved or (mode == "latency" and not had_fully_correct_best)
    stale = 0 if reset else _as_int(state.get(stale_key)) + 1
    verify_count = _as_int(state.get("verify_count")) + 1
    next_state = {
        "verify_count": verify_count,
        "mode": mode,
        "correctness_stale_verify_count": stale
        if mode == "correctness"
        else _as_int(state.get("correctness_stale_verify_count")),
        "latency_stale_verify_count": stale if mode == "latency" else _as_int(state.get("latency_stale_verify_count")),
    }
    _atomic_json(state_path, next_state)

    stop_path = workspace / _STOP
    patience = _as_int(policy.get(f"{mode}_patience"))
    if patience <= 0 or stale < patience:
        return None
    if mode == "correctness" and _as_float(best.get("reward")) < _as_float(policy.get("correctness_min_reward")):
        return None
    stop = {
        "reason": "no_latency_improvement" if mode == "latency" else "no_verify_improvement",
        "mode": mode,
        "patience": patience,
        "stale_verify_count": stale,
        "verify_count": verify_count,
    }
    _atomic_json(stop_path, stop)
    return stop


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    payload = _payload()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    result = None
    if mode == "pre":
        result = pre(payload)
    elif mode == "post":
        result = post(payload)
    if result is not None:
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
