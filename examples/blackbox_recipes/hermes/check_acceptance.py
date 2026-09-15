#!/usr/bin/env python3
"""Machine-check the one-sample Hermes SWE-bench acceptance contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, ok: bool, detail: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--expected-reward", type=float, default=1.0)
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    aggregate_path = root / "result.json"
    metadata_path = root / "sample_metadata.json"
    aggregate: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    if aggregate_path.is_file():
        try:
            loaded = _load(aggregate_path)
            is_object = isinstance(loaded, dict)
            aggregate = loaded if is_object else {}
            _check("aggregate-json", is_object, "result.json is an object", checks)
        except (OSError, json.JSONDecodeError) as exc:
            _check("aggregate-json", False, f"cannot parse result.json: {exc}", checks)
    else:
        _check("aggregate-json", False, "missing result.json", checks)
    if metadata_path.is_file():
        try:
            loaded = _load(metadata_path)
            metadata = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            metadata = {}
    expected_instance = args.instance_id or metadata.get("instance_id")

    _check("num-prompts", aggregate.get("num_prompts") == 1, f"num_prompts={aggregate.get('num_prompts')!r}", checks)
    _check(
        "num-scored-sessions",
        aggregate.get("num_scored_sessions") == 1,
        f"num_scored_sessions={aggregate.get('num_scored_sessions')!r}",
        checks,
    )
    scores = aggregate.get("scores")
    _check("scores-shape", isinstance(scores, list) and len(scores) == 1, f"scores={scores!r}", checks)
    score = scores[0] if isinstance(scores, list) and len(scores) == 1 else None
    _check("reward-one", score == args.expected_reward, f"score={score!r}", checks)

    task_logs = list((root / "logs").rglob("task.log")) if (root / "logs").exists() else []
    _check("one-task-log", len(task_logs) == 1, f"task_logs={len(task_logs)}", checks)
    task_log_text = task_logs[0].read_text(encoding="utf-8", errors="replace") if len(task_logs) == 1 else ""
    _check("verifier-resolved", "resolved=True" in task_log_text, "task log contains resolved=True", checks)
    _check("agent-finished", "finished=True" in task_log_text, "task log contains finished=True", checks)

    trajectory_paths = sorted((root / "logs").rglob("trajectory.json")) if (root / "logs").exists() else []
    trajectory_meta: dict[str, Any] = {}
    if len(trajectory_paths) == 1:
        try:
            loaded = _load(trajectory_paths[0])
            trajectory_meta = loaded if isinstance(loaded, dict) else {}
            _check("trajectory-json", isinstance(loaded, dict), "trajectory.json is an object", checks)
        except (OSError, json.JSONDecodeError) as exc:
            _check("trajectory-json", False, f"cannot parse trajectory.json: {exc}", checks)
    _check("one-trajectory-file", len(trajectory_paths) == 1, f"trajectory_files={len(trajectory_paths)}", checks)
    trajectory_items = trajectory_meta.get("trajectories") if isinstance(trajectory_meta, dict) else None
    _check(
        "one-trajectory",
        trajectory_meta.get("num_trajectories") == 1
        and isinstance(trajectory_items, list)
        and len(trajectory_items) == 1,
        f"num_trajectories={trajectory_meta.get('num_trajectories')!r}",
        checks,
    )
    item = trajectory_items[0] if isinstance(trajectory_items, list) and len(trajectory_items) == 1 else {}
    _check("trajectory-finished", item.get("finished") is True, f"finished={item.get('finished')!r}", checks)
    _check(
        "trajectory-reward",
        item.get("reward_score") == args.expected_reward,
        f"reward_score={item.get('reward_score')!r}",
        checks,
    )

    npz_paths = sorted((root / "logs").rglob("trajectory.npz")) if (root / "logs").exists() else []
    arrays_ok = False
    arrays_detail = "missing trajectory.npz"
    if len(npz_paths) == 1:
        try:
            import numpy as np

            with np.load(npz_paths[0], allow_pickle=False) as arrays:
                required = {"traj0_prompt_ids", "traj0_response_ids", "traj0_response_mask", "traj0_response_logprobs"}
                missing = sorted(required - set(arrays.files))
                lengths = {
                    key: int(arrays[key].shape[0])
                    for key in ("traj0_response_ids", "traj0_response_mask", "traj0_response_logprobs")
                    if key in arrays
                }
                arrays_ok = (
                    not missing and len(set(lengths.values())) == 1 and all(value > 0 for value in lengths.values())
                )
                arrays_detail = f"missing={missing}, lengths={lengths}"
        except Exception as exc:
            arrays_detail = f"cannot inspect trajectory.npz: {type(exc).__name__}: {exc}"
    _check("trajectory-arrays", len(npz_paths) == 1 and arrays_ok, arrays_detail, checks)

    hermes_results = sorted((root / "hermes").rglob("result.json")) if (root / "hermes").exists() else []
    runner_result: dict[str, Any] = {}
    if len(hermes_results) == 1:
        try:
            loaded = _load(hermes_results[0])
            runner_result = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            runner_result = {}
    _check("one-hermes-result", len(hermes_results) == 1, f"hermes_results={len(hermes_results)}", checks)
    _check(
        "hermes-completed",
        runner_result.get("status") == "completed" and runner_result.get("finished") is True,
        f"status={runner_result.get('status')!r}, finished={runner_result.get('finished')!r}",
        checks,
    )
    if expected_instance:
        _check(
            "instance-id-recorded",
            isinstance(expected_instance, str) and bool(expected_instance),
            f"instance_id={expected_instance!r}",
            checks,
        )

    passed = all(check["ok"] for check in checks)
    report = {
        "schema_version": 1,
        "passed": passed,
        "instance_id": expected_instance,
        "num_prompts": aggregate.get("num_prompts"),
        "num_scored_sessions": aggregate.get("num_scored_sessions"),
        "scores": scores,
        "session_id": trajectory_meta.get("session_id"),
        "trajectory_count": trajectory_meta.get("num_trajectories"),
        "finished": item.get("finished"),
        "reward": score,
        "verifier_resolved": "resolved=True" in task_log_text,
        "paths": {
            "result": str(aggregate_path),
            "task_log": str(task_logs[0]) if task_logs else None,
            "trajectory": str(trajectory_paths[0]) if trajectory_paths else None,
            "hermes_result": str(hermes_results[0]) if hermes_results else None,
        },
        "checks": checks,
    }
    (root / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    handoff = [
        "# Hermes recipe handoff",
        "",
        f"- 通过：`{passed}`",
        f"- instance_id：`{expected_instance or 'unknown'}`",
        f"- reward：`{score}`",
        f"- session：`{trajectory_meta.get('session_id')}`",
        f"- 检查项：`{sum(check['ok'] for check in checks)}/{len(checks)}`",
        "",
        "详见 `acceptance.json`；训练更新不在本首版范围内。",
    ]
    (root / "handoff.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "instance_id": expected_instance, "score": score}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
