"""Translate Harbor verifier output into Uni-Agent rewards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import TaskResult

_OUTPUT_TAIL_CHARS = 8000


def _primary_reward(rewards: Any) -> tuple[float | None, str | None]:
    if not isinstance(rewards, dict) or not rewards:
        return None, "Harbor trial produced no verifier rewards"

    if "reward" in rewards:
        value = rewards["reward"]
        key = "reward"
    elif len(rewards) == 1:
        key, value = next(iter(rewards.items()))
    else:
        return None, "Harbor verifier returned multiple metrics without a primary 'reward' key"

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, f"Harbor verifier reward {key!r} is not numeric"
    return float(value), None


def _exception_summary(exception: Any) -> dict[str, Any] | None:
    if not isinstance(exception, dict) or not exception:
        return None
    return {
        key: exception.get(key)
        for key in ("exception_type", "exception_message", "occurred_at")
        if exception.get(key) is not None
    }


def _step_summaries(steps: Any) -> list[dict[str, Any]] | None:
    if not isinstance(steps, list):
        return None

    summaries = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        verifier_result = step.get("verifier_result")
        summaries.append(
            {
                "step_name": step.get("step_name"),
                "rewards": verifier_result.get("rewards") if isinstance(verifier_result, dict) else None,
                "exception": _exception_summary(step.get("exception_info")),
            }
        )
    return summaries


def task_result_from_harbor_trial(
    payload: dict[str, Any],
    *,
    trial_dir: Path,
    cli_exit_code: int,
    stdout: str,
    stderr: str,
    elapsed: float,
) -> TaskResult:
    """Translate Harbor's persisted TrialResult into Uni-Agent's result shape."""
    verifier_result = payload.get("verifier_result")
    rewards = verifier_result.get("rewards") if isinstance(verifier_result, dict) else None
    score, reward_error = _primary_reward(rewards)
    exception = _exception_summary(payload.get("exception_info"))

    error: str | None = None
    if cli_exit_code != 0:
        detail = (stderr or stdout).strip()[-2000:]
        error = f"Harbor CLI exited with code {cli_exit_code}" + (f": {detail}" if detail else "")
    elif exception is not None:
        exception_type = exception.get("exception_type", "HarborTrialError")
        exception_message = exception.get("exception_message", "trial failed")
        error = f"{exception_type}: {exception_message}"
    elif reward_error is not None:
        error = reward_error

    completed = error is None and score is not None
    scalar_reward = score if completed and score is not None else 0.0
    resolved = completed and scalar_reward == 1.0
    timings = {
        key: payload.get(key)
        for key in (
            "started_at",
            "finished_at",
            "environment_setup",
            "agent_setup",
            "agent_execution",
            "verifier",
        )
        if payload.get(key) is not None
    }

    eval_report = {
        "trial_id": payload.get("id"),
        "trial_name": payload.get("trial_name"),
        "trial_uri": payload.get("trial_uri"),
        "trial_path": str(trial_dir),
        "artifact_path": str(trial_dir / "artifacts"),
        "cli_exit_code": cli_exit_code,
        "rewards": rewards,
        "exception": exception,
        "steps": _step_summaries(payload.get("step_results")),
        "timings": timings,
        "stdout_tail": stdout[-_OUTPUT_TAIL_CHARS:],
        "stderr_tail": stderr[-_OUTPUT_TAIL_CHARS:],
        "error": error,
    }
    extra_info = {
        "resolved": resolved,
        "eval_completed": completed,
        "eval_execution_time": elapsed,
        "eval_report": eval_report,
        "agent": payload.get("agent_info"),
    }
    return TaskResult(
        reward=scalar_reward,
        accuracy=scalar_reward,
        finished=None,
        extra_info=extra_info,
    )
