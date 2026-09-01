"""Map uni-agent metrics onto verl trainer metric keys.

Collection stays on dotted internal names. This module is the shared format
exit: training wandb keys and eval console/JSONL keys are the same.

Zero verl imports. The trainer feeds timing stages into ``timing_raw`` so
``compute_timing_metrics`` can add the ``timing_s/`` prefix; :func:`aggregate`
applies that prefix itself for eval and other local sinks.
"""

from __future__ import annotations

from typing import Any

from .collector import Metric

# Internal collector name -> timing_raw stage (trainer then prefixes timing_s/).
_TIMING_STAGES: dict[str, str] = {
    "task.total_s": "task/total",
    "task.generate_s": "task/generate",
    "reward.s": "task/reward",
    "env.start_s": "task/env_start",
    "gateway.request_s": "gateway/request",
    "gateway.encode_s": "gateway/encode",
    "gateway.backend_generate_s": "gateway/backend_generate",
    "gateway.decode_s": "gateway/decode",
    "agent.run_s": "agent/run",
}

# Counters are not trainer-stage timings; they land on metrics as-is.
_COUNTER_KEYS: dict[str, str] = {
    "gateway.requests": "gateway/requests",
}


def _unwrap(value: Any) -> Any:
    return getattr(value, "data", value)


def _as_metrics(value: Any) -> dict[str, dict[str, Any]]:
    value = _unwrap(value)
    if not isinstance(value, dict) or not value:
        return {}
    return {name: entry for name, entry in value.items() if isinstance(entry, dict) and "values" in entry}


def _task_scalars(metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for name, entry in metrics.items():
        metric = Metric.from_dict(entry)
        if metric.values:
            scalars[name] = float(metric.aggregate())
    return scalars


def _mean_max(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), max(values)


def reduce_agent_metrics(
    metrics: list[Any],
    mask: Any | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Reduce trajectory ``agent_metrics`` rows to trainer timing + scalar dicts.

    Empty metrics (non-last trajectories in a multi-chain session) are skipped
    so session-level timings are not double-counted. ``mask`` is a boolean
    sequence aligned with ``metrics`` (``False`` = padding).

    Returns ``(timing_raw, metrics)``. Timing keys have no ``timing_s/`` prefix.
    """
    task_rows: list[dict[str, float]] = []
    for index, raw in enumerate(metrics):
        if mask is not None and not bool(mask[index]):
            continue
        scalars = _task_scalars(_as_metrics(raw))
        if scalars:
            task_rows.append(scalars)
    if not task_rows:
        return {}, {}

    grouped: dict[str, list[float]] = {}
    for row in task_rows:
        for name, value in row.items():
            grouped.setdefault(name, []).append(value)

    timing_raw: dict[str, float] = {}
    metrics: dict[str, float] = {}
    for name, values in grouped.items():
        mean_value, max_value = _mean_max(values)
        stage = _TIMING_STAGES.get(name)
        if stage is not None:
            timing_raw[f"{stage}/mean"] = mean_value
            timing_raw[f"{stage}/max"] = max_value
            continue
        counter = _COUNTER_KEYS.get(name)
        if counter is not None:
            metrics[f"{counter}/mean"] = mean_value
            metrics[f"{counter}/max"] = max_value

    slowest = max(task_rows, key=lambda row: row.get("task.total_s", float("-inf")))
    for name, value in slowest.items():
        stage = _TIMING_STAGES.get(name)
        if stage is not None:
            metrics[f"agent_loop/slowest/{stage}"] = value

    return timing_raw, metrics


def aggregate(metrics: list[Any]) -> dict[str, float]:
    """Eval/local sink: same keys the trainer logs after ``compute_timing_metrics``."""
    timing_raw, scalars = reduce_agent_metrics(metrics)
    out = {f"timing_s/{name}": value for name, value in timing_raw.items()}
    out.update(scalars)
    return out


def reduce_prompt_summaries(summaries: list[Any]) -> dict[str, float]:
    """Reduce prompt ``agent_metrics_summary`` rows to L0 ``agent_loop/*`` counters.

    Each summary is consumed once (the caller de-dupes by uid). Timing sufficient
    stats stay on the TQ field for down-drill; they are not re-emitted here so
    they cannot collide with the batch-level ``timing_s/task/*`` keys.
    """
    rows = [_unwrap(item) for item in summaries]
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return {}

    success_sessions = sum(int(row.get("success_sessions", 0)) for row in rows)
    failed_sessions = sum(int(row.get("failed_sessions", 0)) for row in rows)
    unfinished_sessions = sum(int(row.get("unfinished_sessions", 0)) for row in rows)
    failed_prompts = sum(1 for row in rows if int(row.get("success_sessions", 0)) <= 0)
    return {
        "agent_loop/failed_prompts": float(failed_prompts),
        "agent_loop/success_sessions": float(success_sessions),
        "agent_loop/failed_sessions": float(failed_sessions),
        "agent_loop/unfinished_sessions": float(unfinished_sessions),
    }
