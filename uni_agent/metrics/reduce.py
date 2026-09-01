"""Reduce per-task metrics to flat scalars.

``reduce_metrics`` takes a dict of :class:`~uni_agent.metrics.collector.Metric`
instances or sample lists and collapses each key to one number. The aggregation
is taken from the ``Metric`` when present; otherwise the key name decides
(``max`` / ``min`` in the name, else mean). A list of per-task metrics dicts is
reduced per task first, then across tasks. Callers choose metric names; this
module does not.
"""

from __future__ import annotations

from typing import Any

from .collector import Metric


def reduce_metrics(
    metrics: dict[str, Metric | list[Any] | dict[str, Any]] | list[dict[str, dict[str, Any]]],
) -> dict[str, float]:
    """Collapse each metric to a single scalar."""
    if isinstance(metrics, list):
        grouped: dict[str, list[float]] = {}
        for item in metrics:
            for name, entry in item.items():
                metric = Metric.from_dict(entry)
                if metric.values:
                    grouped.setdefault(name, []).append(metric.aggregate())
        return reduce_metrics(grouped)

    out: dict[str, float] = {}
    for key, val in metrics.items():
        if isinstance(val, Metric):
            if val.values:
                out[key] = val.aggregate()
            continue
        if isinstance(val, dict) and "values" in val:
            metric = Metric.from_dict(val)
            if metric.values:
                out[key] = metric.aggregate()
            continue
        values = [float(v) for v in val]
        if not values:
            continue
        key_lower = key.lower()
        if "max" in key_lower:
            out[key] = max(values)
        elif "min" in key_lower:
            out[key] = min(values)
        else:
            out[key] = sum(values) / len(values)
    return out


_PROMPT_COUNT_KEYS = (
    "expected_sessions",
    "success_sessions",
    "failed_sessions",
    "unfinished_sessions",
)


def _metrics_sufficient_stats(metrics: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, float]]:
    """Collapse one task's metrics to ``{name: {count, sum, min, max}}``.

    Each metric contributes one task-level scalar (the collector aggregation),
    so a later merge across sessions is lossless.
    """
    if not metrics:
        return {}
    stats: dict[str, dict[str, float]] = {}
    for name, entry in metrics.items():
        if not isinstance(entry, dict) or "values" not in entry:
            continue
        metric = Metric.from_dict(entry)
        if not metric.values:
            continue
        value = float(metric.aggregate())
        stats[name] = {"count": 1.0, "sum": value, "min": value, "max": value}
    return stats


def merge_sufficient_stats(
    rows: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Merge per-session sufficient stats without losing min/max."""
    merged: dict[str, dict[str, float]] = {}
    for row in rows:
        for name, stats in row.items():
            if name in _PROMPT_COUNT_KEYS or not isinstance(stats, dict):
                continue
            incoming_count = float(stats.get("count", 0.0))
            if incoming_count <= 0:
                continue
            existing = merged.get(name)
            if existing is None:
                merged[name] = {
                    "count": incoming_count,
                    "sum": float(stats["sum"]),
                    "min": float(stats["min"]),
                    "max": float(stats["max"]),
                }
                continue
            existing["count"] += incoming_count
            existing["sum"] += float(stats["sum"])
            existing["min"] = min(existing["min"], float(stats["min"]))
            existing["max"] = max(existing["max"], float(stats["max"]))
    return merged


def build_prompt_metrics_summary(
    *,
    expected_sessions: int,
    success_sessions: int,
    failed_sessions: int,
    unfinished_sessions: int,
    metrics: list[dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Prompt-level TQ payload: session outcomes plus merged sufficient stats."""
    summary: dict[str, Any] = {
        "expected_sessions": int(expected_sessions),
        "success_sessions": int(success_sessions),
        "failed_sessions": int(failed_sessions),
        "unfinished_sessions": int(unfinished_sessions),
    }
    summary.update(merge_sufficient_stats([_metrics_sufficient_stats(item) for item in metrics or []]))
    return summary
