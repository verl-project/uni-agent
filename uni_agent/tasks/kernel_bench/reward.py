"""Deterministic KernelBench reward calculation for verifier-produced metrics."""

from __future__ import annotations

import math
from typing import Any

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "ast": 0.05,
    "compile": 0.25,
    "correctness": 0.40,
    "speedup": 0.30,
    "target_speedup": 2.0,
}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _is_true(value: Any) -> bool:
    return value is True


def _speedup(metrics: dict[str, Any]) -> float:
    containers = [metrics]
    perf_data = metrics.get("perf_data")
    if isinstance(perf_data, dict):
        containers.append(perf_data)
        implementation = perf_data.get("implementation")
        if isinstance(implementation, dict):
            containers.append(implementation)
    for container in containers:
        for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
            if key in container:
                return max(0.0, _as_float(container[key]))
    return 0.0


def normalize_metrics(
    metrics: dict[str, Any] | None,
    *,
    verify: dict[str, Any] | None = None,
    perf: dict[str, Any] | None = None,
    op_name: str = "operator",
) -> dict[str, Any]:
    """Normalize either recipe metrics or the verifier's raw JSON artifacts."""

    result = dict(metrics or {})
    verify = verify or {}
    perf = perf or {}
    for key in ("total_cases", "passed_cases", "failed_cases", "compile_ok", "output_observed"):
        if key not in result and key in verify:
            result[key] = verify[key]
    if "perf_data" not in result and perf:
        result["perf_data"] = perf

    total = max(0, _as_int(result.get("total_cases")))
    passed = max(0, _as_int(result.get("passed_cases")))
    failed = max(0, _as_int(result.get("failed_cases"), max(total - passed, 0)))
    if total and passed > total:
        passed = total
    correctness_ok = _is_true(result.get("correctness_ok")) or bool(total and passed == total and failed == 0)
    compile_ok = bool(
        _is_true(result.get("compile_ok")) or _is_true(result.get("output_observed")) or correctness_ok or passed > 0
    )
    pass_rate = min(1.0, max(0.0, passed / total)) if total else float(correctness_ok)

    result.update(
        {
            "op_name": str(result.get("op_name") or op_name),
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": failed,
            "pass_rate": round(pass_rate, 6),
            "compile_ok": compile_ok,
            "correctness_ok": correctness_ok,
            "success": bool(result.get("success")) or correctness_ok,
        }
    )
    if not correctness_ok:
        result.setdefault("error_type", "compilation_failed" if not compile_ok else "correctness_failed")
    return result


def reward_breakdown(
    metrics: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return the preserved partial-credit reward and its compact components."""

    configured = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    pass_rate = min(1.0, max(0.0, _as_float(metrics.get("pass_rate"))))
    correctness_ok = metrics.get("correctness_ok") is True
    ast_score = configured["ast"] if metrics.get("ast_check_ok") else 0.0
    compile_score = configured["compile"] if metrics.get("compile_ok") else 0.0
    correctness_score = configured["correctness"] * pass_rate
    speedup = _speedup(metrics)
    target = max(configured["target_speedup"], 1e-9)
    speedup_score = 0.0
    if speedup > 0 and (correctness_ok or pass_rate > 0):
        speedup_score = configured["speedup"] * min(speedup / target, 1.0) * pass_rate
    total = max(0.0, ast_score + compile_score + correctness_score + speedup_score)
    return {
        "ast": round(ast_score, 4),
        "compile": round(compile_score, 4),
        "correctness": round(correctness_score, 4),
        "speedup": round(speedup_score, 4),
        "raw_speedup": round(speedup, 4),
        "total": round(total, 4),
    }


def attach_reward(metrics: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, Any]:
    result = dict(metrics)
    components = reward_breakdown(result, weights)
    result["reward"] = components["total"]
    result["reward_components"] = components
    return result


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info=None) -> dict[str, float]:
    """verl reward-function compatibility; task-side reward remains authoritative."""

    del data_source, solution_str, ground_truth
    score = _as_float(extra_info.get("reward_score")) if isinstance(extra_info, dict) else 0.0
    return {"score": score}
