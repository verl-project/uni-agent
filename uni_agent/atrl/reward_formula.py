#!/usr/bin/env python3
"""Shared reward formula for Triton/Ascend KernelBench rollouts.

Single source of truth for the partial-credit reward computation, imported by
both:

- ``snapshot_verify_best.py`` -- inside the sandbox, after each ``run_verify``
  call, to snapshot the best implementation and precompute its reward.
- ``reward.py`` -- on the trainer side, during RL reward evaluation.

Both sides MUST derive identical reward values from the same metrics, so the
formula lives here once. Weights come from the ``TRITON_REWARD_*`` environment
variables; their sum is the natural reward upper bound (1.30 with the weights
shipped in ``train_claude_code_megatron_async_lyl.sh``: AST 0.02 + compile
0.15 + correctness 0.60 + bonus 0.13 + speedup 0.40).

The total is clamped to ``>= 0`` only -- it is NOT clamped to 1.0. Clamping to
1.0 collapsed GRPO's within-group speedup discrimination: every correct
rollout with ``speedup >= 0.5 * target`` saturated at 1.0, zeroing the
within-group variance on the speedup axis. Leaving the upper bound at the
weight sum keeps correct rollouts spread across ``[0.90, 1.30]`` by speedup.

This module is pure-stdlib (``os`` only) so it runs unchanged in the sandbox
Python environment and on the trainer.
"""

from __future__ import annotations

import os
from typing import Any


# --------------------------------------------------------------------------- #
# Numeric / environment helpers
# --------------------------------------------------------------------------- #

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def float_env(name: str, default: float, *fallback_names: str) -> float:
    for key in (name, *fallback_names):
        value = os.environ.get(key)
        if value not in (None, ""):
            return float(value)
    return default


def int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return safe_int(value, default)


def truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "")


# --------------------------------------------------------------------------- #
# Metric extraction
# --------------------------------------------------------------------------- #

def pass_rate(metrics: dict[str, Any]) -> float:
    total = safe_int(metrics.get("total_cases"), 0)
    passed = safe_int(metrics.get("passed_cases"), 0)
    if total <= 0:
        return 1.0 if bool(metrics.get("correctness_ok") or metrics.get("success")) else 0.0
    return max(0.0, min(passed / max(total, 1), 1.0))


def compile_ok(metrics: dict[str, Any]) -> bool:
    """3-condition compile check used inside the reward formula.

    Mirrors the inline check historically embedded in ``reward_breakdown`` on
    both sides. The looser 5-condition variant (also accepting
    ``passed_cases > 0`` / non-empty ``perf_data``) is NOT used here so that
    reward scores stay byte-identical to the pre-refactor ``reward_breakdown``.
    """
    return bool(
        metrics.get("compile_ok")
        or metrics.get("output_observed")
        or metrics.get("correctness_ok")
    )


def speedup_from_metrics(metrics: dict[str, Any]) -> float:
    """Speedup-vs-torch read from ``metrics['perf_data']`` (0.0 if missing).

    This supersedes the undefined ``_speedup`` helper that previously made
    ``snapshot_verify_best.py`` raise ``NameError`` inside ``_reward_breakdown``.
    """
    perf_data = metrics.get("perf_data")
    if not isinstance(perf_data, dict):
        return 0.0
    for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
        if key in perf_data:
            return max(0.0, safe_float(perf_data.get(key), 0.0))
    return 0.0


def has_verify_signal(metrics: dict[str, Any] | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    return any(
        key in metrics
        for key in (
            "verified_success",
            "verify_exit",
            "total_cases",
            "passed_cases",
            "failed_cases",
            "pass_rate",
            "compile_ok",
            "correctness_ok",
        )
    )


def verify_artifacts_confirm_success(
    verify: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> bool:
    """Require matching raw counts and a successful current-run summary."""
    if not isinstance(verify, dict) or not isinstance(summary, dict):
        return False
    count_keys = ("passed_cases", "total_cases", "failed_cases")
    if not all(key in verify and key in summary for key in count_keys):
        return False
    try:
        verify_counts = tuple(int(verify[key]) for key in count_keys)
        summary_counts = tuple(int(summary[key]) for key in count_keys)
        verify_exit = summary["verify_exit"]
    except (KeyError, TypeError, ValueError):
        return False
    if type(verify_exit) is not int:
        return False
    passed, total, failed = verify_counts
    return bool(
        summary.get("verified_success") is True
        and verify_exit == 0
        and verify_counts == summary_counts
        and total > 0
        and passed == total
        and failed == 0
    )


# --------------------------------------------------------------------------- #
# Reward formula (single source of truth)
# --------------------------------------------------------------------------- #

def reward_breakdown(metrics: dict[str, Any]) -> dict[str, Any]:
    """Stable partial-credit reward components from a metrics dict.

    Weights are read from ``TRITON_REWARD_*`` env vars. ``total`` is the raw
    weighted sum clamped to ``>= 0`` only (no 1.0 upper clamp): with the
    shipped weights a fully-correct + on-target rollout scores 1.30, preserving
    speedup discrimination across ``[0.90, 1.30]`` for GRPO.
    """
    ast_reward = float_env("TRITON_REWARD_AST_OK", 0.0)
    compile_reward = float_env("TRITON_REWARD_COMPILE_OK", 0.10)
    correctness_reward = float_env("TRITON_REWARD_CORRECTNESS_OK", 0.55)
    all_correct_bonus = float_env("TRITON_REWARD_ALL_CORRECT_BONUS", 0.10)
    speedup_reward = float_env("TRITON_REWARD_SPEEDUP_MAX", 0.40)

    ast_ok = bool(metrics.get("ast_check_ok", False))
    c_ok = compile_ok(metrics)
    p_rate = pass_rate(metrics)
    success = bool(metrics.get("success", False))
    corr_ok = bool(metrics.get("correctness_ok", False))

    ast_score = ast_reward if ast_ok else 0.0
    compile_score = compile_reward if c_ok else 0.0
    correctness_score = correctness_reward * p_rate
    all_correct_score = all_correct_bonus if corr_ok else 0.0

    speedup = speedup_from_metrics(metrics)
    speedup_score = 0.0
    if speedup > 0.0 and (success or corr_ok or p_rate > 0.0):
        target = max(float_env("TRITON_REWARD_TARGET_SPEEDUP", 2.0), 1e-6)
        speedup_score = speedup_reward * max(0.0, min(speedup / target, 1.0)) * max(p_rate, 0.0)

    total = round(
        max(0.0, ast_score + compile_score + correctness_score + all_correct_score + speedup_score),
        4,
    )
    return {
        "ast": round(ast_score, 4),
        "compile": round(compile_score, 4),
        "correctness": round(correctness_score, 4),
        "all_correct_bonus": round(all_correct_score, 4),
        "speedup": round(speedup_score, 4),
        "pass_rate": round(p_rate, 6),
        "raw_speedup": round(speedup, 6),
        "total": total,
    }


def metric_reward(metrics: dict[str, Any] | None) -> float:
    """Reward for a metrics dict, preferring a precomputed ``reward`` field.

    Used by the sandbox snapshot path, which writes the reward back into the
    metrics dict via ``attach_reward_fields`` and then reads it back here.
    """
    if not isinstance(metrics, dict):
        return 0.0
    if "reward" in metrics:
        return safe_float(metrics.get("reward"), 0.0)
    return reward_breakdown(metrics)["total"]


def attach_reward_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add flat reward fields to a metrics dict in place (and return it)."""
    components = reward_breakdown(metrics)
    for key in ("reward_components", "benchmark_ok", "output_observed"):
        metrics.pop(key, None)
    metrics["reward"] = components["total"]
    metrics["reward_ast"] = components["ast"]
    metrics["reward_compile"] = components["compile"]
    metrics["reward_correctness"] = components["correctness"]
    metrics["reward_all_correct_bonus"] = components["all_correct_bonus"]
    metrics["reward_speedup"] = components["speedup"]
    metrics["raw_speedup"] = components["raw_speedup"]
    # Same-impl speedup for metrics_best consumers; do not read a sibling
    # perf_result_best.json that may belong to a faster different impl.
    metrics["speedup_vs_torch"] = components["raw_speedup"]
    return metrics


# --------------------------------------------------------------------------- #
# Best-version ranking (which snapshot is "best" across the agent loop)
# --------------------------------------------------------------------------- #

def rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Sort key for choosing the best version across the agent loop.

    Higher tuple = strictly better. Components: correctness > pass_rate >
    compile_ok > raw_speedup > total_reward > passed_cases > -failed_cases.
    ``raw_speedup`` and ``total`` give speedup-aware discrimination so a faster
    correct version outranks a slower correct one (aligns with
    ``TRITON_TRAIN_BEST_FIRST``).
    """
    correctness_ok = bool(metrics.get("correctness_ok") or metrics.get("success"))
    components = reward_breakdown(metrics)
    return (
        1.0 if correctness_ok else 0.0,
        components["pass_rate"],
        1.0 if compile_ok(metrics) else 0.0,
        components["raw_speedup"],
        components["total"],
        float(safe_int(metrics.get("passed_cases"), 0)),
        -float(safe_int(metrics.get("failed_cases"), 0)),
    )
