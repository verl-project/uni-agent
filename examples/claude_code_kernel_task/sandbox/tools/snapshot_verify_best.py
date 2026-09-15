#!/usr/bin/env python3
"""Keep the highest-ranked agent-time verifier result and implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _speedup(metrics: dict[str, Any]) -> float:
    values = [metrics]
    perf = metrics.get("perf_data")
    if isinstance(perf, dict):
        values.append(perf)
        implementation = perf.get("implementation")
        if isinstance(implementation, dict):
            values.append(implementation)
    for value in values:
        for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
            if key in value:
                return max(0.0, _number(value[key]))
    return 0.0


def _reward(metrics: dict[str, Any]) -> float:
    pass_rate = _number(metrics.get("pass_rate"))
    speedup = _speedup(metrics)
    total = 0.05
    if metrics.get("compile_ok") is True:
        total += 0.25
    total += 0.40 * pass_rate
    if speedup > 0 and pass_rate > 0:
        total += 0.30 * min(speedup / 2.0, 1.0) * pass_rate
    return round(total, 4)


def _metrics(
    op_name: str,
    verify: dict[str, Any],
    summary: dict[str, Any] | None,
    perf: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = summary or {}
    total = _integer(verify.get("total_cases"), _integer(summary.get("total_cases")))
    passed = _integer(verify.get("passed_cases"), _integer(summary.get("passed_cases")))
    failed = _integer(verify.get("failed_cases"), max(total - passed, 0))
    correct = total > 0 and passed == total and failed == 0
    compile_ok = correct or passed > 0 or summary.get("compile_ok") is True
    result: dict[str, Any] = {
        "op_name": op_name,
        "success": correct,
        "ast_check_ok": True,
        "compile_ok": compile_ok,
        "correctness_ok": correct,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": round(passed / total, 6) if total else 0.0,
    }
    if correct and perf:
        result["perf_data"] = perf
    result["reward"] = _reward(result)
    return result


def _rank(metrics: dict[str, Any]) -> tuple[float, ...]:
    correct = metrics.get("correctness_ok") is True
    return (
        float(correct),
        _number(metrics.get("pass_rate")),
        float(metrics.get("compile_ok") is True),
        _speedup(metrics) if correct else 0.0,
        float(_integer(metrics.get("passed_cases"))),
        -float(_integer(metrics.get("failed_cases"))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--verify-result", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--perf-result")
    args = parser.parse_args()

    workspace = Path.cwd()
    verify = _read(workspace / args.verify_result)
    if verify is None:
        return 0
    summary = _read(workspace / args.summary) if args.summary else None
    perf = _read(workspace / args.perf_result) if args.perf_result else None
    metrics = _metrics(args.op_name, verify, summary, perf)
    staged = workspace / "output" / "verify" / f"{args.op_name}_triton_ascend_impl.py"
    if not staged.is_file():
        return 0
    best_metrics = workspace / "metrics_best.json"
    previous = _read(best_metrics)
    if previous is not None and _rank(metrics) <= _rank(previous):
        print("[best-snapshot] kept existing best")
        return 0

    best_impl = workspace / "src" / f"{args.op_name}_triton_ascend_impl_best.py"
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    metrics.update(
        {
            "implementation_path": f"src/{args.op_name}_triton_ascend_impl_best.py",
            "implementation_sha256": digest,
        }
    )
    temporary_impl = best_impl.with_suffix(".tmp")
    temporary_metrics = best_metrics.with_suffix(".tmp")
    shutil.copy2(staged, temporary_impl)
    temporary_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_impl, best_impl)
    os.replace(temporary_metrics, best_metrics)
    print(
        f"[best-snapshot] updated: passed={metrics['passed_cases']}/{metrics['total_cases']} "
        f"speedup={_speedup(metrics):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
