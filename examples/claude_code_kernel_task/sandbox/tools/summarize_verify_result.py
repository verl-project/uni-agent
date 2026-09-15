#!/usr/bin/env python3
"""Write a compact, repair-oriented summary of verifier output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _message(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("error_msg") or item.get("error") or item)
    return str(item)


def summarize(data: dict[str, Any], exit_code: int) -> dict[str, Any]:
    total = int(data.get("total_cases") or 0)
    passed = int(data.get("passed_cases") or 0)
    failed = int(data.get("failed_cases") if data.get("failed_cases") is not None else max(total - passed, 0))
    failures = data.get("failures") if isinstance(data.get("failures"), list) else []
    examples = []
    for item in failures[:4]:
        examples.append(
            {
                "case_idx": item.get("case_idx") if isinstance(item, dict) else None,
                "error_type": item.get("error_type") if isinstance(item, dict) else None,
                "reason": _message(item).strip().splitlines()[-1][:500],
            }
        )
    output_observed = passed > 0 or any(
        marker in _message(item).lower() for item in failures for marker in ("mismatch", "output", "mere=", "mare=")
    )
    return {
        "verified_success": exit_code == 0 and total > 0 and passed == total and failed == 0,
        "verify_exit": exit_code,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": round(passed / total, 6) if total else 0.0,
        "compile_ok": output_observed,
        "output_observed": output_observed,
        "failure_examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("verify_result")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--write-json", required=True)
    args = parser.parse_args()
    source = Path(args.verify_result)
    if not source.is_file():
        print(f"[verifier-summary] missing {source} after exit={args.exit_code}")
        return 0
    try:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[verifier-summary] cannot read {source}: {exc}")
        return 0
    result = summarize(data, args.exit_code)
    Path(args.write_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[verifier-summary] "
        f"passed={result['passed_cases']}/{result['total_cases']} "
        f"compile_ok={result['compile_ok']} exit={args.exit_code}"
    )
    for failure in result["failure_examples"]:
        print(f"[verifier-summary] {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
