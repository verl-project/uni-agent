"""Scan a directory of `<run_id>/interaction_result.json` files and emit one
row per pytest invocation found in `action`.

Output: a JSONL file where each line is:
  {run_id, step_idx, execution_time, exit_reason, command}

Detection rule: the action contains one of the tokens
  - `pytest`, `pytest3`, ... (as a standalone command word)
  - `python[X.Y] -m pytest`
  - `py.test`

Usage:
  python examples/agent_interaction/extract_pytest_steps.py \
      --log-dir /mnt/hdfs/went/logs/swebench_qwen3_coder_trajectories_1 \
      --out /tmp/pytest_steps.jsonl
"""

# ruff: noqa: E501

import argparse
import json
import re
import sys
from pathlib import Path

PYTEST_RE = re.compile(r"(?:^|[\s;&|`(])(python[0-9.]*\s+-m\s+pytest|pytest[0-9]*|py\.test)\b")


def first_pytest_cmd(action: str) -> str | None:
    """Return the first line of `action` that contains a pytest invocation, trimmed."""
    for line in action.splitlines():
        if PYTEST_RE.search(line):
            return line.strip()[:200]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files = sorted(args.log_dir.glob("*/interaction_result.json"))
    print(f"Scanning {len(files)} trajectories")

    n_pytest = 0
    with args.out.open("w", encoding="utf-8") as f:
        for p in files:
            try:
                data = json.load(p.open())
            except (json.JSONDecodeError, OSError) as e:
                print(f"skip {p}: {e}", file=sys.stderr)
                continue
            run_id = p.parent.name
            for s in data.get("trajectory", []):
                action = s.get("action") or ""
                cmd = first_pytest_cmd(action)
                if cmd is None:
                    continue
                f.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "step_idx": s.get("step_idx"),
                            "execution_time": s.get("execution_time"),
                            "exit_reason": s.get("exit_reason"),
                            "command": cmd,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_pytest += 1

    print(f"Found {n_pytest} pytest step(s); wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
