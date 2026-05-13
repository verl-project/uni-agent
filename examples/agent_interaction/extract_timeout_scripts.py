"""Pull out, from each timeout step in a trajectory, the body of the script
that timed out (e.g. `reproduce_issue.py`, `test_xxx.py`).

How it works:
  For every timeout step, we scan back through earlier steps and find the most
  recent `str_replace_editor create --path <file> --file_text '<body>'` whose
  <file> matches the script being executed in the timeout step. The action
  fields are stored verbatim in interaction_result.json, so the script body is
  recoverable byte-for-byte.

Usage:
  python examples/agent_interaction/extract_timeout_scripts.py \
      --log-dir /mnt/hdfs/went/logs/swebench_qwen3_coder_trajectories \
      --out timeout_scripts.md
"""

# ruff: noqa: E501

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any


def parse_script_from_timeout_cmd(action: str) -> str | None:
    """Extract the script path from an action like `python /path/to/script.py [args]`.

    Returns None if the action doesn't look like a python invocation we can attribute.
    """
    if not action:
        return None
    # Strip common shell prefixes
    a = action.strip()
    # Match `[cd /xxx && ] [timeout N] python[3] [-flags] /path/script.py`
    m = re.search(r"\bpython3?\s+(?:-[A-Za-z]+\s+)*(\S+\.py)\b", a)
    if m:
        return m.group(1)
    return None


def parse_file_text_from_create(action: str) -> tuple[str | None, str | None]:
    """For actions of the form
      str_replace_editor create --path <P> --file_text <BODY>
    return (path, body). Otherwise (None, None).

    The action string is what ToolsManager.get_tool_bash_command produced, i.e.
    shlex-quoted bash. We use shlex to parse it back.
    """
    if not action.startswith("str_replace_editor "):
        return None, None
    try:
        toks = shlex.split(action)
    except ValueError:
        return None, None
    if len(toks) < 2 or toks[1] != "create":
        return None, None

    path = None
    file_text = None
    i = 2
    while i < len(toks):
        if toks[i] == "--path" and i + 1 < len(toks):
            path = toks[i + 1]
            i += 2
        elif toks[i] == "--file_text" and i + 1 < len(toks):
            file_text = toks[i + 1]
            i += 2
        else:
            i += 1
    return path, file_text


def find_script_body(steps: list[dict[str, Any]], up_to_idx: int, script_path: str) -> tuple[int, str] | None:
    """Walk steps[0:up_to_idx] backwards, return (step_idx, body) of the latest
    `create` that wrote `script_path`. None if not found.
    """
    for i in range(up_to_idx - 1, -1, -1):
        a = steps[i].get("action", "") or ""
        path, body = parse_file_text_from_create(a)
        if path is not None and (path == script_path or path.endswith("/" + script_path) or script_path.endswith(path)):
            if body is not None:
                return steps[i].get("step_idx", i + 1), body
    return None


def process_trajectory(path: Path) -> list[dict[str, Any]]:
    """Return a list of records, one per timeout step, with the resolved script body."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [{"error": f"failed to parse {path}: {e}"}]

    steps = data.get("trajectory", [])
    records: list[dict[str, Any]] = []
    for i, s in enumerate(steps):
        if s.get("exit_reason") != "timeout_error":
            continue
        action = s.get("action", "") or ""
        script = parse_script_from_timeout_cmd(action)
        rec: dict[str, Any] = {
            "run_id": path.parent.name,
            "step_idx": s.get("step_idx"),
            "timeout_cmd": action,
            "resolved_script": script,
            "body_source_step": None,
            "body": None,
        }
        if script is not None:
            hit = find_script_body(steps, up_to_idx=i, script_path=script.lstrip("/"))
            if hit is None:
                # also try basename match
                hit = find_script_body(steps, up_to_idx=i, script_path=Path(script).name)
            if hit is not None:
                rec["body_source_step"], rec["body"] = hit
        records.append(rec)
    return records


def write_markdown(out_path: Path, all_records: list[dict[str, Any]]) -> None:
    lines: list[str] = ["# Timeout-script bodies\n"]
    lines.append(f"Total timeout steps: **{len(all_records)}**\n")

    # Group by run_id
    by_run: dict[str, list[dict[str, Any]]] = {}
    for r in all_records:
        if "error" in r:
            continue
        by_run.setdefault(r["run_id"], []).append(r)

    for run_id, recs in by_run.items():
        lines.append(f"\n---\n\n## `{run_id}` — {len(recs)} timeout(s)\n")
        for r in recs:
            lines.append(f"### Step {r['step_idx']}\n")
            lines.append(f"- timeout command:\n\n```bash\n{r['timeout_cmd'].strip()}\n```\n")
            if r["body"] is None:
                lines.append("**Could not locate script body** (no preceding `create` matched).\n")
                continue
            lines.append(f"- script body was written at step {r['body_source_step']}:\n")
            lines.append("```python")
            lines.append(r["body"])
            lines.append("```\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory of <run_id>/interaction_result.json")
    parser.add_argument("--out", type=Path, default=Path("timeout_scripts.md"))
    parser.add_argument(
        "--only-run-ids",
        nargs="*",
        default=None,
        help="If given, restrict to these run_ids (faster on large hdfs trees).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        print(f"log-dir {log_dir} does not exist")
        return 2

    if args.only_run_ids:
        paths = [log_dir / rid / "interaction_result.json" for rid in args.only_run_ids]
        paths = [p for p in paths if p.is_file()]
    else:
        paths = sorted(log_dir.glob("*/interaction_result.json"))

    print(f"Scanning {len(paths)} trajectories")

    all_records: list[dict[str, Any]] = []
    for p in paths:
        recs = process_trajectory(p)
        all_records.extend(recs)

    timeout_count = sum(1 for r in all_records if "error" not in r)
    print(f"Found {timeout_count} timeout step(s)")
    write_markdown(args.out, all_records)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
