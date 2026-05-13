"""Offline analyzer for legacy Uni-Agent rollout logs.

Works on `interaction_result.json` files written before the per-turn timing
fields (llm_time / tool_outcome / step_start_ts / ...) were added. Uses only
the fields that have always existed:

  - top-level: execution_time, metrics.tool_calls, metrics.generate_sequences
  - per step:  step_idx, action, observation, execution_time, exit_reason

What this script answers:
  1. Which trajectories are tool-dominated? (sum_tool / total_time large)
  2. Which trajectories experienced timeout / terminal_not_alive / format_error?
  3. For each interesting trajectory, what commands ran in each turn and how long?

Usage:
  python examples/agent_interaction/analyze_legacy.py \
      --log-dir /mnt/hdfs/went/logs/swebench_qwen3_coder_trajectories \
      --top-k 20 \
      --tool-ratio-threshold 0.5 \
      --out-dir ./legacy_analysis
"""

# ruff: noqa: E501

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StepRow:
    step_idx: int
    exit_reason: str
    action: str
    observation: str
    execution_time: float | None  # tool-call duration; None on non-tool exits

    @property
    def is_timeout(self) -> bool:
        return self.exit_reason == "timeout_error"

    @property
    def is_terminal_dead(self) -> bool:
        return self.exit_reason == "terminal_not_alive"

    @property
    def is_format_error(self) -> bool:
        return self.exit_reason == "format_error"

    @property
    def tool_hint(self) -> str:
        """Best-effort tool name guess from the raw action string.

        Format error / token limit / abnormal exit -> empty action; mark unknown.
        """
        if not self.action:
            return "-"
        a = self.action.strip()
        # ToolsManager.get_tool_bash_command formats non-bash tools as
        # "<tool_name> <subcommand?> --k v ...". For execute_bash it returns
        # the raw command, and for submit it returns "echo '<<<Finished>>>'".
        if a.startswith("echo '<<<Finished>>>'"):
            return "submit"
        first = a.split(None, 1)[0]
        # Known structured tools used in this project
        if first in {"str_replace_editor", "search", "search_arxiv", "finish"}:
            return first
        # Anything else is treated as raw shell from execute_bash
        return "execute_bash"


@dataclass
class TrajectoryRow:
    path: Path
    run_id: str
    execution_time: float
    sum_tool: float  # from metrics.tool_calls (cumulative)
    sum_llm: float  # from metrics.generate_sequences
    num_turns: int
    last_exit_reason: str
    steps: list[StepRow] = field(default_factory=list)

    @property
    def tool_ratio(self) -> float:
        if self.execution_time <= 0:
            return 0.0
        return self.sum_tool / self.execution_time

    @property
    def llm_ratio(self) -> float:
        if self.execution_time <= 0:
            return 0.0
        return self.sum_llm / self.execution_time

    @property
    def num_timeouts(self) -> int:
        return sum(1 for s in self.steps if s.is_timeout)

    @property
    def num_terminal_dead(self) -> int:
        return sum(1 for s in self.steps if s.is_terminal_dead)

    @property
    def num_format_errors(self) -> int:
        return sum(1 for s in self.steps if s.is_format_error)

    @property
    def max_step_exec(self) -> float:
        return max((s.execution_time or 0.0) for s in self.steps) if self.steps else 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_trajectory(path: Path) -> TrajectoryRow | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    steps_raw: list[dict[str, Any]] = data.get("trajectory", [])
    steps: list[StepRow] = []
    for s in steps_raw:
        steps.append(
            StepRow(
                step_idx=s.get("step_idx", -1),
                exit_reason=s.get("exit_reason", ""),
                action=s.get("action", "") or "",
                observation=s.get("observation", "") or "",
                execution_time=s.get("execution_time"),
            )
        )

    metrics = data.get("metrics") or {}
    return TrajectoryRow(
        path=path,
        run_id=path.parent.name,
        execution_time=float(data.get("execution_time", 0.0) or 0.0),
        sum_tool=float(metrics.get("tool_calls", 0.0) or 0.0),
        sum_llm=float(metrics.get("generate_sequences", 0.0) or 0.0),
        num_turns=len(steps),
        last_exit_reason=(steps[-1].exit_reason if steps else ""),
        steps=steps,
    )


def discover(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("*/interaction_result.json"))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... <{len(text) - max_chars} chars elided>"


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def write_per_trajectory_jsonl(out_path: Path, trajs: list[TrajectoryRow]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for t in trajs:
            f.write(
                json.dumps(
                    {
                        "run_id": t.run_id,
                        "path": str(t.path),
                        "execution_time": t.execution_time,
                        "sum_tool": t.sum_tool,
                        "sum_llm": t.sum_llm,
                        "tool_ratio": t.tool_ratio,
                        "llm_ratio": t.llm_ratio,
                        "num_turns": t.num_turns,
                        "last_exit_reason": t.last_exit_reason,
                        "num_timeouts": t.num_timeouts,
                        "num_terminal_dead": t.num_terminal_dead,
                        "num_format_errors": t.num_format_errors,
                        "max_step_exec": t.max_step_exec,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_per_step_jsonl(out_path: Path, trajs: list[TrajectoryRow]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for t in trajs:
            for s in t.steps:
                f.write(
                    json.dumps(
                        {
                            "run_id": t.run_id,
                            "step_idx": s.step_idx,
                            "exit_reason": s.exit_reason,
                            "tool_hint": s.tool_hint,
                            "execution_time": s.execution_time,
                            "is_timeout": s.is_timeout,
                            "is_terminal_dead": s.is_terminal_dead,
                            "is_format_error": s.is_format_error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def write_overview(out_path: Path, trajs: list[TrajectoryRow]) -> None:
    """Top-level distribution + categorical counts."""
    lines: list[str] = []
    n = len(trajs)
    lines.append("# Overview\n")
    lines.append(f"Total trajectories: **{n}**\n")
    if n == 0:
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    exec_sorted = sorted(t.execution_time for t in trajs)
    tool_ratio_sorted = sorted(t.tool_ratio for t in trajs)
    llm_ratio_sorted = sorted(t.llm_ratio for t in trajs)

    lines.append("## execution_time distribution (s)\n")
    lines.append(
        f"- mean={sum(exec_sorted) / n:.1f} | p50={_quantile(exec_sorted, 0.5):.1f} | "
        f"p90={_quantile(exec_sorted, 0.9):.1f} | p99={_quantile(exec_sorted, 0.99):.1f} | "
        f"max={exec_sorted[-1]:.1f}\n"
    )

    lines.append("## tool_ratio distribution (sum_tool / execution_time)\n")
    lines.append(
        f"- mean={sum(tool_ratio_sorted) / n:.2%} | p50={_quantile(tool_ratio_sorted, 0.5):.2%} | "
        f"p90={_quantile(tool_ratio_sorted, 0.9):.2%} | max={tool_ratio_sorted[-1]:.2%}\n"
    )
    lines.append("## llm_ratio distribution (sum_llm / execution_time)\n")
    lines.append(
        f"- mean={sum(llm_ratio_sorted) / n:.2%} | p50={_quantile(llm_ratio_sorted, 0.5):.2%} | "
        f"p90={_quantile(llm_ratio_sorted, 0.9):.2%} | max={llm_ratio_sorted[-1]:.2%}\n"
    )

    lines.append("## Categorical counts\n")
    lines.append(f"- trajectories with any timeout_error:        **{sum(1 for t in trajs if t.num_timeouts)}**")
    lines.append(f"- trajectories with any terminal_not_alive:   **{sum(1 for t in trajs if t.num_terminal_dead)}**")
    lines.append(f"- trajectories with any format_error:         **{sum(1 for t in trajs if t.num_format_errors)}**")

    # Exit-reason histogram (last step)
    from collections import Counter
    last_exit_counts = Counter(t.last_exit_reason for t in trajs)
    lines.append("\n## Last-step exit_reason histogram\n")
    lines.append("| exit_reason | count |")
    lines.append("|---|---:|")
    for k, v in sorted(last_exit_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{k or '(empty)'}` | {v} |")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def render_trajectory_detail(t: TrajectoryRow, action_chars: int, obs_chars: int) -> list[str]:
    """One markdown subsection per trajectory."""
    lines: list[str] = []
    lines.append(f"### `{t.run_id}`\n")
    lines.append(f"- file: `{t.path}`")
    lines.append(
        f"- execution_time: **{t.execution_time:.1f}s** | num_turns: {t.num_turns} | "
        f"last_exit: `{t.last_exit_reason}`"
    )
    lines.append(
        f"- sum_tool: **{t.sum_tool:.1f}s** ({t.tool_ratio:.1%}) | "
        f"sum_llm: {t.sum_llm:.1f}s ({t.llm_ratio:.1%})"
    )
    lines.append(
        f"- #timeouts: **{t.num_timeouts}** | #terminal_dead: **{t.num_terminal_dead}** | "
        f"#format_errors: {t.num_format_errors} | max_step_exec: {t.max_step_exec:.1f}s\n"
    )

    # Per-turn table
    lines.append("| step | tool | exec(s) | exit_reason | action (first line) |")
    lines.append("|---:|---|---:|---|---|")
    for s in t.steps:
        action_first = (s.action.splitlines()[0] if s.action else "").strip().replace("|", "\\|")
        if len(action_first) > 80:
            action_first = action_first[:77] + "..."
        et = "n/a" if s.execution_time is None else f"{s.execution_time:.2f}"
        lines.append(
            f"| {s.step_idx} | `{s.tool_hint}` | {et} | {s.exit_reason} | `{action_first}` |"
        )

    # Spotlight: timeout / terminal_dead / format_error -- with full action body
    spotlight = [
        s for s in t.steps if s.is_timeout or s.is_terminal_dead or s.is_format_error
    ]
    if spotlight:
        lines.append("\n**Notable invocations:**\n")
        for s in spotlight:
            et = "n/a" if s.execution_time is None else f"{s.execution_time:.2f}s"
            lines.append(
                f"- step **{s.step_idx}** ({s.tool_hint}, exit=`{s.exit_reason}`, exec={et}):"
            )
            if s.action:
                lines.append("\n```bash")
                lines.append(_truncate(s.action, action_chars))
                lines.append("```")
            if s.observation:
                lines.append("Observation (truncated):\n")
                lines.append("```text")
                lines.append(_truncate(s.observation, obs_chars))
                lines.append("```")
    lines.append("")
    return lines


def write_section(
    out_path: Path,
    title: str,
    description: str,
    trajs: list[TrajectoryRow],
    top_k: int,
    action_chars: int,
    obs_chars: int,
) -> None:
    lines: list[str] = [f"# {title}\n", description, ""]
    if not trajs:
        lines.append("_(no matching trajectories)_")
    else:
        lines.append(f"Showing top {min(top_k, len(trajs))} of {len(trajs)} matches.\n")
        for t in trajs[:top_k]:
            lines.extend(render_trajectory_detail(t, action_chars, obs_chars))
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy-log analyzer (no new timing fields needed)")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory containing <run_id>/interaction_result.json")
    parser.add_argument("--out-dir", type=Path, default=Path("./legacy_analysis"), help="Where to write reports.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K trajectories to render in each section.")
    parser.add_argument(
        "--tool-ratio-threshold",
        type=float,
        default=0.5,
        help="A trajectory is 'tool-dominated' if sum_tool/execution_time >= this value.",
    )
    parser.add_argument("--action-chars", type=int, default=2000, help="Max chars of action body to embed.")
    parser.add_argument("--obs-chars", type=int, default=1000, help="Max chars of observation body to embed.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only scan the first N files (for quick smoke tests). 0 = no limit.",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        logger.error("log-dir %s does not exist", log_dir)
        return 2

    paths = discover(log_dir)
    if args.limit > 0:
        paths = paths[: args.limit]
    logger.info("Found %d trajectory files under %s", len(paths), log_dir)
    if not paths:
        return 1

    trajs: list[TrajectoryRow] = []
    for p in paths:
        t = load_trajectory(p)
        if t is not None:
            trajs.append(t)
    logger.info("Loaded %d trajectories successfully", len(trajs))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Flat JSONL dumps for ad-hoc downstream analysis (pandas/DuckDB)
    write_per_trajectory_jsonl(args.out_dir / "per_trajectory.jsonl", trajs)
    write_per_step_jsonl(args.out_dir / "per_step.jsonl", trajs)

    # 2. Overview
    write_overview(args.out_dir / "overview.md", trajs)

    # 3. Tool-dominated section (by tool_ratio desc, only those above threshold)
    tool_dom = [t for t in trajs if t.tool_ratio >= args.tool_ratio_threshold]
    tool_dom.sort(key=lambda t: (t.tool_ratio, t.execution_time), reverse=True)
    write_section(
        args.out_dir / "tool_dominated.md",
        title=f"Tool-dominated trajectories (tool_ratio >= {args.tool_ratio_threshold:.0%})",
        description=(
            "Trajectories where the cumulative tool time accounts for at least the "
            "configured fraction of total execution_time. These are the ones whose "
            "tail latency cannot be explained by LLM generation alone."
        ),
        trajs=tool_dom,
        top_k=args.top_k,
        action_chars=args.action_chars,
        obs_chars=args.obs_chars,
    )

    # 4. Long-tail by total execution time (regardless of who dominates)
    by_total = sorted(trajs, key=lambda t: t.execution_time, reverse=True)
    write_section(
        args.out_dir / "longest_total.md",
        title="Longest trajectories by execution_time",
        description="Wall-clock longest trajectories. Tool ratio is included so you can see at a glance whether LLM or tools drove the long tail.",
        trajs=by_total,
        top_k=args.top_k,
        action_chars=args.action_chars,
        obs_chars=args.obs_chars,
    )

    # 5. Trajectories with timeout_error
    with_timeout = [t for t in trajs if t.num_timeouts > 0]
    with_timeout.sort(key=lambda t: (t.num_timeouts, t.execution_time), reverse=True)
    write_section(
        args.out_dir / "timeout_trajectories.md",
        title="Trajectories with timeout_error",
        description=(
            "Every trajectory that hit at least one `timeout_error`. The `exec(s)` "
            "column for those steps mixes the command's own deadline with the "
            "interrupt+probe recovery overhead -- granular separation requires "
            "the new timing fields (`tool_time` / `tool_recovery_time`)."
        ),
        trajs=with_timeout,
        top_k=args.top_k,
        action_chars=args.action_chars,
        obs_chars=args.obs_chars,
    )

    # 6. Trajectories with terminal_not_alive (most severe failure)
    with_dead = [t for t in trajs if t.num_terminal_dead > 0]
    with_dead.sort(key=lambda t: t.execution_time, reverse=True)
    write_section(
        args.out_dir / "terminal_dead_trajectories.md",
        title="Trajectories with terminal_not_alive",
        description="Trajectories where the sandbox shell stopped responding after a command -- usually a deployment-level issue, not an agent issue.",
        trajs=with_dead,
        top_k=args.top_k,
        action_chars=args.action_chars,
        obs_chars=args.obs_chars,
    )

    logger.info("Wrote outputs under %s", args.out_dir)
    for name in [
        "overview.md",
        "tool_dominated.md",
        "longest_total.md",
        "timeout_trajectories.md",
        "terminal_dead_trajectories.md",
        "per_trajectory.jsonl",
        "per_step.jsonl",
    ]:
        logger.info("  - %s", args.out_dir / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
