"""Validate that a pipeline-smoke training log contains real RL update signals."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _float_values(pattern: str, content: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, content)]


def verify_training_log(content: str, *, expected_step: int) -> dict[str, int | float]:
    """Raise ``ValueError`` unless the log proves rollout, reward, and update."""
    if expected_step < 1:
        raise ValueError("expected_step must be positive")

    problems: list[str] = []
    summaries = [line for line in content.splitlines() if "generate_sequences summary:" in line]
    # Workers emit summaries independently; async prefetch and multiple GPUs
    # do not have a one-summary-per-training-step relationship.
    if not summaries:
        problems.append("missing rollout summary")
    for line in summaries:
        input_match = re.search(r"num_input_prompts=(\d+)", line)
        success_match = re.search(r"num_success_sessions=(\d+)", line)
        output_match = re.search(r"num_success_outputs=(\d+)", line)
        if input_match is None or int(input_match.group(1)) == 0:
            problems.append("rollout summary has no input prompts")
        if success_match is None or int(success_match.group(1)) == 0:
            problems.append("rollout summary has no successful sessions")
        if success_match is None or output_match is None or int(output_match.group(1)) != int(success_match.group(1)):
            problems.append("rollout summary success session/output counts do not match")
        for field in ("num_failed_sessions", "num_unfinished_episodes", "num_failed_uids"):
            match = re.search(rf"{field}=(\d+)", line)
            if match is None or int(match.group(1)) != 0:
                problems.append(f"rollout summary has invalid {field}")

    steps = [int(value) for value in re.findall(r"training/global_step:(\d+)", content)]
    expected_steps = list(range(1, expected_step + 1))
    if steps != expected_steps:
        problems.append(f"training steps are {steps}, expected {expected_steps}")

    rewards = _float_values(rf"critic/rewards/mean:({_FLOAT})", content)
    if len(rewards) != expected_step or not all(math.isfinite(value) for value in rewards):
        problems.append("missing or non-finite reward metrics")
    elif not any(value > 0 for value in rewards):
        problems.append("no positive task reward")

    grad_norms = _float_values(rf"actor/grad_norm:({_FLOAT})", content)
    if len(grad_norms) != expected_step or not all(math.isfinite(value) for value in grad_norms):
        problems.append("missing or non-finite actor gradients")
    elif not any(abs(value) > 0 for value in grad_norms):
        problems.append("no non-zero actor gradient")

    update_actor_times = _float_values(rf"timing_s/update_actor:({_FLOAT})", content)
    update_weights_times = _float_values(rf"timing_s/update_weights:({_FLOAT})", content)
    if len(update_actor_times) != expected_step or not all(
        math.isfinite(value) and value > 0 for value in update_actor_times
    ):
        problems.append("missing, non-finite, or non-positive actor-update timing")
    if len(update_weights_times) != expected_step or not all(
        math.isfinite(value) and value > 0 for value in update_weights_times
    ):
        problems.append("missing, non-finite, or non-positive rollout-weight synchronization timing")

    if problems:
        raise ValueError("; ".join(dict.fromkeys(problems)))

    return {
        "global_step": max(steps),
        "rollout_summaries": len(summaries),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "nonzero_grad_steps": sum(math.isfinite(value) and abs(value) > 0 for value in grad_norms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--agent-log-dir", required=True, type=Path)
    parser.add_argument("--expected-step", required=True, type=int)
    args = parser.parse_args()
    try:
        summary = verify_training_log(
            args.log_file.read_text(encoding="utf-8", errors="replace"),
            expected_step=args.expected_step,
        )
        finish_sessions = verify_finish_evidence(args.agent_log_dir)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Training signal verification failed: {exc}") from exc
    print(
        "Verified rollout -> reward -> policy update: "
        f"global_step={summary['global_step']}, "
        f"reward_mean_range={summary['reward_min']}..{summary['reward_max']}, "
        f"nonzero_grad_steps={summary['nonzero_grad_steps']}, correct_finish_sessions={finish_sessions}"
    )


def verify_finish_evidence(log_dir: Path) -> int:
    """Require a correct real finish observation and a saved rewarded trajectory."""
    import json

    count = 0
    for task_log in log_dir.glob("step_*/session-*/task.log"):
        content = task_log.read_text(encoding="utf-8")
        if "pipeline_smoke: correct_finish_observed" not in content:
            continue
        trajectory_path = task_log.with_name("trajectory.json")
        if not trajectory_path.is_file():
            raise ValueError(f"missing trajectory for finish evidence: {task_log.parent.name}")
        data = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if any(t.get("finished") is True and t.get("reward_score") == 1.0 for t in data["trajectories"]):
            count += 1
    if count == 0:
        raise ValueError("no correct real finish call with a saved rewarded trajectory")
    return count


if __name__ == "__main__":
    main()
