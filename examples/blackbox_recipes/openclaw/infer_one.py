"""Run one task against an existing endpoint, without Ray or a training loop."""
import argparse
import asyncio
import dataclasses
import json
import os
from pathlib import Path

from uni_agent.tasks.registry import get_task


async def run(args):
    data = json.loads(args.task.read_text(encoding="utf-8"))
    model = {"base_url": args.base_url, "model_name": args.model, "api_key": os.getenv("API_KEY", "EMPTY")}
    data["agent"]["model"] = model
    data["agent"]["artifact_dir"] = str(args.output.resolve().parent / "trajectories")
    task = get_task(data)
    result = await task.run()
    report = dataclasses.asdict(result)
    report["task_completed"] = (result.reward == 1.0 and result.finished is True
        and result.extra_info.get("agent", {}).get("trajectory_audit", {}).get("verified") is True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("task_completed", "reward", "accuracy", "finished")}))
    return report["task_completed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
