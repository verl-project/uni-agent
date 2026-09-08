"""Check that the post-agent verifier accepts good code and rejects wrong code."""
import asyncio
import json
from pathlib import Path
import sys

from uni_agent.sandbox.docker import DockerSandbox
from uni_agent.tasks.terminal_bench.reward import compute_reward
from dataset import task_payload

CORRECT = '''def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out
'''
WRONG = 'def merge_intervals(intervals): return intervals\n'


async def main(output):
    rows = []
    for label, candidate, expected in [("correct", CORRECT, 1), ("incorrect", WRONG, 0)]:
        async with DockerSandbox(image="openclaw-recipe-runtime:2026.9.2", pull_policy="never",
                                 run_args=["--network", "none"]) as sandbox:
            await sandbox.write_file("/workspace/solution.py", candidate)
            result = await compute_reward(task_payload()["metadata"], sandbox)
            assert result["reward"] == expected, (label, result)
            rows.append({"case": label, "reward": result["reward"], "eval_report": result["eval_report"]})
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
    print(json.dumps([{k: row[k] for k in ("case", "reward")} for row in rows]))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
