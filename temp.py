import asyncio
import os

from datasets import load_dataset

from uni_agent.sandbox import SandboxConfig, build_sandbox
from uni_agent.tasks.swe_bench.reward import compute_reward

INSTANCE = "pylint-dev__pylint-6528"


async def main():
    p = os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")
    ds = load_dataset("parquet", data_files=p, split="train")
    row = next(r for r in ds if r["extra_info"]["tools_kwargs"]["task"]["metadata"]["instance_id"] == INSTANCE)
    m = row["extra_info"]["tools_kwargs"]["task"]["metadata"]
    image = row["extra_info"]["tools_kwargs"]["task"]["sandbox"]["image"]

    async with build_sandbox(SandboxConfig(provider="modal", image=image, runtime_timeout=1200)) as sb:
        await sb.write_file("/tmp/gold.patch", m["patch"])
        r = await sb.exec(["git", "apply", "--whitespace=fix", "/tmp/gold.patch"], workdir="/testbed")
        print("git apply gold exit:", r.exit_code)
        result = await compute_reward(m, sb)
        ts = (result.get("eval_report") or {}).get("test_status") or {}
        print("resolved:", result["resolved"], "| eval_completed:", result["eval_completed"])
        print("FAIL_TO_PASS failures:", ts.get("FAIL_TO_PASS", {}).get("failure"))
        print("PASS_TO_PASS failures:", ts.get("PASS_TO_PASS", {}).get("failure"))


asyncio.run(main())
