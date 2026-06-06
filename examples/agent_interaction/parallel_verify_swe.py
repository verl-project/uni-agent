# ruff: noqa: E501
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

import ray
from datasets import load_dataset
from tqdm import tqdm

from uni_agent.async_logging import add_file_handler, cleanup_handlers
from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.reward import load_reward_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 512))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
DATA_PATH = os.getenv("DATA_PATH", "/home/tiger/data/swe_agent/swe_bench_multilingual_modal.parquet")


async def run_sample(sample):
    run_id = str(uuid.uuid4())
    instance = sample["extra_info"]["tools_kwargs"]
    impl = os.getenv("DEPLOYMENT", "vefaas").lower()

    case_deployment = dict(instance["env"].get("deployment", {}))
    if not case_deployment.get("image"):
        raise KeyError("No image found in instance.env.deployment.image or instance.env.image")

    if impl == "vefaas":
        defaults = {
            "type": "vefaas",
            "command": "curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}",
            "timeout": 600.0,
            "startup_timeout": 180.0,
        }
    elif impl == "modal":
        defaults = {
            "type": "modal",
            "startup_timeout": 600.0,
            "runtime_timeout": 600.0,
            "deployment_timeout": 3600.0,
        }
    elif impl == "":
        raise ValueError("DEPLOYMENT must be set")
    else:
        raise ValueError(f"Invalid environment implementation: {impl}")

    # Case config wins; defaults fill in whatever the case didn't specify.
    deployment_config = {**defaults, **case_deployment}

    env_config = {
        "deployment": deployment_config,
        "env_variables": {
            "PIP_PROGRESS_BAR": "off",
            "PIP_CACHE_DIR": "~/.cache/pip",
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "TQDM_DISABLE": "1",
            "GIT_PAGER": "cat",
        },
        "post_setup_cmd": instance["env"]["post_setup_cmd"],
    }
    env_config = AgentEnvConfig(**env_config)
    env = AgentEnv(run_id=run_id, env_config=env_config)

    reward_config = {
        "name": instance["reward"]["name"],
        "run_id": run_id,
        "metadata": instance["reward"]["metadata"],
        "env": env,
        "eval_timeout": 600.0,
    }
    reward_spec = load_reward_spec(reward_config)
    add_file_handler(Path(f"/tmp/eval_gold_patch/{run_id}.log"), run_id)

    try:
        await env.start()
        await reward_spec.apply_gold_patch()
        _, result = await reward_spec.compute_reward()
    except Exception as e:
        logger.error(f"Error running sample {run_id}: {e}")
        result = {"resolved": False, "eval_completed": False, "eval_execution_time": None}
    finally:
        await env.close()
        cleanup_handlers(run_id)
    return result


@ray.remote
class TestEvalActor:
    _semaphore = asyncio.Semaphore(max(1, GLOBAL_CONCURRENCY // NUM_WORKERS))

    async def run_single(self, sample):
        async with self._semaphore:
            return await run_sample(sample)


def _rule(text: str = "", width: int = 50, ch: str = "─") -> str:
    """A centered-title horizontal rule. Emoji-safe (left-aligned rows below it
    carry the values, so we never depend on monospace emoji width)."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def main():
    ray.init()
    # data_path = "/home/tiger/data/swe_agent/swe_rebench_filtered.parquet"
    # data_path = "/home/tiger/data/swe_agent/r2e_gym_subset.parquet"
    dataset = load_dataset("parquet", data_files=DATA_PATH, split="train")
    samples = dataset.to_list()
    logger.info(f"loaded {len(samples)} samples from {DATA_PATH}")
    logger.info(
        f"deployment={os.getenv('DEPLOYMENT', 'vefaas')} workers={NUM_WORKERS} concurrency={GLOBAL_CONCURRENCY}"
    )

    workers = [TestEvalActor.remote() for _ in range(NUM_WORKERS)]
    # one future per sample (round-robin across workers) so we can stream
    # per-sample progress; the actor semaphore still bounds real concurrency.
    futures = [workers[i % len(workers)].run_single.remote(s) for i, s in enumerate(samples)]
    fut_to_idx = {f: i for i, f in enumerate(futures)}

    begin_time = time.time()
    results: list = [None] * len(futures)
    ok = wa = tle = 0
    remaining = list(futures)
    with tqdm(
        total=len(futures),
        desc="🚀 eval",
        colour="green",
        unit="inst",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    ) as pbar:
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            for d in done:
                res = ray.get(d)
                results[fut_to_idx[d]] = res
                if res.get("resolved"):
                    ok += 1
                elif res.get("eval_completed"):
                    wa += 1
                else:
                    tle += 1
                rate = ok / pbar.n * 100 if pbar.n else 0.0
                pbar.set_postfix_str(f"✅{ok} ❌WA{wa} ⏱TLE{tle} | {rate:.0f}% pass")
                pbar.update(1)
    end_time = time.time()

    all_num = len(results)
    success_num = len([r for r in results if r.get("resolved")])
    fail_wa_num = len([r for r in results if not r.get("resolved") and r.get("eval_completed")])
    fail_tle_num = len([r for r in results if not r.get("resolved") and not r.get("eval_completed")])

    def instance_name(sample):
        return sample["extra_info"]["tools_kwargs"]["reward"]["metadata"]["instance_id"]

    fail_wa_names = [
        instance_name(sample)
        for sample, item in zip(samples, results, strict=False)
        if not item.get("resolved") and item.get("eval_completed")
    ]
    fail_tle_names = [
        instance_name(sample)
        for sample, item in zip(samples, results, strict=False)
        if not item.get("resolved") and not item.get("eval_completed")
    ]

    exec_times = [r["eval_execution_time"] for r in results if r.get("eval_execution_time") is not None]
    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0
    pass_rate = success_num / all_num * 100 if all_num else 0.0
    wall = end_time - begin_time

    summary = "\n".join(
        [
            "",
            _rule("🧪 eval summary"),
            f"  ✅ resolved   {success_num:>4}   ({pass_rate:.1f}%)",
            f"  ❌ wrong-ans  {fail_wa_num:>4}",
            f"  ⏱  timeout    {fail_tle_num:>4}",
            f"  Σ  total      {all_num:>4}",
            _rule(f"avg {avg_exec_time:.1f}s · wall {wall:.1f}s · n={len(exec_times)}"),
            "",
        ]
    )
    print(summary)

    logger.info(f"fail_wa instance names: {fail_wa_names}")
    logger.info(f"fail_tle instance names: {fail_tle_names}")


if __name__ == "__main__":
    main()
