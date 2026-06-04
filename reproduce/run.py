# ruff: noqa: E501
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

import ray
from tqdm import tqdm

from datasets import load_dataset
from uni_agent.async_logging import add_file_handler, cleanup_handlers
from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.reward import load_reward_spec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 64))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))


async def run_sample(sample):
    run_id = str(uuid.uuid4())
    instance = sample["extra_info"]["tools_kwargs"]
    impl = os.getenv("DEPLOYMENT", "vefaas").lower()

    # SWE preprocessors emit ``env.deployment.image`` (nested, matching
    # ``AgentEnvConfig`` / ``DeployConfig``). Older parquets used flat
    # ``env.image``; accept both so a stale parquet doesn't silently break.
    instance_image = instance["env"].get("deployment", {}).get("image") or instance["env"].get("image")
    if instance_image is None:
        raise KeyError("No image found in instance.env.deployment.image or instance.env.image")

    if impl == "vefaas":
        deployment_config = {
            "type": "vefaas",
            "image": instance_image,
            "command": "curl -fsSL https://vefaas-swe.tos-cn-beijing.ivolces.com/swe-rex/install_1.4.0.sh | bash -s -- {token}",
            "timeout": 600.0,
            "startup_timeout": 180.0,
            "function_id": os.getenv("VEFAAS_FUNCTION_ID"),
            "function_route": os.getenv("VEFAAS_FUNCTION_ROUTE"),
        }
    elif impl == "modal":
        deployment_config = {
            "type": "modal",
            "image": instance_image,
            "startup_timeout": 600.0,
            "runtime_timeout": 600.0,
            "deployment_timeout": 3600.0,
        }
    elif impl == "":
        raise ValueError("DEPLOYMENT must be set")
    else:
        raise ValueError(f"Invalid environment implementation: {impl}")

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
    add_file_handler(Path(f"reproduce/logs/{run_id}.log"), run_id)

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
    _semaphore = asyncio.Semaphore(GLOBAL_CONCURRENCY // NUM_WORKERS)

    async def run_batch(self, samples):
        tasks = [self.run_single(sample) for sample in samples]
        return await asyncio.gather(*tasks)

    async def run_single(self, sample):
        async with self._semaphore:
            return await run_sample(sample)


def main():
    ray.init()
    data_path = "./reproduce/swe_bench_verified_modal.parquet"
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    samples = dataset.to_list()
    workers = [TestEvalActor.remote() for _ in range(NUM_WORKERS)]
    # one future per sample (round-robin across workers) so we can track
    # per-sample progress; the actor semaphore still bounds real concurrency.
    futures = [workers[i % len(workers)].run_single.remote(s) for i, s in enumerate(samples)]
    fut_to_idx = {f: i for i, f in enumerate(futures)}

    begin_time = time.time()
    results = [None] * len(futures)
    ok = wa = tle = 0
    remaining = list(futures)
    with tqdm(
        total=len(futures),
        desc="🚀 eval",
        colour="green",
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
                pbar.set_postfix_str(f"✅{ok} ❌WA{wa} ⏱TLE{tle}")
                pbar.update(1)
    end_time = time.time()
    logger.info(f"time cost: {end_time - begin_time:.2f}s")
    all_num = len(results)
    success_num = len([item for item in results if item.get("resolved")])
    fail_wa_num = len([item for item in results if not item.get("resolved") and item.get("eval_completed")])
    fail_tle_num = len([item for item in results if not item.get("resolved") and not item.get("eval_completed")])

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

    logger.info(
        f"all_num: {all_num}, success_num: {success_num}, fail_wa_num: {fail_wa_num}, fail_tle_num: {fail_tle_num}"
    )
    logger.info(f"avg_execution_time: {avg_exec_time:.2f}s (n={len(exec_times)})")

    logger.info(f"fail_wa instance names: {fail_wa_names}")
    logger.info(f"fail_tle instance names: {fail_tle_names}")


if __name__ == "__main__":
    main()
