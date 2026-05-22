# ruff: noqa: E501
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

import ray
from datasets import load_dataset

from uni_agent.async_logging import add_file_handler, cleanup_handlers
from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.reward import load_reward_spec

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


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
    add_file_handler(Path(f"/tmp/eval_gold_patch/{run_id}.log"), run_id)

    await env.start()
    await reward_spec.apply_gold_patch()
    _, result = await reward_spec.compute_reward()
    await env.close()
    cleanup_handlers(run_id)
    return result


@ray.remote
class TestEvalActor:
    _semaphore = asyncio.Semaphore(64)

    async def run_batch(self, samples):
        tasks = [self.run_single(sample) for sample in samples]
        return await asyncio.gather(*tasks)

    async def run_single(self, sample):
        async with self._semaphore:
            return await run_sample(sample)


def main():
    ray.init()
    # data_path = "/home/tiger/data/swe_agent/swe_rebench_filtered.parquet"
    # data_path = "/home/tiger/data/swe_agent/r2e_gym_subset.parquet"
    data_path = "/home/tiger/data/swe_agent/swe_bench_verified_modal.parquet"
    dataset = load_dataset("parquet", data_files=data_path, split="train")
    samples = dataset.to_list()
    workers = [TestEvalActor.remote() for _ in range(8)]
    futures = []
    chunk_size = (len(samples) - 1) // len(workers) + 1
    for i in range(len(workers)):
        chunk = samples[i * chunk_size : (i + 1) * chunk_size]
        futures.append(workers[i].run_batch.remote(chunk))
    # each future returns a list of per-sample results (one chunk per worker)

    begin_time = time.time()
    results_chunk = ray.get(futures)
    end_time = time.time()
    logger.info(f"time cost: {end_time - begin_time:.2f}s")
    results = [item for chunk in results_chunk for item in chunk]
    all_num = len(results)
    success_num = len([item for item in results if item["resolved"]])
    fail_wa_num = len([item for item in results if not item["resolved"] and item["eval_completed"]])
    fail_tle_num = len([item for item in results if not item["resolved"] and not item["eval_completed"]])

    def instance_name(sample):
        return sample["extra_info"]["tools_kwargs"]["reward"]["metadata"]["instance_id"]

    fail_wa_names = [
        instance_name(sample)
        for sample, item in zip(samples, results, strict=False)
        if not item["resolved"] and item["eval_completed"]
    ]
    fail_tle_names = [
        instance_name(sample)
        for sample, item in zip(samples, results, strict=False)
        if not item["resolved"] and not item["eval_completed"]
    ]

    exec_times = [r["eval_execution_time"] for r in results if r.get("eval_execution_time") is not None]
    avg_exec_time = sum(exec_times) / len(exec_times)

    logger.info(
        f"all_num: {all_num}, success_num: {success_num}, fail_wa_num: {fail_wa_num}, fail_tle_num: {fail_tle_num}"
    )
    logger.info(f"avg_execution_time: {avg_exec_time:.2f}s (n={len(exec_times)})")

    logger.info(f"fail_wa instance names: {fail_wa_names}")
    logger.info(f"fail_tle instance names: {fail_tle_names}")


if __name__ == "__main__":
    main()
