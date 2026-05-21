# ruff: noqa: E501
"""Parallel gold-solution verification for Terminal-Bench v2.

Each parquet row is self-contained:
  * ``extra_info.tools_kwargs.env`` is a full ``AgentEnvConfig`` dict (modal
    deployment + image + per-task timeouts + env_variables + post_setup_cmd).
  * ``extra_info.tools_kwargs.reward`` includes ``name``, ``metadata``
    (with ``solution_archive`` / ``tests_archive`` blobs) and ``eval_timeout``.

So this script just spreads those into the constructors and runs
``apply_gold_solution`` + ``compute_reward`` per sample, fanned out via Ray.
"""

import argparse
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import ray
from datasets import load_dataset

from uni_agent.async_logging import add_file_handler, cleanup_handlers
from uni_agent.interaction import AgentEnv, AgentEnvConfig
from uni_agent.reward import load_reward_spec

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


SUCCESS_REWARD_THRESHOLD = 0.5


async def run_sample(sample: dict) -> dict:
    run_id = str(uuid.uuid4())
    instance = sample["extra_info"]["tools_kwargs"]

    env_config = AgentEnvConfig(**instance["env"])
    env = AgentEnv(run_id=run_id, env_config=env_config)

    reward_config: dict[str, Any] = {"run_id": run_id, "env": env, **instance["reward"]}
    reward_spec = load_reward_spec(reward_config)

    add_file_handler(Path(f"/tmp/eval_gold_patch_tbench/{run_id}.log"), run_id)
    try:
        await env.start()
        await reward_spec.apply_gold_solution()
        _, result = await reward_spec.compute_reward()
    finally:
        await env.close()
        cleanup_handlers(run_id)
    return result


@ray.remote
class TestEvalActor:
    _semaphore = asyncio.Semaphore(8)

    async def run_batch(self, samples: list[dict]) -> list[dict]:
        tasks = [self.run_single(sample) for sample in samples]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def run_single(self, sample: dict) -> dict:
        async with self._semaphore:
            try:
                return await run_sample(sample)
            except Exception as exc:
                # Surface failures as structured results so aggregation isn't lost.
                return {
                    "eval_completed": False,
                    "eval_execution_time": None,
                    "reward": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }


def _task_id(sample: dict) -> str:
    return sample["extra_info"]["tools_kwargs"]["reward"]["metadata"].get("task_id", "<unknown>")


def _is_resolved(result: dict) -> bool:
    return float(result.get("reward", 0.0)) >= SUCCESS_REWARD_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="/home/tiger/data/swe_agent/terminal_bench_v2_modal.parquet",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only verify the first N samples (smoke testing).",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task_id allowlist (verify only matching rows).",
    )
    args = parser.parse_args()

    ray.init()

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()

    if args.task_ids:
        wanted = {t.strip() for t in args.task_ids.split(",") if t.strip()}
        samples = [s for s in samples if _task_id(s) in wanted]
    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        logger.warning("no samples selected; exiting")
        return

    num_workers = min(args.num_workers, len(samples))
    workers = [TestEvalActor.remote() for _ in range(num_workers)]
    chunk_size = (len(samples) - 1) // num_workers + 1
    futures = [workers[i].run_batch.remote(samples[i * chunk_size : (i + 1) * chunk_size]) for i in range(num_workers)]

    logger.info(f"verifying {len(samples)} samples across {num_workers} workers (chunk_size={chunk_size})")
    begin_time = time.time()
    results_chunk = ray.get(futures)
    elapsed = time.time() - begin_time
    logger.info(f"time cost: {elapsed:.2f}s")

    results = [item for chunk in results_chunk for item in chunk]
    all_num = len(results)
    success_num = sum(_is_resolved(r) for r in results)
    fail_wa_num = sum(not _is_resolved(r) and r.get("eval_completed") for r in results)
    fail_tle_num = sum(not _is_resolved(r) and not r.get("eval_completed") for r in results)

    fail_wa_names = [
        _task_id(s) for s, r in zip(samples, results, strict=True) if not _is_resolved(r) and r.get("eval_completed")
    ]
    fail_tle_names = [
        _task_id(s)
        for s, r in zip(samples, results, strict=True)
        if not _is_resolved(r) and not r.get("eval_completed")
    ]

    exec_times = [r["eval_execution_time"] for r in results if r.get("eval_execution_time") is not None]
    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0

    logger.info(
        f"all_num: {all_num}, success_num: {success_num}, fail_wa_num: {fail_wa_num}, fail_tle_num: {fail_tle_num}"
    )
    logger.info(f"avg_execution_time: {avg_exec_time:.2f}s (n={len(exec_times)})")
    logger.info(f"fail_wa task_ids (test ran, reward<{SUCCESS_REWARD_THRESHOLD}): {fail_wa_names}")
    logger.info(f"fail_tle task_ids (test did not complete): {fail_tle_names}")

    errored = [(_task_id(s), r["error"]) for s, r in zip(samples, results, strict=True) if "error" in r]
    if errored:
        logger.warning(f"{len(errored)} samples raised exceptions:")
        for name, err in errored:
            logger.warning(f"  {name}: {err}")


if __name__ == "__main__":
    main()
