"""Parallel gold-patch verification for SWE-bench.

Runs each dataset row's SWE-bench task in oracle mode (``run_gold_patch=True``):
apply the gold patch in the sandbox, run the tests, and score. Every instance
should resolve -- it's the data-quality baseline you run before training. Results
are bucketed as resolved (ok) / wrong-answer (wa) / timeout-or-error (tle) and
streamed to a live progress bar.
"""

import argparse
import asyncio
import logging
import os
import time

import ray
from datasets import load_dataset
from tqdm import tqdm

from uni_agent.tasks import get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)

GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 512))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
SANDBOX_PROVIDER = os.getenv("SANDBOX_PROVIDER", "modal")
RUNTIME_TIMEOUT = float(os.getenv("RUNTIME_TIMEOUT", 3600))


@ray.remote
class TestEvalActor:
    _semaphore = asyncio.Semaphore(max(1, GLOBAL_CONCURRENCY // NUM_WORKERS))

    async def run_single(self, sample: dict) -> dict:
        async with self._semaphore:
            task_config = sample["extra_info"]["tools_kwargs"]["task"]
            instance_id = task_config["metadata"]["instance_id"]
            try:
                task_config["run_gold_patch"] = True
                task_config["log_dir"] = "/tmp/eval_gold_patch"
                task_config["sandbox"]["provider"] = SANDBOX_PROVIDER
                task_config["sandbox"]["runtime_timeout"] = RUNTIME_TIMEOUT
                result = await get_task(task_config).run()
                info = result.info or {}
                resolved = bool(info.get("resolved", result.reward))
                return {
                    "instance_id": instance_id,
                    "resolved": resolved,
                    "eval_completed": bool(info.get("eval_completed", True)),
                    "eval_execution_time": info.get("eval_execution_time"),
                }
            except Exception as e:
                logger.error(f"error verifying {instance_id}: {type(e).__name__}: {e}")
                return {
                    "instance_id": instance_id,
                    "resolved": False,
                    "eval_completed": False,
                    "eval_execution_time": None,
                    "error": f"{type(e).__name__}: {e}",
                }


def _rule(text: str = "", width: int = 50, ch: str = "-") -> str:
    """A centered-title horizontal rule."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")),
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Only verify the first N samples (smoke testing).")
    args = parser.parse_args()

    ray.init()

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return

    logger.info(f"loaded {len(samples)} samples from {args.data_path}")
    logger.info(f"provider={SANDBOX_PROVIDER} workers={args.num_workers} concurrency={GLOBAL_CONCURRENCY}")

    num_workers = min(args.num_workers, len(samples))
    workers = [TestEvalActor.remote() for _ in range(num_workers)]
    # One future per sample (round-robin across workers) so we can stream
    # per-sample progress; the actor semaphore still bounds real concurrency.
    futures = [workers[i % num_workers].run_single.remote(s) for i, s in enumerate(samples)]
    fut_to_idx = {f: i for i, f in enumerate(futures)}

    begin_time = time.time()
    results: list = [None] * len(futures)
    ok = wa = tle = 0
    remaining = list(futures)
    with tqdm(
        total=len(futures),
        desc="eval",
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
                rate = ok / (pbar.n + 1) * 100
                pbar.set_postfix_str(f"resolved={ok} WA={wa} TLE={tle} | {rate:.0f}% pass")
                pbar.update(1)
    wall = time.time() - begin_time

    all_num = len(results)
    success_num = sum(1 for r in results if r.get("resolved"))
    fail_wa_num = sum(1 for r in results if not r.get("resolved") and r.get("eval_completed"))
    fail_tle_num = sum(1 for r in results if not r.get("resolved") and not r.get("eval_completed"))

    fail_wa_names = [r["instance_id"] for r in results if not r.get("resolved") and r.get("eval_completed")]
    fail_tle_names = [r["instance_id"] for r in results if not r.get("resolved") and not r.get("eval_completed")]

    exec_times = [r["eval_execution_time"] for r in results if r.get("eval_execution_time") is not None]
    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0
    pass_rate = success_num / all_num * 100 if all_num else 0.0

    summary = "\n".join(
        [
            "",
            _rule("eval summary"),
            f"  resolved    {success_num:>4}   ({pass_rate:.1f}%)",
            f"  wrong-ans   {fail_wa_num:>4}",
            f"  timeout/err {fail_tle_num:>4}",
            f"  total       {all_num:>4}",
            _rule(f"avg {avg_exec_time:.1f}s | wall {wall:.1f}s | n={len(exec_times)}"),
            "",
        ]
    )
    print(summary)

    logger.info(f"fail_wa instance names: {fail_wa_names}")
    logger.info(f"fail_tle instance names: {fail_tle_names}")

    errored = [(r["instance_id"], r["error"]) for r in results if r.get("error")]
    if errored:
        logger.warning(f"{len(errored)} samples raised exceptions (showing up to 10):")
        for name, err in errored[:10]:
            logger.warning(f"  {name}: {err}")


if __name__ == "__main__":
    main()
