"""Parallel oracle-solution verification.

Runs each dataset row task in oracle mode (``run_oracle_solution=True``):
results are counted by status and streamed to a live progress bar.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import ray
from datasets import load_dataset
from tqdm import tqdm

from uni_agent.logging import LogContext, sample_logging
from uni_agent.tasks import TaskConfigResolver, get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
logger = logging.getLogger(__name__)

GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 512))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))


class OracleEvalActor:
    def __init__(self, log_dir: str | None, max_concurrency: int):
        self.log_dir = log_dir
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single(self, task_config: dict) -> dict:
        async with self._semaphore:
            instance_id = task_config["metadata"]["instance_id"]
            log_id = f"verify-{uuid4().hex}"
            log_path = str(Path(self.log_dir).expanduser() / log_id / "task.log") if self.log_dir else None
            async with sample_logging.from_context(LogContext(log_id, log_path)):
                try:
                    result = await get_task(task_config).run()
                    info = result.extra_info or {}
                    solved = bool(info.get("resolved", result.reward))
                    task_result = {
                        "instance_id": instance_id,
                        "log_id": log_id,
                        "solved": solved,
                        "eval_execution_time": info.get("eval_execution_time"),
                        "status": "completed",
                    }
                    return task_result
                except Exception as e:
                    logger.error(f"error verifying {instance_id}: {type(e).__name__}: {e}")
                    return {
                        "instance_id": instance_id,
                        "log_id": log_id,
                        "solved": False,
                        "eval_execution_time": None,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                    }


def _prepare_task(sample: dict, resolver: TaskConfigResolver) -> dict:
    """Merge run-level Task Config (including ``sandbox.image_map``) onto the sample, then pin oracle eval."""
    sample_config = sample["extra_info"]["tools_kwargs"]["task"]
    task_config = resolver.resolve(sample_config)
    sandbox_provider = os.getenv("SANDBOX_PROVIDER")
    if sandbox_provider:
        task_config["sandbox"]["provider"] = sandbox_provider
    task_config["run_oracle_solution"] = True
    return task_config


def _rule(text: str = "", width: int = 50, ch: str = "-") -> str:
    """A centered-title horizontal rule."""
    if not text:
        return ch * width
    pad = max(0, width - len(text) - 2)
    return f"{ch * (pad // 2)} {text} {ch * (pad - pad // 2)}"


def _allocate_worker_concurrency(total_concurrency: int, num_workers: int) -> list[int]:
    """Split a global concurrency budget across Ray actors without exceeding it."""
    per_worker, remainder = divmod(total_concurrency, num_workers)
    return [per_worker + (worker_index < remainder) for worker_index in range(num_workers)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=GLOBAL_CONCURRENCY,
        help="Maximum in-flight oracle tasks across all Ray actors (env GLOBAL_CONCURRENCY).",
    )
    parser.add_argument(
        "--task-config",
        default=None,
        help="Run-level Task Config YAML.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only verify the first N samples (smoke testing).")
    parser.add_argument(
        "--log-dir",
        default=os.getenv("UNI_AGENT_LOG_DIR", "/tmp/eval_gold_patch"),
        help="Root directory for per-sample logs; use an empty value to disable file logging.",
    )
    parser.add_argument("--result-path", default=None, help="Optional JSON path for summary and per-sample results.")
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")

    ray.init()

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return

    resolver = TaskConfigResolver.from_file(args.task_config) if args.task_config else TaskConfigResolver()
    try:
        tasks = [_prepare_task(sample, resolver) for sample in samples]
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("failed to resolve Task Config: %s", exc)
        return

    num_workers = min(args.num_workers, len(tasks), args.concurrency)
    worker_concurrency = _allocate_worker_concurrency(args.concurrency, num_workers)
    sandbox_providers = sorted({task["sandbox"]["provider"] for task in tasks})

    logger.info(f"loaded {len(tasks)} samples from {args.data_path}")
    logger.info(
        "providers=%s workers=%s concurrency=%s worker_concurrency=%s config=%s",
        sandbox_providers,
        num_workers,
        args.concurrency,
        worker_concurrency,
        args.task_config or "parquet",
    )

    workers = [
        ray.remote(OracleEvalActor).remote(args.log_dir, max_concurrency) for max_concurrency in worker_concurrency
    ]
    # One future per sample (round-robin across workers) so we can stream
    # per-sample progress; the actor semaphore still bounds real concurrency.
    futures = [workers[i % num_workers].run_single.remote(task) for i, task in enumerate(tasks)]
    fut_to_idx = {f: i for i, f in enumerate(futures)}

    begin_time = time.time()
    results: list = [None] * len(futures)
    status_counts = {
        "completed": 0,
        "error": 0,
    }
    solved_count = 0
    unsolved_count = 0
    remaining = list(futures)
    with tqdm(
        total=len(futures),
        desc="oracle verification",
        unit="sample",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    ) as pbar:
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            for d in done:
                res = ray.get(d)
                results[fut_to_idx[d]] = res
                status_counts[res["status"]] += 1
                solved_count += int(res["solved"])
                unsolved_count += int(not res["solved"])
                pbar.set_postfix_str(f"solved={solved_count} unsolved={unsolved_count}")
                pbar.update(1)
    wall = time.time() - begin_time

    all_num = len(results)
    solved_rate = solved_count / all_num * 100 if all_num else 0.0
    exec_times = [result["eval_execution_time"] for result in results if result["eval_execution_time"] is not None]
    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0

    summary = "\n".join(
        [
            "",
            _rule("oracle summary"),
            f"  solved     {solved_count:>4}   ({solved_rate:.1f}%)",
            f"  completed  {status_counts['completed']:>4}",
            f"  error      {status_counts['error']:>4}",
            f"  total      {all_num:>4}",
            _rule(f"avg {avg_exec_time:.1f}s | wall {wall:.1f}s"),
            "",
        ]
    )
    print(summary)

    unsolved_instance_ids = [result["instance_id"] for result in results if not result["solved"]]
    logger.info("unsolved instance ids: %s", unsolved_instance_ids)

    errored = [(result["instance_id"], result["error"]) for result in results if result.get("error")]
    if errored:
        logger.warning(f"{len(errored)} samples raised exceptions (showing up to 10):")
        for name, err in errored[:10]:
            logger.warning(f"  {name}: {err}")

    summary_data = {
        "data_path": args.data_path,
        "total": all_num,
        "status_counts": status_counts,
        "solved": solved_count,
        "solved_rate": solved_rate / 100.0,
        "average_eval_execution_time": avg_exec_time,
        "wall_time": wall,
    }
    if args.result_path:
        output_path = Path(args.result_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary_data, "results": results}, indent=2, default=str))
        logger.info("wrote oracle results to %s", output_path)


if __name__ == "__main__":
    main()
