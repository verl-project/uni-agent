"""Parallel agent inference against a running OpenAI-compatible API.

Talks *directly* to a policy server you started yourself (no GPUs on the driver).
For the variant that has verl bring the engine up and routes rollouts through the
agent framework gateway, see ``parallel_infer_verl.py``.

Bring up an OpenAI-compatible policy server, then run this against it:

    vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --served-model-name Qwen3-Coder-30B-A3B-Instruct \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --tensor-parallel-size 4

    BASE_URL=http://localhost:8000/v1 MODEL=Qwen3-Coder-30B-A3B-Instruct \
        python examples/inference/parallel_infer_api.py \
        --task-config examples/inference/task_config.yaml --limit 8

``--task-config`` (see ``examples/inference/task_config.yaml``) accepts one Task
Config mapping or a list keyed by ``name``. Each sample routes to the matching
entry and is merged on top. The endpoint (--base-url / --model / --api-key or env
BASE_URL / MODEL / API_KEY) is layered onto agent.model last.
"""

import argparse
import asyncio
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 128))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))


class InferenceActor:
    def __init__(self, max_concurrency: int):
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single(self, task: dict, log_context: LogContext) -> dict:
        async with self._semaphore:
            instance_id = task.get("metadata", {}).get("instance_id", "<unknown>")
            async with sample_logging.from_context(log_context):
                try:
                    result = await get_task(task).run()
                    info = result.info or {}
                    resolved = bool(info.get("resolved", result.reward))
                    return {
                        "instance_id": instance_id,
                        "log_id": log_context.log_id,
                        "resolved": resolved,
                        "eval_completed": bool(info.get("eval_completed", True)),
                        "eval_execution_time": info.get("eval_execution_time"),
                    }
                except Exception as e:
                    logger.error(f"error running {instance_id}: {type(e).__name__}: {e}")
                    return {
                        "instance_id": instance_id,
                        "log_id": log_context.log_id,
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


def _allocate_worker_concurrency(total_concurrency: int, num_workers: int) -> list[int]:
    """Split a global concurrency budget across Ray actors without exceeding it."""
    per_worker, remainder = divmod(total_concurrency, num_workers)
    return [per_worker + (worker_index < remainder) for worker_index in range(num_workers)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel agent inference.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples (smoke testing).")
    parser.add_argument(
        "--task-config",
        required=True,
        help="YAML mapping or list of Task Config defaults, routed by each sample's task name (required).",
    )

    # Policy endpoint
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="OpenAI-compatible endpoint (env BASE_URL).")
    parser.add_argument("--api-key", default=os.getenv("API_KEY"), help="Bearer key (env API_KEY).")
    parser.add_argument("--model", default=os.getenv("MODEL", ""), help="Served model name (env MODEL).")

    parser.add_argument("--n", type=int, default=1, help="Rollouts per instance (pass rate averages over all).")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=GLOBAL_CONCURRENCY,
        help="Maximum in-flight rollouts across all Ray actors (env GLOBAL_CONCURRENCY).",
    )
    parser.add_argument(
        "--log-dir",
        default=os.getenv("UNI_AGENT_LOG_DIR", "/tmp/uni_agent_logs"),
        help="Root directory for per-rollout logs; use an empty value to disable file logging.",
    )
    args = parser.parse_args()
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if NUM_WORKERS <= 0:
        parser.error("NUM_WORKERS must be positive")

    resolver = TaskConfigResolver.from_file(args.task_config)
    runtime_model = {"api_key": args.api_key}
    if args.base_url:
        runtime_model["base_url"] = args.base_url
    if args.model:
        runtime_model["model_name"] = args.model

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        logger.warning("no samples selected; exiting")
        return

    resolved_tasks: list[dict] = []
    try:
        for sample in samples:
            sample_config = sample["extra_info"]["tools_kwargs"]["task"]
            resolved = resolver.resolve(sample_config, runtime_model=runtime_model)
            task_name = resolved["name"]
            if not resolved.get("agent", {}).get("model", {}).get("base_url"):
                raise ValueError(f"no policy endpoint for sample task {task_name!r}")
            resolved_tasks.append(resolved)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("failed to resolve Task Config: %s", exc)
        return

    n = max(1, args.n)
    work_items: list[tuple[dict, LogContext]] = []
    for sample_index, task in enumerate(resolved_tasks):
        for rollout_index in range(n):
            log_id = f"external-{sample_index}-{rollout_index}-{uuid4().hex}"
            log_path = str(Path(args.log_dir).expanduser() / log_id / "task.log") if args.log_dir else None
            work_items.append((task, LogContext(log_id=log_id, log_path=log_path)))

    num_workers = min(NUM_WORKERS, len(work_items), args.concurrency)
    worker_concurrency = _allocate_worker_concurrency(args.concurrency, num_workers)

    logger.info(f"loaded {len(work_items)} rollouts ({n}x) from {args.data_path}")
    logger.info(
        "workers=%s concurrency=%s worker_concurrency=%s config=yaml:%s",
        num_workers,
        args.concurrency,
        worker_concurrency,
        args.task_config,
    )

    workers = [ray.remote(InferenceActor).remote(max_concurrency) for max_concurrency in worker_concurrency]
    futures = [
        workers[i % num_workers].run_single.remote(task, log_context)
        for i, (task, log_context) in enumerate(work_items)
    ]

    fut_to_idx = {f: i for i, f in enumerate(futures)}

    begin_time = time.time()
    results: list = [None] * len(futures)
    ok = wa = tle = 0
    remaining = list(futures)
    with tqdm(
        total=len(futures),
        desc="infer",
        unit="roll",
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

    fail_wa_names = sorted({r["instance_id"] for r in results if not r.get("resolved") and r.get("eval_completed")})
    fail_tle_names = sorted(
        {r["instance_id"] for r in results if not r.get("resolved") and not r.get("eval_completed")}
    )

    exec_times = [r["eval_execution_time"] for r in results if r.get("eval_execution_time") is not None]
    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0.0
    pass_rate = success_num / all_num * 100 if all_num else 0.0

    summary = "\n".join(
        [
            "",
            _rule("inference summary"),
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
        logger.warning(f"{len(errored)} rollouts raised exceptions (showing up to 10):")
        for name, err in errored[:10]:
            logger.warning(f"  {name}: {err}")


if __name__ == "__main__":
    main()
