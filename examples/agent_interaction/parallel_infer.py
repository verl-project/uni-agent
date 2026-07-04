# ruff: noqa: E501
"""Parallel agent inference for SWE-bench.

Point it at an already-running server and a preprocessed parquet:

    BASE_URL=http://localhost:8000/v1 MODEL=Qwen3-Coder-30B-A3B-Instruct \
        python examples/agent_interaction/parallel_infer.py --limit 8
"""

import argparse
import asyncio
import logging
import os
import time

import ray
from datasets import load_dataset
from tqdm import tqdm

from uni_agent.agents import get_agent_cls
from uni_agent.tasks import get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Infra knobs, injected via env (same as parallel_verify_swe.py).
GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 128))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
SANDBOX_PROVIDER = os.getenv("SANDBOX_PROVIDER", "modal")
RUNTIME_TIMEOUT = float(os.getenv("RUNTIME_TIMEOUT", 3600))


@ray.remote
class InferenceActor:
    _semaphore = asyncio.Semaphore(max(1, GLOBAL_CONCURRENCY // NUM_WORKERS))

    async def run_single(self, sample: dict, agent_cfg: dict) -> dict:
        async with self._semaphore:
            task = sample["extra_info"]["tools_kwargs"]["task"]
            instance_id = task["metadata"]["instance_id"]
            try:
                # Splice inference-time knobs onto the dataset's task dict, then let
                # get_task parse it (the agent registry resolves the concrete config).
                task["sandbox"]["provider"] = SANDBOX_PROVIDER
                task["sandbox"]["runtime_timeout"] = RUNTIME_TIMEOUT
                task["agent"] = agent_cfg
                result = await get_task(task).run()
                info = result.info or {}
                resolved = bool(info.get("resolved", result.reward))
                return {
                    "instance_id": instance_id,
                    "resolved": resolved,
                    "eval_completed": bool(info.get("eval_completed", True)),
                    "eval_execution_time": info.get("eval_execution_time"),
                }
            except Exception as e:
                logger.error(f"error running {instance_id}: {type(e).__name__}: {e}")
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
    parser = argparse.ArgumentParser(description="Parallel agent inference for SWE-bench.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")),
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples (smoke testing).")
    parser.add_argument("--agent", default=os.getenv("AGENT", "code_act"), help="Registered agent name to run.")

    # Policy endpoint: the server is started out-of-band; we only point at it.
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="OpenAI-compatible endpoint (env BASE_URL).")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "EMPTY"), help="Bearer key (env API_KEY).")
    parser.add_argument("--model", default=os.getenv("MODEL", ""), help="Served model name (env MODEL).")

    # Sampling / rollout knobs.
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max tokens per model response.")
    parser.add_argument("--max-steps", type=int, default=50, help="Max tool-calling turns per episode.")
    parser.add_argument("--n", type=int, default=1, help="Rollouts per instance (pass rate averages over all).")
    args = parser.parse_args()

    if not args.base_url:
        logger.error("no policy endpoint: set BASE_URL (or --base-url) to the OpenAI-compatible server")
        return

    sampling_params: dict = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    if args.model:
        sampling_params["model"] = args.model  # forwarded to the OpenAI chat.completions call
    agent_cfg: dict = {
        "name": args.agent,
        "model": {"base_url": args.base_url, "api_key": args.api_key, "sampling_params": sampling_params},
    }
    # max_steps is code_act's turn budget; include it only for agents that declare it.
    if "max_steps" in get_agent_cls(args.agent).config_model.model_fields:
        agent_cfg["max_steps"] = args.max_steps

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    samples = dataset.to_list()
    if args.limit is not None:
        samples = samples[: args.limit]
    n = max(1, args.n)
    samples = [s for s in samples for _ in range(n)]  # fan out N rollouts per instance
    if not samples:
        logger.warning("no samples selected; exiting")
        return

    logger.info(f"loaded {len(samples)} rollouts ({n}x) from {args.data_path}")
    logger.info(f"agent={args.agent} provider={SANDBOX_PROVIDER} endpoint={args.base_url} model={args.model or '<default>'}")
    logger.info(f"workers={args.num_workers} concurrency={GLOBAL_CONCURRENCY} sampling={sampling_params}")

    num_workers = min(args.num_workers, len(samples))
    workers = [InferenceActor.remote() for _ in range(num_workers)]
    # One future per rollout (round-robin across workers) so we can stream
    # per-rollout progress; the actor semaphore still bounds real concurrency.
    futures = [workers[i % num_workers].run_single.remote(s, agent_cfg) for i, s in enumerate(samples)]
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
    fail_tle_names = sorted({r["instance_id"] for r in results if not r.get("resolved") and not r.get("eval_completed")})

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
