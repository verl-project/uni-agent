# ruff: noqa: E501
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

``--task-config`` (a YAML task config; see ``examples/inference/task_config.yaml``)
is required and provides run-level defaults; each sample's task dict is merged on top.
The endpoint (--base-url / --model / --api-key or env BASE_URL / MODEL / API_KEY) is
layered onto agent.model last.
"""

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import ray
import yaml
from datasets import load_dataset
from tqdm import tqdm

from uni_agent.framework.task_runner import resolve_task_config
from uni_agent.tasks import get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 128))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))


class InferenceActor:
    _semaphore = asyncio.Semaphore(max(1, GLOBAL_CONCURRENCY // NUM_WORKERS))

    async def run_single(self, sample: dict, task_defaults: dict) -> dict:
        async with self._semaphore:
            base_task = sample["extra_info"]["tools_kwargs"]["task"]
            instance_id = base_task["metadata"]["instance_id"]
            try:
                # Run-level YAML provides defaults, the sample wins on top, and the
                # live endpoint is injected last.
                model_cfg = task_defaults.get("agent", {}).get("model", {})
                task = resolve_task_config(
                    {"task": base_task},
                    session_base_url=model_cfg.get("base_url"),
                    task_defaults=task_defaults,
                    api_key=model_cfg.get("api_key", "EMPTY"),
                    model_name=model_cfg.get("model_name"),
                )
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


def _load_task_yaml(path: str) -> dict:
    """Load a YAML task-config file into a dict."""
    raw = yaml.safe_load(Path(path).expanduser().read_text())
    if isinstance(raw, list):
        if not raw or not isinstance(raw[0], dict):
            raise ValueError(f"--task-config {path!r}: a YAML list must start with a mapping (the task config).")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(f"--task-config {path!r} must be a YAML mapping (the task config), got {type(raw).__name__}")
    return raw


def build_task_defaults(args: argparse.Namespace) -> dict:
    """Load required run-level Task Config defaults and attach the live endpoint."""
    overrides = _load_task_yaml(args.task_config)

    # endpoint = runtime state, layered onto the agent's model.
    agent = overrides.setdefault("agent", {})
    agent.setdefault("name", args.agent)
    model = agent.setdefault("model", {})
    if args.base_url:
        model["base_url"] = args.base_url
    if args.model:
        model["model_name"] = args.model
    model.setdefault("api_key", args.api_key)

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel agent inference.")
    parser.add_argument(
        "--data-path",
        default=os.getenv("DATA_PATH", os.path.expanduser("~/data/swe_agent/swe_bench_verified.parquet")),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N samples (smoke testing).")
    parser.add_argument("--agent", default="react", help="Registered agent name to run.")
    parser.add_argument(
        "--task-config",
        required=True,
        help="Path to a YAML task config (name/sandbox/agent/...), deep-merged onto each sample's task dict (required).",
    )

    # Policy endpoint
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="OpenAI-compatible endpoint (env BASE_URL).")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "EMPTY"), help="Bearer key (env API_KEY).")
    parser.add_argument("--model", default=os.getenv("MODEL", ""), help="Served model name (env MODEL).")

    parser.add_argument("--n", type=int, default=1, help="Rollouts per instance (pass rate averages over all).")
    args = parser.parse_args()

    task_defaults = build_task_defaults(args)
    agent_cfg = task_defaults.get("agent", {})
    model_cfg = agent_cfg.get("model", {})
    if not model_cfg.get("base_url"):
        logger.error("no policy endpoint: set BASE_URL (or --base-url), or agent.model.base_url in --task-config")
        return

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
    logger.info(
        f"task={task_defaults.get('name') or '<dataset>'} agent={agent_cfg.get('name', args.agent)} "
        f"provider={task_defaults.get('sandbox', {}).get('provider')} "
        f"endpoint={model_cfg.get('base_url')} model={model_cfg.get('model_name') or '<default>'}"
    )
    logger.info(
        f"workers={NUM_WORKERS} concurrency={GLOBAL_CONCURRENCY} "
        f"sampling=temp{model_cfg.get('temperature')}/top_p{model_cfg.get('top_p')}/top_k{model_cfg.get('top_k')} "
        f"max_tokens_per_turn={model_cfg.get('max_tokens_per_turn')} "
        f"max_total_tokens={model_cfg.get('max_total_tokens')} "
        f"config=yaml:{args.task_config}"
    )

    num_workers = min(NUM_WORKERS, len(samples))
    workers = [ray.remote(InferenceActor).remote() for _ in range(num_workers)]
    futures = [workers[i % num_workers].run_single.remote(s, task_defaults) for i, s in enumerate(samples)]

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
