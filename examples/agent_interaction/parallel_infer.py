# ruff: noqa: E501
"""Parallel agent inference for SWE-bench.

Bring up an OpenAI-compatible policy server, then run this against it:

    vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --served-model-name Qwen3-Coder-30B-A3B-Instruct \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --tensor-parallel-size 4

    BASE_URL=http://localhost:8000/v1 MODEL=Qwen3-Coder-30B-A3B-Instruct \
        python examples/agent_interaction/parallel_infer.py --limit 8

By default the agent config is built from the per-flag knobs. Pass ``--task-config``
to load a YAML task config instead, deep-merged onto each sample's task dict
(overriding agent / sandbox / ... while the per-sample image + metadata survive). A
one-item ``- name: ...`` list wrapper is accepted:

    - name: swe_bench
      agent:
        name: code_act
        max_steps: 100
        tools:
          - name: stateful_shell
            command_timeout: 120
          - name: str_replace_editor
        model:
          temperature: 0.8
          top_p: 0.9
          top_k: -1
          max_total_tokens: 65536

Either way, the endpoint (--base-url / --model / --api-key or env BASE_URL / MODEL /
API_KEY) is layered onto agent.model last.
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

from uni_agent.agents import get_agent_cls
from uni_agent.tasks import get_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


GLOBAL_CONCURRENCY = int(os.getenv("GLOBAL_CONCURRENCY", 128))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))


class InferenceActor:
    _semaphore = asyncio.Semaphore(max(1, GLOBAL_CONCURRENCY // NUM_WORKERS))

    async def run_single(self, sample: dict, task_overrides: dict) -> dict:
        async with self._semaphore:
            base_task = sample["extra_info"]["tools_kwargs"]["task"]
            instance_id = base_task["metadata"]["instance_id"]
            try:
                # Deep-merge this run's task config onto the dataset's per-sample task
                # (overrides win; the sample's image + metadata survive), then let
                # get_task parse it (the agent registry resolves the concrete config).
                task = _deep_merge(base_task, task_overrides)
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


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` on top of ``base``, returning a new dict.

    Nested dicts merge key-wise (``overrides`` wins); lists and scalars replace
    wholesale. ``base`` is never mutated. (Same semantics as the agent-loop's.)
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    result = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_task_yaml(path: str) -> dict:
    """Load a YAML task-config file into a dict.

    Accepts either a bare mapping or a one-item ``- name: ...`` list wrapping it
    (the list form is the convention used elsewhere in the repo, e.g. the agent-loop
    configs loaded via ``yaml.safe_load(...)[0]``).
    """
    raw = yaml.safe_load(Path(path).expanduser().read_text())
    if isinstance(raw, list):
        if not raw or not isinstance(raw[0], dict):
            raise ValueError(f"--task-config {path!r}: a YAML list must start with a mapping (the task config).")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(f"--task-config {path!r} must be a YAML mapping (the task config), got {type(raw).__name__}")
    return raw


def build_task_overrides(args: argparse.Namespace) -> dict:
    """Build the task-config dict deep-merged onto every sample's task.

    Two sources, one shape:

    * ``--task-config FILE`` -- a (partial) task config YAML; its ``agent`` section
      supersedes the per-flag knobs.
    * otherwise -- the per-flag agent knobs wrapped as ``{"agent": {...}}``.

    Either way, the endpoint (``--base-url`` / ``--model`` / ``--api-key``, env
    ``BASE_URL`` / ``MODEL`` / ``API_KEY``) is then layered onto ``agent.model``.
    """
    if args.task_config:
        overrides = _load_task_yaml(args.task_config)
    else:
        model_cfg: dict = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        if args.max_tokens is not None:
            model_cfg["max_tokens_per_turn"] = args.max_tokens
        if args.max_total_tokens is not None:
            model_cfg["max_total_tokens"] = args.max_total_tokens
        agent_cfg: dict = {"name": args.agent, "model": model_cfg}
        # max_steps is code_act's turn budget; include it only for agents that declare it.
        if "max_steps" in get_agent_cls(args.agent).config_model.model_fields:
            agent_cfg["max_steps"] = args.max_steps
        overrides = {"agent": agent_cfg}

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
    parser.add_argument("--agent", default="code_act", help="Registered agent name to run.")
    parser.add_argument(
        "--task-config",
        help="Path to a YAML task config (name/sandbox/agent/...) deep-merged onto each sample's task dict. "
        "Its `agent` section supersedes the per-flag knobs; the endpoint "
        "(--base-url/--model/--api-key, env BASE_URL/MODEL/API_KEY) is still layered on.",
    )

    # Policy endpoint
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="OpenAI-compatible endpoint (env BASE_URL).")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "EMPTY"), help="Bearer key (env API_KEY).")
    parser.add_argument("--model", default=os.getenv("MODEL", ""), help="Served model name (env MODEL).")

    # Sampling / rollout knobs (keep temperature/top_p/top_k aligned with training).
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=-1, help="Top-k sampling; -1 disables it.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-turn generation cap (max_tokens per model response); omit to fall back to --max-total-tokens.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=65536,
        help="Whole-episode generation budget (sum of completion tokens across turns); omit for unbounded.",
    )
    parser.add_argument("--max-steps", type=int, default=100, help="Max tool-calling turns per episode.")
    parser.add_argument("--n", type=int, default=1, help="Rollouts per instance (pass rate averages over all).")
    args = parser.parse_args()

    task_overrides = build_task_overrides(args)
    agent_cfg = task_overrides.get("agent", {})
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
        f"task={task_overrides.get('name') or '<dataset>'} agent={agent_cfg.get('name', args.agent)} "
        f"provider={task_overrides.get('sandbox', {}).get('provider')} "
        f"endpoint={model_cfg.get('base_url')} model={model_cfg.get('model_name') or '<default>'}"
    )
    logger.info(
        f"workers={NUM_WORKERS} concurrency={GLOBAL_CONCURRENCY} "
        f"sampling=temp{model_cfg.get('temperature')}/top_p{model_cfg.get('top_p')}/top_k{model_cfg.get('top_k')} "
        f"max_tokens_per_turn={model_cfg.get('max_tokens_per_turn')} "
        f"max_total_tokens={model_cfg.get('max_total_tokens')} "
        f"config={('yaml:' + args.task_config) if args.task_config else 'flags'}"
    )

    num_workers = min(NUM_WORKERS, len(samples))
    workers = [ray.remote(InferenceActor)() for _ in range(num_workers)]
    futures = [workers[i % num_workers].run_single.remote(s, task_overrides) for i, s in enumerate(samples)]

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
