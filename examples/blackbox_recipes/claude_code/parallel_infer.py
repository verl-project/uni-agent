"""Standalone inference runner for the blackbox claude-code recipe.

Spins up vLLM + gateway + a reward worker, runs agent sessions in parallel,
and reports resolve rate. Does NOT start the Megatron trainer.

Reuses the recipe's existing training config
(config/claude_code_megatron_v1.yaml); its megatron/optimizer sections are
inert here since this driver never builds the actor worker group — only the
rollout, agent_framework, model, and reward sections are read.

Usage:
    python examples/blackbox_recipes/claude_code/parallel_infer.py \
        --model-path ~/models/Qwen3.5-9B \
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \
        --max-samples 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np
import ray

from uni_agent.framework.entry import build_agent_framework, build_gateway_manager
from verl.experimental.reward_loop.reward_loop import RewardLoopWorker
from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import tq
from verl.workers.rollout.llm_server import LLMServerManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    force=True,
)
logger = logging.getLogger(__name__)

# ── Recipe-specific constants (only these two differ between recipes) ──────
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
_CONFIG_NAME = "claude_code_megatron_v1"
_DEFAULT_TOOL_IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest"


# =====================================================================
# Dataset loading (inlined; keeps the driver self-contained)
# =====================================================================


def _remap_image_to_local(image_name: str) -> str:
    parts = image_name.split("/")
    if len(parts) > 1 and "." in parts[0]:
        basename = parts[-1]
    else:
        basename = image_name
    basename = basename.replace("_1776_", "__")
    if ":" in basename:
        basename = basename.rsplit(":", 1)[0]
    return f"{basename}:latest"


def _remap_sample_images(sample: dict[str, Any]) -> dict[str, Any]:
    extra_info = sample.get("extra_info")
    if not extra_info:
        return sample
    tools_kwargs = extra_info.get("tools_kwargs", {})
    env = tools_kwargs.get("env", {})
    image = env.get("image")
    if not image:
        return sample
    local_image = _remap_image_to_local(image)
    if local_image != image:
        logger.debug("Remapping image: %s -> %s", image, local_image)
        env["image"] = local_image
    return sample


def _inject_reward_fields(sample: dict[str, Any]) -> None:
    extra_info = sample.get("extra_info", {})
    tools_kwargs = extra_info.get("tools_kwargs", {})
    reward_config = tools_kwargs.get("reward", {})
    sample.setdefault("data_source", reward_config.get("name", "unknown"))
    sample.setdefault("reward_model", {"ground_truth": {}})


def load_swe_dataset(data_path: str, max_samples: int = -1) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = os.path.expanduser(data_path)
    logger.info("Loading dataset from: %s", path)
    samples = pq.read_table(path).to_pylist()
    for i, sample in enumerate(samples):
        samples[i] = _remap_sample_images(sample)
        _inject_reward_fields(samples[i])
    if max_samples > 0:
        samples = samples[:max_samples]
    logger.info("Loaded %d samples", len(samples))
    return samples


# =====================================================================
# Config
# =====================================================================


def _load_config(
    *,
    model_path: str,
    engine: str,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    n: int,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
    gateway_count: int,
    max_concurrent_sessions: int,
    tool_image: str | None,
    run_timeout: int,
) -> Any:
    """Compose the recipe's training config and override inference fields.

    The megatron/actor/optimizer sections are left untouched and never read.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        config = compose(config_name=_CONFIG_NAME)

    OmegaConf.set_struct(config, False)

    config.actor_rollout_ref.model.path = os.path.expanduser(model_path)

    ro = config.actor_rollout_ref.rollout
    ro.name = engine
    ro.mode = "async"
    ro.prompt_length = prompt_length
    ro.response_length = response_length
    ro.max_model_len = prompt_length + response_length + 1024
    ro.max_num_batched_tokens = ro.max_model_len
    ro.n = n
    ro.temperature = temperature
    ro.top_p = top_p
    ro.tensor_model_parallel_size = tensor_parallel_size
    ro.gpu_memory_utilization = float(os.getenv("ROLLOUT_GPU_MEM_UTIL", "0.7"))
    ro.nnodes = nnodes
    ro.n_gpus_per_node = n_gpus_per_node
    ro.calculate_log_probs = True
    ro.enable_sleep_mode = False

    af = ro.custom.agent_framework
    af.gateway_count = gateway_count
    runner_name = next(iter(af.agent_runners.keys()))
    runner_cfg = af.agent_runners[runner_name]
    runner_cfg.max_concurrent_sessions = max_concurrent_sessions
    if tool_image:
        runner_cfg.runner_kwargs.tool_image = tool_image
    runner_cfg.runner_kwargs.run_timeout = run_timeout

    config.trainer.nnodes = nnodes
    config.trainer.n_gpus_per_node = n_gpus_per_node

    OmegaConf.set_struct(config, True)
    return config


# =====================================================================
# Batch + score capture
# =====================================================================


def _build_prompts(samples: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    raw_prompts = [sample["prompt"] for sample in samples]
    uids = [str(uuid4()) for _ in samples]
    tools_kwargs_list = [dict((sample.get("extra_info") or {}).get("tools_kwargs", {})) for sample in samples]
    prompts = tu.get_tensordict(
        tensor_dict={
            "raw_prompt": raw_prompts,
            "uid": uids,
            "data_source": [sample["data_source"] for sample in samples],
            "reward_model": [sample["reward_model"] for sample in samples],
            "tools_kwargs": tools_kwargs_list,
        },
        non_tensor_dict={"global_steps": 0},
    )
    return prompts, uids


def _install_tq_capture() -> tuple[dict[str, float], dict[str, str]]:
    """Monkeypatch the process-local TransferQueue to capture rm_scores in-memory.

    Runner dispatch is a Ray task, but session finalize/score/TQ-writes happen
    in this driver process, so patching ``tq`` here captures every write.
    """
    captured_scores: dict[str, float] = {}
    uid_status: dict[str, str] = {}

    async def _fake_put(*, key, partition_id=None, tag=None, **kwargs):
        if isinstance(tag, dict) and "status" in tag:
            uid_status[str(key)] = str(tag["status"])

    async def _fake_batch_put(*, keys=None, fields=None, tags=None, partition_id=None, **kwargs):
        if fields is None or keys is None or "rm_scores" not in fields:
            return
        rm = fields["rm_scores"]  # nested tensor; rm[i] is trajectory i's response scores
        for i, key in enumerate(keys):
            row = rm[i]
            captured_scores[str(key)] = float(row[-1].item()) if row.numel() else 0.0

    tq.async_kv_put = _fake_put
    tq.async_kv_batch_put = _fake_batch_put
    return captured_scores, uid_status


def _report(samples, uids, captured_scores) -> dict[str, Any]:
    uid_to_index = {uid: i for i, uid in enumerate(uids)}
    per_sample_sum = [0.0] * len(samples)
    per_sample_cnt = [0] * len(samples)
    for key, score in captured_scores.items():
        # key format: {uid}_{session_index}_{index}
        uid = key.rsplit("_", 2)[0]
        idx = uid_to_index.get(uid)
        if idx is None:
            continue
        per_sample_sum[idx] += score
        per_sample_cnt[idx] += 1
    per_sample_scores = [
        per_sample_sum[i] / per_sample_cnt[i] if per_sample_cnt[i] else 0.0 for i in range(len(samples))
    ]
    resolved = sum(1 for s in per_sample_scores if s > 0)
    mean = float(np.mean(per_sample_scores)) if per_sample_scores else 0.0
    logger.info(
        "Resolved %d / %d samples (%.2f%%), mean score: %.4f",
        resolved,
        len(samples),
        100.0 * resolved / max(len(samples), 1),
        mean,
    )
    return {"resolved": resolved, "total": len(samples), "mean_score": mean, "per_sample_scores": per_sample_scores}


# =====================================================================
# Runner
# =====================================================================


def run_inference(
    *,
    model_path: str,
    data_path: str,
    prompt_length: int,
    response_length: int,
    temperature: float,
    top_p: float,
    n: int,
    max_samples: int,
    engine: str,
    nnodes: int,
    n_gpus_per_node: int,
    tensor_parallel_size: int,
    gateway_count: int,
    max_concurrent_sessions: int,
    tool_image: str | None,
    run_timeout: int,
) -> dict[str, Any]:
    if not ray.is_initialized():
        ray.init()

    config = _load_config(
        model_path=model_path,
        engine=engine,
        prompt_length=prompt_length,
        response_length=response_length,
        temperature=temperature,
        top_p=top_p,
        n=n,
        nnodes=nnodes,
        n_gpus_per_node=n_gpus_per_node,
        tensor_parallel_size=tensor_parallel_size,
        gateway_count=gateway_count,
        max_concurrent_sessions=max_concurrent_sessions,
        tool_image=tool_image,
        run_timeout=run_timeout,
    )

    samples = load_swe_dataset(data_path, max_samples=max_samples)
    if not samples:
        raise ValueError("No samples to process")

    logger.info("Initializing LLM server manager...")
    llm_server_manager = LLMServerManager.create(config=config)
    llm_client = llm_server_manager.get_client()

    gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
    reward_worker = ray.remote(RewardLoopWorker).remote(config, None)
    framework = build_agent_framework(
        config=config,
        gateway_manager=gateway_manager,
        reward_loop_worker_handles=[reward_worker],
    )

    prompts, uids = _build_prompts(samples)
    captured_scores, _uid_status = _install_tq_capture()

    logger.info("Starting %d sample(s), %d session(s) each...", len(samples), n)
    try:
        try:
            asyncio.run(framework.generate_sequences(prompts))
        except RuntimeError as exc:
            # Framework raises RuntimeError when every rollout fails; downgrade to a
            # warning so we still report a (zero) resolve rate instead of crashing.
            logger.warning("generate_sequences failed: %s", exc)

        if not captured_scores:
            logger.warning(
                "No trajectory scores captured — all rollouts may have failed (see the "
                "generate_sequences summary above), or the TransferQueue monkeypatch did not "
                "reach the writer; resolve rate will be reported as 0."
            )

        return _report(samples, uids, captured_scores)
    finally:
        # Always tear down gateway actors, even if generate/report raised, so a
        # failed run does not leak the Ray actor pool.
        asyncio.run(gateway_manager.shutdown())


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Blackbox claude-code standalone inference")
    parser.add_argument("--model-path", "--model", type=str, default="~/models/Qwen3.5-9B")
    parser.add_argument("--data-path", type=str, default="~/data/swe_agent/swe_bench_verified.parquet")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--prompt-length", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=131072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--gateway-count", type=int, default=1)
    parser.add_argument("--max-concurrent-sessions", type=int, default=8)
    parser.add_argument("--tool-image", type=str, default=_DEFAULT_TOOL_IMAGE)
    parser.add_argument("--run-timeout", type=int, default=7200)
    parser.add_argument("--max-turns", type=int, default=100)
    args = parser.parse_args()

    # Set before ray.init so runner Ray tasks inherit it.
    os.environ["AGENT_MAX_TURNS"] = str(args.max_turns)

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_samples=args.max_samples,
        engine=args.engine,
        nnodes=args.nnodes,
        n_gpus_per_node=args.n_gpus_per_node,
        tensor_parallel_size=args.tensor_parallel_size,
        gateway_count=args.gateway_count,
        max_concurrent_sessions=args.max_concurrent_sessions,
        tool_image=args.tool_image,
        run_timeout=args.run_timeout,
    )


if __name__ == "__main__":
    main()
