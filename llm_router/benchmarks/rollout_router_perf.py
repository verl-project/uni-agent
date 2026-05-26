"""Compare rollout latency with legacy sticky routing vs LLMRouter policy.

This benchmark is intentionally opt-in and GPU-heavy. It starts two local Ray
nodes, launches two vLLM rollout replicas, then reuses those replicas while
switching only the load-balancer policy.

Example:

    CUDA_VISIBLE_DEVICES=2,3 \
    LLM_ROUTER_E2E_GPU_IDS=2,3 \
    python -m llm_router.benchmarks.rollout_router_perf
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import ray

from llm_router import LLMRouter
from llm_router.load_balancer import LoadBalancer
from llm_router.tests.integration.test_rollout_dual_node_e2e import (
    _DEFAULT_DATASET,
    _DEFAULT_MODEL,
    _build_config,
    _load_swe_prompt,
    _start_two_node_cluster,
)
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _summary(latencies: list[float], wall_time: float) -> dict[str, float]:
    return {
        "count": len(latencies),
        "wall_s": wall_time,
        "mean_s": statistics.fmean(latencies) if latencies else 0.0,
        "median_s": statistics.median(latencies) if latencies else 0.0,
        "p90_s": _percentile(latencies, 0.90),
        "min_s": min(latencies) if latencies else 0.0,
        "max_s": max(latencies) if latencies else 0.0,
    }


async def _timed_generate(
    server_manager: AsyncLLMServerManager,
    request_id: str,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    *,
    weight_version: str,
) -> float:
    start = time.perf_counter()
    await server_manager.generate(
        request_id,
        prompt_ids=prompt_ids,
        sampling_params=dict(sampling_params),
        weight_version=weight_version,
    )
    return time.perf_counter() - start


async def _clear_rollout_caches(manager: LLMRouter) -> None:
    await asyncio.gather(*[replica.clear_kv_cache() for replica in manager.rollout_replicas])


async def _warmup_engines(
    manager: LLMRouter,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
) -> None:
    await asyncio.gather(
        *[
            server.generate.remote(
                request_id=f"bench-engine-warmup-{idx}",
                prompt_ids=prompt_ids,
                sampling_params=dict(sampling_params),
                weight_version="bench-warmup",
            )
            for idx, server in enumerate(manager.server_handles)
        ]
    )
    await _clear_rollout_caches(manager)


async def _run_multi_turn_same_session(
    *,
    server_manager: AsyncLLMServerManager,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    label: str,
    turns: int,
) -> dict[str, Any]:
    latencies = []
    current_prompt = list(prompt_ids)
    wall_start = time.perf_counter()
    for turn in range(turns):
        start = time.perf_counter()
        output = await server_manager.generate(
            f"{label}-session",
            prompt_ids=current_prompt,
            sampling_params=dict(sampling_params),
            weight_version=label,
        )
        latencies.append(time.perf_counter() - start)
        current_prompt.extend(list(output.token_ids))
        current_prompt = current_prompt[: int(os.environ.get("LLM_ROUTER_BENCH_MAX_PROMPT_IDS", "1400"))]
    return _summary(latencies, time.perf_counter() - wall_start)


async def _run_shared_prefix_fanout(
    *,
    server_manager: AsyncLLMServerManager,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    label: str,
    concurrency: int,
) -> dict[str, Any]:
    await server_manager.generate(
        f"{label}-warm",
        prompt_ids=prompt_ids,
        sampling_params=dict(sampling_params),
        weight_version=label,
    )
    await asyncio.sleep(0.1)

    wall_start = time.perf_counter()
    latencies = await asyncio.gather(
        *[
            _timed_generate(
                server_manager,
                f"{label}-fanout-{idx}",
                prompt_ids,
                sampling_params,
                weight_version=label,
            )
            for idx in range(concurrency)
        ]
    )
    return _summary(latencies, time.perf_counter() - wall_start)


async def _make_server_manager(
    cfg: Any,
    manager: LLMRouter,
    *,
    policy_name: str,
    context_aware: bool,
    label: str,
) -> tuple[AsyncLLMServerManager, ray.actor.ActorHandle]:
    from omegaconf import open_dict

    local_cfg = cfg.copy()
    with open_dict(local_cfg.actor_rollout_ref.rollout):
        local_cfg.actor_rollout_ref.rollout.context_aware_scheduling.enable = context_aware
    lb = LoadBalancer.options(name=f"llm_router_bench_{label}_{os.getpid()}").remote(
        server_ids=manager.server_ids,
        policy_name=policy_name,
        routing_cache_size=10000,
        gpu_hit_threshold=64,
        cpu_hit_threshold=64,
        load_threshold=int(os.environ.get("LLM_ROUTER_BENCH_LOAD_THRESHOLD", "1024")),
        record_acquire_history=True,
    )
    return (
        AsyncLLMServerManager(
            local_cfg,
            list(zip(manager.server_ids, manager.server_handles, strict=True)),
            lb,
        ),
        lb,
    )


async def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    cfg = _build_config(args.model)
    cfg.actor_rollout_ref.rollout.prompt_length = args.prompt_len
    cfg.actor_rollout_ref.rollout.response_length = args.response_len
    cfg.actor_rollout_ref.rollout.max_model_len = args.max_model_len
    cfg.actor_rollout_ref.rollout.max_num_batched_tokens = args.max_model_len

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    prompt_ids = tokenizer.apply_chat_template(
        _load_swe_prompt(args.dataset),
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_ids = list(prompt_ids[: args.prompt_len])
    sampling_params = {
        "max_tokens": args.decode_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }

    manager = await LLMRouter.create(cfg)
    try:
        legacy_manager, legacy_lb = await _make_server_manager(
            cfg,
            manager,
            policy_name="legacy_sticky",
            context_aware=False,
            label="legacy",
        )
        router_manager, router_lb = await _make_server_manager(
            cfg,
            manager,
            policy_name="rule_based",
            context_aware=True,
            label="router",
        )

        results: dict[str, Any] = {
            "model": str(args.model),
            "dataset": str(args.dataset),
            "prompt_len": len(prompt_ids),
            "decode_tokens": args.decode_tokens,
            "server_ids": list(manager.server_ids),
            "server_addresses": list(manager.server_addresses),
            "workloads": {},
        }

        await _warmup_engines(manager, prompt_ids, sampling_params)

        for label, server_manager, lb in [
            ("legacy_off", legacy_manager, legacy_lb),
            ("llm_router_on", router_manager, router_lb),
        ]:
            await _clear_rollout_caches(manager)
            same_session = await _run_multi_turn_same_session(
                server_manager=server_manager,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                label=f"{label}-same",
                turns=args.turns,
            )
            history_after_same = ray.get(lb.debug_acquire_history.remote())

            await _clear_rollout_caches(manager)
            fanout = await _run_shared_prefix_fanout(
                server_manager=server_manager,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                label=f"{label}-fanout",
                concurrency=args.concurrency,
            )
            history_after_fanout = ray.get(lb.debug_acquire_history.remote())
            results["workloads"][label] = {
                "same_session": same_session,
                "shared_prefix_fanout": fanout,
                "routes": history_after_fanout,
                "same_session_routes": history_after_same,
            }

        off = results["workloads"]["legacy_off"]["shared_prefix_fanout"]["wall_s"]
        on = results["workloads"]["llm_router_on"]["shared_prefix_fanout"]["wall_s"]
        results["shared_prefix_fanout_speedup"] = off / on if on else None
        return results
    finally:
        await _clear_rollout_caches(manager)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--prompt-len", type=int, default=int(os.environ.get("LLM_ROUTER_BENCH_PROMPT_LEN", "1024")))
    parser.add_argument("--response-len", type=int, default=int(os.environ.get("LLM_ROUTER_BENCH_RESPONSE_LEN", "16")))
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("LLM_ROUTER_BENCH_MAX_MODEL_LEN", "1536")),
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=int(os.environ.get("LLM_ROUTER_BENCH_DECODE_TOKENS", "1")),
    )
    parser.add_argument("--turns", type=int, default=int(os.environ.get("LLM_ROUTER_BENCH_TURNS", "4")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("LLM_ROUTER_BENCH_CONCURRENCY", "4")))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cluster = _start_two_node_cluster()
    try:
        results = asyncio.run(_run_benchmark(args))
        text = json.dumps(results, indent=2, sort_keys=True)
        print(text)
        if args.output:
            args.output.write_text(text)
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
