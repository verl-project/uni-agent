"""Heavy rollout-only e2e for LLMRouter on a single host split into two Ray nodes.

This test is opt-in because it starts two local Ray raylets and two vLLM
replicas for Qwen3-8B. It validates the rollout path, not PPO training:

1. ``LLMRouter.create`` launches two rollout replicas on a simulated two-node
   Ray cluster.
2. A real ``AsyncLLMServerManager.generate`` call with no prefix reports falls
   back to the least-in-flight replica.
3. A second turn for the same session routes back to the same replica after the
   first generation reports its GPU/HBM prefix.
4. A CPU-tier Mooncake-style report routes a new session to the CPU owner.
5. A GPU-tier report wins over a CPU-tier report for the same prefix.

Run manually, for example:

    LLM_ROUTER_ROLLOUT_DUAL_NODE_E2E=1 \
    LLM_ROUTER_E2E_GPU_IDS=0,1 \
    pytest llm_router/tests/integration/test_rollout_dual_node_e2e.py -v -s
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL = Path("/data1/zqzhai/models/Qwen/Qwen3-8B")
_DEFAULT_DATASET = Path("/data1/zqzhai/dataset/verl_swe_bench_verified")


@pytest.mark.skipif(
    os.environ.get("LLM_ROUTER_ROLLOUT_DUAL_NODE_E2E") != "1",
    reason="set LLM_ROUTER_ROLLOUT_DUAL_NODE_E2E=1 to run the Qwen3-8B rollout e2e",
)
def test_rollout_uses_llm_router_tiered_routing_on_simulated_two_nodes(tmp_path: Path):
    model_path = Path(os.environ.get("LLM_ROUTER_E2E_MODEL", _DEFAULT_MODEL))
    dataset_path = Path(os.environ.get("LLM_ROUTER_E2E_DATASET", _DEFAULT_DATASET))
    if not model_path.exists():
        pytest.skip(f"model path not found: {model_path}")
    if not dataset_path.exists():
        pytest.skip(f"dataset path not found: {dataset_path}")

    output_path = tmp_path / "rollout-dual-node-e2e.json"
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARN")
    env.setdefault("VLLM_USE_V1", "1")
    env["LLM_ROUTER_RECORD_ACQUIRE_HISTORY"] = "1"
    env["LLM_ROUTER_E2E_OUTPUT"] = str(output_path)
    env["LLM_ROUTER_E2E_MODEL"] = str(model_path)
    env["LLM_ROUTER_E2E_DATASET"] = str(dataset_path)
    env["LLM_ROUTER_E2E_SUBPROCESS"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "llm_router.tests.integration.test_rollout_dual_node_e2e"],
        cwd=_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(env.get("LLM_ROUTER_E2E_TIMEOUT", "1800")),
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-240:])
        pytest.fail(f"rollout dual-node e2e failed with code {result.returncode}\n{tail}")

    summary = json.loads(output_path.read_text())
    assert summary["server_ids"] == ["replica-0", "replica-1"]
    assert summary["least_loaded_route"] == "replica-0"
    assert summary["gpu_followup_route"] == summary["least_loaded_route"]
    assert summary["cpu_route"] == "replica-1"
    assert summary["gpu_priority_route"] == "replica-0"
    assert summary["num_ray_nodes"] == 2


def _gpu_ids_for_two_nodes() -> list[str]:
    configured = os.environ.get("LLM_ROUTER_E2E_GPU_IDS")
    if configured:
        gpu_ids = [item.strip() for item in configured.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            gpu_ids = ["0", "1"]
    if len(gpu_ids) < 2:
        raise RuntimeError(
            "LLM_ROUTER_E2E_GPU_IDS or CUDA_VISIBLE_DEVICES must expose at least two GPUs"
        )
    return gpu_ids[:2]


def _dataset_file(path: Path) -> Path:
    if path.is_file():
        return path
    parquet = path / "swe_bench_verified.parquet"
    if parquet.exists():
        return parquet
    candidates = sorted(path.glob("*.parquet"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"no parquet dataset found under {path}")


def _load_swe_prompt(dataset_path: Path) -> list[dict[str, str]]:
    import pandas as pd

    row = pd.read_parquet(_dataset_file(dataset_path)).iloc[0]
    return [dict(message) for message in row["prompt"]]


def _build_config(model_path: Path) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = str(_ROOT / "verl" / "verl" / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer")

    cfg.trainer.n_gpus_per_node = 1
    cfg.trainer.nnodes = 2
    cfg.trainer.logger = ["console"]
    cfg.actor_rollout_ref.model.path = str(model_path)
    cfg.actor_rollout_ref.model.tokenizer_path = str(model_path)
    cfg.actor_rollout_ref.model.trust_remote_code = True
    cfg.actor_rollout_ref.rollout.name = "vllm"
    cfg.actor_rollout_ref.rollout.mode = "async"
    cfg.actor_rollout_ref.rollout.n_gpus_per_node = 1
    cfg.actor_rollout_ref.rollout.nnodes = 2
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    cfg.actor_rollout_ref.rollout.data_parallel_size = 1
    cfg.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
    cfg.actor_rollout_ref.rollout.prompt_length = int(os.environ.get("LLM_ROUTER_E2E_PROMPT_LEN", "1024"))
    cfg.actor_rollout_ref.rollout.response_length = int(os.environ.get("LLM_ROUTER_E2E_RESPONSE_LEN", "8"))
    cfg.actor_rollout_ref.rollout.max_model_len = int(os.environ.get("LLM_ROUTER_E2E_MAX_MODEL_LEN", "1536"))
    cfg.actor_rollout_ref.rollout.max_num_batched_tokens = int(
        os.environ.get("LLM_ROUTER_E2E_MAX_BATCHED_TOKENS", "1536")
    )
    cfg.actor_rollout_ref.rollout.max_num_seqs = 2
    cfg.actor_rollout_ref.rollout.gpu_memory_utilization = float(
        os.environ.get("LLM_ROUTER_E2E_GPU_MEMORY_UTILIZATION", "0.72")
    )
    cfg.actor_rollout_ref.rollout.enforce_eager = True
    cfg.actor_rollout_ref.rollout.enable_chunked_prefill = False
    cfg.actor_rollout_ref.rollout.agent.num_workers = 1
    cfg.actor_rollout_ref.rollout.prometheus.enable = False
    cfg.actor_rollout_ref.rollout.disable_log_stats = True
    cfg.actor_rollout_ref.rollout.skip_tokenizer_init = False
    cfg.actor_rollout_ref.rollout.load_format = "auto"
    cfg.actor_rollout_ref.rollout.val_kwargs.temperature = 0
    cfg.actor_rollout_ref.rollout.val_kwargs.top_p = 1.0
    cfg.actor_rollout_ref.rollout.val_kwargs.top_k = -1
    with open_dict(cfg.actor_rollout_ref.rollout.agent):
        cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class = "llm_router.LLMRouter"
    with open_dict(cfg.actor_rollout_ref.rollout):
        cfg.actor_rollout_ref.rollout.context_aware_scheduling = {
            "enable": True,
            "prewarm_enable": False,
            "prefix_probe_stride": 64,
            "routing_cache_size": 10000,
        }
    with open_dict(cfg.actor_rollout_ref):
        cfg.actor_rollout_ref.llm_router = {
            "policy": "rule_based",
            "gpu_hit_threshold": 64,
            "cpu_hit_threshold": 64,
            "load_threshold": 1024,
            "routing_cache_size": 10000,
        }
    cfg.reward.reward_model.enable = False
    return cfg


def _start_two_node_cluster():
    import ray
    from ray.cluster_utils import Cluster

    gpu_ids = _gpu_ids_for_two_nodes()
    cpus_per_node = int(os.environ.get("LLM_ROUTER_E2E_CPUS_PER_NODE", "8"))
    object_store_memory = int(os.environ.get("LLM_ROUTER_E2E_OBJECT_STORE_BYTES", str(2 * 1024 * 1024 * 1024)))
    common_env = {
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "1"),
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "WARN"),
        "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
        "LLM_ROUTER_RECORD_ACQUIRE_HISTORY": os.environ.get(
            "LLM_ROUTER_RECORD_ACQUIRE_HISTORY",
            "1",
        ),
    }

    cluster = Cluster()
    for index, gpu_id in enumerate(gpu_ids):
        node_env = dict(common_env)
        node_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cluster.add_node(
            num_cpus=cpus_per_node,
            num_gpus=1,
            object_store_memory=object_store_memory,
            resources={f"llm_router_e2e_node_{index}": 1},
            env_vars=node_env,
        )
    ray.init(address=cluster.address, ignore_reinit_error=True, runtime_env={"env_vars": common_env})
    return cluster


async def _run_rollout_routes(model_path: Path, dataset_path: Path) -> dict[str, Any]:
    import ray
    from transformers import AutoTokenizer

    from llm_router import LLMRouter
    from llm_router.connector.prefix_hash import iter_prefix_signatures
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

    cfg = _build_config(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    prompt_ids = tokenizer.apply_chat_template(
        _load_swe_prompt(dataset_path),
        add_generation_prompt=True,
        tokenize=True,
    )
    prompt_budget = cfg.actor_rollout_ref.rollout.prompt_length
    prompt_ids = list(prompt_ids[:prompt_budget])
    if len(prompt_ids) < 128:
        raise RuntimeError(f"SWE prompt unexpectedly short after tokenization: {len(prompt_ids)}")

    manager = await LLMRouter.create(cfg)
    server_manager = AsyncLLMServerManager(
        cfg,
        list(zip(manager.server_ids, manager.server_handles, strict=True)),
        manager.load_balancer,
    )
    sampling_params = {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1}
    stride = int(cfg.actor_rollout_ref.rollout.context_aware_scheduling.prefix_probe_stride)

    out1 = await server_manager.generate(
        "least-session",
        prompt_ids=prompt_ids,
        sampling_params=dict(sampling_params),
        weight_version="least",
    )
    followup_prompt = prompt_ids + list(out1.token_ids)
    if tokenizer.eos_token_id is not None:
        followup_prompt.append(int(tokenizer.eos_token_id))
    followup_prompt = followup_prompt[: cfg.actor_rollout_ref.rollout.max_model_len - 1]
    await server_manager.generate(
        "least-session",
        prompt_ids=followup_prompt,
        sampling_params=dict(sampling_params),
        weight_version="least",
    )

    cpu_sigs = iter_prefix_signatures(prompt_ids, "cpu", stride=stride)
    ray.get(manager.load_balancer.report_prefixes.remote("replica-1", cpu_sigs, tier="cpu"))
    await server_manager.generate(
        "cpu-session",
        prompt_ids=prompt_ids,
        sampling_params=dict(sampling_params),
        weight_version="cpu",
    )

    priority_sigs = iter_prefix_signatures(prompt_ids, "priority", stride=stride)
    ray.get(manager.load_balancer.report_prefixes.remote("replica-0", priority_sigs, tier="gpu"))
    ray.get(manager.load_balancer.report_prefixes.remote("replica-1", priority_sigs, tier="cpu"))
    await server_manager.generate(
        "priority-session",
        prompt_ids=prompt_ids,
        sampling_params=dict(sampling_params),
        weight_version="priority",
    )

    history = ray.get(manager.load_balancer.debug_acquire_history.remote())
    routes = {item["session_id"]: item["server_id"] for item in history}
    ray_nodes = [node for node in ray.nodes() if node.get("Alive")]
    return {
        "server_ids": list(manager.server_ids),
        "server_addresses": list(manager.server_addresses),
        "history": history,
        "least_loaded_route": history[0]["server_id"],
        "gpu_followup_route": history[1]["server_id"],
        "cpu_route": routes["cpu-session"],
        "gpu_priority_route": routes["priority-session"],
        "num_ray_nodes": len(ray_nodes),
        "ray_nodes": [
            {
                "node_id": node.get("NodeID"),
                "resources": node.get("Resources", {}),
            }
            for node in ray_nodes
        ],
        "prompt_len": len(prompt_ids),
    }


def _run_subprocess_entrypoint() -> None:
    import ray

    output_path = Path(os.environ["LLM_ROUTER_E2E_OUTPUT"])
    cluster = _start_two_node_cluster()
    try:
        summary = asyncio.run(
            _run_rollout_routes(
                Path(os.environ["LLM_ROUTER_E2E_MODEL"]),
                Path(os.environ["LLM_ROUTER_E2E_DATASET"]),
            )
        )
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    if os.environ.get("LLM_ROUTER_E2E_SUBPROCESS") != "1":
        raise SystemExit(
            textwrap.dedent(
                """\
                Run through pytest with:
                  LLM_ROUTER_ROLLOUT_DUAL_NODE_E2E=1
                  LLM_ROUTER_E2E_GPU_IDS=0,1
                """
            )
        )
    _run_subprocess_entrypoint()
