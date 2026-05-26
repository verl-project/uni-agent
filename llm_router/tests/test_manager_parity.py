"""GPU parity: LLMRouter behaves like stock AgentLoopManager.

This is a compatibility test, not a throughput benchmark. It runs the same
minimal rollout twice:

1. stock ``verl.experimental.agent_loop.AgentLoopManager``
2. ``llm_router.LLMRouter``

Each run happens in a fresh Python subprocess and Ray lifecycle. vLLM/Ray keep
GPU actors alive long enough that comparing both managers in one process can
make the candidate run see ``0`` available GPUs on single-GPU test machines.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

requires_gpu = pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES"),
    reason="GPU required for end-to-end rollout",
)
requires_model = pytest.mark.skipif(
    not os.environ.get("LLM_ROUTER_PARITY_MODEL"),
    reason="set LLM_ROUTER_PARITY_MODEL to a local small HF model path",
)

_ROOT = Path(__file__).resolve().parents[2]


@requires_gpu
@requires_model
def test_llm_router_drop_in_parity(tmp_path: Path):
    ref_path = tmp_path / "stock.json"
    cand_path = tmp_path / "llm_router.json"

    _run_isolated_manager("stock", ref_path)
    _run_isolated_manager("llm_router", cand_path)

    ref = json.loads(ref_path.read_text())
    cand = json.loads(cand_path.read_text())

    assert ref["batch_size"] == cand["batch_size"]
    assert ref["responses_shape"] == cand["responses_shape"]
    assert ref["response_mask_shape"] == cand["response_mask_shape"]
    assert ref["attention_mask_shape"] == cand["attention_mask_shape"]
    assert ref["input_ids_shape"] == cand["input_ids_shape"]

    # Greedy validation sampling should make the first tokens exactly stable.
    assert ref["first_response_tokens"] == cand["first_response_tokens"]


def _run_isolated_manager(kind: str, output_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARN")
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env["LLM_ROUTER_PARITY_KIND"] = kind
    env["LLM_ROUTER_PARITY_OUTPUT"] = str(output_path)

    cmd = [sys.executable, "-m", "llm_router.tests.test_manager_parity"]
    result = subprocess.run(
        cmd,
        cwd=_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(env.get("LLM_ROUTER_PARITY_TIMEOUT", "900")),
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-160:])
        pytest.fail(f"{kind} isolated manager run failed with code {result.returncode}\n{tail}")


def _minimal_config() -> Any:
    """Return a minimal-but-valid verl config for GPU parity testing."""
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = str(_ROOT / "verl" / "verl" / "trainer" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer")

    model_path = os.environ["LLM_ROUTER_PARITY_MODEL"]
    cfg.trainer.n_gpus_per_node = 1
    cfg.trainer.nnodes = 1
    cfg.trainer.logger = ["console"]
    cfg.actor_rollout_ref.model.path = model_path
    cfg.actor_rollout_ref.model.tokenizer_path = model_path
    cfg.actor_rollout_ref.model.trust_remote_code = True
    cfg.actor_rollout_ref.rollout.name = "vllm"
    cfg.actor_rollout_ref.rollout.mode = "async"
    cfg.actor_rollout_ref.rollout.n_gpus_per_node = 1
    cfg.actor_rollout_ref.rollout.nnodes = 1
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    cfg.actor_rollout_ref.rollout.data_parallel_size = 1
    cfg.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
    cfg.actor_rollout_ref.rollout.prompt_length = 32
    cfg.actor_rollout_ref.rollout.response_length = 8
    cfg.actor_rollout_ref.rollout.max_model_len = 64
    cfg.actor_rollout_ref.rollout.max_num_batched_tokens = 128
    cfg.actor_rollout_ref.rollout.max_num_seqs = 2
    cfg.actor_rollout_ref.rollout.gpu_memory_utilization = 0.25
    cfg.actor_rollout_ref.rollout.enforce_eager = True
    cfg.actor_rollout_ref.rollout.enable_chunked_prefill = False
    cfg.actor_rollout_ref.rollout.agent.num_workers = 1
    with open_dict(cfg.actor_rollout_ref.rollout.agent):
        cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class = "llm_router.LLMRouter"
    cfg.actor_rollout_ref.rollout.prometheus.enable = False
    cfg.actor_rollout_ref.rollout.disable_log_stats = True
    cfg.actor_rollout_ref.rollout.skip_tokenizer_init = False
    cfg.actor_rollout_ref.rollout.load_format = "auto"
    cfg.actor_rollout_ref.rollout.val_kwargs.temperature = 0
    cfg.actor_rollout_ref.rollout.val_kwargs.top_p = 1.0
    cfg.actor_rollout_ref.rollout.val_kwargs.top_k = -1
    with open_dict(cfg.actor_rollout_ref):
        cfg.actor_rollout_ref.llm_router = {"policy": "legacy_sticky"}
    cfg.reward.reward_model.enable = False
    return cfg


def _build_two_prompts():
    import numpy as np
    import torch

    from verl.protocol import DataProto

    prompts = torch.tensor(
        [
            [151644, 872, 198, 9707, 151645],
            [151644, 872, 198, 3838, 151645],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(prompts)
    position_ids = torch.arange(prompts.shape[1], dtype=torch.long).unsqueeze(0).repeat(prompts.shape[0], 1)
    raw_prompt = np.array(
        [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ],
        dtype=object,
    )
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors={
            "raw_prompt": raw_prompt,
            "agent_name": np.array(["single_turn_agent", "single_turn_agent"], dtype=object),
            "index": np.array([0, 1], dtype=object),
        },
        meta_info={"validate": True, "global_steps": 0},
    )


def _summarize_output(output) -> dict[str, Any]:
    responses = output.batch["responses"]
    return {
        "batch_size": list(output.batch.batch_size),
        "responses_shape": list(responses.shape),
        "response_mask_shape": list(output.batch["response_mask"].shape),
        "attention_mask_shape": list(output.batch["attention_mask"].shape),
        "input_ids_shape": list(output.batch["input_ids"].shape),
        "first_response_tokens": responses[:, :8].tolist(),
    }


def _run_subprocess_entrypoint() -> None:
    import ray

    kind = os.environ["LLM_ROUTER_PARITY_KIND"]
    output_path = Path(os.environ["LLM_ROUTER_PARITY_OUTPUT"])
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
            "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "WARN"),
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "1"),
            "NCCL_P2P_DISABLE": os.environ.get("NCCL_P2P_DISABLE", "1"),
        }
    }
    ray.init(num_cpus=4, num_gpus=1, runtime_env=runtime_env, ignore_reinit_error=True)
    try:
        cfg = _minimal_config()
        prompts = _build_two_prompts()
        if kind == "stock":
            from verl.experimental.agent_loop import AgentLoopManager

            manager = AgentLoopManager.create(cfg)
        elif kind == "llm_router":
            from verl.utils.import_utils import load_class_from_fqn

            cls = load_class_from_fqn(
                cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class,
                "AgentLoopManager",
            )
            manager = cls.create(cfg)
        else:
            raise ValueError(f"unknown parity kind: {kind}")

        output = manager.generate_sequences(prompts)
        output_path.write_text(json.dumps(_summarize_output(output), sort_keys=True))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    if "LLM_ROUTER_PARITY_KIND" not in os.environ:
        raise SystemExit(
            textwrap.dedent(
                """\
                Run this module via pytest, or set:
                  LLM_ROUTER_PARITY_KIND=stock|llm_router
                  LLM_ROUTER_PARITY_OUTPUT=/path/to/output.json
                """
            )
        )
    _run_subprocess_entrypoint()
