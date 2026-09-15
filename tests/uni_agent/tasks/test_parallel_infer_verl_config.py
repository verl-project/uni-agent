from __future__ import annotations

from argparse import Namespace

import pytest

from examples.agent_aware_router.run_infer import init_config as init_router_config
from examples.inference.parallel_infer_verl import init_config


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    "entrypoint,simulated_runner",
    [(init_config, None), (init_router_config, None), (init_router_config, "example.simulated_runner")],
)
def test_inference_sampling_uses_run_options_and_preserves_length_configuration(entrypoint, simulated_runner):
    args = Namespace(
        temperature=0.4,
        top_p=0.85,
        top_k=20,
        n=2,
        nnodes=1,
        n_gpus_per_node=1,
        model_path="/tmp/test-model",
        engine="vllm",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        enable_rollout_routing_replay=False,
        tool_parser="qwen3_coder",
        gateway_count=1,
        concurrency=4,
        task_config="/not-loaded-until-task-preparation.yaml",
        log_dir="/tmp/test-inference",
        num_workers=1,
        max_model_len=32768,
        max_num_seqs=16,
        enable_mooncake=False,
        kv_events=False,
        router_config_path="uni_agent/agent_aware_router/configs/agent_aware_router.yaml",
        simulated_runner_fqn=simulated_runner,
        load_threshold=0.5,
        prompt_length=2048,
        response_length=8192,
    )

    config = entrypoint(
        args,
        task_configs=[{"agent": {"model": {"sampling_params_override": {"temperature": 1.0}}}}],
        served_model_name="policy",
    )

    rollout = config.actor_rollout_ref.rollout
    for sampling in (rollout, rollout.val_kwargs):
        assert sampling.temperature == 0.4
        assert sampling.top_p == 0.85
        assert sampling.top_k == 20
    assert rollout.val_kwargs.do_sample is True
    expected_prompt_length = args.prompt_length if entrypoint is init_router_config else 4096
    expected_response_length = args.response_length if entrypoint is init_router_config else 65536
    assert rollout.prompt_length == config.data.max_prompt_length == expected_prompt_length
    assert rollout.response_length == config.data.max_response_length == expected_response_length
    task_runner = rollout.custom.agent_framework.agent_runners.task
    if simulated_runner:
        assert task_runner.runner_fqn == simulated_runner
        assert not task_runner.runner_kwargs
    else:
        assert task_runner.runner_kwargs.task_config_path == args.task_config
