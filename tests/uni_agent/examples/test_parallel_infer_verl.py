import argparse

from examples.inference.parallel_infer_verl import init_config


def test_init_config_disables_memory_lifecycle_for_standalone_inference():
    args = argparse.Namespace(
        n=1,
        nnodes=1,
        n_gpus_per_node=1,
        model_path="/tmp/model",
        engine="vllm",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_rollout_routing_replay=False,
        tool_parser="hermes",
        gateway_count=1,
        concurrency=1,
        task_config="task.yaml",
        log_dir=None,
    )

    config = init_config(args, task_configs=[{"agent": {"model": {}}}], served_model_name="model")

    rollout = config.actor_rollout_ref.rollout
    assert rollout.free_cache_engine is False
    assert rollout.enable_sleep_mode is False
