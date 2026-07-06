"""Shared entry helpers for OpenClaw fully-async train_entry modules."""

from __future__ import annotations

from time import time


def patch_fully_async(rollouter_cls, trainer_cls) -> None:
    """Patch verl fully-async globals to custom rollouter/trainer classes."""
    from verl.experimental.fully_async_policy import fully_async_main

    fully_async_main.FullyAsyncRollouter = rollouter_cls
    fully_async_main.FullyAsyncTrainer = trainer_cls


def run(config) -> None:
    """Run verl fully-async training with the provided Hydra config."""
    from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    start_time = time()
    auto_set_device(config)
    # Align rollout pool config with actor_rollout_ref rollouter config.
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=FullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")
