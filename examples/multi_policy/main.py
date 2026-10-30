from __future__ import annotations

import importlib
import logging
import os
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from examples.multi_policy.trainer.multi_policy_trainer import MultiPolicyTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _apply_example_patch(config) -> None:
    """Apply an example-owned runtime patch before trainer construction.

    verl is treated as a shared/public package, so example-specific fixes live
    in the example directory and are loaded here through the optional
    ``config.example_patch_fqn`` key (e.g.
    ``examples.multi_policy.verl_patch``, which makes each policy's Ray
    placement-group name prefix unique).
    """
    patch_fqn = OmegaConf.select(config, "example_patch_fqn", default=None)
    if not patch_fqn:
        return
    module = importlib.import_module(str(patch_fqn))
    apply_patch = getattr(module, "apply_patch", None)
    if callable(apply_patch):
        apply_patch()
        logger.info("applied example patch from %s", patch_fqn)


def run_multi_policy(config, task_runner_class) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env(config)
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        # runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        # Hydra composes configs as OmegaConf structs, which reject adding keys
        # (e.g. env_vars) that the YAML does not declare. Convert to a plain
        # mutable container before injecting TRANSFER_QUEUE_ENABLE / PYTHONPATH.
        runtime_env_kwargs = OmegaConf.to_container(ray_init_kwargs.get("runtime_env", {}), resolve=True) or {}

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        # Force Ray workers to import THIS checkout's example patch
        # (worker_process_setup_hook). Workers inherit a cluster/node
        # PYTHONPATH that may point at another uni-marl checkout whose
        # verl_patch lacks apply_worker_patch, which would break the hook.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        runtime_env_vars = runtime_env_kwargs.setdefault("env_vars", {})
        existing_pythonpath = runtime_env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        runtime_env_vars["PYTHONPATH"] = repo_root + (":" + existing_pythonpath if existing_pythonpath else "")

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        logger.info("ray init kwargs: %s", ray_init_kwargs)
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = (getattr(config, "ray_kwargs", {}) or {}).get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote
class MultiPolicyTaskRunner:
    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        """Initialize the agent loop manager to generate multi-agent rollouts."""
        if self.config is None or self.trainer is None:
            raise RuntimeError("trainer and config must be initialized before agent_loop_manager")

        from examples.multi_policy.framework.multi_agent_entry import (
            MultiPolicyRolloutAdapter,
        )

        self.agent_loop_manager = MultiPolicyRolloutAdapter.create(
            config=self.config,
            policy_clients=self.trainer.get_policy_clients(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
            policy_configs=self.trainer.get_policy_configs(),
        )
        return self.agent_loop_manager

    def run(self, config: DictConfig):
        import transfer_queue as tq

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        try:
            _apply_example_patch(config)
            self.trainer = MultiPolicyTrainer(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            # Ray owns the trainer's policy workers, placement groups, Gateway
            # actors, and framework worker through this task runner actor. They
            # are reclaimed with the Ray job, matching verl v1 lifecycle.
            tq.close()


@hydra.main(config_path="config", config_name="multi_policy", version_base=None)
def main(config):
    auto_set_device(config)
    run_multi_policy(config, task_runner_class=MultiPolicyTaskRunner)


if __name__ == "__main__":
    main()
