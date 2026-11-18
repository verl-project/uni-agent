"""Minimal separate-async single-policy v1 PPO trainer for multi-agent orchestration.

Mirrors SinglePPOTrainer (sync): the outer MultiPolicyTrainer owns the shared
dataloader and the shared ReplayBufferAsync. Inheriting PPOTrainerSeparateAsync
gives us:

- native trainer/FSDP setup plus standalone vLLM runtime components;
- get_llm_client() -> FullyAsyncLLMServerClient (partial-rollout resume);
- on_step_end(): standalone_checkpoint_manager.update_weights (abort/resume built-in);
- on_step_begin(): hybrid switch-to-trainer semantics (with enable_switch=False).
"""

from __future__ import annotations

from copy import deepcopy

from verl.checkpoint_engine import CheckpointEngineManager
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import (
    HybridEngineMode,
    PPOTrainerSeparateAsync,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.rollout.llm_server import LLMServerManager


class SingleAsyncPPOTrainer(PPOTrainerSeparateAsync):
    """Separate-async v1 PPO trainer driven by the outer multi-agent trainer."""

    def init_training_runtime(self) -> None:
        """Initialize the primary trainer-owned runtime, excluding standalone rollout.

        Calling the v1 base implementation directly establishes the policy's
        trainer placement groups, distributed model workers, and hybrid
        rollout replicas without entering ``PPOTrainerSeparateAsync._setup()``,
        which would immediately allocate standalone rollout GPUs.
        """
        PPOTrainer._setup(self)

    def init_standalone_rollout_runtime(self) -> None:
        """Initialize the native v1 standalone rollout runtime."""
        hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)

        # Standalone replicas have dedicated GPUs; keep their memory budget
        # separate from the hybrid replicas that share GPUs with training.
        standalone_config = deepcopy(self.config)
        standalone_rollout_config = standalone_config.actor_rollout_ref.rollout
        standalone_memory = standalone_rollout_config.get("standalone_gpu_memory_utilization")
        if standalone_memory is not None:
            standalone_rollout_config.gpu_memory_utilization = standalone_memory

        self.standalone_server_manager = LLMServerManager.create(
            config=standalone_config,
            start_rank=hybrid_num_replicas,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

        if self._enable_hybrid_replicas:
            self.current_mode = HybridEngineMode.ROLLOUT
            self.add_replicas_to_balancer()
        else:
            # Rollout is served exclusively by the standalone pool; there are no
            # colocated replicas on the training GPUs to switch into ROLLOUT mode.
            self.current_mode = HybridEngineMode.TRAINER

    def _build_replay_buffer(self):
        """The outer trainer owns the shared async replay buffer."""
        return None

    def _init_dataloader(self):
        """The outer trainer owns the shared prompt stream."""
        return None

    def fit(self, agent_loop_manager):
        raise RuntimeError(
            "SingleAsyncPPOTrainer cannot run its own fit(): it has no dataloader or "
            "replay buffer. Use MultiPolicyTrainer for multi-agent orchestration."
        )

    def step(self, metrics, timing_raw):
        raise RuntimeError(
            "SingleAsyncPPOTrainer cannot run its own step(): it has no dataloader or "
            "replay buffer. Use MultiPolicyTrainer for multi-agent orchestration."
        )
