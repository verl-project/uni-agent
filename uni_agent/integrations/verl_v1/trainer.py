"""Native V1 training modes with uni-agent-owned validation and clients."""

from __future__ import annotations

from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, PPOTrainerSeparateAsync, PPOTrainerSync

from .client import CancellableFullyAsyncLLMServerClient, CancellableLLMServerClient
from .validation import ValidationMixin


def _client_kwargs(config):
    return {
        "cancellation_timeout": float(
            OmegaConf.select(
                config, "actor_rollout_ref.rollout.custom.agent_framework.cancellation_timeout_seconds", default=5.0
            )
        )
    }


class UniAgentSyncTrainer(ValidationMixin, PPOTrainerSync):
    """Reuse native synchronous training with owned validation settlement."""

    def get_llm_client(self):
        return self.llm_server_manager.get_client(client_cls=CancellableLLMServerClient, **_client_kwargs(self.config))


class UniAgentColocateAsyncTrainer(ValidationMixin, PPOTrainerColocateAsync):
    """Reuse colocated training, including the native partial-rollout client."""

    def get_llm_client(self):
        return self.llm_server_manager.get_client(
            client_cls=CancellableFullyAsyncLLMServerClient, **_client_kwargs(self.config)
        )


class UniAgentSeparateAsyncTrainer(ValidationMixin, PPOTrainerSeparateAsync):
    """Reuse separate-async training, including native weight synchronization."""

    def get_llm_client(self):
        return self.standalone_server_manager.get_client(
            client_cls=CancellableFullyAsyncLLMServerClient, **_client_kwargs(self.config)
        )


def get_trainer_class(mode):
    """Select an integration locally; never overwrite verl's trainer registry."""
    classes = {
        "sync": UniAgentSyncTrainer,
        "colocate_async": UniAgentColocateAsyncTrainer,
        "separate_async": UniAgentSeparateAsyncTrainer,
    }
    if mode not in classes:
        raise ValueError(f"Unsupported uni-agent V1 trainer mode: {mode!r}")
    return classes[mode]
