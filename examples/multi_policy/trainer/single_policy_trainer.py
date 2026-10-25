"""Minimal single-policy v1 PPO trainer for multi-agent orchestration.

``MultiPolicyTrainer`` owns one shared dataloader + one shared replay
buffer at the outer level, so per-policy trainers do not need to build their
own. This subclass keeps every verl sync-trainer capability (resource-pool /
worker / vLLM init, tokenizer, phase methods, checkpoint manager) and only
skips the two outer-owned pieces:

- ``_build_replay_buffer``: the outer trainer builds the shared ReplayBuffer.
- ``_init_dataloader``: the outer trainer loads the dataset once, not once per
  policy (avoids N full dataset loads for N policies).

The tokenizer is intentionally kept: ``_balance_batch`` (called by the outer
update path) reads ``self.tokenizer.eos_token_id``, and the outer dataloader
also needs the tokenizer/processor from the first policy trainer.

Mirror verl trainer_base (v1 sync trainer, commit 130aa20-era). Because this
class *removes* init steps (unlike verl's own subclasses, which mostly *add*),
any new mandatory init step added by verl in ``__init__``/``_setup`` must be
re-checked here on upgrade.
"""

from __future__ import annotations

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class SinglePPOTrainer(PPOTrainerSync):
    """Synchronous v1 PPO trainer without outer-owned dataloader/replay buffer."""

    def init_training_runtime(self) -> None:
        """Initialize the colocated trainer/rollout runtime without syncing weights."""
        self._setup()

    def _build_replay_buffer(self):
        """The outer trainer owns the shared replay buffer; a policy never samples."""
        return None

    def _init_dataloader(self):
        """The outer trainer owns the shared prompt stream; skip per-policy dataset load."""
        return None

    def fit(self, agent_loop_manager):
        """Refuse to run the standalone training loop.

        This is a partial trainer: ``_init_dataloader`` and
        ``_build_replay_buffer`` are disabled, so ``fit()`` would silently
        consume verl's default (unused) data config. Multi-agent training must
        go through ``MultiPolicyTrainer``, which owns the shared
        dataloader/replay buffer and drives the phase methods directly.
        """
        raise RuntimeError(
            "SinglePPOTrainer cannot run its own fit(): it has no dataloader or "
            "replay buffer. Use MultiPolicyTrainer for multi-agent orchestration."
        )

    def step(self, metrics, timing_raw):
        """Refuse to run the standalone training step (see :meth:`fit`)."""
        raise RuntimeError(
            "SinglePPOTrainer cannot run its own step(): it has no dataloader or "
            "replay buffer. Use MultiPolicyTrainer for multi-agent orchestration."
        )
