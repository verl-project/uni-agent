"""OpenClaw RL trainer for verl ``fully_async_policy``."""

from __future__ import annotations

import ray

from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer as _FullyAsyncTrainerActor,
)
from verl.utils.debug import marked_timer

_FullyAsyncTrainerBase = _FullyAsyncTrainerActor.__ray_actor_class__


class _OpenClawFullyAsyncTrainerImpl(_FullyAsyncTrainerBase):
    """Implementation class (decorated with ``@ray.remote`` below)."""

    def _fit_compute_advantage(self, batch):
        """Reward-direct advantage: broadcast the signed PRM score to all response tokens."""

        metrics = self.metrics
        timing_raw = self.timing_raw
        reward_tensor = self.reward_tensor  # [B, response_len], score on the last response token

        with marked_timer("adv", timing_raw, color="brown"):
            batch.batch["token_level_scores"] = reward_tensor
            batch.batch["token_level_rewards"] = reward_tensor

            response_mask = batch.batch["response_mask"].to(reward_tensor.dtype)
            # The single non-zero (last-token) entry collapses to the per-sample signed score.
            seq_score = reward_tensor.sum(dim=-1, keepdim=True)  # [B, 1]
            advantages = seq_score * response_mask
            batch.batch["advantages"] = advantages
            batch.batch["returns"] = advantages

            scores = seq_score.squeeze(-1)
            if scores.numel() > 0:
                metrics["openclaw/rl/reward_mean"] = scores.mean().item()
                metrics["openclaw/rl/reward_pos_frac"] = (scores > 0).float().mean().item()
                metrics["openclaw/rl/reward_neg_frac"] = (scores < 0).float().mean().item()
                metrics["openclaw/rl/learnable_frac"] = (scores != 0).float().mean().item()
        return batch


OpenClawFullyAsyncTrainer = ray.remote(num_cpus=10)(_OpenClawFullyAsyncTrainerImpl)
