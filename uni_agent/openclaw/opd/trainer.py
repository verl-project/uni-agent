"""OpenClaw OPD trainer for verl ``fully_async_policy``."""

from __future__ import annotations

import ray

from verl.experimental.fully_async_policy.fully_async_trainer import (
    FullyAsyncTrainer as _FullyAsyncTrainerActor,
)
from verl.utils.debug import marked_timer

_FullyAsyncTrainerBase = _FullyAsyncTrainerActor.__ray_actor_class__


class _OpenClawOPDTrainerImpl(_FullyAsyncTrainerBase):
    """Implementation class (decorated with ``@ray.remote`` below)."""

    def _fit_compute_advantage(self, batch):
        """Zero the task advantage; the OPD signal lives in the distillation loss."""
        metrics = self.metrics
        timing_raw = self.timing_raw
        reward_tensor = self.reward_tensor  # zeros for OPD samples

        with marked_timer("adv", timing_raw, color="brown"):
            zeros = reward_tensor * 0.0
            batch.batch["token_level_scores"] = zeros
            batch.batch["token_level_rewards"] = zeros
            batch.batch["advantages"] = zeros
            batch.batch["returns"] = zeros
            metrics["openclaw/opd/num_samples"] = float(reward_tensor.shape[0])
        return batch


OpenClawOPDTrainer = ray.remote(num_cpus=10)(_OpenClawOPDTrainerImpl)
