"""OpenClaw Binary RL policy loss, registered into verl."""

from __future__ import annotations

from typing import Any, Optional

import torch

from uni_agent.openclaw.losses._core import DEFAULT_EPS_CLIP_HIGH, DEFAULT_EPS_CLIP_LOW
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.utils import torch_functional as verl_F
from verl.workers.config import ActorConfig


@register_policy_loss("openclaw_grpo")
def compute_openclaw_grpo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """GRPO with asymmetric PPO clipping (OpenClaw Binary RL)."""
    assert config is not None
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else DEFAULT_EPS_CLIP_LOW
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else DEFAULT_EPS_CLIP_HIGH

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
