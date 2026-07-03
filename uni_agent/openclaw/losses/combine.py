"""OpenClaw Combine (OPD + GRPO) loss, registered into verl."""

from __future__ import annotations

import os
from typing import Any

import torch
from tensordict import TensorDict

from uni_agent.openclaw.losses._core import DEFAULT_EPS_CLIP_HIGH, DEFAULT_EPS_CLIP_LOW
from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig
from verl.workers.utils.padding import no_padding_2_padding


def _weight(env_name: str, default: float) -> float:
    try:
        return float(os.environ.get(env_name, default))
    except (TypeError, ValueError):
        return default


def _bool_env(env_name: str, default: bool = False) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@register_distillation_loss(DistillationLossSettings(names=["openclaw_combine"], use_estimator=True))  # type: ignore[arg-type]
def compute_openclaw_combine_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-token PPO surrogate over the combined OPD + GRPO advantage.

    Returns ``(per_token_pg_loss (bsz, seqlen), metrics)``. With
    ``use_policy_gradient=false`` and ``use_task_rewards=false`` the framework
    simply aggregates this tensor with ``agg_loss`` -- giving the single-clip
    combine objective.
    """
    w_opd = _weight("OPENCLAW_COMBINE_W_OPD", 1.0)
    w_rl = _weight("OPENCLAW_COMBINE_W_RL", 1.0)
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else DEFAULT_EPS_CLIP_LOW
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else DEFAULT_EPS_CLIP_HIGH

    new_logp = no_padding_2_padding(model_output["log_probs"], data)
    teacher_logp = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)

    old_logp = data["old_log_probs"]
    if old_logp.is_nested:
        old_logp = old_logp.to_padded_tensor(0.0)

    advantages = data["advantages"]
    if advantages.is_nested:
        advantages = advantages.to_padded_tensor(0.0)

    response_mask = data["response_mask"]
    if response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)

    teacher_adv = teacher_logp - old_logp

    # RL-only turns carry teacher_logprobs == rollout_log_probs, so teacher_adv is
    # only as small as the rollout-vs-train log-prob drift (not strictly zero).
    # When OPENCLAW_COMBINE_ZERO_TEACHER_ON_RL is set, gate teacher_adv to exactly
    # zero on RL-only rows using the validity marker carried in teacher_ids
    # (1 = real teacher signal, 0 = RL-only). Default off -> behaviour unchanged.
    if _bool_env("OPENCLAW_COMBINE_ZERO_TEACHER_ON_RL", False) and "teacher_ids" in data.keys():
        teacher_valid = no_padding_2_padding(data["teacher_ids"], data)
        if teacher_valid.dim() > teacher_adv.dim():
            teacher_valid = teacher_valid.squeeze(-1)
        teacher_adv = teacher_adv * (teacher_valid > 0).to(teacher_adv.dtype)

    combined_adv = w_opd * teacher_adv + w_rl * advantages

    negative_approx_kl = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    pg_losses1 = -combined_adv * ratio
    pg_losses2 = -combined_adv * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
    per_token_pg = torch.maximum(pg_losses1, pg_losses2)

    mask_bool = response_mask.bool()
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()[mask_bool].mean() if mask_bool.any() else torch.tensor(0.0)
    metrics = {
        "distillation/combine_clipfrac": Metric(AggregationType.MEAN, clipfrac),
        "distillation/combine_teacher_adv": Metric(
            AggregationType.MEAN, teacher_adv[mask_bool].mean() if mask_bool.any() else torch.tensor(0.0)
        ),
        "distillation/combine_reward_adv": Metric(
            AggregationType.MEAN, advantages[mask_bool].mean() if mask_bool.any() else torch.tensor(0.0)
        ),
    }
    return per_token_pg, metrics
