"""OpenClaw On-Policy Distillation (OPD) loss, registered into verl."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig
from verl.workers.utils.padding import no_padding_2_padding


@register_distillation_loss(DistillationLossSettings(names=["openclaw_opd"], use_estimator=True))  # type: ignore[arg-type]
def compute_openclaw_opd_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Per-token reverse-KL estimate ``student_logp - teacher_logp``.

    Returns ``(distillation_losses (bsz, seqlen), metrics)``. Under
    ``use_policy_gradient=True`` the framework negates this to form the OPD
    advantage ``teacher_logp - student_logp`` and applies the PPO surrogate.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()

    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape, (
        f"shape mismatch: student={student_log_probs.shape} "
        f"teacher={teacher_log_probs.shape} mask={response_mask_bool.shape}"
    )

    distillation_losses = student_log_probs - teacher_log_probs

    valid = distillation_losses[response_mask_bool]
    metrics = {
        "distillation/opd_advantage": Metric(
            AggregationType.MEAN, (teacher_log_probs - student_log_probs)[response_mask_bool].mean()
        ),
        "distillation/abs_loss": Metric(AggregationType.MEAN, valid.abs().mean()),
    }
    return distillation_losses, metrics
