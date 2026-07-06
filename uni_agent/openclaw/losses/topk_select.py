"""OpenClaw Top-K-Select OPD loss, registered into verl."""

from __future__ import annotations

import os
from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig
from verl.workers.utils.padding import no_padding_2_padding


def _select_candidate_teacher_logp(
    teacher_cand: torch.Tensor,  # (B, S, C)
    student_logp: torch.Tensor,  # (B, S)
    data: TensorDict,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(selected_teacher_logp (B,S), overlap_ratio_metric_tensor)``.

    Per-token candidate selection. When student/teacher top-K ids are present we
    select by top-K overlap; otherwise we fall back to the candidate that
    maximizes the teacher log-prob (a token-optimal distillation proxy).
    """
    B, S, C = teacher_cand.shape
    overlap_metric = torch.tensor(0.0)

    student_topk = data.get("student_topk_ids", None)
    teacher_topk_cand = data.get("teacher_topk_ids_cand", None)

    if mode == "shortest":
        return teacher_cand[..., 0], overlap_metric

    if student_topk is not None and teacher_topk_cand is not None:
        student_topk = no_padding_2_padding(student_topk, data)  # (B, S, K)
        teacher_topk_cand = no_padding_2_padding(teacher_topk_cand, data)  # (B, S, C, K)
        # overlap[b,s,c] = |student_topk ∩ teacher_topk_cand|
        st = student_topk.unsqueeze(2).unsqueeze(-1)  # (B,S,1,K,1)
        tc = teacher_topk_cand.unsqueeze(-2)  # (B,S,C,1,K)
        overlap = (st == tc).any(dim=-1).sum(dim=-1).float()  # (B,S,C)
        if mode == "sequence_optimal":
            sel = overlap.sum(dim=1).argmax(dim=-1)  # (B,)
            sel = sel.view(B, 1).expand(B, S)
        else:  # token_optimal
            sel = overlap.argmax(dim=-1)  # (B,S)
        overlap_metric = overlap.gather(-1, sel.unsqueeze(-1)).squeeze(-1).mean()
    else:
        # No overlap info: pick the candidate with the largest teacher logp.
        sel = teacher_cand.argmax(dim=-1)  # (B,S)

    selected = teacher_cand.gather(-1, sel.unsqueeze(-1)).squeeze(-1)  # (B,S)
    return selected, overlap_metric


@register_distillation_loss(DistillationLossSettings(names=["openclaw_topk_select"], use_estimator=True))  # type: ignore[arg-type]
def compute_openclaw_topk_select_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Overlap-selected multi-candidate OPD reverse-KL estimate.

    Returns ``(distillation_losses (bsz, seqlen), metrics)`` where the per-token
    loss is ``student_logp - teacher_sel_logp``; with ``use_policy_gradient=true``
    the advantage becomes ``teacher_sel_logp - student_logp``.
    """
    mode = os.environ.get("OPENCLAW_TOPK_SELECT_MODE", "token_optimal")
    student_logp = no_padding_2_padding(model_output["log_probs"], data)

    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()

    teacher_cand = data.get("teacher_logprobs_cand", None)
    overlap_metric = torch.tensor(0.0)
    if teacher_cand is not None:
        teacher_cand = no_padding_2_padding(teacher_cand, data)  # (B, S, C)
        teacher_sel, overlap_metric = _select_candidate_teacher_logp(teacher_cand, student_logp, data, mode)
    else:
        # Graceful fallback to single-candidate OPD.
        teacher_sel = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)

    assert teacher_sel.shape == student_logp.shape == response_mask_bool.shape

    distillation_losses = student_logp - teacher_sel
    valid = distillation_losses[response_mask_bool]
    metrics = {
        "distillation/topk_select_opd_advantage": Metric(
            AggregationType.MEAN, (teacher_sel - student_logp)[response_mask_bool].mean()
        ),
        "distillation/topk_select_overlap": Metric(AggregationType.MEAN, overlap_metric),
        "distillation/abs_loss": Metric(AggregationType.MEAN, valid.abs().mean()),
    }
    return distillation_losses, metrics
