"""Pure-torch math for the OpenClaw RL / OPD / Combine objectives."""

from __future__ import annotations

import torch

DEFAULT_EPS_CLIP_LOW = 0.2
DEFAULT_EPS_CLIP_HIGH = 0.28


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    """Mean of ``values`` over positions where ``mask`` is truthy."""
    mask = mask.to(values.dtype)
    if dim is None:
        denom = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denom
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (values * mask).sum(dim=dim) / denom


def opd_token_advantage(teacher_logp: torch.Tensor, student_logp: torch.Tensor) -> torch.Tensor:
    """Token-level OPD advantage: ``teacher_logp - student_logp``.

    This is the directional distillation signal: positive where the teacher is
    more confident than the student on the sampled token.
    """
    return teacher_logp - student_logp


def combined_advantage(
    teacher_logp: torch.Tensor,
    student_logp: torch.Tensor,
    reward: torch.Tensor,
    *,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
) -> torch.Tensor:
    """Hybrid OPD + GRPO advantage, per OpenClaw ``combine_loss``.

    ``reward`` is broadcast (already shaped like ``student_logp`` or a scalar
    per row). The OPD term is token-level.
    """
    return w_opd * opd_token_advantage(teacher_logp, student_logp) + w_rl * reward


def asymmetric_ppo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    eps_clip_low: float = DEFAULT_EPS_CLIP_LOW,
    eps_clip_high: float = DEFAULT_EPS_CLIP_HIGH,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO clipped surrogate with asymmetric clipping (token-mean aggregation).

    Returns ``(pg_loss, metrics)``. ``advantages`` may be per-token (OPD/Combine)
    or broadcast scalar reward (GRPO).
    """
    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - eps_clip_low, 1 + eps_clip_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_loss = masked_mean(pg_losses, response_mask)

    metrics = {
        "actor/pg_clipfrac": float(pg_clipfrac.detach().item()),
        "actor/ppo_kl": float(ppo_kl.detach().item()),
    }
    return pg_loss, metrics


def reverse_kl_estimate(student_logp: torch.Tensor, teacher_logp: torch.Tensor) -> torch.Tensor:
    """k1 reverse-KL estimate ``student_logp - teacher_logp`` (per token).

    Used as the distillation loss tensor. With policy-gradient distillation the
    advantage is ``-(student - teacher) = teacher - student`` (the OPD signal).
    """
    return student_logp - teacher_logp


# ---------------------------------------------------------------------------
# Top-K hint selection (overlap-based k*), pure helpers.
# ---------------------------------------------------------------------------


def overlap_count_per_token(
    student_topk_ids: torch.Tensor,
    teacher_cand_topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Count token-id overlap between the student top-K set and each teacher
    candidate's top-K set.

    Args:
        student_topk_ids: ``(R, K_s)`` student top-K ids per response token.
        teacher_cand_topk_ids: ``(C, R, K_t)`` teacher top-K ids per candidate.

    Returns:
        ``(C, R)`` overlap counts ``O[c, t] = |S^student_t ∩ S^teacher_{c,t}|``.
    """
    C, R, _ = teacher_cand_topk_ids.shape
    out = torch.zeros((C, R), dtype=torch.long, device=teacher_cand_topk_ids.device)
    for c in range(C):
        for t in range(R):
            s_set = set(student_topk_ids[t].tolist())
            cand = teacher_cand_topk_ids[c, t].tolist()
            out[c, t] = sum(1 for v in cand if v in s_set)
    return out


def select_k_star_per_token(overlap: torch.Tensor, mode: str = "token_optimal") -> torch.Tensor:
    """Select the best teacher candidate index per token from overlap counts.

    Args:
        overlap: ``(C, R)`` overlap counts.
        mode: ``"shortest"`` (always candidate 0, shortest hint),
              ``"token_optimal"`` (argmax overlap per token),
              ``"sequence_optimal"`` (single argmax over summed overlap).

    Returns:
        ``(R,)`` selected candidate index per token.
    """
    C, R = overlap.shape
    if mode == "shortest" or C == 1:
        return torch.zeros((R,), dtype=torch.long, device=overlap.device)
    if mode == "sequence_optimal":
        k = int(overlap.sum(dim=1).argmax().item())
        return torch.full((R,), k, dtype=torch.long, device=overlap.device)
    # token_optimal (default)
    return overlap.argmax(dim=0)
