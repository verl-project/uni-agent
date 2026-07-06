"""Build a per-turn OPD ``RolloutSample`` (teacher log-probs carried in-batch)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from uni_agent.openclaw.common.sample_builder import build_base_tensors, build_teacher_full_seq
from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import RolloutSample


def build_opd_rollout_sample(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_logprobs: list[float],
    teacher_log_probs: list[float],
    sample_id: str,
    uid: str,
    pad_token_id: int,
    prompt_length: int,
    response_length: int,
    param_version: int,
    teacher_topk_log_probs: Optional[list[list[float]]] = None,
    teacher_topk_indices: Optional[list[list[int]]] = None,
    distill_topk: int = 0,
    rollout_status: Optional[dict] = None,
    generate_seconds: float = 0.0,
    turn: int = 0,
    eval_score: Optional[float] = None,
) -> RolloutSample:
    """Build a single-row OPD :class:`RolloutSample` with teacher signal."""
    base = build_base_tensors(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_logprobs=response_logprobs,
        pad_token_id=pad_token_id,
        prompt_length=prompt_length,
        response_length=response_length,
    )

    # OPD has no task reward; carry zero rm_scores so the trainer's
    # _fit_compute_reward skips the reward model (the GenRM is the proxy's
    # judge/teacher, not a verl reward manager).
    rm_scores_t = torch.zeros((1, response_length), dtype=torch.float32)

    use_topk = int(distill_topk or 0) > 0
    K = int(distill_topk) if use_topk else 1
    teacher_lp_full, teacher_id_full = build_teacher_full_seq(
        teacher_log_probs=teacher_log_probs,
        prompt_length=prompt_length,
        response_length=response_length,
        valid_response_length=base.valid_response_length,
        input_ids_row=base.input_ids_row,
        K=K,
        teacher_topk_log_probs=teacher_topk_log_probs if use_topk else None,
        teacher_topk_indices=teacher_topk_indices if use_topk else None,
    )

    tensors = {
        **base.tensors,
        "rm_scores": rm_scores_t,
        "teacher_logprobs": teacher_lp_full.unsqueeze(0),  # [1, seq_len, K]
        "teacher_ids": teacher_id_full.unsqueeze(0),  # [1, seq_len, K]
    }

    non_tensors = {
        "uid": np.array([uid], dtype=object),
        "__num_turns__": np.array([1], dtype=np.int32),
        "min_global_steps": np.array([int(param_version)], dtype=object),
        "max_global_steps": np.array([int(param_version)], dtype=object),
        "openclaw_turn": np.array([int(turn)], dtype=object),
        "openclaw_opd_topk": np.array([int(distill_topk or 0)], dtype=object),
        "openclaw_eval_score": np.array([float(eval_score) if eval_score is not None else 0.0], dtype=object),
    }

    meta_info = {
        "metrics": [{"generate_sequences": float(generate_seconds), "tool_calls": 0.0}],
    }

    full_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    return RolloutSample(
        full_batch=full_batch,
        sample_id=sample_id,
        epoch=0,
        rollout_status=rollout_status or {},
    )
