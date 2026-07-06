"""Build per-turn RolloutSample rows for OpenClaw Combine (RL + OPD)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from uni_agent.openclaw.common.sample_builder import build_base_tensors, build_teacher_full_seq
from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import RolloutSample


def build_combine_rollout_sample(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_logprobs: list[float],
    teacher_log_probs: list[float],
    score: float,
    sample_id: str,
    uid: str,
    pad_token_id: int,
    prompt_length: int,
    response_length: int,
    param_version: int,
    rollout_status: Optional[dict] = None,
    generate_seconds: float = 0.0,
    turn: int = 0,
    sample_kind: str = "opd+rl",
) -> RolloutSample:
    """Build one combine training sample.

    `score` is written onto the last valid response token in `rm_scores`.
    `teacher_log_probs` is response-aligned and remapped into full-sequence
    `teacher_logprobs` ([1, seq_len, 1]) with the one-token left shift used by
    verl's no-padding conversion. `teacher_ids` carries the rl-only validity
    marker (see module docstring).
    """
    base = build_base_tensors(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_logprobs=response_logprobs,
        pad_token_id=pad_token_id,
        prompt_length=prompt_length,
        response_length=response_length,
    )

    rm_scores_t = torch.zeros((1, response_length), dtype=torch.float32)
    rm_scores_t[0, base.valid_response_length - 1] = float(score)

    teacher_lp_full, _ = build_teacher_full_seq(
        teacher_log_probs=teacher_log_probs,
        prompt_length=prompt_length,
        response_length=response_length,
        valid_response_length=base.valid_response_length,
        input_ids_row=base.input_ids_row,
        K=1,
    )

    # rl-only validity marker carried in teacher_ids (1 = real teacher signal at
    # the response predictor slots, 0 = rl-only). Combine's estimator loss never
    # reads teacher_ids as vocab indices, so repurposing it is safe.
    is_rl_only = sample_kind == "rl_only"
    teacher_valid_full = torch.zeros((base.seq_len, 1), dtype=torch.long)
    if not is_rl_only:
        start = prompt_length - 1
        for j in range(base.valid_response_length):
            pos = start + j
            if pos >= base.seq_len:
                break
            teacher_valid_full[pos, 0] = 1

    tensors = {
        **base.tensors,
        "rm_scores": rm_scores_t,
        "teacher_logprobs": teacher_lp_full.unsqueeze(0),
        "teacher_ids": teacher_valid_full.unsqueeze(0),
    }

    non_tensors = {
        "uid": np.array([uid], dtype=object),
        "__num_turns__": np.array([1], dtype=np.int32),
        "min_global_steps": np.array([int(param_version)], dtype=object),
        "max_global_steps": np.array([int(param_version)], dtype=object),
        "openclaw_turn": np.array([int(turn)], dtype=object),
        "openclaw_prm_score": np.array([float(score)], dtype=object),
        "openclaw_combine_kind": np.array([sample_kind], dtype=object),
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
