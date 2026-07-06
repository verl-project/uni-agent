"""Convert a captured OpenClaw main turn into a verl ``RolloutSample``."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from uni_agent.openclaw.common.sample_builder import build_base_tensors
from uni_agent.openclaw.common.sample_builder import left_pad as _left_pad  # noqa: F401  (back-compat re-export)
from uni_agent.openclaw.common.sample_builder import right_pad as _right_pad  # noqa: F401  (back-compat re-export)
from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import RolloutSample


def build_rollout_sample(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_logprobs: list[float],
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
    has_next_state: bool = True,
) -> RolloutSample:
    """Build a single-row :class:`RolloutSample` from a captured main turn."""
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

    tensors = {**base.tensors, "rm_scores": rm_scores_t}

    non_tensors = {
        "uid": np.array([uid], dtype=object),
        "__num_turns__": np.array([1], dtype=np.int32),
        "min_global_steps": np.array([int(param_version)], dtype=object),
        "max_global_steps": np.array([int(param_version)], dtype=object),
        "openclaw_prm_score": np.array([float(score)], dtype=object),
        "openclaw_has_next_state": np.array([bool(has_next_state)], dtype=object),
        "openclaw_turn": np.array([int(turn)], dtype=object),
    }

    meta_info = {
        # Consumed by detach_utils.addition_process -> processing_times / tool_calls_times.
        "metrics": [{"generate_sequences": float(generate_seconds), "tool_calls": 0.0}],
    }

    full_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    return RolloutSample(
        full_batch=full_batch,
        sample_id=sample_id,
        epoch=0,
        rollout_status=rollout_status or {},
    )
