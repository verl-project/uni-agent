"""Shared tensor builders for OpenClaw fully-async RolloutSample rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def left_pad(values: Sequence, target_len: int, pad=None, *, pad_id=None):
    return_mask = pad_id is not None
    pad = pad_id if return_mask else pad
    values = list(values)
    if len(values) >= target_len:
        padded = values[-target_len:]
        mask = [1] * len(padded)
    else:
        pad_len = target_len - len(values)
        padded = [pad] * pad_len + values
        mask = [0] * pad_len + [1] * len(values)
    return (padded, mask) if return_mask else padded


def right_pad(values: Sequence, target_len: int, pad=None, *, pad_id=None):
    return_mask = pad_id is not None
    pad = pad_id if return_mask else pad
    values = list(values)
    if len(values) >= target_len:
        padded = values[:target_len]
        mask = [1] * len(padded)
    else:
        pad_len = target_len - len(values)
        padded = values + [pad] * pad_len
        mask = [1] * len(values) + [0] * pad_len
    return (padded, mask) if return_mask else padded


@dataclass
class BaseTensors:
    tensors: dict[str, torch.Tensor]
    valid_response_length: int
    input_ids_row: torch.Tensor
    seq_len: int


def build_base_tensors(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    response_logprobs: list[float],
    pad_token_id: int,
    prompt_length: int,
    response_length: int,
) -> BaseTensors:
    """Build the common tensor payload shared by RL/OPD/Combine samples."""
    valid_response_length = min(len(response_ids), response_length)

    prompts = left_pad(prompt_ids, prompt_length, pad_token_id)
    responses = right_pad(response_ids, response_length, pad_token_id)
    response_mask = right_pad([1] * valid_response_length, response_length, 0)
    rollout_log_probs = right_pad(response_logprobs, response_length, 0.0)

    prompt_mask = [0 if t == pad_token_id else 1 for t in prompts]
    attention_mask_row = torch.tensor(prompt_mask + response_mask, dtype=torch.long)
    input_ids_row = torch.tensor(prompts + responses, dtype=torch.long)
    position_ids_row = torch.clamp(torch.cumsum(attention_mask_row, dim=0) - 1, min=0)

    tensors = {
        "prompts": torch.tensor([prompts], dtype=torch.long),
        "responses": torch.tensor([responses], dtype=torch.long),
        "response_mask": torch.tensor([response_mask], dtype=torch.long),
        "input_ids": input_ids_row.unsqueeze(0),
        "attention_mask": attention_mask_row.unsqueeze(0),
        "position_ids": position_ids_row.unsqueeze(0),
        "rollout_log_probs": torch.tensor([rollout_log_probs], dtype=torch.float32),
    }
    return BaseTensors(
        tensors=tensors,
        valid_response_length=valid_response_length,
        input_ids_row=input_ids_row,
        seq_len=input_ids_row.numel(),
    )


def build_teacher_full_seq(
    *,
    teacher_log_probs: list[float],
    prompt_length: int,
    response_length: int,
    valid_response_length: int,
    input_ids_row: torch.Tensor,
    K: int,
    teacher_topk_log_probs: list[list[float]] | None = None,
    teacher_topk_indices: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project response-aligned teacher signals into full-sequence predictor slots."""
    seq_len = int(prompt_length + response_length)
    teacher_lp_full = torch.zeros((seq_len, K), dtype=torch.float32)
    teacher_id_full = torch.zeros((seq_len, K), dtype=torch.long)

    # Predictor slots for response token j are at (prompt_length - 1 + j).
    start = prompt_length - 1
    for j in range(valid_response_length):
        pos = start + j
        if pos < 0 or pos >= seq_len:
            break

        if (
            K > 1
            and teacher_topk_log_probs is not None
            and teacher_topk_indices is not None
            and j < len(teacher_topk_log_probs)
            and j < len(teacher_topk_indices)
        ):
            lp_row = right_pad(teacher_topk_log_probs[j], K, 0.0)
            id_row = right_pad(teacher_topk_indices[j], K, 0)
            teacher_lp_full[pos] = torch.tensor(lp_row, dtype=torch.float32)
            teacher_id_full[pos] = torch.tensor(id_row, dtype=torch.long)
            continue

        # K == 1 (or no top-k payload): fill first column from token logprob.
        lp = teacher_log_probs[j] if j < len(teacher_log_probs) else 0.0
        teacher_lp_full[pos, 0] = float(lp)

        # Predictor target token id for slot `pos`.
        target_idx = pos + 1
        if 0 <= target_idx < input_ids_row.numel():
            teacher_id_full[pos, 0] = int(input_ids_row[target_idx].item())

    return teacher_lp_full, teacher_id_full
