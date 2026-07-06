"""Unit tests for the shared OpenClaw sample-builder helpers.

These cover the trickiest, most regression-prone piece of the online/OPD/Combine
data path -- the per-turn tensor layout and the full-sequence teacher alignment
-- without importing verl/torch-distributed machinery. Only
:mod:`uni_agent.openclaw.common.sample_builder` (torch + protocol) is exercised.
"""

import pytest
import torch

from uni_agent.openclaw.common.sample_builder import (
    build_base_tensors,
    build_teacher_full_seq,
    left_pad,
    right_pad,
)


def test_left_pad_and_right_pad():
    tokens, attn = left_pad([1, 2, 3], 5, pad_id=0)
    assert tokens == [0, 0, 1, 2, 3]
    assert attn == [0, 0, 1, 1, 1]

    tokens, attn = right_pad([1, 2, 3], 5, pad_id=0)
    assert tokens == [1, 2, 3, 0, 0]
    assert attn == [1, 1, 1, 0, 0]

    # truncation keeps the right side for prompts, left side for responses
    assert left_pad([1, 2, 3, 4], 2, pad_id=0)[0] == [3, 4]
    assert right_pad([1, 2, 3, 4], 2, pad_id=0)[0] == [1, 2]


def test_build_base_tensors_shapes_and_padding():
    prompt_length, response_length = 6, 4
    base = build_base_tensors(
        prompt_ids=[11, 12, 13],
        response_ids=[21, 22],
        response_logprobs=[-0.1, -0.2],
        pad_token_id=0,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    t = base.tensors
    assert base.valid_response_length == 2
    assert base.seq_len == prompt_length + response_length

    assert t["prompts"].shape == (1, prompt_length)
    assert t["responses"].shape == (1, response_length)
    assert t["input_ids"].shape == (1, prompt_length + response_length)
    assert t["attention_mask"].shape == (1, prompt_length + response_length)
    assert t["position_ids"].shape == (1, prompt_length + response_length)
    assert t["rollout_log_probs"].shape == (1, response_length)

    # prompt left-padded, response right-padded
    assert t["prompts"][0].tolist() == [0, 0, 0, 11, 12, 13]
    assert t["responses"][0].tolist() == [21, 22, 0, 0]
    assert t["response_mask"][0].tolist() == [1, 1, 0, 0]
    # rollout log probs padded with 0.0 after the 2 valid tokens
    assert t["rollout_log_probs"][0].tolist()[:2] == pytest.approx([-0.1, -0.2], abs=1e-6)
    assert t["rollout_log_probs"][0].tolist()[2:] == [0.0, 0.0]

    # position_ids: cumsum(attention_mask) - 1, clipped at 0
    attn = t["attention_mask"][0]
    expected_pos = torch.clip(attn.cumsum(dim=-1) - 1, min=0)
    assert torch.equal(t["position_ids"][0], expected_pos)
    assert base.input_ids_row.tolist() == [0, 0, 0, 11, 12, 13, 21, 22, 0, 0]


def _no_padding_response_slice(full_seq_values, attention_mask, valid_response_length):
    """Reproduce verl ``no_padding_2_padding`` for a single un-padded sequence.

    The engine unpads the full sequence (keeping attention_mask==1 positions in
    order) then the loss slices ``values[seq_offset - resp_len - 1 : seq_offset - 1]``
    -- the one-token left shift onto the response predictor slots.
    """
    mask = attention_mask.bool()
    values = full_seq_values[mask]  # no-padding values in sequence order
    seq_offset = values.shape[0]
    resp_len = valid_response_length
    return values[seq_offset - resp_len - 1 : seq_offset - 1]


def test_build_teacher_full_seq_alignment_roundtrip():
    prompt_length, response_length = 6, 4
    teacher_vals = [-0.5, -0.6, -0.7]
    base = build_base_tensors(
        prompt_ids=[11, 12, 13],
        response_ids=[21, 22, 23],
        response_logprobs=[-0.1, -0.2, -0.3],
        pad_token_id=0,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    teacher_lp_full, teacher_id_full = build_teacher_full_seq(
        teacher_log_probs=teacher_vals,
        prompt_length=prompt_length,
        response_length=response_length,
        valid_response_length=base.valid_response_length,
        input_ids_row=base.input_ids_row,
        K=1,
    )
    assert teacher_lp_full.shape == (prompt_length + response_length, 1)
    assert teacher_id_full.shape == (prompt_length + response_length, 1)

    # values placed at predictor slots prompt_length-1 + j
    start = prompt_length - 1
    for j, v in enumerate(teacher_vals):
        assert abs(teacher_lp_full[start + j, 0].item() - v) < 1e-6

    # round trip through the verl no-padding/left-shift slice recovers the values
    recovered = _no_padding_response_slice(
        teacher_lp_full.squeeze(-1),
        base.tensors["attention_mask"][0],
        base.valid_response_length,
    )
    assert recovered.tolist() == pytest.approx(teacher_vals, abs=1e-6)


def test_build_teacher_full_seq_topk():
    prompt_length, response_length, K = 4, 3, 2
    base = build_base_tensors(
        prompt_ids=[11, 12],
        response_ids=[21, 22],
        response_logprobs=[-0.1, -0.2],
        pad_token_id=0,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    topk_lp = [[-0.1, -0.9], [-0.2, -0.8]]
    topk_idx = [[5, 6], [7, 8]]
    teacher_lp_full, teacher_id_full = build_teacher_full_seq(
        teacher_log_probs=[],
        prompt_length=prompt_length,
        response_length=response_length,
        valid_response_length=base.valid_response_length,
        input_ids_row=base.input_ids_row,
        K=K,
        teacher_topk_log_probs=topk_lp,
        teacher_topk_indices=topk_idx,
    )
    assert teacher_lp_full.shape == (prompt_length + response_length, K)
    start = prompt_length - 1
    assert teacher_lp_full[start].tolist() == pytest.approx([-0.1, -0.9], abs=1e-6)
    assert teacher_id_full[start].tolist() == [5, 6]
    assert teacher_lp_full[start + 1].tolist() == pytest.approx([-0.2, -0.8], abs=1e-6)
    assert teacher_id_full[start + 1].tolist() == [7, 8]
