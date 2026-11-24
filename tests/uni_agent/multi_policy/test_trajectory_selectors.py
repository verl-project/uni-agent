"""Tests for the multi_policy trajectory selectors."""

from __future__ import annotations

import pytest

from examples.multi_policy.trajectory_selectors.selectors import longest
from uni_agent.gateway.session import Trajectory


def _trajectory(
    *,
    response_ids: list[int],
    response_mask: list[int] | None = None,
    num_turns: int = 1,
) -> Trajectory:
    return Trajectory(
        prompt_ids=[1],
        response_ids=response_ids,
        response_mask=response_mask if response_mask is not None else [1] * len(response_ids),
        num_turns=num_turns,
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_longest_keeps_the_longest_trajectory():
    short = _trajectory(response_ids=[1])
    long = _trajectory(response_ids=[1, 2, 3])
    medium = _trajectory(response_ids=[1, 2])

    kept = longest("session-1", [short, long, medium])

    assert len(kept) == 1
    assert kept[0] is long


@pytest.mark.cpu
@pytest.mark.level0
def test_longest_prefers_model_tokens_over_raw_length():
    # More response ids, but most of them are context tokens (mask 0).
    padded = _trajectory(response_ids=[1, 2, 3], response_mask=[1, 0, 0])
    generated = _trajectory(response_ids=[1, 2], response_mask=[1, 1])

    kept = longest("session-1", [padded, generated])

    assert len(kept) == 1
    assert kept[0] is generated


@pytest.mark.cpu
@pytest.mark.level0
def test_longest_breaks_ties_with_list_order():
    # Equal metrics; the later list position wins, matching the built-in longest.
    earlier = _trajectory(response_ids=[1])
    later = _trajectory(response_ids=[1])

    kept = longest("session-1", [earlier, later])

    assert len(kept) == 1
    assert kept[0] is later


@pytest.mark.cpu
@pytest.mark.level0
def test_longest_handles_empty_and_single():
    assert longest("session-1", []) == []

    only = _trajectory(response_ids=[1])
    kept = longest("session-1", [only])

    assert len(kept) == 1
    assert kept[0] is only
