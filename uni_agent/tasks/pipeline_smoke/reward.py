"""Correctness and tool-compliance reward for the pipeline smoke task."""

from __future__ import annotations


def normalize_answer(value: object) -> str:
    """Normalize harmless presentation differences without changing semantics."""
    return " ".join(str(value).strip().casefold().split())


def compute_score(answer: object, expected_answer: object) -> float:
    """Return a deterministic binary reward for an exact normalized match."""
    normalized_answer = normalize_answer(answer)
    normalized_expected = normalize_answer(expected_answer)
    if not normalized_answer or not normalized_expected:
        return 0.0
    return float(normalized_answer == normalized_expected)


def compute_reward(answer: object, expected_answer: object, *, used_finish_tool: bool) -> float:
    """Reward a correct answer, with full credit for using the control tool."""
    accuracy = compute_score(answer, expected_answer)
    if not accuracy:
        return 0.0
    return 1.0 if used_finish_tool else 0.5
