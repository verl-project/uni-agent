from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REWARD_PATH = Path(__file__).parents[3] / "uni_agent" / "tasks" / "hotpotqa" / "reward.py"
SPEC = importlib.util.spec_from_file_location("hotpotqa_reward", REWARD_PATH)
assert SPEC is not None and SPEC.loader is not None
REWARD_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARD_MODULE)
compute_score = REWARD_MODULE.compute_score


@pytest.mark.parametrize(
    ("solution", "ground_truths", "expected"),
    [
        (r"Final answer: \boxed{alpha beta}", ["alpha beta"], 1.0),
        (r"Final answer: \boxed{alpha beta gamma}", ["alpha beta"], 0.0),
        (r"Final answer: \boxed{Alpha Beta}", ["alphabeta"], 1.0),
        (r"Earlier: \boxed{wrong}. Final: \boxed{alpha beta}", ["wrong"], 0.0),
        (r"Final answer: \boxed{alpha beta}", ["wrong", "alpha beta"], 1.0),
        ("alpha beta", ["alpha beta"], 0.0),
    ],
)
def test_compute_score_matches_official_memagent_verifier(
    solution: str,
    ground_truths: list[str],
    expected: float,
) -> None:
    """Catch replacing the official normalized exact match with partial overlap."""

    assert compute_score(solution, ground_truths) == expected


def test_compute_score_returns_zero_for_no_ground_truths() -> None:
    """Keep the integration safe when a malformed sample has no accepted answer."""

    assert compute_score(r"Final answer: \boxed{alpha beta}", []) == 0.0
