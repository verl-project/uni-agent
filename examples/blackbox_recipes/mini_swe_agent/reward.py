"""Trainer reward shim; the unified SWE task reports the actual reward."""

from __future__ import annotations


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """Read the score posted by the unified task runner."""
    del data_source, solution_str, ground_truth
    score = 0.0
    if extra_info:
        score = float(extra_info.get("reward", extra_info.get("reward_score", 0.0)))
    return {"score": score}
