"""Reward function for the multi-agent blackbox example."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

_EXPECTED_KEYS = (
    "target",
    "expected",
    "expected_text",
    "expected_answer",
    "expected_answers",
    "answer",
    "ground_truth",
    "ground_truth_raw",
)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _answer_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        candidates: list[str] = []
        for key in _EXPECTED_KEYS:
            if key in value:
                candidates.extend(_answer_candidates(value[key]))
        return candidates
    if isinstance(value, (list, tuple, set)):
        candidates = []
        for item in value:
            candidates.extend(_answer_candidates(item))
        return candidates
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "(")) and stripped.endswith(("]", ")")):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                pass
            else:
                if isinstance(parsed, (list, tuple, set)):
                    return _answer_candidates(parsed)
        return [stripped] if stripped else []
    return [str(value).strip()]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a rollout-level reward using verl's custom reward API.

    ``MultiAgentGatewayFramework`` supplies MAS context under
    ``extra_info.runner_reward_info.reward_context``. Return flat numeric
    metrics: verl's reward manager wraps them in ``reward_extra_info``.
    Production integrations should replace this with a task evaluator.
    """
    extra_info = dict(extra_info or {})
    runner_info = extra_info.get("runner_reward_info") or {}
    context = runner_info.get("reward_context") or {}
    final_result = context.get("final_result", solution_str)
    agent_outputs = context.get("agent_outputs") or {}

    expected_candidates = list(
        dict.fromkeys(
            candidate for candidate in _answer_candidates(ground_truth) if candidate
        )
    )

    final_text = _text(final_result).strip()
    if expected_candidates:
        final_text_lower = final_text.lower()
        score = (
            1.0
            if any(
                candidate.lower() in final_text_lower
                for candidate in expected_candidates
            )
            else 0.0
        )
    else:
        # No usable ground truth: never reward arbitrary output. Returning 0.0
        # keeps GRPO from treating every non-empty answer as correct.
        score = 0.0

    return {
        "score": score,
        "acc": score,
        "num_agent_outputs": len(agent_outputs),
    }
