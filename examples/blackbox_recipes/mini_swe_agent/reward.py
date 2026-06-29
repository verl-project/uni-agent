"""Reward utilities for the blackbox SWE-agent recipe.

Contains:
- build_reward_context: extract reward metadata + eval_timeout from tools_kwargs
- compute_score: thin reward function that reads reward_score from extra_info
- evaluate_in_env: run reward evaluation in Docker env (shared by both runners)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def build_reward_context(tools_kwargs: dict) -> tuple[dict[str, Any], int]:
    """Extract reward metadata and eval_timeout from per-sample tools_kwargs."""
    reward_config = tools_kwargs.get("reward", {})
    metadata = {
        "data_source": reward_config.get("name", "unknown"),
        "reward_model": reward_config.get("metadata", {}),
    }
    eval_timeout = int(os.environ.get("SWE_AGENT_EVAL_TIMEOUT", "600"))
    return metadata, eval_timeout


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """Read reward_score from extra_info, injected by the agent runner."""
    score = 0.0
    if extra_info and "reward_score" in extra_info:
        score = float(extra_info["reward_score"])
    return {"score": score}


def _get_reward_spec(data_source: str):
    """Load reward spec class by data_source name."""
    from uni_agent.reward.registry import REWARD_SPEC_REGISTRY, _load_reward_spec_module

    if data_source not in REWARD_SPEC_REGISTRY:
        _load_reward_spec_module(data_source)
    cls = REWARD_SPEC_REGISTRY.get(data_source)
    if cls is None:
        raise ValueError(f"Unknown data_source: {data_source}. Available: {list(REWARD_SPEC_REGISTRY.keys())}")
    return cls


async def evaluate_in_env(
    env,
    metadata: dict[str, Any],
    eval_timeout: int = 600,
) -> tuple[float, dict]:
    """Run reward evaluation in the Docker env.

    Returns (score, eval_result) where score is 1.0/0.0 and
    eval_result contains details (eval_completed, resolved, etc.).
    """
    data_source = metadata.get("data_source", "unknown")
    reward_model = metadata.get("reward_model", {})

    spec_cls = _get_reward_spec(data_source)
    spec_metadata = reward_model.get("ground_truth", reward_model)

    spec = spec_cls(
        run_id="swe_bb_eval",
        metadata=spec_metadata,
        env=env,
        eval_timeout=eval_timeout,
    )

    resolved, result = await spec.compute_reward()
    score = 1.0 if resolved else 0.0
    return score, result
