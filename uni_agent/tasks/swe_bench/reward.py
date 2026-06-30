"""Reward wiring for the SWE-bench task.

The scorer itself lives in the shared reward registry
(:mod:`uni_agent.reward.swe_bench`); here we only declare which spec + params
this task scores with. ``run.py`` passes this to
:func:`~uni_agent.reward.load_reward_spec` itself at run time.
"""

from __future__ import annotations

from typing import Any


def reward_config() -> dict[str, Any]:
    """Reward-spec config for SWE-bench (consumed by ``load_reward_spec``)."""
    return {"name": "swe_bench"}
