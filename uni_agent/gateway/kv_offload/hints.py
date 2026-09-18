"""Deterministic Agent Hint scoring for Gateway KV-cache offload.

Only information available before backend generation is used.  Keeping this
module free of vLLM imports makes the scoring rule cheap to unit test in
Gateway-only environments.
"""

from __future__ import annotations

from dataclasses import dataclass


MIN_USEFUL_REMAINING_TOKENS = 1024


@dataclass(frozen=True)
class DynamicPriorityInputs:
    """Runtime snapshot used to score one generation request."""

    tools_available: bool
    has_active_chain: bool
    received_tool_result: bool
    context_tokens: int
    trajectory_capacity: int | None
    remaining_capacity: int | None
    rollback_applied: bool
    active_chain_count: int


@dataclass(frozen=True)
class PriorityDecision:
    """Final priority plus its explainable score components."""

    priority: int
    continuation_score: int
    recompute_cost_score: int
    remaining_capacity_score: int
    chain_state_score: int

    @property
    def band(self) -> str:
        if self.priority < 30:
            return "cold"
        if self.priority < 60:
            return "normal"
        if self.priority < 80:
            return "high"
        return "hot"


def _continuation_score(inputs: DynamicPriorityInputs) -> int:
    if inputs.received_tool_result:
        return 55
    if inputs.has_active_chain:
        return 40
    if inputs.tools_available:
        return 25
    return 10


def _recompute_cost_score(context_tokens: int) -> int:
    if context_tokens < 1024:
        return 0
    if context_tokens < 4096:
        return 5
    if context_tokens < 16_384:
        return 10
    if context_tokens < 32_768:
        return 15
    if context_tokens < 65_536:
        return 22
    return 30


def _remaining_capacity_score(inputs: DynamicPriorityInputs) -> int:
    capacity = inputs.trajectory_capacity
    remaining = inputs.remaining_capacity
    if capacity is None or remaining is None or capacity <= 0:
        return 0
    if remaining < MIN_USEFUL_REMAINING_TOKENS or remaining / capacity < 0.05:
        return -15
    if remaining >= 8192 and remaining / capacity >= 0.25:
        return 10
    return 0


def compute_dynamic_priority(inputs: DynamicPriorityInputs) -> PriorityDecision:
    """Compute a stable 0..100 KV priority from pre-generation state."""

    continuation = _continuation_score(inputs)
    recompute = _recompute_cost_score(max(0, inputs.context_tokens))
    remaining = _remaining_capacity_score(inputs)
    chain_state = (5 if inputs.rollback_applied else 0) - (5 if inputs.active_chain_count > 1 else 0)
    priority = max(0, min(100, continuation + recompute + remaining + chain_state))
    return PriorityDecision(
        priority=priority,
        continuation_score=continuation,
        recompute_cost_score=recompute,
        remaining_capacity_score=remaining,
        chain_state_score=chain_state,
    )
