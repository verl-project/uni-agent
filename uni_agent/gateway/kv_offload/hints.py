"""Connector-owned configuration and scoring for agent-aware KV offload.

Only information available before backend generation is used.  Keeping this
module free of vLLM imports makes the scoring rule cheap to unit test in
CPU-only environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from uni_agent.gateway.agent_hint import AgentRuntimeHint

MIN_USEFUL_REMAINING_TOKENS = 1024


@dataclass(frozen=True)
class AgentHintConfig:
    """Cache policy settings from kv_connector_extra_config.agent_hint_config."""

    active_lease_seconds: float = 300.0
    min_offload_priority: int = 0
    admit_without_hint: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.active_lease_seconds, bool)
            or not isinstance(self.active_lease_seconds, int | float)
            or not math.isfinite(self.active_lease_seconds)
            or self.active_lease_seconds <= 0
        ):
            raise ValueError("active_lease_seconds must be finite and positive")
        if type(self.min_offload_priority) is not int or not 0 <= self.min_offload_priority <= 100:
            raise ValueError("min_offload_priority must be an integer between 0 and 100")
        if type(self.admit_without_hint) is not bool:
            raise ValueError("admit_without_hint must be a bool")

    @classmethod
    def from_dict(cls, raw) -> AgentHintConfig:
        if not isinstance(raw, dict):
            raise ValueError("agent_hint_config must be an object")
        unknown = set(raw) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unsupported agent_hint_config options: {sorted(unknown)}; scoring is always dynamic")
        return cls(**raw)


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


def _continuation_score(inputs: AgentRuntimeHint) -> int:
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


def _remaining_capacity_score(inputs: AgentRuntimeHint) -> int:
    capacity = inputs.trajectory_capacity
    remaining = inputs.remaining_capacity
    if capacity is None or remaining is None or capacity <= 0:
        return 0
    if remaining < MIN_USEFUL_REMAINING_TOKENS or remaining / capacity < 0.05:
        return -15
    if remaining >= 8192 and remaining / capacity >= 0.25:
        return 10
    return 0


def compute_dynamic_priority(inputs: AgentRuntimeHint) -> PriorityDecision:
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
