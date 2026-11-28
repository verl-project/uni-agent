"""Backend-neutral, versioned snapshot of agent state before generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentRuntimeHint:
    tools_available: bool
    has_active_chain: bool
    received_tool_result: bool
    context_tokens: int
    trajectory_capacity: int | None
    remaining_capacity: int | None
    rollback_applied: bool
    active_chain_count: int
    trajectory_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 2, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Any) -> AgentRuntimeHint | None:
        """Reject incomplete, legacy or malformed snapshots as absent hints."""
        if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 2:
            return None
        boolean_fields = ("tools_available", "has_active_chain", "received_tool_result", "rollback_applied")
        if any(type(raw.get(name)) is not bool for name in boolean_fields):
            return None
        if any(type(raw.get(name)) is not int or raw[name] < 0 for name in ("context_tokens", "active_chain_count")):
            return None
        for name in ("trajectory_capacity", "remaining_capacity"):
            if name not in raw:
                return None
            value = raw[name]
            if value is not None and (type(value) is not int or value < (1 if name == "trajectory_capacity" else 0)):
                return None
        capacity, remaining = raw["trajectory_capacity"], raw["remaining_capacity"]
        if (capacity is None) != (remaining is None) or (capacity is not None and remaining > capacity):
            return None
        if not isinstance(raw.get("trajectory_id"), str):
            return None
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})
