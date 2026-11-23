"""Response-side token ranges for committed Gateway generations."""

from dataclasses import dataclass, replace
from typing import Any

from uni_agent.gateway.session.types import AgentContext


@dataclass(frozen=True)
class ResponseSpan:
    response_start: int
    response_end: int
    tool_call_ids: tuple[str, ...]
    agent_context: AgentContext | None


def decoded_tool_call_ids(assistant_message: dict[str, Any]) -> tuple[str, ...]:
    """Read IDs from the decoded assistant message used by provider adapters."""
    return tuple(call["id"] for call in assistant_message.get("tool_calls") or () if call.get("id"))


def trim_response_spans(spans: list[ResponseSpan], response_length: int) -> None:
    """Apply a response tail trim while preserving the generation ordinals."""
    for index in range(len(spans) - 1, -1, -1):
        span = spans[index]
        if span.response_end <= response_length:
            break
        spans[index] = replace(
            span, response_start=min(span.response_start, response_length), response_end=response_length
        )


def export_generation_spans(spans: list[ResponseSpan], session_id: str, chain_id: int) -> list[dict[str, Any]]:
    """Materialize independent metadata for the surviving trajectory."""
    return [
        {
            "session_id": session_id,
            "chain_id": chain_id,
            "generation_index": index,
            "response_start": span.response_start,
            "response_end": span.response_end,
            "tool_call_ids": list(span.tool_call_ids),
            "agent_context": span.agent_context.copy() if span.agent_context is not None else {},
        }
        for index, span in enumerate(spans)
        if span.response_start < span.response_end or span.tool_call_ids
    ]
