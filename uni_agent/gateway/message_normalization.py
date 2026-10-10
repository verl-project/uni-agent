"""Canonicalize message fragments shared by Gateway wire adapters and codecs."""

from __future__ import annotations

from typing import Any


def _merge_content(previous: Any, current: Any) -> Any:
    if not current:
        return previous
    if not previous:
        return current
    if isinstance(previous, str) and isinstance(current, str):
        return f"{previous}\n{current}"
    if isinstance(previous, list) and isinstance(current, list):
        return [*previous, {"type": "text", "text": "\n"}, *current]

    def as_parts(content: Any) -> list[Any]:
        if isinstance(content, list):
            return list(content)
        return [{"type": "text", "text": str(content)}]

    return [*as_parts(previous), {"type": "text", "text": "\n"}, *as_parts(current)]


def coalesce_consecutive_assistant_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge Responses reasoning/function items into one assistant message.

    The Responses protocol may represent one assistant turn as separate
    ``reasoning`` and ``function_call`` items. The internal Chat Completions
    shape needs one assistant message carrying ``reasoning_content`` and
    ``tool_calls`` so session prefix matching and incremental encoding see the
    same history on every continuation.
    """
    result: list[dict[str, Any]] = []
    for message in messages:
        if not result or result[-1].get("role") != "assistant" or message.get("role") != "assistant":
            copied = dict(message)
            if isinstance(message.get("tool_calls"), list):
                copied["tool_calls"] = list(message["tool_calls"])
            result.append(copied)
            continue

        previous = result[-1]
        previous["content"] = _merge_content(previous.get("content", ""), message.get("content", ""))

        current_reasoning = message.get("reasoning_content") or ""
        if current_reasoning:
            previous_reasoning = previous.get("reasoning_content") or ""
            previous["reasoning_content"] = (
                f"{previous_reasoning}\n{current_reasoning}" if previous_reasoning else current_reasoning
            )

        current_tool_calls = message.get("tool_calls") or []
        if current_tool_calls:
            previous["tool_calls"] = [*previous.get("tool_calls", []), *current_tool_calls]

        if "name" not in previous and "name" in message:
            previous["name"] = message["name"]
    return result


def canonicalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize system/developer placement and Responses assistant fragments."""
    coalesced = coalesce_consecutive_assistant_messages(messages)
    system_messages = [message for message in coalesced if message.get("role") == "system"]
    if not system_messages:
        return coalesced

    first_system = dict(system_messages[0])
    for message in system_messages[1:]:
        first_system["content"] = _merge_content(first_system.get("content", ""), message.get("content", ""))
    if len(system_messages) == 1 and coalesced and coalesced[0].get("role") == "system":
        return coalesced
    return [first_system] + [message for message in coalesced if message.get("role") != "system"]
