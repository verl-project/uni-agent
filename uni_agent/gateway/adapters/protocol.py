"""Lightweight adapter-side request typing.

These types document the Anthropic wire fragments this adapter lowers. They are
intentionally partial: validation remains in the adapter and the internal shape
stays OpenAI-like for chat templates.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class AnthropicContentBlock(TypedDict, total=False):
    type: str
    text: str
    source: dict[str, Any]
    id: str
    name: str
    input: dict[str, Any]
    tool_use_id: str
    content: str | list[dict[str, Any]]


class AnthropicMessage(TypedDict):
    role: str
    content: str | list[AnthropicContentBlock]


class AnthropicRequest(TypedDict):
    messages: list[AnthropicMessage]
    system: NotRequired[str | list[AnthropicContentBlock]]
    tools: NotRequired[list[dict[str, Any]]]
    tool_choice: NotRequired[str | dict[str, Any]]
