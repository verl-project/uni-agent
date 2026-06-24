"""Lightweight adapter-side wire protocol typing.

These types document the provider wire fragments the adapters lower. They are
intentionally partial: validation remains in each adapter and the internal
shape stays OpenAI-like for chat templates.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class OpenAIChatCompletionFunction(TypedDict, total=False):
    """``tool_calls[i].function`` object inside an OpenAI chat message."""

    name: str
    arguments: Any  # OpenAI spec is JSON string; gateway also accepts dict (Qwen-style chat templates)


class OpenAIChatCompletionToolCall(TypedDict, total=False):
    """One entry in an OpenAI assistant message's ``tool_calls`` array."""

    id: str
    type: Literal["function"]
    function: OpenAIChatCompletionFunction


class OpenAIChatMessage(TypedDict, total=False):
    """A single OpenAI chat-completion message.

    Includes OpenAI-compatible extension ``reasoning_content`` that the gateway
    preserves on input.
    """

    role: str
    content: Any
    name: str
    tool_calls: list[OpenAIChatCompletionToolCall]
    tool_call_id: str
    reasoning_content: str | None


class OpenAIChatCompletionTool(TypedDict, total=False):
    """One entry in an OpenAI request ``tools`` array."""

    type: Literal["function"]
    function: dict[str, Any]


class OpenAIChatCompletionRequest(TypedDict, total=False):
    """Incoming ``POST /v1/chat/completions`` request body shape.

    ``chat_template_kwargs`` is an OpenAI-compatible server extension used by
    the gateway to forward per-request chat template overrides (e.g.
    ``enable_thinking``) into ``MessageCodec``.
    """

    model: str
    messages: list[OpenAIChatMessage]
    tools: list[OpenAIChatCompletionTool]
    tool_choice: Any
    stream: bool
    n: int
    response_format: Any
    chat_template_kwargs: dict[str, Any]
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop: str | list[str]


class OpenAIChatCompletionUsage(TypedDict):
    """Token usage block inside an OpenAI response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatCompletionChoice(TypedDict):
    """One entry in an OpenAI response ``choices`` array.

    ``finish_reason`` is the OpenAI-spec value mapped from the backend's raw
    stop reason by the OpenAI adapter.
    """

    index: int
    message: OpenAIChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class OpenAIChatCompletionResponse(TypedDict):
    """Outgoing ``POST /v1/chat/completions`` response body shape."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAIChatCompletionUsage


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
    model: NotRequired[str]
    messages: list[AnthropicMessage]
    system: NotRequired[str | list[AnthropicContentBlock]]
    tools: NotRequired[list[dict[str, Any]]]
    tool_choice: NotRequired[str | dict[str, Any]]
    stream: NotRequired[bool]
    max_tokens: NotRequired[int]
    stop_sequences: NotRequired[list[str]]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    top_k: NotRequired[int]
