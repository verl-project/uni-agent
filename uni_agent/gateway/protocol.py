"""OpenAI-compatible Chat Completions HTTP protocol types for the gateway.

These TypedDicts give the actor's HTTP layer a single source of truth
for request / response shape so that ``_handle_chat_completions`` no
longer constructs anonymous dicts. They also serve as documentation for
the expected shape of these objects.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class ChatCompletionFunction(TypedDict, total=False):
    """``tool_calls[i].function`` object inside a ChatMessage."""

    name: str
    arguments: Any  # OpenAI spec is JSON string; gateway also accepts dict (Qwen-style chat templates)


class ChatCompletionToolCall(TypedDict, total=False):
    """One entry in an assistant message's ``tool_calls`` array."""

    id: str
    type: Literal["function"]
    function: ChatCompletionFunction


class ChatMessage(TypedDict, total=False):
    """A single chat-completion message (system / user / assistant / tool).

    Includes OpenAI-compatible extension ``reasoning_content`` that the
    gateway preserves on input.
    """

    role: str
    content: Any
    name: str
    tool_calls: list[ChatCompletionToolCall]
    tool_call_id: str
    reasoning_content: str | None


class ChatCompletionTool(TypedDict, total=False):
    """One entry in the request ``tools`` array."""

    type: Literal["function"]
    function: dict[str, Any]


class ChatCompletionRequest(TypedDict, total=False):
    """Incoming ``POST /v1/chat/completions`` request body shape.

    ``chat_template_kwargs`` is an OpenAI-compatible server extension
    used by the gateway to forward per-request chat template overrides
    (e.g. ``enable_thinking``) into ``MessageCodec``.
    """

    model: str
    messages: list[ChatMessage]
    tools: list[ChatCompletionTool]
    tool_choice: Any
    stream: bool
    n: int
    response_format: Any
    chat_template_kwargs: dict[str, Any]
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int


class ChatCompletionUsage(TypedDict):
    """Token usage block inside the response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(TypedDict):
    """One entry in the response ``choices`` array.

    ``finish_reason`` is the OpenAI-spec value mapped from the backend's
    raw stop reason by ``MessageCodec.decode_response``.
    """

    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class ChatCompletionResponse(TypedDict):
    """Outgoing ``POST /v1/chat/completions`` response body shape."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage
