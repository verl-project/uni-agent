"""OpenAI Chat Completions wire <-> InternalGenerationRequest.

Actor calls this before session; MessageCodec never sees wire shape after later
tasks.
"""

from __future__ import annotations

import json
from typing import Any

from uni_agent.gateway.session.codec import MalformedRequestError
from uni_agent.gateway.session.request import InternalGenerationRequest

OPENAI_ALLOWED_SAMPLING_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop"})


def _normalize_message_content(content: Any) -> Any:
    """Normalize message content: coerce None to empty string, validate type."""
    if isinstance(content, list | dict | str):
        return content
    if content is None:
        return ""
    raise MalformedRequestError(f"Unsupported content type: {type(content).__name__}")


def _normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """Validate tool_calls and parse JSON-string function arguments."""
    if not isinstance(tool_calls, list):
        raise MalformedRequestError("tool_calls must be a list")
    result = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise MalformedRequestError("tool_calls entries must be objects")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise MalformedRequestError("tool_call.function must be an object")

        normalized_tool_call = dict(tool_call)
        normalized_function = dict(function)
        arguments = normalized_function.get("arguments")
        if isinstance(arguments, str):
            try:
                normalized_function["arguments"] = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                pass
        normalized_tool_call["function"] = normalized_function
        result.append(normalized_tool_call)
    return result


def _normalize_message(message: Any) -> dict[str, Any]:
    """Normalize a single message and filter to known OpenAI chat fields."""
    if not isinstance(message, dict):
        raise MalformedRequestError("messages entries must be objects")

    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise MalformedRequestError("message.role must be a non-empty string")

    normalized: dict[str, Any] = {
        "role": role,
        "content": _normalize_message_content(message.get("content", "")),
    }
    if "name" in message:
        name = message["name"]
        if not isinstance(name, str):
            raise MalformedRequestError("message.name must be a string")
        normalized["name"] = name
    if "tool_calls" in message:
        normalized["tool_calls"] = _normalize_tool_calls(message["tool_calls"])
    if "tool_call_id" in message:
        normalized["tool_call_id"] = str(message["tool_call_id"])
    if "reasoning_content" in message:
        reasoning_content = message["reasoning_content"]
        if reasoning_content is not None and not isinstance(reasoning_content, str):
            raise MalformedRequestError("message.reasoning_content must be a string or null")
        normalized["reasoning_content"] = reasoning_content
    return normalized


def _validate_tools(tools: Any) -> list[Any] | None:
    """Validate tools structure. Does not modify content."""
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    return tools


def openai_to_internal(
    payload: dict,
    *,
    base_sampling_params: dict,
    allowed_sampling_keys: frozenset[str],
) -> InternalGenerationRequest:
    """Lower an OpenAI chat-completions request into the internal request shape."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise MalformedRequestError("messages must be non-empty")
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        raise MalformedRequestError("chat_template_kwargs must be an object")

    tool_choice_payload = payload.get("tool_choice")
    tool_choice = tool_choice_payload.lower() if isinstance(tool_choice_payload, str) else "auto"
    tools = _validate_tools(payload.get("tools"))
    if tool_choice == "none":
        tools = None

    sampling_params = dict(base_sampling_params)
    for key in allowed_sampling_keys:
        if key in payload:
            sampling_params[key] = payload[key]

    return {
        "messages": [_normalize_message(message) for message in messages],
        "tools": tools,
        "chat_template_kwargs": dict(chat_template_kwargs) if chat_template_kwargs else {},
        "sampling_params": sampling_params,
    }
