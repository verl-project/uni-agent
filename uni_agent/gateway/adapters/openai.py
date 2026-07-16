"""OpenAI Chat Completions wire <-> InternalGenerationRequest.

Actor calls this before session; MessageCodec never sees wire shape.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from fastapi.responses import StreamingResponse

from uni_agent.gateway.session.session import GenerationOutcome
from uni_agent.gateway.session.types import InternalGenerationRequest

from .types import MalformedRequestError

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_OPENAI_ERROR_TYPE_BY_STATUS = {
    # Only status codes the gateway actually emits today (gateway.py /
    # session.py: 400 malformed/JSON, 409 concurrent session generation,
    # 500 internal). 401/403/404/422/429 are intentionally omitted: this
    # gateway has no auth / rate-limit / routing paths that surface them.
    400: "invalid_request_error",
    409: "conflict_error",
}


def openai_error_body(status_code: int, message: str) -> dict[str, Any]:
    error_type = _OPENAI_ERROR_TYPE_BY_STATUS.get(
        status_code,
        "invalid_request_error" if 400 <= status_code < 500 else "internal_server_error",
    )
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": None,
            "param": None,
        }
    }


def openai_build_response(outcome: GenerationOutcome, *, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": outcome.assistant_msg, "finish_reason": outcome.finish_reason}],
        "usage": {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
        },
    }


def openai_stream_response(outcome: GenerationOutcome, *, model: str) -> StreamingResponse:
    """Synthesize an OpenAI chat.completion.chunk SSE stream from a completed outcome."""
    chunk_id = f"chatcmpl-{uuid4().hex}"
    created = int(time.time())

    def _chunk(delta: dict[str, Any], finish: str | None, usage: dict[str, int] | None = None) -> str:
        body = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            body["usage"] = usage
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    async def _gen() -> AsyncIterator[bytes]:
        msg = outcome.assistant_msg
        yield _chunk({"role": "assistant"}, None).encode()
        if isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"]:
            yield _chunk({"reasoning_content": msg["reasoning_content"]}, None).encode()
        if isinstance(msg.get("content"), str) and msg["content"]:
            yield _chunk({"content": msg["content"]}, None).encode()
        if msg.get("tool_calls"):
            tool_calls = [{**tool_call, "index": idx} for idx, tool_call in enumerate(msg["tool_calls"])]
            yield _chunk({"tool_calls": tool_calls}, None).encode()
        usage = {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
        }
        yield _chunk({}, outcome.finish_reason, usage).encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


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


def openai_to_internal(
    payload: dict,
    *,
    base_sampling_params: dict,
    allowed_sampling_keys: frozenset[str],
) -> InternalGenerationRequest:
    """Lower an OpenAI chat-completions request into the internal request shape."""
    # Capability gates: these OpenAI request features have no gateway/session
    # equivalent yet, so reject them before building the internal request.
    n_value = payload.get("n", 1)
    if n_value != 1:
        raise MalformedRequestError(f"n={n_value} is not supported (only n=1)")
    if payload.get("response_format") is not None:
        raise MalformedRequestError("response_format is not supported")

    tool_choice_payload = payload.get("tool_choice", "auto")
    if isinstance(tool_choice_payload, str):
        tool_choice = tool_choice_payload.lower()
        if tool_choice not in {"auto", "none"}:
            raise MalformedRequestError(
                f'tool_choice="{tool_choice_payload}" is not supported (only "auto" / "none" are supported)'
            )
    elif isinstance(tool_choice_payload, dict):
        raise MalformedRequestError(
            'tool_choice with a specific function is not supported (only "auto" / "none" are supported)'
        )
    else:
        raise MalformedRequestError("tool_choice must be a string or object")

    # Required payload fields.
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise MalformedRequestError("messages must be non-empty")

    # Tool injection policy.
    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    if tool_choice == "none":
        tools = None

    # Sampling params are gateway-owned allowlist merges, not message canonicalization.
    sampling_params = dict(base_sampling_params)
    for key in allowed_sampling_keys:
        if key in payload:
            sampling_params[key] = payload[key]

    return {
        "messages": [_normalize_message(message) for message in messages],
        "tools": tools,
        "sampling_params": sampling_params,
    }
