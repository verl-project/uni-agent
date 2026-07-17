"""Anthropic Messages wire -> InternalGenerationRequest."""

from __future__ import annotations

import copy
import json
import logging
import re
import secrets
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from uni_agent.gateway.session.session import GenerationOutcome
from uni_agent.gateway.session.types import InternalGenerationRequest

from .types import MalformedRequestError

_BILLING_HEADER_RE = re.compile(r"^\s*x-anthropic-billing-header:[^\n]*\n?", re.IGNORECASE | re.MULTILINE)
# content_filter/function_call are defensive: current internal finish_reason
# values are stop/length/tool_calls.
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}

logger = logging.getLogger("gateway")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# Only status codes the gateway actually emits today (gateway.py / session.py:
# 400 malformed/JSON, 409 concurrent session generation, 500 internal). Other
# Anthropic types (authentication_error / permission_error / not_found_error /
# request_too_large / rate_limit_error / overloaded_error) are intentionally
# omitted: this gateway is not an Anthropic proxy and has no auth / rate-limit
# / capacity paths that would surface those codes. Anything not listed falls
# back to "api_error" via .get() default.
ANTHROPIC_ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    409: "invalid_request_error",
    500: "api_error",
}


def anthropic_error_body(status_code: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": ANTHROPIC_ERROR_TYPE_BY_STATUS.get(status_code, "api_error"),
            "message": message,
        },
    }


def _tool_call_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            logger.warning("Invalid Anthropic tool call JSON arguments; using empty input")
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.warning("Anthropic tool call JSON arguments were not an object; using empty input")
        return {}
    logger.warning("Anthropic tool call arguments were not an object or JSON string; using empty input")
    return {}


def _outcome_to_blocks(assistant_msg: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    # Outbound thinking/reasoning is intentionally not synthesized here:
    # `assistant_msg` from session/decode currently has no `reasoning_content`
    # field, and adding outbound wrapping before there is a producer would be
    # defensive code at a trusted boundary (AGENTS §4 rule 3). When session
    # decode grows reasoning_content, mirror OpenAI streaming and emit Anthropic
    # thinking blocks at that point.
    content = assistant_msg.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})

    tool_calls = assistant_msg.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{secrets.token_hex(8)}",
                    "name": function.get("name", "tool"),
                    "input": _tool_call_input(function.get("arguments")),
                }
            )

    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def anthropic_build_response(outcome: GenerationOutcome, *, model: str) -> dict[str, Any]:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": _outcome_to_blocks(outcome.assistant_msg),
        "stop_reason": _STOP_REASON_MAP.get(outcome.finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": outcome.prompt_tokens, "output_tokens": outcome.completion_tokens},
    }


def anthropic_stream_response(outcome: GenerationOutcome, *, model: str) -> StreamingResponse:
    """Synthesize an Anthropic Messages SSE stream from a completed outcome.

    Whole-turn synthesis (backend is not token-streaming): message_start,
    per-block start/delta/stop, message_delta with stop_reason+usage,
    message_stop.
    """
    blocks = _outcome_to_blocks(outcome.assistant_msg)
    stop_reason = _STOP_REASON_MAP.get(outcome.finish_reason, "end_turn")
    msg_id = f"msg_{secrets.token_hex(12)}"

    def _event(name: str, data: dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

    async def _gen() -> AsyncIterator[bytes]:
        yield _event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": outcome.prompt_tokens, "output_tokens": 0},
                },
            },
        )
        for idx, block in enumerate(blocks):
            if block["type"] == "text":
                start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block["text"]}
            else:
                start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)}
            yield _event("content_block_start", {"type": "content_block_start", "index": idx, "content_block": start})
            yield _event("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": delta})
            yield _event("content_block_stop", {"type": "content_block_stop", "index": idx})
        yield _event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": outcome.prompt_tokens, "output_tokens": outcome.completion_tokens},
            },
        )
        yield _event("message_stop", {"type": "message_stop"})

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


def _strip_billing_header(text: str) -> str:
    # Claude Code can send this request-scoped billing header in system text; if
    # kept, it changes prompt prefixes and causes cache/template prefix drift.
    return _BILLING_HEADER_RE.sub("", text).strip()


def _system_to_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return _strip_billing_header(system)
    if not isinstance(system, list):
        raise MalformedRequestError("system must be a string or text block list")

    parts: list[str] = []
    for block in system:
        if not isinstance(block, dict):
            raise MalformedRequestError("system blocks must be objects")
        if block.get("type") == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise MalformedRequestError("system text must be a string")
            parts.append(text)
        else:
            raise MalformedRequestError(f"Unsupported system block type: {block.get('type')}")
    return _strip_billing_header("\n".join(part for part in parts if part))


def _image_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise MalformedRequestError("image.source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise MalformedRequestError("base64 image source requires media_type and data")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        url = source.get("url")
        if not isinstance(url, str):
            raise MalformedRequestError("url image source requires url")
    else:
        raise MalformedRequestError(f"Unsupported image source type: {source_type}")
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_messages(tool_call_id: str, content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "tool", "tool_call_id": tool_call_id, "content": content}]
    if content is None:
        return [{"role": "tool", "tool_call_id": tool_call_id, "content": ""}]
    if not isinstance(content, list):
        raise MalformedRequestError("tool_result.content must be a string or block list")

    messages: list[dict[str, Any]] = []
    texts: list[str] = []
    emitted_tool_message = False

    def flush_text() -> None:
        nonlocal emitted_tool_message
        if not texts:
            return
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": "\n".join(texts)})
        emitted_tool_message = True
        texts.clear()

    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("tool_result content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise MalformedRequestError("tool_result text must be a string")
            texts.append(text)
        elif block_type == "image":
            # OpenAI/vLLM tool-role messages cannot carry images, so preserve
            # visual payloads as a following user message downgrade. The
            # assistant's tool_calls must be closed by at least one tool
            # message before any user follows; emit an empty tool sentinel if
            # neither buffered text nor a prior tool message has done that.
            if not texts and not emitted_tool_message:
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": ""})
                emitted_tool_message = True
            flush_text()
            messages.append({"role": "user", "content": [_image_block_to_openai_part(block)]})
        else:
            raise MalformedRequestError(f"Unsupported tool_result block type: {block_type}")
    flush_text()
    if not messages:
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": ""})
    return messages


def _user_messages_from_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise MalformedRequestError("user content must be a string or block list")

    messages: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []  # A single openai message can carry multiple text/image parts.

    def flush_user() -> None:
        # Collapse pure-text parts to a single string so downstream chat
        # templates take their well-trodden string path; keep list form only
        # when an image is present (image_url has no string fallback in the
        # OpenAI protocol). Claude Code splits content into multiple text
        # blocks to attach cache_control per block, so join with \n to keep
        # those blocks from running together.
        if not parts:
            return
        if any(part.get("type") != "text" for part in parts):
            content = list(parts)
        else:
            content = "\n".join(part["text"] for part in parts)
        messages.append({"role": "user", "content": content})
        parts.clear()

    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("user content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise MalformedRequestError("user text must be a string")
            parts.append({"type": "text", "text": text})
        elif block_type == "image":
            parts.append(_image_block_to_openai_part(block))
        elif block_type == "tool_result":
            flush_user()
            tool_call_id = block.get("tool_use_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise MalformedRequestError("tool_result.tool_use_id must be a non-empty string")
            messages.extend(_tool_result_messages(tool_call_id, block.get("content", "")))
        else:
            raise MalformedRequestError(f"Unsupported user block type: {block_type}")
    flush_user()
    return messages


def _assistant_message_from_content(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise MalformedRequestError("assistant content must be a string or block list")

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("assistant content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise MalformedRequestError("assistant text must be a string")
            texts.append(text)
        elif block_type == "tool_use":
            tool_id = block.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                raise MalformedRequestError("tool_use.id must be a non-empty string")
            name = block.get("name")
            if not isinstance(name, str):
                raise MalformedRequestError("tool_use.name must be a string")
            arguments = block.get("input", {})
            if not isinstance(arguments, dict):
                raise MalformedRequestError("tool_use.input must be an object")
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        elif block_type == "thinking":
            # Self-hosted SGLang never emits Anthropic thinking blocks, so seeing
            # one in inbound history means the client mixed sources. Drop with a
            # warning rather than crash; <think>...</think> tokens, if present,
            # are already preserved as plain text on the same assistant message.
            logger.warning("Dropping Anthropic thinking block from assistant history; multi-turn prefix may drift")
            continue
        elif block_type == "redacted_thinking":
            # Anthropic-encrypted reasoning signature with byte-level fidelity;
            # no faithful representation in our chat-template path, and silently
            # mapping it to plain text would corrupt training distribution.
            raise MalformedRequestError("redacted_thinking blocks are not supported")
        else:
            raise MalformedRequestError(f"Unsupported assistant block type: {block_type}")

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _fold_reminder_into_user(message: dict[str, Any], text: str, *, prepend: bool = False) -> None:
    """Fold a system-text reminder into an existing user message in place.

    Content is str | list of Anthropic blocks; both forms are handled. Mid-list
    `<system-reminder>` text is anchored to a neighbouring user message rather
    than hoisted to index 0, since many chat templates reject system messages
    beyond index 0. `text` is the fully-wrapped reminder string — the caller is
    responsible for adding the `<system-reminder>...</system-reminder>` envelope
    before passing it in.
    """
    content = message.get("content", "")
    if isinstance(content, list):
        part = {"type": "text", "text": text}
        if prepend:
            content.insert(0, part)
        else:
            content.append(part)
        return
    if not content:
        message["content"] = text
    elif prepend:
        message["content"] = f"{text}\n{content}"
    else:
        message["content"] = f"{content}\n{text}"


def _fold_mid_list_system_into_user(messages: Any) -> Any:
    """Fold every `messages[]`-level system entry into a neighbouring user
    message as a `<system-reminder>` block, returning a system-free list.

    Anthropic carries the real system prompt in the top-level `system` field; a
    `role: system` entry inside `messages` is a mid-conversation reminder that
    clients like Claude Code inject. Many chat templates reject any system
    message past index 0 *and* reject consecutive same-role messages, so a
    reminder must be merged into an adjacent user message rather than emitted
    standalone. We fold index-0 too: keeping it as a leading system would be
    pushed to index 1 by the top-level system prompt, which is exactly what
    templates reject. Non-list input is passed through for the caller to reject.

    Strategy: single forward pass, preferring the preceding user (append). When
    no preceding user exists yet, buffer the reminder and prepend it into the
    next user; if neither exists, emit a standalone user at the end (last
    resort: violates same-role-adjacency only when the whole conversation has
    no user message, which is itself ill-formed).
    """
    if not isinstance(messages, list):
        return messages
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        return messages

    out: list[Any] = []
    pending = ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            system_text = _system_to_text(message.get("content", ""))
            if not system_text:
                continue
            reminder = f"<system-reminder>\n{system_text}\n</system-reminder>"
            prev_user = next(
                (m for m in reversed(out) if isinstance(m, dict) and m.get("role") == "user"),
                None,
            )
            if prev_user is not None:
                _fold_reminder_into_user(prev_user, reminder)
            else:
                pending = f"{pending}\n{reminder}" if pending else reminder
            continue
        if pending and isinstance(message, dict) and message.get("role") == "user":
            _fold_reminder_into_user(message, pending, prepend=True)
            pending = ""
        out.append(message)
    if pending:
        out.append({"role": "user", "content": pending})
    return out


def _messages_to_internal(messages: Any) -> list[dict[str, Any]]:
    # Caller folds mid-list system into user first, so only user/assistant
    # remain here (a stray system role is therefore unsupported).
    if not isinstance(messages, list) or not messages:
        raise MalformedRequestError("messages must be non-empty")

    result: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise MalformedRequestError("messages entries must be objects")
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            result.extend(_user_messages_from_content(content))
        elif role == "assistant":
            result.append(_assistant_message_from_content(content))
        else:
            raise MalformedRequestError(f"Unsupported message role: {role}")
    return result


def _convert_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise MalformedRequestError("tools entries must be objects")
        name = tool.get("name")
        if not isinstance(name, str):
            raise MalformedRequestError("tool.name must be a string")
        input_schema = copy.deepcopy(tool.get("input_schema", {}))
        function: dict[str, Any] = {"name": name, "parameters": input_schema}
        description = tool.get("description")
        if isinstance(description, str):
            function["description"] = description
        # cache_control is request/cache metadata; internal tools keep only the
        # OpenAI function schema.
        converted.append({"type": "function", "function": function})
    return converted


def _apply_tool_choice(payload: dict[str, Any], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    tool_choice = payload.get("tool_choice", "auto")
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return tools
        if tool_choice == "none":
            return None
        raise MalformedRequestError(f"Unsupported tool_choice: {tool_choice}")
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return tools
        if choice_type == "none":
            return None
        raise MalformedRequestError(f"Unsupported tool_choice type: {choice_type}")
    raise MalformedRequestError("tool_choice must be a string or object")


def anthropic_to_internal(
    payload: dict,
    *,
    base_sampling_params: dict,
    allowed_sampling_keys: frozenset[str],
) -> InternalGenerationRequest:
    """Lower an Anthropic Messages request into the internal request shape."""
    if not isinstance(payload, dict):
        raise MalformedRequestError("payload must be an object")

    # Message/system lowering handles Anthropic-specific compatibility downgrades
    # before the session sees the OpenAI-like template-facing canonical.
    messages = _messages_to_internal(_fold_mid_list_system_into_user(payload.get("messages")))
    system_text = _system_to_text(payload.get("system"))
    if system_text:
        messages.insert(0, {"role": "system", "content": system_text})

    # Tool conversion and sampling stay adapter-owned; the session consumes only
    # internal tools plus codec-keyed sampling params.
    tools = _apply_tool_choice(payload, _convert_tools(payload.get("tools")))
    sampling_params = dict(base_sampling_params)
    for key in ("max_tokens", "temperature", "top_p", "top_k"):
        if key in allowed_sampling_keys and key in payload:
            sampling_params[key] = payload[key]
    if "stop" in allowed_sampling_keys and "stop_sequences" in payload:
        sampling_params["stop"] = payload["stop_sequences"]
    # Anthropic cache_control can appear on request blocks; it is intentionally
    # ignored because generation sampling params have no equivalent field.
    return {
        "messages": messages,
        "tools": tools,
        "sampling_params": sampling_params,
    }
