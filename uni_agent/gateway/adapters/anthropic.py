"""Anthropic Messages wire -> InternalGenerationRequest."""

from __future__ import annotations

import re
from typing import Any

from uni_agent.gateway.session.codec import MalformedRequestError
from uni_agent.gateway.session.request import InternalGenerationRequest

_BILLING_HEADER_RE = re.compile(r"^\s*x-anthropic-billing-header:[^\n]*\n?", re.IGNORECASE | re.MULTILINE)
_SAME_NAME_SAMPLING_KEYS = ("max_tokens", "temperature", "top_p", "top_k")


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


def _text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _content_parts_or_text(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if any(part.get("type") != "text" for part in parts):
        return list(parts)
    return "".join(part["text"] for part in parts)


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
        if texts:
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": "".join(texts)})
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
            if not texts and not emitted_tool_message:
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": ""})
                emitted_tool_message = True
            flush_text()
            # OpenAI/vLLM tool-role messages cannot carry images, so preserve
            # visual payloads as a following user message downgrade.
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
    parts: list[dict[str, Any]] = []

    def flush_user() -> None:
        if parts:
            messages.append({"role": "user", "content": _content_parts_or_text(parts)})
            parts.clear()

    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("user content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if not isinstance(text, str):
                raise MalformedRequestError("user text must be a string")
            parts.append(_text_part(text))
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
        elif block_type in {"thinking", "redacted_thinking"}:
            # Full Anthropic thinking signature semantics have no faithful chat
            # template representation here, so these blocks are skipped.
            continue
        else:
            raise MalformedRequestError(f"Unsupported assistant block type: {block_type}")

    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _messages_to_internal(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise MalformedRequestError("messages must be non-empty")

    result: list[dict[str, Any]] = []
    pending_reminder = ""

    def fold_reminder_into_user(message: dict[str, Any], text: str, *, prepend: bool = False) -> None:
        content = message.get("content", "")
        if isinstance(content, str):
            if not content:
                message["content"] = text
            elif prepend:
                message["content"] = f"{text}\n{content}"
            else:
                message["content"] = f"{content}\n{text}"
        elif isinstance(content, list):
            part = _text_part(text)
            if prepend:
                content.insert(0, part)
            else:
                content.append(part)
        else:
            message["content"] = text

    for message in messages:
        if not isinstance(message, dict):
            raise MalformedRequestError("messages entries must be objects")
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            user_messages = _user_messages_from_content(content)
            if pending_reminder:
                folded = False
                for user_message in user_messages:
                    if user_message.get("role") == "user":
                        fold_reminder_into_user(user_message, pending_reminder, prepend=True)
                        folded = True
                        break
                if folded:
                    pending_reminder = ""
                else:
                    # A tool_result-only Anthropic user message converts to tool
                    # role messages; emit the reminder before it instead of
                    # letting system text leap over the tool result.
                    result.append({"role": "user", "content": pending_reminder})
                    pending_reminder = ""
            result.extend(user_messages)
        elif role == "assistant":
            result.append(_assistant_message_from_content(content))
        elif role == "system":
            # Many chat templates reject system messages beyond index 0. Folding
            # mid-list system text as a Slime-style reminder preserves position
            # better than hoisting it to the top.
            system_text = _system_to_text(content)
            reminder = f"<system-reminder>\n{system_text}\n</system-reminder>" if system_text else ""
            if not reminder:
                continue
            folded = False
            for previous_message in reversed(result):
                if previous_message.get("role") == "user":
                    fold_reminder_into_user(previous_message, reminder)
                    folded = True
                    break
            if not folded:
                pending_reminder = f"{pending_reminder}\n{reminder}" if pending_reminder else reminder
        else:
            raise MalformedRequestError(f"Unsupported message role: {role}")
    if pending_reminder:
        result.append({"role": "user", "content": pending_reminder})
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
        tool_type = tool.get("type")
        if tool_type not in (None, "custom"):
            raise MalformedRequestError(f"Unsupported Anthropic tool type: {tool_type}")
        name = tool.get("name")
        if not isinstance(name, str):
            raise MalformedRequestError("tool.name must be a string")
        function: dict[str, Any] = {"name": name, "parameters": tool.get("input_schema", {})}
        description = tool.get("description")
        if isinstance(description, str):
            function["description"] = description
        # cache_control is request/cache metadata; internal tools keep only the
        # OpenAI function schema and pass input_schema through unchanged.
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


def _sampling_params(
    payload: dict[str, Any],
    *,
    base_sampling_params: dict[str, Any],
    allowed_sampling_keys: frozenset[str],
) -> dict[str, Any]:
    sampling_params = dict(base_sampling_params)
    for key in _SAME_NAME_SAMPLING_KEYS:
        if key in allowed_sampling_keys and key in payload:
            sampling_params[key] = payload[key]
    if "stop" in allowed_sampling_keys and "stop_sequences" in payload:
        sampling_params["stop"] = payload["stop_sequences"]
    # Anthropic cache_control can appear on request blocks; it is intentionally
    # ignored because generation sampling params have no equivalent field.
    return sampling_params


def anthropic_to_internal(
    payload: dict,
    *,
    base_sampling_params: dict,
    allowed_sampling_keys: frozenset[str],
) -> InternalGenerationRequest:
    """Lower an Anthropic Messages request into the internal request shape."""
    if not isinstance(payload, dict):
        raise MalformedRequestError("payload must be an object")

    messages = _messages_to_internal(payload.get("messages"))
    system_text = _system_to_text(payload.get("system"))
    if system_text:
        messages.insert(0, {"role": "system", "content": system_text})

    tools = _apply_tool_choice(payload, _convert_tools(payload.get("tools")))
    return {
        "messages": messages,
        "tools": tools,
        "chat_template_kwargs": {},
        "sampling_params": _sampling_params(
            payload,
            base_sampling_params=base_sampling_params,
            allowed_sampling_keys=allowed_sampling_keys,
        ),
    }
