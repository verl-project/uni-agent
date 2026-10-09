"""OpenAI Responses wire adapter for per-session gateway endpoints.

The gateway session itself uses a compact Chat Completions-shaped request. This
module owns Responses-specific input items, custom-tool preservation, output
envelopes, errors, and SSE streaming. It does not own tokenization or session
state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi.responses import StreamingResponse

from uni_agent.gateway.message_normalization import canonicalize_messages
from uni_agent.gateway.session.session import GenerationOutcome
from uni_agent.gateway.session.types import InternalGenerationRequest

from .openai import openai_to_internal
from .types import MalformedRequestError

logger = logging.getLogger("gateway")

_RESPONSES_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
_TEXT_BLOCK_TYPES = {"input_text", "output_text", "reasoning_text", "summary_text", "text"}
_NAMESPACE_SEPARATOR = "__"
_DEFAULT_NAMESPACE = "functions"
_SKIPPED_ITEM_TYPES = {"additional_tools", "compaction_trigger", "context_compaction"}
_HOSTED_TOOL_TYPES = {
    "apply_patch",
    "computer",
    "computer_use_preview",
    "image_generation",
    "shell",
    "tool_search",
    "web_search",
    "web_search_preview",
}
_IGNORED_TOOL_TYPES = {"code_interpreter", "file_search", "local_shell"}


def responses_error_body(status_code: int, message: str, *, param: str | None = None) -> dict[str, Any]:
    """Return the error envelope expected by Responses clients."""
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error" if 400 <= status_code < 500 else "internal_server_error",
            "code": None,
            "param": param,
        }
    }


def _content_to_text(content: Any, *, param: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_to_text(item, param=param) for item in content)
    if isinstance(content, dict):
        if "output" in content:
            return _content_to_text(content["output"], param=param)
        block_type = content.get("type")
        if block_type in _TEXT_BLOCK_TYPES or block_type is None:
            return _content_to_text(content.get("text", content.get("content", "")), param=param)
        if block_type == "encrypted_content":
            value = content.get("encrypted_content", "")
            return value if isinstance(value, str) else str(value)
        if block_type == "input_image":
            return "[Image omitted]"
        salvaged = content.get("text", content.get("content", content.get("refusal")))
        if salvaged is not None:
            return _content_to_text(salvaged, param=param)
        raise MalformedRequestError(f"Unsupported Responses content block {block_type!r} at {param}")
    if isinstance(content, int | float | bool):
        return str(content)
    raise MalformedRequestError(f"Unsupported content value at {param}: {type(content).__name__}")


def _content_to_chat_content(content: Any, *, param: str) -> Any:
    """Lower Responses text/image blocks to the canonical chat content shape."""
    blocks = content if isinstance(content, list) else [content]
    parts: list[dict[str, Any]] = []
    has_image = False
    for index, block in enumerate(blocks):
        block_param = f"{param}[{index}]"
        if isinstance(block, dict) and block.get("type") == "input_image":
            image_url = block.get("image_url", block.get("url"))
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not image_url:
                data = block.get("data") or block.get("image_data")
                if isinstance(data, str) and data:
                    mime = block.get("mime_type") or block.get("media_type") or "image/png"
                    image_url = data if data.startswith("data:") else f"data:{mime};base64,{data}"
            if isinstance(image_url, str) and image_url:
                parts.append({"type": "image_url", "image_url": {"url": image_url}})
                has_image = True
                continue
            logger.warning("Responses input_image has no usable URL at %s", block_param)
        text = _content_to_text(block, param=block_param)
        if text:
            parts.append({"type": "text", "text": text})
    return parts if has_image else _content_to_text(content, param=param)


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False)


def _custom_arguments(value: Any) -> dict[str, str]:
    """Keep arbitrary custom input in a parser-compatible envelope."""
    if isinstance(value, str):
        return {"input": value}
    if isinstance(value, dict) and isinstance(value.get("input"), str):
        return {"input": value["input"]}
    return {"input": json.dumps(value, ensure_ascii=False)}


def _custom_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
        "additionalProperties": False,
    }


def _clean_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    cleaned = dict(schema)
    for key in ("$schema", "title"):
        cleaned.pop(key, None)
    if cleaned.get("required") in (None, []):
        cleaned.pop("required", None)
    if cleaned.get("additionalProperties") in (None, {}):
        cleaned.pop("additionalProperties", None)
    return cleaned


def _declared_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = source.get("name") or tool.get("type")
    return str(name) if name else None


def _flatten_tool(
    tool: dict[str, Any],
    *,
    namespace: str | None = None,
    drop_deferred: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    """Build the parser-facing function view and remember original tool kinds."""
    tool_type = tool.get("type")
    if tool_type == "namespace":
        name = tool.get("name")
        nested_tools = tool.get("tools")
        if nested_tools is None:
            logger.warning("Skipping empty Responses namespace tool group %r", name)
            return [], {}, set()
        if not isinstance(name, str) or not name.strip():
            raise MalformedRequestError("Namespace tools require a non-empty name")
        if not isinstance(nested_tools, list):
            raise MalformedRequestError("Namespace tools require a tools list")
        container = name.strip()
        next_namespace = (
            namespace
            if container in ("", _DEFAULT_NAMESPACE)
            else (f"{namespace}{_NAMESPACE_SEPARATOR}{container}" if namespace else container)
        )
        converted: list[dict[str, Any]] = []
        kinds: dict[str, str] = {}
        namespaced: set[str] = set()
        for nested in nested_tools:
            if not isinstance(nested, dict):
                raise MalformedRequestError("Each namespace tool must be an object")
            child, child_kinds, child_names = _flatten_tool(
                nested, namespace=next_namespace, drop_deferred=drop_deferred
            )
            converted.extend(child)
            kinds.update(child_kinds)
            namespaced.update(child_names)
        return converted, kinds, namespaced

    if tool_type in _IGNORED_TOOL_TYPES:
        raise MalformedRequestError(f"Responses tool type {tool_type!r} is not supported by this gateway")
    if tool_type in _HOSTED_TOOL_TYPES:
        source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = source.get("name") or tool_type
        if not isinstance(name, str) or not name:
            raise MalformedRequestError(f"Hosted Responses tool {tool_type!r} requires a name")
        qualified_name = f"{namespace}{_NAMESPACE_SEPARATOR}{name}" if namespace else name
        parameters = _clean_schema(
            source.get("parameters", source.get("input_schema", {"type": "object", "properties": {}}))
        )
        declaration = {
            "type": "function",
            "function": {
                "name": qualified_name,
                "description": str(source.get("description") or ""),
                "parameters": parameters,
            },
        }
        return [declaration], {qualified_name: str(tool_type)}, {qualified_name} if namespace else set()
    if tool_type not in {"function", "custom"}:
        raise MalformedRequestError(f"Unsupported Responses tool type {tool_type!r}")
    if drop_deferred and tool.get("defer_loading"):
        return [], {}, set()

    source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = source.get("name")
    if not isinstance(name, str) or not name:
        raise MalformedRequestError(f"{tool_type} tools require a non-empty name")
    qualified_name = f"{namespace}{_NAMESPACE_SEPARATOR}{name}" if namespace else name
    parameters = (
        _custom_parameters()
        if tool_type == "custom"
        else _clean_schema(source.get("parameters", {"type": "object", "properties": {}}))
    )
    declaration = {
        "type": "function",
        "function": {
            "name": qualified_name,
            "description": str(source.get("description") or ""),
            "parameters": parameters,
        },
    }
    return [declaration], {qualified_name: str(tool_type)}, {qualified_name} if namespace else set()


def _merge_tool_channels(top: list[Any], additional: list[Any]) -> list[Any]:
    if not top:
        return list(additional)
    if not additional:
        return list(top)
    top_names = {_declared_name(tool) for tool in top}
    return list(top) + [tool for tool in additional if _declared_name(tool) not in top_names]


def _collect_tool_defs(payload: dict[str, Any]) -> list[Any]:
    top = payload.get("tools") or []
    if not isinstance(top, list):
        raise MalformedRequestError("tools must be a list")
    additional: list[Any] = []
    input_value = payload.get("input")
    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, dict) and item.get("type") == "additional_tools":
                extra = item.get("tools")
                if not isinstance(extra, list):
                    raise MalformedRequestError("additional_tools requires a tools list")
                additional.extend(extra)
    return _merge_tool_channels(list(top), additional)


def _convert_tools(tools: Any, *, drop_deferred: bool = False) -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    if tools is None:
        return [], {}, set()
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    converted: list[dict[str, Any]] = []
    kinds: dict[str, str] = {}
    namespaced: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            raise MalformedRequestError("Each tool must be an object")
        child, child_kinds, child_names = _flatten_tool(tool, drop_deferred=drop_deferred)
        converted.extend(child)
        kinds.update(child_kinds)
        namespaced.update(child_names)
    return converted, kinds, namespaced


def _flush_assistant_pending(messages: list[dict[str, Any]], pending: dict[str, list[Any]] | None) -> None:
    if pending is None:
        return
    content = "".join(pending["content"])
    reasoning = "\n".join(part for part in pending["reasoning"] if part)
    tool_calls = pending["tool_calls"]
    if content or reasoning or tool_calls:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)


def _messages_from_input(payload: dict[str, Any]) -> list[dict[str, Any]]:
    input_value = payload.get("input")
    if not isinstance(input_value, str | list):
        raise MalformedRequestError("input must be a string or a list")
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise MalformedRequestError("instructions must be a string")
        if instructions:
            messages.append({"role": "system", "content": instructions})
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return canonicalize_messages(messages)

    pending: dict[str, list[Any]] | None = None

    def ensure_pending() -> dict[str, list[Any]]:
        nonlocal pending
        if pending is None:
            pending = {"content": [], "reasoning": [], "tool_calls": []}
        return pending

    for index, item in enumerate(input_value):
        param = f"input[{index}]"
        if not isinstance(item, dict):
            raise MalformedRequestError(f"Each input item must be an object at {param}")
        item_type = item.get("type", "message")
        if item_type in _SKIPPED_ITEM_TYPES:
            continue
        if item_type == "message":
            role = item.get("role")
            if not isinstance(role, str) or role not in {"system", "developer", "user", "assistant"}:
                raise MalformedRequestError(f"Unsupported Responses message role {role!r} at {param}.role")
            content = _content_to_chat_content(item.get("content", ""), param=f"{param}.content")
            if role == "assistant":
                if isinstance(content, list):
                    raise MalformedRequestError("image content is only supported in user messages")
                ensure_pending()["content"].append(content)
                continue
            _flush_assistant_pending(messages, pending)
            pending = None
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": content})
            continue
        if item_type == "reasoning":
            summary = _content_to_text(item.get("summary") or item.get("content") or [], param=param)
            if summary:
                ensure_pending()["reasoning"].append(summary)
            continue
        if item_type in {"function_call", "custom_tool_call"}:
            name = item.get("name")
            call_id = item.get("call_id")
            if not isinstance(name, str) or not name:
                raise MalformedRequestError(f"{item_type} requires a name")
            if not isinstance(call_id, str) or not call_id:
                raise MalformedRequestError(f"{item_type} requires a call_id")
            namespace = item.get("namespace")
            if isinstance(namespace, str) and namespace.strip() not in ("", _DEFAULT_NAMESPACE):
                name = f"{namespace}{_NAMESPACE_SEPARATOR}{name}"
            raw_arguments = item.get("arguments", item.get("input"))
            arguments = _custom_arguments(raw_arguments) if item_type == "custom_tool_call" else raw_arguments
            ensure_pending()["tool_calls"].append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue
        if item_type in {"function_call_output", "custom_tool_call_output", "computer_call_output"}:
            _flush_assistant_pending(messages, pending)
            pending = None
            call_id = item.get("call_id") or item.get("id")
            if not call_id:
                raise MalformedRequestError(f"{item_type} requires a call_id")
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": str(call_id), "content": output})
            continue
        if item_type in {"web_search_call", "computer_call", "tool_search_call"}:
            name = {"web_search_call": "web_search", "computer_call": "computer"}.get(item_type, "tool_search")
            raw_arguments = item.get("action", item.get("actions", item.get("arguments", {})))
            ensure_pending()["tool_calls"].append(
                {
                    "id": str(item.get("call_id") or item.get("id") or uuid4().hex),
                    "type": "function",
                    "function": {"name": name, "arguments": _json_arguments(raw_arguments)},
                }
            )
            continue
        if item_type in {"compaction", "compaction_summary"}:
            _flush_assistant_pending(messages, pending)
            pending = None
            messages.append({"role": "user", "content": ""})
            continue
        raise MalformedRequestError(f"Unsupported Responses input item type {item_type!r} at {param}")
    _flush_assistant_pending(messages, pending)
    if not any(message.get("role") in {"system", "user"} for message in messages):
        messages.append({"role": "user", "content": ""})
    return canonicalize_messages(messages)


def responses_to_internal(
    payload: dict[str, Any],
    *,
    base_sampling_params: dict[str, Any],
    allowed_sampling_keys: frozenset[str],
) -> InternalGenerationRequest:
    if not isinstance(payload, dict):
        raise MalformedRequestError("Request body must be a JSON object")
    if "max_tokens" in payload:
        raise MalformedRequestError("max_tokens is not a Responses field; use max_output_tokens")
    if payload.get("background") is True:
        raise MalformedRequestError("background Responses are not supported")
    if payload.get("store") is True:
        raise MalformedRequestError("stored Responses are not supported")
    if payload.get("previous_response_id") is not None:
        raise MalformedRequestError("previous_response_id is not supported; send the full input history")

    text_config = payload.get("text")
    if text_config is not None:
        if not isinstance(text_config, dict):
            raise MalformedRequestError("text must be an object")
        response_format = text_config.get("format")
        if response_format is not None:
            if not isinstance(response_format, dict) or response_format.get("type") != "text":
                raise MalformedRequestError(
                    "Only Responses text.format.type='text' is supported; structured output is not enabled"
                )

    parallel_tool_calls = payload.get("parallel_tool_calls", True)
    if type(parallel_tool_calls) is not bool:
        raise MalformedRequestError("parallel_tool_calls must be a boolean")

    if parallel_tool_calls is False:
        raise MalformedRequestError("parallel_tool_calls=false is not supported")

    tool_choice_payload = payload.get("tool_choice", "auto")
    if isinstance(tool_choice_payload, str):
        tool_choice = tool_choice_payload.lower()
        if tool_choice not in {"auto", "none"}:
            raise MalformedRequestError(
                f'tool_choice="{tool_choice_payload}" is not supported (only "auto" / "none" are supported)'
            )
    elif isinstance(tool_choice_payload, dict):
        if tool_choice_payload.get("type") == "function":
            raise MalformedRequestError(
                'tool_choice with a specific function is not supported (only "auto" / "none" are supported)'
            )
        raise MalformedRequestError('tool_choice object is not supported (only "auto" / "none" are supported)')
    else:
        raise MalformedRequestError("tool_choice must be a string or object")

    converted_tools, _, _ = _convert_tools(_collect_tool_defs(payload), drop_deferred=True)
    chat_payload: dict[str, Any] = {
        "messages": _messages_from_input(payload),
        "tools": None if tool_choice == "none" else converted_tools or None,
        "tool_choice": tool_choice,
    }
    for key in ("temperature", "top_p", "top_k", "stop"):
        if payload.get(key) is not None:
            chat_payload[key] = payload[key]
    if payload.get("max_output_tokens") is not None:
        chat_payload["max_tokens"] = payload["max_output_tokens"]
    return openai_to_internal(
        chat_payload,
        base_sampling_params=base_sampling_params,
        allowed_sampling_keys=allowed_sampling_keys,
    )


def _tool_metadata(payload: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
    _, kinds, namespaced = _convert_tools(_collect_tool_defs(payload))
    return kinds, namespaced


def _split_namespaced_name(name: str, namespaced_names: set[str]) -> tuple[str | None, str]:
    if name in namespaced_names:
        namespace, _, bare_name = name.partition(_NAMESPACE_SEPARATOR)
        if namespace and bare_name:
            return namespace, bare_name
    return None, name


def _custom_input(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    if isinstance(arguments, dict) and isinstance(arguments.get("input"), str):
        return arguments["input"]
    return json.dumps(arguments, ensure_ascii=False) if arguments is not None else ""


def _usage(outcome: GenerationOutcome) -> dict[str, Any]:
    return {
        "input_tokens": outcome.prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": outcome.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
    }


def _output_items(outcome: GenerationOutcome, payload: dict[str, Any]) -> list[dict[str, Any]]:
    message = outcome.assistant_msg
    kinds, namespaced_names = _tool_metadata(payload)
    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        output.append(
            {
                "id": f"rs_{uuid4().hex}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "status": "completed",
            }
        )
    content = message.get("content")
    if isinstance(content, str) and content:
        output.append(
            {
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": [], "logprobs": []}],
            }
        )
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        namespace, bare_name = _split_namespaced_name(name, namespaced_names)
        call_id = str(tool_call.get("id") or f"call_{uuid4().hex}")
        arguments = function.get("arguments", "")
        if kinds.get(name) == "custom":
            item: dict[str, Any] = {
                "id": f"ctc_{uuid4().hex}",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": call_id,
                "name": bare_name,
                "input": _custom_input(arguments),
            }
        else:
            item = {
                "id": f"fc_{uuid4().hex}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": bare_name,
                "arguments": _json_arguments(arguments),
            }
        if namespace:
            item["namespace"] = namespace
        output.append(item)
    return output


def _response_status(finish_reason: str) -> str:
    return "incomplete" if finish_reason in {"length", "max_tokens"} else "completed"


def _response_text(output: list[dict[str, Any]]) -> str:
    return "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text" and isinstance(part.get("text"), str)
    )


def _response_base(
    payload: dict[str, Any],
    *,
    response_id: str,
    model: str,
    created_at: int,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "instructions": payload.get("instructions"),
        "max_output_tokens": payload.get("max_output_tokens"),
        "model": model,
        "output": output,
        "output_text": _response_text(output),
        "parallel_tool_calls": bool(payload.get("parallel_tool_calls", True)),
        "previous_response_id": None,
        "reasoning": payload.get("reasoning"),
        "store": False,
        "temperature": payload.get("temperature"),
        "text": payload.get("text", {"format": {"type": "text"}}),
        "tool_choice": payload.get("tool_choice", "auto").lower(),
        "tools": payload.get("tools") or [],
        "top_p": payload.get("top_p"),
        "truncation": payload.get("truncation", "disabled"),
        "usage": usage,
        "metadata": payload.get("metadata") or {},
    }


def responses_build_response(
    outcome: GenerationOutcome,
    *,
    payload: dict[str, Any] | None = None,
    model: str,
    response_id: str | None = None,
) -> dict[str, Any]:
    payload = dict(payload or {})
    output = _output_items(outcome, payload)
    status = _response_status(outcome.finish_reason)
    return _response_base(
        payload,
        response_id=response_id or f"resp_{uuid4().hex}",
        model=model,
        created_at=int(time.time()),
        status=status,
        output=output,
        usage=_usage(outcome),
        incomplete_reason="max_output_tokens" if status == "incomplete" else None,
    )


def _event_to_sse(event: dict[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


async def _as_generation(value: GenerationOutcome | Callable[[], Awaitable[GenerationOutcome]]) -> GenerationOutcome:
    if callable(value):
        return await value()
    return value


def responses_stream_response(
    run_generation: GenerationOutcome | Callable[[], Awaitable[GenerationOutcome]],
    *,
    payload: dict[str, Any] | None = None,
    model: str,
    heartbeat_interval_s: float = 15.0,
) -> StreamingResponse:
    """Serve a Responses SSE stream without blocking the client during generation."""
    payload = dict(payload or {})
    response_id = f"resp_{uuid4().hex}"
    created_at = int(time.time())

    async def _gen() -> AsyncIterator[bytes]:
        sequence_number = 0

        def event(event_type: str, **fields: Any) -> dict[str, Any]:
            nonlocal sequence_number
            result = {"type": event_type, "sequence_number": sequence_number, **fields}
            sequence_number += 1
            return result

        in_progress = _response_base(
            payload,
            response_id=response_id,
            model=model,
            created_at=created_at,
            status="in_progress",
            output=[],
            usage=None,
        )
        yield _event_to_sse(event("response.created", response=in_progress))
        yield _event_to_sse(event("response.in_progress", response=in_progress))
        task = asyncio.create_task(_as_generation(run_generation))
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=heartbeat_interval_s)
                if not done:
                    yield _event_to_sse(event("response.in_progress", response=in_progress))
            outcome = task.result()
            output = _output_items(outcome, payload)
            for output_index, item in enumerate(output):
                item_type = item["type"]
                if item_type == "reasoning":
                    in_flight = {**item, "status": "in_progress", "summary": []}
                elif item_type == "message":
                    in_flight = {**item, "status": "in_progress", "content": []}
                elif item_type == "custom_tool_call":
                    in_flight = {**item, "status": "in_progress", "input": ""}
                else:
                    in_flight = {**item, "status": "in_progress", "arguments": ""}
                yield _event_to_sse(event("response.output_item.added", output_index=output_index, item=in_flight))
                if item_type == "reasoning":
                    text = item["summary"][0]["text"]
                    yield _event_to_sse(
                        event(
                            "response.reasoning_summary_text.delta",
                            item_id=item["id"],
                            output_index=output_index,
                            summary_index=0,
                            delta=text,
                        )
                    )
                    yield _event_to_sse(
                        event(
                            "response.reasoning_summary_text.done",
                            item_id=item["id"],
                            output_index=output_index,
                            summary_index=0,
                            text=text,
                        )
                    )
                elif item_type == "message":
                    text = item["content"][0]["text"]
                    yield _event_to_sse(
                        event(
                            "response.output_text.delta",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=0,
                            delta=text,
                        )
                    )
                    yield _event_to_sse(
                        event(
                            "response.output_text.done",
                            item_id=item["id"],
                            output_index=output_index,
                            content_index=0,
                            text=text,
                        )
                    )
                elif item_type == "custom_tool_call":
                    yield _event_to_sse(
                        event(
                            "response.custom_tool_call_input.delta",
                            item_id=item["id"],
                            output_index=output_index,
                            delta=item["input"],
                        )
                    )
                    yield _event_to_sse(
                        event(
                            "response.custom_tool_call_input.done",
                            item_id=item["id"],
                            output_index=output_index,
                            input=item["input"],
                        )
                    )
                else:
                    yield _event_to_sse(
                        event(
                            "response.function_call_arguments.delta",
                            item_id=item["id"],
                            output_index=output_index,
                            delta=item["arguments"],
                        )
                    )
                    yield _event_to_sse(
                        event(
                            "response.function_call_arguments.done",
                            item_id=item["id"],
                            output_index=output_index,
                            arguments=item["arguments"],
                        )
                    )
                yield _event_to_sse(event("response.output_item.done", output_index=output_index, item=item))
            completed = responses_build_response(outcome, payload=payload, model=model, response_id=response_id)
            yield _event_to_sse(event(f"response.{completed['status']}", response=completed))
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            logger.exception("Responses generation failed")
            failed = _response_base(
                payload,
                response_id=response_id,
                model=model,
                created_at=created_at,
                status="failed",
                output=[],
                usage=None,
            )
            failed["error"] = {"code": "internal_error", "message": str(exc)}
            yield _event_to_sse(event("response.failed", response=failed))
            yield _event_to_sse(event("error", code="internal_error", message=str(exc), param=None))

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_RESPONSES_HEADERS)
