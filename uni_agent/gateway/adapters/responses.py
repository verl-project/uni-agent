"""OpenAI Responses API adapter used by Codex.

The session core operates on a compact chat-shaped request. Codex emits the
Responses wire format, so this adapter owns the translation in both directions
and keeps the session/Gateway trajectory logic unchanged.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from uni_agent.gateway.message_normalization import canonicalize_messages

if TYPE_CHECKING:
    from uni_agent.gateway.session.session import GenerationOutcome
    from uni_agent.gateway.session.types import InternalGenerationRequest
from .types import MalformedRequestError

_RESPONSES_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise MalformedRequestError("Responses message content must be a string or list")
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            raise MalformedRequestError("Responses message content items must be objects")
        kind = item.get("type")
        if kind in {"input_text", "output_text", "summary_text", "text"}:
            text = item.get("text", "")
            if not isinstance(text, str):
                raise MalformedRequestError("Responses text content must contain string text")
            parts.append(text)
    return "".join(parts)


def _response_input_to_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise MalformedRequestError("instructions must be a string")
        if instructions:
            messages.append({"role": "system", "content": instructions})

    input_value = payload.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return canonicalize_messages(messages)
    if not isinstance(input_value, list) or not input_value:
        raise MalformedRequestError("input must be a non-empty string or list")

    for item in input_value:
        if not isinstance(item, dict):
            raise MalformedRequestError("input items must be objects")
        item_type = item.get("type", "message")
        if item_type == "message":
            role = item.get("role")
            if role == "developer":
                role = "system"
            if role not in {"system", "user", "assistant"}:
                raise MalformedRequestError(f"unsupported Responses message role: {role!r}")
            messages.append({"role": role, "content": _text_content(item.get("content", ""))})
        elif item_type == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise MalformedRequestError("function_call.name must be a non-empty string")
            call_id = str(item.get("call_id") or item.get("id") or uuid4().hex)
            arguments = item.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise MalformedRequestError("function_call.arguments must be valid JSON") from exc
            if not isinstance(arguments, dict):
                raise MalformedRequestError("function_call.arguments must decode to an object")
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            )
        elif item_type == "function_call_output":
            call_id = item.get("call_id") or item.get("id")
            if not call_id:
                raise MalformedRequestError("function_call_output.call_id is required")
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": str(call_id), "content": output})
        elif item_type == "reasoning":
            summary = item.get("summary") or []
            text = _text_content(summary)
            if text:
                messages.append({"role": "assistant", "content": "", "reasoning_content": text})
        elif item_type in {"computer_call_output", "custom_tool_call_output"}:
            call_id = item.get("call_id") or item.get("id")
            output = item.get("output", "")
            if call_id:
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": str(call_id), "content": output})
        else:
            # Codex includes metadata/tool items that do not belong in the
            # model chat context. Ignore them rather than dropping the request.
            continue
    if not messages or not any(message.get("role") in {"user", "system"} for message in messages):
        raise MalformedRequestError("input does not contain a user/system message")
    return canonicalize_messages(messages)


def _responses_tools_to_chat(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise MalformedRequestError("tools entries must be objects")
        kind = tool.get("type")
        if kind == "function":
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise MalformedRequestError("function tool name must be a non-empty string")
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        elif kind == "namespace":
            namespace = tool.get("name")
            if not isinstance(namespace, str) or not namespace:
                raise MalformedRequestError("namespace tool name must be a non-empty string")
            for function in tool.get("tools", []):
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                result.append(
                    {
                        "type": "function",
                        "function": {
                            **function,
                            "name": f"{namespace}.{function['name']}",
                        },
                    }
                )
        # Hosted tools such as web_search are owned by Codex and do not map to
        # a model function in the Gateway; omit them from the chat request.
    return result or None


def responses_to_internal(
    payload: dict[str, Any],
    *,
    base_sampling_params: dict[str, Any],
    allowed_sampling_keys: frozenset[str],
) -> InternalGenerationRequest:
    if not isinstance(payload, dict):
        raise MalformedRequestError("Responses body must be an object")
    messages = _response_input_to_messages(payload)
    sampling_params = dict(base_sampling_params)
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
        ("max_tokens", "max_tokens"),
    ):
        if source in payload and target in allowed_sampling_keys:
            sampling_params[target] = payload[source]
    return {
        "messages": messages,
        "tools": _responses_tools_to_chat(payload.get("tools")),
        "sampling_params": sampling_params,
    }


def _usage(outcome: GenerationOutcome) -> dict[str, int]:
    return {
        "input_tokens": outcome.prompt_tokens,
        "output_tokens": outcome.completion_tokens,
        "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
    }


def _response_status(finish_reason: str) -> str:
    return "incomplete" if finish_reason in {"length", "max_tokens"} else "completed"


def _output_items(outcome: GenerationOutcome) -> list[dict[str, Any]]:
    message = outcome.assistant_msg
    item_status = _response_status(outcome.finish_reason)
    items: list[dict[str, Any]] = []
    if message.get("reasoning_content"):
        items.append(
            {
                "id": f"rs_{uuid4().hex}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": message["reasoning_content"]}],
                "status": item_status,
            }
        )
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        function = call.get("function") or {}
        call_id = str(call.get("id") or uuid4().hex)
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        items.append(
            {
                "id": call_id,
                "type": "function_call",
                "status": item_status,
                "call_id": call_id,
                "name": function.get("name", ""),
                "arguments": arguments,
            }
        )
    content = message.get("content")
    if isinstance(content, str) and content:
        items.append(
            {
                "id": f"msg_{uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": item_status,
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )
    return items


def responses_build_response(
    outcome: GenerationOutcome, *, model: str, response_id: str | None = None
) -> dict[str, Any]:
    output = _output_items(outcome)
    output_text = "".join(
        item["text"]
        for output_item in output
        if output_item.get("type") == "message"
        for item in output_item.get("content", [])
        if item.get("type") == "output_text" and isinstance(item.get("text"), str)
    )
    status = _response_status(outcome.finish_reason)
    response = {
        "id": response_id or f"resp_{uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "output_text": output_text,
        "usage": _usage(outcome),
        "parallel_tool_calls": True,
    }
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    return response


def _event(event_type: str, **fields: Any) -> bytes:
    payload = {"type": event_type, **fields}
    encoded = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {encoded}\n\n".encode()


def responses_stream_response(outcome: GenerationOutcome, *, model: str):
    from fastapi.responses import StreamingResponse

    response_id = f"resp_{uuid4().hex}"
    response = responses_build_response(outcome, model=model, response_id=response_id)

    async def _gen() -> AsyncIterator[bytes]:
        in_progress = {key: value for key, value in response.items() if key != "incomplete_details"}
        in_progress["status"] = "in_progress"
        in_progress["output"] = []
        yield _event("response.created", response=in_progress)
        yield _event("response.in_progress", response=in_progress)
        for output_index, item in enumerate(response["output"]):
            yield _event(
                "response.output_item.added", output_index=output_index, item={**item, "status": "in_progress"}
            )
            if item["type"] == "message":
                content = item["content"][0]
                yield _event(
                    "response.content_part.added",
                    output_index=output_index,
                    content_index=0,
                    item_id=item["id"],
                    part=content,
                )
                yield _event(
                    "response.output_text.delta",
                    output_index=output_index,
                    content_index=0,
                    item_id=item["id"],
                    delta=content["text"],
                )
                yield _event(
                    "response.output_text.done",
                    output_index=output_index,
                    content_index=0,
                    item_id=item["id"],
                    text=content["text"],
                )
                yield _event(
                    "response.content_part.done",
                    output_index=output_index,
                    content_index=0,
                    item_id=item["id"],
                    part=content,
                )
            elif item["type"] == "function_call":
                yield _event(
                    "response.function_call_arguments.delta",
                    output_index=output_index,
                    item_id=item["id"],
                    delta=item["arguments"],
                )
                yield _event(
                    "response.function_call_arguments.done",
                    output_index=output_index,
                    item_id=item["id"],
                    arguments=item["arguments"],
                )
            yield _event("response.output_item.done", output_index=output_index, item=item)
        terminal_event = "response.incomplete" if response["status"] == "incomplete" else "response.completed"
        yield _event(terminal_event, response=response)

    return StreamingResponse(_gen(), media_type="text/event-stream", headers=_RESPONSES_HEADERS)
