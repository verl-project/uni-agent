from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import TYPE_CHECKING, Any

import httpx
from PIL import Image

if TYPE_CHECKING:
    from uni_agent.trainer.framework.types import SessionHandle
    from verl.tools.schemas import ToolResponse
else:
    SessionHandle = Any
    ToolResponse = Any


IMAGE_ZOOM_IN_TOOL_NAME = "image_zoom_in_tool"
GATEWAY_REQUEST_TIMEOUT_SECONDS = 300.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, Image.Image):
        buffer = BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    if isinstance(value, dict):
        if "bytes" in value:
            return _json_ready(value["bytes"])
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _tool_kwargs_for_name(tools_kwargs: dict | None) -> dict[str, Any]:
    if not isinstance(tools_kwargs, dict):
        return {}

    maybe_tool_kwargs = tools_kwargs.get(IMAGE_ZOOM_IN_TOOL_NAME)
    return maybe_tool_kwargs if isinstance(maybe_tool_kwargs, dict) else {}


def _parse_tool_arguments(arguments: object) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_message_from_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not choices:
        raise ValueError("chat completion response did not include choices")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat completion response choice did not include a message")
    return message


def _tool_response_to_openai_tool_message(*, tool_call_id: str, tool_response: ToolResponse) -> dict[str, Any]:
    content: list[dict[str, Any]] = []

    if tool_response.video:
        raise NotImplementedError("ToolResponse video content is not supported by the DeepEyes gateway recipe")

    if tool_response.text is not None:
        content.append({"type": "text", "text": str(tool_response.text)})
    for image in tool_response.image or []:
        content.append({"type": "image", "image": _json_ready(image)})
    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def _select_tool(tool_config: list[Any] | None):
    if not tool_config:
        raise ValueError("tool_config is required for deepeyes_agent_runner")

    for tool in tool_config:
        if getattr(tool, "name", None) == IMAGE_ZOOM_IN_TOOL_NAME:
            return tool
    raise ValueError(f"tool_config must include {IMAGE_ZOOM_IN_TOOL_NAME}")


async def deepeyes_agent_runner(
    *,
    raw_prompt: list[dict],
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict | None = None,
    tool_config: list[Any] | None = None,
    max_turns: int = 5,
    **kwargs,
) -> None:
    """Run a DeepEyes multi-turn image zoom-in tool loop against the gateway."""
    del sample_index, kwargs
    if session.base_url is None:
        raise ValueError("session.base_url is required for deepeyes_agent_runner")

    image_tool = _select_tool(tool_config)
    image_tool_kwargs = _tool_kwargs_for_name(tools_kwargs)
    create_kwargs = dict(image_tool_kwargs.get("create_kwargs") or {})
    if "image" not in create_kwargs and "image" in image_tool_kwargs:
        create_kwargs["image"] = image_tool_kwargs["image"]
    execute_kwargs = dict(image_tool_kwargs.get("execute_kwargs") or {})
    release_kwargs = dict(image_tool_kwargs.get("release_kwargs") or {})

    tool_instance_id: str | None = None
    messages = _json_ready(list(raw_prompt))

    try:
        tool_instance_id, _ = await image_tool.create(
            instance_id=f"{session.session_id}-image_zoom_in_tool",
            create_kwargs=create_kwargs,
        )
        tool_schema = image_tool.get_openai_tool_schema().model_dump(exclude_none=True)

        async with httpx.AsyncClient(timeout=GATEWAY_REQUEST_TIMEOUT_SECONDS) as client:
            for turn_index in range(max(0, max_turns)):
                response = await client.post(
                    f"{session.base_url}/chat/completions",
                    json={
                        "model": "deepeyes",
                        "messages": messages,
                        "tools": [tool_schema],
                    },
                )
                response.raise_for_status()

                assistant_message = _assistant_message_from_response(response.json())
                messages.append(dict(assistant_message))

                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls or turn_index + 1 >= max_turns:
                    break

                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    parameters = _parse_tool_arguments(function.get("arguments"))
                    tool_response, _, _ = await image_tool.execute(
                        tool_instance_id,
                        parameters=parameters,
                        **execute_kwargs,
                    )
                    messages.append(
                        _tool_response_to_openai_tool_message(
                            tool_call_id=tool_call.get("id", ""),
                            tool_response=tool_response,
                        )
                    )
    finally:
        if tool_instance_id is not None:
            await image_tool.release(tool_instance_id, **release_kwargs)
