"""DeepEyes multimodal ReAct agent."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Any

from PIL import Image
from pydantic import Field

from uni_agent.agents.react.model import OpenAICompatibleChatModel
from uni_agent.tools import Toolbox, ToolResult

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent
from .tool import ImageZoomInConfig, ImageZoomInResult, ImageZoomInTool, coerce_image

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You can call functions to assist with the user query. "
    "Important: You must call only one function at a time. "
    "Put the final concise answer inside exactly one <answer>...</answer> block."
)


class DeepEyesAgentConfig(AgentConfig):
    """DeepEyes policy loop and image-tool settings."""

    name: str = "deepeyes"
    max_turns: int = Field(default=4, gt=0)
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    request_max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries after transient HTTP failures. Set to zero for long-running policy requests.",
    )
    image_tool: ImageZoomInConfig = Field(default_factory=ImageZoomInConfig)


@register_agent("deepeyes")
class DeepEyesAgent(Agent):
    """Sequential multimodal ReAct loop whose only action is image zoom."""

    config_model = DeepEyesAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: DeepEyesAgentConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("deepeyes: config.model.base_url is not set")

        prompt = _with_system_prompt(messages)
        source_image = _first_image(prompt, timeout=cfg.image_tool.fetch_timeout_seconds)
        transcript = messages_for_gateway(prompt)
        logger.info(
            "DeepEyes agent start: source_image=%sx%s messages=%s max_turns=%s",
            source_image.width,
            source_image.height,
            len(transcript),
            cfg.max_turns,
        )
        tool = ImageZoomInTool(
            sandbox,
            image=source_image,
            **cfg.image_tool.model_dump(),
        )
        toolbox = Toolbox([tool])
        request_params = cfg.model.sampling_params()
        request_params["tool_choice"] = "auto"
        if cfg.model.max_tokens_per_turn is not None:
            request_params["max_tokens"] = cfg.model.max_tokens_per_turn
        model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params=request_params,
            tools_schemas=toolbox.schemas(),
            timeout=cfg.request_timeout_seconds,
            max_retries=cfg.request_max_retries,
        )

        info: dict[str, Any] = {
            "steps": 0,
            "tool_calls": 0,
            "tool_successes": 0,
            "tool_errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        final_answer: str | None = None
        termination_reason = "unknown"
        try:
            async with toolbox.entered(retry=1, timeout=60):
                for turn_index in range(cfg.max_turns):
                    info["steps"] = turn_index + 1
                    logger.info("DeepEyes agent turn %s/%s", turn_index + 1, cfg.max_turns)
                    turn_params = dict(request_params)
                    if cfg.model.max_total_tokens is not None:
                        remaining_tokens = cfg.model.max_total_tokens - info["completion_tokens"]
                        if remaining_tokens <= 0:
                            termination_reason = "token_limit"
                            logger.warning(
                                "DeepEyes exhausted max_total_tokens=%s before turn %s",
                                cfg.model.max_total_tokens,
                                turn_index + 1,
                            )
                            break
                        turn_params["max_tokens"] = min(
                            int(turn_params.get("max_tokens", remaining_tokens)),
                            remaining_tokens,
                        )
                    content, tool_calls, generation = await model.query(
                        transcript,
                        sampling_params=turn_params,
                    )
                    info["prompt_tokens"] += int(generation.get("prompt_tokens", 0))
                    info["completion_tokens"] += int(generation.get("completion_tokens", 0))
                    info["total_tokens"] = info["prompt_tokens"] + info["completion_tokens"]

                    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
                    if tool_calls:
                        assistant_message["tool_calls"] = tool_calls
                    transcript.append(assistant_message)
                    logger.info("DeepEyes assistant:\n%s", content)

                    finish_reason = generation.get("finish_reason")
                    if not tool_calls:
                        final_answer = assistant_text(content)
                        termination_reason = "token_limit" if finish_reason == "length" else "finished"
                        break

                    info["tool_calls"] += len(tool_calls)
                    if turn_index + 1 >= cfg.max_turns:
                        info["tool_errors"] += len(tool_calls)
                        termination_reason = "max_turns"
                        logger.warning("DeepEyes exhausted max_turns=%s with pending tool calls", cfg.max_turns)
                        break

                    for tool_call in tool_calls:
                        function = tool_call.get("function") or {}
                        tool_name = str(function.get("name", ""))
                        logger.info("DeepEyes tool call %s: %s", tool_name, function.get("arguments"))
                        result = await toolbox.call(
                            tool_name,
                            function.get("arguments"),
                        )
                        successful = (
                            result.status == "ok"
                            and isinstance(result, ImageZoomInResult)
                            and result.info.get("success") is True
                        )
                        if successful:
                            info["tool_successes"] += 1
                        else:
                            info["tool_errors"] += 1
                        transcript.append(_tool_message(str(tool_call.get("id", "")), result))
                        logger.info("DeepEyes tool result status=%s: %s", result.status, result.text or "")
                else:
                    termination_reason = "max_turns"
        finally:
            await model.aclose()

        finished = termination_reason == "finished"
        info["termination_reason"] = termination_reason
        logger.info(
            "DeepEyes agent done: reason=%s turns=%s calls=%s successes=%s errors=%s final_answer=%s",
            termination_reason,
            info["steps"],
            info["tool_calls"],
            info["tool_successes"],
            info["tool_errors"],
            final_answer,
        )
        return AgentResult(
            output={"final_answer": final_answer, "termination_reason": termination_reason},
            transcript=transcript,
            info=info,
            finished=finished,
        )


def _with_system_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("DeepEyes task prompt must be a non-empty message list")
    copied = [dict(message) for message in messages]
    if copied[0].get("role") != "system":
        copied.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    return copied


def _first_image(messages: list[dict[str, Any]], *, timeout: float):
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if "image" in part:
                return coerce_image(part["image"], timeout=timeout)
            if "bytes" in part:
                return coerce_image({"bytes": part["bytes"]}, timeout=timeout)
            if part.get("type") == "image_url" or "image_url" in part:
                return coerce_image({"image_url": part.get("image_url")}, timeout=timeout)
    raise ValueError("DeepEyes prompt does not contain a source image")


def _tool_message(tool_call_id: str, result: ToolResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if result.text is not None:
        content.append({"type": "text", "text": str(result.text)})
    if isinstance(result, ImageZoomInResult):
        content.extend({"type": "image_url", "image_url": {"url": image_data_url(image)}} for image in result.images)
    if not content:
        content.append({"type": "text", "text": ""})
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def messages_for_gateway(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return serializable OpenAI messages with canonical image URL blocks."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("raw_prompt must be a non-empty message list")
    return [_message_for_gateway(message) for message in messages]


def _message_for_gateway(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise TypeError("raw_prompt messages must be mappings")
    normalized = dict(message)
    content = normalized.get("content", "")
    if isinstance(content, list):
        normalized["content"] = [_content_part_for_gateway(part) for part in content]
    elif not isinstance(content, str):
        raise TypeError("message content must be text or a content-part list")
    return normalized


def _content_part_for_gateway(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    image_keys = {"image", "image_url", "bytes"}
    if part.get("type") not in {"image", "image_url"} and not image_keys.intersection(part):
        return dict(part)
    if "image" in part:
        payload = part["image"]
    elif "bytes" in part:
        payload = {"bytes": part["bytes"]}
    else:
        image_url = part.get("image_url")
        payload = image_url.get("url") if isinstance(image_url, dict) else image_url
    return {"type": "image_url", "image_url": {"url": image_data_url(payload)}}


def image_data_url(value: Any) -> str:
    """Encode a supported in-memory image value as an OpenAI data URL."""

    if isinstance(value, str):
        return value
    if isinstance(value, Image.Image):
        buffer = BytesIO()
        value.convert("RGB").save(buffer, format="PNG")
        value = buffer.getvalue()
    elif isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]
    if isinstance(value, bytes | bytearray):
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    raise TypeError(f"unsupported image payload: {type(value).__name__}")


def assistant_text(content: Any) -> str:
    """Flatten an OpenAI assistant content value to final-answer text."""

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", "")).strip() for part in content if isinstance(part, dict) and part.get("text")
    ).strip()
