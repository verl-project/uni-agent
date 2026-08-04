"""Model-scoped codec for tokenizer, processor, tool-parser, and decode paths.

This layer stays within the model boundary: it applies chat templates, handles
processor-backed multimodal inputs, parses tools, and decodes backend outputs.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from verl.utils.tokenizer import normalize_token_ids
from verl.utils.tokenizer.chat_template import apply_chat_template as _apply_chat_template
from verl.utils.tokenizer.chat_template import initialize_turn_separator

logger = logging.getLogger("gateway")

# Map backend stop_reason values into the gateway's internal finish_reason vocabulary.
_FINISH_REASON_MAP = {
    "completed": "stop",
    "stop": "stop",
    "matched_stop": "stop",
    "eos": "stop",
    "length": "length",
    "max_tokens": "length",
    "aborted": "stop",
    "abort": "stop",
}

_SGLANG_TOOL_PARSER_ALIASES = {
    "qwen3_xml": "qwen3_coder",
}

_VLLM_TOOL_PARSER_ALIASES = {
    "qwen": "qwen3_xml",
    "qwen25": "qwen3_xml",
    "qwen3": "qwen3_xml",
}


def initialize_generation_prompt(processing_class, **apply_chat_template_kwargs) -> list[int]:
    """Initialize the token suffix inserted by ``add_generation_prompt=True``."""
    without_generation_prompt = normalize_token_ids(
        _apply_chat_template(
            processing_class,
            [{"role": "user", "content": ""}],
            add_generation_prompt=False,
            **apply_chat_template_kwargs,
        )
    )
    with_generation_prompt = normalize_token_ids(
        _apply_chat_template(
            processing_class,
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            **apply_chat_template_kwargs,
        )
    )
    if with_generation_prompt[: len(without_generation_prompt)] != without_generation_prompt:
        raise ValueError("Generation prompt is not a stable token suffix")
    return with_generation_prompt[len(without_generation_prompt) :]


def _canonicalize_tool_arguments_for_comparison(arguments: Any) -> tuple[str, Any]:
    if isinstance(arguments, dict | list):
        return ("json", arguments)
    if isinstance(arguments, str):
        try:
            return ("json", json.loads(arguments))
        except json.JSONDecodeError:
            return ("raw", arguments)
    return ("raw", arguments)


def _process_tool_calls_sglang(
    text: str,
    tools: list[dict[str, Any]],
    parser_name: str,
) -> tuple[str, list[Any]]:
    from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
    from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    sglang_tools = [SglTool(type=tool["type"], function=SglFunction(**tool["function"])) for tool in tools]
    parser = FunctionCallParser(sglang_tools, parser_name)
    if not parser.has_tool_call(text):
        return text, []

    content, calls = parser.parse_non_stream(text)
    return content, [SimpleNamespace(name=call.name, arguments=call.parameters) for call in calls]


def _process_tool_calls_vllm(
    text: str,
    tools: list[dict[str, Any]],
    parser_name: str,
    tokenizer,
) -> tuple[str, list[Any]]:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
    from vllm.tool_parsers import ToolParserManager

    parser_cls = ToolParserManager.get_tool_parser(parser_name)
    vllm_tools = [ChatCompletionToolsParam(**tool) if isinstance(tool, dict) else tool for tool in tools]
    parser = parser_cls(tokenizer, tools=vllm_tools)
    request = SimpleNamespace(
        tools=vllm_tools,
        tool_choice="auto",
        skip_special_tokens=True,
    )
    parsed = parser.extract_tool_calls(text, request)
    if not parsed.tools_called:
        return text, []
    return parsed.content or "", [tool_call.function for tool_call in parsed.tool_calls]


def _extract_tool_calls_with_sglang_or_vllm(
    text: str,
    tools: list[dict[str, Any]],
    parser_name: str,
    tokenizer,
) -> tuple[str, list[Any]]:
    sglang_name = _SGLANG_TOOL_PARSER_ALIASES.get(parser_name, parser_name)
    try:
        return _process_tool_calls_sglang(text, tools, sglang_name)
    except ModuleNotFoundError:
        pass
    except Exception:
        logger.warning("SGLang tool-call parsing failed; trying vLLM", exc_info=True)

    vllm_name = _VLLM_TOOL_PARSER_ALIASES.get(parser_name, parser_name)
    try:
        return _process_tool_calls_vllm(text, tools, vllm_name, tokenizer)
    except ModuleNotFoundError:
        pass
    except Exception:
        logger.warning("vLLM tool-call parsing failed; returning raw text", exc_info=True)

    return text, []


class MessageCodec:
    """Model-scoped request codec used by gateway sessions.

    ``_GatewayActor`` owns one codec per actor and injects it into
    ``GatewaySession`` instances. The codec renders chat templates, handles
    multimodal processor inputs, and decodes backend token outputs without
    reading session state.
    """

    def __init__(
        self,
        tokenizer,
        *,
        processor=None,
        vision_info_extractor=None,
        vision_info_extractor_kwargs: dict[str, Any] | None = None,
        tool_parser_name: str | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self._vision_info_extractor = vision_info_extractor or self._default_vision_info_extractor
        self._vision_info_extractor_kwargs = dict(vision_info_extractor_kwargs or {})
        self._apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        processing_class = self._processor if self._processor is not None else tokenizer
        self._generation_prompt = initialize_generation_prompt(
            processing_class,
            **self._apply_chat_template_kwargs,
        )
        self._turn_separator = initialize_turn_separator(
            processing_class,
            **self._apply_chat_template_kwargs,
        )
        self._tool_parser_name = tool_parser_name

    @property
    def generation_prompt(self) -> list[int]:
        """Return the configured chat template's generation-prompt token suffix."""
        return list(self._generation_prompt)

    @property
    def turn_separator(self) -> list[int]:
        """Return the configured chat template's inter-turn separator tokens."""
        return list(self._turn_separator)

    async def _default_vision_info_extractor(
        self,
        messages: list[dict[str, Any]],
        *,
        image_patch_size: int,
        **_extra: Any,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        # Lazy import so callers without multi-modal needs do not load
        # qwen_vl_utils. ``_extra`` absorbs ``vision_info_extractor_kwargs`` that
        # ``extract_multi_modal_data`` forwards for custom extractors; the
        # default path needs nothing beyond ``messages`` and patch size.
        from qwen_vl_utils import process_vision_info

        return process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )

    async def extract_multi_modal_data(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[Any] | None, list[Any] | None]:
        """Extract image and video inputs when a processor-backed request needs them."""
        if self._processor is None:
            return None, None

        has_multi_modal_blocks = False
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"image", "image_url", "video", "video_url"}:
                    has_multi_modal_blocks = True
                    break
            if has_multi_modal_blocks:
                break

        if not has_multi_modal_blocks:
            return None, None

        return await self._vision_info_extractor(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            **self._vision_info_extractor_kwargs,
        )

    def _encode_prompt_text(
        self,
        prompt: str,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Encode rendered prompt text with the configured tokenizer or processor."""
        if self._processor is None:
            return normalize_token_ids(self._tokenizer.encode(prompt, add_special_tokens=False))

        videos = video_data
        video_metadata = None
        if videos is not None:
            videos, video_metadata = zip(*videos, strict=False)
            videos, video_metadata = list(videos), list(video_metadata)
        model_inputs = self._processor(
            text=[prompt],
            images=image_data,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_sample_frames=False,
        )
        return normalize_token_ids(model_inputs["input_ids"])

    def encode_full(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Encode a full chat history into prompt token IDs."""
        processing_class = self._processor if self._processor is not None else self._tokenizer
        raw_prompt = _apply_chat_template(
            processing_class,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **self._apply_chat_template_kwargs,
        )
        return self._encode_prompt_text(raw_prompt, image_data, video_data)

    def encode_incremental(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> list[int]:
        """Encode continuation messages using a dummy-user anchored delta."""
        if not messages:
            return []

        processing_class = self._processor if self._processor is not None else self._tokenizer
        anchor_content = [{"type": "text", "text": ""}] if self._processor is not None else ""
        anchor = [{"role": "user", "content": anchor_content}]

        if any(message.get("role") == "assistant" for message in messages[1:]):
            raise ValueError("An incremental assistant message may only appear first")

        # TODO: Replace this user/tool empty-user fallback with continuous-token merging.
        # A user -> tool anchor is not valid for every chat template.
        anchor_prompt = _apply_chat_template(
            processing_class,
            anchor,
            add_generation_prompt=False,
            tokenize=False,
            **self._apply_chat_template_kwargs,
        )
        full_prompt = _apply_chat_template(
            processing_class,
            anchor + messages,
            add_generation_prompt=True,
            tokenize=False,
            **self._apply_chat_template_kwargs,
        )
        prefix_prompt = anchor_prompt
        if self._turn_separator:
            separator_text = self._tokenizer.decode(self._turn_separator, skip_special_tokens=False)
            if not separator_text or not anchor_prompt.endswith(separator_text):
                raise ValueError("Turn separator is not a stable text suffix")
            prefix_prompt = anchor_prompt[: -len(separator_text)]
        if not full_prompt.startswith(prefix_prompt):
            raise ValueError("Incremental chat template is not prefix-stable")
        return self._encode_prompt_text(
            full_prompt[len(prefix_prompt) :],
            image_data,
            video_data,
        )

    async def decode_response(
        self,
        response_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop_reason: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Decode model output tokens into an assistant message and finish reason."""
        if self._tool_parser_name and tools:
            response_text = self._tokenizer.decode(response_ids, skip_special_tokens=False)
            content, function_calls = _extract_tool_calls_with_sglang_or_vllm(
                response_text,
                tools,
                self._tool_parser_name,
                self._tokenizer,
            )
            if function_calls:
                tool_calls = [
                    {
                        "id": f"call_{uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": fc.name, "arguments": fc.arguments},
                    }
                    for fc in function_calls
                ]
                message = {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                }
                return message, "tool_calls"
        response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, stop_reason) if stop_reason else "stop"
        return {"role": "assistant", "content": response_text}, finish_reason

    def canonicalize_message_for_prefix_comparison(self, message: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize one message before session prefix comparison."""
        normalized = dict(message)
        normalized.pop("tool_call_id", None)
        tool_calls = normalized.get("tool_calls")
        if not isinstance(tool_calls, list):
            return normalized

        normalized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            normalized_tool_call = dict(tool_call)
            normalized_tool_call.pop("id", None)
            function = normalized_tool_call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                normalized_function = dict(function)
                normalized_function["arguments"] = _canonicalize_tool_arguments_for_comparison(function["arguments"])
                normalized_tool_call["function"] = normalized_function
            normalized_tool_calls.append(normalized_tool_call)
        normalized["tool_calls"] = normalized_tool_calls
        return normalized
