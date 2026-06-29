"""Model-scoped codec for tokenizer, processor, tool-parser, and sampling boundaries."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.tokenizer.chat_template import apply_chat_template as _apply_chat_template
from verl.utils.tokenizer.chat_template import initialize_system_prompt
from verl.utils.tokenizer import normalize_token_ids


class MalformedRequestError(ValueError):
    """Raised when request payload normalization finds unsupported input."""

    pass


_DEFAULT_ALLOWED_REQUEST_SAMPLING_PARAM_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
    }
)


# Map backend stop_reason values to OpenAI-spec finish_reason values.
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


def _canonicalize_tool_arguments_for_comparison(arguments: Any) -> tuple[str, Any]:
    if isinstance(arguments, dict | list):
        return ("json", arguments)
    if isinstance(arguments, str):
        try:
            return ("json", json.loads(arguments))
        except json.JSONDecodeError:
            return ("raw", arguments)
    return ("raw", arguments)


class MessageCodec:
    """Model-scoped request codec used by gateway sessions and actors.

    ``_GatewayActor`` owns one codec per actor and injects it into
    ``GatewaySession`` instances. The codec normalizes OpenAI-shaped request
    payloads, renders chat templates, handles multimodal processor inputs, and
    decodes backend token outputs without reading session state.
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
        base_sampling_params: dict[str, Any] | None = None,
        allowed_request_sampling_param_keys: set[str] | frozenset[str] | None = None,
    ):
        self._tokenizer = tokenizer
        self._processor = processor
        self._vision_info_extractor = vision_info_extractor or self._default_vision_info_extractor
        self._vision_info_extractor_kwargs = dict(vision_info_extractor_kwargs or {})
        self._apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self._base_sampling_params = dict(base_sampling_params or {})
        self._allowed_request_sampling_param_keys = (
            _DEFAULT_ALLOWED_REQUEST_SAMPLING_PARAM_KEYS
            if allowed_request_sampling_param_keys is None
            else frozenset(allowed_request_sampling_param_keys)
        )
        self._system_prompt = initialize_system_prompt(
            tokenizer,
            **self._apply_chat_template_kwargs,
        )
        self._tool_parser = ToolParser.get_tool_parser(tool_parser_name, tokenizer) if tool_parser_name else None

    async def _default_vision_info_extractor(
        self,
        messages: list[dict[str, Any]],
        *,
        image_patch_size: int,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        # Keep the dataset dependency lazy so custom extractors do not pay for
        # RLHFDataset imports unless they actually use the default path.
        from verl.utils.dataset.rl_dataset import RLHFDataset

        return await RLHFDataset.process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            config=self._vision_info_extractor_kwargs.get("config"),
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

    def encode_full(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        request_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Encode a full chat history into prompt token IDs."""
        chat_template_kwargs = {**self._apply_chat_template_kwargs, **(request_chat_template_kwargs or {})}
        if self._processor is not None:
            raw_prompt = _apply_chat_template(
                self._processor,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                **chat_template_kwargs,
            )
            videos = video_data
            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos, strict=False)
                videos, video_metadata = list(videos), list(video_metadata)
            model_inputs = self._processor(
                text=[raw_prompt],
                images=image_data,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_sample_frames=False,
            )
            return normalize_token_ids(model_inputs["input_ids"])

        return normalize_token_ids(
            _apply_chat_template(
                self._tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        )

    # TODO: check if delta tokenization is better than remove_system_prompt
    def encode_incremental(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        request_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Encode continuation messages without the cached system prompt prefix."""
        chat_template_kwargs = {**self._apply_chat_template_kwargs, **(request_chat_template_kwargs or {})}
        if self._processor is not None:
            raw_prompt = _apply_chat_template(
                self._processor,
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **chat_template_kwargs,
            )
            videos = video_data
            video_metadata = None
            if videos is not None:
                videos, video_metadata = zip(*videos, strict=False)
                videos, video_metadata = list(videos), list(video_metadata)
            model_inputs = self._processor(
                text=[raw_prompt],
                images=image_data,
                videos=videos,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_sample_frames=False,
            )
            ids = normalize_token_ids(model_inputs["input_ids"])
        else:
            ids = normalize_token_ids(
                _apply_chat_template(
                    self._tokenizer,
                    messages,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            )
        return ids[len(self._system_prompt) :]

    async def decode_response(
        self,
        response_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop_reason: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Decode model output tokens into an assistant message and finish reason."""
        if self._tool_parser is not None and tools:
            parsed_tools = None
            try:
                from verl.tools.schemas import OpenAIFunctionToolSchema

                parsed_tools = [OpenAIFunctionToolSchema(**t) if isinstance(t, dict) else t for t in tools]
            except Exception:
                pass
            content, function_calls = await self._tool_parser.extract_tool_calls(response_ids, parsed_tools)
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

    def normalize_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize chat messages, tool schemas, and chat-template kwargs."""
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

        return {
            "messages": [_normalize_message(message) for message in messages],
            "tools": tools,
            "chat_template_kwargs": dict(chat_template_kwargs) if chat_template_kwargs else {},
        }

    def canonicalize_message_for_prefix_comparison(self, message: dict[str, Any]) -> dict[str, Any]:
        """Canonicalize one message before session prefix comparison."""
        normalized = dict(message)
        tool_calls = normalized.get("tool_calls")
        if not isinstance(tool_calls, list):
            return normalized

        normalized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            normalized_tool_call = dict(tool_call)
            function = normalized_tool_call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                normalized_function = dict(function)
                normalized_function["arguments"] = _canonicalize_tool_arguments_for_comparison(function["arguments"])
                normalized_tool_call["function"] = normalized_function
            normalized_tool_calls.append(normalized_tool_call)
        normalized["tool_calls"] = normalized_tool_calls
        return normalized

    def build_sampling_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Merge allowed per-request sampling params with codec defaults."""
        sampling_params = dict(self._base_sampling_params)
        for key in self._allowed_request_sampling_param_keys:
            if key in payload:
                sampling_params[key] = payload[key]
        return sampling_params
