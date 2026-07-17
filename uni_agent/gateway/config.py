"""Public config dataclass for GatewayActor wiring.

Carries model, codec, and session knobs that entry.py forwards to the
gateway actor. Backend is NOT in this
config: it is injected separately by GatewayManager so the codec/
session boundary has no view of the LLM client lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GatewayActorConfig:
    """Model and session configuration forwarded into each gateway actor.

    Attributes:
        tokenizer: Tokenizer used by the message codec.
        processor: Optional multimodal processor used for vision requests.
        tool_parser_name: Optional VERL tool parser name for decoding tool calls.
        apply_chat_template_kwargs: Default kwargs passed to chat-template rendering.
        base_sampling_params: Sampling params applied before per-request overrides.
        allowed_request_sampling_param_keys: Request sampling keys accepted by the
            provider adapters when merging payload sampling params.
        vision_info_extractor: Optional async extractor for image/video inputs.
        vision_info_extractor_kwargs: Static kwargs forwarded to the extractor.
        prompt_length: Optional prompt-token budget stored on gateway sessions.
        response_length: Optional response-token budget stored on gateway sessions.
    """

    tokenizer: Any
    processor: Any | None = None
    tool_parser_name: str | None = None
    apply_chat_template_kwargs: dict[str, Any] | None = None
    base_sampling_params: dict[str, Any] | None = None
    allowed_request_sampling_param_keys: set[str] | frozenset[str] | None = None
    vision_info_extractor: Callable | None = None
    vision_info_extractor_kwargs: dict[str, Any] | None = None
    prompt_length: int | None = None
    response_length: int | None = None

    def __post_init__(self) -> None:
        if self.response_length is not None and self.response_length <= 0:
            raise ValueError(f"response_length must be positive when set, got {self.response_length}")
