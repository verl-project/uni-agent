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
            codec when merging payload sampling params.
        vision_info_extractor: Optional async extractor for image/video inputs.
        vision_info_extractor_kwargs: Static kwargs forwarded to the extractor.
        prompt_length: Optional prompt-token budget stored on gateway sessions.
        response_length: Optional response-token budget stored on gateway sessions.
        enable_parallel_session_generation: Allow concurrent backend generation
            calls within one gateway session while serializing prepare/commit.
        ignore_cch_for_prefix_hash: Treat Claude Code ``cch=...`` hash churn as
            non-semantic only for prefix-hash comparison.
    """

    tokenizer: Any
    processor: Any | None = None
    tool_parser_name: str | None = None
    apply_chat_template_kwargs: dict[str, Any] | None = None
    base_sampling_params: dict[str, Any] | None = None
    allowed_request_sampling_param_keys: frozenset[str] | None = None
    vision_info_extractor: Callable | None = None
    vision_info_extractor_kwargs: dict[str, Any] | None = None
    prompt_length: int | None = None
    response_length: int | None = None
    enable_parallel_session_generation: bool = False
    ignore_cch_for_prefix_hash: bool = False

    def __post_init__(self) -> None:
        for name in ("enable_parallel_session_generation", "ignore_cch_for_prefix_hash"):
            value = getattr(self, name)
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool, got {type(value).__name__}")
