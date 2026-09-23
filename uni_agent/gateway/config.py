"""Public config dataclass for GatewayActor wiring.

Carries model, codec, and session knobs that entry.py forwards to the
gateway actor. The backend client is injected separately by GatewayManager;
only its rollout name is forwarded here to select a matching tool parser.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KVCacheHintConfig:
    """Agent KV hint settings; these do not enable backend CPU offloading.

    Static mode uses priority for requests without tools and tool_priority for
    requests with tools. Dynamic mode computes priority from trajectory state.
    lease_seconds sets the lifetime of each refreshed hint in either mode.
    """

    enabled: bool = False
    priority_mode: str = "static"
    lease_seconds: float = 300.0
    priority: int = 50
    tool_priority: int = 90

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if self.priority_mode not in ("static", "dynamic"):
            raise ValueError("priority_mode must be 'static' or 'dynamic'")
        if (
            isinstance(self.lease_seconds, bool)
            or not isinstance(self.lease_seconds, int | float)
            or not math.isfinite(self.lease_seconds)
            or self.lease_seconds <= 0
        ):
            raise ValueError("lease_seconds must be finite and positive")
        for name, value in (
            ("priority", self.priority),
            ("tool_priority", self.tool_priority),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                raise ValueError(f"{name} must be an integer between 0 and 100")


@dataclass(frozen=True)
class GatewayActorConfig:
    """Model and session configuration forwarded into each gateway actor.

    Attributes:
        tokenizer: Tokenizer used by the message codec.
        processor: Optional multimodal processor used for vision requests.
        tool_parser_name: Optional tool parser name for decoding tool calls.
        rollout_backend: Inference backend name used to select the matching
            SGLang or vLLM parser. Other backends use verl's parser registry.
        enable_tool_parser_cache: Whether to reuse parser instances within an
            actor-scoped codec. Disable for parsers that require request-scoped
            instances; enabled by default for the parser-construction speedup.
        hf_model_type: Root Hugging Face ``config.json`` model type used to
            select the Continuous Token builder.
        apply_chat_template_kwargs: Default kwargs passed to chat-template rendering.
        mm_processor_kwargs: Static multimodal processor kwargs used by the
            Continuous Token builder.
        allowed_request_sampling_param_keys: Extra request sampling keys accepted by
            provider adapters in addition to the default max_tokens and stop.
            None or an empty set keeps those defaults; this setting cannot remove them.
        vision_info_extractor: Optional async extractor for image/video inputs.
        vision_info_extractor_kwargs: Static kwargs forwarded to the extractor.
        prompt_length: Optional prompt component of the total trajectory capacity.
        response_length: Optional response component of the total trajectory capacity.
            The gateway enforces their sum when both values are set.
        enable_last_assistant_rollback: Whether latest-assistant rewrites may
            rollback and reuse an existing chain. Enabled by default.
        kv_cache_offload_config: Immutable agent KV hint settings.
        coalesce_reserved_exact_requests: Whether exact provider-normalized
            requests in the same session share an in-flight result, including
            first-turn and new-chain requests. Enabled by default; disable for
            independent concurrent sampling of identical requests.
    """

    tokenizer: Any
    processor: Any | None = None
    tool_parser_name: str | None = None
    rollout_backend: str | None = None
    enable_tool_parser_cache: bool = True
    hf_model_type: str | None = None
    apply_chat_template_kwargs: dict[str, Any] | None = None
    mm_processor_kwargs: dict[str, Any] | None = None
    allowed_request_sampling_param_keys: set[str] | frozenset[str] | None = None
    vision_info_extractor: Callable | None = None
    vision_info_extractor_kwargs: dict[str, Any] | None = None
    prompt_length: int | None = None
    response_length: int | None = None
    enable_last_assistant_rollback: bool = True
    kv_cache_offload_config: KVCacheHintConfig = field(default_factory=KVCacheHintConfig)
    coalesce_reserved_exact_requests: bool = True

    def __post_init__(self) -> None:
        if type(self.enable_tool_parser_cache) is not bool:
            raise ValueError(
                f"enable_tool_parser_cache must be a bool, got {type(self.enable_tool_parser_cache).__name__}"
            )
        if type(self.enable_last_assistant_rollback) is not bool:
            raise ValueError(
                "enable_last_assistant_rollback must be a bool, "
                f"got {type(self.enable_last_assistant_rollback).__name__}"
            )
        if not isinstance(self.kv_cache_offload_config, KVCacheHintConfig):
            raise ValueError("kv_cache_offload_config must be a KVCacheHintConfig")
        if type(self.coalesce_reserved_exact_requests) is not bool:
            raise ValueError(
                "coalesce_reserved_exact_requests must be a bool, "
                f"got {type(self.coalesce_reserved_exact_requests).__name__}"
            )
        if self.prompt_length is not None and self.prompt_length <= 0:
            raise ValueError(f"prompt_length must be positive when set, got {self.prompt_length}")
        if self.response_length is not None and self.response_length <= 0:
            raise ValueError(f"response_length must be positive when set, got {self.response_length}")
