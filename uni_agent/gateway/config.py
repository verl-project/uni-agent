"""Public config dataclass for GatewayActor wiring.

Carries model, codec, and session knobs that entry.py forwards to the
gateway actor. The backend client is injected separately by GatewayManager;
only its rollout name is forwarded here to select a matching tool parser.
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
        coalesce_reserved_exact_requests: Whether exact provider-normalized
            requests in the same session share an in-flight result, including
            first-turn and new-chain requests. Enabled by default; disable for
            independent concurrent sampling of identical requests.
        task_metrics_mode: Local Task Metrics mode. ``off`` preserves the legacy
            path; ``shadow`` and ``primary`` enable local fact projection.
        direct_state_sync_enabled: Whether Gateway lifecycle state is mirrored to
            the Router through the recoverable Direct path.
        direct_state_sync_max_queue_events: Maximum pending lifecycle events per Gateway source.
        direct_state_sync_max_queue_bytes: Maximum pending lifecycle bytes per Gateway source.
        direct_state_sync_max_retries: Consecutive delivery attempts before the view becomes stale.
        direct_state_sync_retry_backoff_s: Delay between retryable delivery attempts.
        direct_state_sync_max_terminal_entities: Bounded terminal-session snapshot retention.
        global_telemetry_enabled: Whether Gateway facts are sent through the Global telemetry runtime.
        global_telemetry_event_types: Existing event families selected for Global forwarding.
        global_telemetry_max_queue_events: Maximum pending telemetry events per Gateway source.
        global_telemetry_max_queue_bytes: Maximum pending telemetry bytes per Gateway source.
        global_telemetry_max_batch_events: Maximum events in one telemetry batch.
        global_telemetry_max_batch_bytes: Maximum bytes in one telemetry batch.
        global_telemetry_flush_interval_s: Maximum source-side batching delay.
        global_telemetry_max_retries: Delivery attempts before an observation batch is incomplete.
        global_telemetry_retry_backoff_s: Delay between retryable telemetry delivery attempts.
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
    coalesce_reserved_exact_requests: bool = True
    task_metrics_mode: str = "off"
    direct_state_sync_enabled: bool = False
    direct_state_sync_max_queue_events: int = 1024
    direct_state_sync_max_queue_bytes: int = 1024 * 1024
    direct_state_sync_max_retries: int = 3
    direct_state_sync_retry_backoff_s: float = 0.01
    direct_state_sync_max_terminal_entities: int = 4096
    global_telemetry_enabled: bool = False
    global_telemetry_event_types: tuple[str, ...] = (
        "SessionOpened",
        "GenerationFinished",
        "SessionClosed",
    )
    global_telemetry_max_queue_events: int = 4096
    global_telemetry_max_queue_bytes: int = 8 * 1024 * 1024
    global_telemetry_max_batch_events: int = 128
    global_telemetry_max_batch_bytes: int = 256 * 1024
    global_telemetry_flush_interval_s: float = 0.02
    global_telemetry_max_retries: int = 3
    global_telemetry_retry_backoff_s: float = 0.01

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
        if type(self.coalesce_reserved_exact_requests) is not bool:
            raise ValueError(
                "coalesce_reserved_exact_requests must be a bool, "
                f"got {type(self.coalesce_reserved_exact_requests).__name__}"
            )
        if not isinstance(self.task_metrics_mode, str) or self.task_metrics_mode not in {
            "off",
            "shadow",
            "primary",
        }:
            raise ValueError(
                f"task_metrics_mode must be one of 'off', 'shadow', or 'primary', got {self.task_metrics_mode!r}"
            )
        if type(self.direct_state_sync_enabled) is not bool:
            raise ValueError(
                f"direct_state_sync_enabled must be a bool, got {type(self.direct_state_sync_enabled).__name__}"
            )
        if self.direct_state_sync_max_queue_events <= 0:
            raise ValueError("direct_state_sync_max_queue_events must be positive")
        if self.direct_state_sync_max_queue_bytes <= 0:
            raise ValueError("direct_state_sync_max_queue_bytes must be positive")
        if self.direct_state_sync_max_retries <= 0:
            raise ValueError("direct_state_sync_max_retries must be positive")
        if self.direct_state_sync_retry_backoff_s < 0:
            raise ValueError("direct_state_sync_retry_backoff_s must be non-negative")
        if self.direct_state_sync_max_terminal_entities <= 0:
            raise ValueError("direct_state_sync_max_terminal_entities must be positive")
        if type(self.global_telemetry_enabled) is not bool:
            raise ValueError(
                f"global_telemetry_enabled must be a bool, got {type(self.global_telemetry_enabled).__name__}"
            )
        if isinstance(self.global_telemetry_event_types, str):
            raise ValueError("global_telemetry_event_types must be a sequence of event names")
        event_types = tuple(dict.fromkeys(self.global_telemetry_event_types))
        if not event_types or any(not isinstance(event_type, str) or not event_type for event_type in event_types):
            raise ValueError("global_telemetry_event_types must contain non-empty event names")
        object.__setattr__(self, "global_telemetry_event_types", event_types)
        if (
            min(
                self.global_telemetry_max_queue_events,
                self.global_telemetry_max_queue_bytes,
                self.global_telemetry_max_batch_events,
                self.global_telemetry_max_batch_bytes,
                self.global_telemetry_max_retries,
            )
            <= 0
        ):
            raise ValueError("Global telemetry queue, batch, and retry budgets must be positive")
        if self.global_telemetry_flush_interval_s < 0:
            raise ValueError("global_telemetry_flush_interval_s must be non-negative")
        if self.global_telemetry_retry_backoff_s < 0:
            raise ValueError("global_telemetry_retry_backoff_s must be non-negative")
        if self.prompt_length is not None and self.prompt_length <= 0:
            raise ValueError(f"prompt_length must be positive when set, got {self.prompt_length}")
        if self.response_length is not None and self.response_length <= 0:
            raise ValueError(f"response_length must be positive when set, got {self.response_length}")
