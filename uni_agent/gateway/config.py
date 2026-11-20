"""Public config dataclass for GatewayActor wiring.

Carries model, codec, and session knobs that entry.py forwards to the
gateway actor. The backend client is injected separately by GatewayManager;
only its rollout name is forwarded here to select a matching tool parser.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any


def _concrete_ip(value: str, field: str) -> str:
    """Normalize an address that can be used both for listening and in a URL."""
    try:
        if not isinstance(value, str) or "%" in value:
            raise ValueError
        address = ip_address(value)
        if address.is_unspecified or address.is_multicast or address.is_link_local:
            raise ValueError
    except ValueError:
        raise ValueError(f"{field} must be a concrete, unscoped IPv4 or IPv6 address") from None
    return str(address)


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
        use_tree_trajectory_manager: Select TreeGatewaySession for shared-prefix
            trajectories when true. Defaults to false, using GatewaySession
            to manage linear chains directly.
        bind_address: Optional local IP used for both HTTP listening and session
            URLs. Defaults to the actor's Ray node IP. Mutually exclusive with
            bind_address_by_node.
        bind_address_by_node: Optional mapping from each Ray node IP to a local
            HTTP address on that node. A gateway's node must be in the mapping;
            entries are not shared defaults for other nodes.
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
    use_tree_trajectory_manager: bool = False
    bind_address: str | None = None
    bind_address_by_node: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.bind_address is not None and self.bind_address_by_node is not None:
            raise ValueError("bind_address and bind_address_by_node are mutually exclusive")
        if self.bind_address is not None:
            object.__setattr__(self, "bind_address", _concrete_ip(self.bind_address, "bind_address"))
        if self.bind_address_by_node is not None:
            if not isinstance(self.bind_address_by_node, Mapping) or not self.bind_address_by_node:
                raise ValueError("bind_address_by_node must be a nonempty mapping of node IPs to local IPs")
            addresses = {}
            for node, address in self.bind_address_by_node.items():
                node = _concrete_ip(node, "bind_address_by_node key")
                if node in addresses:
                    raise ValueError(f"bind_address_by_node has duplicate normalized node IP {node}")
                addresses[node] = _concrete_ip(address, "bind_address_by_node value")
            object.__setattr__(self, "bind_address_by_node", addresses)
        if type(self.use_tree_trajectory_manager) is not bool:
            raise ValueError("use_tree_trajectory_manager must be a bool")
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
        if self.prompt_length is not None and self.prompt_length <= 0:
            raise ValueError(f"prompt_length must be positive when set, got {self.prompt_length}")
        if self.response_length is not None and self.response_length <= 0:
            raise ValueError(f"response_length must be positive when set, got {self.response_length}")

    def resolve_bind_address(self, ray_node_address: str) -> str:
        """Select this actor's local HTTP address without changing Ray networking."""
        if self.bind_address_by_node is not None:
            node = _concrete_ip(ray_node_address.strip("[]"), "Ray node address")
            try:
                return self.bind_address_by_node[node]
            except KeyError:
                raise ValueError(f"bind_address_by_node has no address for Ray node {node}") from None
        return self.bind_address or ray_node_address.strip("[]")
