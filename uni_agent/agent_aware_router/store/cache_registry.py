"""Shared cache layout for replicas with the same configuration."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from math import gcd, lcm
from typing import Literal

MatchKind = Literal["full", "sliding", "chunked", "mamba"]
_MATCH_KINDS: dict[str, MatchKind] = {
    "full_attention": "full",
    "mla_attention": "full",
    "sink_full_attention": "full",
    "sliding_window": "sliding",
    "sliding_window_mla": "sliding",
    "chunked_local_attention": "chunked",
    "mamba": "mamba",
}
_NON_PREFIX_KINDS = {"encoder_only_attention", "cross_attention"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheGroup:
    group_idx: int
    kind: str
    block_size: int
    match_kind: MatchKind | None
    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    prefix_cacheable: bool | None = None
    prefix_replay_tokens: int = 0

    @property
    def participates(self) -> bool:
        return self.match_kind is not None


@dataclass(frozen=True)
class CacheRegistry:
    groups: tuple[CacheGroup, ...]
    local_hash_unit: int
    vllm_version: str
    model: str | None
    mamba_cache_mode: str | None
    tensor_parallel_size: int
    kv_cache_capacity_tokens: int | None = None
    prefix_replay_tokens: int = 0

    @property
    def required_groups(self) -> tuple[CacheGroup, ...]:
        return tuple(group for group in self.groups if group.participates)

    def group(self, group_idx: int) -> CacheGroup:
        for group in self.groups:
            if group.group_idx == group_idx:
                return group
        raise KeyError(group_idx)


@dataclass(frozen=True)
class CacheRegistryState:
    status: str = "unavailable"
    registry: CacheRegistry | None = None
    reason: str | None = None


class _RegistryError(ValueError):
    def __init__(self, reason: str, status: str = "invalid"):
        super().__init__(reason)
        self.status = status


def _mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise _RegistryError(f"invalid_{name}")
    return value


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise _RegistryError(f"invalid_{name}")
    return value


def build_cache_registry(metadata) -> CacheRegistryState:
    """Build a cache registry from vLLM metadata."""
    try:
        raw = _mapping(metadata, "metadata")
        status = raw.get("status")
        if status != "ok":
            if not isinstance(status, str) or status not in {"not_ready", "headless", "unsupported", "error"}:
                raise _RegistryError("invalid_metadata_status")
            return CacheRegistryState(
                status="unavailable" if status == "not_ready" else status,
                reason=str(raw.get("reason") or f"metadata_{status}"),
            )
        if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
            raise _RegistryError("unsupported_metadata_schema", "unsupported")
        version = raw.get("vllm_version")
        if not isinstance(version, str) or not version:
            raise _RegistryError("missing_vllm_version")
        cache = _mapping(raw.get("cache_config"), "cache_config")
        if cache.get("enable_prefix_caching") is False:
            return CacheRegistryState(status="disabled", reason="prefix_caching_disabled")
        if cache.get("enable_prefix_caching") is not True:
            raise _RegistryError("missing_prefix_caching_config")
        parallel = _mapping(raw.get("parallel_config"), "parallel_config")
        tp = _positive_int(parallel.get("tensor_parallel_size"), "tensor_parallel_size")
        _positive_int(parallel.get("pipeline_parallel_size"), "pipeline_parallel_size")
        for key in (
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        ):
            if _positive_int(parallel.get(key), key) != 1:
                raise _RegistryError(f"unsupported_{key}", "unsupported")
        model = _mapping(raw.get("model_config"), "model_config")
        model_name = model.get("model")
        if model_name is not None and not isinstance(model_name, str):
            raise _RegistryError("invalid_model_name")
        mode = cache.get("mamba_cache_mode")
        if mode is not None and (not isinstance(mode, str) or mode not in {"align", "all", "none"}):
            raise _RegistryError("unsupported_mamba_cache_mode", "unsupported")
        groups_raw = raw.get("groups")
        if not isinstance(groups_raw, list) or not groups_raw:
            raise _RegistryError("missing_groups")
        groups = []
        inferred_auxiliary_groups = []
        seen = set()
        for item in groups_raw:
            item = _mapping(item, "group")
            idx = item.get("group_idx")
            if type(idx) is not int or idx < 0 or idx in seen:
                raise _RegistryError("invalid_or_duplicate_group_idx")
            seen.add(idx)
            kind = item.get("kind")
            if not isinstance(kind, str) or not kind:
                raise _RegistryError(f"missing_kind:group={idx}")
            block_size = _positive_int(item.get("block_size"), f"block_size:group={idx}")
            # Metadata without prefix policy fields uses kind-based matching.
            cacheable = item.get("prefix_cacheable")
            replay = item.get("prefix_replay_tokens", 0)
            if "prefix_cacheable" in item or "prefix_replay_tokens" in item:
                if type(cacheable) is not bool:
                    raise _RegistryError(f"invalid_prefix_cacheable:group={idx}")
                if "prefix_replay_tokens" not in item or type(replay) is not int or replay < 0:
                    raise _RegistryError(f"invalid_prefix_replay_tokens:group={idx}")
                if cacheable and replay:
                    raise _RegistryError(f"unsupported_cacheable_prefix_replay:group={idx}", "unsupported")
            elif kind == "unknown":
                cacheable = False
                inferred_auxiliary_groups.append(idx)
            excluded = cacheable is False or kind in _NON_PREFIX_KINDS
            match_kind = None if excluded else _MATCH_KINDS.get(kind)
            if not excluded and match_kind is None:
                raise _RegistryError(f"unsupported_kind:group={idx},kind={kind}", "unsupported")
            window = item.get("sliding_window")
            if window is not None or match_kind == "sliding":
                window = _positive_int(window, f"sliding_window:group={idx}")
            chunk = None
            if match_kind == "chunked":
                chunk = _positive_int(model.get("attention_chunk_size"), f"attention_chunk_size:group={idx}")
            groups.append(CacheGroup(idx, kind, block_size, match_kind, window, chunk, cacheable, replay))
        replay_windows = {group.prefix_replay_tokens for group in groups if group.prefix_replay_tokens}
        if len(replay_windows) > 1:
            raise _RegistryError("inconsistent_prefix_replay_tokens", "unsupported")
        required = [group for group in groups if group.participates]
        if not required:
            raise _RegistryError("no_prefix_cache_groups", "unsupported")
        if any(group.kind == "mamba" for group in groups) and mode == "all":
            # Mamba "all" mode aligns prefix matches to the LCM of all group sizes.
            unit = lcm(*(group.block_size for group in groups))
        else:
            unit = gcd(*(group.block_size for group in required))
            for key in ("prefix_match_unit", "hash_block_size"):
                configured = cache.get(key)
                if configured is not None:
                    configured = _positive_int(configured, key)
                    if any(group.block_size % configured for group in required):
                        raise _RegistryError(f"incompatible_{key}")
                    unit = gcd(unit, configured)
        capacity = raw.get("kv_cache_capacity_tokens")
        if type(capacity) is not int or capacity <= 0:
            capacity = None
        registry = CacheRegistry(
            tuple(sorted(groups, key=lambda g: g.group_idx)),
            unit,
            version,
            model_name,
            mode,
            tp,
            capacity,
            next(iter(replay_windows), 0),
        )
        if inferred_auxiliary_groups:
            logger.warning(
                "KV metadata has no prefix policy for unknown groups %s; treating them as non-cacheable",
                inferred_auxiliary_groups,
            )
        return CacheRegistryState(status="registered", registry=registry)
    except _RegistryError as exc:
        return CacheRegistryState(status=exc.status, reason=str(exc))
