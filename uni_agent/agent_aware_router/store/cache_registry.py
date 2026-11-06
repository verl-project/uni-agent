"""Shared cache layout for replicas with the same configuration."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import ceil, gcd, lcm
from typing import Literal

from ..types.kv_cache import GroupReservationSpec

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
class CacheGroupSpec:
    """Immutable layout and prefix-matching policy for one cache group."""

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
    groups: tuple[CacheGroupSpec, ...]
    local_hash_unit: int
    vllm_version: str
    model: str | None
    mamba_cache_mode: str | None
    tensor_parallel_size: int
    kv_cache_capacity_tokens: int | None = None
    prefix_replay_tokens: int = 0
    reservation_groups: tuple[GroupReservationSpec, ...] = ()
    pool_blocks: int | None = None

    @property
    def required_groups(self) -> tuple[CacheGroupSpec, ...]:
        return tuple(group for group in self.groups if group.participates)

    @property
    def reservation_specs(self) -> tuple[GroupReservationSpec, ...]:
        return self.reservation_groups or _reservation_specs(self.groups, self.mamba_cache_mode, None)[0]

    def group(self, group_idx: int) -> CacheGroupSpec:
        for group in self.groups:
            if group.group_idx == group_idx:
                return group
        raise KeyError(group_idx)

    def normalize_event(self, event):
        """Fill old event fields only when this layout identifies one group."""
        if event.is_clear:
            return event
        group_idx = event.group_idx
        if group_idx is None and len(self.groups) == 1:
            group_idx = self.groups[0].group_idx
        try:
            group = self.group(group_idx)
        except KeyError:
            return event  # Ambiguous and unknown groups remain unsupported.
        tokens = event.raw_token_ids
        if (
            event.is_store
            and event.group_idx is None
            and event.kind is None
            and group.match_kind == "full"
            and tokens is not None
            and len(event.block_hashes) == len(tokens) // event.block_size
        ):
            # Old dense Full events can include an uncached incomplete tail.
            # Sparse/masked events need their entire token range for identity.
            tokens = tokens[: len(event.block_hashes) * event.block_size]
        return replace(
            event,
            group_idx=group_idx,
            kind=group.kind if event.kind is None else event.kind,
            sliding_window=group.sliding_window if event.sliding_window is None else event.sliding_window,
            raw_token_ids=tokens,
        )


@dataclass(frozen=True)
class CacheRegistryState:
    status: str = "unavailable"
    registry: CacheRegistry | None = None
    reason: str | None = None
    inferred: bool = False


def legacy_full_registry(block_size: int) -> CacheRegistryState:
    """Adapt an untagged old event source under its single-Full assumption.

    An event cannot prove the engine's topology. This compatibility contract
    applies only while metadata is unavailable and events omit group and kind.
    """
    registry = CacheRegistry(
        groups=(CacheGroupSpec(0, "full_attention", block_size, "full"),),
        local_hash_unit=block_size,
        vllm_version="unknown",
        model=None,
        mamba_cache_mode=None,
        tensor_parallel_size=1,
    )
    return CacheRegistryState("registered", registry, "legacy_single_full_assumption", inferred=True)


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


def _parse_parallel_layout(raw) -> int:
    """Validate supported parallel dimensions and return tensor parallel size."""
    parallel = _mapping(raw, "parallel_config")
    tp = _positive_int(parallel.get("tensor_parallel_size"), "tensor_parallel_size")
    _positive_int(parallel.get("pipeline_parallel_size"), "pipeline_parallel_size")
    for key in (
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if _positive_int(parallel.get(key), key) != 1:
            raise _RegistryError(f"unsupported_{key}", "unsupported")
    return tp


def _parse_model_layout(raw, cache: Mapping) -> tuple[Mapping, str | None, str | None]:
    """Validate model identity and cache mode before interpreting group policies."""
    model = _mapping(raw, "model_config")
    name = model.get("model")
    if name is not None and not isinstance(name, str):
        raise _RegistryError("invalid_model_name")
    mode = cache.get("mamba_cache_mode")
    if mode is not None and (not isinstance(mode, str) or mode not in {"align", "all", "none"}):
        raise _RegistryError("unsupported_mamba_cache_mode", "unsupported")
    return model, name, mode


def _parse_group_prefix_policy(item: Mapping, idx: int, kind: str) -> tuple[bool | None, int, bool]:
    """Resolve explicit policy or the legacy non-cacheable unknown-group fallback."""
    cacheable = item.get("prefix_cacheable")
    replay = item.get("prefix_replay_tokens", 0)
    inferred_auxiliary = False
    if "prefix_cacheable" in item or "prefix_replay_tokens" in item:
        if type(cacheable) is not bool:
            raise _RegistryError(f"invalid_prefix_cacheable:group={idx}")
        if "prefix_replay_tokens" not in item or type(replay) is not int or replay < 0:
            raise _RegistryError(f"invalid_prefix_replay_tokens:group={idx}")
        if cacheable and replay:
            raise _RegistryError(f"unsupported_cacheable_prefix_replay:group={idx}", "unsupported")
    elif kind == "unknown":
        cacheable = False
        inferred_auxiliary = True
    return cacheable, replay, inferred_auxiliary


def _parse_group_spec(item: Mapping, idx: int, model: Mapping) -> tuple[CacheGroupSpec, bool]:
    """Parse one group after its identifier has passed collection-level checks."""
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind:
        raise _RegistryError(f"missing_kind:group={idx}")
    block_size = _positive_int(item.get("block_size"), f"block_size:group={idx}")
    cacheable, replay, inferred_auxiliary = _parse_group_prefix_policy(item, idx, kind)
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
    return CacheGroupSpec(idx, kind, block_size, match_kind, window, chunk, cacheable, replay), inferred_auxiliary


def _parse_group_specs(raw, model: Mapping) -> tuple[list[CacheGroupSpec], list[int]]:
    """Validate unique identifiers and retain input order for group validation."""
    if not isinstance(raw, list) or not raw:
        raise _RegistryError("missing_groups")
    groups = []
    inferred_auxiliary_groups = []
    seen = set()
    for item in raw:
        item = _mapping(item, "group")
        idx = item.get("group_idx")
        if type(idx) is not int or idx < 0 or idx in seen:
            raise _RegistryError("invalid_or_duplicate_group_idx")
        seen.add(idx)
        group, inferred_auxiliary = _parse_group_spec(item, idx, model)
        groups.append(group)
        if inferred_auxiliary:
            inferred_auxiliary_groups.append(idx)
    return groups, inferred_auxiliary_groups


def _resolve_prefix_layout(groups: list[CacheGroupSpec], cache: Mapping, mode: str | None) -> tuple[int, int]:
    """Validate cross-group prefix policy and resolve its hash unit and replay window."""
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
    return unit, next(iter(replay_windows), 0)


def _reservation_profile(profile, group_ids):
    """Optional budget failure leaves the layout usable with unknown private costs."""
    if not isinstance(profile, Mapping) or type(profile.get("schema_version")) is not int:
        return {}, None, 0
    raw_groups = profile.get("groups")
    max_len = profile.get("max_model_len")
    if profile["schema_version"] != 1 or not isinstance(raw_groups, list) or type(max_len) is not int or max_len <= 0:
        return {}, None, 0
    budgets = {}
    for item in raw_groups:
        if not isinstance(item, Mapping):
            return {}, None, 0
        idx, count = item.get("group_idx"), item.get("max_blocks_per_request")
        if type(idx) is not int or idx in budgets or type(count) is not int or count < 0:
            return {}, None, 0
        budgets[idx] = count
    if budgets.keys() != group_ids:
        return {}, None, 0
    pool = profile.get("num_blocks")
    return budgets, pool if type(pool) is int and pool > 0 else None, max_len


def _reservation_specs(groups, mode, profile):
    """Keep complete-prefix reservations; obtain private-state budgets natively."""
    budgets, pool, max_len = _reservation_profile(profile, {group.group_idx for group in groups})
    specs = []
    for group in groups:
        shared = group.participates and (group.match_kind != "mamba" or mode == "all")
        if shared:
            peak = budgets.get(group.group_idx)
            if peak is not None and max_len:
                extra = max(0, peak - ceil(max_len / group.block_size))
            else:
                extra = None if group.match_kind == "mamba" else 0
        else:
            extra = budgets.get(group.group_idx, 0 if group.kind == "encoder_only_attention" else None)
        specs.append(GroupReservationSpec(group.group_idx, group.block_size, shared, extra))
    return tuple(specs), pool


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
        tp = _parse_parallel_layout(raw.get("parallel_config"))
        model, model_name, mode = _parse_model_layout(raw.get("model_config"), cache)
        groups, inferred_auxiliary_groups = _parse_group_specs(raw.get("groups"), model)
        unit, replay = _resolve_prefix_layout(groups, cache, mode)
        capacity = raw.get("kv_cache_capacity_tokens")
        if type(capacity) is not int or capacity <= 0:
            capacity = None
        reservation_specs, pool_blocks = _reservation_specs(groups, mode, raw.get("cache_reservation_profile"))
        registry = CacheRegistry(
            tuple(sorted(groups, key=lambda g: g.group_idx)),
            unit,
            version,
            model_name,
            mode,
            tp,
            capacity,
            replay,
            reservation_specs,
            pool_blocks,
        )
        if inferred_auxiliary_groups:
            logger.warning(
                "KV metadata has no prefix policy for unknown groups %s; treating them as non-cacheable",
                inferred_auxiliary_groups,
            )
        return CacheRegistryState(status="registered", registry=registry)
    except _RegistryError as exc:
        return CacheRegistryState(status=exc.status, reason=str(exc))
