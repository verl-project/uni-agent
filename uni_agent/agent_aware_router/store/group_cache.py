"""Replica-local event endpoints and four prefix-cache matching rules.

The caller (KVCacheStore) serializes writes and queries. Intermediate local
hashes are temporary; only reported remote endpoints become residency entries.
Per-group dispatch hints are tracked separately from reported residency.
"""

from __future__ import annotations

import logging
import time
from collections import ChainMap, Counter, OrderedDict, defaultdict
from dataclasses import dataclass, replace
from itertools import islice
from math import lcm
from typing import TYPE_CHECKING, Literal, NamedTuple

from ..types.kv_cache import LOCAL_GPU_DOMAIN
from ..utils.hash import get_prefix_hashes_incremental
from .cache_registry import CacheGroupSpec, CacheRegistry
from .prefix_identity import PrefixIdentityMismatch

if TYPE_CHECKING:
    from ..types.kv_cache import KVCacheEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheEndpoint:
    local_hash: str
    tokens: int


class _PendingStore(NamedTuple):
    enqueued_at: float
    event: KVCacheEvent
    retained: Counter[str]


@dataclass(frozen=True)
class PrefixEstimate:
    status: str
    estimated_hit_tokens: int
    query_tokens: int
    lookup_limit_tokens: int
    local_hash_unit: int | None
    reason: str | None = None
    replayed_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        return self.estimated_hit_tokens / self.query_tokens if self.query_tokens else 0.0

    @property
    def estimated_saved_tokens(self) -> int:
        return max(0, self.estimated_hit_tokens - self.replayed_tokens)

    @property
    def saved_token_rate(self) -> float:
        return self.estimated_saved_tokens / self.query_tokens if self.query_tokens else 0.0


class ReplicaCacheIndex:
    """Runtime cache index for all cache groups on one replica.

    Tracks event endpoints, pending dependencies, and prefix estimates.
    """

    def __init__(
        self,
        registry: CacheRegistry,
        replica_id: str,
        *,
        pending_limit=1024,
        pending_ttl=60.0,
        history_limit=8192,
        native_hashes=None,
    ):
        self.registry = registry
        self.native_hashes = native_hashes
        self.native_reconstruction_paused = False
        self._rejected_native_events: OrderedDict[tuple, None] = OrderedDict()
        self._restored_revision = -1
        self.replica_id = replica_id
        self.epoch = 0
        self.anchors: OrderedDict[str, CacheEndpoint] = OrderedDict()
        self.dispatch_hints: dict[int, OrderedDict[str, int]] = {}
        # group, remote hash -> actual endpoint
        self.entries: dict[tuple, CacheEndpoint] = {}
        # Incremental Stored/Removed counts, independent of hash resolution.
        # Partial aliases may share physical blocks.
        self.retained_hashes: Counter[tuple[int, str]] = Counter()
        self.by_group: dict[int, dict[str, set[str]]] = defaultdict(dict)
        self.references: Counter[str] = Counter()
        self.pending: list[_PendingStore] = []
        self.pending_limit = pending_limit
        self.pending_ttl = pending_ttl
        self.history_limit = history_limit
        self.status = "warming"
        self.reason: str | None = "no_events"
        self.diagnostics: Counter[str] = Counter()
        self._small_events: set[tuple[int, int]] = set()
        self._identity_revision = 0
        self._retried_revision: int | None = native_hashes.identity_revision if native_hashes else 0

    def clear(self) -> None:
        self.epoch += 1
        self.anchors.clear()
        self.dispatch_hints.clear()
        self.entries.clear()
        self.retained_hashes.clear()
        self.by_group.clear()
        self.references.clear()
        self.pending.clear()
        self._restored_revision = -1
        self._small_events.clear()
        self._identity_revision = 0
        self._retried_revision = self.native_hashes.identity_revision if self.native_hashes else 0
        self.status, self.reason = "ready", None

    def record_dispatch(self, hashes: list[str]) -> int:
        """Predict expressible block endpoints from the canonical local hash chain.

        Predictions use each group's block boundaries and matching rules, but
        never establish remote identities or increment reported residency.
        """
        if self.status == "unsupported":
            return 0
        added = sum(self._record_group_dispatch(group, hashes) for group in self.registry.required_groups)
        if any(self.dispatch_hints.values()) and self.status == "warming":
            self.status, self.reason = "ready", None
        return added

    def _record_group_dispatch(self, group, hashes):
        reported = self.by_group.get(group.group_idx, {})
        hints = self.dispatch_hints.setdefault(group.group_idx, OrderedDict())
        unit = self.registry.local_hash_unit
        added = 0
        for i, local in enumerate(hashes):
            end = (i + 1) * unit
            if end % group.block_size == 0 and local not in reported:
                added += local not in hints
                hints[local] = end
                hints.move_to_end(local)
        # Bound predictions independently of native residency for every group.
        while len(hints) > self.history_limit:
            hints.popitem(last=False)
        return added

    def _invalidate_dispatch_hints(self, group, remote_hashes):
        hints = self.dispatch_hints.get(group)
        if not hints:
            return
        for remote in remote_hashes:
            endpoint = self.anchors.get(remote)
            if endpoint is None:
                # The identity is unknown, so any prediction in this group
                # might have been removed. Other groups remain independent.
                hints.clear()
                return
            hints.pop(endpoint.local_hash, None)

    def _matching_index(self, group):
        reported = self.by_group.get(group.group_idx, {})
        hints = self.dispatch_hints.get(group.group_idx)
        if hints:
            return ChainMap(reported, hints)
        return reported

    def _reject_event(self, reason: str) -> None:
        """Consume an uninterpretable event without erasing known residency."""
        self.diagnostics[reason] += 1
        if self.diagnostics[reason] == 1:
            logger.warning("Skipping KV event: replica=%s epoch=%s reason=%s", self.replica_id, self.epoch, reason)

    def apply(self, events: list[KVCacheEvent]) -> None:
        # Drop stale waiting references before new events or peer identities can resolve them.
        self._expire_pending()
        # Share token hashing between group events within this received batch.
        chains: dict[tuple, dict[int, str]] = {}
        for event in events:
            if event.is_clear:
                self.clear()
                chains.clear()
                continue
            self._apply_event(event, chains)
        self._restore_identities()
        self._retry(chains)
        self._prune_anchors()

    def _apply_event(self, event: KVCacheEvent, chains: dict) -> None:
        """Filter and dispatch one non-Clear event, preserving its state transition."""
        domain = event.storage_domain
        if domain is None:
            self._reject_event("missing_or_unknown_medium")
            return
        if domain != LOCAL_GPU_DOMAIN:
            # These sources cannot establish identities for local GPU
            # events either. Their removal cannot delete local entries.
            self.diagnostics["excluded_storage_domain"] += 1
            return
        try:
            group = self.registry.group(event.group_idx)
        except KeyError:
            reason = f"unregistered_group:{event.group_idx}"
            self._reject_event(reason)
            # An unknown required group prevents a complete hybrid estimate,
            # but known groups can still process their Stored/Removed events.
            self.status, self.reason = "unsupported", reason
            return
        if not group.participates:
            return
        if event.is_remove:
            self._invalidate_dispatch_hints(group.group_idx, event.block_hashes)
            self._remove(group.group_idx, event.block_hashes)
        elif event.is_store:
            if not self._accept_stored_event(event, group):
                return
            resolved = self._store(event, chains)
            self._record_stored_result(event, group.group_idx, resolved)
        if self.status == "warming":
            self.status, self.reason = "ready", None

    def _accept_stored_event(self, event: KVCacheEvent, group: CacheGroupSpec) -> bool:
        """Check Stored identity eligibility and diagnose supported partial events."""
        if event.kind != group.kind or (group.match_kind == "sliding" and event.sliding_window != group.sliding_window):
            reason = f"group_metadata_mismatch:{group.group_idx}"
            self._reject_event(reason)
            if event.kind == "unknown":
                self.status, self.reason = "unsupported", reason
            return False
        lookahead = self.registry.lookahead_tokens
        rows = event.lookahead_token_ids
        if rows is not None and any(len(row) != lookahead for row in rows):
            self.status, self.reason = "unsupported", "lookahead_length_mismatch"
            self._reject_event(self.reason)
            return False
        if lookahead and rows is None:
            self._reject_event("missing_event_lookahead")
            return False
        if event.lora_id not in (None, 0) or event.lora_name or any(event.extra_keys or ()):
            self.diagnostics["excluded_cache_identity"] += 1
            return False
        if event.block_size < group.block_size:
            self.diagnostics["smaller_event_block_size"] += 1
            warning_key = (group.group_idx, event.block_size)
            if warning_key not in self._small_events:
                self._small_events.add(warning_key)
                logger.warning(
                    "Smaller KV event block size: replica=%s epoch=%s group=%s group_block_size=%s "
                    "event_block_size=%s local_hash_unit=%s",
                    self.replica_id,
                    self.epoch,
                    group.group_idx,
                    group.block_size,
                    event.block_size,
                    self.registry.local_hash_unit,
                )
        return True

    def _record_stored_result(self, event: KVCacheEvent, group_idx: int, resolved: bool | None) -> None:
        """Count one arrival and defer unresolved events when the layout is supported.

        Retries resolve previously counted arrivals and must not call this method.
        """
        if resolved is not None:
            self.retained_hashes.update((group_idx, remote) for remote in event.block_hashes)
        if resolved is not False:
            return
        if self.status == "unsupported":
            return
        for pending in self.pending:
            if pending.event == event:
                pending.retained.update(event.block_hashes)
                break
        else:
            self.pending.append(_PendingStore(time.monotonic(), event, Counter(event.block_hashes)))
        excess = len(self.pending) - self.pending_limit
        if excess > 0:
            self._drop_pending(excess, "pending_overflow")

    def _store(self, event: KVCacheEvent, chains: dict, evicted: frozenset[str] = frozenset()) -> bool | None:
        """Return True if resolved, False if waiting for identities, None if rejected."""
        if not event.block_hashes:
            return True
        if event.raw_token_ids is None:
            return self._reject_event("missing_tokens")
        if evicted and len(event.raw_token_ids) != len(event.block_hashes) * event.block_size:
            # Sparse events locate each surviving hash by identity. A removed
            # unknown hash must no longer block the other endpoints.
            rows = event.lookahead_token_ids
            kept = [i for i, remote in enumerate(event.block_hashes) if remote not in evicted]
            event = replace(
                event,
                block_hashes=[event.block_hashes[i] for i in kept],
                lookahead_token_ids=tuple(rows[i] for i in kept) if rows is not None else None,
            )
        signature = (
            event.group_idx,
            tuple(event.block_hashes),
            event.parent_block_hash,
            event.raw_token_ids,
            event.lookahead_token_ids,
            event.block_size,
        )
        if signature in self._rejected_native_events:
            return None
        try:
            resolved = self._resolve_native_endpoints(event)
        except PrefixIdentityMismatch:
            self.native_reconstruction_paused = True
            self._rejected_native_events[signature] = None
            while len(self._rejected_native_events) > self.history_limit:
                self._rejected_native_events.popitem(last=False)
            return self._reject_event("native_hash_mismatch")
        native_resolved = resolved is not None
        if not native_resolved:
            resolved = self._resolve_local_endpoints(event, chains)
        if resolved is False or resolved is None:
            return resolved
        # Resolvers validate the complete event before any residency is changed.
        for remote, endpoint in resolved:
            if remote not in evicted:
                self._put(remote, endpoint, event.group_idx)
        if native_resolved:
            self.diagnostics["native_events_resolved"] += 1
        return True

    def _resolve_native_endpoints(self, event: KVCacheEvent) -> list[tuple[str, CacheEndpoint]] | None:
        """Return verified native endpoints, or None to use local resolution."""
        if self.native_hashes is None or self.native_reconstruction_paused:
            return None
        identities = self.native_hashes.resolve_event(
            event,
            allow_reconstruct=self.registry.group(event.group_idx).match_kind == "full",
            lookahead=self.registry.lookahead_tokens,
        )
        if identities is None:
            return None
        return [
            (remote, CacheEndpoint(identity.local_hash, identity.tokens))
            for remote, identity in zip(event.block_hashes, identities, strict=True)
        ]

    def _resolve_local_endpoints(
        self, event: KVCacheEvent, chains: dict
    ) -> list[tuple[str, CacheEndpoint]] | Literal[False] | None:
        """Resolve token endpoints; False waits for dependencies and None rejects."""
        unit = self.registry.local_hash_unit
        parent = CacheEndpoint("0", 0)
        if event.parent_block_hash is not None:
            parent = self.anchors.get(event.parent_block_hash)
            if parent is None:
                return False
        tokens = event.raw_token_ids
        if not tokens or len(tokens) % unit or parent.tokens % unit:
            return self._reject_event("unrepresentable_event_boundary")
        dense = len(tokens) == len(event.block_hashes) * event.block_size
        if not dense and any(remote not in self.anchors for remote in event.block_hashes):
            return False
        lookahead = self.registry.lookahead_tokens
        if lookahead:
            ends = (
                range(event.block_size, len(tokens) + 1, event.block_size)
                if dense
                else [self.anchors[remote].tokens - parent.tokens for remote in event.block_hashes]
            )
            tokens = event.hash_input_tokens(ends, lookahead)
            if tokens is None:
                return self._reject_event("inconsistent_lookahead_tokens")
        chain = self._local_hash_chain(parent, tokens, chains)
        return self._event_endpoints(event, parent, chain, dense)

    def _local_hash_chain(self, parent: CacheEndpoint, tokens, chains: dict) -> dict[int, str]:
        """Reuse token hashing across group events within the current batch."""
        unit = self.registry.local_hash_unit
        key = (parent, tokens)
        chain = chains.get(key)
        if chain is None:
            values, _ = get_prefix_hashes_incremental(
                tokens, unit, int(parent.local_hash), 0, lookahead=self.registry.lookahead_tokens
            )
            chain = {parent.tokens + i * unit: str(value) for i, value in enumerate(values, start=1)}
            chains[key] = chain
        return chain

    def _event_endpoints(
        self, event: KVCacheEvent, parent: CacheEndpoint, chain: dict[int, str], dense: bool
    ) -> list[tuple[str, CacheEndpoint]] | None:
        """Validate dense positions or known sparse positions without writing residency."""
        resolved = []
        previous = parent.tokens
        for i, remote in enumerate(event.block_hashes):
            if dense:
                end = parent.tokens + (i + 1) * event.block_size
                if end not in chain:
                    return self._reject_event("unrepresentable_remote_endpoint")
                endpoint = CacheEndpoint(chain[end], end)
            else:
                # Sparse hashes omit null/masked blocks while tokens describe
                # the entire range. Only an existing identity can locate them.
                endpoint = self.anchors[remote]
                if chain.get(endpoint.tokens) != endpoint.local_hash or endpoint.tokens <= previous:
                    return self._reject_event("sparse_endpoint_mismatch")
            resolved.append((remote, endpoint))
            previous = endpoint.tokens
        return resolved

    def _put(self, remote, endpoint, group):
        self.dispatch_hints.get(group, {}).pop(endpoint.local_hash, None)
        if self.pending and self.anchors.get(remote) != endpoint:
            self._identity_revision += 1
        self.anchors[remote] = endpoint
        self.anchors.move_to_end(remote)
        key = (group, remote)
        existing = self.entries.get(key)
        if existing == endpoint:
            return
        if existing is not None:
            old = self.by_group[group][existing.local_hash]
            old.discard(remote)
            if not old:
                del self.by_group[group][existing.local_hash]
        else:
            self.references[remote] += 1
        self.entries[key] = endpoint
        self.by_group[group].setdefault(endpoint.local_hash, set()).add(remote)

    def _restore_identities(self):
        native = self.native_hashes
        if native is None or self.status == "unsupported":
            return
        if self._restored_revision == native.identity_revision:
            return
        self._restored_revision = native.identity_revision
        if len(self.entries) < len(self.retained_hashes):
            for group, remote in self.retained_hashes:
                key = (group, remote)
                identity = native.identities.get(remote)
                if key not in self.entries and identity is not None:
                    self._put(remote, CacheEndpoint(identity.local_hash, identity.tokens), group)

    def _remove(self, group: int, hashes: list[str]) -> None:
        removals = Counter(hashes)
        self._release_hashes(group, removals)
        if not self.pending:
            return
        # Charge removals to confirmed copies first. Pending arrivals own at
        # most the total still retained for each hash; a later rejection
        # must only roll back that arrival's surviving copies.
        remaining = {remote: self.retained_hashes[group, remote] for remote in removals}
        for item in self.pending:
            if item.event.group_idx != group:
                continue
            for remote in remaining.keys() & item.retained.keys():
                count = item.retained[remote]
                keep = min(count, remaining[remote])
                remaining[remote] -= keep
                if keep < count:
                    if keep:
                        item.retained[remote] = keep
                    else:
                        del item.retained[remote]
                    self._retried_revision = None
        self.pending = [item for item in self.pending if item.retained]
        for remote, confirmed in remaining.items():
            if not confirmed and self.retained_hashes[group, remote]:
                self._discard_endpoint(group, remote)

    def _release_hashes(self, group: int, removals: Counter[str]) -> None:
        for remote, count in removals.items():
            key = (group, remote)
            if self.retained_hashes[key] > count:
                self.retained_hashes[key] -= count
                continue
            self.retained_hashes.pop(key, None)
            self._discard_endpoint(group, remote)

    def _discard_endpoint(self, group: int, remote: str) -> None:
        endpoint = self.entries.pop((group, remote), None)
        if endpoint is None:
            return
        self.references[remote] -= 1
        if not self.references[remote]:
            del self.references[remote]
        local = self.by_group[group][endpoint.local_hash]
        local.discard(remote)
        if not local:
            del self.by_group[group][endpoint.local_hash]

    def _retry(self, chains: dict) -> None:
        if self.status == "unsupported":
            return
        revision = self._identity_revision + (self.native_hashes.identity_revision if self.native_hashes else 0)
        while revision != self._retried_revision:
            self._retried_revision = revision
            old = self.pending
            self.pending = []
            for item in old:
                evicted = frozenset(item.event.block_hashes).difference(item.retained)
                resolved = self._store(item.event, chains, evicted)
                if resolved is False:
                    self.pending.append(item)
                elif resolved is None:
                    self._release_hashes(item.event.group_idx, item.retained)
            revision = self._identity_revision + (self.native_hashes.identity_revision if self.native_hashes else 0)

    def _expire_pending(self) -> None:
        # Missing Stored dependencies only hide additional residency; they do
        # not invalidate entries already established by other events.
        if not self.pending:
            return
        now = time.monotonic()
        # Retries and removals preserve enqueue order.
        expired = 0
        for item in self.pending:
            if now - item.enqueued_at <= self.pending_ttl:
                break
            expired += 1
        if expired:
            self._drop_pending(expired, "pending_expired")

    def _drop_pending(self, count: int, reason: str) -> None:
        for item in self.pending[:count]:
            self._release_hashes(item.event.group_idx, item.retained)
        del self.pending[:count]
        self.diagnostics[reason] += count

    def _prune_anchors(self) -> None:
        excess = len(self.anchors) - len(self.references) - self.history_limit
        if excess <= 0:
            return
        history = (remote for remote in self.anchors if remote not in self.references)
        for remote in list(islice(history, excess)):
            del self.anchors[remote]

    def estimate(self, hashes: list[str], query_tokens: int) -> PrefixEstimate:
        unit = self.registry.local_hash_unit
        limit = self.registry.lookup_limit(query_tokens)
        self._expire_pending()
        status, reason = self.status, self.reason
        if status != "ready" or query_tokens == 0:
            return PrefixEstimate(status if query_tokens else "empty", 0, query_tokens, limit, unit, reason)
        candidate = self._common_hit(hashes, min(limit, len(hashes) * unit))
        if self.registry.drop_eagle_block:
            candidate = max(0, candidate - unit)  # The unitary Full manager recomputes its last matched block.
        replay = self.registry.prefix_replay_tokens
        # vLLM abandons hits no longer than the replay window. Longer hits
        # still count as H cached tokens, while only H - W tokens save prefill.
        if candidate <= replay:
            candidate = 0
        return PrefixEstimate(
            "ready",
            candidate,
            query_tokens,
            limit,
            unit,
            replayed_tokens=replay if candidate else 0,
        )

    def _common_hit(self, hashes, candidate):
        """Intersect group matches, reusing verified Full coverage for shorter candidates."""
        groups = sorted(self.registry.required_groups, key=lambda group: group.match_kind != "full")
        indexes = {group.group_idx: self._matching_index(group) for group in groups}
        if any(not indexes[group.group_idx] and group.match_kind != "chunked" for group in groups):
            return 0
        alignment = self.registry.prefix_match_alignment
        candidate = candidate // alignment * alignment
        validated = {}
        while candidate:
            for group in groups:
                previous_hit = validated.get(group.group_idx, 0)
                if previous_hit == candidate or (group.match_kind == "full" and previous_hit > candidate):
                    continue
                hit = _find_hit(group, indexes[group.group_idx], hashes, candidate, self.registry.local_hash_unit)
                validated[group.group_idx] = hit
                if hit < candidate:
                    candidate = hit // alignment * alignment
                    break
            else:
                break
        return candidate


def _find_hit(group: CacheGroupSpec, index: dict, hashes: list[str], limit: int, unit: int) -> int:
    """Largest valid endpoint <= limit in the supplied group index."""
    if group.match_kind == "mamba":
        return _find_mamba_hit(index, hashes, limit, unit)
    if group.match_kind == "full":
        return _find_full_hit(group, index, hashes, limit, unit)
    if group.match_kind == "sliding":
        return _find_sliding_hit(group, index, hashes, limit, unit)
    if group.match_kind == "chunked":
        return _find_chunked_hit(group, index, hashes, limit, unit)
    raise ValueError(f"Unsupported match kind: {group.match_kind}")


def _find_mamba_hit(index: dict, hashes: list[str], limit: int, unit: int) -> int:
    for i in range(limit // unit - 1, -1, -1):
        if hashes[i] in index:
            return (i + 1) * unit
    return 0


def _find_full_hit(group: CacheGroupSpec, index: dict, hashes: list[str], limit: int, unit: int) -> int:
    # Mamba "all" can make the canonical unit larger than this group's
    # physical blocks. Check only endpoints expressible in both units.
    block = lcm(group.block_size, unit)
    hit = 0
    for end in range(block, limit + 1, block):
        if hashes[end // unit - 1] not in index:
            break
        hit = end
    # Check explicit partial endpoints only within the first incomplete block.
    tail = min(limit, hit + block - 1) // unit
    for i in range(tail - 1, hit // unit - 1, -1):
        if hashes[i] in index:
            return (i + 1) * unit
    return hit


def _find_sliding_hit(group: CacheGroupSpec, index: dict, hashes: list[str], limit: int, unit: int) -> int:
    block = group.block_size
    needed = max(1, (group.sliding_window - 1 + block - 1) // block)
    run = 0
    top = 0
    for end in range(limit // block * block, 0, -block):
        if end % unit == 0 and hashes[end // unit - 1] in index:
            if not run:
                top = end
            run += 1
            if run >= needed:
                return top
        else:
            run = 0
    return top if run else 0  # A shorter window is valid at the prompt start.


def _find_chunked_hit(group: CacheGroupSpec, index: dict, hashes: list[str], limit: int, unit: int) -> int:
    block = group.block_size
    # At a chunk boundary the preceding chunk requires no cached blocks.
    # If a miss moves the candidate into another chunk, re-evaluate there.
    candidate = limit // block * block
    while candidate > 0:
        start = candidate // group.attention_chunk_size * group.attention_chunk_size
        hit = start // block * block
        while hit + block <= candidate and (hit + block) % unit == 0 and hashes[(hit + block) // unit - 1] in index:
            hit += block
        hit = hit // unit * unit
        if hit == candidate:
            return hit
        candidate = hit
    return 0
