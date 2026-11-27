# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DataStore — unified data access layer for all stores."""

from __future__ import annotations

from threading import Lock
from typing import Any

from ..types import Layer
from ..types.kv_cache import KVCacheUpdate, NativeHashCodecFactory
from .cache_registry import CacheRegistryState
from .capabilities import KVState, MetricsState, RequestState
from .inflight import InflightState
from .kv_cache_store import KVCacheStore
from .per_replica_store import PerReplicaStore
from .per_request_store import PerRequestStore


class DataStore:
    """Composition root and compatibility facade for the store capabilities.

    New code uses ``metrics``, ``kv``, ``requests`` and ``inflight``. Flat methods remain
    compatibility delegates, including legacy scratch access; production
    callers use named request operations instead of owning scratch keys.

    Usage:
        store = DataStore()
        # Write metrics
        store.refresh_metrics({'node1': {'kv_cache_usage_perc': 45.0}})
        # Read metrics
        value = store.get_metric('node1', 'kv_cache_usage_perc')
        # Inspect reported cache residency
        store.per_replica_block_counts()
    """

    _lock = Lock()
    _inflight_backing: tuple[PerReplicaStore, KVCacheStore, PerRequestStore] | None = None
    _inflight: InflightState | None = None

    def __init__(self) -> None:
        with DataStore._lock:
            self._metrics = PerReplicaStore.singleton()
            self._kv = KVCacheStore.singleton()
            self._per_request = PerRequestStore.singleton()
            self._init_capabilities()

    def _init_capabilities(self) -> None:
        self.metrics = MetricsState(self._metrics)
        self.kv = KVState(self._kv, self.metrics)
        self.requests = RequestState(self._per_request)
        backing = (self._metrics, self._kv, self._per_request)
        inflight = DataStore._inflight
        if inflight is None or DataStore._inflight_backing != backing:
            inflight = InflightState(self.metrics, self.kv, self.requests)
            DataStore._inflight_backing = backing
            DataStore._inflight = inflight
        self.inflight = inflight

    # ── PerReplicaStore operations ─────────────────────────────────────────

    def get_metric(self, replica_id: str, key: str) -> Any:
        """Query a single metric by canonical key.

        Args:
            replica_id: Target node.
            key: Canonical metric key (e.g., ``MetricKey.KV_CACHE_USAGE_PERC``).

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if absent.

        Raises:
            KeyError: If key is not a valid canonical key.
        """
        return self.metrics.get_metric(replica_id, key)

    def get_metrics(self, replica_id: str) -> dict[str, Any]:
        """Get a node's full metrics snapshot.

        Args:
            replica_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node is absent.
        """
        return self.metrics.get_metrics(replica_id)

    def get_metric_replica_ids(self) -> list[str]:
        """Return all node IDs that have metrics in the store."""
        return self.metrics.get_metric_node_ids()

    def refresh_metrics(self, new_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Batch refresh metrics from collectors.

        For each node in ``new_data``: merge with existing data
        (new values overwrite same keys).  Nodes NOT in ``new_data``
        are left untouched.

        Args:
            new_data: Dict of {replica_id: {canonical_key: value}}.

        Returns:
            ``{replica_id: full_merged_snapshot}`` (see ``PerReplicaStore.refresh``).
        """
        return self.metrics.refresh_metrics(new_data)

    # ── KVCacheStore operations ─────────────────────────────────────────

    def cache_registry_state(self, replica_id) -> CacheRegistryState:
        """This replica's layout used by group-aware indexing and matching."""
        return self.kv.cache_registry_state(replica_id)

    def refresh_replica_metadata(self, replica_id, metadata):
        return self.kv.refresh_replica_metadata(replica_id, metadata)

    def metadata_status(self, replica_id):
        return self.kv.metadata_status(replica_id)

    def get_cache_status(self):
        return self.kv.get_cache_status()

    def set_native_hash_codec_factory(self, factory: NativeHashCodecFactory) -> None:
        self.kv.set_native_hash_codec_factory(factory)

    def get_block_size(self) -> int | None:
        """Get learned block size.

        Returns:
            Block size in tokens, or None if not yet learned.
        """
        return self.kv.get_block_size()

    def set_block_size(self, size: int) -> None:
        """Set block size (learned from first BlockStored event).

        Args:
            size: Block size in tokens.
        """
        return self.kv.set_block_size(size)

    def hash_schema(self) -> tuple[int, int | None, int]:
        return self.kv.hash_schema()

    def apply_kv_update(self, update: KVCacheUpdate) -> None:
        self.kv.apply_kv_update(update)

    def estimate_prefixes(self, replica_ids, hashes, query_tokens):
        return self.kv.estimate_prefixes(replica_ids, hashes, query_tokens)

    def invalidate_kv_replica(self, replica_id: str) -> None:
        """Clear the replica index and gate hits and dispatch predictions."""
        self.kv.invalidate_kv_replica(replica_id)

    def mark_kv_replica_ready(self, replica_id: str) -> None:
        """Expose residency restored by the completed event replay."""
        self.kv.mark_kv_replica_ready(replica_id)

    def enable_kv_dispatch_estimates(self, replica_id: str) -> None:
        """Discard incomplete replay state and permit new dispatch estimates."""
        self.kv.enable_kv_dispatch_estimates(replica_id)

    def clear_kv_resident_blocks(self, replica_id: str) -> None:
        """Clear reported residency without discarding request reservations."""
        self.kv.clear_kv_resident_blocks(replica_id)

    # ── KV-cache load (load signal) ─────────────────────────────────────

    def kv_cache_load(self, replica_id: str) -> float:
        """Estimate retention pressure from event counts / GPU pool blocks.

        This differs from native in-use occupancy: unreferenced blocks can still
        retain reusable cache. Hybrid partial aliases are not physical-block
        deduplicated. Missing pool capacity keeps the original zero default.
        """
        return self.kv.kv_cache_load(replica_id)

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return per-replica GPU cache retention estimates from events.

        Thin pass-through to ``KVCacheStore``; used by the periodic kv-events
        tally log to surface retained-block counts per replica.
        """
        return self.kv.per_replica_block_counts()

    def record_dispatch_blocks(self, replica_id: str, hashes: list[str]) -> int:
        return self.kv.record_dispatch_blocks(replica_id, hashes)

    def get_layer_prefix_hit_rate(
        self, replica_id: str, hash_strs: list[str], layer: Layer = Layer.GPU, *, query_tokens: int | None = None
    ) -> float:
        return self.kv.get_layer_prefix_hit_rate(replica_id, hash_strs, layer, query_tokens=query_tokens)

    def incr_block_refs(self, replica_id: str, hash_strs: list[str]) -> None:
        """Add one reference to every hash on a replica (multiset insert).

        Thin pass-through to ``KVCacheStore`` (see :meth:`KVCacheStore.incr_block_refs`).
        """
        return self.kv.incr_block_refs(replica_id, hash_strs)

    def decr_block_refs(self, replica_id: str, hash_strs: list[str]) -> None:
        """Drop one reference to every hash on a replica (multiset remove)."""
        return self.kv.decr_block_refs(replica_id, hash_strs)

    def inflight_block_count(self, replica_id: str) -> int:
        """Number of distinct hashes currently referenced on ``replica_id``."""
        return self.kv.inflight_block_count(replica_id)

    def put_marker(self, request_id: str, replica_id: str, hash_strs: list[str]) -> None:
        """Store a per-request snapshot row (see :meth:`KVCacheStore.put_marker`)."""
        return self.kv.put_marker(request_id, replica_id, hash_strs)

    def get_marker(self, request_id: str) -> dict[str, Any] | None:
        """Return a copy of a request's marker row without clearing it (``None`` if absent)."""
        return self.kv.get_marker(request_id)

    def pop_marker(self, request_id: str) -> dict[str, Any] | None:
        """Remove and return a request's marker row (``None`` if absent)."""
        return self.kv.pop_marker(request_id)

    # ── PerReplicaStore incremental write ──────────────────────────────────

    def incr_metric(self, replica_id: str, key: str, delta: int | float = 1) -> int | float:
        """Apply a signed delta to one metric for one node (inflight ±1).

        Routes to ``PerReplicaStore.incr`` (not ``refresh``) so callers can apply deltas
        without tracking the absolute value themselves.

        Returns:
            The new value of ``key``.
        """
        return self.metrics.incr_metric(replica_id, key, delta)

    def incr_metrics(self, replica_id: str, deltas: dict[str, int | float]) -> dict[str, int | float]:
        """Apply multiple signed deltas to one node under a single lock.

        Batched variant of :meth:`incr_metric` for inflight accounting,
        which applies several deltas per dispatch (INFLIGHT / DISPATCHED /
        PROMPT_LEN_SUM) — batching avoids one lock cycle per key on the hot path.

        Returns:
            ``{canonical_key: new_value}`` for every key in ``deltas``.
        """
        return self.metrics.incr_metrics(replica_id, deltas)

    # ── Request-state compatibility delegates ─────────────────────────

    def get_sticky_binding(self, request_id: str) -> str | None:
        """Return the bound replica_id for ``request_id`` (None if cold/evicted)."""
        return self.requests.get_sticky_binding(request_id)

    def put_sticky_binding(self, request_id: str, replica_id: str) -> None:
        """Bind / refresh ``request_id → replica_id`` (driven by ``on_acquire``)."""
        return self.requests.put_sticky_binding(request_id, replica_id)

    def invalidate_sticky_binding(self, request_id: str) -> None:
        """Drop one request_id's sticky binding."""
        return self.requests.invalidate_sticky_binding(request_id)

    def invalidate_sticky_replica(self, replica_id: str) -> None:
        """Drop every sticky binding pointing at a removed replica."""
        return self.requests.invalidate_sticky_replica(replica_id)

    def clear_sticky_bindings(self) -> int:
        """Drop every sticky binding regardless of target replica.

        Backs the #7115 protocol's ``clear_sticky_cache`` (trainer phase
        switches / fully-async rebalancing).

        Returns:
            Number of request rows a binding was dropped from.
        """
        return self.requests.clear_sticky_bindings()

    def sticky_status(self) -> dict:
        """Return a debugging snapshot of the sticky bindings."""
        return self.requests.sticky_status()

    # ── Legacy scratch access; new code uses named requests operations ──

    def incr_per_request(self, request_id: str, key: str, delta: int | float = 1):
        """Apply a signed delta to one per-request value; return the new value.

        Routes to ``PerRequestStore.incr`` — the per-request store is a generic
        ``request_id → {key: value}`` scratch space (not tied to any one metric),
        so the caller owns ``key`` and its semantics. Mirrors ``incr_metric``.
        """
        return self._per_request.incr(request_id, key, delta)

    def get_per_request(self, request_id: str, key: str, default: Any = None):
        """Return one per-request value (``default`` if unset/evicted)."""
        return self._per_request.get(request_id, key, default)

    def set_per_request(self, request_id: str, key: str, value: Any) -> None:
        """Set one per-request value (creates the row if new)."""
        self._per_request.set(request_id, key, value)

    def del_per_request(self, request_id: str, key: str) -> None:
        """Drop one per-request value (no-op if absent)."""
        self._per_request.delete(request_id, key)
