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

from typing import Any

from ..types import MetricKey
from ..types.kv_cache import KVCacheUpdate, NativeHashCodecFactory
from .cache_registry import CacheRegistryState
from .kv_cache_store import KVCacheStore
from .per_replica_store import PerReplicaStore
from .per_request_store import PerRequestStore

_STICKY_KEY = "sticky_replica"


class DataStore:
    """Unified data access layer — single entry point for all store operations.

    Wraps the singleton stores and exposes a unified interface for all
    reads and writes.  Stateless — instantiate once and reuse.

    Usage:
        store = DataStore()
        # Write metrics
        store.refresh_metrics({'node1': {'kv_cache_usage_perc': 45.0}})
        # Read metrics
        value = store.get_metric('node1', 'kv_cache_usage_perc')
        # Inspect reported cache residency
        store.per_replica_block_counts()
    """

    def __init__(self) -> None:
        self._metrics = PerReplicaStore.singleton()
        self._kv = KVCacheStore.singleton()
        self._per_request = PerRequestStore.singleton()

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
        return self._metrics.get(replica_id, key)

    def get_metrics(self, replica_id: str) -> dict[str, Any]:
        """Get a node's full metrics snapshot.

        Args:
            replica_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node is absent.
        """
        return self._metrics.get(replica_id)

    def get_metric_replica_ids(self) -> list[str]:
        """Return all node IDs that have metrics in the store."""
        return self._metrics.all_ids()

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
        return self._metrics.refresh(new_data)

    # ── KVCacheStore operations ─────────────────────────────────────────

    def cache_registry_state(self, replica_id) -> CacheRegistryState:
        """This replica's layout used by group-aware indexing and matching."""
        return self._kv.registry_state(replica_id)

    def refresh_replica_metadata(self, replica_id, metadata, *, revision=None):
        return self._kv.refresh_replica_metadata(replica_id, metadata, revision=revision)

    def metadata_revision(self, replica_id):
        return self._kv.metadata_revision(replica_id)

    def replay_revision(self, replica_id):
        return self._kv.replay_revision(replica_id)

    def request_metadata_refresh(self, replica_id, reason):
        return self._kv.request_metadata_refresh(replica_id, reason)

    def metadata_refresh_requests(self):
        return self._kv.metadata_refresh_requests()

    def metadata_status(self, replica_id):
        status = self._kv.metadata_status(replica_id)
        status["capacity_known"] = self.capacity_blocks(replica_id) is not None
        return status

    def get_cache_status(self):
        statuses = self._kv.get_cache_status()
        for node, status in statuses.items():
            status["capacity_known"] = self.capacity_blocks(node) is not None
        return statuses

    def drain_cache_diagnostics(self):
        return self._kv.drain_cache_diagnostics()

    def record_cache_diagnostic(self, replica_id, stage, reason, details=None, *, count=1):
        self._kv.record_cache_diagnostic(replica_id, stage, reason, details, count=count)

    def set_native_hash_codec_factory(self, factory: NativeHashCodecFactory) -> None:
        self._kv.set_native_hash_codec_factory(factory)

    def capacity_blocks(self, replica_id) -> int | None:
        """Current node's usable pool budget, preferring validated native metadata."""
        if self._kv.dispatch_schema(replica_id) is None:
            return None
        capacity = self._kv.capacity_blocks(replica_id)
        if capacity is None:
            capacity = self.get_metrics(replica_id).get(MetricKey.KV_CACHE_CAPACITY_BLOCKS)
        return capacity if type(capacity) is int and capacity >= 0 else None

    def hash_schema(self) -> tuple[int, int | None]:
        return self._kv.hash_schema()

    def apply_kv_update(self, update: KVCacheUpdate) -> None:
        self._kv.apply_update(update)

    def estimate_prefixes(self, replica_ids, hashes, query_tokens):
        return self._kv.estimate_prefixes(replica_ids, hashes, query_tokens)

    def invalidate_kv_replica(self, replica_id: str) -> None:
        """Clear the replica index and gate hits and dispatch predictions."""
        self._kv.invalidate_replica(replica_id)

    def mark_kv_replica_ready(self, replica_id: str, *, revision: int | None = None) -> None:
        """Expose residency restored by the completed event replay."""
        self._kv.mark_replica_ready(replica_id, revision=revision)

    def enable_kv_dispatch_estimates(self, replica_id: str) -> None:
        """Discard incomplete replay state and permit new dispatch estimates."""
        self._kv.enable_dispatch_estimates(replica_id)

    def clear_kv_resident_blocks(self, replica_id: str) -> None:
        """Clear reported residency without discarding request reservations."""
        self._kv.clear_resident_blocks(replica_id)

    # ── KV-cache load (load signal) ─────────────────────────────────────

    def kv_cache_load(self, replica_id: str) -> float:
        """Estimate retention pressure from event counts / GPU pool blocks.

        This differs from native in-use occupancy: unreferenced blocks can still
        retain reusable cache. Hybrid partial aliases are not physical-block
        deduplicated. Missing pool capacity keeps the original zero default.
        """
        retained = self._kv.per_replica_block_counts().get(replica_id, 0)
        if retained <= 0:
            return 0.0
        total = self._metrics.get(replica_id, MetricKey.NUM_GPU_BLOCKS)
        if not total or total <= 0:
            return 0.0
        return min(1.0, retained / float(total))

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return per-replica GPU cache retention estimates from events.

        Thin pass-through to ``KVCacheStore``; used by the periodic kv-events
        tally log to surface retained-block counts per replica.
        """
        return self._kv.per_replica_block_counts()

    def dispatch_schema(self, replica_id):
        return self._kv.dispatch_schema(replica_id)

    def inflight_block_count(self, replica_id):
        return self._kv.inflight_block_count(replica_id)

    def inflight_reserved_blocks(self, replica_id):
        return self._kv.inflight_reserved_blocks(replica_id)

    def request_block_need(self, prompt_tokens, saved_tokens=0, *, replica_id):
        return self._kv.request_block_need(prompt_tokens, saved_tokens, replica_id=replica_id)

    def get_marker(self, request_id):
        return self._kv.get_marker(request_id)

    def pin_request(self, request_id, replica_id, hashes, schema, *, reservations, predict=True):
        return self._kv.pin_request(request_id, replica_id, hashes, schema, reservations=reservations, predict=predict)

    def unpin_request(self, request_id):
        self._kv.unpin_request(request_id)

    # ── PerReplicaStore incremental write ──────────────────────────────────

    def incr_metric(self, replica_id: str, key: str, delta: int | float = 1) -> int | float:
        """Apply a signed delta to one metric for one node (inflight ±1).

        Routes to ``PerReplicaStore.incr`` (not ``refresh``) so a stateless delta
        emitter (``InflightParser``) can move a running counter without
        tracking the absolute value itself.

        Returns:
            The new value of ``key``.
        """
        return self._metrics.incr(replica_id, key, delta)

    def incr_metrics(self, replica_id: str, deltas: dict[str, int | float]) -> dict[str, int | float]:
        """Apply multiple signed deltas to one node under a single lock.

        Batched variant of :meth:`incr_metric` for the ``on_acquire`` parser,
        which emits several deltas per dispatch (INFLIGHT / DISPATCHED /
        PROMPT_LEN_SUM) — batching avoids one lock cycle per key on the hot path.

        Returns:
            ``{canonical_key: new_value}`` for every key in ``deltas``.
        """
        return self._metrics.incr_many(replica_id, deltas)

    # ── Sticky bindings (a per-request value stored under _STICKY_KEY) ───

    def get_sticky_binding(self, request_id: str) -> str | None:
        """Return the bound replica_id for ``request_id`` (None if cold/evicted)."""
        return self._per_request.get(request_id, _STICKY_KEY)

    def put_sticky_binding(self, request_id: str, replica_id: str) -> None:
        """Bind / refresh ``request_id → replica_id`` (driven by ``on_acquire``)."""
        self._per_request.set(request_id, _STICKY_KEY, replica_id)

    def invalidate_sticky_binding(self, request_id: str) -> None:
        """Drop one request_id's sticky binding."""
        self._per_request.delete(request_id, _STICKY_KEY)

    def invalidate_sticky_replica(self, replica_id: str) -> None:
        """Drop every sticky binding pointing at a removed replica."""
        self._per_request.delete_where(_STICKY_KEY, replica_id)

    def clear_sticky_bindings(self) -> int:
        """Drop every sticky binding regardless of target replica.

        Backs the #7115 protocol's ``clear_sticky_cache`` (trainer phase
        switches / fully-async rebalancing).

        Returns:
            Number of request rows a binding was dropped from.
        """
        return self._per_request.delete_key(_STICKY_KEY)

    def sticky_status(self) -> dict:
        """Return a debugging snapshot of the sticky bindings."""
        return {"max_size": self._per_request.max_size, "size": self._per_request.count(_STICKY_KEY)}

    # ── PerRequestStore operations ─────────────────────────────────────

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
