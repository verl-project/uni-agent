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

from ..types import Layer
from .capabilities import KVState, MetricsState, RequestState
from .kv_cache_store import KVCacheStore
from .per_replica_store import PerReplicaStore
from .per_request_store import PerRequestStore


class DataStore:
    """Composition root and compatibility facade for the three store capabilities.

    New code uses ``metrics``, ``kv`` and ``requests``. Flat methods remain
    compatibility delegates, including legacy scratch access; production
    callers use named request operations instead of owning scratch keys.

    Usage:
        store = DataStore()
        # Write metrics
        store.refresh_metrics({'node1': {'kv_cache_usage_perc': 45.0}})
        # Read metrics
        value = store.get_metric('node1', 'kv_cache_usage_perc')
        # Write KV cache blocks
        store.add_kv_blocks('node1', ['hash1', 'hash2'])
    """

    def __init__(self) -> None:
        self._metrics = PerReplicaStore.singleton()
        self._kv = KVCacheStore.singleton()
        self._per_request = PerRequestStore.singleton()
        self.metrics = MetricsState(self._metrics)
        self.kv = KVState(self._kv, self.metrics)
        self.requests = RequestState(self._per_request)

    # ── PerReplicaStore operations ─────────────────────────────────────────

    def get_metric(self, node_id: str, key: str) -> Any:
        """Query a single metric by canonical key.

        Args:
            node_id: Target node.
            key: Canonical metric key (e.g., ``MetricKey.KV_CACHE_USAGE_PERC``).

        Returns:
            Metric value; falls back to ``METRIC_SPECS`` default if absent.

        Raises:
            KeyError: If key is not a valid canonical key.
        """
        return self.metrics.get_metric(node_id, key)

    def get_metrics(self, node_id: str) -> dict[str, Any]:
        """Get a node's full metrics snapshot.

        Args:
            node_id: Target node.

        Returns:
            Dict of canonical_key → value; empty dict if node is absent.
        """
        return self.metrics.get_metrics(node_id)

    def get_metric_node_ids(self) -> list[str]:
        """Return all node IDs that have metrics in the store."""
        return self.metrics.get_metric_node_ids()

    def refresh_metrics(self, new_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Batch refresh metrics from collectors.

        For each node in ``new_data``: merge with existing data
        (new values overwrite same keys).  Nodes NOT in ``new_data``
        are left untouched.

        Args:
            new_data: Dict of {node_id: {canonical_key: value}}.

        Returns:
            ``{node_id: full_merged_snapshot}`` (see ``PerReplicaStore.refresh``).
        """
        return self.metrics.refresh_metrics(new_data)

    # ── KVCacheStore operations ─────────────────────────────────────────

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

    def add_kv_blocks(self, node_id: str, block_hashes: list[str], layer: Layer = Layer.GPU) -> None:
        """Add KV cache blocks to a node.

        Args:
            node_id: Target node.
            block_hashes: List of local prefix hashes to add.
            layer: Cache layer (``Layer.GPU``/``Layer.CPU``/``Layer.SSD``).
        """
        return self.kv.add_kv_blocks(node_id, block_hashes, layer)

    def record_dispatch_blocks(self, node_id: str, block_hashes: list[str]) -> int:
        """Mark a dispatched request's prefix blocks resident on a node (GPU).

        The acquire path calls this so the resident index learns the prefix
        immediately instead of waiting for the engine's kv-events; idempotent
        against the later ``BlockStored``.

        Args:
            node_id: The replica the request was dispatched to.
            block_hashes: The request's full-block chained prefix hashes.

        Returns:
            Number of blocks this call newly made resident (0 for a full hit).
        """
        return self.kv.record_dispatch_blocks(node_id, block_hashes)

    def remove_kv_blocks(self, node_id: str, block_hashes: list[str], layer: Layer = Layer.GPU) -> None:
        """Remove KV cache blocks from a node.

        Args:
            node_id: Target node.
            block_hashes: List of local prefix hashes to remove.
            layer: Cache layer (``Layer.GPU``/``Layer.CPU``/``Layer.SSD``).
        """
        return self.kv.remove_kv_blocks(node_id, block_hashes, layer)

    def clear_kv_node(self, node_id: str) -> None:
        """Clear all KV cache state for a node: resident blocks, block refs, markers.

        Args:
            node_id: Target node.
        """
        return self.kv.clear_kv_node(node_id)

    def get_kv_block_count(self) -> int:
        """Return the number of unique block hashes currently cached."""
        return self.kv.get_kv_block_count()

    def kv_node_has_blocks(self, node_id: str) -> bool:
        """Return True if node_id appears in at least one cached block."""
        return self.kv.kv_node_has_blocks(node_id)

    def has_kv_block(self, block_hash: str) -> bool:
        """Return True if block_hash is present in the cache index."""
        return self.kv.has_kv_block(block_hash)

    # ── KV cache prefix hit rate queries ────────────────────────────────

    def get_layer_prefix_hit_rate(
        self,
        node_id: str,
        hash_strs: list[str],
        layer: Layer = Layer.GPU,
    ) -> float:
        """Query prefix-cache hit rate for a node at a given layer.

        Args:
            node_id: Target node.
            hash_strs: Precomputed ``str(h)`` chain of the prompt's full-block
                prefix hashes (computed once by the caller and shared across
                replicas/layers).
            layer: Cache layer (``Layer.GPU``/``Layer.CPU``/``Layer.SSD``).

        Only the GPU layer is backed by a reverse index today; CPU/SSD return
        0.0 (see ``KVCacheStore.get_layer_prefix_hit_rate``).

        Returns:
            Hit rate 0.0–1.0.
        """
        return self.kv.get_layer_prefix_hit_rate(node_id, hash_strs, layer)

    # ── KV-cache load (load signal) ─────────────────────────────────────

    def kv_cache_load(self, node_id: str) -> float:
        """KV-cache load = ``retained_blocks / num_gpu_blocks`` (∈ [0,1]).

        Retained blocks (free-pool blocks hash-marked for reuse) over the GPU
        block pool — rises as distinct prefixes accumulate and signals impending
        LRU eviction. Returns 0.0 when retained blocks or ``num_gpu_blocks`` is
        unavailable (e.g. mc-off groups emit no kv-events).
        """
        return self.kv.kv_cache_load(node_id)

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return ``{replica_id: number of distinct prefix blocks it retains}``.

        Thin pass-through to ``KVCacheStore``; used by the periodic kv-events
        tally log to surface retained-block counts per replica.
        """
        return self.kv.per_replica_block_counts()

    # ── Block reference counts and per-request markers ──────────────────

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

    def incr_metric(self, node_id: str, key: str, delta: int | float = 1) -> int | float:
        """Apply a signed delta to one metric for one node (inflight ±1).

        Routes to ``PerReplicaStore.incr`` (not ``refresh``) so a stateless delta
        emitter (``InflightParser``) can move a running counter without
        tracking the absolute value itself.

        Returns:
            The new value of ``key``.
        """
        return self.metrics.incr_metric(node_id, key, delta)

    def incr_metrics(self, node_id: str, deltas: dict[str, int | float]) -> dict[str, int | float]:
        """Apply multiple signed deltas to one node under a single lock.

        Batched variant of :meth:`incr_metric` for the ``on_acquire`` parser,
        which emits several deltas per dispatch (INFLIGHT / DISPATCHED /
        PROMPT_LEN_SUM) — batching avoids one lock cycle per key on the hot path.

        Returns:
            ``{canonical_key: new_value}`` for every key in ``deltas``.
        """
        return self.metrics.incr_metrics(node_id, deltas)

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
