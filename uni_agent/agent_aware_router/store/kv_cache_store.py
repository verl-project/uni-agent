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

"""KVCacheStore — backend-agnostic data carrier for KV cache mapping tables."""

from __future__ import annotations

import threading
from collections.abc import Iterable

from ..types import Layer


class KVCacheStore:
    """Mutable data carrier for KV cache mapping tables.

    Attributes:
        block_size: Learned block size (None until first BlockStored event).
        replicas_by_block: local prefix hash → set of replica_ids that cache
            it on GPU.  CPU/SSD blocks are counted but not indexed here.
    """

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self.block_size: int | None = None
        self.replicas_by_block: dict[str, set[str]] = {}
        # Per-layer per-replica block counts, maintained alongside replicas_by_block.
        self._replica_layer_counts: dict[Layer, dict[str, int]] = {
            Layer.GPU: {},
            Layer.CPU: {},
            Layer.SSD: {},
        }
        # In-flight *held* blocks: replica_id → {block hash → ref count}. Unlike
        # ``replicas_by_block`` (resident, evictable) these are pinned by in-flight
        # requests, which only the router's own acquire/release can track — kv-events
        # carry no request identity. Same lock as the reverse index.
        self._inflight_blocks: dict[str, dict[str, int]] = {}
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Replica management ──────────────────────────────────────────────

    def clear_replica(self, replica_id: str) -> None:
        """Clear a replica's resident blocks from the reverse index.

        O(n) in the number of unique blocks, but replica count is typically small (< 100).
        """
        with self._lock:
            stale_hashes: list[str] = []
            for bh, replicas in self.replicas_by_block.items():
                if replica_id not in replicas:
                    continue
                replicas.discard(replica_id)
                if not replicas:
                    stale_hashes.append(bh)
            for bh in stale_hashes:
                del self.replicas_by_block[bh]
            for layer_counts in self._replica_layer_counts.values():
                layer_counts.pop(replica_id, None)

    # ── Block management ────────────────────────────────────────────────

    def add_blocks(self, replica_id: str, block_hashes: Iterable[str], layer: Layer = Layer.GPU) -> None:
        """Add blocks to a replica at a layer, updating the reverse index.

        Only GPU blocks are indexed in ``replicas_by_block`` (CPU/SSD counted
        only); GPU adds are idempotent — a block already resident is a no-op.
        """
        with self._lock:
            self._add_blocks_locked(replica_id, block_hashes, layer)

    def record_dispatch_blocks(self, replica_id: str, hash_strs: list[str]) -> int:
        """Mark a dispatched request's prefix blocks *resident* on ``replica_id`` (GPU).

        Called by the acquire path so ``gpu_hit`` sees the prefix immediately
        instead of trailing the kv-event stream; idempotent against the engine's
        later ``BlockStored`` for the same hashes, and the blocks stay indexed
        after release until the engine evicts them (``BlockRemoved``).

        Args:
            replica_id: The replica the request was dispatched to.
            hash_strs: The request's full-block chained prefix hashes (empty is
                a no-op).

        Returns:
            Number of blocks this call newly made resident (0 for a full hit).
        """
        if not hash_strs:
            return 0
        with self._lock:
            return self._add_blocks_locked(replica_id, hash_strs, Layer.GPU)

    def remove_blocks(self, replica_id: str, block_hashes: Iterable[str], layer: Layer = Layer.GPU) -> None:
        """Remove blocks from a replica at a layer, updating the reverse index.

        Symmetric to :meth:`add_blocks`: a GPU block not resident on the replica
        is a no-op, so a removal for a block it never had cannot under-count.
        """
        with self._lock:
            self._remove_blocks_locked(replica_id, block_hashes, layer)

    def _add_blocks_locked(self, replica_id: str, block_hashes: Iterable[str], layer: Layer) -> int:
        """Index + count blocks; caller holds ``_lock``. Returns the blocks added.

        GPU counting is transition-based (idempotent); CPU/SSD have no reverse
        index to test against and count unconditionally.
        """
        layer_counts = self._replica_layer_counts.setdefault(layer, {})
        added = 0
        for bh in block_hashes:
            if layer == Layer.GPU:
                replicas = self.replicas_by_block.get(bh)
                if replicas is not None and replica_id in replicas:
                    continue  # already resident — idempotent, never double-count
                if replicas is None:
                    self.replicas_by_block[bh] = {replica_id}
                else:
                    replicas.add(replica_id)
            layer_counts[replica_id] = layer_counts.get(replica_id, 0) + 1
            added += 1
        return added

    def _remove_blocks_locked(self, replica_id: str, block_hashes: Iterable[str], layer: Layer) -> None:
        """Un-index + uncount blocks. Caller holds ``_lock`` (see :meth:`remove_blocks`)."""
        layer_counts = self._replica_layer_counts.setdefault(layer, {})
        for bh in block_hashes:
            if layer == Layer.GPU:
                replicas = self.replicas_by_block.get(bh)
                if replicas is None or replica_id not in replicas:
                    continue  # not resident — never decrement into the negative
                replicas.discard(replica_id)
                if not replicas:
                    del self.replicas_by_block[bh]
            layer_counts[replica_id] = layer_counts.get(replica_id, 0) - 1

    # ── In-flight block holding (pin / unpin) ───────────────────────────

    def pin_inflight_blocks(self, replica_id: str, hash_strs: list[str]) -> None:
        """Bump the ref count of every block an in-flight request holds.

        Deduplicated by hash; blocks already resident in the prefix cache are
        pinned too (occupancy, not new allocations). Empty ``hash_strs`` pins
        nothing.
        """
        with self._lock:
            held = self._inflight_blocks.setdefault(replica_id, {})
            for h in hash_strs:
                held[h] = held.get(h, 0) + 1

    def unpin_inflight_blocks(self, replica_id: str, hash_strs: list[str]) -> None:
        """Drop one reference to every block a finishing request held.

        A block leaves the held set only when its last holder releases it —
        a shared prefix survives one of its requests completing. ``hash_strs``
        must be the same list the request pinned at acquire.
        """
        with self._lock:
            held = self._inflight_blocks.get(replica_id)
            if not held:
                return
            for h in hash_strs:
                ref_cnt = held.get(h, 0) - 1
                if ref_cnt > 0:
                    held[h] = ref_cnt
                else:
                    held.pop(h, None)

    def inflight_block_count(self, replica_id: str) -> int:
        """Number of distinct blocks currently held (pinned by in-flight requests) on ``replica_id``."""
        with self._lock:
            return len(self._inflight_blocks.get(replica_id, {}))

    def _is_cached_locked(self, replica_id: str, hash_str: str) -> bool:
        """Whether ``replica_id`` already caches ``hash_str``. Caller holds ``_lock``."""
        cached = self.replicas_by_block.get(hash_str)
        return cached is not None and replica_id in cached

    # ── Retained-cache size ─────────────────────────────────────────────

    def per_replica_block_counts(self) -> dict[str, int]:
        """Return ``{replica_id: number of distinct GPU prefix blocks it retains}``.

        GPU-only count — feeds the retained-load formula. Maintained incrementally,
        O(replicas). Divide by the per-replica block pool size for occupancy.
        """
        with self._lock:
            return dict(self._replica_layer_counts.get(Layer.GPU, {}))

    # ── Prefix hit rate queries ─────────────────────────────────────────

    def get_layer_prefix_hit_rate(
        self,
        node_id: str,
        hash_strs: list[str],
        layer: Layer = Layer.GPU,
    ) -> float:
        """Prefix-cache hit rate for a node at a layer, ∈ [0.0, 1.0].

        GPU: walk the local reverse index (``replicas_by_block``) along the
        supplied ``hash_strs`` chain until a hash isn't cached on this node.

        CPU/SSD return 0.0 today — not just because the mooncake collector is
        unwired, but because ``add_blocks`` only indexes GPU blocks into
        ``replicas_by_block`` (CPU/SSD are counted in ``_replica_layer_counts``
        only). Supporting CPU/SSD hit queries requires extending the reverse
        index to be layer-keyed (``dict[Layer, dict[str, set[str]]]``) and
        indexing those blocks in ``add_blocks``; until then ``layer`` here is a
        placeholder that short-circuits to 0.0.
        """
        if layer != Layer.GPU or self.block_size is None:
            return 0.0
        with self._lock:
            return self._gpu_hit_rate_locked(node_id, hash_strs)

    def _gpu_hit_rate_locked(self, node_id: str, hash_strs: list[str]) -> float:
        """Walk the reverse index along the hash chain. Caller holds ``_lock``."""
        if not hash_strs:
            return 0.0
        matched = 0
        for i, hs in enumerate(hash_strs):
            if not self._is_cached_locked(node_id, hs):
                break  # chain break — this node doesn't cache this hash
            matched = i + 1
        return matched / len(hash_strs)
