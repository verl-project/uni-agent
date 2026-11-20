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

"""Ordered cache events and a single group-aware prefix index per replica."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..types.kv_cache import LOCAL_GPU_DOMAIN, KVCacheUpdate, NativeHashCodecFactory, NativeHashCodecResult
from .cache_registry import CacheRegistryState, build_cache_registry
from .group_cache import PrefixEstimate, ReplicaCacheIndex
from .prefix_identity import PrefixIdentityIndex


class KVCacheStore:
    """Own the engine layouts, stream progress, and per-replica cache indexes."""

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self.registry_generation = 0
        self.group_caches: dict[str, ReplicaCacheIndex] = {}
        self._native_codec_factory: NativeHashCodecFactory | None = None
        self._replica_metadata: Mapping[str, Mapping] = {}
        self._replica_registries: dict[str, CacheRegistryState] = {}
        self._native_indexes: dict[object, PrefixIdentityIndex] = {}
        self._native_by_node: dict[str, PrefixIdentityIndex] = {}
        self._native_results: dict[str, NativeHashCodecResult] = {}
        self._metadata_errors: dict[str, str] = {}
        self._gated_replicas: set[str] = set()
        self.block_size: int | None = None
        self._inflight_blocks: dict[str, dict[str, int]] = {}
        self._inflight_pins: dict[str, dict] = {}
        self._lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def registry_state(self, node) -> CacheRegistryState:
        with self._lock:
            return self._node_layout_state_locked(node)

    def hash_schema(self) -> tuple[int, int | None, int]:
        with self._lock:
            unit, lookahead = self._hash_parameters_locked()
            return self.registry_generation, unit, lookahead

    def _hash_parameters_locked(self):
        return next(
            (
                (state.registry.local_hash_unit, state.registry.lookahead_tokens)
                for state in self._replica_registries.values()
                if state.registry is not None
            ),
            (None, 0),
        )

    def _node_layout_state_locked(self, node):
        return self._replica_registries.get(node, CacheRegistryState(reason="replica_metadata_unavailable"))

    def set_native_hash_codec_factory(self, factory: NativeHashCodecFactory) -> None:
        """Bind a backend before collection; backend replacement discards residency."""
        with self._lock:
            if factory == self._native_codec_factory:
                return
            indexes: dict[object, PrefixIdentityIndex] = {}
            by_node = {}
            results = {}
            for node, metadata in self._replica_metadata.items():
                registry = self._node_layout_state_locked(node).registry
                result, index = self._build_native_binding(metadata, registry, factory, indexes)
                if result is not None:
                    results[node] = result
                if index is not None:
                    by_node[node] = index
            self._native_codec_factory = factory
            self._native_indexes = indexes
            self._native_by_node = by_node
            self._native_results = results
            self.group_caches.clear()

    @staticmethod
    def _native_required(registry) -> bool:
        if registry is None:
            return False
        full_sizes = [group.block_size for group in registry.required_groups if group.match_kind == "full"]
        return bool(full_sizes and registry.local_hash_unit not in full_sizes)

    def _build_native_binding(self, metadata, registry, factory, indexes):
        if not self._native_required(registry):
            return None, None
        result = (
            factory(metadata, registry.local_hash_unit)
            if factory is not None
            else NativeHashCodecResult(reason="native_codec_unbound")
        )
        if result.codec is None:
            return result, None
        key = result.codec.profile_key
        index = indexes.get(key)
        if index is None:
            index = indexes[key] = PrefixIdentityIndex(result.codec)
        return result, index

    def _metadata_status_locked(self, node):
        state = self._node_layout_state_locked(node)
        result = self._native_results.get(node)
        index = self.group_caches.get(node)
        paused = bool(index is not None and index.native_reconstruction_paused)
        native = "not_required"
        native_reason = None
        if self._native_required(state.registry):
            native = (
                "paused" if paused else "ready" if result is not None and result.codec is not None else "unavailable"
            )
            native_reason = (
                "native_hash_mismatch" if paused else result.reason if result is not None else "native_codec_unbound"
            )
        return {
            "ready": state.status == "registered" and not native_reason and node not in self._metadata_errors,
            "reason": self._metadata_errors.get(node) or state.reason or native_reason,
            "matching": state.status,
            "native": native,
            "native_details": dict(result.details) if result is not None else {},
            "gated": node in self._gated_replicas,
        }

    def metadata_status(self, node):
        with self._lock:
            return self._metadata_status_locked(node)

    def get_cache_status(self):
        with self._lock:
            nodes = self._replica_metadata.keys() | self._metadata_errors.keys()
            return {node: self._metadata_status_locked(node) for node in sorted(nodes)}

    def refresh_replica_metadata(self, node, metadata):
        """Apply one validated source snapshot without resetting unrelated caches or holds."""
        with self._lock:
            if not isinstance(metadata, Mapping) or metadata.get("status") != "ok":
                reason = (
                    metadata.get("reason", "metadata_unavailable")
                    if isinstance(metadata, Mapping)
                    else "invalid_metadata"
                )
                self._metadata_errors[node] = str(reason)
                return False
            old = self._replica_registries.get(node, CacheRegistryState())
            previous_raw = self._replica_metadata.get(node)
            incoming = deepcopy(dict(metadata))
            errors = incoming.get("metadata_errors")
            if errors is not None and not isinstance(errors, Mapping):
                self._metadata_errors[node] = "invalid_metadata_errors"
                return False
            error_reason = None
            if errors:
                error = next(iter(errors.values()))
                error_reason = str(error.get("reason", "metadata_unavailable") if isinstance(error, Mapping) else error)
            state = build_cache_registry(incoming)
            if not state.has_layout:
                reason = state.reason or "invalid_metadata"
                self._metadata_errors[node] = reason
                return False
            registry = state.registry
            if registry is not None:
                for other, peer in self._replica_registries.items():
                    if other == node or peer.registry is None:
                        continue
                    candidate = peer.registry
                    if any(
                        getattr(registry, key) is not None
                        and getattr(candidate, key) is not None
                        and getattr(registry, key) != getattr(candidate, key)
                        for key in ("layout_key", "model", "mamba_cache_mode")
                    ):
                        state = replace(
                            state, status="unsupported", registry=None, reason="incompatible_replica_layout"
                        )
                        break
            old_matching = (
                (old.registry.layout_key, old.registry.model, old.registry.mamba_cache_mode) if old.registry else None
            )
            new_matching = (
                (state.registry.layout_key, state.registry.model, state.registry.mamba_cache_mode)
                if state.registry
                else None
            )
            changed = old_matching != new_matching or (
                previous_raw is not None
                and previous_raw.get("native_hash_config") != incoming.get("native_hash_config")
            )
            # A factory failure must leave the current snapshot available for a retry.
            indexes = self._native_indexes.copy()
            result, index = self._build_native_binding(incoming, state.registry, self._native_codec_factory, indexes)
            old_hash_parameters = self._hash_parameters_locked()
            self._replica_metadata = {**self._replica_metadata, node: incoming}
            self._replica_registries[node] = state
            if old_hash_parameters != self._hash_parameters_locked():
                self.registry_generation += 1
            if result is None:
                self._native_results.pop(node, None)
            else:
                self._native_results[node] = result
            if index is None:
                self._native_by_node.pop(node, None)
            else:
                self._native_by_node[node] = index
            if error_reason is not None:
                self._metadata_errors[node] = error_reason
            else:
                self._metadata_errors.pop(node, None)
            if changed:
                self._drop_cache_locked(node)
                self._gated_replicas.add(node)
            active = {index.codec.profile_key for index in self._native_by_node.values()}
            self._native_indexes = {key: value for key, value in indexes.items() if key in active}
            return bool(changed and state.registry is not None)

    def apply_update(self, update: KVCacheUpdate) -> None:
        """Apply ordered cache events under one lock."""
        with self._lock:
            node = update.replica_id
            if self.block_size is None:
                # The existing load scorer uses the first local GPU block size.
                for event in update.events:
                    if event.is_store and event.raw_token_ids is not None and event.storage_domain == LOCAL_GPU_DOMAIN:
                        self.block_size = event.block_size
                        break
            registry = self._node_layout_state_locked(node).registry
            if registry is None:
                if any(event.is_clear for event in update.events):
                    self._drop_cache_locked(node)
                return  # Valid events still feed observability with an unknown layout.
            native = self._native_by_node.get(node)
            revision = native.identity_revision if native else 0
            index = self._replica_index_locked(node)
            index.apply([registry.normalize_event(event) for event in update.events])
            while native is not None and revision != native.identity_revision:
                revision = native.identity_revision
                for cache in self.group_caches.values():
                    if cache.native_hashes is native:
                        cache.apply([])

    def estimate_prefixes(self, replica_ids, hashes, query_tokens) -> dict[str, PrefixEstimate]:
        with self._lock:
            result = {}
            for node in replica_ids:
                state = self._node_layout_state_locked(node)
                index = self.group_caches.get(node)
                unit = state.registry.local_hash_unit if state.registry else None
                if node in self._gated_replicas:
                    result[node] = PrefixEstimate("warming", 0, query_tokens, 0, unit, "stream_rebuilding")
                elif state.registry is None:
                    result[node] = PrefixEstimate(state.status, 0, query_tokens, 0, None, state.reason)
                elif index is None:
                    result[node] = PrefixEstimate(
                        "warming", 0, query_tokens, state.registry.lookup_limit(query_tokens), unit, "no_events"
                    )
                else:
                    result[node] = index.estimate(hashes, query_tokens)
            return result

    def invalidate_replica(self, replica_id: str) -> None:
        """Discard residency and gate estimates while preserving request holds."""
        with self._lock:
            self._gated_replicas.add(replica_id)
            self._drop_cache_locked(replica_id)

    def mark_replica_ready(self, replica_id: str) -> None:
        """Expose the known resident subset after transport rebuilding."""
        with self._lock:
            self._gated_replicas.discard(replica_id)

    def enable_dispatch_estimates(self, replica_id: str) -> None:
        """Discard the partial view and allow fresh dispatch predictions."""
        with self._lock:
            self._drop_cache_locked(replica_id)
            self._gated_replicas.discard(replica_id)

    def clear_resident_blocks(self, replica_id: str) -> None:
        """Clear residency without changing request holds or gating state."""
        with self._lock:
            self._drop_cache_locked(replica_id)

    def clear_replica(self, replica_id: str) -> None:
        """Discard residency and request reservations without changing stream progress."""
        with self._lock:
            self._drop_cache_locked(replica_id)
            self._inflight_blocks.pop(replica_id, None)
            for rid in [rid for rid, marker in self._inflight_pins.items() if marker["replica"] == replica_id]:
                del self._inflight_pins[rid]

    def per_replica_block_counts(self) -> dict[str, int]:
        """Count reported group/hash identities, including unresolved Stored events."""
        with self._lock:
            return {node: len(index.retained_hashes) for node, index in self.group_caches.items()}

    def _drop_cache_locked(self, node):
        self.group_caches.pop(node, None)

    def _replica_index_locked(self, node):
        index = self.group_caches.get(node)
        if index is None:
            index = self.group_caches[node] = ReplicaCacheIndex(
                self._node_layout_state_locked(node).registry, node, native_hashes=self._native_by_node.get(node)
            )
        return index

    def record_dispatch_blocks(self, replica_id: str, hashes: list[str]) -> int:
        """Record group-aligned prefix predictions for this replica."""
        with self._lock:
            registry = self._node_layout_state_locked(replica_id).registry
            if not hashes or registry is None or replica_id in self._gated_replicas:
                return 0
            return self._replica_index_locked(replica_id).record_dispatch(hashes)

    def incr_block_refs(self, replica_id: str, hash_strs: list[str]) -> None:
        """Add one reference to every hash on ``replica_id`` (multiset insert).

        Deduplicated by hash: a hash already referenced counts once more per
        ``incr_block_refs`` call, not once per block. Blocks already resident in
        the prefix cache are ref'd too. Empty ``hash_strs`` is a no-op.
        """
        with self._lock:
            held = self._inflight_blocks.setdefault(replica_id, {})
            for h in hash_strs:
                held[h] = held.get(h, 0) + 1

    def decr_block_refs(self, replica_id: str, hash_strs: list[str]) -> None:
        """Drop one reference to every hash on ``replica_id`` (multiset remove).

        A hash leaves the table when its last reference goes. ``hash_strs`` must
        mirror the references previously added for them to reach zero.
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
        """Number of distinct hashes currently referenced on ``replica_id``."""
        with self._lock:
            return len(self._inflight_blocks.get(replica_id, {}))

    def put_marker(self, request_id: str, replica_id: str, hash_strs: list[str]) -> None:
        """Store a per-request snapshot row (overwrites any previous row).

        Caller-defined payload; the caller owns its semantics.
        """
        with self._lock:
            self._inflight_pins[request_id] = {"replica": replica_id, "hashes": list(hash_strs)}

    def get_marker(self, request_id: str) -> dict[str, Any] | None:
        """Return a copy of ``request_id``'s marker row without clearing it (``None`` if absent)."""
        with self._lock:
            marker = self._inflight_pins.get(request_id)
            if marker is None:
                return None
            return {"replica": marker["replica"], "hashes": list(marker["hashes"])}

    def pop_marker(self, request_id: str) -> dict[str, Any] | None:
        """Remove and return ``request_id``'s marker row (``None`` if absent).

        Read-and-clear is one locked step, so a duplicated consumer can never
        process the same row twice — the second one finds no row.
        """
        with self._lock:
            return self._inflight_pins.pop(request_id, None)
