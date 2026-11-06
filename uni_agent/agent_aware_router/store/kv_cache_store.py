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

import logging
import threading
from collections import Counter
from collections.abc import Mapping

from ..types.kv_cache import DispatchSchema, KVCacheUpdate
from ..types.native_hash import NativeHashCodecFactory
from .cache_registry import CacheRegistryState, build_cache_registry, legacy_full_registry
from .group_cache import PrefixEstimate, ReplicaCacheIndex
from .prefix_identity import PrefixIdentityIndex

logger = logging.getLogger(__name__)


class KVCacheStore:
    """Own the pool layout, stream progress, and per-replica cache indexes."""

    _instance: KVCacheStore | None = None

    def __init__(self) -> None:
        self._registry_state = CacheRegistryState()
        self._legacy_layout_allowed = True
        self.registry_generation = 0
        self.group_caches: dict[str, ReplicaCacheIndex] = {}
        self._native_codec_factory: NativeHashCodecFactory | None = None
        self._replica_metadata: Mapping[str, Mapping] = {}
        self._native_indexes: dict[object, PrefixIdentityIndex] = {}
        self._native_by_node: dict[str, PrefixIdentityIndex] = {}
        self._stream_errors: dict[str, str] = {}
        self._event_sequences: dict[str, int] = {}
        self._inflight_blocks: dict[str, Counter] = {}
        self._inflight_pins: dict[str, dict] = {}
        self._inflight_private: dict[str, dict[str, tuple[DispatchSchema, int | None]]] = {}
        self._lock = threading.Lock()

    @classmethod
    def singleton(cls) -> KVCacheStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def registry_state(self) -> CacheRegistryState:
        with self._lock:
            return self._registry_state

    def hash_schema(self) -> tuple[int, int | None]:
        with self._lock:
            registry = self._registry_state.registry
            return self.registry_generation, registry.local_hash_unit if registry else None

    def configure_registry(self, metadata, *, replica_metadata=None) -> CacheRegistryState:
        """Install a validated layout before collection, retaining no old residency."""
        state = build_cache_registry(metadata)
        with self._lock:
            self._registry_state = state
            self._legacy_layout_allowed = (
                isinstance(metadata, Mapping)
                and "groups" not in metadata
                and metadata.get("status") in {"not_ready", "unsupported", "error"}
                and not (
                    isinstance(metadata.get("cache_config"), Mapping)
                    and metadata["cache_config"].get("enable_prefix_caching") is False
                )
            )
            self._replica_metadata = dict(replica_metadata or {})
            self._configure_native_indexes_locked()
            self.registry_generation += 1
            self.group_caches.clear()
            self._stream_errors.clear()
            self._event_sequences.clear()
        return state

    def set_native_hash_codec_factory(self, factory: NativeHashCodecFactory) -> None:
        """Bind a backend before collection; backend replacement discards residency."""
        with self._lock:
            if factory == self._native_codec_factory:
                return
            self._native_codec_factory = factory
            self._configure_native_indexes_locked()
            self.group_caches.clear()

    def _configure_native_indexes_locked(self) -> None:
        self._native_indexes.clear()
        self._native_by_node.clear()
        registry = self._registry_state.registry
        factory = self._native_codec_factory
        if registry is None or factory is None:
            return
        full_sizes = [group.block_size for group in registry.required_groups if group.match_kind == "full"]
        if not full_sizes or registry.local_hash_unit in full_sizes:
            return
        for node, metadata in self._replica_metadata.items():
            codec = factory(metadata, registry.local_hash_unit)
            if codec is None:
                continue
            index = self._native_indexes.get(codec.profile_key)
            if index is None:
                index = self._native_indexes[codec.profile_key] = PrefixIdentityIndex(codec)
            self._native_by_node[node] = index

    def _event_registry_locked(self, events):
        state = self._registry_state
        inferred = state.inferred
        if state.registry is not None and not inferred:
            return state.registry
        if not inferred and not self._legacy_layout_allowed:
            return None
        # An untagged old producer is compatible under a single-Full contract.
        # Group tags or any changed event block size revoke that assumption,
        # including partial-event sizes that need authoritative group metadata.
        tagged = any(not event.is_clear and (event.group_idx is not None or event.kind is not None) for event in events)
        sizes = {
            event.block_size
            for event in events
            if event.is_store
            and isinstance(event.medium, str)
            and event.medium.upper() == "GPU"
            and (event.locality or "LOCAL").upper() == "LOCAL"
            and event.ownership is None
        }
        if inferred:
            sizes.add(state.registry.local_hash_unit)
        if tagged or len(sizes) > 1:
            reason = "group_metadata_unavailable" if tagged else "ambiguous_legacy_block_size"
            self._registry_state = CacheRegistryState("unsupported", reason=reason)
            self._legacy_layout_allowed = False
            self.registry_generation += 1
            self.group_caches.clear()
            logger.warning("Cannot infer a single Full cache layout from native events: %s", reason)
            return None
        if state.registry is None and sizes:
            self._registry_state = legacy_full_registry(sizes.pop())
            self.registry_generation += 1
            logger.warning("KV metadata unavailable; assuming one Full group for untagged legacy cache events")
        return self._registry_state.registry

    def apply_update(self, update: KVCacheUpdate) -> bool:
        """Check stream integrity, adapt old fields, and apply events under one lock."""
        with self._lock:
            node = update.node_id
            if update.reset_stream:
                self.group_caches.pop(node, None)
                self._stream_errors.pop(node, None)
                self._event_sequences.pop(node, None)
                return True
            if update.sequence is not None:
                last = self._event_sequences.get(node)
                if last is not None and update.sequence <= last:
                    return False
                self._event_sequences[node] = update.sequence
                if last is not None and update.sequence != last + 1:
                    self._stream_errors[node] = f"kv_sequence_gap:{last + 1}:{update.sequence}"
                    self.group_caches.pop(node, None)
                    logger.warning("KV event gap for %s: expected=%s received=%s", node, last + 1, update.sequence)
            if update.stream_error:
                self._stream_errors[node] = update.stream_error
                self.group_caches.pop(node, None)
                return False
            if node in self._stream_errors:
                if not any(event.is_clear for event in update.events):
                    return False
                self._stream_errors.pop(node, None)
            if any(event.is_clear for event in update.events):
                self._clear_inflight_locked(node)
            registry = self._event_registry_locked(update.events)
            if registry is None:
                if any(event.is_clear for event in update.events):
                    self.group_caches.pop(node, None)
                return True  # Valid events still feed observability with an unknown layout.
            native = self._native_by_node.get(node)
            revision = native.identity_revision if native else 0
            index = self._replica_index_locked(node)
            index.apply([registry.normalize_event(event) for event in update.events])
            while native is not None and native.enabled and revision != native.identity_revision:
                revision = native.identity_revision
                for cache in self.group_caches.values():
                    if cache.native_hashes is native:
                        cache.apply([])
            return True

    def estimate_prefixes(self, node_ids, hashes, query_tokens) -> dict[str, PrefixEstimate]:
        with self._lock:
            state = self._registry_state
            result = {}
            for node in node_ids:
                index = self.group_caches.get(node)
                unit = state.registry.local_hash_unit if state.registry else None
                if node in self._stream_errors:
                    result[node] = PrefixEstimate("warming", 0, query_tokens, 0, unit, self._stream_errors[node])
                elif state.registry is None:
                    result[node] = PrefixEstimate(state.status, 0, query_tokens, 0, None, state.reason)
                elif index is None:
                    result[node] = PrefixEstimate(
                        "warming", 0, query_tokens, max(0, (query_tokens - 1) // unit * unit), unit, "no_events"
                    )
                else:
                    result[node] = index.estimate(hashes, query_tokens)
            return result

    def clear_replica(self, replica_id: str) -> None:
        """Discard residency and request reservations without changing stream progress."""
        with self._lock:
            self.group_caches.pop(replica_id, None)
            self._clear_inflight_locked(replica_id)

    def per_replica_block_counts(self) -> dict[str, int]:
        """Count reported group/hash identities, including unresolved Stored events."""
        with self._lock:
            return {node: len(index.retained_hashes) for node, index in self.group_caches.items()}

    def _replica_index_locked(self, node):
        index = self.group_caches.get(node)
        if index is None:
            index = self.group_caches[node] = ReplicaCacheIndex(
                self._registry_state.registry, node, native_hashes=self._native_by_node.get(node)
            )
        return index

    def _dispatch_schema_locked(self) -> DispatchSchema | None:
        registry = self._registry_state.registry
        if registry is None:
            return None
        return DispatchSchema(self.registry_generation, registry.local_hash_unit, registry.reservation_specs)

    def dispatch_schema(self) -> DispatchSchema | None:
        with self._lock:
            return self._dispatch_schema_locked()

    def _clear_inflight_locked(self, replica_id):
        self._inflight_blocks.pop(replica_id, None)
        self._inflight_private.pop(replica_id, None)
        for request in [rid for rid, marker in self._inflight_pins.items() if marker["replica"] == replica_id]:
            del self._inflight_pins[request]

    def _incr_refs_locked(self, replica_id, hashes, schema, group_idx=0):
        self._inflight_blocks.setdefault(replica_id, Counter()).update((schema, group_idx, h) for h in hashes)

    def _decr_refs_locked(self, replica_id, hashes, schema, group_idx=0):
        held = self._inflight_blocks.get(replica_id)
        if held is None:
            return
        for h in hashes:
            key = (schema, group_idx, h)
            if held[key] > 1:
                held[key] -= 1
            else:
                held.pop(key, None)
        if not held:
            self._inflight_blocks.pop(replica_id, None)

    def inflight_block_count(self, replica_id) -> int:
        """Estimated held pool blocks: distinct group refs plus private reservations."""
        with self._lock:
            return self._reserved_blocks_locked(replica_id)

    def _reserved_blocks_locked(self, replica_id):
        shared = len(self._inflight_blocks.get(replica_id, {}))
        private = self._inflight_private.get(replica_id, {})
        return shared + sum(count or 0 for _, count in private.values())

    def inflight_reserved_blocks(self, replica_id) -> int | None:
        """One pool-block budget for every layout; unknown after incompatible changes."""
        with self._lock:
            schema = self._dispatch_schema_locked()
            if schema is None or schema.request_blocks(1) is None:
                return None
            held = self._inflight_blocks.get(replica_id, {})
            private = self._inflight_private.get(replica_id, {})
            if any(key[0] != schema for key in held) or any(s != schema or n is None for s, n in private.values()):
                return None
            return self._reserved_blocks_locked(replica_id)

    def request_block_need(self, prompt_tokens, saved_tokens=0):
        with self._lock:
            schema = self._dispatch_schema_locked()
            return schema.request_blocks(prompt_tokens, saved_tokens) if schema is not None else None

    def get_marker(self, request_id):
        with self._lock:
            marker = self._inflight_pins.get(request_id)
            return {**marker, "hashes": list(marker["hashes"])} if marker is not None else None

    def _unpin_request_locked(self, request_id):
        marker = self._inflight_pins.pop(request_id, None)
        if marker is not None:
            node, schema = marker["replica"], marker["schema"]
            for group in marker.get("reservations", ()):
                self._decr_refs_locked(node, group.block_hashes, schema, group.group_idx)
            private = self._inflight_private.get(node)
            if private is not None:
                private.pop(request_id, None)
                if not private:
                    self._inflight_private.pop(node, None)

    def unpin_request(self, request_id):
        """Release exactly the acquire-time group refs and private reservation once."""
        with self._lock:
            self._unpin_request_locked(request_id)

    def pin_request(self, request_id, replica_id, hashes, schema, *, reservations, predict=True) -> int:
        """Atomically retain group refs, private budgets, prediction and release marker."""
        with self._lock:
            self._unpin_request_locked(request_id)
            for group in reservations:
                self._incr_refs_locked(replica_id, group.block_hashes, schema, group.group_idx)
            private = (
                None
                if any(g.private_blocks is None for g in reservations)
                else sum(g.private_blocks for g in reservations)
            )
            self._inflight_private.setdefault(replica_id, {})[request_id] = (schema, private)
            self._inflight_pins[request_id] = {
                "replica": replica_id,
                "hashes": list(hashes),
                "schema": schema,
                "reservations": tuple(reservations),
            }
            return self._record_dispatch_locked(replica_id, hashes, schema) if predict else 0

    def _record_dispatch_locked(self, replica_id, hashes, schema) -> int:
        if (
            not hashes
            or schema is None
            or schema != self._dispatch_schema_locked()
            or replica_id in self._stream_errors
        ):
            return 0
        return self._replica_index_locked(replica_id).record_dispatch(hashes)
