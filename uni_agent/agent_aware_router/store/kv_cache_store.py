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
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace

from ..types.kv_cache import DispatchSchema, KVCacheUpdate, NativeHashCodecFactory, NativeHashCodecResult
from .cache_registry import CacheRegistryState, build_cache_registry
from .group_cache import PrefixEstimate, ReplicaCacheIndex
from .prefix_identity import PrefixIdentityIndex

logger = logging.getLogger(__name__)


class KVCacheStore:
    """Own the pool layout, stream progress, and per-replica cache indexes."""

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
        self._schema_generations: dict[str, int] = {}
        self._metadata_revisions: Counter[str] = Counter()
        self._replay_revisions: Counter[str] = Counter()
        self._metadata_requests: dict[str, int] = {}
        self._metadata_errors: dict[str, str] = {}
        self._diagnostics: dict[str, Counter] = {}
        self._diagnostic_deltas: Counter = Counter()
        self._diagnostic_logged: dict[tuple, float] = {}
        self._index_diagnostics: dict[str, tuple[ReplicaCacheIndex, Counter]] = {}
        self._gated_replicas: set[str] = set()
        self._inflight_blocks: dict[str, Counter] = {}
        self._inflight_pins: dict[str, dict] = {}
        self._inflight_private: dict[str, dict[str, tuple[DispatchSchema | None, int | None]]] = {}
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

    def hash_schema(self) -> tuple[int, int | None]:
        with self._lock:
            return self.registry_generation, self._hash_unit_locked()

    def _hash_unit_locked(self):
        return next(
            (
                state.registry.local_hash_unit
                for state in self._replica_registries.values()
                if state.registry is not None
            ),
            None,
        )

    def _node_layout_state_locked(self, node):
        return self._replica_registries.get(node, CacheRegistryState(reason="replica_metadata_unavailable"))

    def capacity_blocks(self, node) -> int | None:
        """Read only this replica's normalized capacity; never borrow another node's."""
        with self._lock:
            state = self._replica_registries.get(node)
            return state.capacity_blocks if state is not None else None

    def set_native_hash_codec_factory(self, factory: NativeHashCodecFactory) -> None:
        """Bind a backend before collection; backend replacement discards residency."""
        with self._lock:
            if factory == self._native_codec_factory:
                return
            self._native_codec_factory = factory
            self._native_indexes.clear()
            self._native_by_node.clear()
            self._native_results.clear()
            for node in self._replica_metadata:
                self._bind_native_index_locked(node)
            self.group_caches.clear()
            self._index_diagnostics.clear()

    def _native_required_locked(self, node) -> bool:
        registry = self._node_layout_state_locked(node).registry
        if registry is None:
            return False
        full_sizes = [group.block_size for group in registry.required_groups if group.match_kind == "full"]
        return bool(full_sizes and registry.local_hash_unit not in full_sizes)

    def _bind_native_index_locked(self, node) -> None:
        self._native_by_node.pop(node, None)
        self._native_results.pop(node, None)
        if not self._native_required_locked(node):
            return
        registry = self._node_layout_state_locked(node).registry
        result = (
            self._native_codec_factory(self._replica_metadata[node], registry.local_hash_unit)
            if self._native_codec_factory is not None
            else NativeHashCodecResult(reason="native_codec_unbound")
        )
        self._native_results[node] = result
        if result.codec is None:
            self._record_diagnostic_locked(node, "codec", result.reason or "native_codec_unavailable", result.details)
            return
        key = result.codec.profile_key
        index = self._native_indexes.get(key)
        if index is None:
            index = self._native_indexes[key] = PrefixIdentityIndex(result.codec)
        self._native_by_node[node] = index

    def _record_diagnostic_locked(self, node, stage, reason, details=None, *, count=1, warning=True):
        reason = reason.split(":", 1)[0]
        self._diagnostics.setdefault(node, Counter())[(stage, reason)] += count
        key = node, stage, reason
        self._diagnostic_deltas[key] += count
        now = time.monotonic()
        if warning and (key not in self._diagnostic_logged or now - self._diagnostic_logged[key] >= 300):
            self._diagnostic_logged[key] = now
            logger.warning(
                "KV cache capability unavailable: replica=%s stage=%s reason=%s details=%s",
                node,
                stage,
                reason,
                dict(details or {}),
            )

    def _collect_index_diagnostics_locked(self, node, index):
        old_index, previous = self._index_diagnostics.get(node, (None, Counter()))
        if old_index is not index:
            previous = Counter()
        for reason, count in (index.diagnostics - previous).items():
            stage = (
                "pending" if reason.startswith("pending_") else "identity" if reason.startswith("native_") else "event"
            )
            self._record_diagnostic_locked(node, stage, reason, count=count, warning=reason != "native_events_resolved")
        self._index_diagnostics[node] = index, index.diagnostics.copy()

    def drain_cache_diagnostics(self):
        with self._lock:
            result = [(*key, count) for key, count in self._diagnostic_deltas.items()]
            self._diagnostic_deltas.clear()
            return result

    def record_cache_diagnostic(self, node, stage, reason, details=None, *, count=1):
        with self._lock:
            self._record_diagnostic_locked(node, stage, reason, details, count=count)

    def _metadata_status_locked(self, node):
        state = self._node_layout_state_locked(node)
        result = self._native_results.get(node)
        index = self.group_caches.get(node)
        paused = bool(index is not None and index.native_reconstruction_paused)
        native = "not_required"
        native_reason = None
        if self._native_required_locked(node):
            native = (
                "paused" if paused else "ready" if result is not None and result.codec is not None else "unavailable"
            )
            native_reason = (
                "native_hash_mismatch" if paused else result.reason if result is not None else "native_codec_unbound"
            )
        budget_known = bool(state.reservation_groups) and all(
            g.private_blocks is not None for g in state.reservation_groups
        )
        reason = (
            self._metadata_errors.get(node)
            or state.reason
            or ("unknown_request_budget" if not budget_known else native_reason)
        )
        ready = (
            state.status in {"registered", "disabled"}
            and budget_known
            and native not in {"paused", "unavailable"}
            and node not in self._metadata_errors
        )
        return {
            "ready": ready,
            "reason": reason,
            "matching": state.status,
            "native": native,
            "native_details": dict(result.details) if result is not None else {},
            "capacity_known": state.capacity_blocks is not None,
            "budget_known": budget_known,
            "gated": node in self._gated_replicas,
            "revision": self._metadata_revisions[node],
            "refresh_pending": node in self._metadata_requests,
            "counts": {f"{stage}:{why}": count for (stage, why), count in self._diagnostics.get(node, {}).items()},
        }

    def metadata_status(self, node):
        with self._lock:
            return self._metadata_status_locked(node)

    def get_cache_status(self):
        with self._lock:
            nodes = set(self._replica_metadata) | set(self._diagnostics) | set(self._metadata_requests)
            return {node: self._metadata_status_locked(node) for node in sorted(nodes)}

    def metadata_revision(self, node):
        with self._lock:
            return self._metadata_revisions[node]

    def replay_revision(self, node):
        with self._lock:
            return self._replay_revisions[node]

    def _request_metadata_refresh_locked(self, node, reason):
        if node not in self._metadata_requests or reason == "stream_reset":
            self._metadata_revisions[node] += 1
        self._metadata_requests[node] = self._metadata_revisions[node]
        return self._metadata_revisions[node]

    def request_metadata_refresh(self, node, reason):
        with self._lock:
            return self._request_metadata_refresh_locked(node, reason)

    def metadata_refresh_requests(self):
        with self._lock:
            return dict(self._metadata_requests)

    @staticmethod
    def _reservation_key(state):
        registry = state.registry
        return registry.local_hash_unit if registry else None, state.reservation_groups

    def refresh_replica_metadata(self, node, metadata, *, revision=None):
        """Apply one validated source snapshot without resetting unrelated caches or holds."""
        with self._lock:
            if revision is not None and revision != self._metadata_revisions[node]:
                return False
            if not isinstance(metadata, Mapping) or metadata.get("status") != "ok":
                reason = (
                    metadata.get("reason", "metadata_unavailable")
                    if isinstance(metadata, Mapping)
                    else "invalid_metadata"
                )
                self._metadata_errors[node] = str(reason)
                self._record_diagnostic_locked(node, "metadata", str(reason))
                return False
            old = self._replica_registries.get(node, CacheRegistryState())
            previous_raw = self._replica_metadata.get(node)
            previous_status = self._metadata_status_locked(node)
            incoming = dict(metadata)
            errors = incoming.get("metadata_errors") or {}
            same_layout = previous_raw is not None and all(
                previous_raw.get(key) == incoming.get(key)
                for key in (
                    "schema_version",
                    "vllm_version",
                    "groups",
                    "cache_config",
                    "parallel_config",
                    "model_config",
                )
            )
            same_identity = previous_raw is not None and (
                previous_raw.get("native_hash_config") == incoming.get("native_hash_config")
                or incoming.get("native_hash_config") is None
                and "native_hash_config" in errors
            )
            if same_layout and errors and node not in self._metadata_requests:
                for field in ("native_hash_config", "cache_reservation_profile"):
                    if (
                        incoming.get(field) is None
                        and previous_raw.get(field) is not None
                        and (field == "native_hash_config" or same_identity)
                    ):
                        incoming[field] = previous_raw[field]
            state = build_cache_registry(incoming)
            if not state.reservation_groups:
                reason = state.reason or "invalid_metadata"
                self._metadata_errors[node] = reason
                self._record_diagnostic_locked(node, "metadata", reason)
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
            if self._reservation_key(old) != self._reservation_key(state):
                self._schema_generations[node] = self._schema_generations.get(node, self.registry_generation) + 1
            old_unit = self._hash_unit_locked()
            self._replica_metadata = {**self._replica_metadata, node: incoming}
            self._replica_registries[node] = state
            if old_unit != self._hash_unit_locked():
                self.registry_generation += 1
            self._bind_native_index_locked(node)
            if errors:
                error = next(iter(errors.values()))
                reason = error.get("reason", "metadata_unavailable") if isinstance(error, Mapping) else str(error)
                self._metadata_errors[node] = str(reason)
                for part, why in errors.items():
                    reason = why.get("reason", "metadata_unavailable") if isinstance(why, Mapping) else str(why)
                    self._record_diagnostic_locked(node, "metadata", str(reason), {"part": part, "error": why})
            else:
                self._metadata_errors.pop(node, None)
                self._metadata_requests.pop(node, None)
            cache = self.group_caches.get(node)
            if cache is not None and not changed:
                cache.native_hashes = self._native_by_node.get(node)
                if not errors:
                    cache.native_reconstruction_paused = False
                    cache.apply([])
            if changed:
                self._drop_cache_locked(node)
                self._gated_replicas.add(node)
                self._replay_revisions[node] += 1
            if state.reason and state.status != "disabled":
                self._record_diagnostic_locked(node, "unit" if "unit" in state.reason else "metadata", state.reason)
            if not previous_status["ready"] and self._metadata_status_locked(node)["ready"]:
                logger.info("KV cache metadata recovered: replica=%s", node)
                self._record_diagnostic_locked(node, "metadata", "recovered", warning=False)
            active = {index.codec.profile_key for index in self._native_by_node.values()}
            self._native_indexes = {key: value for key, value in self._native_indexes.items() if key in active}
            return bool(changed and state.registry is not None)

    def apply_update(self, update: KVCacheUpdate) -> None:
        """Apply ordered cache events under one lock."""
        with self._lock:
            node = update.replica_id
            registry = self._node_layout_state_locked(node).registry
            if registry is None:
                if any(event.is_clear for event in update.events):
                    self._drop_cache_locked(node)
                return  # Valid events still feed observability with an unknown layout.
            native = self._native_by_node.get(node)
            revision = native.identity_revision if native else 0
            index = self._replica_index_locked(node)
            index.apply([registry.normalize_event(event) for event in update.events])
            self._collect_index_diagnostics_locked(node, index)
            if index.native_reconstruction_paused:
                self._request_metadata_refresh_locked(node, "native_hash_mismatch")
            while native is not None and revision != native.identity_revision:
                revision = native.identity_revision
                for cache in self.group_caches.values():
                    if cache.native_hashes is native:
                        cache.apply([])
                        self._collect_index_diagnostics_locked(cache.replica_id, cache)

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
                        "warming", 0, query_tokens, max(0, (query_tokens - 1) // unit * unit), unit, "no_events"
                    )
                else:
                    result[node] = index.estimate(hashes, query_tokens)
                    self._collect_index_diagnostics_locked(node, index)
            return result

    def invalidate_replica(self, replica_id: str) -> None:
        """Discard residency and gate estimates while preserving request holds."""
        with self._lock:
            self._gated_replicas.add(replica_id)
            self._drop_cache_locked(replica_id)

    def mark_replica_ready(self, replica_id: str, *, revision: int | None = None) -> None:
        """Expose the known resident subset after transport rebuilding."""
        with self._lock:
            if revision is not None and revision != self._replay_revisions[replica_id]:
                return
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
            self._clear_inflight_locked(replica_id)

    def per_replica_block_counts(self) -> dict[str, int]:
        """Count reported group/hash identities, including unresolved Stored events."""
        with self._lock:
            return {node: len(index.retained_hashes) for node, index in self.group_caches.items()}

    def _drop_cache_locked(self, node):
        self.group_caches.pop(node, None)
        self._index_diagnostics.pop(node, None)

    def _replica_index_locked(self, node):
        index = self.group_caches.get(node)
        if index is None:
            index = self.group_caches[node] = ReplicaCacheIndex(
                self._node_layout_state_locked(node).registry, node, native_hashes=self._native_by_node.get(node)
            )
        return index

    def _dispatch_schema_locked(self, node) -> DispatchSchema | None:
        state = self._node_layout_state_locked(node)
        registry = state.registry
        specs = state.reservation_groups
        if not specs:
            return None
        return DispatchSchema(
            self._schema_generations.get(node, self.registry_generation),
            registry.local_hash_unit if registry is not None else None,
            specs,
        )

    def dispatch_schema(self, node) -> DispatchSchema | None:
        with self._lock:
            return self._dispatch_schema_locked(node)

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
            schema = self._dispatch_schema_locked(replica_id)
            if schema is None or schema.request_blocks(1) is None:
                return None
            held = self._inflight_blocks.get(replica_id, {})
            private = self._inflight_private.get(replica_id, {})
            if any(key[0] != schema for key in held) or any(s != schema or n is None for s, n in private.values()):
                return None
            return self._reserved_blocks_locked(replica_id)

    def request_block_need(self, prompt_tokens, saved_tokens=0, *, replica_id):
        with self._lock:
            schema = self._dispatch_schema_locked(replica_id)
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
                if schema is None or any(g.private_blocks is None for g in reservations)
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
            or schema.hash_unit is None
            or schema != self._dispatch_schema_locked(replica_id)
            or replica_id in self._gated_replicas
        ):
            return 0
        return self._replica_index_locked(replica_id).record_dispatch(hashes)
