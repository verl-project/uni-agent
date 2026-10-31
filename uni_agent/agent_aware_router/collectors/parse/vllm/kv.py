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

"""VLLMKVParser — vLLM KV-cache event parser.

Parses msgpack payloads from ZMQ and returns structured update commands.
Store writes are handled by Collector via DataStore.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from copy import copy, deepcopy

import msgpack

from ....collectors.parse import KVCacheUpdate, Parser
from ....collectors.parse.vllm.kv_event import KVCacheEvent
from ....types import Layer
from ....types.cache_lifecycle import BRIDGE_SCHEMA_KEY, CacheLifecycleState, KVEventFrame
from ....utils.hash import compute_hash

logger = logging.getLogger(__name__)

_FILTERED_EVENT_LOG_INTERVAL_S = 30.0


class VLLMKVParser(Parser):
    """vLLM KV-cache parser — msgpack payload → KVCacheUpdate.

    Each event type has a dedicated handler that folds its blocks into a single
    ``KVCacheUpdate`` accumulator; ``parse`` is pure dispatch.

    Attributes:
        remote_to_local_block_hash: Mapping from vLLM remote block_hash
            to locally-computed prefix hash (str).  Used for chained
            hash computation.
        _block_size: Learned block size from first eligible GPU stored event.
    """

    # vLLM BlockStored/BlockRemoved ``medium`` → canonical layer.
    # Only None (older vLLM) defaults to GPU; unknown media remain unclassified.
    _MEDIUM_TO_LAYER: dict[str, Layer] = {"GPU": Layer.GPU, "CPU": Layer.CPU, "SSD": Layer.SSD}

    def __init__(self, *, group_aware: bool = False, lifecycle_reader=None) -> None:
        self.group_aware = group_aware
        self.lifecycle_reader = lifecycle_reader
        self.remote_to_local_block_hash: dict[str, str] = {}
        self._block_size: int | None = None
        self._filtered_event_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self._filter_last_logged: dict[tuple[str, str], float] = {}
        self._contexts: dict[tuple, tuple[dict, int | None]] = {}
        self._context_key = None

    def retain_nodes(self, node_ids):
        """Keep hash mappings only for event sources resumed by the transport."""
        self._contexts = {key: value for key, value in self._contexts.items() if key[0] in node_ids}
        if self._context_key is not None and self._context_key[0] not in node_ids:
            self._context_key = None
            self.remote_to_local_block_hash = {}
            self._block_size = None

    def snapshot(self):
        """Copy parser state while retaining the shared lifecycle reader."""
        result = copy(self)
        result.remote_to_local_block_hash = self.remote_to_local_block_hash.copy()
        result._contexts = deepcopy(self._contexts)
        result._filtered_event_counts = deepcopy(self._filtered_event_counts)
        result._filter_last_logged = self._filter_last_logged.copy()
        return result

    def _switch_context(self, node_id, lifecycle=None):
        identity = (lifecycle.instance_id, lifecycle.epoch) if lifecycle else (None, None)
        key = (node_id, *identity)
        if key == self._context_key:
            return
        if self._context_key is not None:
            self._contexts[self._context_key] = (self.remote_to_local_block_hash, self._block_size)
        # A remote hash mapping belongs to one replica generation. Retire older
        # contexts for this replica; never clear another replica's mappings.
        for old in list(self._contexts):
            if old[0] == node_id and old != key:
                del self._contexts[old]
        self.remote_to_local_block_hash, self._block_size = self._contexts.get(key, ({}, None))
        self._context_key = key

    def parse(self, raw_data: bytes | str, node_id: str) -> KVCacheUpdate | None:
        """Parse msgpack payload and return structured update command.

        Handles both single batches (real-time) and multiple batches (replay):
          - Single: [timestamp, [event, ...], ...]
          - Multiple: [[timestamp, [...], ...], [timestamp, [...], ...]]
        Each event may be a legacy positional array or a mapping with named
        fields and a ``type`` discriminator.

        Args:
            raw_data: ZMQ payload bytes (msgpack-encoded).
            node_id: The endpoint that sent this payload.

        Returns:
            KVCacheUpdate with operations to apply, or None if parsing failed.
        """
        sequence = None
        if isinstance(raw_data, KVEventFrame):
            sequence = raw_data.sequence
            if raw_data.reset_stream:
                self._switch_context(node_id)
                self.remote_to_local_block_hash.clear()
                self._block_size = None
                return KVCacheUpdate(node_id=node_id, reset_stream=True)
            if raw_data.error:
                self._switch_context(node_id)
                self.remote_to_local_block_hash.clear()
                return KVCacheUpdate(node_id=node_id, stream_error=raw_data.error)
            raw_data = raw_data.payload
        if isinstance(raw_data, str):
            logger.debug("VLLMKVParser received string data, expected bytes — skipping")
            return None

        bridged = False
        try:
            raw = msgpack.unpackb(raw_data, raw=False)
            lifetime = None
            if isinstance(raw, dict) and BRIDGE_SCHEMA_KEY in raw:
                bridged = True
                if raw[BRIDGE_SCHEMA_KEY] != 1:
                    raise ValueError("unsupported cache event bridge schema")
                lifetime = CacheLifecycleState.from_dict(raw["state"])
                sequence = raw["sequence"]
                if raw.get("error"):
                    return KVCacheUpdate(node_id=node_id, lifecycle=lifetime, stream_error=raw["error"])
                raw = msgpack.unpackb(raw["payload"], raw=False) if raw["payload"] else [0, []]
            current = self.lifecycle_reader(node_id) if self.lifecycle_reader else None
            if current is not None and (
                lifetime is None
                or current.instance_id != lifetime.instance_id
                or current.epoch != lifetime.epoch
                or current.index_state != "ready"
            ):
                return None  # Reject obsolete identity before touching legacy mappings.
            self._switch_context(node_id, lifetime)

            if not isinstance(raw, list | dict) or len(raw) == 0:
                raise ValueError(f"Unexpected msgpack format (type={type(raw).__name__})")
            event_payloads = raw if isinstance(raw, list) and isinstance(raw[0], list | dict) else [raw]

            update = KVCacheUpdate(
                node_id=node_id, group_events=[] if self.group_aware else None, lifecycle=lifetime, sequence=sequence
            )
            for payload in event_payloads:
                events = KVCacheEvent.from_raw(payload, default_node_id=node_id)
                for event in events:
                    if self.group_aware:
                        update.group_events.append(event)
                        continue
                    if event.event_type == "stored":
                        self._on_block_stored(event, update)
                    elif event.event_type == "removed":
                        self._on_block_removed(event, update)
                    elif event.event_type == "clear":
                        self._on_all_blocks_cleared(event, update)
                    else:
                        raise ValueError(f"Unknow event.event_type {event.event_type}.")
            return update

        except (msgpack.UnpackException, ValueError, TypeError, KeyError) as exc:
            preview = bytes(raw_data[:32]).hex() if isinstance(raw_data, bytes | bytearray) else str(raw_data)[:64]
            logger.warning(
                f"Failed to parse msgpack payload from node {node_id}: {exc!r} (len={len(raw_data)}, head={preview})"
            )
            if bridged:
                return KVCacheUpdate(node_id=node_id, stream_error="invalid_bridge_envelope")
            return None

    @classmethod
    def _medium_to_layer(cls, medium: str | None) -> Layer | None:
        """Classify a medium; legacy None means GPU, unknown strings return None."""
        if medium is None:
            return Layer.GPU
        return cls._MEDIUM_TO_LAYER.get(medium.upper())

    # ── Event handlers ──────────────────────────────────────────────────

    def _record_filtered_event(self, event: KVCacheEvent, layer: Layer | None) -> None:
        """Count excluded media without mutating the GPU routing cache view."""
        assert event.medium is not None
        medium = event.medium.upper()
        key = (event.node_id, medium)
        counts = self._filtered_event_counts[key]
        counts[f"{event.event_type}_events"] += 1
        counts[f"{event.event_type}_blocks"] += len(event.block_hashes)
        now = time.monotonic()
        last_logged = self._filter_last_logged.get(key)
        if last_logged is not None and now - last_logged < _FILTERED_EVENT_LOG_INTERVAL_S:
            return
        self._filter_last_logged[key] = now
        logger.log(
            logging.WARNING if layer is None else logging.INFO,
            "KV events excluded from router cache view (node=%s, medium=%r, reason=%s): "
            "stored_events=%d removed_events=%d stored_blocks=%d removed_blocks=%d; "
            "only GPU residency is tracked",
            event.node_id,
            medium,
            "unknown medium" if layer is None else "non-GPU medium",
            counts["stored_events"],
            counts["removed_events"],
            counts["stored_blocks"],
            counts["removed_blocks"],
        )

    def _on_block_stored(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Handle BlockStored: learn block_size, compute local hashes, fold into update."""
        layer = self._medium_to_layer(event.medium)
        if layer != Layer.GPU:
            self._record_filtered_event(event, layer)
            return
        token_ids = event.token_ids
        if token_ids is None:
            logger.debug("Stored event has no token_ids — skipping")
            return

        if self._block_size is None and event.block_size is not None:
            self._block_size = event.block_size
            update.set_block_size(event.block_size)

        seed = 0
        local_parent_hash = seed
        if event.parent_block_hash is not None:
            local_parent_str = self.remote_to_local_block_hash.get(event.parent_block_hash)
            if local_parent_str is not None:
                local_parent_hash = int(local_parent_str)

        local_hashes: list[str] = []
        for i, block_bytes in enumerate(token_ids):
            if i >= len(event.block_hashes):
                break
            local_hash_int = compute_hash(
                local_parent_hash,
                block_bytes,
                seed=seed,
            )
            local_hash_str = str(local_hash_int)
            bh = event.block_hashes[i]
            self.remote_to_local_block_hash[bh] = local_hash_str
            local_hashes.append(local_hash_str)
            local_parent_hash = local_hash_int  # chain

        update.add(Layer.GPU, local_hashes)

    def _on_block_removed(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Handle BlockRemoved: convert remote hashes to local, fold into update."""
        layer = self._medium_to_layer(event.medium)
        if layer != Layer.GPU:
            self._record_filtered_event(event, layer)
            return
        local_hashes = [
            self.remote_to_local_block_hash[bh] for bh in event.block_hashes if bh in self.remote_to_local_block_hash
        ]
        for bh in event.block_hashes:
            self.remote_to_local_block_hash.pop(bh, None)

        update.remove(Layer.GPU, local_hashes)

    def _on_all_blocks_cleared(self, event: KVCacheEvent, update: KVCacheUpdate) -> None:
        """Handle AllBlocksCleared: mark the replica for a full clear."""
        self.remote_to_local_block_hash.clear()
        update.clear()
