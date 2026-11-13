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

"""Apply collector updates to store capabilities in their existing protocol order."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..insight.collector import CollectorObserver, UpdateObserver
from ..store.data_store import DataStore
from ..types import InflightUpdate
from .parse import KVCacheUpdate, MetricsUpdate, StickyUpdate

logger = logging.getLogger(__name__)


class StoreUpdater:
    """The collector application boundary; DataStore owns inflight transitions.

    This preserves the existing per-store locking and ordered callback semantics.
    A multi-command update is not a transaction across stores.
    """

    def __init__(self, store: DataStore, observer: UpdateObserver) -> None:
        self._metrics = store.metrics
        self._kv = store.kv
        self._requests = store.requests
        self._inflight = store.inflight
        self._observer = observer

    def apply_inflight(self, update: InflightUpdate) -> None:
        """Delegate the whole transition to DataStore, then observe applied effects."""
        result = self._inflight.apply(update)
        if result.blocks_recorded is not None:
            self._observer.on_blocks_recorded(result.blocks_recorded)
        for applied in result.metrics:
            self._observer.on_metrics_applied(
                MetricsUpdate(applied.node_id, applied.deltas, is_delta=True, request_id=update.request_id),
                applied.values,
            )

    def apply_kv(self, update: KVCacheUpdate) -> None:
        if update.block_size is not None:
            self._kv.set_block_size(update.block_size)
        if update.clear_all:
            self._kv.clear_kv_node(update.node_id)
        for layer, hashes in update.remove_blocks.items():
            if hashes:
                self._kv.remove_kv_blocks(update.node_id, hashes, layer=layer)
        for layer, hashes in update.add_blocks.items():
            if hashes:
                self._kv.add_kv_blocks(update.node_id, hashes, layer=layer)

        self._observer.on_kv_applied(update)

    def apply_metrics(self, update: MetricsUpdate) -> None:
        if update.is_delta:
            values = self._metrics.incr_metrics(update.node_id, update.metrics)
        else:
            values = self._metrics.refresh_metrics({update.node_id: update.metrics}).get(update.node_id, {})
        self._observer.on_metrics_applied(update, values)

    def apply_sticky(self, update: StickyUpdate) -> None:
        """Apply a StickyUpdate through the request-state capability."""
        if update.action == "put":
            self._requests.put_sticky_binding(update.request_id, update.replica_id)
        elif update.action == "invalidate":
            self._requests.invalidate_sticky_binding(update.request_id)
        elif update.action == "invalidate_replica":
            for rid in update.replica_ids:
                self._requests.invalidate_sticky_replica(rid)
        else:
            logger.warning(f"unknown StickyUpdate action: {update.action}")


def default_updater(parser_stats: Callable[[], Any] = lambda: None) -> StoreUpdater:
    """Compose singleton-backed capabilities at the application boundary."""
    store = DataStore()
    return StoreUpdater(
        store,
        CollectorObserver(store.metrics, store.kv, parser_stats),
    )
