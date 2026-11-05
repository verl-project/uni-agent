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
from ..store.capabilities import KVState, MetricsState, RequestState
from ..store.data_store import DataStore
from .parse import InflightBlockUpdate, KVCacheUpdate, MetricsUpdate, StickyUpdate
from .sink import InflightContext
from .transport.callback import StatisticEvent

logger = logging.getLogger(__name__)


class StoreUpdater:
    """The only store writer on the collector path; observations follow writes.

    This preserves the existing per-store locking and ordered callback semantics.
    A multi-command update is not a transaction across stores.
    """

    def __init__(
        self,
        metrics: MetricsState,
        kv: KVState,
        requests: RequestState,
        observer: UpdateObserver,
    ) -> None:
        self._metrics = metrics
        self._kv = kv
        self._requests = requests
        self._observer = observer

    def prepare_inflight(self, event: StatisticEvent) -> InflightContext:
        """Gather bookings in the original order before the pure parser nets deltas."""
        rid = event.request_id
        if event.event == "on_acquire":
            turn = self._requests.next_turn(rid) if rid else 0
            stale = self._kv.get_marker(rid) if rid else None
            booked_len = self._requests.get_prompt_length(rid) if rid else 0
            if rid:
                self._requests.book_prompt_length(rid, len(event.prompt_ids))
            return InflightContext(turn, stale, self._kv.get_block_size() or 0, booked_len)
        return InflightContext(
            turn=self._requests.get_turn(rid) if rid else 0,
            booked_len=self._requests.get_prompt_length(rid) if rid else 0,
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

    def apply_inflight_blocks(self, update: InflightBlockUpdate) -> None:
        """Apply an ``InflightBlockUpdate`` command to the held-block book — mechanically.

        The acquire/release protocol decisions (when to pin, when a re-acquire
        must first unwind a stale pin) live in ``InflightParser``; this applier
        only maps each command to store operations. Unpin sets always come from
        the request's marker row (``KVCacheStore.pop_marker``) — the ground
        truth of what that dispatch pinned — so a release whose memo was later
        replaced still unwinds exactly the original set.
        """
        if update.action == "pin":
            self._kv.incr_block_refs(update.node_id, update.block_hashes)
            # Resident record runs after the ref count, on the same chain;
            # idempotent against the engine's later ``BlockStored``.
            recorded = 0
            if update.mark_resident:
                recorded = self._kv.record_dispatch_blocks(update.node_id, update.block_hashes)
            self._kv.put_marker(update.request_id, update.node_id, update.block_hashes)
            if update.mark_resident:
                self._observer.on_blocks_recorded(recorded)
        elif update.action == "unpin":
            marker = self._kv.pop_marker(update.request_id)
            if marker is None:
                # Nothing was pinned at acquire — the held set never knew about this request.
                logger.debug(f"release with no pinned blocks request={update.request_id} on {update.node_id}")
                return
            self._kv.decr_block_refs(marker["replica"], marker["hashes"])
        elif update.action == "stale_unpin":
            marker = self._kv.pop_marker(update.request_id)
            if marker is None:
                logger.debug(f"stale-unpin found no marker request={update.request_id}")
                return
            self._kv.decr_block_refs(marker["replica"], marker["hashes"])
        else:
            logger.warning(f"unknown InflightBlockUpdate action: {update.action}")

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
        store.metrics,
        store.kv,
        store.requests,
        CollectorObserver(store.metrics, store.kv, parser_stats),
    )
