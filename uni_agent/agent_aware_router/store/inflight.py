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

"""DataStore's inflight accounting, independent of collectors and parsers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..types import InflightUpdate, MetricKey
from ..utils.hash import get_prefix_hashes_incremental
from .capabilities import KVState, MetricsState, RequestState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Booking:
    replica_id: str
    turn: int
    prompt_length: int


@dataclass(frozen=True)
class InflightMetricsApplied:
    """A signed change and the metric values returned by the successful write."""

    node_id: str
    deltas: dict[str, int]
    values: dict[str, int | float]


@dataclass
class InflightResult:
    """Applied effects for the updater to observe, in accounting order."""

    metrics: list[InflightMetricsApplied] = field(default_factory=list)
    blocks_recorded: int | None = None


class InflightState:
    """Own dispatch transitions across DataStore's metrics, KV and request state.

    One collector serializes calls to its DataStore instance. Active bookings
    survive optional request-memo eviction and remain until release/replacement.
    These writes use the existing store locks, not a cross-store transaction.
    """

    def __init__(self, metrics: MetricsState, kv: KVState, requests: RequestState) -> None:
        self._metrics = metrics
        self._kv = kv
        self._requests = requests
        self._active: dict[str, _Booking] = {}

    def apply(self, update: InflightUpdate) -> InflightResult:
        if update.action == "acquire":
            return self._acquire(update)
        if update.action == "release":
            return self._release(update)
        raise ValueError(f"Unknown inflight action: {update.action}")

    def _incr(self, result: InflightResult, node_id: str, deltas: dict[str, int]) -> None:
        values = self._metrics.incr_metrics(node_id, deltas)
        result.metrics.append(InflightMetricsApplied(node_id, deltas, values))

    def _unpin(self, request_id: str) -> None:
        # The marker is the acquire-time truth, independent of request memos.
        marker = self._kv.pop_marker(request_id)
        if marker is not None:
            self._kv.decr_block_refs(marker["replica"], marker["hashes"])

    def _acquire(self, update: InflightUpdate) -> InflightResult:
        rid = update.request_id
        plen = len(update.prompt_ids)
        metrics = {
            MetricKey.INFLIGHT_COUNT: 1,
            MetricKey.INFLIGHT_TOKENS: plen,
            MetricKey.DISPATCHED_COUNT: 1,
            MetricKey.PROMPT_LEN_SUM: plen,
        }
        result = InflightResult()
        if rid is None:
            self._incr(result, update.node_id, metrics)
            return result

        active = self._active.get(rid)
        turn = max(self._requests.get_turn(rid), active.turn if active else 0) + 1
        block_size = self._kv.get_block_size() or 0
        hash_strs = []
        if update.prompt_ids and block_size > 0:
            # Resolve before ANY writes, including stale unwind. Invalid tokens
            # must leave a previous active dispatch and its booking intact.
            hashes, _ = get_prefix_hashes_incremental(list(update.prompt_ids), block_size, 0, 0)
            hash_strs = [str(h) for h in hashes]

        if active is not None:
            logger.warning(f"re-acquire without release request={rid} on {active.replica_id} — unwinding old dispatch")
            self._unpin(rid)
            self._incr(
                result,
                active.replica_id,
                {
                    MetricKey.INFLIGHT_COUNT: -1,
                    MetricKey.INFLIGHT_TOKENS: -active.prompt_length,
                    MetricKey.INFLIGHT_TURN_SUM: -active.turn,
                },
            )
        if hash_strs:
            self._kv.incr_block_refs(update.node_id, hash_strs)
            result.blocks_recorded = self._kv.record_dispatch_blocks(update.node_id, hash_strs)
            self._kv.put_marker(rid, update.node_id, hash_strs)

        metrics[MetricKey.INFLIGHT_TURN_SUM] = turn
        self._incr(result, update.node_id, metrics)
        # Commit only after resolution and block/metric application succeed.
        self._requests.book_turn(rid, turn)
        self._requests.book_prompt_length(rid, plen)
        self._active[rid] = _Booking(update.node_id, turn, plen)
        return result

    def _release(self, update: InflightUpdate) -> InflightResult:
        rid = update.request_id
        result = InflightResult()
        metrics = {MetricKey.INFLIGHT_COUNT: -1, MetricKey.COMPLETED_COUNT: 1}
        if rid is not None:
            active = self._active.get(rid)
            if active is None or active.replica_id != update.node_id:
                # Unknown/duplicate/old-replica releases cannot debit a booking.
                return result
            self._unpin(rid)
            metrics[MetricKey.INFLIGHT_TURN_SUM] = -active.turn
            metrics[MetricKey.INFLIGHT_TOKENS] = -active.prompt_length
        # Unattributed releases retain the legacy count-only behavior.
        self._incr(result, update.node_id, metrics)
        if rid is not None:
            self._active.pop(rid)
        return result
