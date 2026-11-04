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

"""InflightParser — Balancer acquire/release callbacks → netted metric deltas + block commands.

the collector gathers the per-event state from the store and hands it to :meth:`InflightParser.parse`
as keyword arguments; the parser decides and emits ``InflightBlockUpdate`` pin/unpin/stale_unpin
commands plus fully-netted ``MetricsUpdate`` deltas.
"""

from __future__ import annotations

import logging
from typing import Any

from ....collectors.parse import InflightBlockUpdate, MetricsUpdate, Parser
from ....collectors.transport.callback import StatisticEvent
from ....types import MetricKey
from ....utils.hash import get_prefix_hashes_incremental

logger = logging.getLogger(__name__)


class InflightParser(Parser):
    """Pure acquire/release protocol decisions → netted deltas + block commands.

    Args:
        raw_data: The raw callback payload — an ``on_acquire`` / ``on_release``
            ``StatisticEvent``; anything else is ignored (``None``).
        node_id: Source endpoint id (empty for the callback transport — the
            event carries its own replica).
        turn: The request's turn after the collector's increment (acquire) or
            its stored turn (release); 0 when unattributable.
        stale: The request's still-held pin marker (``get_marker`` peek), or
            ``None`` — its presence marks the previous release as dropped.
        block_size: The learned KV block size (store read by the collector);
            0 = not yet learned → nothing resolvable, nothing pinned.
        booked_len: The request's currently-booked prompt length — the stale
            dispatch's booking netted out on acquire, folded out on release.
    """

    def parse(
        self,
        raw_data: bytes | str | Any,
        node_id: str,
        *,
        turn: int = 0,
        stale: dict[str, Any] | None = None,
        block_size: int = 0,
        booked_len: int = 0,
    ) -> list | None:
        """Dispatch on acquire/release; ignore other events / non-event payloads."""
        if not isinstance(raw_data, StatisticEvent):
            return None

        event = raw_data
        if event.replica_id is None:
            return None

        if event.event == "on_acquire":
            return self._parse_acquire(event, turn=turn, stale=stale, block_size=block_size, booked_len=booked_len)
        if event.event == "on_release":
            return self._parse_release(event, turn=turn, booked_len=booked_len)
        return None

    # ── on_acquire ───────────────────────────────────────────────────────

    def _parse_acquire(
        self,
        event: StatisticEvent,
        *,
        turn: int,
        stale: dict[str, Any] | None,
        block_size: int,
        booked_len: int,
    ) -> list:
        """Netted acquire: optional stale unwind + pin command + metric deltas."""
        plen = len(event.prompt_ids)
        updates: list[InflightBlockUpdate | MetricsUpdate] = []
        metrics: dict[str, int] = {
            MetricKey.INFLIGHT_COUNT: 1,
            MetricKey.INFLIGHT_TOKENS: plen,
            MetricKey.DISPATCHED_COUNT: 1,
            MetricKey.PROMPT_LEN_SUM: plen,
        }
        if event.request_id is None:
            logger.debug(
                "dispatch (on_acquire) missing request_id — skipping turn and block pinning (raw token count booked)"
            )
            return [MetricsUpdate(node_id=event.replica_id, metrics=metrics, is_delta=True)]

        metrics[MetricKey.INFLIGHT_TURN_SUM] = turn
        # A still-held marker means the previous dispatch never released (a
        # dropped COMPLETED_COUNT) — emit the unwind and net its still-booked
        # tokens out of this acquire.
        if stale is not None:
            logger.warning(
                f"re-acquire without release request={event.request_id} on {stale['replica']} "
                f"— unpinning the previous block set"
            )
            updates.append(
                InflightBlockUpdate(node_id=event.replica_id, request_id=event.request_id, action="stale_unpin")
            )
            metrics[MetricKey.INFLIGHT_TOKENS] = plen - booked_len
            # The dropped release's INFLIGHT_COUNT -1 never arrived either; fold
            # it into this acquire so the gauge nets to the one turn genuinely
            # in flight.
            metrics[MetricKey.INFLIGHT_COUNT] = metrics.get(MetricKey.INFLIGHT_COUNT, 0) - 1

        # Resolve the dispatch's prefix chain, computed fresh (the chain is
        # deterministic — no memo). Nothing resolvable (no prompt, block size
        # not yet learned, or a sub-block prompt) pins nothing.
        hash_strs = self._resolve_chain(event, block_size)
        if hash_strs:
            updates.append(
                InflightBlockUpdate(
                    node_id=event.replica_id,
                    request_id=event.request_id,
                    action="pin",
                    block_hashes=hash_strs,
                    mark_resident=True,
                )
            )
        updates.append(
            MetricsUpdate(node_id=event.replica_id, metrics=metrics, is_delta=True, request_id=event.request_id)
        )
        return updates

    # ── on_release ───────────────────────────────────────────────────────

    def _parse_release(self, event: StatisticEvent, *, turn: int, booked_len: int) -> list:
        """Netted release: unpin command + metric deltas folded from acquire-time rows."""
        metrics: dict[str, int] = {
            MetricKey.INFLIGHT_COUNT: -1,
            MetricKey.COMPLETED_COUNT: 1,
        }
        updates: list[InflightBlockUpdate | MetricsUpdate] = []
        if event.request_id is None:
            logger.debug("release (on_release) missing request_id — skipping turn/token/block subtraction")
        else:
            updates.append(InflightBlockUpdate(node_id=event.replica_id, request_id=event.request_id, action="unpin"))
            metrics[MetricKey.INFLIGHT_TURN_SUM] = -turn
            metrics[MetricKey.INFLIGHT_TOKENS] = -booked_len
        updates.append(
            MetricsUpdate(node_id=event.replica_id, metrics=metrics, is_delta=True, request_id=event.request_id)
        )
        return updates

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_chain(self, event: StatisticEvent, block_size: int) -> list[str]:
        """Resolve the event prompt's full-block prefix hash chain, computed fresh.

        Stateless sibling of ``resolve_prefix_hashes`` (which memoizes the
        per-request checkpoint in the store): the chain is deterministic, so it
        is simply recomputed per dispatch. Returns ``[]`` when there is no
        prompt or no learned block size.
        """
        if not event.prompt_ids or block_size <= 0:
            return []
        hashes, _ = get_prefix_hashes_incremental(list(event.prompt_ids), block_size, 0, 0)
        return [str(h) for h in hashes]
