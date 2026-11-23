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
from ....types.kv_cache import DispatchSchema, GroupReservation, InflightBooking
from ....utils.hash import get_prefix_hashes_incremental

logger = logging.getLogger(__name__)


class InflightParser(Parser):
    """Pure acquire/release protocol decisions → netted deltas + block commands.

    Args:
        raw_data: The raw callback payload — an ``on_acquire`` / ``on_release``
            ``StatisticEvent``; anything else is ignored (``None``).
        replica_id: Source endpoint id (empty for the callback transport — the
            event carries its own replica).
        turn: The request's turn after the collector's increment (acquire) or
            its stored turn (release); 0 when unattributable.
        booking: The active dispatch's replica, prompt length, and turn.
        schema: Hash identity and reservation units captured by the collector.
    """

    def parse(
        self,
        raw_data: bytes | str | Any,
        replica_id: str,
        *,
        turn: int = 0,
        booking: InflightBooking | None = None,
        schema: DispatchSchema | None = None,
    ) -> list | None:
        """Dispatch on acquire/release; ignore other events / non-event payloads."""
        if not isinstance(raw_data, StatisticEvent):
            return None

        event = raw_data
        if event.replica_id is None:
            return None

        if event.event == "on_acquire":
            return self._parse_acquire(event, turn=turn, booking=booking, schema=schema)
        if event.event == "on_release":
            return self._parse_release(event, booking=booking)
        return None

    # ── on_acquire ───────────────────────────────────────────────────────

    def _parse_acquire(
        self,
        event: StatisticEvent,
        *,
        turn: int,
        booking: InflightBooking | None,
        schema: DispatchSchema | None,
    ) -> list:
        """Netted acquire: optional stale unwind + pin command + metric deltas."""
        plen = len(event.prompt_ids or ())
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
            return [MetricsUpdate(replica_id=event.replica_id, metrics=metrics, is_delta=True)]

        metrics[MetricKey.INFLIGHT_TURN_SUM] = turn
        if booking is not None:
            logger.warning(
                f"re-acquire without release request={event.request_id} on {booking.replica_id} "
                "— unpinning the previous block set"
            )
            updates.append(
                InflightBlockUpdate(replica_id=booking.replica_id, request_id=event.request_id, action="stale_unpin")
            )
            unwind = {
                MetricKey.INFLIGHT_COUNT: -1,
                MetricKey.INFLIGHT_TOKENS: -booking.prompt_len,
                MetricKey.INFLIGHT_TURN_SUM: -booking.turn,
            }
            if booking.replica_id == event.replica_id:
                for key, delta in unwind.items():
                    metrics[key] += delta
            else:
                updates.append(MetricsUpdate(replica_id=booking.replica_id, metrics=unwind, is_delta=True))

        # Resolve this dispatch's prefix chain and reservations. Missing
        # allocation metadata retains an unknown reservation until release.
        hash_strs = self._resolve_chain(event, schema)
        reservations = self._resolve_reservations(event, schema, hash_strs)
        updates.append(
            InflightBlockUpdate(
                replica_id=event.replica_id,
                request_id=event.request_id,
                action="pin",
                block_hashes=hash_strs,
                mark_resident=True,
                schema=schema,
                booking=InflightBooking(event.replica_id, plen, turn),
                reservations=reservations,
            )
        )
        updates.append(
            MetricsUpdate(replica_id=event.replica_id, metrics=metrics, is_delta=True, request_id=event.request_id)
        )
        return updates

    # ── on_release ───────────────────────────────────────────────────────

    def _parse_release(self, event: StatisticEvent, *, booking: InflightBooking | None) -> list:
        """Netted release: unpin command + metric deltas folded from acquire-time rows."""
        metrics: dict[str, int] = {
            MetricKey.INFLIGHT_COUNT: -1,
            MetricKey.COMPLETED_COUNT: 1,
        }
        updates: list[InflightBlockUpdate | MetricsUpdate] = []
        if event.request_id is None:
            logger.debug("release (on_release) missing request_id — skipping turn/token/block subtraction")
        elif booking is None or booking.replica_id != event.replica_id:
            # Count the release callback without discharging another replica's booking.
            del metrics[MetricKey.INFLIGHT_COUNT]
        else:
            updates.append(
                InflightBlockUpdate(replica_id=event.replica_id, request_id=event.request_id, action="unpin")
            )
            metrics[MetricKey.INFLIGHT_TURN_SUM] = -booking.turn
            metrics[MetricKey.INFLIGHT_TOKENS] = -booking.prompt_len
        updates.append(
            MetricsUpdate(replica_id=event.replica_id, metrics=metrics, is_delta=True, request_id=event.request_id)
        )
        return updates

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_chain(self, event: StatisticEvent, schema: DispatchSchema | None) -> list[str]:
        """Hash once at the registry unit, then select reservation endpoints."""
        if not event.prompt_ids or schema is None or schema.hash_unit is None:
            return []
        hashes, _ = get_prefix_hashes_incremental(
            list(event.prompt_ids), schema.hash_unit, 0, 0, lookahead=schema.lookahead_tokens
        )
        return [str(h) for h in hashes]

    def _resolve_reservations(self, event, schema, hash_strs):
        if schema is None:
            return ()
        if event.prompt_ids is None:
            # Shared block demand depends on the missing prompt. Fixed private
            # budgets remain known, and release still unpins the whole request.
            return tuple(
                GroupReservation(group.group_idx, private_blocks=None if group.share_prefix else group.private_blocks)
                for group in schema.groups
            )
        if not event.prompt_ids:
            return ()
        chains = {schema.hash_unit: tuple(hash_strs)}
        groups = []
        for group in schema.groups:
            hashes = ()
            private = group.private_blocks
            if group.share_prefix:
                if schema.hash_unit is None:
                    if private is not None:
                        private += len(event.prompt_ids) // group.block_size
                else:
                    if group.block_size not in chains:
                        values, _ = get_prefix_hashes_incremental(
                            list(event.prompt_ids), group.block_size, 0, 0, lookahead=schema.lookahead_tokens
                        )
                        chains[group.block_size] = tuple(map(str, values))
                    hashes = chains[group.block_size]
                    if schema.lookahead_tokens and private is not None:
                        # A full prompt block waiting for its suffix is private to this dispatch.
                        private += len(event.prompt_ids) // group.block_size - len(hashes)
            groups.append(GroupReservation(group.group_idx, hashes, private))
        return tuple(groups)
