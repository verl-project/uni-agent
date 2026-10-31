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

"""InflightParser — Balancer callback → netted metric deltas + block commands.

The parser owns the acquire/release protocol for the router's in-flight
accounts. It is the only business-logic layer in the statistic chain: the
store is a pure CRUD carrier and the collector applies the parser's output
mechanically.

Two access tiers keep the layering clean:

- **Scratch state** (per-request rows in ``PerRequestStore`` and the hash-chain
  memo) is read and written directly through the injected ``DataStore`` —
  these are caches/drafts with no cross-component protocol meaning. The turn
  counter, the dispatch prompt-length row, and ``resolve_prefix_hashes``'s
  memo live here (the memo's incremental-extend optimization needs the live
  row reference, so it cannot round-trip through commands).
- **Protocol state** (held-block reference counts, pin markers, per-replica
  metrics) is only ever emitted as commands: :class:`InflightBlockUpdate`
  objects the collector applies, plus fully-netted ``MetricsUpdate`` deltas.
  The parser reads protocol state (``get_marker`` peek) but never writes it.

Mirrors verl ``GlobalRequestLoadBalancer._inflight_requests``: acquire bumps
the chosen replica's in-flight count by +1, release decrements it by -1.

Beyond the running inflight gauge, acquire/release also bump the per-replica
cumulative DISPATCHED_COUNT / COMPLETED_COUNT counters — the monotonic
siblings of the gauge that carry dispatch / completion volume per replica.
Summed across replicas they give the global dispatched/completed totals; their
ratio is the realized-throughput share.

``INFLIGHT_TOKENS`` is the token-weighted sibling of the inflight gauge: acquire
adds the dispatched prompt's token count; releases carry no token list (verl
#7115), so the release-side negative delta is folded from the acquire-time
per-request row.
``INFLIGHT_TURN_SUM`` likewise adds the request's current turn on acquire and
subtracts the same turn on release (release changes neither), keeping the
per-replica turn-weighted sum balanced.

Held-block account: on acquire the parser resolves the request's prefix hash
chain and emits a ``pin`` command (the collector ref-counts the blocks, marks
them resident so ``gpu_hit`` sees the prefix at once, and snapshots the exact
set as the request's marker row). On release it emits ``unpin`` — the collector
unwinds the marker row, since releases carry no prompt under verl #7115 and the
memo may have been replaced in between. A marker still present at re-acquire
means the previous dispatch never released (a dropped release): the parser
emits ``stale_unpin`` and nets the stale dispatch's booked tokens / count out
of this acquire's deltas.

Event → output mapping (a list, applied in order):
- ``on_acquire`` → optional ``stale_unpin`` + optional ``pin`` +
  ``MetricsUpdate``: ``INFLIGHT_COUNT``/``DISPATCHED_COUNT`` +1,
  ``INFLIGHT_TOKENS``/``PROMPT_LEN_SUM`` +prompt length,
  ``INFLIGHT_TURN_SUM`` +turn — all netted. Without ``request_id`` only the
  raw gauge/count deltas are emitted (nothing to attribute turn/pins to).
- ``on_release`` → ``unpin`` + ``MetricsUpdate``: ``INFLIGHT_COUNT`` -1,
  ``COMPLETED_COUNT`` +1, ``INFLIGHT_TOKENS``/``INFLIGHT_TURN_SUM`` subtracted
  from the acquire-time rows.

``on_servers_removed`` is intentionally a no-op (returns ``None``): verl never
removes servers and maintains ``_inflight_requests`` purely via symmetric
acquire/release. Clearing on removal would be unsafe — in-flight requests for
a removed replica still complete and fire their ``on_release`` (-1), which
would drive the counter negative if we'd zeroed it on removal. The cumulative
counts are likewise left intact (they are lifetime totals). Faithful
simulation = don't touch the counters on removal.
"""

from __future__ import annotations

import logging
from typing import Any

from ....collectors.parse import InflightBlockUpdate, MetricsUpdate, Parser
from ....collectors.transport.callback import StatisticEvent
from ....store.data_store import DataStore
from ....types import MetricKey
from ....utils.prefix_cache import resolve_prefix_hashes

logger = logging.getLogger(__name__)

# Per-request bookkeeping key: prompt length recorded at dispatch. verl #7115
# releases carry no token list, so the release-side ``INFLIGHT_TOKENS`` delta
# is folded from this row — same acquire-record / release-consume shape as the
# per-request "turn" counter.
_PROMPT_LEN_KEY = "prompt_len"


class InflightParser(Parser):
    """Parse ``StatisticEvent`` → netted inflight deltas + held-block commands.

    Args:
        store: ``DataStore`` for the parser's scratch state (per-request turn /
            prompt-length rows, hash-chain memo). Protocol state is never
            written here — it travels to the collector as commands.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store

    def parse(self, raw_data: bytes | str | Any, node_id: str) -> list | None:
        """Dispatch on acquire/release; ignore other events / non-event payloads."""
        if not isinstance(raw_data, StatisticEvent):
            return None

        event = raw_data
        if event.replica_id is None:
            return None

        if event.event == "on_acquire":
            return self._parse_acquire(event)
        if event.event == "on_release":
            return self._parse_release(event)
        return None

    # ── on_acquire ───────────────────────────────────────────────────────

    def _parse_acquire(self, event: StatisticEvent) -> list:
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

        # Scratch: the dispatch's turn feeds the replica's INFLIGHT_TURN_SUM.
        metrics[MetricKey.INFLIGHT_TURN_SUM] = self._store.incr_per_request(event.request_id, "turn")
        # Protocol read: a marker still held means the previous dispatch never
        # released (a dropped COMPLETED_COUNT) — emit the unwind and net its
        # still-booked tokens out of this acquire.
        stale = self._store.get_marker(event.request_id)
        if stale is not None:
            logger.warning(
                f"re-acquire without release request={event.request_id} on {stale['replica']} "
                f"— unpinning the previous block set"
            )
            updates.append(
                InflightBlockUpdate(node_id=event.replica_id, request_id=event.request_id, action="stale_unpin")
            )
            metrics[MetricKey.INFLIGHT_TOKENS] = plen - self._store.get_per_request(
                event.request_id, _PROMPT_LEN_KEY, 0
            )
            # The dropped release's INFLIGHT_COUNT -1 never arrived either; fold
            # it into this acquire so the gauge nets to the one turn genuinely
            # in flight.
            metrics[MetricKey.INFLIGHT_COUNT] = metrics.get(MetricKey.INFLIGHT_COUNT, 0) - 1
        self._store.set_per_request(event.request_id, _PROMPT_LEN_KEY, plen)

        # Scratch: resolve the dispatch's prefix chain (memoized per request).
        # Nothing resolvable (no prompt, block size not yet learned, or a
        # sub-block prompt) pins nothing.
        hash_strs = (
            resolve_prefix_hashes(list(event.prompt_ids), event.request_id, self._store) if event.prompt_ids else []
        )
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

    def _parse_release(self, event: StatisticEvent) -> list:
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
            metrics[MetricKey.INFLIGHT_TURN_SUM] = -self._store.get_per_request(event.request_id, "turn", 0)
            metrics[MetricKey.INFLIGHT_TOKENS] = -self._store.get_per_request(event.request_id, _PROMPT_LEN_KEY, 0)
        updates.append(
            MetricsUpdate(node_id=event.replica_id, metrics=metrics, is_delta=True, request_id=event.request_id)
        )
        return updates
