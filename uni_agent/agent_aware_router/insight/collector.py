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

"""Observe successful collector updates without owning any store writes."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

from ..collectors.parse import KVCacheUpdate, MetricsUpdate
from ..store.contracts import KVReader, MetricsReader
from ..types import EmitKey, MetricKey
from . import WriteEvent, WriteKind, emitter

logger = logging.getLogger(__name__)

_METRICS_LOG_INTERVAL_S = 30
_DISPATCH_LOG_INTERVAL_S = 5.0
_KV_EVENT_LOG_INTERVAL_S = 500


# Cumulative metrics tracked for windowed deltas in the evidence log. Single
# source of truth — ``_delta`` consumers below read these by key, and the
# per-replica prev-snapshot iterates the same tuple.
_CUMULATIVE_KEYS: tuple[str, ...] = (
    MetricKey.TTFT_SECONDS_SUM,
    MetricKey.TTFT_COUNT,
    MetricKey.QUEUE_TIME_SECONDS_SUM,
    MetricKey.QUEUE_TIME_COUNT,
    MetricKey.TPOT_SECONDS_SUM,
    MetricKey.TPOT_COUNT,
    MetricKey.PROMPT_TOKENS,
    MetricKey.PROMPT_TOKENS_CACHED,
    MetricKey.GENERATION_TOKENS,
    MetricKey.EXTERNAL_PREFIX_CACHE_HITS,
    MetricKey.ESTIMATED_FLOPS_PER_GPU,
)


def _avg(delta_sum: float, delta_cnt: float) -> float:
    """Windowed average = delta_sum / delta_cnt, or NaN if no samples."""
    return delta_sum / delta_cnt if delta_cnt > 0 else float("nan")


def _ms(value: float) -> str:
    """Format a seconds value as millis for the evidence log ('-' if NaN)."""
    return f"{value * 1000:.1f}" if value == value else "-"


class UpdateObserver(Protocol):
    def on_kv_applied(self, update: KVCacheUpdate) -> None: ...

    def on_metrics_applied(self, update: MetricsUpdate, values: dict[str, Any]) -> None: ...

    def on_blocks_recorded(self, count: int) -> None: ...


class CollectorObserver:
    """Per-collector evidence windows and insight events, emitted after application."""

    def __init__(
        self,
        metrics: MetricsReader,
        kv: KVReader,
        parser_stats: Callable[[], Any] = lambda: None,
    ) -> None:
        self._metrics = metrics
        self._kv = kv
        self._parser_stats = parser_stats
        self._metrics_poll_count = 0
        self._metrics_prev: dict[str, dict[str, float]] = {}
        self._kv_event_counts: dict[str, int] = defaultdict(int)
        self._kv_block_counts: dict[str, int] = defaultdict(int)
        self._kv_last_logged_total = 0
        self._dispatch_last_log = 0.0
        self._dispatch_recorded_blocks = 0

    def on_blocks_recorded(self, count: int) -> None:
        self._dispatch_recorded_blocks += count

    def on_kv_applied(self, update: KVCacheUpdate) -> None:
        # Tally for periodic summary — observe BlockStored/BlockRemoved flow.
        n_added = sum(len(v) for v in update.add_blocks.values())
        n_removed = sum(len(v) for v in update.remove_blocks.values())
        if update.clear_all:
            self._kv_event_counts["clear"] += 1
        if n_added:
            self._kv_event_counts["stored"] += 1
            self._kv_block_counts["stored"] += n_added
        if n_removed:
            self._kv_event_counts["removed"] += 1
            self._kv_block_counts["removed"] += n_removed
            emitter.on_write(
                WriteEvent(
                    kind=WriteKind.KV_REMOVED,
                    node=update.node_id,
                    deltas={EmitKey.KV_EVICTIONS: n_removed},
                )
            )
        total = sum(self._kv_event_counts.values())
        if total - self._kv_last_logged_total >= _KV_EVENT_LOG_INTERVAL_S:
            self._kv_last_logged_total = total
            # Parser-side translation failures: non-zero means the event-side
            # retained count is drifting above the engine's truth.
            hash_map_stats = self._parser_stats()
            logger.debug(
                f"kv-events tally: events={dict(self._kv_event_counts)} "
                f"blocks={dict(self._kv_block_counts)} (total_events={total}) | "
                f"retained_blocks/replica={self._kv.per_replica_block_counts()} | "
                f"hash-map={dict(hash_map_stats or {})}"
            )

    def on_metrics_applied(self, update: MetricsUpdate, values: dict[str, Any]) -> None:
        if update.is_delta:
            is_acquire = MetricKey.DISPATCHED_COUNT in update.metrics
            is_release = MetricKey.COMPLETED_COUNT in update.metrics
            is_unwind = not is_acquire and update.metrics.get(MetricKey.INFLIGHT_COUNT, 0) < 0
            if is_acquire or is_release or is_unwind:
                # Stale-dispatch corrections refresh the old replica's gauges
                # without inventing a completion counter increment.
                emitter.on_write(
                    WriteEvent(
                        kind=WriteKind.ACQUIRE if is_acquire else WriteKind.RELEASE,
                        node=update.node_id,
                        deltas=update.metrics,
                        new_values=values,
                        turn_sum=values.get(MetricKey.INFLIGHT_TURN_SUM),
                        inflight_count=values.get(MetricKey.INFLIGHT_COUNT),
                    )
                )
            self._maybe_log_dispatch_stats()
            return
        emitter.on_write(
            WriteEvent(
                kind=WriteKind.POLL,
                node=update.node_id,
                new_values=values,
                load=self._kv.kv_cache_load(update.node_id),
            )
        )

        # Periodic visibility into what the collector fed the router — compare
        # against vllm's own "GPU KV cache usage" engine-stats log line.
        self._metrics_poll_count += 1
        if self._metrics_poll_count % _METRICS_LOG_INTERVAL_S == 0:
            # Emit evidence for ALL known replicas, not just the one that happened
            # to be polled at this poll-count tick. Metrics polling is serial
            # (one replica per poll), so emitting only ``update.node_id`` here
            # sampled ~1/N of replicas per window → some replicas never got an
            # evidence line (e.g. 4/8 seen). Each replica keeps its own
            # ``_metrics_prev`` baseline, so windowed deltas stay correct.
            for nid in self._metrics.get_metric_node_ids():
                self._log_evidence_window(nid)

    def _maybe_log_dispatch_stats(self) -> None:
        """Emit per-replica dispatched/completed/inflight_turn_sum/prompt_len_sum counters at most every interval.

        Reads each dispatched replica's cumulative counters from PerReplicaStore and
        logs them (the ``router-dispatch`` line); the plot derives trailing-5-min
        dispatched / completed / avg-turn / RPM / avg-prompt-len from their per-replica
        deltas. Time-throttled so the cadence is load-independent (idle stretches emit nothing).
        """
        now = time.monotonic()
        if now - self._dispatch_last_log < _DISPATCH_LOG_INTERVAL_S:
            return
        self._dispatch_last_log = now
        logger.debug(f"router-dispatch blocks: router_recorded={self._dispatch_recorded_blocks}")
        for rep in self._metrics.get_metric_node_ids():
            snap = self._metrics.get_metrics(rep)
            dispatched = snap.get(MetricKey.DISPATCHED_COUNT, 0)
            if not dispatched:  # skip replicas that never received a dispatch
                continue
            completed = snap.get(MetricKey.COMPLETED_COUNT, 0)
            inflight_turn_sum = snap.get(MetricKey.INFLIGHT_TURN_SUM, 0)
            prompt_len_sum = snap.get(MetricKey.PROMPT_LEN_SUM, 0)
            logger.info(
                f"router-dispatch replica={rep} dispatched={dispatched} completed={completed} "
                f"inflight_turn_sum={inflight_turn_sum} prompt_len_sum={prompt_len_sum}"
            )

    def _log_evidence_window(self, node_id: str) -> None:
        """Emit a windowed evidence summary for one replica.

        Deltas vs the previous snapshot, over ~``_METRICS_LOG_INTERVAL_S``
        polls (≈30 s). Reads the merged store snapshot (not the per-poll
        update) so a transiently-missing scrape line doesn't zero a
        cumulative counter.
        """
        snap = self._metrics.get_metrics(node_id)
        prev = self._metrics_prev.get(node_id, {})

        def _delta(key: str) -> float:
            cur = float(snap.get(key, 0) or 0)
            return cur - float(prev.get(key, cur) or 0)

        # kv = retained blocks (cached-freeable + running-with-hash); usage =
        # vLLM's running-only fraction. Emit both: cache-fill (kv) vs running
        # pressure (usage).
        kv = self._kv.kv_cache_load(node_id)
        usage_raw = snap.get(MetricKey.KV_CACHE_USAGE_PERC)
        run = snap.get(MetricKey.NUM_REQUESTS_RUNNING)
        wait = snap.get(MetricKey.NUM_REQUESTS_WAITING)

        # Windowed TTFT/queue/TPOT averages (delta_sum / delta_count).
        ttft_avg = _avg(_delta(MetricKey.TTFT_SECONDS_SUM), _delta(MetricKey.TTFT_COUNT))
        queue_avg = _avg(_delta(MetricKey.QUEUE_TIME_SECONDS_SUM), _delta(MetricKey.QUEUE_TIME_COUNT))
        # prefill_time = TTFT - queue (TTFT includes the queue wait).
        prefill_t = (ttft_avg - queue_avg) if (ttft_avg == ttft_avg and queue_avg == queue_avg) else float("nan")
        tpot_avg = _avg(_delta(MetricKey.TPOT_SECONDS_SUM), _delta(MetricKey.TPOT_COUNT))

        # Token deltas over the window (prefill computed vs cached, decode, external).
        d_prefill = _delta(MetricKey.PROMPT_TOKENS)
        d_cached = _delta(MetricKey.PROMPT_TOKENS_CACHED)
        d_decode = _delta(MetricKey.GENERATION_TOKENS)
        d_external = _delta(MetricKey.EXTERNAL_PREFIX_CACHE_HITS)
        d_flops = _delta(MetricKey.ESTIMATED_FLOPS_PER_GPU)
        cache_hit_pct = 100.0 * d_cached / (d_cached + d_prefill) if (d_cached + d_prefill) > 0 else float("nan")

        kv_str = f"{kv:.3f}" if isinstance(kv, float) else kv
        usage_str = f"{float(usage_raw):.3f}" if usage_raw is not None else "-"
        hit_str = f"{cache_hit_pct:.1f}" if cache_hit_pct == cache_hit_pct else "-"
        logger.debug(
            f"vllm-evidence replica={node_id} kv={kv_str} usage={usage_str} run={run} wait={wait} | "
            f"TTFT={_ms(ttft_avg)}ms queue={_ms(queue_avg)}ms prefillT={_ms(prefill_t)}ms TPOT={_ms(tpot_avg)}ms | "
            f"prefill={int(d_prefill)} cached={int(d_cached)} (hit={hit_str}%) "
            f"decode={int(d_decode)} external={int(d_external)} flops={int(d_flops)} [poll #{self._metrics_poll_count}]"
        )

        # Snapshot current cumulative values for next window's delta.
        self._metrics_prev[node_id] = {k: float(snap.get(k, 0) or 0) for k in _CUMULATIVE_KEYS}
