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

"""A-class end-to-end: strategy ``score()`` → emitter ``on_score`` / ``on_route`` → rl_insight.

With emit on, a recording rl_insight double captures the strategy-component
histograms (load/s_cache or the capacity fractions) and the per-call
route_latency sample — verifying the 6 A-class signals across both slow_cut
modes and every ``score()`` return path (including the sticky short-circuit).
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from uni_agent import rl_insight as facade
from uni_agent.agent_aware_router.store.data_store import DataStore
from uni_agent.agent_aware_router.store.per_replica_store import PerReplicaStore
from uni_agent.agent_aware_router.store.per_request_store import PerRequestStore
from uni_agent.agent_aware_router.strategies.base import ReplicaInfo
from uni_agent.agent_aware_router.strategies.kvc_aware import KVCacheAwareStrategy
from uni_agent.agent_aware_router.types import MetricKey, SlowCut

pytestmark = [pytest.mark.level0, pytest.mark.cpu]

PROMPT_IDS = [1, 2, 3]


class _RecordingRLInsight:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def init(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - test double
        """No-op stand-in for rl_insight.init."""

    def metric_count(self, name, amount, documentation="", **labels):
        self.calls.append(("counter", name, amount, dict(labels)))

    def metric_gauge(self, name, value, documentation="", **labels):
        self.calls.append(("gauge", name, value, dict(labels)))

    def metric_histogram(self, name, value, documentation="", *, buckets=None, **labels):
        self.calls.append(("histogram", name, value, dict(labels)))


def _strat(**kwargs) -> KVCacheAwareStrategy:
    defaults = dict(
        alpha=0.7,
        load_threshold=0.9,
        layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        load_weights=(0.4, 0.2, 0.1, 0.3),
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> _RecordingRLInsight:
    PerReplicaStore._instance = None
    PerRequestStore._instance = None
    monkeypatch.setenv(facade.ENABLE_ENV, "1")
    monkeypatch.setattr(facade, "_enabled", True)  # pre-flip gate cache
    rl = _RecordingRLInsight()
    monkeypatch.setitem(sys.modules, "rl_insight", rl)
    yield rl
    PerReplicaStore._instance = None
    PerRequestStore._instance = None


def _names(rl: _RecordingRLInsight, metric_type: str | None = None) -> set[str]:
    return {c[1] for c in rl.calls if metric_type is None or c[0] == metric_type}


def test_prefix_load_aware_emits_load_and_scache_per_replica(recording):
    ds = DataStore()
    ds.refresh_metrics({"s0": {MetricKey.NUM_REQUESTS_RUNNING: 1, MetricKey.NUM_GPU_BLOCKS: 10}})
    strat = _strat(slow_cut=SlowCut.PREFIX_LOAD_AWARE, do_shortcut=False)

    strat.score(PROMPT_IDS, ds, [ReplicaInfo(replica_id="s0")])

    assert {"load", "s_cache"} <= _names(recording, "histogram")
    # components carry the replica label
    load_call = next(c for c in recording.calls if c[1] == "load")
    assert load_call[3] == {"replica": "s0"}


def test_capacity_token_aware_emits_capacity_fractions(recording):
    ds = DataStore()
    ds.set_block_size(16)
    ds.refresh_metrics({"s0": {MetricKey.NUM_GPU_BLOCKS: 100, MetricKey.KV_CACHE_USAGE_PERC: 0.2}})
    strat = _strat(slow_cut=SlowCut.CAPACITY_TOKEN_AWARE, do_shortcut=False)

    strat.score(PROMPT_IDS, ds, [ReplicaInfo(replica_id="s0")])

    # cap = 100 blocks × 16 = 1600 tokens; kv_perc=0.2 → avail=1280 → ratio 0.8.
    # Components are emitted as fractions of capacity (default-bucket friendly).
    assert {"avail_ratio", "need_ratio", "remaining_ratio"} <= _names(recording, "histogram")
    avail_call = next(c for c in recording.calls if c[1] == "avail_ratio")
    assert avail_call[2] == pytest.approx(0.8)
    assert avail_call[3] == {"replica": "s0"}


def test_capacity_token_aware_skips_emit_without_capacity(recording):
    # NUM_GPU_BLOCKS not yet scraped → cap=0, raw token values are garbage:
    # routing still happens, but no capacity fractions are emitted.
    ds = DataStore()
    ds.set_block_size(16)
    strat = _strat(slow_cut=SlowCut.CAPACITY_TOKEN_AWARE, do_shortcut=False)

    strat.score(PROMPT_IDS, ds, [ReplicaInfo(replica_id="s0")])

    assert not (_names(recording, "histogram") & {"avail_ratio", "need_ratio", "remaining_ratio"})


def test_route_latency_fires_once_per_score_call_with_no_replica_label(recording):
    ds = DataStore()
    ds.refresh_metrics({"s0": {MetricKey.NUM_REQUESTS_RUNNING: 1, MetricKey.NUM_GPU_BLOCKS: 10}})
    strat = _strat(slow_cut=SlowCut.PREFIX_LOAD_AWARE, do_shortcut=False)

    strat.score(PROMPT_IDS, ds, [ReplicaInfo(replica_id="s0")])

    route_calls = [c for c in recording.calls if c[1] == "route_latency_seconds"]
    assert len(route_calls) == 1  # exactly one sample per score() call
    assert route_calls[0][0] == "histogram"
    assert route_calls[0][3] == {}  # global — no replica label


def test_route_latency_fires_even_on_sticky_short_circuit(recording):
    ds = DataStore()
    ds.refresh_metrics({"s0": {MetricKey.NUM_REQUESTS_RUNNING: 0, MetricKey.NUM_GPU_BLOCKS: 10}})
    ds.put_sticky_binding("r1", "s0")
    strat = _strat(slow_cut=SlowCut.PREFIX_LOAD_AWARE, do_shortcut=True)

    strat.score(PROMPT_IDS, ds, [ReplicaInfo(replica_id="s0")], request_id="r1")

    # sticky short-circuit returns before component scoring, but the try/finally
    # still records one route_latency sample — and no score components.
    assert len([c for c in recording.calls if c[1] == "route_latency_seconds"]) == 1
    assert "load" not in _names(recording)
