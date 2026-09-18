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

"""CPU unit tests for the kvc-aware router rl-insight emitter.

A fake rl_insight records every metric_* call so the tests assert the exact
primitive → rl_insight-API mapping driven by ``EMIT_SPECS``, without a live
rl-insight server or Ray. Covers env-gating, the type fan-out, both A-class
mount points (``on_score``/``on_route``) and all four ``on_write`` kinds.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from uni_agent import rl_insight as facade
from uni_agent.agent_aware_router.insight.emitter import (
    WriteEvent,
    WriteKind,
    emitter,
)
from uni_agent.agent_aware_router.types.emit_spec import EMIT_SPECS, EmitKey

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


class FakeRLInsight:
    """Records metric_* calls as uniform dicts."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def init(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - test double
        """No-op stand-in for rl_insight.init."""

    def metric_count(self, name, amount, documentation="", **labels):
        self.calls.append(
            {
                "method": "counter",
                "name": name,
                "value": amount,
                "doc": documentation,
                "labels": dict(labels),
            }
        )

    def metric_gauge(self, name, value, documentation="", **labels):
        self.calls.append(
            {
                "method": "gauge",
                "name": name,
                "value": value,
                "doc": documentation,
                "labels": dict(labels),
            }
        )

    def metric_histogram(self, name, value, documentation="", **labels):
        self.calls.append(
            {
                "method": "histogram",
                "name": name,
                "value": value,
                "doc": documentation,
                "labels": dict(labels),
            }
        )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    """Start each test with emit off and a clean gate cache; restore after."""
    monkeypatch.delenv(facade.ENABLE_ENV, raising=False)
    monkeypatch.setattr(facade, "_enabled", None)
    yield


@pytest.fixture
def enabled_emitter(monkeypatch: pytest.MonkeyPatch) -> FakeRLInsight:
    """Enable emit and inject a recording fake as the rl_insight module.

    The fake goes into ``sys.modules`` so the facade's real gate → import →
    init path runs; only the module itself is the test double.
    """
    monkeypatch.setenv(facade.ENABLE_ENV, "1")
    monkeypatch.setattr(facade, "_enabled", True)  # pre-flip gate cache
    fake = FakeRLInsight()
    monkeypatch.setitem(sys.modules, "rl_insight", fake)
    yield fake


def _by_name(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in calls}


# ── env gating ───────────────────────────────────────────────────────


def test_disabled_short_circuits_before_importing_rl_insight(monkeypatch: pytest.MonkeyPatch):
    # env off — the facade must short-circuit before it ever imports rl_insight.
    monkeypatch.delenv(facade.ENABLE_ENV, raising=False)
    monkeypatch.setattr(facade, "_enabled", None)
    assert facade._get_rl_insight() is None
    emitter.on_score("n0", {"load": 0.5})
    emitter.on_route(0.1)
    emitter.on_write(WriteEvent(kind=WriteKind.KV_REMOVED, node="n0", deltas={EmitKey.KV_EVICTIONS: 1}))


# ── A-class: on_score / on_route ─────────────────────────────────────


def test_on_score_emits_each_component_with_replica_label(
    enabled_emitter: FakeRLInsight,
):
    emitter.on_score("n0", {"load": 0.3, "s_cache": 0.7})

    names = [c["name"] for c in enabled_emitter.calls]
    assert names == [EmitKey.LOAD, EmitKey.S_CACHE]
    assert all(c["method"] == "histogram" for c in enabled_emitter.calls)
    assert all(c["labels"] == {"replica": "n0"} for c in enabled_emitter.calls)
    assert _by_name(enabled_emitter.calls)[EmitKey.LOAD]["value"] == 0.3


def test_on_score_skips_unknown_component_keys(enabled_emitter: FakeRLInsight):
    emitter.on_score("n0", {"load": 0.3, "not_a_metric": 9.9})

    assert [c["name"] for c in enabled_emitter.calls] == [EmitKey.LOAD]


def test_on_route_emits_global_latency_histogram(enabled_emitter: FakeRLInsight):
    emitter.on_route(0.005)

    assert len(enabled_emitter.calls) == 1
    call = enabled_emitter.calls[0]
    assert (call["method"], call["name"], call["value"]) == (
        "histogram",
        EmitKey.ROUTE_LATENCY_SECONDS,
        0.005,
    )
    assert call["labels"] == {}  # global — no replica label


# ── B-class: on_write by kind ────────────────────────────────────────


def test_on_write_poll_emits_levels_cumulatives_and_load(
    enabled_emitter: FakeRLInsight,
):
    new_values = {
        EmitKey.KV_CACHE_USAGE_PERC: 0.5,
        EmitKey.NUM_REQUESTS_RUNNING: 2,
        EmitKey.NUM_REQUESTS_WAITING: 1,
        EmitKey.PROMPT_TOKENS: 1000,
        EmitKey.PROMPT_TOKENS_CACHED: 200,
        EmitKey.EXTERNAL_PREFIX_CACHE_HITS: 50,
        EmitKey.ESTIMATED_FLOPS_PER_GPU: 999,
    }
    emitter.on_write(WriteEvent(kind=WriteKind.POLL, node="n0", new_values=new_values, load=0.42))

    by_name = _by_name(enabled_emitter.calls)
    assert set(by_name) == set(new_values) | {EmitKey.KV_CACHE_LOAD}
    assert len(enabled_emitter.calls) == 8
    assert all(c["method"] == "gauge" for c in enabled_emitter.calls)  # all gauges
    assert by_name[EmitKey.KV_CACHE_LOAD]["value"] == 0.42
    assert by_name[EmitKey.PROMPT_TOKENS]["value"] == 1000  # cumulative forwarded as-is
    assert all(c["labels"] == {"replica": "n0"} for c in enabled_emitter.calls)


def test_on_write_acquire_emits_counters_tokens_and_avg_turn(
    enabled_emitter: FakeRLInsight,
):
    emitter.on_write(
        WriteEvent(
            kind=WriteKind.ACQUIRE,
            node="n0",
            deltas={EmitKey.DISPATCHED_COUNT: 1, EmitKey.PROMPT_LEN_SUM: 2163},
            new_values={EmitKey.INFLIGHT_TOKENS: 2163},
            turn_sum=3,
            inflight_count=2,
        )
    )

    by_name = _by_name(enabled_emitter.calls)
    assert set(by_name) == {
        EmitKey.DISPATCHED_COUNT,
        EmitKey.PROMPT_LEN_SUM,
        EmitKey.INFLIGHT_TOKENS,
        EmitKey.INFLIGHT_AVG_TURN,
    }
    assert by_name[EmitKey.DISPATCHED_COUNT]["method"] == "counter"
    assert by_name[EmitKey.DISPATCHED_COUNT]["value"] == 1
    assert by_name[EmitKey.PROMPT_LEN_SUM]["value"] == 2163
    assert by_name[EmitKey.INFLIGHT_TOKENS]["method"] == "gauge"
    assert by_name[EmitKey.INFLIGHT_TOKENS]["value"] == 2163
    # emitter-side division: turn_sum(3) / inflight_count(2) = 1.5
    assert by_name[EmitKey.INFLIGHT_AVG_TURN]["method"] == "gauge"
    assert by_name[EmitKey.INFLIGHT_AVG_TURN]["value"] == 1.5


def test_on_write_release_emits_completed_tokens_and_avg_turn_zero_when_idle(
    enabled_emitter: FakeRLInsight,
):
    emitter.on_write(
        WriteEvent(
            kind=WriteKind.RELEASE,
            node="n0",
            deltas={EmitKey.COMPLETED_COUNT: 1},
            new_values={EmitKey.INFLIGHT_TOKENS: 0},
            turn_sum=0,
            inflight_count=0,
        )
    )

    by_name = _by_name(enabled_emitter.calls)
    assert set(by_name) == {
        EmitKey.COMPLETED_COUNT,
        EmitKey.INFLIGHT_TOKENS,
        EmitKey.INFLIGHT_AVG_TURN,
    }
    assert by_name[EmitKey.COMPLETED_COUNT]["method"] == "counter"
    assert by_name[EmitKey.INFLIGHT_AVG_TURN]["value"] == 0.0  # count 0 → defined as 0


def test_on_write_kv_removed_emits_eviction_delta(enabled_emitter: FakeRLInsight):
    emitter.on_write(WriteEvent(kind=WriteKind.KV_REMOVED, node="n0", deltas={EmitKey.KV_EVICTIONS: 7}))

    assert len(enabled_emitter.calls) == 1
    call = enabled_emitter.calls[0]
    assert (call["method"], call["name"], call["value"]) == (
        "counter",
        EmitKey.KV_EVICTIONS,
        7,
    )


def test_on_write_acquire_without_optional_fields_skips_them(
    enabled_emitter: FakeRLInsight,
):
    # Only dispatched delta present; no inflight_tokens, no turn fields.
    emitter.on_write(
        WriteEvent(
            kind=WriteKind.ACQUIRE,
            node="n0",
            deltas={EmitKey.DISPATCHED_COUNT: 1},
        )
    )

    assert [c["name"] for c in enabled_emitter.calls] == [EmitKey.DISPATCHED_COUNT]


# ── contract sanity: every WriteEvent-reachable primitive is in EMIT_SPECS ──


def test_emit_spec_contract_covers_all_referenced_primitives():
    referenced = {
        EmitKey.KV_CACHE_USAGE_PERC,
        EmitKey.NUM_REQUESTS_RUNNING,
        EmitKey.NUM_REQUESTS_WAITING,
        EmitKey.PROMPT_TOKENS,
        EmitKey.PROMPT_TOKENS_CACHED,
        EmitKey.EXTERNAL_PREFIX_CACHE_HITS,
        EmitKey.ESTIMATED_FLOPS_PER_GPU,
        EmitKey.KV_CACHE_LOAD,
        EmitKey.DISPATCHED_COUNT,
        EmitKey.PROMPT_LEN_SUM,
        EmitKey.INFLIGHT_TOKENS,
        EmitKey.INFLIGHT_AVG_TURN,
        EmitKey.COMPLETED_COUNT,
        EmitKey.KV_EVICTIONS,
        EmitKey.LOAD,
        EmitKey.ROUTE_LATENCY_SECONDS,
    }
    assert referenced <= set(EMIT_SPECS)
