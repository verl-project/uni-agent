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

"""Tests for agent_aware_router strategy seams (runtime strategy classes + registry).

These are the minimal collaborator seams the Balancer constructs against
(see detailed_balancer.md §2.3). Construction, registry dispatch, and routing
are tested here.
"""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig
from uni_agent.agent_aware_router.store.group_cache import PrefixEstimate
from uni_agent.agent_aware_router.strategies import ReplicaInfo, StrategyRegistry, route
from uni_agent.agent_aware_router.strategies.kvc_aware import KVCacheAwareStrategy

# ============================================================
# ReplicaInfo
# ============================================================


class TestReplicaInfo:
    """R03: ReplicaInfo value type."""

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_r03_carries_only_replica_id(self):
        """
        Feature: ReplicaInfo carries a replica_id and no actor handle
        Description: construct ReplicaInfo(replica_id="s0")
        Expectation: ri.replica_id == "s0" and has no handle attribute
        """
        ri = ReplicaInfo(replica_id="s0")
        assert ri.replica_id == "s0"
        assert not hasattr(ri, "handle")


# ============================================================
# route() — real ranking
# ============================================================


# ---- Strategy registry + route (comprehensive, from test_strategy.py) ----


class TestStrategyRegistry:
    @pytest.mark.cpu
    @pytest.mark.level0
    def test_builtin_registered(self):
        """
        Feature: built-in KVCacheAwareStrategy is pre-registered for KVCAwareStrategyConfig
        Description: look up KVCAwareStrategyConfig in the registry
        Expectation: returns KVCacheAwareStrategy class
        """

        assert StrategyRegistry.get(KVCAwareStrategyConfig) is KVCacheAwareStrategy

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_register_and_get(self):
        """
        Feature: custom strategy can be registered and retrieved by config type
        Description: register a dummy config class, then call get() with it
        Expectation: get() returns the registered strategy class
        """

        class _DummyConfig:
            pass

        class _DummyStrategy:
            def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
                return [0.0] * len(replicas)

        StrategyRegistry.register(_DummyConfig, _DummyStrategy)
        try:
            assert StrategyRegistry.get(_DummyConfig) is _DummyStrategy
        finally:
            StrategyRegistry._registry.pop(_DummyConfig, None)

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_get_unknown_raises(self):
        """
        Feature: looking up an unregistered config type raises KeyError
        Description: call get() with a class that was never registered
        Expectation: raises KeyError
        """

        class _UnknownConfig:
            pass

        with pytest.raises(KeyError):
            StrategyRegistry.get(_UnknownConfig)


# --------------------------------------------------------------------------- #
# Test doubles for route() composition tests
# --------------------------------------------------------------------------- #


class FakeRouteDataProvider:
    def __init__(self, reserved_blocks=None):
        self._reserved_blocks = reserved_blocks or {}

    def get_metric(self, replica_id, key):
        return 0.0

    def capacity_blocks(self, replica_id):
        return 10

    def inflight_reserved_blocks(self, replica_id):
        return self._reserved_blocks[replica_id]

    def inflight_block_count(self, replica_id):
        return self._reserved_blocks[replica_id]

    def request_block_need(self, prompt_tokens, saved_tokens, *, replica_id):
        return 1

    def hash_schema(self):
        return 0, None, 0

    def estimate_prefixes(self, replica_ids, hashes, query_tokens):
        return {node: PrefixEstimate("unavailable", 0, query_tokens, 0, None) for node in replica_ids}


class ConstantStrategy:
    def __init__(self, scores):
        self._scores = scores

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
        return list(self._scores)


class BadLengthStrategy:
    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
        return [1.0]


class RaisingStrategy:
    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None):
        raise KeyError("boom")


def _replicas(*ids):
    return [ReplicaInfo(replica_id=rid) for rid in ids]


PROMPT_IDS = [1, 2, 3]


def _strat(**kwargs):
    defaults = dict(
        load_threshold=0.9,
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


# --------------------------------------------------------------------------- #
# route() — real ranking
# --------------------------------------------------------------------------- #


class TestRoute:
    @pytest.mark.cpu
    @pytest.mark.level0
    def test_single_strategy_descending(self):
        """
        Feature: route() returns replica ids sorted by score descending
        """
        provider = FakeRouteDataProvider({})
        ranking = route(
            ConstantStrategy([0.2, 0.5, 0.1]),
            PROMPT_IDS,
            provider,
            _replicas("rep_a", "rep_b", "rep_c"),
        )
        assert ranking == ["rep_b", "rep_a", "rep_c"]

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_overloaded_present_in_ranking(self, monkeypatch):
        """Capacity scoring keeps an overloaded replica in the ranking."""
        provider = FakeRouteDataProvider(reserved_blocks={"rep_a": 2, "rep_b": 6, "overloaded": 10})
        monkeypatch.setattr(
            "uni_agent.agent_aware_router.strategies.routing.random.shuffle",
            lambda _: pytest.fail("unexpected fallback to random routing"),
        )

        ranking = route(_strat(), PROMPT_IDS, provider, _replicas("overloaded", "rep_b", "rep_a"))
        assert ranking[0] == "rep_a"
        assert set(ranking) == {"rep_a", "rep_b", "overloaded"}

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_empty_pool_raises(self):
        """
        Feature: route() raises RuntimeError when the replica list is empty
        """
        with pytest.raises(RuntimeError):
            route(ConstantStrategy([]), PROMPT_IDS, FakeRouteDataProvider({}), [])

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_length_mismatch_falls_back_to_random(self):
        """
        Feature: route() falls back to random order when score() returns wrong-length list
        """
        ranking = route(
            BadLengthStrategy(),
            PROMPT_IDS,
            FakeRouteDataProvider({}),
            _replicas("rep_a", "rep_b"),
        )
        assert set(ranking) == {"rep_a", "rep_b"}

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_strategy_exception_falls_back_to_random(self):
        """
        Feature: exceptions from score() cause route() to fall back to random order
        """
        ranking = route(RaisingStrategy(), PROMPT_IDS, FakeRouteDataProvider({}), _replicas("rep_a", "rep_b"))
        assert set(ranking) == {"rep_a", "rep_b"}

    @pytest.mark.cpu
    @pytest.mark.level0
    def test_nan_score_ranked_last(self):
        """
        Feature: non-finite (NaN) scores are ranked last
        """
        provider = FakeRouteDataProvider({})
        ranking = route(
            ConstantStrategy([float("nan"), 0.1, 0.5]),
            PROMPT_IDS,
            provider,
            _replicas("nan_rep", "low_rep", "high_rep"),
        )
        assert ranking[0] == "high_rep"
        assert ranking[-1] == "nan_rep"


# --------------------------------------------------------------------------- #
# from_config() classmethod
# --------------------------------------------------------------------------- #
