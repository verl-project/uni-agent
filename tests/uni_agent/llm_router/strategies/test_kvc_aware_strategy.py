"""Unit tests for the LLM router strategy module (strategies/ package).

Unified combined score (one pass, no fast/slow branching):
    S = α·S_cache + (1-α)·S_load
    S_cache = w_gpu·gpu_hit + w_cpu·cpu_hit + w_ssd·ssd_hit   (weights sum to 1)
    S_load  = 1 - load                                         (bigger = less loaded)
    load    = a·kv + b·min(1, running/max_num_seqs) + c·min(1, waiting/max_num_seqs)
              (a+b+c=1; default 0.4/0.3/0.3; bigger = more loaded)

Overload (used only by the sticky short-circuit): ``load > load_threshold``
(default 0.9). Combined scoring never consults overload.
Default cache weights: {gpu:0.7, cpu:0.2, ssd:0.1}.
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.metric_spec import MetricKey
from uni_agent.llm_router.strategies import (
    KVCacheAwareStrategy,
    RoutingStrategy,
    StickySessionTable,
    StrategyError,
    route,
)
from uni_agent.llm_router.strategies.base import ReplicaInfo
from uni_agent.llm_router.strategies.kvc_aware import STICKY_TOP_SCORE

pytestmark = [pytest.mark.ut, pytest.mark.cpu]
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _strat(**kwargs) -> KVCacheAwareStrategy:
    """Build a KVCacheAwareStrategy with required boilerplate fields filled in.

    ``max_num_seqs=64`` is pinned so the load formula's running term is
    deterministic regardless of the ``MAX_NUM_SEQS`` env var.
    """
    defaults = dict(
        alpha=0.7,
        load_threshold=0.9,
        layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        collector_names=["vllm_zmq"],
        weight=1.0,
        load_fn="normalized",
        load_weights=(0.4, 0.3, 0.3),
        max_num_seqs=64,
    )
    defaults.update(kwargs)
    return KVCacheAwareStrategy(**defaults)


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


PROMPT_IDS = [1, 2, 3]


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeRouteDataProvider:
    """In-memory replica metrics for unit tests.

    Each replica entry is a plain dict with the following optional keys:
      kv_cache_usage_perc  – KV cache usage ratio (default 1.0)
      num_requests_running – requests in flight (default 0)
      num_requests_waiting – requests in the queue (default 0)
      gpu_hit_pct          – GPU prefix cache hit percent 0-100 (default 0)
      tiers                – dict mapping tier name to hit rate (default {})
    """

    def __init__(self, data: dict[str, dict]):
        self._data = data

    def get_metric(self, replica_id: str, key: str) -> float | int:
        entry = self._data.get(replica_id, {})
        if key == MetricKey.KV_CACHE_USAGE_PERC:
            return entry.get("kv_cache_usage_perc", 1.0)
        if key == MetricKey.NUM_REQUESTS_RUNNING:
            return entry.get("num_requests_running", 0)
        if key == MetricKey.NUM_REQUESTS_WAITING:
            return entry.get("num_requests_waiting", 0)
        return entry.get(key, 0.0)

    def get_metrics(self, replica_id: str) -> dict:
        entry = self._data.get(replica_id, {})
        return {
            MetricKey.KV_CACHE_USAGE_PERC: entry.get("kv_cache_usage_perc", 1.0),
            MetricKey.NUM_REQUESTS_RUNNING: entry.get("num_requests_running", 0),
            MetricKey.NUM_REQUESTS_WAITING: entry.get("num_requests_waiting", 0),
        }

    def get_gpu_prefix_hit_rate(self, prompt_ids: list[int]) -> dict[str, int]:
        """Returns {replica_id: hit_percent 0-100} for replicas with hits."""
        result = {}
        for replica_id, entry in self._data.items():
            pct = entry.get("gpu_hit_pct", 0)
            if pct > 0:
                result[replica_id] = pct
        return result

    def get_tier_prefix_hit_rate(self, replica_id: str, prompt_ids: list[int], tier: str) -> float:
        return self._data.get(replica_id, {}).get("tiers", {}).get(tier, 0.0)


class ConstantStrategy:
    """Returns a fixed per-replica score list (for route() composition tests)."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        return list(self._scores)


class BadLengthStrategy:
    """Returns a wrong-length list to exercise the contract check in route()."""

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        return [1.0]


class RaisingStrategy:
    """Raises inside score() to exercise route()'s exception wrapping."""

    def score(self, prompt_ids, provider, replicas, request_id=None, sticky_table=None) -> list[float]:
        raise KeyError("boom")


# --------------------------------------------------------------------------- #
# Unified combined score (one pass: α·S_cache + (1-α)·S_load)
# --------------------------------------------------------------------------- #

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestKVCAwareCombinedScore:
    def test_three_layer_cache_weighted_sum(self):
        """
        Feature: S_cache is a three-layer weighted sum; load term from S_load
        Description: two light-load replicas (running=0); rep_a has gpu+cpu+ssd hits
        Expectation: scores = [0.766, 0.322]; rep_a ranks first
          rep_a: load=0.4·0.2=0.08 → s_load=0.92; s_cache=0.70; score=0.7·0.70+0.3·0.92=0.766
          rep_b: load=0.4·0.4=0.16 → s_load=0.84; s_cache=0.10; score=0.7·0.10+0.3·0.84=0.322
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.2,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.6, "ssd": 0.2},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.4,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.3, "ssd": 0.4},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))
        assert scores == pytest.approx([0.766, 0.322])
        assert route([(strat, 1.0)], PROMPT_IDS, provider, _replicas("rep_a", "rep_b")) == ["rep_a", "rep_b"]

    def test_gpu_dominates_when_tiers_empty(self):
        """
        Feature: with no tier hits, S_cache = w_gpu·gpu_hit; load light
        Description: rep_a gpu_hit_pct=70; rep_b none; both running=0
        Expectation: scores = [0.619, 0.252]
          rep_a: load=0.08→s_load=0.92; s_cache=0.49; score=0.7·0.49+0.3·0.92=0.619
          rep_b: load=0.16→s_load=0.84; s_cache=0;    score=0.3·0.84=0.252
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.2,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 70,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.4,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"))
        assert scores == pytest.approx([0.619, 0.252])

    def test_high_load_penalizes_but_full_formula_applied(self):
        """
        Feature: a saturated replica (load>0.9) gets the FULL formula (no zeroing);
        its s_load≈0 drags the score down despite high cache.
        Description: "loaded" kv=1,r=64,w=1000 (load≈0.98); "light" kv=0.2,r=0 (load=0.08)
        Expectation: light outranks loaded; loaded score still reflects its cache term (not zeroed)
          loaded: waiting clamped (1000/64→1.0) → load=0.4+0.3+0.3=1.0→s_load=0; s_cache=0.63;
                  score=0.7·0.63+0.3·0=0.441  (cache term 0.441 present despite saturation)
          light:  load=0.08→s_load=0.92; s_cache=0.35; score=0.7·0.35+0.3·0.92=0.521
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "loaded": {
                    "kv_cache_usage_perc": 1.0,
                    "num_requests_running": 64,
                    "num_requests_waiting": 1000,
                    "gpu_hit_pct": 90,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "light": {
                    "kv_cache_usage_perc": 0.2,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 50,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("loaded", "light"))
        assert scores == pytest.approx([0.441, 0.521], abs=1e-4)
        assert scores[1] > scores[0]  # high load penalizes below light
        # loaded's score equals the full formula — cache term (0.441) is NOT zeroed
        # load=1.0 (waiting clamped) → s_load=0 → score = 0.7·0.63 + 0.3·0 = 0.441
        assert scores[0] == pytest.approx(0.7 * 0.63 + 0.3 * 0.0, abs=1e-4)

    def test_no_cache_pure_load(self):
        """
        Feature: with no cache hits, score collapses to (1-α)·s_load = (1-α)·(1-load)
        Description: idle (load=0) vs kv-full (load=0.4); both no cache, running=0
        Expectation: scores = [0.30, 0.18]
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "idle": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "full": {
                    "kv_cache_usage_perc": 1.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("idle", "full"))
        assert scores == pytest.approx([0.30, 0.18])


# --------------------------------------------------------------------------- #
# StrategyRegistry
# --------------------------------------------------------------------------- #


class TestKVCAwareLoad:
    def test_load_formula_monotonic_in_kv(self):
        """
        Feature: higher kv_usage → higher load → lower s_load → lower score
        Description: three replicas with kv 0 / 0.5 / 1.0 (running=0); no cache
        Expectation: scores decrease as kv rises
          idle:   load=0    → s_load=1.0  → score=0.30
          mid:    load=0.2  → s_load=0.8  → score=0.24
          loaded: load=0.7  → s_load=0.3  → score=0.09   (kv=1,running=64: load=0.4+0.3=0.7)
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "idle": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "mid": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "loaded": {
                    "kv_cache_usage_perc": 1.0,
                    "num_requests_running": 64,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("idle", "mid", "loaded"))
        assert scores == pytest.approx([0.30, 0.24, 0.09])
        assert scores[0] > scores[1] > scores[2]

    def test_running_increases_load(self):
        """
        Feature: running/max_num_seqs contributes to load; clamped to 1.0
        Description: kv=0.5 fixed; running 0 / 32 / 64; no cache
        Expectation: scores decrease as running rises
          r=0:  load=0.2        → s_load=0.80 → score=0.24
          r=32: load=0.2+0.15=0.35 → s_load=0.65 → score=0.195
          r=64: load=0.2+0.30=0.50 → s_load=0.50 → score=0.15
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "r0": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "r32": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 32,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "r64": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 64,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("r0", "r32", "r64"))
        assert scores == pytest.approx([0.24, 0.195, 0.15])
        assert scores[0] > scores[1] > scores[2]

    def test_waiting_increases_load(self):
        """
        Feature: min(1, waiting/max_num_seqs) contributes to load
        Description: kv=0,running=0; waiting 0 vs 10; no cache
        Expectation: waiting replica scores lower
          w=0:  load=0 → s_load=1.0 → score=0.30
          w=10: load=0.3·(10/64)=0.0469 → s_load=0.9531 → score=0.2859
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "w0": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "w10": {
                    "kv_cache_usage_perc": 0.0,
                    "num_requests_running": 0,
                    "num_requests_waiting": 10,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("w0", "w10"))
        assert scores == pytest.approx([0.30, 0.3 * (1 - 0.3 * (10 / 64))])
        assert scores[0] > scores[1]

    def test_missing_metrics_defaults_to_high_load(self):
        """
        Feature: unknown replica defaults to kv=1.0 → load=0.4 (not 1.0); no cache
        Description: score a replica whose id is absent from the provider
        Expectation: load=0.4·1.0=0.4 → s_load=0.6 → score=0.3·0.6=0.18
        """
        strat = _strat()
        provider = FakeRouteDataProvider({})
        scores = strat.score(PROMPT_IDS, provider, _replicas("ghost"))
        assert scores == pytest.approx([0.18])


# --------------------------------------------------------------------------- #
# _cache_score: three-layer weighted hit (gpu + cpu + ssd)
# --------------------------------------------------------------------------- #
class TestKVCAwareCacheScore:
    def test_three_layer_weighted_sum(self):
        """
        Feature: _cache_score = w_gpu·gpu + w_cpu·cpu + w_ssd·ssd
        Description: gpu_hit_pct=80, cpu=0.6, ssd=0.2 with default weights
        Expectation: 0.7*0.8 + 0.2*0.6 + 0.1*0.2 = 0.70
        """
        strat = _strat()
        provider = FakeRouteDataProvider({"rep": {"gpu_hit_pct": 80, "tiers": {"cpu": 0.6, "ssd": 0.2}}})
        gpu_hit_pct = provider.get_gpu_prefix_hit_rate(PROMPT_IDS)  # {"rep": 80}
        assert strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS, gpu_hit_pct) == pytest.approx(0.70)

    def test_gpu_only_when_tier_none(self):
        """
        Feature: None tier hit rate is treated as 0.0
        Description: provider returns None for tiers (mooncake placeholder); gpu_hit_pct=80
        Expectation: 0.7*0.8 + 0 + 0 = 0.56
        """

        class _NoneProvider(FakeRouteDataProvider):
            def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
                return None

        strat = _strat()
        provider = _NoneProvider({"rep": {"gpu_hit_pct": 80, "tiers": {}}})
        gpu_hit_pct = {"rep": 80}
        assert strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS, gpu_hit_pct) == pytest.approx(0.56)

    def test_no_hit_returns_zero(self):
        """
        Feature: no gpu hit and no tier hits → _cache_score = 0.0
        Description: replica absent from gpu_hit_pct; tiers all 0
        Expectation: 0.0
        """
        strat = _strat()
        provider = FakeRouteDataProvider({"rep": {"tiers": {"cpu": 0.0, "ssd": 0.0}}})
        assert strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS, {}) == pytest.approx(0.0)

    def test_custom_weights_respected(self):
        """
        Feature: _cache_score honors custom layer_weights
        Description: weights {gpu:0.5,cpu:0.3,ssd:0.2}; all hits = 1.0 (gpu_hit_pct=100)
        Expectation: 0.5 + 0.3 + 0.2 = 1.0
        """
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        provider = FakeRouteDataProvider({"rep": {"gpu_hit_pct": 100, "tiers": {"cpu": 1.0, "ssd": 1.0}}})
        gpu_hit_pct = {"rep": 100}
        assert strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS, gpu_hit_pct) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Tier weights in the cache term
# --------------------------------------------------------------------------- #
class TestKVCAwareTierWeights:
    def test_cpu_weight_higher_than_ssd(self):
        """
        Feature: cpu tier weight (0.2) > ssd tier weight (0.1) in the cache term
        Description: two light-load replicas; one has cpu hit, other ssd hit
        Expectation: cpu-hit replica scores higher
          cpu_hit: load=0.2→s_load=0.8; s_cache=0.2·0.6=0.12; score=0.7·0.12+0.3·0.8=0.324
          ssd_hit: load=0.2→s_load=0.8; s_cache=0.1·0.8=0.08; score=0.7·0.08+0.3·0.8=0.296
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "cpu_hit": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.6, "ssd": 0.0},
                },
                "ssd_hit": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.8},
                },
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("cpu_hit", "ssd_hit"))
        assert scores == pytest.approx([0.324, 0.296])
        assert scores[0] > scores[1]

    def test_tier_none_treated_as_zero(self):
        """
        Feature: None return from get_tier_prefix_hit_rate is treated as 0.0
        Description: provider returns None for tier hit rate
        Expectation: score = (1-α)·s_load (S_cache=0), no TypeError
          rep: load=0.2→s_load=0.8; s_cache=0; score=0.3·0.8=0.24
        """

        class _NoneProvider(FakeRouteDataProvider):
            def get_tier_prefix_hit_rate(self, replica_id, prompt_ids, tier):
                return None

        strat = _strat()
        provider = _NoneProvider(
            {
                "rep": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {},
                }
            }
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep"))
        assert scores == pytest.approx([0.24])


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
class TestKVCAwareConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alpha": 1.5},
            {"alpha": -0.1},
            {"load_threshold": 0},
            {"load_threshold": 1.0},
            {"layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": -0.1}},  # negative weight
            {"layer_weights": {"nvme": 1.0}},  # illegal key
            {"layer_weights": {"gpu": 1.0, "cpu": 0.2, "ssd": 0.1}},  # sum 1.3 != 1
            {"layer_weights": {"gpu": 0.7, "cpu": 0.3}},  # missing ssd
            {"load_fn": "does-not-exist"},  # unknown load fn
            {"load_weights": (0.5, 0.3)},  # len != 3
            {"load_weights": (0.5, 0.5, 0.5)},  # sum 1.5 != 1
            {"load_weights": (-0.1, 0.6, 0.5)},  # negative
        ],
    )
    def test_invalid_construction_raises(self, kwargs):
        """
        Feature: invalid constructor arguments raise StrategyError
        Description: construct KVCacheAwareStrategy with each invalid kwarg
        Expectation: raises StrategyError for each case
        """
        with pytest.raises(StrategyError):
            _strat(**kwargs)

    def test_valid_three_key_weights_accepted(self):
        """
        Feature: three-key layer_weights summing to 1.0 construct successfully
        """
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        assert strat.layer_weights == {"gpu": 0.5, "cpu": 0.3, "ssd": 0.2}

    def test_default_load_fn_is_normalized(self):
        """
        Feature: default load_fn is "normalized"; load_weights default (0.4,0.3,0.3)
        """
        strat = _strat()
        assert strat.load_fn == "normalized"
        assert strat.load_weights == (0.4, 0.3, 0.3)

    def test_load_fn_kv_over_pressure_selectable(self):
        """
        Feature: load_fn="kv_over_pressure" (legacy) is selectable
        """
        strat = _strat(load_fn="kv_over_pressure")
        assert strat.load_fn == "kv_over_pressure"


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #
class TestStrategyContract:
    def test_protocol_satisfied(self):
        """
        Feature: KVCacheAwareStrategy satisfies the RoutingStrategy Protocol
        """
        strat = _strat()
        assert isinstance(strat, RoutingStrategy)

    def test_output_length_matches_replicas(self):
        """
        Feature: score() returns a list with same length as replicas
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 90,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 2,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 10,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas)
        assert len(scores) == len(replicas)

    def test_stateless_repeatable(self):
        """
        Feature: calling score() twice on the same inputs produces identical results
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 2,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.5, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat.score(PROMPT_IDS, provider, replicas) == pytest.approx(strat.score(PROMPT_IDS, provider, replicas))


# --------------------------------------------------------------------------- #
# route() composition
# --------------------------------------------------------------------------- #


class TestFromConfig:
    def test_from_config_correct_fields(self):
        """
        Feature: from_config() transfers config fields to the strategy instance
        Description: build a KVCAwareStrategyConfig with non-default values, then from_config()
        Expectation: strategy alpha, load_threshold, layer_weights match the config;
                     load_fn/weights come from code defaults (not config)
        """
        from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(
            alpha=0.6,
            load_threshold=0.85,
            layer_weights={"gpu": 0.6, "cpu": 0.3, "ssd": 0.1},
            weight=0.9,
            collector_names=["vllm_zmq"],
        )
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.alpha == pytest.approx(0.6)
        assert strat.load_threshold == pytest.approx(0.85)
        assert strat.layer_weights == {"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}
        assert strat.load_fn == "normalized"  # code default
        assert strat.load_weights == (0.4, 0.3, 0.3)  # code default

    def test_from_config_scores_match_direct(self, monkeypatch):
        """
        Feature: a strategy built via from_config() produces the same scores as one built directly
        Description: pin MAX_NUM_SEQS=64 so both resolve the same max_num_seqs
        Expectation: both strategies return approx-equal score lists
        """
        from uni_agent.llm_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setenv("MAX_NUM_SEQS", "64")  # from_config resolves max_num_seqs from env
        cfg = KVCAwareStrategyConfig(
            alpha=0.7,
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
            weight=1.0,
            collector_names=["vllm_zmq"],
        )
        strat_from_cfg = KVCacheAwareStrategy.from_config(cfg)
        strat_direct = _strat()  # max_num_seqs=64 pinned
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "kv_cache_usage_perc": 0.92,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat_from_cfg.score(PROMPT_IDS, provider, replicas) == pytest.approx(
            strat_direct.score(PROMPT_IDS, provider, replicas)
        )


# --------------------------------------------------------------------------- #
# Sticky-session short-circuit (is_overloaded uses load > load_threshold)
# --------------------------------------------------------------------------- #
class TestStickyShortCircuit:
    """Sticky replica wins when bound + present + not overloaded; else fall through.

    Overload now means ``load > load_threshold`` (default 0.9) — i.e. the bound
    replica is genuinely saturated (kv≈1, running≈max_num_seqs, big backlog).
    """

    def _provider(self, **per_replica):
        """Build a FakeRouteDataProvider from {rep_id: metrics_dict}."""
        return FakeRouteDataProvider(per_replica)

    # ── is_overloaded ──────────────────────────────────────────────────────
    def test_is_overloaded_true_when_saturated(self):
        """Feature: is_overloaded True when load > load_threshold (0.9).
        Description: kv=1.0, running=64 (mns), waiting=1000 → load≈0.98 > 0.9
        Expectation: overloaded
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 1.0, "num_requests_running": 64, "num_requests_waiting": 1000}
        )
        assert strat.is_overloaded(provider, ReplicaInfo("rep_a")) is True

    def test_is_overloaded_false_when_light(self):
        """Feature: is_overloaded False when load <= load_threshold.
        Description: kv=0.3, running=0, waiting=0 → load=0.12 < 0.9
        Expectation: not overloaded
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(rep_a={"kv_cache_usage_perc": 0.3, "num_requests_running": 0})
        assert strat.is_overloaded(provider, ReplicaInfo("rep_a")) is False

    # ── score() sticky short-circuit ───────────────────────────────────────
    def test_sticky_hit_not_overloaded_short_circuits(self):
        """Feature: bound + present + not overloaded → sticky replica gets top score.
        Description: sticky binds r1→rep_b; rep_b light (load=0.12); rep_a has better
        combined score but must NOT win.
        Expectation: scores = [0.0, STICKY_TOP_SCORE]; route() picks rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        sticky = StickySessionTable()
        sticky.put("r1", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas, "r1", sticky)
        assert scores == [0.0, STICKY_TOP_SCORE]
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1", sticky)
        assert ranking[0] == "rep_b"

    def test_sticky_hit_overloaded_falls_back_to_combined(self):
        """Feature: bound but saturated (load>0.9) → no short-circuit, combined scoring.
        Description: sticky binds r1→rep_b; rep_b saturated (kv=1,r=64,w=1000);
        rep_a light with gpu hit.
        Expectation: rep_a wins (combined), not the saturated sticky rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={
                "kv_cache_usage_perc": 1.0,
                "num_requests_running": 64,
                "num_requests_waiting": 1000,
                "gpu_hit_pct": 0,
            },
        )
        replicas = _replicas("rep_a", "rep_b")
        sticky = StickySessionTable()
        sticky.put("r1", "rep_b")
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1", sticky)
        assert ranking[0] == "rep_a"

    def test_sticky_no_binding_cold_start_combined(self):
        """Feature: no sticky binding → combined scoring (cold start).
        Expectation: best combined replica wins (rep_a)
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        sticky = StickySessionTable()
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1", sticky)
        assert ranking[0] == "rep_a"

    def test_sticky_bound_replica_removed_falls_back(self):
        """Feature: bound replica no longer in pool → fall back to combined.
        Expectation: rep_a wins (combined), no KeyError/crash
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        sticky = StickySessionTable()
        sticky.put("r1", "rep_gone")  # bound replica not in pool
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, "r1", sticky)
        assert ranking[0] == "rep_a"

    def test_sticky_none_request_id_combined(self):
        """Feature: request_id=None → combined scoring (no sticky lookup).
        Expectation: combined scoring, rep_a wins
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        ranking = route([(strat, 1.0)], PROMPT_IDS, provider, replicas, None, None)
        assert ranking[0] == "rep_a"
