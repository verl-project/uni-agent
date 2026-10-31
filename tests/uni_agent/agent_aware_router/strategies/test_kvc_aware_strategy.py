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

"""Unit tests for the LLM router strategy module (strategies/ package).

Overload (used only by the sticky short-circuit): ``kv_cache_usage_perc >
load_threshold`` (default 0.5). Fallback scoring never consults overload.
Default cache weights: {gpu:0.7, cpu:0.2, ssd:0.1}.
"""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.strategies import route
from uni_agent.agent_aware_router.strategies.base import ReplicaInfo
from uni_agent.agent_aware_router.strategies.kvc_aware import (
    STICKY_TOP_SCORE,
    KVCacheAwareStrategy,
    StrategyError,
)
from uni_agent.agent_aware_router.strategies.routing import RoutingStrategy
from uni_agent.agent_aware_router.types import Layer, MetricKey, OverloadMode, SlowCut

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _strat(**kwargs) -> KVCacheAwareStrategy:
    """Build a KVCacheAwareStrategy with required boilerplate fields filled in.

    Calls ``set_capacity(64, 1024)`` — mimics what the Balancer does after
    construction.
    """
    defaults = dict(
        load_threshold=0.9,
        layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
    )
    defaults.update(kwargs)
    strat = KVCacheAwareStrategy(**defaults)
    strat.set_capacity(64, 1024)
    return strat


def _replicas(*ids: str) -> list[ReplicaInfo]:
    return [ReplicaInfo(replica_id=rid) for rid in ids]


PROMPT_IDS = [1, 2, 3]

# Fixed soft-pick tie-break seed: the band cases assert on the *set* of winners
# over many draws, which is only reproducible when the RNG is pinned.
SOFT_PICK_SEED = 1234


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeRouteDataProvider:
    """In-memory replica metrics for unit tests.

    Each replica entry is a plain dict with the following optional keys:
      kv_cache_usage_perc  – KV cache usage ratio (default 1.0)
      num_requests_running – requests in flight (default 0)
      num_requests_waiting – requests in the queue (default 0)
      inflight_count       – in-flight acquire/release counter (default 0)
      inflight_tokens      – raw in-flight prompt tokens, observation-only (default 0)
      inflight_blocks      – held KV blocks read via inflight_block_count(),
                             dedup + ref counted (default 0)
      gpu_hit_pct          – GPU prefix cache hit percent 0-100 (default 0)
      tiers                – dict mapping tier name to hit rate (default {})
    """

    def __init__(self, data: dict[str, dict], sticky: dict[str, str] | None = None):
        self._data = data
        self._sticky = sticky or {}
        self._per_request: dict[str, dict] = {}

    def get_sticky_binding(self, request_id: str) -> str | None:
        return self._sticky.get(request_id)

    def put_sticky_binding(self, request_id: str, replica_id: str) -> None:
        self._sticky[request_id] = replica_id

    def get_metric(self, replica_id: str, key: str) -> float | int:
        entry = self._data.get(replica_id, {})
        if key == MetricKey.KV_CACHE_USAGE_PERC:
            return entry.get("kv_cache_usage_perc", 1.0)
        if key == MetricKey.NUM_REQUESTS_RUNNING:
            return entry.get("num_requests_running", 0)
        if key == MetricKey.NUM_REQUESTS_WAITING:
            return entry.get("num_requests_waiting", 0)
        if key == MetricKey.INFLIGHT_COUNT:
            return entry.get("inflight_count", 0)
        return entry.get(key, 0.0)

    def get_metrics(self, replica_id: str) -> dict:
        entry = self._data.get(replica_id, {})
        return {
            MetricKey.KV_CACHE_USAGE_PERC: entry.get("kv_cache_usage_perc", 1.0),
            MetricKey.NUM_REQUESTS_RUNNING: entry.get("num_requests_running", 0),
            MetricKey.NUM_REQUESTS_WAITING: entry.get("num_requests_waiting", 0),
            MetricKey.INFLIGHT_COUNT: entry.get("inflight_count", 0),
        }

    def get_layer_prefix_hit_rate(self, replica_id: str, hash_strs: list[str], layer: str = Layer.GPU) -> float:
        entry = self._data.get(replica_id, {})
        if layer == Layer.GPU:
            return entry.get("gpu_hit_pct", 0) / 100.0
        return entry.get("tiers", {}).get(layer, 0.0)

    def kv_cache_load(self, replica_id: str) -> float | None:
        # Unit tests key the load signal on kv_cache_usage_perc (no kv-events /
        # retained blocks simulated); mirror it so the load formula sees it.
        return self._data.get(replica_id, {}).get("kv_cache_usage_perc", 1.0)

    def inflight_block_count(self, replica_id: str) -> int:
        return int(self._data.get(replica_id, {}).get("inflight_blocks", 0))

    def get_metric_node_ids(self) -> list[str]:
        return list(self._data.keys())

    def get_block_size(self) -> int | None:
        # Block size is learned from KV events; tests default to vLLM's 16.
        return 16

    def get_per_request(self, request_id: str, key: str, default=None):
        return self._per_request.get(request_id, {}).get(key, default)

    def set_per_request(self, request_id: str, key: str, value) -> None:
        self._per_request.setdefault(request_id, {})[key] = value


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
# _cache_score: three-layer weighted hit (gpu + cpu + ssd)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestKVCAwareCacheScore:
    def test_custom_weights_respected(self):
        """
        Feature: _cache_score honors custom layer_weights
        Description: weights {gpu:0.5,cpu:0.3,ssd:0.2}; all hits = 1.0 (gpu_hit_pct=100)
        Expectation: 0.5 + 0.3 + 0.2 = 1.0
        """
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        provider = FakeRouteDataProvider({"rep": {"gpu_hit_pct": 100, "tiers": {"cpu": 1.0, "ssd": 1.0}}})
        s_cache, _ = strat._cache_score(provider, ReplicaInfo("rep"), PROMPT_IDS)
        assert s_cache == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Tier weights in the cache term
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level1
class TestKVCAwareTierWeights:
    def test_cpu_weight_higher_than_ssd(self):
        """
        Feature: cpu tier weight (0.2) > ssd tier weight (0.1) in the cache score
        Description: two replicas with gpu_hit=0; one has cpu hit, other ssd hit
        Expectation: cpu-hit replica scores higher
          cpu_hit: s_cache = 0.2·0.6 = 0.12
          ssd_hit: s_cache = 0.1·0.8 = 0.08
        """
        strat = _strat()
        provider = FakeRouteDataProvider(
            {
                "cpu_hit": {"gpu_hit_pct": 0, "tiers": {"cpu": 0.6, "ssd": 0.0}},
                "ssd_hit": {"gpu_hit_pct": 0, "tiers": {"cpu": 0.0, "ssd": 0.8}},
            }
        )
        s_cache_cpu, _ = strat._cache_score(provider, ReplicaInfo("cpu_hit"), PROMPT_IDS)
        s_cache_ssd, _ = strat._cache_score(provider, ReplicaInfo("ssd_hit"), PROMPT_IDS)
        assert s_cache_cpu == pytest.approx(0.12)
        assert s_cache_ssd == pytest.approx(0.08)
        assert s_cache_cpu > s_cache_ssd


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestKVCAwareConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"load_threshold": 0},
            {"load_threshold": 1.0},
            {"layer_weights": {"gpu": 0.7, "cpu": 0.2, "ssd": -0.1}},
            {"layer_weights": {"nvme": 1.0}},
            {"layer_weights": {"gpu": 1.0, "cpu": 0.2, "ssd": 0.1}},
            {"layer_weights": {"gpu": 0.7, "cpu": 0.3}},
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
        strat = _strat(layer_weights={"gpu": 0.5, "cpu": 0.3, "ssd": 0.2})
        assert strat.layer_weights == {"gpu": 0.5, "cpu": 0.3, "ssd": 0.2}


# --------------------------------------------------------------------------- #
# set_capacity
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestSetCapacity:
    @staticmethod
    def _new_strat() -> KVCacheAwareStrategy:
        """A fresh strategy with set_capacity not yet called."""
        return KVCacheAwareStrategy(
            load_threshold=0.9,
            layer_weights={"gpu": 0.7, "cpu": 0.2, "ssd": 0.1},
        )

    def test_set_capacity_updates_max_num_seqs(self):
        """
        Feature: set_capacity records max_num_seqs
        Description: call set_capacity(16, 1024) on a fresh strategy
        Expectation: _max_num_seqs == 16
        """
        strat = self._new_strat()
        strat.set_capacity(16, 1024)
        assert strat._max_num_seqs == 16

    @pytest.mark.parametrize("max_num_seqs,block_size", [(0, 0), (-1, -1)])
    def test_set_capacity_rejects_non_positive(self, max_num_seqs, block_size):
        """
        Feature: set_capacity rejects zero or negative capacity
        Description: call set_capacity with (0, 0) and (-1, -1)
        Expectation: raises StrategyError
        """
        strat = self._new_strat()
        with pytest.raises(StrategyError):
            strat.set_capacity(max_num_seqs, block_size)


# --------------------------------------------------------------------------- #
# Interface contract
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestStrategyContract:
    def test_score_length_and_stateless_repeatable(self):
        """
        Feature: score() returns a replica-length list and is stateless across calls
        Description: call score() twice on the same two-replica inputs
        Expectation: len(scores) == len(replicas); the two calls produce identical results
        """
        strat = _strat()
        assert isinstance(strat, RoutingStrategy)

        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    # cap=1600 → avail 1600 (rep_a) vs 32 (rep_b): strict argmax, not a tie.
                    "inflight_tokens": 0,
                    "inflight_blocks": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.5,
                    "num_requests_running": 2,
                    "num_requests_waiting": 0,
                    "inflight_tokens": 100,
                    "inflight_blocks": 97,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.5, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas)
        assert len(scores) == len(replicas)
        assert strat.score(PROMPT_IDS, provider, replicas) == pytest.approx(scores)


# --------------------------------------------------------------------------- #
# route() composition
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestFromConfig:
    def test_from_config_maps_fields_and_matches_direct_construction(self):
        """
        Feature: from_config maps fields and behaves like direct construction
        Description: build from a non-default cfg; compare fields and score() against a directly-built strategy
        Expectation: fields match the config; _max_num_seqs is None until set_capacity; score matches direct
        """
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        # The config persists the user-tunable knob (load_threshold); the
        # remaining knobs fall back to DEFAULT_STRATEGY_KNOBS — mirror that
        # here (with one override).
        cfg.layer_weights = {"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}
        cfg.seed = SOFT_PICK_SEED  # same tie-break stream as the direct build below
        strat_from_cfg = KVCacheAwareStrategy.from_config(cfg)

        # ── field mapping ──
        assert strat_from_cfg.load_threshold == pytest.approx(0.85)
        assert strat_from_cfg.layer_weights == {"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}
        assert strat_from_cfg._max_num_seqs is None  # not set until set_capacity()

        # ── behavioral equivalence: from_config vs direct construction ──
        strat_from_cfg.set_capacity(64, 1024)
        strat_direct = _strat(
            load_threshold=0.85,
            layer_weights={"gpu": 0.6, "cpu": 0.3, "ssd": 0.1},
            seed=SOFT_PICK_SEED,
        )
        provider = FakeRouteDataProvider(
            {
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.3,
                    "num_requests_running": 1,
                    "num_requests_waiting": 0,
                    "inflight_tokens": 0,
                    "gpu_hit_pct": 80,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
                "rep_b": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.92,
                    "num_requests_running": 0,
                    "num_requests_waiting": 0,
                    # cap=1600 → avail 48 < gate 160 → rep_a wins deterministically.
                    "inflight_tokens": 100,
                    "inflight_blocks": 97,
                    "gpu_hit_pct": 0,
                    "tiers": {"cpu": 0.0, "ssd": 0.0},
                },
            }
        )
        replicas = _replicas("rep_a", "rep_b")
        assert strat_from_cfg.score(PROMPT_IDS, provider, replicas) == pytest.approx(
            strat_direct.score(PROMPT_IDS, provider, replicas)
        )

    def test_seed_non_int_config_rejected_like_direct_construction(self):
        """from_config and direct construction agree on non-int seeds: StrategyError,
        never a silent coercion (int(True)==1, int(12.5)==12, or '1234' passing as int)."""
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        for bad_seed in (True, 12.5, "1234"):
            cfg = KVCAwareStrategyConfig(load_threshold=0.85)
            cfg.seed = bad_seed
            with pytest.raises(StrategyError, match="seed"):
                KVCacheAwareStrategy.from_config(cfg)
            with pytest.raises(StrategyError, match="seed"):
                _strat(seed=bad_seed)  # _strat fills the required boilerplate knobs


# --------------------------------------------------------------------------- #
# Debug env overrides (UNI_AGENT_ROUTER_*) read in from_config
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestFromConfigDebugEnv:
    """``from_config`` applies ``UNI_AGENT_ROUTER_*`` overrides when debug is on.

    The strategy reads the process env directly (no rollout.custom plumbing):
    each tunable knob is read via ``get_debug_var``, coerced to its default
    type, and merged into the local constructor kwargs — ``cfg`` itself is
    never mutated. Off by default.
    """

    @staticmethod
    def _env_with(debug: str | None, **knobs: str) -> dict[str, str]:
        env: dict[str, str] = {}
        if debug is not None:
            env["UNI_AGENT_ROUTER_DEBUG"] = debug
        for k, v in knobs.items():
            env[f"UNI_AGENT_ROUTER_{k.upper()}"] = v
        return env

    def test_debug_off_ignores_env(self, monkeypatch):
        """Gate off → overrides never reach the strategy (default behavior)."""
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with(None, SLOW_CUT="least-inflight", OVERLOAD_MODE="None"),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.slow_cut == SlowCut.CAPACITY_TOKEN_AWARE  # default, not least-inflight

    def test_pure_sticky_baseline(self, monkeypatch):
        """slow_cut=least-inflight + overload_mode=None → sticky never falls back."""
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", SLOW_CUT="least-inflight", OVERLOAD_MODE="None"),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.slow_cut == SlowCut.LEAST_INFLIGHT
        assert strat.overload_mode == OverloadMode.NONE
        assert strat.do_shortcut is True

    def test_coercion_types(self, monkeypatch):
        """bool / enum / float / dict-json all coerce to their knob types."""
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with(
                "1",
                TIE_TOLERANCE="0.2",
                DO_SHORTCUT="false",
                LAYER_WEIGHTS='{"gpu": 0.6, "cpu": 0.3, "ssd": 0.1}',
            ),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        strat = KVCacheAwareStrategy.from_config(cfg)
        assert strat.tie_tolerance == pytest.approx(0.2)
        assert strat.do_shortcut is False
        assert strat.layer_weights == {Layer.GPU: 0.6, Layer.CPU: 0.3, Layer.SSD: 0.1}

    def test_tie_tolerance_env_override_and_error_layering(self, monkeypatch):
        """TIE_TOLERANCE: coercion first (ConfigError), then range (StrategyError)."""
        from uni_agent.agent_aware_router.config.base import ConfigError
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", TIE_TOLERANCE="0.2"),
        )
        assert KVCacheAwareStrategy.from_config(cfg).tie_tolerance == pytest.approx(0.2)

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", TIE_TOLERANCE="high"),
        )
        with pytest.raises(ConfigError, match="tie_tolerance"):
            KVCacheAwareStrategy.from_config(cfg)

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", TIE_TOLERANCE="1.5"),
        )
        with pytest.raises(StrategyError, match="tie_tolerance"):
            KVCacheAwareStrategy.from_config(cfg)

    def test_seed_env_override_and_error_layering(self, monkeypatch):
        """SEED: coercion first (ConfigError for non-int strings), then __init__ type validation."""
        from uni_agent.agent_aware_router.config.base import ConfigError
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", SEED="1234"),
        )
        assert KVCacheAwareStrategy.from_config(cfg)._seed == 1234

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", SEED="high"),
        )
        with pytest.raises(ConfigError, match="seed"):
            KVCacheAwareStrategy.from_config(cfg)

    @pytest.mark.parametrize(
        ("knob", "value", "match"),
        [
            ("SLOW_CUT", "nope", "slow_cut"),
            ("OVERLOAD_MODE", "sometimes", "overload_mode"),
            ("DO_SHORTCUT", "maybe", "do_shortcut"),
            ("TIE_TOLERANCE", "high", "tie_tolerance"),
            ("LAYER_WEIGHTS", "not json", "layer_weights"),
            ("LAYER_WEIGHTS", "[1, 2]", "layer_weights"),
            ("LAYER_WEIGHTS", '{"npu": 0.5}', "layer_weights"),
        ],
    )
    def test_bad_values_raise_configerror(self, monkeypatch, knob, value, match):
        """Malformed env values fail fast (ConfigError), not silent wrong strategy."""
        from uni_agent.agent_aware_router.config.base import ConfigError
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        monkeypatch.setattr(
            "uni_agent.agent_aware_router.debug.os.environ",
            self._env_with("1", **{knob: value}),
        )
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        with pytest.raises(ConfigError, match=match):
            KVCacheAwareStrategy.from_config(cfg)

    def test_config_object_not_polluted(self, monkeypatch):
        """Debug overrides land in constructor kwargs, never on the config object.

        Reusing the same cfg after disabling debug must produce a default
        strategy — no residual override leaks across constructions.
        """
        from uni_agent.agent_aware_router.config.strategy import KVCAwareStrategyConfig

        env_on = self._env_with("1", TIE_TOLERANCE="0.2")
        monkeypatch.setattr("uni_agent.agent_aware_router.debug.os.environ", env_on)
        cfg = KVCAwareStrategyConfig(load_threshold=0.85)
        assert KVCacheAwareStrategy.from_config(cfg).tie_tolerance == pytest.approx(0.2)
        assert not hasattr(cfg, "tie_tolerance")  # cfg untouched

        monkeypatch.setattr("uni_agent.agent_aware_router.debug.os.environ", {})
        assert KVCacheAwareStrategy.from_config(cfg).tie_tolerance == pytest.approx(0.05)  # default, not 0.2


# --------------------------------------------------------------------------- #
# Sticky-session short-circuit (is_overloaded uses kv_perc > load_threshold)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestStickyShortCircuit:
    """Sticky replica wins when bound + present + not overloaded; else fall through.

    With the default ``overload_mode=KV_CACHE_USAGE_PERC``, overload means
    ``kv_cache_usage_perc > load_threshold`` (default 0.5) — the bound replica
    is genuinely saturated. The "saturated" case below therefore feeds
    kv_cache_usage_perc=1.0.
    """

    def _provider(self, sticky=None, **per_replica):
        """Build a FakeRouteDataProvider from {rep_id: metrics_dict} + optional sticky."""
        return FakeRouteDataProvider(per_replica, sticky=sticky)

    # ── score() sticky short-circuit ───────────────────────────────────────
    def test_sticky_hit_not_overloaded_short_circuits(self):
        """Feature: bound + present + not overloaded → sticky replica gets top score.
        Description: sticky binds r1→rep_b; rep_b light (kv=0.3); rep_a has better
        fallback score but must NOT win.
        Expectation: scores = [0.0, STICKY_TOP_SCORE]; route() picks rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            sticky={"r1": "rep_b"},
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={"kv_cache_usage_perc": 0.3, "num_requests_running": 0, "gpu_hit_pct": 0},
        )
        replicas = _replicas("rep_a", "rep_b")
        scores = strat.score(PROMPT_IDS, provider, replicas, "r1")
        assert scores == [0.0, STICKY_TOP_SCORE]
        ranking = route(strat, PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_b"

    def test_sticky_hit_overloaded_falls_back_to_combined(self):
        """Feature: bound but saturated (kv=1.0 > load_threshold=0.9) → no short-circuit,
        fallback scoring.
        Description: sticky binds r1→rep_b; rep_b saturated (kv_cache_usage_perc=1.0);
        rep_a light with gpu hit.
        Expectation: rep_a wins (fallback scoring), not the saturated sticky rep_b
        """
        strat = _strat(load_threshold=0.9)
        provider = self._provider(
            sticky={"r1": "rep_b"},
            rep_a={"kv_cache_usage_perc": 0.2, "num_requests_running": 0, "gpu_hit_pct": 80},
            rep_b={
                "kv_cache_usage_perc": 1.0,
                "num_requests_running": 64,
                "num_requests_waiting": 1000,
                "inflight_count": 64,
                "gpu_hit_pct": 0,
            },
        )
        replicas = _replicas("rep_a", "rep_b")
        ranking = route(strat, PROMPT_IDS, provider, replicas, "r1")
        assert ranking[0] == "rep_a"

    # ── REMAINING_CAP sibling gate ─────────────────────────────────────────
    # cap = num_gpu_blocks(100) × block_size(16) = 1600; thresh = 1600 × (1-0.5) = 800
    # (load_threshold pinned to 0.5 for this fixture). Under water: 60 held blocks →
    # avail 640. Headroom: 10 held → avail 1440.

    def test_sibling_gate_ignores_removed_replicas(self):
        """Feature: REMAINING_CAP sibling gate weighs the current pool only
        Description: bound rep under water; a removed replica's metric row (never
        purged) still holds 0 blocks — the only node with gate-passing headroom;
        every pool replica is under water too
        Expectation: is_overloaded returns False (keep sticky) — the ghost must not
        count as an escape hatch
        """
        strat = _strat(overload_mode=OverloadMode.REMAINING_CAP, load_threshold=0.5)
        provider = self._provider(
            rep_a={"num_gpu_blocks": 100, "inflight_blocks": 60},
            rep_b={"num_gpu_blocks": 100, "inflight_blocks": 60},
            ghost={"num_gpu_blocks": 100, "inflight_blocks": 0},
        )
        assert strat.is_overloaded(provider, ReplicaInfo("rep_a"), _replicas("rep_a", "rep_b")) is False

    def test_sibling_gate_breaks_sticky_on_live_headroom(self):
        """Feature: REMAINING_CAP sibling gate follows live headroom
        Description: bound rep under water; a pool sibling passes the gate
        Expectation: is_overloaded returns True → sticky falls through to combined scoring
        """
        strat = _strat(overload_mode=OverloadMode.REMAINING_CAP, load_threshold=0.5)
        provider = self._provider(
            rep_a={"num_gpu_blocks": 100, "inflight_blocks": 60},
            rep_b={"num_gpu_blocks": 100, "inflight_blocks": 10},
        )
        assert strat.is_overloaded(provider, ReplicaInfo("rep_a"), _replicas("rep_a", "rep_b")) is True

    def test_sticky_holds_when_only_ghost_has_headroom(self):
        """Feature: sticky short-circuit wiring passes the pool to the overload gate
        Description: bound rep under water; only a removed replica's lingering row
        passes the gate
        Expectation: sticky still short-circuits — scores = [STICKY_TOP_SCORE, 0.0]
        """
        strat = _strat(overload_mode=OverloadMode.REMAINING_CAP, load_threshold=0.5)
        provider = self._provider(
            sticky={"r1": "rep_a"},
            rep_a={"num_gpu_blocks": 100, "inflight_blocks": 60},
            rep_b={"num_gpu_blocks": 100, "inflight_blocks": 60},
            ghost={"num_gpu_blocks": 100, "inflight_blocks": 0},
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert scores == [STICKY_TOP_SCORE, 0.0]


# --------------------------------------------------------------------------- #
# Fallback modes: slow_cut selects the fallback scoring after a sticky miss
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestFallbackModes:
    """``slow_cut`` selects the fallback scoring used after the sticky
    short-circuit misses (``least-inflight`` mirrors verl
    GlobalRequestLoadBalancer)."""

    def test_overloaded_sticky_defers_to_the_capacity_book(self):
        """Bound replica over the sticky gate → the capacity book decides, not the binding.

        rep_a holds 97 blocks (avail = 1600-1552 = 48 < thresh=800 → filtered), rep_b wins.
        """
        strat = _strat(load_threshold=0.9)
        provider = FakeRouteDataProvider(
            {
                # cap must be > 0 (num_gpu_blocks) so the post-fallback ranking has a
                # strict winner: with cap=0 both replicas tie and soft-pick randomizes.
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.95,
                    "num_requests_running": 64,
                    "num_requests_waiting": 1000,
                    "inflight_tokens": 1550,
                    "inflight_blocks": 97,
                },
                "rep_b": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 1.0},
            },
            sticky={"r1": "rep_a"},
        )
        ranking = route(strat, PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_b"  # saturated bound replica is not re-picked

    def test_miss_routes_to_least_inflight(self):
        """slow_cut=least-inflight: pick the replica with the fewest in-flight requests."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"inflight_count": 5}, "rep_b": {"inflight_count": 2}},
        )
        ranking = route(strat, PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_b"

    def test_inflight_tie_keeps_pool_order(self):
        """slow_cut=least-inflight tie-break: equal inflight → first replica in pool order."""
        strat = _strat(slow_cut=SlowCut.LEAST_INFLIGHT)
        provider = FakeRouteDataProvider(
            {"rep_a": {"inflight_count": 3}, "rep_b": {"inflight_count": 3}},
        )
        ranking = route(strat, PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), "r1")
        assert ranking[0] == "rep_a"


# --------------------------------------------------------------------------- #
# Capacity-gated token routing (slow_cut=capacity-token-aware)
# --------------------------------------------------------------------------- #
@pytest.mark.cpu
@pytest.mark.level0
class TestCapacityTokenAware:
    """``slow_cut=capacity-token-aware``: ``avail = cap - held_blocks × block_size`` gates
    and ranks (the router's own ref-count book via ``inflight_block_count`` — neither
    ``kv_perc`` nor ``INFLIGHT_TOKENS`` decides).

    Every strategy is built with ``SOFT_PICK_SEED`` (see ``_cap_strat``): soft-pick draws
    the winner from the tolerance band, so band cases are only reproducible with a seed.
    """

    def _cap_strat(self, **kwargs) -> KVCacheAwareStrategy:
        kwargs.setdefault("slow_cut", SlowCut.CAPACITY_TOKEN_AWARE)
        # Gate baseline pinned at 0.9: the fixtures below do arithmetic
        # against thresh = cap·(1−0.9) = 160 at cap=1600; threshold
        # sensitivity is covered by test_capacity_gate_threshold_via_load_threshold.
        kwargs.setdefault("load_threshold", 0.9)
        kwargs.setdefault("seed", SOFT_PICK_SEED)
        return _strat(**kwargs)

    def test_gate_filters_booked_replicas_and_picks_max_remaining(self):
        """Eligible = ``avail >= cap·(1-load_threshold)``; the largest ``remaining = avail - need`` wins.

        rep_a cache-rich but full (avail=16 < 160) → filtered; rep_b (remaining=797)
        beats eligible rep_c (477) → argmax picks rep_b.
        """
        strat = self._cap_strat()
        replicas = _replicas("rep_a", "rep_b", "rep_c")
        provider = FakeRouteDataProvider(
            {
                # rep_a: perfect cache but its held blocks nearly fill the pool → filtered.
                "rep_a": {
                    "num_gpu_blocks": 100,
                    "inflight_tokens": 1584,
                    "inflight_blocks": 99,
                    "gpu_hit_pct": 100,
                },
                # rep_b: no cache but plenty of room → avail=800.
                "rep_b": {"num_gpu_blocks": 100, "inflight_tokens": 800, "inflight_blocks": 50, "gpu_hit_pct": 0},
                # rep_c: eligible but less remaining than rep_b → argmax is non-trivial.
                "rep_c": {"num_gpu_blocks": 100, "inflight_tokens": 1120, "inflight_blocks": 70, "gpu_hit_pct": 0},
            },
        )
        # The capacity path is the ONLY path (no sticky binding), so this is also the unbound-request case.
        scores = strat.score(PROMPT_IDS, provider, replicas, request_id="r1")
        assert scores == [0.0, STICKY_TOP_SCORE, 0.0]
        # anti-kidnapping via route(): the cache-rich rep_a is dropped despite best cache
        assert route(strat, PROMPT_IDS, provider, replicas, "r1")[0] == "rep_b"

    def test_avail_tracks_inflight_blocks_not_tokens_or_kv_perc(self):
        """Free capacity = ``cap − held_blocks × block_size``; kv_perc and INFLIGHT_TOKENS are observation-only.

        rep_b's token ledger reads 0 (its prompts all hit prefix cache) yet its 94 held
        blocks are real occupancy: avail=96 < 160 → filtered; kv-rich rep_a wins.
        """
        strat = self._cap_strat()
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "kv_cache_usage_perc": 0.99, "inflight_tokens": 0},
                "rep_b": {
                    "num_gpu_blocks": 100,
                    "kv_cache_usage_perc": 0.0,
                    "inflight_tokens": 1500,
                    "inflight_blocks": 94,
                },
            },
        )
        scores = strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b"), request_id="r1")
        assert scores == [STICKY_TOP_SCORE, 0.0]

    def test_capacity_gate_threshold_via_load_threshold(self):
        """Gate threshold = ``cap·(1-load_threshold)``; load_threshold tunes it.

        cap=1600, do_shortcut=False so the binding can't short-circuit into the sticky
        branch: 0.99 (thresh=16) keeps both eligible; 0.85 (thresh=240) filters rep_a (avail=160).
        """
        data = {
            "rep_a": {"num_gpu_blocks": 100, "inflight_tokens": 1450, "inflight_blocks": 91, "gpu_hit_pct": 100},
            "rep_b": {"num_gpu_blocks": 100, "inflight_tokens": 800, "inflight_blocks": 50, "gpu_hit_pct": 0},
        }
        # Low threshold (thresh=1600·(1-0.99)=16): both eligible → rep_b wins on remaining.
        low = self._cap_strat(do_shortcut=False, load_threshold=0.99)
        p_low = FakeRouteDataProvider(dict(data))
        p_low.put_sticky_binding("r1", "rep_b")
        s_low = low.score(PROMPT_IDS, p_low, _replicas("rep_a", "rep_b"), request_id="r1")
        assert s_low[1] == STICKY_TOP_SCORE
        # High threshold (thresh=1600·(1-0.85)=240): rep_a (avail=160) filtered, rep_b wins.
        high = self._cap_strat(do_shortcut=False, load_threshold=0.85)
        p_high = FakeRouteDataProvider(dict(data))
        p_high.put_sticky_binding("r1", "rep_b")
        s_high = high.score(PROMPT_IDS, p_high, _replicas("rep_a", "rep_b"), request_id="r1")
        assert s_high[1] == STICKY_TOP_SCORE
        assert s_high[0] == 0.0  # rep_a filtered by the higher gate

    def test_soft_pick_randomizes_within_tolerance_band(self):
        """tie_tolerance soft-pick: near-equal remaining values share the win.

        |best| band = 0.05×797 ≈ 40: rep_c (gap 16) is in, rep_d (gap 624) and
        filtered rep_a are out; repeated calls return both winners.
        """
        strat = self._cap_strat(do_shortcut=False, tie_tolerance=0.05)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "inflight_blocks": 99},
                "rep_b": {"num_gpu_blocks": 100, "inflight_blocks": 50},
                "rep_c": {"num_gpu_blocks": 100, "inflight_blocks": 51},
                # Distant third stays outside the band (gap 624 ≫ 40) — excluded from the pick.
                "rep_d": {"num_gpu_blocks": 100, "inflight_blocks": 89},
            }
        )
        winners = {
            strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c", "rep_d"), request_id="r1").index(
                STICKY_TOP_SCORE
            )
            for _ in range(200)
        }
        assert winners == {1, 2}

    def test_soft_pick_strict_outside_tolerance_band(self):
        """Soft-pick only randomizes inside the band — a clear win stays deterministic.

        rep_b remaining=1597 vs rep_c 781: band = 0.05×1597 ≈ 80 ≪ 816 → strict argmax every call.
        """
        strat = self._cap_strat(do_shortcut=False, tie_tolerance=0.05)
        provider = FakeRouteDataProvider(
            {
                "rep_a": {"num_gpu_blocks": 100, "inflight_blocks": 99},
                "rep_b": {"num_gpu_blocks": 100, "inflight_blocks": 0},
                "rep_c": {"num_gpu_blocks": 100, "inflight_blocks": 51},
            }
        )
        winners = {
            strat.score(PROMPT_IDS, provider, _replicas("rep_a", "rep_b", "rep_c"), request_id="r1").index(
                STICKY_TOP_SCORE
            )
            for _ in range(50)
        }
        assert winners == {1}

    def test_soft_pick_negative_pool_uses_abs_best_band(self):
        """Band = tolerance × |best|: with a negative pool the band widens with |best|.

        [-5800, -6016, -6300], tol=0.05 → band 290: 0 and 1 (216 apart) share the
        win, 2 (500 apart) is out. All-equal values keep every index (diffs 0 ≤
        band) — the candidate set can never be empty.
        """
        strat = self._cap_strat(tie_tolerance=0.05)
        picked = {
            strat._soft_pick([-5800.0, -6016.0, -6300.0], maximize=True, tolerance=strat.tie_tolerance)
            for _ in range(50)
        }
        assert picked == {0, 1}
        picked = {strat._soft_pick([-5.0, -5.0], maximize=True, tolerance=strat.tie_tolerance) for _ in range(50)}
        assert picked == {0, 1}
