"""Unit tests for the selectable load-score functions (strategies/load_score.py).

All functions return ``load ∈ [0,1]`` with the convention **bigger = more loaded**
(the inverse of the strategy's ``s_load``; the strategy converts via
``s_load = 1 - load``).

Normalized formula (default):
    running_usage = min(1, running / max_num_seqs)
    waiting_usage = min(1, waiting / max_num_seqs)
    load = a·kv + b·running_usage + c·waiting_usage      (a+b+c=1)
Default weights (a, b, c) = (0.4, 0.3, 0.3).
"""

from __future__ import annotations

import pytest

from uni_agent.llm_router.strategies.load_score import (
    DEFAULT_LOAD_WEIGHTS,
    DEFAULT_MAX_NUM_SEQS,
    LOAD_FNS,
    get_load_fn,
    load_kv_over_pressure,
    load_normalized,
    resolve_max_num_seqs,
)

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


# --------------------------------------------------------------------------- #
# load_normalized
# --------------------------------------------------------------------------- #
class TestLoadNormalized:
    def test_idle_replica_is_zero(self):
        """kv=0, running=0, waiting=0 → load=0 (all three terms vanish)."""
        assert load_normalized(0.0, 0, 0, max_num_seqs=64) == pytest.approx(0.0)

    def test_kv_only_contribution(self):
        """kv=0.5, running/waiting=0 → load = a·kv = 0.4·0.5 = 0.2."""
        assert load_normalized(0.5, 0, 0, max_num_seqs=64) == pytest.approx(0.2)

    def test_running_and_kv(self):
        """kv=0.5, running=32 (half of mns=64), waiting=0 → 0.4·0.5 + 0.3·0.5 = 0.35."""
        assert load_normalized(0.5, 32, 0, max_num_seqs=64) == pytest.approx(0.35)

    def test_running_clamped_to_one(self):
        """running > max_num_seqs → running_usage clamped to 1.0 (load stays ≤1)."""
        # kv=0.8, running=128 (>64) → run_usage=1.0; load = 0.4·0.8 + 0.3·1 = 0.62
        assert load_normalized(0.8, 128, 0, max_num_seqs=64) == pytest.approx(0.62)

    def test_waiting_term(self):
        """kv=0, running=0, waiting=10 → wait_usage=10/64; load = 0.3·(10/64)=0.046875."""
        assert load_normalized(0.0, 0, 10, max_num_seqs=64) == pytest.approx(0.3 * (10 / 64))

    def test_custom_weights_change_load(self):
        """weights (0.6,0.2,0.2) on kv=0.5,running=32 → 0.6·0.5 + 0.2·0.5 = 0.4."""
        load = load_normalized(0.5, 32, 0, max_num_seqs=64, weights=(0.6, 0.2, 0.2))
        assert load == pytest.approx(0.4)
        # and differs from the default-weight result (0.35)
        assert load != pytest.approx(0.35)

    def test_near_saturated_exceeds_threshold(self):
        """kv=1, running=mns, large waiting → load > 0.9 (overloaded regime)."""
        load = load_normalized(1.0, 64, 1000, max_num_seqs=64)
        assert load > 0.9
        assert load <= 1.0

    def test_max_num_seqs_zero_safe(self):
        """max_num_seqs=0 → running_usage AND waiting_usage degrade to 1.0 (no div-by-zero)."""
        # kv=0, running=5, waiting=0 → 0.4·0 + 0.3·1.0 + 0.3·1.0 = 0.6
        load = load_normalized(0.0, 5, 0, max_num_seqs=0)
        assert load == pytest.approx(0.6)  # 0.3·1.0 + 0.3·1.0


# --------------------------------------------------------------------------- #
# load_kv_over_pressure (legacy, = 1 - old s_load)
# --------------------------------------------------------------------------- #
class TestLoadKvOverPressure:
    def test_idle_is_zero(self):
        """kv=0,r=0,w=0 → old s_load=1.0 → load = 0.0 (consistent with normalized)."""
        assert load_kv_over_pressure(0.0, 0, 0, max_num_seqs=64) == pytest.approx(0.0)

    def test_inverts_old_formula(self):
        """load = 1 - (1-kv)/(1+running+waiting). kv=0.5,r=1,w=0 → 1 - 0.25 = 0.75."""
        assert load_kv_over_pressure(0.5, 1, 0, max_num_seqs=64) == pytest.approx(0.75)

    def test_near_full(self):
        """kv=0.9,r=5,w=0 → old s_load=0.1/6≈0.01667 → load≈0.98333."""
        assert load_kv_over_pressure(0.9, 5, 0, max_num_seqs=64) == pytest.approx(1 - 0.1 / 6)

    def test_ignores_max_num_seqs_and_weights(self):
        """legacy formula does not use max_num_seqs/weights — same result regardless."""
        a = load_kv_over_pressure(0.5, 1, 0, max_num_seqs=64)
        b = load_kv_over_pressure(0.5, 1, 0, max_num_seqs=999)
        c = load_kv_over_pressure(0.5, 1, 0, max_num_seqs=64, weights=(0.9, 0.05, 0.05))
        assert a == pytest.approx(b) == pytest.approx(c)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class TestLoadRegistry:
    def test_default_weights_tuple(self):
        """DEFAULT_LOAD_WEIGHTS == (0.4, 0.3, 0.3) and sums to 1."""
        assert DEFAULT_LOAD_WEIGHTS == (0.4, 0.3, 0.3)
        assert sum(DEFAULT_LOAD_WEIGHTS) == pytest.approx(1.0)

    def test_default_max_num_seqs(self):
        """DEFAULT_MAX_NUM_SEQS == 64 (matches infer_multi.sh / vLLM 24GB default)."""
        assert DEFAULT_MAX_NUM_SEQS == 64

    def test_registry_has_both_functions(self):
        """LOAD_FNS exposes 'normalized' (default) and 'kv_over_pressure' (legacy)."""
        assert LOAD_FNS["normalized"] is load_normalized
        assert LOAD_FNS["kv_over_pressure"] is load_kv_over_pressure

    def test_get_load_fn_returns_callable(self):
        """get_load_fn(name) returns the registered function."""
        assert get_load_fn("normalized") is load_normalized
        assert get_load_fn("kv_over_pressure") is load_kv_over_pressure

    def test_get_load_fn_unknown_raises(self):
        """unknown name → KeyError."""
        with pytest.raises(KeyError):
            get_load_fn("does-not-exist")


# --------------------------------------------------------------------------- #
# resolve_max_num_seqs (env)
# --------------------------------------------------------------------------- #
class TestResolveMaxNumSeqs:
    def test_default_when_env_unset(self, monkeypatch):
        """MAX_NUM_SEQS unset → default 64."""
        monkeypatch.delenv("MAX_NUM_SEQS", raising=False)
        assert resolve_max_num_seqs() == 64

    def test_env_override(self, monkeypatch):
        """MAX_NUM_SEQS=128 → 128."""
        monkeypatch.setenv("MAX_NUM_SEQS", "128")
        assert resolve_max_num_seqs() == 128

    def test_env_invalid_falls_back_to_default(self, monkeypatch):
        """MAX_NUM_SEQS=garbage → fall back to default (no crash)."""
        monkeypatch.setenv("MAX_NUM_SEQS", "not-a-number")
        assert resolve_max_num_seqs() == 64
