"""Tests for the router debug facility (debug.py).

debug.py defines the ``UNI_AGENT_ROUTER_*`` namespace and its detection /
per-variable query primitives only — knob-name validation and value coercion
are balancer-side (see balancer/test_balancer_unit.py), and the driver's env
assembly lives in examples/agent_aware_router/run_infer.py.
"""

from __future__ import annotations

import pytest

from uni_agent.agent_aware_router.debug import (
    DEBUG_ENV,
    ENV_PREFIX,
    get_debug_var,
    is_debug_enabled,
)

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


class TestIsDebugEnabled:
    """The master switch gates the whole namespace."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " 1 "])
    def test_truthy_values(self, value):
        assert is_debug_enabled({DEBUG_ENV: value}) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "garbage"])
    def test_falsy_values(self, value):
        assert is_debug_enabled({DEBUG_ENV: value}) is False

    def test_unset_is_false(self):
        assert is_debug_enabled({}) is False


class TestGetDebugVar:
    """Per-variable query primitive."""

    def test_returns_raw_value_when_enabled(self):
        env = {DEBUG_ENV: "1", ENV_PREFIX + "SLOW_CUT": "least-inflight"}
        assert get_debug_var("SLOW_CUT", env) == "least-inflight"

    def test_name_case_insensitive(self):
        env = {DEBUG_ENV: "1", ENV_PREFIX + "SLOW_CUT": "x"}
        assert get_debug_var("slow_cut", env) == "x"

    def test_gate_off_returns_none_even_when_set(self):
        env = {ENV_PREFIX + "SLOW_CUT": "least-inflight"}
        assert get_debug_var("SLOW_CUT", env) is None

    def test_unset_or_empty_returns_none(self):
        assert get_debug_var("SLOW_CUT", {DEBUG_ENV: "1"}) is None
        assert get_debug_var("SLOW_CUT", {DEBUG_ENV: "1", ENV_PREFIX + "SLOW_CUT": ""}) is None
