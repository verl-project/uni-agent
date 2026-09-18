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

"""CPU unit tests for the unified rl-insight emit facade.

The facade is a thin pass-through; most of its behaviour is covered indirectly
by the emitter and adapter tests (which drive the metric/trace channels end to
end through a fake backend). Pinned here: the env-off short-circuit (never
import ``rl_insight`` with the switch off) and the failure isolation (a
wiring error degrades to permanently-off instead of breaking the caller).
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

import uni_agent.rl_insight as facade

pytestmark = [pytest.mark.level0, pytest.mark.cpu]


def test_disabled_metric_short_circuits_before_import(monkeypatch: pytest.MonkeyPatch):
    # env off — the gate returns None and rl_insight is never imported.
    monkeypatch.delenv(facade.ENABLE_ENV, raising=False)
    monkeypatch.setattr(facade, "_enabled", None)
    monkeypatch.setattr(facade, "_dead", False)
    assert facade._get_rl_insight() is None
    facade.metric_count("x", 1)
    facade.metric_gauge("x", 2)
    facade.metric_histogram("x", 3)


def test_wiring_failure_isolates_callers(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    """init() raising must not propagate; the failure is warned once; later emits stay silent no-ops."""
    monkeypatch.setenv(facade.ENABLE_ENV, "1")
    monkeypatch.setattr(facade, "_enabled", None)
    monkeypatch.setattr(facade, "_dead", False)

    calls = []
    boom = types.SimpleNamespace(
        init=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backend down")),
        metric_count=lambda *a, **k: calls.append("count"),
    )
    monkeypatch.setitem(sys.modules, "rl_insight", boom)

    with caplog.at_level(logging.WARNING, logger=facade.__name__):
        assert facade._get_rl_insight() is None  # first call: warned, None
        facade.metric_count("x", 1)  # degraded: silent no-op, no re-init
    wiring_warnings = [r.getMessage() for r in caplog.records if "rl-insight wiring failed" in r.getMessage()]
    assert wiring_warnings == [
        "rl-insight wiring failed (RuntimeError: backend down); metric emit disabled for this process"
    ]
    assert calls == []


def test_dead_latch_prevents_retry(monkeypatch: pytest.MonkeyPatch):
    """Once latched dead, the gate returns None even before touching rl_insight."""
    monkeypatch.setattr(facade, "_enabled", True)
    monkeypatch.setattr(facade, "_dead", True)
    assert facade._get_rl_insight() is None
