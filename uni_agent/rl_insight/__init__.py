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

"""Unified emit facade: routes metric + trace to rl-insight through one entry point.

  * **metric** (``metric_count`` / ``metric_gauge`` / ``metric_histogram``) →
    the bare ``rl_insight`` package (verl's ``RLInsightLogger`` does not
    expose these three APIs).
  * **trace** (``trace_span``) → verl ``RLInsightLogger.trace_span``, which
    owns the version gate and lazy init for the trace path.

Both channels share one env switch, ``VERL_RL_INSIGHT_ENABLE`` (the same flag
verl/trainer reads). The metric path is lazily wired: with the switch off the
facade never imports ``rl_insight`` and pays ~zero cost; wiring failures are
logged once and degrade to permanently-off (observability must never crash
its host).
"""

from __future__ import annotations

import logging
import os
from typing import Any

__all__ = [
    "ENABLE_ENV",
    "metric_count",
    "metric_gauge",
    "metric_histogram",
    "trace_span",
]

logger = logging.getLogger(__name__)

ENABLE_ENV = "VERL_RL_INSIGHT_ENABLE"


_enabled: bool | None = None
_dead: bool = False


def _get_rl_insight() -> Any | None:
    """Return the wired ``rl_insight`` module, or ``None`` when off/unavailable.

    Combines env gate + lazy import + one-shot init (re-init is a no-op inside
    rl_insight itself). Import/init failures log once and latch ``_dead``:
    later emits stay silent no-ops instead of raising into the hot path.
    """
    global _enabled, _dead
    if _enabled is None:
        _enabled = os.getenv(ENABLE_ENV) == "1"
    if not _enabled or _dead:
        return None
    try:
        import rl_insight

        rl_insight.init()
    except Exception as exc:  # noqa: BLE001 - telemetry must not break its host
        _dead = True
        logger.warning(
            "rl-insight wiring failed (%s: %s); metric emit disabled for this process", type(exc).__name__, exc
        )
        return None
    return rl_insight


# ── metric channel → bare rl_insight ─────────────────────────────────


def metric_count(name: str, amount: float = 1.0, documentation: str = "", **labels: Any) -> None:
    """Record a counter increment (no-op when emit is off)."""
    api = _get_rl_insight()
    if api is None:
        return
    api.metric_count(name, amount, documentation, **labels)


def metric_gauge(name: str, value: float, documentation: str = "", **labels: Any) -> None:
    """Record a gauge value (no-op when emit is off)."""
    api = _get_rl_insight()
    if api is None:
        return
    api.metric_gauge(name, value, documentation, **labels)


def metric_histogram(name: str, value: float, documentation: str = "", **labels: Any) -> None:
    """Record one histogram sample (no-op when emit is off).

    Values must be normalized to units the ``prometheus_client`` default
    buckets resolve.
    """
    api = _get_rl_insight()
    if api is None:
        return
    api.metric_histogram(name, value, documentation, **labels)


# ── trace channel → verl RLInsightLogger ──────────────────────────────


def trace_span(
    name: str,
    *,
    start_time_ns: int,
    end_time_ns: int,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Report one completed span through verl's RLInsightLogger (version-gated, no-op when off).

    verl's ``trace_span`` owns the env gate, the ``>=0.3.0`` version check, the
    capability probe, and lazy init — so this forwards unconditionally and lets
    verl no-op when rl-insight is off or too old.
    """
    from verl.utils.tracking import RLInsightLogger

    RLInsightLogger.trace_span(
        name=name,
        start_time_ns=start_time_ns,
        end_time_ns=end_time_ns,
        attributes=attributes,
    )
