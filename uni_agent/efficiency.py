"""Task-local elapsed-time accounting; no runtime or transport dependency.

Collectors cover terminal episodes, including exceptions and cancellation. They
exclude episodes that are still running or whose process was killed. Durations
are cumulative operation seconds, not wall-clock utilization: concurrent or
nested stages can overlap and must not be added as disjoint time fractions.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_metrics: ContextVar[dict[str, float] | None] = ContextVar("rollout_efficiency", default=None)


@contextmanager
def collect_efficiency() -> Iterator[dict[str, float]]:
    """Collect one episode; child coroutines share it, independent episodes do not."""
    metrics = {
        f"{stage}/{field}": 0.0
        for stage in ("sandbox_startup", "tool", "reward")
        for field in ("count", "total_s", "error_count", "cancelled_count")
    }
    metrics.update({f"tool/result_{status}_count": 0.0 for status in ("error", "timeout", "format_error")})
    token = _metrics.set(metrics)
    try:
        yield metrics
    finally:
        _metrics.reset(token)


def record_efficiency(key: str, value: float = 1.0) -> None:
    """Accumulate a scalar only while the caller has an active collector."""
    metrics = _metrics.get()
    if metrics is not None:
        metrics[key] = metrics.get(key, 0.0) + value


@contextmanager
def measure_efficiency(stage: str) -> Iterator[None]:
    """Measure the full call, preserving every exception and cancellation.

    ``error_count`` counts raised exceptions; returned failure statuses are
    recorded separately by the caller. ``count`` includes every exited call.
    """
    metrics = _metrics.get()
    if metrics is None:
        yield
        return
    started = time.perf_counter()
    error = cancelled = 0.0
    try:
        yield
    except asyncio.CancelledError:
        cancelled = 1.0
        raise
    except Exception:
        error = 1.0
        raise
    finally:
        for name, value in (
            ("count", 1.0),
            ("total_s", time.perf_counter() - started),
            ("error_count", error),
            ("cancelled_count", cancelled),
        ):
            key = f"{stage}/{name}"
            metrics[key] = metrics.get(key, 0.0) + value
