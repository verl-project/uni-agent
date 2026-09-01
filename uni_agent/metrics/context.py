"""Metrics context: one collector per task, carried implicitly by a ContextVar.

Each Task's :meth:`~uni_agent.tasks.Task.run` installs a collector via
:class:`task_metrics`; every instrumentation site below it records through the
module-level helpers, which no-op when no task is active.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from .collector import MetricsCollector

_current_collector: contextvars.ContextVar[MetricsCollector | None] = contextvars.ContextVar(
    "uni_agent_metrics", default=None
)


class task_metrics:
    """Bind a fresh :class:`MetricsCollector` to this block; usable as ``with`` or ``async with``."""

    def __init__(self) -> None:
        self.collector = MetricsCollector()
        self._token: contextvars.Token | None = None

    def _enter(self) -> MetricsCollector:
        self._token = _current_collector.set(self.collector)
        return self.collector

    def _exit(self) -> None:
        if self._token is not None:
            _current_collector.reset(self._token)
            self._token = None

    def __enter__(self) -> MetricsCollector:
        return self._enter()

    def __exit__(self, *exc_info) -> bool:
        self._exit()
        return False

    async def __aenter__(self) -> MetricsCollector:
        return self._enter()

    async def __aexit__(self, *exc_info) -> bool:
        self._exit()
        return False


def current_collector() -> MetricsCollector | None:
    """The active task's collector, or ``None`` outside a task."""
    return _current_collector.get()


@contextmanager
def timing(name: str) -> Iterator[None]:
    """Time the block on the active collector; no-op outside a task."""
    collector = _current_collector.get()
    if collector is None:
        yield
        return
    with collector.timing(name):
        yield


def incr(name: str, delta: float = 1) -> None:
    """Add to a counter on the active collector; no-op outside a task."""
    collector = _current_collector.get()
    if collector is not None:
        collector.incr(name, delta)


def set_value(name: str, value: float) -> None:
    """Set a gauge on the active collector; no-op outside a task."""
    collector = _current_collector.get()
    if collector is not None:
        collector.set_value(name, value)
