"""Task metrics collector: in-process samples -> serializable metrics dict.

One collector per task. Each named series is a :class:`Metric`: a list of
numeric samples plus an aggregation type (mean / sum / min / max). ``timing``
sums elapsed seconds; ``incr`` sums deltas; ``observe`` defaults to mean;
``set_value`` keeps the last write.

:meth:`MetricsCollector.metrics` returns a plain ``dict`` (JSON/pickle-safe)
that crosses process and queue boundaries unchanged; aggregation into flat
scalars happens later via :func:`uni_agent.metrics.reduce_metrics`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


class Metric:
    """Accumulate numeric samples and reduce them with one aggregation type."""

    def __init__(
        self,
        aggregation: str | AggregationType,
        value: float | list[float] | Metric | None = None,
    ) -> None:
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)
        if not isinstance(aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.aggregation = aggregation
        self.values: list[float] = []
        if value is not None:
            self.append(value)

    def append(self, value: float | list[float] | Metric) -> None:
        if isinstance(value, Metric):
            self.extend(value)
            return
        if isinstance(value, list):
            self.extend(value)
            return
        self.values.append(float(value))

    def extend(self, values: Metric | list[float]) -> None:
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            values = values.values
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        return self._aggregate(self.values, self.aggregation)

    @classmethod
    def _aggregate(cls, values: list[float], aggregation: AggregationType) -> float:
        if not values:
            raise ValueError("Cannot aggregate an empty metric.")
        match aggregation:
            case AggregationType.MEAN:
                return sum(values) / len(values)
            case AggregationType.SUM:
                return sum(values)
            case AggregationType.MIN:
                return min(values)
            case AggregationType.MAX:
                return max(values)

    def as_dict(self) -> dict[str, Any]:
        return {"aggregation": self.aggregation.value, "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        metric = cls(aggregation=data.get("aggregation", AggregationType.MEAN))
        metric.values = [float(v) for v in data.get("values", ())]
        return metric


class MetricsCollector:
    """Accumulates the metric samples of one task."""

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def _metric(self, name: str, aggregation: AggregationType) -> Metric:
        metric = self._metrics.get(name)
        if metric is None:
            metric = Metric(aggregation=aggregation)
            self._metrics[name] = metric
            return metric
        if metric.aggregation != aggregation:
            raise ValueError(f"Aggregation type mismatch for {name!r}: {metric.aggregation} != {aggregation}")
        return metric

    @contextmanager
    def timing(self, name: str) -> Iterator[None]:
        """Time a block and record elapsed seconds (summed)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start, aggregation=AggregationType.SUM)

    def observe(
        self,
        name: str,
        value: float,
        *,
        aggregation: str | AggregationType = AggregationType.MEAN,
    ) -> None:
        """Record one sample."""
        if isinstance(aggregation, str):
            aggregation = AggregationType(aggregation)
        self._metric(name, aggregation).append(value)

    def incr(self, name: str, delta: float = 1) -> None:
        """Add ``delta`` to a counter (summed)."""
        self.observe(name, delta, aggregation=AggregationType.SUM)

    def set_value(self, name: str, value: float) -> None:
        """Set a gauge (last write wins)."""
        self._metrics[name] = Metric(aggregation=AggregationType.MEAN, value=value)

    def metrics(self) -> dict[str, dict[str, Any]]:
        """Return the task's samples as a plain serializable dict."""
        return {name: metric.as_dict() for name, metric in self._metrics.items() if metric.values}

    def merge(self, other: dict[str, dict[str, Any]]) -> None:
        """Merge another metrics dict into this collector (e.g. a child component's)."""
        for name, entry in other.items():
            incoming = Metric.from_dict(entry)
            existing = self._metrics.get(name)
            if existing is None:
                self._metrics[name] = incoming
            else:
                existing.extend(incoming)


def merge_metrics(*metrics: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Combine metrics dicts into one, concatenating samples for shared names."""
    collector = MetricsCollector()
    for item in metrics:
        if item:
            collector.merge(item)
    return collector.metrics()
