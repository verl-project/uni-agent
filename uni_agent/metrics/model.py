"""Bounded task-metric summaries and cross-process fragment envelope."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AggregationType(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    LAST = "last"


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Fixed-size sufficient statistics for one metric inside one fragment."""

    aggregation: AggregationType
    count: int
    total: float
    minimum: float
    maximum: float
    last: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregation", AggregationType(self.aggregation))
        if self.count <= 0:
            raise ValueError(f"count must be positive, got {self.count}")
        for name in ("total", "minimum", "maximum", "last"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            object.__setattr__(self, name, value)
        if self.minimum > self.maximum:
            raise ValueError(f"minimum must not exceed maximum: {self.minimum} > {self.maximum}")

    @property
    def value(self) -> float:
        match self.aggregation:
            case AggregationType.MEAN:
                return self.total / self.count
            case AggregationType.SUM:
                return self.total
            case AggregationType.MIN:
                return self.minimum
            case AggregationType.MAX:
                return self.maximum
            case AggregationType.LAST:
                return self.last

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "aggregation": self.aggregation.value,
            "count": self.count,
            "sum": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "last": self.last,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricSummary:
        return cls(
            aggregation=AggregationType(str(data["aggregation"])),
            count=int(data["count"]),
            total=float(data["sum"]),
            minimum=float(data["min"]),
            maximum=float(data["max"]),
            last=float(data["last"]),
        )


@dataclass(frozen=True, slots=True)
class MetricsFragment:
    """One source's final, bounded metrics for an episode."""

    episode_id: str
    source_role: str
    source_instance: str
    fragment_id: str
    schema_version: int
    revision: int
    complete: bool
    metrics: dict[str, MetricSummary]
    incomplete_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("episode_id", "source_role", "source_instance", "fragment_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version <= 0:
            raise ValueError(f"schema_version must be positive, got {self.schema_version}")
        if self.revision <= 0:
            raise ValueError(f"revision must be positive, got {self.revision}")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "incomplete_reasons", tuple(self.incomplete_reasons))
        if not all(isinstance(summary, MetricSummary) for summary in self.metrics.values()):
            raise TypeError("metrics values must be MetricSummary instances")

    @property
    def errors(self) -> tuple[str, ...]:
        return self.incomplete_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "source_role": self.source_role,
            "source_instance": self.source_instance,
            "fragment_id": self.fragment_id,
            "schema_version": self.schema_version,
            "revision": self.revision,
            "complete": self.complete,
            "metrics": {name: summary.to_dict() for name, summary in self.metrics.items()},
            "incomplete_reasons": list(self.incomplete_reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricsFragment:
        raw_metrics = data.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("metrics fragment DTO must contain a metrics mapping")
        raw_incomplete_reasons = data.get("incomplete_reasons", data.get("errors", ()))
        if not isinstance(raw_incomplete_reasons, list | tuple):
            raise TypeError("metrics fragment incomplete_reasons must be a list or tuple")
        return cls(
            episode_id=str(data["episode_id"]),
            source_role=str(data["source_role"]),
            source_instance=str(data["source_instance"]),
            fragment_id=str(data["fragment_id"]),
            schema_version=int(data["schema_version"]),
            revision=int(data["revision"]),
            complete=bool(data["complete"]),
            metrics={str(name): MetricSummary.from_dict(summary) for name, summary in raw_metrics.items()},
            incomplete_reasons=tuple(str(reason) for reason in raw_incomplete_reasons),
        )
