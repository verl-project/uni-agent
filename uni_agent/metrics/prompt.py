"""Prompt-level aggregation for bounded episode metric fragments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .model import MetricsFragment, MetricSummary

PROMPT_METRICS_SCHEMA_VERSION = 1
PROMPT_METRICS_SUMMARY_FIELD = "agent_metrics_summary"
PROMPT_METRICS_EXPORT_OWNER_FIELD = "agent_metrics_export_owner"
TRAINER_METRICS_EXPORT_OWNER = "trainer"


class EpisodeMetricsStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EpisodeMetricsObservation:
    episode_id: str
    status: EpisodeMetricsStatus
    fragments: tuple[MetricsFragment, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        object.__setattr__(self, "status", EpisodeMetricsStatus(self.status))
        object.__setattr__(self, "fragments", tuple(self.fragments))
        if not all(isinstance(fragment, MetricsFragment) for fragment in self.fragments):
            raise TypeError("fragments must contain MetricsFragment instances")


@dataclass(frozen=True, slots=True)
class PromptMetricsSummary:
    episode_count: int
    successful_episodes: int
    empty_episodes: int
    failed_episodes: int
    fragment_count: int
    complete: bool
    metrics: dict[str, MetricSummary]
    incomplete_reasons: tuple[str, ...] = ()
    schema_version: int = PROMPT_METRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        counts = (
            self.episode_count,
            self.successful_episodes,
            self.empty_episodes,
            self.failed_episodes,
            self.fragment_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("prompt metrics counts must be non-negative")
        if self.successful_episodes + self.empty_episodes + self.failed_episodes != self.episode_count:
            raise ValueError("episode outcome counts must equal episode_count")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "incomplete_reasons", tuple(self.incomplete_reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_count": self.episode_count,
            "successful_episodes": self.successful_episodes,
            "empty_episodes": self.empty_episodes,
            "failed_episodes": self.failed_episodes,
            "fragment_count": self.fragment_count,
            "complete": self.complete,
            "metrics": {name: summary.to_dict() for name, summary in self.metrics.items()},
            "incomplete_reasons": list(self.incomplete_reasons),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PromptMetricsSummary:
        raw_metrics = data.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise TypeError("prompt metrics summary must contain a metrics mapping")
        raw_incomplete_reasons = data.get("incomplete_reasons", ())
        if not isinstance(raw_incomplete_reasons, list | tuple):
            raise TypeError("prompt metrics incomplete_reasons must be a list or tuple")
        complete = data.get("complete")
        if not isinstance(complete, bool):
            raise TypeError("prompt metrics complete must be a bool")
        return cls(
            schema_version=_required_int(data, "schema_version"),
            episode_count=_required_int(data, "episode_count"),
            successful_episodes=_required_int(data, "successful_episodes"),
            empty_episodes=_required_int(data, "empty_episodes"),
            failed_episodes=_required_int(data, "failed_episodes"),
            fragment_count=_required_int(data, "fragment_count"),
            complete=complete,
            metrics={str(name): MetricSummary.from_dict(summary) for name, summary in raw_metrics.items()},
            incomplete_reasons=tuple(str(reason) for reason in raw_incomplete_reasons),
        )


def aggregate_prompt_metrics(observations: Iterable[EpisodeMetricsObservation]) -> PromptMetricsSummary:
    reasons: list[str] = []
    episode_statuses: dict[str, EpisodeMetricsStatus] = {}
    episode_order: list[str] = []
    episode_has_fragment: set[str] = set()
    fragments: dict[str, tuple[int, MetricsFragment]] = {}
    fragment_order = 0
    for observation in observations:
        previous_status = episode_statuses.get(observation.episode_id)
        if previous_status is None:
            episode_order.append(observation.episode_id)
        elif previous_status != observation.status:
            _append_reason(reasons, f"conflicting outcomes for episode {observation.episode_id}")
        episode_statuses[observation.episode_id] = observation.status
        for fragment in observation.fragments:
            episode_has_fragment.add(observation.episode_id)
            if fragment.episode_id != observation.episode_id:
                _append_reason(reasons, f"fragment episode mismatch for episode {observation.episode_id}")
            previous = fragments.get(fragment.fragment_id)
            if previous is None or fragment.revision > previous[1].revision:
                order = fragment_order if previous is None else previous[0]
                fragments[fragment.fragment_id] = (order, fragment)
            elif fragment.revision == previous[1].revision and fragment != previous[1]:
                _append_reason(reasons, f"conflicting fragment revision for {fragment.fragment_id}")
            fragment_order += 1

    for episode_id in episode_order:
        if episode_id not in episode_has_fragment:
            _append_reason(reasons, f"missing metrics fragment for episode {episode_id}")

    metric_values: dict[str, list[tuple[int, MetricSummary]]] = {}
    for order, fragment in sorted(fragments.values(), key=lambda item: item[0]):
        if fragment.schema_version != 1:
            _append_reason(reasons, f"unsupported metrics fragment schema {fragment.schema_version}")
            continue
        if not fragment.complete:
            if fragment.incomplete_reasons:
                for reason in fragment.incomplete_reasons:
                    _append_reason(reasons, f"{fragment.fragment_id}: {reason}")
            else:
                _append_reason(reasons, f"incomplete metrics fragment {fragment.fragment_id}")
        for name, summary in fragment.metrics.items():
            metric_values.setdefault(name, []).append((order, summary))

    metrics: dict[str, MetricSummary] = {}
    for name, ordered_summaries in metric_values.items():
        aggregations = {summary.aggregation for _, summary in ordered_summaries}
        if len(aggregations) != 1:
            _append_reason(reasons, f"conflicting aggregation for metric {name}")
            continue
        summaries = [summary for _, summary in sorted(ordered_summaries, key=lambda item: item[0])]
        values = [summary.value for summary in summaries]
        metrics[name] = MetricSummary(
            aggregation=summaries[0].aggregation,
            count=len(values),
            total=sum(values),
            minimum=min(values),
            maximum=max(values),
            last=values[-1],
        )

    statuses = [episode_statuses[episode_id] for episode_id in episode_order]
    return PromptMetricsSummary(
        episode_count=len(statuses),
        successful_episodes=statuses.count(EpisodeMetricsStatus.SUCCESS),
        empty_episodes=statuses.count(EpisodeMetricsStatus.EMPTY),
        failed_episodes=statuses.count(EpisodeMetricsStatus.FAILED),
        fragment_count=len(fragments),
        complete=not reasons,
        metrics=metrics,
        incomplete_reasons=tuple(reasons),
    )


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _required_int(data: Mapping[str, Any], name: str) -> int:
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"prompt metrics {name} must be an int")
    return value
