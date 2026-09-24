"""Trainer-side reduction for prompt metrics stored in TransferQueue tags."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .model import AggregationType, MetricSummary
from .prompt import PROMPT_METRICS_SCHEMA_VERSION, PROMPT_METRICS_SUMMARY_FIELD, PromptMetricsSummary

_MAX_INCOMPLETE_REASONS = 128
_SUPPORTED_TRACKING_AGGREGATIONS = frozenset({AggregationType.SUM, AggregationType.MIN, AggregationType.MAX})
_SUMMARY_COUNT_NAMES = (
    "prompts",
    "summaries",
    "complete_prompts",
    "incomplete_prompts",
    "missing_summary_prompts",
    "invalid_summary_prompts",
    "unsupported_schema_prompts",
    "unavailable_summary_prompts",
    "episodes",
    "successful_episodes",
    "empty_episodes",
    "failed_episodes",
    "fragments",
    "aggregation_conflicts",
    "unsupported_aggregation_metrics",
    "unsafe_metric_names",
)


@dataclass(frozen=True, slots=True)
class TrainerMetricsExport:
    metrics: dict[str, int | float]
    incomplete_reasons: tuple[str, ...]


@dataclass(slots=True)
class _MetricAccumulator:
    aggregation: AggregationType
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    prompt_count: int = 0

    def add(self, summary: MetricSummary) -> None:
        self.count += summary.count
        self.total += summary.total
        self.minimum = min(self.minimum, summary.minimum)
        self.maximum = max(self.maximum, summary.maximum)
        self.prompt_count += 1

    @property
    def value(self) -> float:
        match self.aggregation:
            case AggregationType.SUM:
                return self.total
            case AggregationType.MIN:
                return self.minimum
            case AggregationType.MAX:
                return self.maximum
            case _:
                raise AssertionError(f"Unsupported trainer aggregation: {self.aggregation.value}")


def aggregate_prompt_metrics_for_tracking(
    prompt_tags: Mapping[str, Mapping[str, Any]],
    selected_prompt_uids: Iterable[str],
    *,
    partition_id: str,
) -> TrainerMetricsExport:
    """Reduce selected prompt summaries into the trainer's flat tracking schema."""
    selected_uids = tuple(dict.fromkeys(str(uid) for uid in selected_prompt_uids))
    prefix = "training" if partition_id == "train" else "validation"
    metric_prefix = f"{prefix}/agent_metrics"
    counts: Counter[str] = Counter(prompts=len(selected_uids))
    reason_counts: Counter[str] = Counter()
    incomplete_reasons: list[str] = []
    metric_summaries: dict[str, list[MetricSummary]] = {}

    for uid in selected_uids:
        tag = prompt_tags.get(uid)
        raw_summary = tag.get(PROMPT_METRICS_SUMMARY_FIELD) if isinstance(tag, Mapping) else None
        if raw_summary is None:
            counts["missing_summary_prompts"] += 1
            _append_reason(incomplete_reasons, f"{uid}: missing prompt metrics summary")
            continue
        if not isinstance(raw_summary, Mapping):
            counts["invalid_summary_prompts"] += 1
            _append_reason(incomplete_reasons, f"{uid}: invalid prompt metrics summary")
            continue
        schema_version = raw_summary.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            counts["invalid_summary_prompts"] += 1
            _append_reason(incomplete_reasons, f"{uid}: invalid prompt metrics summary")
            continue
        if schema_version != PROMPT_METRICS_SCHEMA_VERSION:
            counts["unsupported_schema_prompts"] += 1
            _append_reason(incomplete_reasons, f"{uid}: unsupported prompt metrics schema {schema_version}")
            continue
        try:
            summary = PromptMetricsSummary.from_dict(raw_summary)
        except (KeyError, TypeError, ValueError):
            counts["invalid_summary_prompts"] += 1
            _append_reason(incomplete_reasons, f"{uid}: invalid prompt metrics summary")
            continue

        counts["summaries"] += 1
        counts["episodes"] += summary.episode_count
        counts["successful_episodes"] += summary.successful_episodes
        counts["empty_episodes"] += summary.empty_episodes
        counts["failed_episodes"] += summary.failed_episodes
        counts["fragments"] += summary.fragment_count
        if summary.complete:
            counts["complete_prompts"] += 1
        else:
            counts["incomplete_prompts"] += 1
            for reason in summary.incomplete_reasons:
                reason_counts[_reason_category(reason)] += 1
                _append_reason(incomplete_reasons, f"{uid}: {reason}")

        for name, metric_summary in summary.metrics.items():
            metric_summaries.setdefault(name, []).append(metric_summary)

    accumulators: dict[str, _MetricAccumulator] = {}
    for name, summaries in metric_summaries.items():
        aggregations = {summary.aggregation for summary in summaries}
        if len(aggregations) != 1:
            counts["aggregation_conflicts"] += 1
            reason_counts["conflict"] += 1
            _append_reason(incomplete_reasons, f"conflicting aggregation for metric {name}")
            continue
        aggregation = summaries[0].aggregation
        if aggregation not in _SUPPORTED_TRACKING_AGGREGATIONS:
            counts["unsupported_aggregation_metrics"] += 1
            reason_counts["unsupported_aggregation"] += 1
            _append_reason(
                incomplete_reasons,
                f"unsupported trainer aggregation for metric {name}: {aggregation.value}",
            )
            continue
        value_key = f"{metric_prefix}/values/{aggregation.value}/{name}"
        coverage_key = f"{metric_prefix}/coverage/sum/{name}/prompts"
        if (
            _trainer_aggregation_for_key(value_key) != aggregation
            or _trainer_aggregation_for_key(coverage_key) != AggregationType.SUM
        ):
            counts["unsafe_metric_names"] += 1
            reason_counts["unsafe_metric_name"] += 1
            _append_reason(
                incomplete_reasons,
                f"metric name conflicts with trainer aggregation rules: {name}",
            )
            continue
        accumulator = _MetricAccumulator(aggregation)
        for summary in summaries:
            accumulator.add(summary)
        accumulators[name] = accumulator

    counts["unavailable_summary_prompts"] = (
        counts["missing_summary_prompts"] + counts["invalid_summary_prompts"] + counts["unsupported_schema_prompts"]
    )
    metrics: dict[str, int | float] = {
        f"{metric_prefix}/summary/sum/{name}": counts[name] for name in _SUMMARY_COUNT_NAMES
    }
    for reason, count in sorted(reason_counts.items()):
        metrics[f"{metric_prefix}/summary/sum/reasons/{reason}"] = count
    for name, accumulator in sorted(accumulators.items()):
        aggregation = accumulator.aggregation.value
        metrics[f"{metric_prefix}/values/{aggregation}/{name}"] = accumulator.value
        metrics[f"{metric_prefix}/coverage/sum/{name}/prompts"] = accumulator.prompt_count
        metrics[f"{metric_prefix}/coverage/sum/{name}/observations"] = accumulator.count

    return TrainerMetricsExport(metrics=metrics, incomplete_reasons=tuple(incomplete_reasons))


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason in reasons:
        return
    if len(reasons) < _MAX_INCOMPLETE_REASONS:
        reasons.append(reason)
    elif len(reasons) == _MAX_INCOMPLETE_REASONS:
        reasons.append("additional incomplete reasons omitted")


def _reason_category(reason: str) -> str:
    if reason.startswith("missing metrics fragment"):
        return "missing_fragment"
    if reason.startswith("unsupported metrics fragment schema"):
        return "unsupported_fragment_schema"
    if reason.startswith("conflicting"):
        return "conflict"
    if ": " in reason or reason.startswith("incomplete metrics fragment"):
        return "incomplete_fragment"
    return "other"


def _trainer_aggregation_for_key(metric_name: str) -> AggregationType | None:
    metric_name = metric_name.lower()
    if "timing_s/" in metric_name or "timing_per_token_ms/" in metric_name:
        return AggregationType.SUM
    if "max" in metric_name or "maximum" in metric_name:
        return AggregationType.MAX
    if "min" in metric_name or "minimum" in metric_name:
        return AggregationType.MIN
    if "sum" in metric_name or "total" in metric_name:
        return AggregationType.SUM
    return None
