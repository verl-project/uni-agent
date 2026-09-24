from __future__ import annotations

from uni_agent.metrics import (
    AggregationType,
    MetricSummary,
    PromptMetricsSummary,
    aggregate_prompt_metrics_for_tracking,
)


def _metric(aggregation: AggregationType, values: list[float]) -> MetricSummary:
    return MetricSummary(
        aggregation=aggregation,
        count=len(values),
        total=sum(values),
        minimum=min(values),
        maximum=max(values),
        last=values[-1],
    )


def _summary(*, metrics: dict[str, MetricSummary], complete: bool = True) -> PromptMetricsSummary:
    return PromptMetricsSummary(
        episode_count=2,
        successful_episodes=1,
        empty_episodes=0,
        failed_episodes=1,
        fragment_count=2,
        complete=complete,
        metrics=metrics,
        incomplete_reasons=() if complete else ("missing metrics fragment for episode failed",),
    )


def test_trainer_export_uses_trainer_safe_aggregation_keys_and_reports_coverage():
    first = _summary(
        metrics={
            "mean": _metric(AggregationType.MEAN, [5, 15]),
            "sum": _metric(AggregationType.SUM, [2, 3]),
            "latency_low": _metric(AggregationType.MIN, [4, 8]),
            "latency_high": _metric(AggregationType.MAX, [4, 8]),
            "last": _metric(AggregationType.LAST, [4, 8]),
        }
    )
    second = _summary(
        metrics={
            "mean": _metric(AggregationType.MEAN, [30]),
            "sum": _metric(AggregationType.SUM, [7]),
            "latency_low": _metric(AggregationType.MIN, [3]),
            "latency_high": _metric(AggregationType.MAX, [12]),
            "last": _metric(AggregationType.LAST, [12]),
        }
    )

    export = aggregate_prompt_metrics_for_tracking(
        {
            "uid-1": {"agent_metrics_summary": first.to_dict()},
            "uid-2": {"agent_metrics_summary": second.to_dict()},
        },
        ["uid-1", "uid-2", "uid-1"],
        partition_id="train",
    )

    assert export.metrics["training/agent_metrics/summary/sum/prompts"] == 2
    assert export.metrics["training/agent_metrics/summary/sum/episodes"] == 4
    assert export.metrics["training/agent_metrics/values/sum/sum"] == 12
    assert export.metrics["training/agent_metrics/values/min/latency_low"] == 3
    assert export.metrics["training/agent_metrics/values/max/latency_high"] == 12
    assert "training/agent_metrics/values/mean/mean" not in export.metrics
    assert "training/agent_metrics/values/last/last" not in export.metrics
    assert export.metrics["training/agent_metrics/summary/sum/unsupported_aggregation_metrics"] == 2
    assert export.metrics["training/agent_metrics/coverage/sum/sum/prompts"] == 2
    assert export.metrics["training/agent_metrics/coverage/sum/sum/observations"] == 3


def test_trainer_export_keeps_missing_and_incomplete_data_explicit():
    incomplete = _summary(metrics={"latency": _metric(AggregationType.SUM, [10])}, complete=False)
    complete_without_latency = _summary(metrics={})
    export = aggregate_prompt_metrics_for_tracking(
        {
            "incomplete": {"agent_metrics_summary": incomplete.to_dict()},
            "missing-metric": {"agent_metrics_summary": complete_without_latency.to_dict()},
            "invalid": {"agent_metrics_summary": {"schema_version": 1}},
            "unsupported": {"agent_metrics_summary": {"schema_version": 2}},
        },
        ["incomplete", "missing-metric", "missing-summary", "invalid", "unsupported"],
        partition_id="val",
    )

    assert export.metrics["validation/agent_metrics/values/sum/latency"] == 10
    assert export.metrics["validation/agent_metrics/coverage/sum/latency/prompts"] == 1
    assert export.metrics["validation/agent_metrics/summary/sum/incomplete_prompts"] == 1
    assert export.metrics["validation/agent_metrics/summary/sum/missing_summary_prompts"] == 1
    assert export.metrics["validation/agent_metrics/summary/sum/invalid_summary_prompts"] == 1
    assert export.metrics["validation/agent_metrics/summary/sum/unsupported_schema_prompts"] == 1
    assert export.metrics["validation/agent_metrics/summary/sum/unavailable_summary_prompts"] == 3
    assert export.metrics["validation/agent_metrics/summary/sum/reasons/missing_fragment"] == 1
    assert len(export.incomplete_reasons) == 4


def test_trainer_export_omits_metric_with_conflicting_aggregation():
    mean_summary = _summary(metrics={"latency": _metric(AggregationType.SUM, [10])})
    max_summary = _summary(metrics={"latency": _metric(AggregationType.MAX, [20])})

    export = aggregate_prompt_metrics_for_tracking(
        {
            "uid-1": {"agent_metrics_summary": mean_summary.to_dict()},
            "uid-2": {"agent_metrics_summary": max_summary.to_dict()},
        },
        ["uid-1", "uid-2"],
        partition_id="train",
    )

    assert "training/agent_metrics/values/sum/latency" not in export.metrics
    assert export.metrics["training/agent_metrics/summary/sum/aggregation_conflicts"] == 1
    assert export.metrics["training/agent_metrics/summary/sum/reasons/conflict"] == 1
    assert export.incomplete_reasons == ("conflicting aggregation for metric latency",)


def test_trainer_export_omits_name_that_selects_wrong_upstream_aggregation():
    summary = _summary(metrics={"request_max": _metric(AggregationType.SUM, [10])})

    export = aggregate_prompt_metrics_for_tracking(
        {"uid": {"agent_metrics_summary": summary.to_dict()}},
        ["uid"],
        partition_id="train",
    )

    assert "training/agent_metrics/values/sum/request_max" not in export.metrics
    assert export.metrics["training/agent_metrics/summary/sum/unsafe_metric_names"] == 1
    assert export.metrics["training/agent_metrics/summary/sum/reasons/unsafe_metric_name"] == 1
