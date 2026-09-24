from __future__ import annotations

from dataclasses import replace

from uni_agent.metrics import (
    AggregationType,
    EpisodeMetricsObservation,
    EpisodeMetricsStatus,
    MetricsFragment,
    MetricSummary,
    aggregate_prompt_metrics,
)


def _fragment(
    episode_id: str,
    value: float,
    *,
    fragment_id: str | None = None,
    revision: int = 1,
    complete: bool = True,
) -> MetricsFragment:
    return MetricsFragment(
        episode_id=episode_id,
        source_role="gateway",
        source_instance="gateway-1",
        fragment_id=fragment_id or f"gateway-1:gateway:{episode_id}",
        schema_version=1,
        revision=revision,
        complete=complete,
        metrics={
            "gateway.request_s": MetricSummary(
                aggregation=AggregationType.MEAN,
                count=10,
                total=value * 10,
                minimum=value,
                maximum=value,
                last=value,
            )
        },
        incomplete_reasons=() if complete else ("request timing missing",),
    )


def test_prompt_summary_preserves_equal_episode_weighting():
    summary = aggregate_prompt_metrics(
        [
            EpisodeMetricsObservation("episode-1", EpisodeMetricsStatus.SUCCESS, (_fragment("episode-1", 10),)),
            EpisodeMetricsObservation("episode-2", EpisodeMetricsStatus.SUCCESS, (_fragment("episode-2", 30),)),
        ]
    )

    assert summary.complete
    assert summary.episode_count == 2
    assert summary.fragment_count == 2
    assert summary.metrics["gateway.request_s"].count == 2
    assert summary.metrics["gateway.request_s"].value == 20


def test_prompt_summary_replaces_duplicate_fragment_with_newer_revision():
    original = _fragment("episode-1", 10, fragment_id="fragment-1")
    replacement = replace(_fragment("episode-1", 30, fragment_id="fragment-1"), revision=2)

    summary = aggregate_prompt_metrics(
        [
            EpisodeMetricsObservation("episode-1", EpisodeMetricsStatus.EMPTY, (original,)),
            EpisodeMetricsObservation("episode-1", EpisodeMetricsStatus.EMPTY, (replacement,)),
        ]
    )

    assert summary.episode_count == 1
    assert summary.empty_episodes == 1
    assert summary.fragment_count == 1
    assert summary.metrics["gateway.request_s"].value == 30


def test_prompt_summary_accepts_multiple_sources_for_one_episode():
    gateway_fragment = _fragment("episode-1", 10)
    framework_fragment = replace(
        _fragment("episode-1", 3),
        source_role="framework",
        source_instance="framework-1",
        fragment_id="framework-1:framework:episode-1",
        metrics={
            "framework.episode_s": MetricSummary(
                aggregation=AggregationType.LAST,
                count=1,
                total=3,
                minimum=3,
                maximum=3,
                last=3,
            )
        },
    )

    summary = aggregate_prompt_metrics(
        [
            EpisodeMetricsObservation(
                "episode-1",
                EpisodeMetricsStatus.SUCCESS,
                (gateway_fragment, framework_fragment),
            )
        ]
    )

    assert summary.complete
    assert summary.episode_count == 1
    assert summary.fragment_count == 2
    assert set(summary.metrics) == {"gateway.request_s", "framework.episode_s"}


def test_prompt_summary_keeps_failure_denominator_and_missing_fragment_explicit():
    summary = aggregate_prompt_metrics([EpisodeMetricsObservation("episode-failed", EpisodeMetricsStatus.FAILED)])

    assert summary.failed_episodes == 1
    assert summary.fragment_count == 0
    assert not summary.complete
    assert summary.metrics == {}
    assert summary.incomplete_reasons == ("missing metrics fragment for episode episode-failed",)


def test_prompt_summary_propagates_fragment_incompleteness():
    summary = aggregate_prompt_metrics(
        [
            EpisodeMetricsObservation(
                "episode-1",
                EpisodeMetricsStatus.SUCCESS,
                (_fragment("episode-1", 10, complete=False),),
            )
        ]
    )

    assert not summary.complete
    assert summary.metrics["gateway.request_s"].value == 10
    assert summary.incomplete_reasons == ("gateway-1:gateway:episode-1: request timing missing",)
    assert summary.to_dict()["failed_episodes"] == 0


def test_prompt_summary_round_trips_through_versioned_dto():
    summary = aggregate_prompt_metrics(
        [EpisodeMetricsObservation("episode-1", EpisodeMetricsStatus.SUCCESS, (_fragment("episode-1", 10),))]
    )

    assert type(summary).from_dict(summary.to_dict()) == summary
