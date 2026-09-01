import asyncio
import os
import time

import pytest

from uni_agent.metrics import (
    AggregationType,
    Metric,
    MetricsCollector,
    current_collector,
    incr,
    reduce_metrics,
    set_value,
    task_metrics,
    timing,
)


def test_metric_mean_sum_min_max():
    mean = Metric(aggregation="mean", value=[1.0, 3.0])
    assert mean.aggregate() == 2.0
    summed = Metric(aggregation=AggregationType.SUM, value=[1.0, 3.0])
    assert summed.aggregate() == 4.0
    assert Metric(aggregation="min", value=[1.0, 3.0]).aggregate() == 1.0
    maximum = Metric(aggregation="max", value=[1.0, 3.0])
    maximum.extend([4.0])
    assert maximum.aggregate() == 4.0


def test_metric_extend_rejects_aggregation_mismatch():
    left = Metric(aggregation="mean", value=1.0)
    right = Metric(aggregation="sum", value=2.0)
    with pytest.raises(ValueError, match="Aggregation type mismatch"):
        left.extend(right)


def test_collector_timing_records_summed_elapsed():
    collector = MetricsCollector()
    with collector.timing("phase"):
        time.sleep(0.01)

    snap = collector.metrics()
    entry = snap["phase"]
    assert entry["aggregation"] == "sum"
    assert len(entry["values"]) == 1
    assert entry["values"][0] >= 0.01


def test_collector_observe_accumulates_samples():
    collector = MetricsCollector()
    collector.observe("phase", 1.0)
    collector.observe("phase", 3.0)

    assert collector.metrics()["phase"] == {"aggregation": "mean", "values": [1.0, 3.0]}
    assert Metric.from_dict(collector.metrics()["phase"]).aggregate() == 2.0


def test_collector_incr_and_set_value():
    collector = MetricsCollector()
    collector.incr("count")
    collector.incr("count", 2)
    collector.set_value("gauge", 7)
    collector.set_value("gauge", 9)  # last write wins

    snap = collector.metrics()
    assert snap["count"] == {"aggregation": "sum", "values": [1.0, 2.0]}
    assert snap["gauge"] == {"aggregation": "mean", "values": [9.0]}


def test_collector_merge_combines_metrics():
    collector = MetricsCollector()
    collector.observe("phase", 2.0)
    collector.incr("count", 1)

    collector.merge(
        {
            "phase": {"aggregation": "mean", "values": [4.0]},
            "count": {"aggregation": "sum", "values": [2.0]},
        }
    )

    snap = collector.metrics()
    assert snap["phase"] == {"aggregation": "mean", "values": [2.0, 4.0]}
    assert snap["count"] == {"aggregation": "sum", "values": [1.0, 2.0]}


def test_merge_metrics_skips_empty_and_concatenates():
    from uni_agent.metrics import merge_metrics

    merged = merge_metrics(
        None,
        {},
        {"phase": {"aggregation": "sum", "values": [1.0]}},
        {"phase": {"aggregation": "sum", "values": [2.0]}, "count": {"aggregation": "sum", "values": [3.0]}},
    )
    assert merged["phase"] == {"aggregation": "sum", "values": [1.0, 2.0]}
    assert merged["count"] == {"aggregation": "sum", "values": [3.0]}


def test_task_metrics_binds_and_restores_context():
    assert current_collector() is None
    with task_metrics() as collector:
        assert current_collector() is collector
        with timing("phase"):
            pass
        incr("count")
        set_value("gauge", 3)
    assert current_collector() is None

    snap = collector.metrics()
    assert snap["phase"]["aggregation"] == "sum"
    assert len(snap["phase"]["values"]) == 1
    assert snap["count"] == {"aggregation": "sum", "values": [1.0]}
    assert snap["gauge"] == {"aggregation": "mean", "values": [3.0]}


def test_task_metrics_async():
    async def run():
        async with task_metrics() as collector:
            with timing("phase"):
                await asyncio.sleep(0)
            return collector.metrics()

    snap = asyncio.run(run())
    assert snap["phase"]["aggregation"] == "sum"
    assert current_collector() is None


def test_module_helpers_noop_outside_task():
    with timing("phase"):
        pass
    incr("count")
    set_value("gauge", 1)
    assert current_collector() is None


def test_reduce_metrics_uses_metric_aggregation_and_key_name():
    flat = reduce_metrics(
        {
            "loss": Metric(aggregation="mean", value=[1.0, 3.0]),
            "phase/max": [5.0, 8.0, 6.0],
            "phase/min": [0.1, 0.05, 0.2],
            "count": [1.0, 3.0],
        }
    )
    assert flat["loss"] == 2.0
    assert flat["phase/max"] == 8.0
    assert flat["phase/min"] == 0.05
    assert flat["count"] == 2.0


def test_reduce_metrics_aggregates_task_scalars():
    rows = [
        {
            "phase": {"aggregation": "sum", "values": [10.0]},
            "nested": {"aggregation": "mean", "values": [6.0]},
            "count": {"aggregation": "sum", "values": [1.0]},
        },
        {
            "phase": {"aggregation": "sum", "values": [20.0]},
            "nested": {"aggregation": "mean", "values": [3.0, 5.0]},
            "count": {"aggregation": "sum", "values": [3.0]},
        },
    ]

    flat = reduce_metrics(rows)

    assert flat["phase"] == 15.0
    # Per-task nested is mean of samples (6.0 and 4.0) -> mean 5.0.
    assert flat["nested"] == 5.0
    assert flat["count"] == 2.0


def test_reduce_metrics_skips_missing_metrics():
    flat = reduce_metrics([{"phase": {"aggregation": "sum", "values": [4.0]}}, {}])

    assert flat == {"phase": 4.0}
    assert reduce_metrics([]) == {}


def test_prompt_metrics_summary_merges_sufficient_stats():
    from uni_agent.metrics import build_prompt_metrics_summary

    summary = build_prompt_metrics_summary(
        expected_sessions=4,
        success_sessions=3,
        failed_sessions=1,
        unfinished_sessions=1,
        metrics=[
            {"task.total_s": {"aggregation": "sum", "values": [10.0]}},
            {"task.total_s": {"aggregation": "sum", "values": [20.0]}},
            {},
        ],
    )
    assert summary["expected_sessions"] == 4
    assert summary["success_sessions"] == 3
    assert summary["failed_sessions"] == 1
    assert summary["unfinished_sessions"] == 1
    assert summary["task.total_s"] == {"count": 2.0, "sum": 30.0, "min": 10.0, "max": 20.0}


def test_aggregate_maps_internal_names_to_verl_keys():
    from uni_agent.metrics import aggregate, reduce_agent_metrics, reduce_prompt_summaries

    rows = [
        {
            "task.total_s": {"aggregation": "sum", "values": [10.0]},
            "task.generate_s": {"aggregation": "sum", "values": [8.0]},
            "gateway.requests": {"aggregation": "sum", "values": [1.0, 1.0]},
        },
        {
            "task.total_s": {"aggregation": "sum", "values": [20.0]},
            "task.generate_s": {"aggregation": "sum", "values": [16.0]},
            "gateway.requests": {"aggregation": "sum", "values": [3.0]},
        },
        {},
    ]
    timing_raw, scalars = reduce_agent_metrics(rows)
    assert timing_raw["task/total/mean"] == 15.0
    assert timing_raw["task/total/max"] == 20.0
    assert scalars["gateway/requests/mean"] == 2.5
    assert scalars["agent_loop/slowest/task/total"] == 20.0
    assert scalars["agent_loop/slowest/task/generate"] == 16.0

    flat = aggregate(rows)
    assert flat["timing_s/task/total/mean"] == 15.0
    assert flat["timing_s/task/generate/max"] == 16.0
    assert flat["gateway/requests/max"] == 3.0

    prompt = reduce_prompt_summaries(
        [
            {"success_sessions": 3, "failed_sessions": 1, "unfinished_sessions": 1},
            {"success_sessions": 0, "failed_sessions": 2, "unfinished_sessions": 0},
        ]
    )
    assert prompt == {
        "agent_loop/failed_prompts": 1.0,
        "agent_loop/success_sessions": 3.0,
        "agent_loop/failed_sessions": 3.0,
        "agent_loop/unfinished_sessions": 1.0,
    }


def test_report_metrics_uses_tracking(monkeypatch):
    from uni_agent.metrics import report as report_mod
    from uni_agent.metrics.report import report_metrics

    calls: list[tuple[str, object]] = []

    class _FakeTracking:
        def __init__(self, project, experiment, default_backend):
            calls.append(("init", (project, experiment, tuple(default_backend))))

        def log(self, data, step):
            calls.append(("log", (dict(data), step)))

        def finish(self):
            calls.append(("finish", None))

    monkeypatch.setattr(report_mod, "_load_tracking", lambda: _FakeTracking)
    report_metrics({"phase": 1.5}, experiment="unit", step=3)

    assert calls == [
        ("init", ("uni_agent", "unit", ("console", "file"))),
        ("log", ({"phase": 1.5}, 3)),
        ("finish", None),
    ]


def test_report_metrics_sets_file_logger_path(monkeypatch, tmp_path):
    from uni_agent.metrics import report as report_mod
    from uni_agent.metrics.report import report_metrics

    seen: dict[str, str | None] = {}

    class _FakeTracking:
        def __init__(self, project, experiment, default_backend):
            seen["path"] = os.environ.get("VERL_FILE_LOGGER_PATH")

        def log(self, data, step):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(report_mod, "_load_tracking", lambda: _FakeTracking)
    monkeypatch.delenv("VERL_FILE_LOGGER_PATH", raising=False)
    filepath = str(tmp_path / "nested" / "metrics.jsonl")
    report_metrics({"phase": 1.0}, filepath=filepath)

    assert seen["path"] == filepath
    assert os.environ.get("VERL_FILE_LOGGER_PATH") is None
    assert (tmp_path / "nested").is_dir()


def test_report_metrics_skips_empty(monkeypatch):
    from uni_agent.metrics import report as report_mod
    from uni_agent.metrics.report import report_metrics

    monkeypatch.setattr(
        report_mod,
        "_load_tracking",
        lambda: (_ for _ in ()).throw(AssertionError("Tracking must not be constructed")),
    )
    report_metrics({})


def test_report_metrics_falls_back_without_verl(monkeypatch, caplog):
    from uni_agent.metrics import report as report_mod
    from uni_agent.metrics.report import report_metrics

    def _missing():
        raise ImportError("no verl")

    monkeypatch.setattr(report_mod, "_load_tracking", _missing)
    caplog.set_level("INFO")
    report_metrics({"phase": 2.0}, experiment="unit")
    assert any("metrics: " in rec.message and "2.0" in rec.message for rec in caplog.records)


def test_task_run_attaches_metrics():
    from uni_agent.tasks.base import Task, TaskConfig, TaskResult

    class _TimedTask(Task):
        async def run(self) -> TaskResult:
            async with task_metrics() as collector:
                with timing("task.total_s"):
                    with timing("reward.s"):
                        pass
                    result = TaskResult(reward=1.0)
                result.metrics = collector.metrics()
                return result

    result = asyncio.run(_TimedTask(TaskConfig(sandbox={"provider": "local"})).run())
    assert result.metrics is not None
    assert result.metrics["task.total_s"]["aggregation"] == "sum"
    assert result.metrics["reward.s"]["aggregation"] == "sum"
    assert current_collector() is None
