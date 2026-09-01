"""Per-task metrics built on stdlib ContextVars.

Each Task's :meth:`~uni_agent.tasks.Task.run` installs a :class:`MetricsCollector`
via :class:`task_metrics`; instrumentation sites record through the module-level
helpers (:func:`timing`, :func:`incr`, :func:`set_value`), which no-op when no
task is active -- so library code stays safe to run uninstrumented.

:meth:`MetricsCollector.metrics` crosses process/queue boundaries;
:func:`reduce_metrics` turns collected samples into a flat scalar dict;
:func:`aggregate` maps those onto the same verl keys the trainer logs;
:class:`MetricsReporter` / :func:`report_metrics` export that dict through
verl's ``Tracking`` (the same sink the trainer uses).
"""

from __future__ import annotations

from .collector import AggregationType, Metric, MetricsCollector, merge_metrics
from .context import current_collector, incr, set_value, task_metrics, timing
from .reduce import build_prompt_metrics_summary, reduce_metrics
from .report import MetricsReporter, report_metrics
from .verl_format import aggregate, reduce_agent_metrics, reduce_prompt_summaries

__all__ = [
    "AggregationType",
    "Metric",
    "MetricsCollector",
    "MetricsReporter",
    "task_metrics",
    "current_collector",
    "timing",
    "incr",
    "set_value",
    "merge_metrics",
    "reduce_metrics",
    "build_prompt_metrics_summary",
    "aggregate",
    "reduce_agent_metrics",
    "reduce_prompt_summaries",
    "report_metrics",
]
