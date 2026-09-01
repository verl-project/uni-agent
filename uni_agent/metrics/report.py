"""Export reduced metrics through verl's Tracking.

Collection and reduction stay verl-free. This module is the local sink: the
same ``Tracking.log`` path the trainer uses, with backends chosen by the caller
(console/file by default). Training still logs via the trainer's own Tracking
instance; this reporter is for every other entry point that already holds a
flat ``{name: scalar}`` dict.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_BACKENDS = ("console", "file")


def _load_tracking():
    from verl.utils.tracking import Tracking

    return Tracking


@contextmanager
def _file_logger_path(filepath: str | None):
    """Point verl's FileLogger at ``filepath`` for the duration of the block."""
    if filepath is None:
        yield
        return
    previous = os.environ.get("VERL_FILE_LOGGER_PATH")
    os.environ["VERL_FILE_LOGGER_PATH"] = filepath
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VERL_FILE_LOGGER_PATH", None)
        else:
            os.environ["VERL_FILE_LOGGER_PATH"] = previous


class MetricsReporter:
    """Thin wrapper around ``verl.utils.tracking.Tracking``.

    ``Tracking`` is imported lazily so ``uni_agent.metrics`` stays importable
    without verl. When Tracking is unavailable the scalars are logged in-process
    instead of raising.
    """

    def __init__(
        self,
        *,
        project: str = "uni_agent",
        experiment: str = "metrics",
        backends: Sequence[str] = _DEFAULT_BACKENDS,
        filepath: str | None = None,
    ) -> None:
        self._tracking: Any | None = None
        try:
            Tracking = _load_tracking()
        except ImportError:
            logger.info("verl Tracking unavailable; metrics will be logged locally")
            return
        with _file_logger_path(filepath):
            if filepath:
                Path(filepath).expanduser().parent.mkdir(parents=True, exist_ok=True)
            self._tracking = Tracking(project, experiment, default_backend=list(backends))

    def log(self, metrics: dict[str, Any], *, step: int = 0) -> None:
        if not metrics:
            return
        if self._tracking is None:
            logger.info("metrics: %s", json.dumps(metrics, default=str))
            return
        self._tracking.log(metrics, step)

    def finish(self) -> None:
        if self._tracking is not None:
            self._tracking.finish()
            self._tracking = None


def report_metrics(
    metrics: dict[str, Any],
    *,
    step: int = 0,
    project: str = "uni_agent",
    experiment: str = "metrics",
    backends: Sequence[str] = _DEFAULT_BACKENDS,
    filepath: str | None = None,
) -> None:
    """One-shot export: construct a reporter, log ``metrics``, then finish."""
    if not metrics:
        return
    reporter = MetricsReporter(
        project=project,
        experiment=experiment,
        backends=backends,
        filepath=filepath,
    )
    try:
        reporter.log(metrics, step=step)
    finally:
        reporter.finish()
