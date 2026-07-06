"""Run-id context and shared config for the logging package.

``run_id`` identifies one sample's log stream, carried implicitly via a ContextVar
(set by ``sample_logging``) or bound explicitly via :func:`get_logger`.
"""

from __future__ import annotations

import contextvars
import logging
import os

# Set by ``sample_logging`` for one sample; read by the dispatch handler and console
# filter to route/gate records.
_current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("uni_agent_run_id", default=None)

_LOG_FORMAT = "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _debug_enabled() -> bool:
    # DEBUG_MODE surfaces per-sample INFO on stdout; it does NOT lower the level to DEBUG.
    return _env_flag("DEBUG_MODE")


# Flush every line to disk (slower). Off by default: files flush on buffer-fill/close.
_FLUSH_EACH_LINE = _env_flag("LOG_FLUSH_EACH_LINE")


def resolve_run_id(record: logging.LogRecord) -> str | None:
    """The record's explicit run_id (from :func:`get_logger`), else the ambient one."""
    run_id = getattr(record, "run_id", None)
    return run_id if run_id is not None else _current_run_id.get()


def get_logger(name: str, run_id: str) -> logging.LoggerAdapter:
    """A logger whose records are tagged with ``run_id``, for explicit callers that
    don't use ``sample_logging`` (e.g. the RL agent loop). Pairs with
    :func:`~uni_agent.logging.add_file_handler` on the same ``run_id``."""
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)  # records are gated per file at the dispatch handler
    return logging.LoggerAdapter(lg, {"run_id": run_id})
