"""Log-ID context and shared config for the logging package.

``log_id`` identifies one execution log and is carried implicitly through a
ContextVar set by ``sample_logging``.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
from dataclasses import dataclass

# Fixed-width logger-name column so the ``|`` separators line up; _AlignedFormatter
# trims each name to _NAME_WIDTH and fills ``shortname``.
_NAME_WIDTH = 22
_LOG_FORMAT = f"%(asctime)s | %(shortname)-{_NAME_WIDTH}s | %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _debug_enabled() -> bool:
    # DEBUG_MODE surfaces per-sample INFO on stdout; it does NOT lower the level to DEBUG.
    return _env_flag("DEBUG_MODE")


# Chatty libraries (incl. Modal's gRPC stack) pinned to WARNING to keep logs on the agent.
_QUIET_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "asyncio", "ray", "hpack", "h2", "grpclib", "modal")


def _resolve_level(level: int | str | None = None) -> int:
    """Resolve a logging level: explicit ``level`` (int or name) wins, then
    ``UNI_AGENT_LOG_LEVEL``, then INFO. Invalid names warn and fall back to INFO.

    Lowering it (e.g. to DEBUG) opens DEBUG records through the root logger and the
    global sinks; DEBUG_MODE keeps controlling only what the console filter lets
    through for LogContext-scoped records.

    UNI_AGENT_LOG_LEVEL should be in (CRITICAL, ERROR, WARNING, INFO, DEBUG)
    """
    raw = level if level is not None else os.getenv("UNI_AGENT_LOG_LEVEL", "")
    if isinstance(raw, int):
        return raw
    raw = str(raw).strip().upper()
    if not raw:
        return logging.INFO
    resolved = logging.getLevelName(raw)
    if not isinstance(resolved, int):
        print(f"[uni-agent logging] invalid log level {raw!r}; falling back to INFO", file=sys.stderr, flush=True)
        return logging.INFO
    return resolved


# Flush every line to disk (slower). Off by default: files flush on buffer-fill/close.
_FLUSH_EACH_LINE = _env_flag("LOG_FLUSH_EACH_LINE")


@dataclass(frozen=True)
class LogContext:
    """Explicit routing information for one logical execution log."""

    log_id: str
    log_path: str | None = None


_current_log_context: contextvars.ContextVar[LogContext | None] = contextvars.ContextVar(
    "uni_agent_log_context",
    default=None,
)


def get_current_log_context() -> LogContext | None:
    """Return the logging context bound to the current execution."""
    return _current_log_context.get()


def _resolve_log_id(record: logging.LogRecord) -> str | None:
    context = get_current_log_context()
    return context.log_id if context is not None else None
