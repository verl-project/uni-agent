"""uni-agent logging built on the stdlib ``logging`` module.

All uni-agent records are isolated on the ``uni_agent`` namespace mount point
(``propagate=False``).

A single dispatch handler there routes each record to its file by log ID; the
ID is carried implicitly by a ContextVar bound through :func:`sample_logging`.
A filtered console handler keeps stdout readable next to a progress bar.

Runtimes pass a :class:`LogContext` across process boundaries and bind it in the
process that executes the workload.
"""

from __future__ import annotations

from .context import LogContext, get_current_log_context
from .session import sample_logging

__all__ = [
    "sample_logging",
    "LogContext",
    "get_current_log_context",
]
