"""Per-execution logging built on the stdlib ``logging`` module.

A single dispatch handler on the root logger routes each record to its file by log ID.
The ID is carried implicitly by a ContextVar bound through :func:`sample_logging`.
A filtered console handler keeps stdout readable next to a progress bar.

Runtimes pass a :class:`LogContext` across process boundaries and bind it in the
process that executes the workload.
"""

from __future__ import annotations

from .context import LogContext
from .session import sample_logging

__all__ = [
    "sample_logging",
    "LogContext",
]
