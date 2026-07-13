"""Per-sample logging built on the stdlib ``logging`` module.

A single dispatch handler on the root logger routes each record to its run's file by
run_id (O(1) lookup). ``run_id`` is carried implicitly by a ContextVar (via
:func:`sample_logging`) or bound explicitly (via :func:`get_logger`). A filtered
console handler keeps stdout readable next to a progress bar.

Typical use (eval): ``with sample_logging(run_id, path): ...``.
Explicit use (RL agent loop): ``get_logger(name, run_id)`` + ``add_file_handler``.
"""

from __future__ import annotations

from .context import current_run_id, get_logger
from .handlers import add_file_handler, cleanup_handlers, install_console_sink
from .session import sample_logging

__all__ = [
    "sample_logging",
    "current_run_id",
    "get_logger",
    "add_file_handler",
    "cleanup_handlers",
    "install_console_sink",
]
