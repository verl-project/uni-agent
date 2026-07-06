"""Per-process setup and the per-sample ``sample_logging`` context manager."""

from __future__ import annotations

import logging
from pathlib import Path

from .context import _current_run_id
from .handlers import _dispatch, add_file_handler, cleanup_handlers, install_console_sink

# Chatty libraries (incl. Modal's gRPC stack) pinned to WARNING to keep logs on the agent.
_QUIET_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "asyncio", "ray", "hpack", "h2", "grpclib", "modal")

_process_logging_ready = False


def _ensure_process_logging() -> None:
    """Configure root logging once per process: swap in our dispatch handler + a
    filtered console sink, pin the level to INFO, and quiet noisy libraries."""
    global _process_logging_ready
    if _process_logging_ready:
        return
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.INFO)
    root.addHandler(_dispatch)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    install_console_sink("INFO")
    _process_logging_ready = True


class sample_logging:
    """Bind ``run_id`` to logs in this block; usable as ``with`` or ``async with``.

    Wires per-process logging on first use. With ``log_path`` the run's records are also
    written there; with ``None`` nothing hits disk. ``run_id`` must be unique per
    concurrent sample or the files collide.
    """

    def __init__(self, run_id: str, log_path: Path | str | None = None, level: str = "info"):
        self.run_id = run_id
        self.log_path = log_path
        self.level = level
        self._token = None

    def _enter(self) -> None:
        _ensure_process_logging()
        if self.log_path is not None:
            add_file_handler(self.log_path, self.run_id, self.level)
        self._token = _current_run_id.set(self.run_id)

    def _exit(self) -> None:
        if self._token is not None:
            _current_run_id.reset(self._token)
            self._token = None
        if self.log_path is not None:
            cleanup_handlers(self.run_id)

    def __enter__(self) -> sample_logging:
        self._enter()
        return self

    def __exit__(self, *exc_info) -> bool:
        self._exit()
        return False

    async def __aenter__(self) -> sample_logging:
        self._enter()
        return self

    async def __aexit__(self, *exc_info) -> bool:
        self._exit()
        return False
