"""uni-agent logging configuration and the per-run ``sample_logging`` context
manager."""

from __future__ import annotations

import logging
from pathlib import Path

from .context import _QUIET_LOGGERS, LogContext, _current_log_context, _resolve_level
from .handlers import _add_file_handler, _cleanup_handler, _dispatch, _install_console_sink, _mount

_process_logging_ready = False


def _setup_console_logging(level: int | str | None = None) -> None:
    """Configure uni-agent logging on the ``uni_agent`` namespace mount point."""
    resolved = _resolve_level(level)
    mount = _mount()
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    mount.setLevel(resolved)
    mount.propagate = False
    _install_console_sink(resolved)


def _ensure_process_logging() -> None:
    """Set up logging once per process: global namespace configuration plus our
    per-run file-dispatch handler (records outside a LogContext are ignored).
    """
    global _process_logging_ready
    if _process_logging_ready:
        return
    mount = _mount()
    if not mount.handlers and mount.level == logging.NOTSET:
        _setup_console_logging()
    if _dispatch not in mount.handlers:
        mount.addHandler(_dispatch)
    _process_logging_ready = True


class sample_logging:
    """Bind a log ID to records in this block; usable as ``with`` or ``async with``.

    Wires per-process logging on first use. With ``log_path`` the run's records are also
    written there; with ``None`` nothing hits disk. ``log_id`` must be unique for every
    concurrently active file in the process.
    """

    def __init__(self, log_id: str, log_path: Path | str | None = None):
        self.log_id = log_id
        self.log_path = log_path
        self._token = None

    @classmethod
    def from_context(cls, context: LogContext) -> sample_logging:
        return cls(context.log_id, context.log_path)

    def _enter(self) -> None:
        _ensure_process_logging()
        if self.log_path is not None:
            _add_file_handler(self.log_path, self.log_id)
        self._token = _current_log_context.set(
            LogContext(
                log_id=self.log_id,
                log_path=str(self.log_path) if self.log_path is not None else None,
            )
        )

    def _exit(self) -> None:
        if self._token is not None:
            _current_log_context.reset(self._token)
            self._token = None
        if self.log_path is not None:
            _cleanup_handler(self.log_id)

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
