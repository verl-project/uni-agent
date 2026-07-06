"""Root-logger handlers: one dispatch handler that routes each record to its run's
file by run_id (O(1)), plus a filtered stdout handler that stays readable next to a
progress bar."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import TextIO

from .context import _DATE_FORMAT, _FLUSH_EACH_LINE, _LOG_FORMAT, _NAME_WIDTH, _debug_enabled, resolve_run_id


class _AlignedFormatter(logging.Formatter):
    """Compact the logger name into a fixed-width ``shortname`` so the ``|`` columns line
    up: drop the ``uni_agent.`` prefix, and if still too long keep the (informative) tail
    behind an ellipsis."""

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        if name.startswith("uni_agent."):
            name = name[len("uni_agent.") :]
        if len(name) > _NAME_WIDTH:
            name = "…" + name[-(_NAME_WIDTH - 1) :]
        record.shortname = name
        return super().format(record)


_formatter = _AlignedFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


class _RunFileDispatch(logging.Handler):
    """Single handler on the root logger. Keeps an open file per run_id and writes
    each record to the matching one; records with no registered run_id are dropped."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(_formatter)
        self._files: dict[str, tuple[TextIO, int]] = {}
        self._lock = threading.Lock()

    def register(self, run_id: str, path: Path, level: str) -> None:
        min_no = getattr(logging, level.upper(), logging.INFO)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_obj = open(path, "a", encoding="utf-8")
        with self._lock:
            previous = self._files.get(run_id)
            self._files[run_id] = (file_obj, min_no)
        if previous is not None:  # replaced an active run_id; close the old file
            try:
                previous[0].close()
            except OSError:
                pass

    def unregister(self, run_id: str) -> None:
        with self._lock:
            entry = self._files.pop(run_id, None)
        if entry is not None:
            try:
                entry[0].close()
            except OSError:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        run_id = resolve_run_id(record)
        if run_id is None:
            return
        with self._lock:
            entry = self._files.get(run_id)
        if entry is None:
            return
        file_obj, min_no = entry
        if record.levelno < min_no:
            return
        try:
            file_obj.write(self.format(record) + "\n")
            if _FLUSH_EACH_LINE:
                file_obj.flush()
        except (ValueError, OSError):
            pass


_dispatch = _RunFileDispatch()


def add_file_handler(file_path: Path | str, run_id: str, level: str = "info") -> str:
    """Open ``file_path`` and route this run_id's records to it until cleanup."""
    _dispatch.register(run_id, Path(file_path), level)
    return run_id


def cleanup_handlers(run_id: str) -> None:
    """Close and forget this run_id's file."""
    _dispatch.unregister(run_id)


def install_dispatch() -> None:
    """Attach the dispatch handler to the root logger (idempotent, additive)."""
    root = logging.getLogger()
    if _dispatch not in root.handlers:
        root.addHandler(_dispatch)


class _ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Run-level records (no run_id) always show; per-sample records are WARNING+ only
        # (readable next to a progress bar), unless DEBUG_MODE also surfaces their INFO.
        if resolve_run_id(record) is None:
            return True
        return _debug_enabled() or record.levelno >= logging.WARNING


_console_handler: logging.Handler | None = None


def install_console_sink(default_level: str | None = None) -> None:
    """(Re)install the stdout handler at ``default_level`` (``None`` -> none). Idempotent."""
    global _console_handler
    root = logging.getLogger()
    if _console_handler is not None:
        root.removeHandler(_console_handler)
        _console_handler = None
    if default_level is None:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(default_level.upper())
    handler.setFormatter(_formatter)
    handler.addFilter(_ConsoleFilter())
    root.addHandler(handler)
    _console_handler = handler


# Register dispatch at import so explicit callers work with no setup; sample_logging
# adds the console sink on first use.
install_dispatch()
