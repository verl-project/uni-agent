"""Root-logger handlers: one dispatch handler that routes each record to its run's
file by log ID, plus a filtered stdout handler that stays readable next to a
progress bar."""

from __future__ import annotations

import atexit
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

from .context import _DATE_FORMAT, _FLUSH_EACH_LINE, _LOG_FORMAT, _NAME_WIDTH, _debug_enabled, _resolve_log_id
from .redaction import _redact_sensitive_text


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
        return _redact_sensitive_text(super().format(record))


_formatter = _AlignedFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)


_FLUSH_INTERVAL = 5.0  # seconds between writer flushes; higher = fewer HDFS round-trips, staler on-disk logs
_QUEUE_MAX = 100_000  # slow sink backs up into dropped records rather than unbounded memory
_STOP = object()


class _LogFileDispatch(logging.Handler):
    """Single root-logger handler: resolve each record's log ID and format it on the
    *calling* thread (cheap, and required while the ContextVar is visible), then
    enqueue all file I/O to a background writer thread. This keeps slow sinks (e.g. an HDFS
    FUSE mount, where every write is a network round-trip) off the asyncio event loop; the
    writer flushes on a fixed cadence (``_FLUSH_INTERVAL``s)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.setFormatter(_formatter)
        self._log_ids: set[str] = set()
        self._lock = threading.Lock()
        self._dropped = 0
        self._dropped_reported = 0
        self._start()
        atexit.register(self._shutdown)

    def _start(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._files: dict[str, TextIO] = {}  # writer-thread-owned
        self._dirty: set[TextIO] = set()
        self._thread = threading.Thread(target=self._run, name="uni-agent-log-writer", daemon=True)
        self._thread.start()

    def _reinit_after_fork(self) -> None:
        # child lost the writer thread; the parent owns the inherited files, so start clean
        self._log_ids = set()
        self._lock = threading.Lock()
        self._dropped = 0
        self._dropped_reported = 0
        self._start()

    # ----- caller thread: enqueue only, never blocks on I/O -----
    def register(self, log_id: str, path: Path) -> None:
        with self._lock:
            self._log_ids.add(log_id)
        self._submit(("open", log_id, str(path)))

    def unregister(self, log_id: str) -> None:
        with self._lock:
            self._log_ids.discard(log_id)
        self._submit(("close", log_id, None))

    def emit(self, record: logging.LogRecord) -> None:
        log_id = _resolve_log_id(record)
        if log_id is None:
            return
        with self._lock:
            registered = log_id in self._log_ids
        if not registered:
            return
        try:
            line = self.format(record) + "\n"
        except Exception:  # a bad format arg must never break the calling coroutine
            return
        self._submit(("write", log_id, line))

    def _submit(self, item: tuple) -> None:
        """Hand an op to the writer without blocking; drop (and count) if the queue is full."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._dropped += 1

    # ----- writer thread: owns every open/write/flush/close -----
    def _run(self) -> None:
        last_flush = time.monotonic()
        while True:
            timeout = max(0.0, _FLUSH_INTERVAL - (time.monotonic() - last_flush))
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._flush()
                last_flush = time.monotonic()
                continue
            if item is _STOP:
                self._drain()
                self._close_all()
                return
            self._apply(item)
            if time.monotonic() - last_flush >= _FLUSH_INTERVAL:
                self._flush()
                last_flush = time.monotonic()

    def _apply(self, item: tuple) -> None:
        op, log_id, arg = item
        if op == "write":
            file_obj = self._files.get(log_id)
            if file_obj is not None:
                try:
                    file_obj.write(arg)
                    if _FLUSH_EACH_LINE:
                        file_obj.flush()
                    else:
                        self._dirty.add(file_obj)
                except (ValueError, OSError):
                    pass
        elif op == "open":
            try:
                path = Path(arg)
                path.parent.mkdir(parents=True, exist_ok=True)
                previous = self._files.pop(log_id, None)
                if previous is not None:  # replaced an active log ID; close the old file
                    self._dirty.discard(previous)
                    try:
                        previous.close()
                    except OSError:
                        pass
                self._files[log_id] = open(path, "a", encoding="utf-8")
            except OSError:
                pass
        elif op == "close":
            file_obj = self._files.pop(log_id, None)
            if file_obj is not None:
                self._dirty.discard(file_obj)
                try:
                    file_obj.flush()
                    file_obj.close()
                except OSError:
                    pass

    def _flush(self) -> None:
        for file_obj in list(self._dirty):
            try:
                file_obj.flush()
            except OSError:
                pass
        self._dirty.clear()
        with self._lock:
            dropped = self._dropped
        if dropped != self._dropped_reported:  # surface drops (not via logging -> no recursion)
            print(
                f"[uni-agent logging] dropped {dropped} log records (writer/sink can't keep up)",
                file=sys.stderr,
                flush=True,
            )
            self._dropped_reported = dropped

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _STOP:
                self._apply(item)

    def _close_all(self) -> None:
        self._flush()
        for file_obj in list(self._files.values()):
            try:
                file_obj.close()
            except OSError:
                pass
        self._files.clear()
        self._dirty.clear()

    def _shutdown(self) -> None:
        """Best-effort drain+flush on normal process exit (daemon thread is killed abruptly otherwise)."""
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)


_dispatch = _LogFileDispatch()

# A forked child (some Ray/multiprocessing start methods) loses the writer thread; restart it.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_dispatch._reinit_after_fork)


def _add_file_handler(file_path: Path | str, log_id: str) -> None:
    _dispatch.register(log_id, Path(file_path))


def _cleanup_handler(log_id: str) -> None:
    _dispatch.unregister(log_id)


class _ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Records outside a LogContext always show; scoped records are WARNING+ only
        # (readable next to a progress bar), unless DEBUG_MODE also surfaces their INFO.
        if _resolve_log_id(record) is None:
            return True
        return _debug_enabled() or record.levelno >= logging.WARNING


_console_handler: logging.Handler | None = None


def _install_console_sink() -> None:
    """Install the filtered INFO console handler once per process."""
    global _console_handler
    root = logging.getLogger()
    if _console_handler is not None and _console_handler in root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_formatter)
    handler.addFilter(_ConsoleFilter())
    root.addHandler(handler)
    _console_handler = handler
