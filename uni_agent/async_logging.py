import os
import sys
import threading
from pathlib import Path

from loguru import logger

# Replace loguru's default stderr sink; all routing is done by the dispatch sink below.
logger.remove()

_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[name]: <12} | {level: <8} | {message}"


def _debug_enabled() -> bool:
    # os.getenv returns a string, so a naive truthiness check treats "0"/"false"/""
    # as enabled. Parse the value explicitly instead.
    return os.getenv("DEBUG_MODE", "").strip().lower() in ("1", "true", "yes", "on")


if _debug_enabled():
    logger.add(
        sys.stdout,
        level="INFO",
        format=_LOG_FORMAT,
        filter=lambda record: "name" in record["extra"],
    )

# run_id -> (open file object, min level no). Touched from loguru's background
# (enqueue) writer thread and from the asyncio thread that adds/removes runs,
# so guard access with a lock.
_run_files: dict[str, tuple] = {}
_lock = threading.Lock()


def _dispatch(message) -> None:
    """Single sink that routes each record to its run's file in O(1).

    This replaces the previous "one loguru sink per run" design, where every
    record fanned out across all live sinks (O(num_runs) filtering per message)
    and each run leaked a sink + background thread + file descriptor when it
    wasn't cleaned up -- the main reason logging fell over under large-scale
    concurrent training.
    """
    record = message.record
    run_id = record["extra"].get("run_id")
    if run_id is None:
        return
    with _lock:
        entry = _run_files.get(run_id)
        if entry is None:
            return
        file_obj, min_no = entry
        if record["level"].no < min_no:
            return
        try:
            file_obj.write(message)
            file_obj.flush()
        except (ValueError, OSError):
            # File was closed/cleaned up between lookup and write; drop late records.
            pass


# One global sink: O(1) dispatch by run_id, a single background writer thread.
logger.add(_dispatch, level="DEBUG", format=_LOG_FORMAT, enqueue=True)


def add_file_handler(file_path: Path, run_id: str, level: str = "info") -> str:
    min_no = logger.level(level.upper()).no
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    file_obj = open(file_path, "a", encoding="utf-8")
    with _lock:
        previous = _run_files.get(run_id)
        _run_files[run_id] = (file_obj, min_no)
    if previous is not None:
        try:
            previous[0].close()
        except OSError:
            pass
    return run_id


def get_logger(name: str, run_id: str):
    return logger.bind(name=name, run_id=run_id)


def cleanup_handlers(run_id: str) -> None:
    with _lock:
        entry = _run_files.pop(run_id, None)
    if entry is not None:
        try:
            entry[0].close()
        except OSError:
            pass
