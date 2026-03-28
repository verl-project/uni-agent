from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

STATUS_META = {
    "queued": {"label": "Queued", "tone": "queued"},
    "running": {"label": "Running", "tone": "running"},
    "verify": {"label": "Verify", "tone": "verify"},
    "completed": {"label": "Completed", "tone": "completed"},
}

VERIFY_MARKERS = (
    "reward_spec",
    "Eval report:",
    "Beginning environment shutdown",
    "Environment shutdown completed",
    "num_turns:",
)

RUNNING_MARKERS = (
    "Beginning environment startup",
    "Runtime initialized",
    "STEP ",
    "MODEL INPUT",
    "ACTION:",
)

STEP_PATTERN = re.compile(r"STEP\s+(\d+)")


@dataclass(slots=True)
class FileSnapshot:
    size: int
    mtime_ns: int


@dataclass(slots=True)
class LogState:
    offset: int = 0
    tail: str = ""


@dataclass(slots=True)
class RunCache:
    run_id: str
    path: Path
    created_at: float
    last_update: float
    status: str = "queued"
    file_state: dict[str, FileSnapshot] = field(default_factory=dict)
    recent_file_events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=80))
    log_states: dict[str, LogState] = field(default_factory=dict)


def infer_status(file_names: set[str], log_excerpt: str) -> str:
    if "interaction_result.json" in file_names:
        return "completed"
    if "run.log" not in file_names:
        return "queued"
    if any(marker in log_excerpt for marker in VERIFY_MARKERS):
        return "verify"
    if any(marker in log_excerpt for marker in RUNNING_MARKERS):
        return "running"
    return "queued"


def extract_current_step(log_excerpt: str) -> int | None:
    matches = STEP_PATTERN.findall(log_excerpt)
    if not matches:
        return None
    return int(matches[-1])


def is_streamable_log(rel_path: str) -> bool:
    lowered = rel_path.lower()
    if lowered == "run.log":
        return True
    return lowered.endswith((".log", ".txt")) and any(keyword in lowered for keyword in ("verify", "eval"))


def make_log_label(rel_path: str) -> str:
    lowered = rel_path.lower()
    if lowered == "run.log":
        return "Run log"
    if "verify" in lowered or "eval" in lowered:
        return "Verify log"
    return Path(rel_path).name


class DashboardMonitor:
    def __init__(
        self,
        log_dir: Path,
        *,
        poll_interval: float = 1.0,
        max_log_tail_chars: int = 60_000,
        max_events: int = 4_000,
    ):
        self.log_dir = log_dir
        self.poll_interval = poll_interval
        self.max_log_tail_chars = max_log_tail_chars
        self.max_events = max_events
        self.runs: dict[str, RunCache] = {}
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.sequence = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scan_once()
        self._thread = threading.Thread(target=self._poll_loop, name="parallel-log-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self.scan_once()
            self._stop_event.wait(self.poll_interval)

    def scan_once(self) -> None:
        with self.lock:
            seen_run_ids: set[str] = set()
            if not self.log_dir.exists():
                return

            for entry in os.scandir(self.log_dir):
                if not entry.is_dir():
                    continue
                seen_run_ids.add(entry.name)
                self._scan_run(Path(entry.path))

            removed_run_ids = sorted(set(self.runs) - seen_run_ids)
            for run_id in removed_run_ids:
                self.runs.pop(run_id, None)
                self._push_event("run_removed", {"run_id": run_id})

    def _scan_run(self, run_path: Path) -> None:
        run_id = run_path.name
        stat = run_path.stat()
        cache = self.runs.get(run_id)
        is_new_run = cache is None
        if cache is None:
            cache = RunCache(
                run_id=run_id,
                path=run_path,
                created_at=stat.st_ctime,
                last_update=stat.st_mtime,
            )
            self.runs[run_id] = cache

        previous_status = cache.status
        previous_files = set(cache.file_state)
        current_files: dict[str, FileSnapshot] = {}
        changed = False
        latest_mtime = stat.st_mtime

        for root, _dirs, files in os.walk(run_path):
            root_path = Path(root)
            for file_name in sorted(files):
                file_path = root_path / file_name
                rel_path = file_path.relative_to(run_path).as_posix()
                file_stat = file_path.stat()
                current_files[rel_path] = FileSnapshot(size=file_stat.st_size, mtime_ns=file_stat.st_mtime_ns)
                latest_mtime = max(latest_mtime, file_stat.st_mtime)

                previous_snapshot = cache.file_state.get(rel_path)
                if is_streamable_log(rel_path):
                    delta = self._read_log_delta(cache, rel_path, file_path, file_stat.st_size)
                    if delta:
                        changed = True
                        self._push_event("log_append", {"run_id": run_id, "source": rel_path, "text": delta})
                elif previous_snapshot is None or (
                    previous_snapshot.size != file_stat.st_size or previous_snapshot.mtime_ns != file_stat.st_mtime_ns
                ):
                    changed = True
                    action = "created" if previous_snapshot is None else "modified"
                    event = {
                        "run_id": run_id,
                        "path": rel_path,
                        "action": action,
                        "size": file_stat.st_size,
                        "updated_at": file_stat.st_mtime,
                    }
                    cache.recent_file_events.appendleft(event)
                    self._push_event("file_changed", event)

        for rel_path in sorted(previous_files - set(current_files)):
            changed = True
            if rel_path in cache.log_states:
                cache.log_states.pop(rel_path, None)
            event = {
                "run_id": run_id,
                "path": rel_path,
                "action": "deleted",
                "size": 0,
                "updated_at": time.time(),
            }
            cache.recent_file_events.appendleft(event)
            self._push_event("file_changed", event)

        cache.file_state = current_files
        cache.last_update = latest_mtime
        run_log_tail = cache.log_states.get("run.log", LogState()).tail
        cache.status = infer_status(set(current_files), run_log_tail)
        if cache.status != previous_status:
            changed = True
            self._push_event("status_changed", {"run_id": run_id, "status": cache.status})

        if changed or is_new_run:
            self._push_event("run_update", {"run": self._serialize_run(cache)})

    def _read_log_delta(self, cache: RunCache, rel_path: str, log_path: Path, current_size: int) -> str:
        log_state = cache.log_states.setdefault(rel_path, LogState())
        start_offset = log_state.offset
        if current_size < start_offset:
            start_offset = 0

        if current_size == start_offset:
            return ""

        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(start_offset)
            delta = handle.read()

        log_state.offset = current_size
        if delta:
            log_state.tail = (log_state.tail + delta)[-self.max_log_tail_chars :]
        return delta

    def _push_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.sequence += 1
        event = {
            "id": self.sequence,
            "type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        }
        self.events.append(event)
        self.condition.notify_all()

    def wait_for_events(self, cursor: int, timeout: float = 2.0) -> list[dict[str, Any]]:
        with self.condition:
            if self.sequence <= cursor:
                self.condition.wait(timeout=timeout)
            return [event for event in self.events if event["id"] > cursor]

    def snapshot(self, query: str = "", status_filter: str = "all") -> dict[str, Any]:
        with self.lock:
            runs = [self._serialize_run(cache) for cache in self.runs.values()]

        if query:
            lowered = query.lower()
            runs = [run for run in runs if lowered in run["run_id"].lower()]
        if status_filter != "all":
            runs = [run for run in runs if run["status"] == status_filter]

        runs.sort(key=lambda item: (item["created_at"], item["run_id"]))
        stats = self._compute_stats(runs)
        return {
            "cursor": self.sequence,
            "log_dir": str(self.log_dir),
            "stats": stats,
            "runs": runs,
        }

    def read_log_chunk(
        self,
        run_id: str,
        source: str,
        *,
        before: int | None = None,
        chunk_size: int = 64_000,
    ) -> dict[str, Any]:
        with self.lock:
            cache = self.runs.get(run_id)
            if cache is None:
                raise FileNotFoundError(f"Run {run_id!r} not found")
            if source not in cache.file_state or not is_streamable_log(source):
                raise FileNotFoundError(f"Log source {source!r} not found for run {run_id!r}")
            file_path = cache.path / source
            file_size = cache.file_state[source].size

        end_offset = file_size if before is None else max(0, min(before, file_size))
        start_offset = max(0, end_offset - max(1, chunk_size))

        with file_path.open("rb") as handle:
            handle.seek(start_offset)
            payload = handle.read(end_offset - start_offset)

        if start_offset > 0:
            newline_idx = payload.find(b"\n")
            if newline_idx != -1:
                start_offset += newline_idx + 1
                payload = payload[newline_idx + 1 :]

        return {
            "run_id": run_id,
            "source": source,
            "label": make_log_label(source),
            "file_size": file_size,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "has_more": start_offset > 0,
            "text": payload.decode("utf-8", errors="replace"),
        }

    def _compute_stats(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {key: 0 for key in STATUS_META}
        most_recent_update = 0.0
        for run in runs:
            counts[run["status"]] += 1
            most_recent_update = max(most_recent_update, run["last_update"])
        return {
            "total": len(runs),
            "queued": counts["queued"],
            "running": counts["running"],
            "verify": counts["verify"],
            "completed": counts["completed"],
            "active": counts["running"] + counts["verify"],
            "most_recent_update": most_recent_update,
        }

    def _serialize_run(self, cache: RunCache) -> dict[str, Any]:
        file_names = sorted(cache.file_state)
        run_log_size = cache.file_state.get("run.log", FileSnapshot(size=0, mtime_ns=0)).size
        run_log_tail = cache.log_states.get("run.log", LogState()).tail
        current_step = extract_current_step(run_log_tail)
        log_sources = [
            {
                "key": rel_path,
                "label": make_log_label(rel_path),
                "size": cache.file_state.get(rel_path, FileSnapshot(size=0, mtime_ns=0)).size,
            }
            for rel_path in sorted(cache.log_states)
            if rel_path in cache.file_state
        ]
        log_contents = {
            rel_path: state.tail for rel_path, state in cache.log_states.items() if rel_path in cache.file_state
        }
        return {
            "run_id": cache.run_id,
            "path": str(cache.path),
            "status": cache.status,
            "status_label": STATUS_META[cache.status]["label"],
            "tone": STATUS_META[cache.status]["tone"],
            "created_at": cache.created_at,
            "last_update": cache.last_update,
            "file_count": len(file_names),
            "files": file_names,
            "run_log_size": run_log_size,
            "current_step": current_step,
            "log_excerpt": run_log_tail,
            "log_sources": log_sources,
            "log_contents": log_contents,
            "recent_file_events": list(cache.recent_file_events),
        }


def make_handler(monitor: DashboardMonitor, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._serve_static("index.html")
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True, "log_dir": str(monitor.log_dir)})
                return
            if parsed.path == "/api/snapshot":
                params = parse_qs(parsed.query)
                self._send_json(
                    monitor.snapshot(
                        query=params.get("q", [""])[0],
                        status_filter=params.get("status", ["all"])[0],
                    )
                )
                return
            if parsed.path == "/api/log":
                params = parse_qs(parsed.query)
                run_id = params.get("run_id", [""])[0]
                source = params.get("source", ["run.log"])[0]
                before_raw = params.get("before", [None])[0]
                chunk_size_raw = params.get("chunk_size", ["64000"])[0]
                if not run_id:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Missing run_id")
                    return
                try:
                    before = int(before_raw) if before_raw not in (None, "") else None
                    chunk_size = int(chunk_size_raw)
                    payload = monitor.read_log_chunk(run_id, source, before=before, chunk_size=chunk_size)
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid before or chunk_size")
                    return
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Requested log source not found")
                    return
                self._send_json(payload)
                return
            if parsed.path == "/api/stream":
                self._stream_events(parsed.query)
                return
            self._serve_static(parsed.path.lstrip("/"))

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _serve_static(self, relative_path: str) -> None:
            safe_path = (static_dir / relative_path).resolve()
            if static_dir.resolve() not in safe_path.parents and safe_path != static_dir.resolve():
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not safe_path.exists() or not safe_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            mime_type, _ = mimetypes.guess_type(str(safe_path))
            payload = safe_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _stream_events(self, query: str) -> None:
            params = parse_qs(query)
            cursor = int(params.get("cursor", ["0"])[0])
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                self._write_sse("connected", {"cursor": monitor.sequence}, event_id=monitor.sequence)
                while True:
                    events = monitor.wait_for_events(cursor, timeout=2.0)
                    if not events:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        continue
                    for event in events:
                        cursor = event["id"]
                        self._write_sse(event["type"], event["payload"], event_id=event["id"])
            except (BrokenPipeError, ConnectionResetError):
                return

        def _write_sse(self, event_type: str, payload: dict[str, Any], *, event_id: int) -> None:
            body = [
                f"id: {event_id}",
                f"event: {event_type}",
                f"data: {json.dumps(payload, ensure_ascii=False)}",
                "",
                "",
            ]
            self.wfile.write("\n".join(body).encode("utf-8"))
            self.wfile.flush()

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a streaming dashboard for uni_agent log directories.")
    parser.add_argument("--log-dir", default="/tmp/swebench_qwen3_coder", help="Directory that contains run folders.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Filesystem polling interval in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor = DashboardMonitor(Path(args.log_dir), poll_interval=args.poll_interval)
    monitor.start()
    static_dir = Path(__file__).parent / "static"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor, static_dir))
    print(f"Serving dashboard on http://{args.host}:{args.port} for {args.log_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        server.server_close()


if __name__ == "__main__":
    main()
