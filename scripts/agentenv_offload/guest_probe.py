#!/usr/bin/env python3
"""Small stateful HTTP probe for sandbox pause/resume experiments.

GET /state reads and hashes every anonymous-memory byte without changing the
oracle. POST /tool changes the oracle explicitly. Command deduplication lives
only in this process's RAM: it survives runtime snapshots, not process restart,
rollback to an earlier snapshot, or loss of the process. It is not an
exactly-once guarantee for external effects.

The random nonce is written only to anonymous memory and HTTP responses, never
to the probe's files or startup log. A VM snapshot necessarily contains it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import random
import secrets
import signal
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
CHUNK = 64 * 1024
NONCE_BYTES = 32


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_start_ticks() -> int | None:
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


class ProbeState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.lock = threading.Lock()
        self.pid = os.getpid()
        self.started_ns = time.time_ns()
        self.start_ticks = process_start_ticks()
        self.counter = 0
        self.memory_bytes = args.memory_mib * MIB
        self.seed = args.seed
        self.entropy = args.entropy
        self.dirty_ratio = args.dirty_ratio
        self.results: dict[str, tuple[tuple[int, str], dict[str, Any] | None]] = {}
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
        self.owned_dir = Path(tempfile.mkdtemp(prefix="probe-", dir=args.workdir)).resolve()
        self.fd_path = self.owned_dir / "fd-source.bin"
        self.journal_path = self.owned_dir / "tool-journal.jsonl"
        self.fd_path.write_bytes(bytes(range(256)) * 16)
        self.reader = self.fd_path.open("rb", buffering=0)
        self.reader.seek(7)
        self.journal = self.journal_path.open("w+b", buffering=0)
        if os.name == "posix":
            self.memory = mmap.mmap(
                -1,
                self.memory_bytes,
                flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        else:
            self.memory = mmap.mmap(-1, self.memory_bytes, access=mmap.ACCESS_WRITE)
        self._fill(0, self.memory_bytes, 0)
        self.memory[:NONCE_BYTES] = secrets.token_bytes(NONCE_BYTES)

    def _fill(self, start: int, count: int, version: int) -> None:
        """Write every byte, including every page, using bounded temporary data."""
        generator = random.Random(f"{self.seed}:{version}")
        end = start + count
        for offset in range(start, end, CHUNK):
            length = min(CHUNK, end - offset)
            if self.entropy == "high":
                data = generator.getrandbits(length * 8).to_bytes(length, "little")
            else:
                data = bytes([(self.seed + version) % 256]) * length
            self.memory[offset : offset + length] = data

    def _snapshot(self) -> dict[str, Any]:
        """Caller holds the lock; independent file reads preserve the open FD."""
        # Hash the entire mapped buffer directly. Slicing mmap creates a bytes
        # copy for every chunk; the buffer protocol preserves the same SHA-256
        # oracle while avoiding those temporary allocations and copies.
        with memoryview(self.memory) as memory:
            memory_sha256 = hashlib.sha256(memory).hexdigest()
        return {
            "schema_version": 1,
            "pid": self.pid,
            "process_started_ns": self.started_ns,
            "process_start_ticks": self.start_ticks,
            "counter": self.counter,
            "nonce": self.memory[:NONCE_BYTES].hex(),
            "memory_sha256": memory_sha256,
            "memory_bytes": self.memory_bytes,
            "entropy": self.entropy,
            "seed": self.seed,
            "dirty_ratio": self.dirty_ratio,
            "fd_offset": self.reader.tell(),
            "fd_sha256": file_digest(self.fd_path),
            "fd_size": self.fd_path.stat().st_size,
            "journal_sha256": file_digest(self.journal_path),
            "journal_size": self.journal_path.stat().st_size,
            "owned_dir": str(self.owned_dir),
            "dedupe_scope": "process-memory-only; no restart or rollback guarantee",
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot()

    def tool(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if set(body) - {"command_id", "read_bytes", "payload"}:
            return 400, {"error": "unknown tool field"}
        command_id = body.get("command_id")
        read_bytes = body.get("read_bytes", 17)
        payload = body.get("payload", "")
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 128:
            return 400, {"error": "command_id must contain 1..128 characters"}
        if type(read_bytes) is not int or not 0 <= read_bytes <= 4096:
            return 400, {"error": "read_bytes must be an integer in 0..4096"}
        if not isinstance(payload, str) or len(payload.encode("utf-8")) > 1024:
            return 400, {"error": "payload must be a string of at most 1024 UTF-8 bytes"}
        signature = (read_bytes, payload)
        with self.lock:
            if command_id in self.results:
                previous_signature, result = self.results[command_id]
                if signature != previous_signature:
                    return 409, {"error": "command_id already used with different arguments"}
                if result is None:
                    return 409, {"error": "command outcome unknown; replay refused"}
                return 200, {**result, "duplicate": True}
            # Reserve before effects. Any error leaves an unknown outcome that
            # cannot be replayed accidentally in the same process.
            self.results[command_id] = (signature, None)
            try:
                data = self.reader.read(read_bytes)
                next_counter = self.counter + 1
                record = {"command_id": command_id, "counter": next_counter, "payload": payload}
                self.journal.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
                os.fsync(self.journal.fileno())
                dirty_bytes = math.ceil((self.memory_bytes - NONCE_BYTES) * self.dirty_ratio)
                self._fill(NONCE_BYTES, dirty_bytes, next_counter)
                self.counter = next_counter
                result = {
                    "ok": True,
                    "duplicate": False,
                    "command_id": command_id,
                    "read_hex": data.hex(),
                    "dirty_bytes": dirty_bytes,
                    "state": self._snapshot(),
                }
                self.results[command_id] = (signature, result)
                return 200, result
            except Exception as exc:
                return 500, {"error": "command outcome unknown", "error_type": type(exc).__name__}

    def close(self) -> None:
        self.reader.close()
        self.journal.close()
        self.memory.close()


class ProbeServer(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: ProbeState) -> None:
        self.state = state
        super().__init__(address, ProbeHandler)


class ProbeHandler(BaseHTTPRequestHandler):
    server: ProbeServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format: str, *args: Any) -> None:
        # Neither request data nor nonce-bearing responses belong in logs.
        return

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {"ok": True, "pid": self.server.state.pid})
        elif self.path == "/state":
            self._reply(200, self.server.state.snapshot())
        else:
            self._reply(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        if self.path not in {"/tool", "/shutdown"}:
            self._reply(404, {"error": "unknown endpoint"})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("chunked requests are not accepted")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 4096:
                raise ValueError("request body exceeds 4096 bytes")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, UnicodeError):
            self._reply(400, {"error": "invalid JSON object or body size"})
            return
        if self.path == "/shutdown":
            if body:
                self._reply(400, {"error": "shutdown requires an empty JSON object"})
                return
            self._reply(200, {"ok": True, "pid": self.server.state.pid})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            status, result = self.server.state.tool(body)
            self._reply(status, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mib", type=int, default=64)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--entropy", choices=("low", "high"), default="low")
    parser.add_argument("--dirty-ratio", type=float, default=0.0)
    args = parser.parse_args()
    if args.memory_mib < 1 or not 0 <= args.port <= 65535 or not 0 <= args.dirty_ratio <= 1:
        parser.error("memory-mib must be positive; port 0..65535; dirty-ratio 0..1")
    # Avoid an accidental core dump serializing the nonce; swap and VM snapshot
    # policy remain the host's responsibility.
    if os.name == "posix":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    state = ProbeState(args)
    try:
        with ProbeServer((args.host, args.port), state) as server:

            def request_stop(signum: int, frame: Any) -> None:
                threading.Thread(target=server.shutdown, daemon=True).start()

            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
            print(
                json.dumps(
                    {
                        "event": "probe_ready",
                        "pid": state.pid,
                        "port": server.server_port,
                        "owned_dir": str(state.owned_dir),
                        "memory_bytes": state.memory_bytes,
                    }
                ),
                flush=True,
            )
            server.serve_forever(poll_interval=0.1)
    finally:
        state.close()


if __name__ == "__main__":
    main()
