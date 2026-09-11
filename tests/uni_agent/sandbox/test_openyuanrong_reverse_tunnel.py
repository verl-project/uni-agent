"""Live reverse-tunnel check through ``uni_agent.sandbox.openyuanrong``.

Creates a real :class:`OpenyuanrongSandbox` (so ``start`` / ``get_tunnel_url`` /
``exec`` / ``stop`` in ``openyuanrong.py`` all run) and probes a local HTTP
server via the sandbox-side tunnel.

CI never collects this: ``.github/workflows/ci.yml`` runs
``pytest tests/uni_agent/ -m='cpu and level0'``. Manual run::

    OPENYUANRONG_SERVER_ADDRESS=... OPENYUANRONG_TOKEN=... PYTHONPATH=. \\
      pytest tests/uni_agent/sandbox/test_openyuanrong_reverse_tunnel.py -m level1
"""

from __future__ import annotations

import asyncio
import http.server
import importlib.util
import os
import shlex
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

_HAS_OPENYUANRONG_SANDBOX = importlib.util.find_spec("openyuanrong_sandbox") is not None
_HAS_OPENYUANRONG_BACKEND = bool(os.getenv("OPENYUANRONG_SERVER_ADDRESS") and os.getenv("OPENYUANRONG_TOKEN"))

_PROXY_PORT = 38197
_PROBE_ATTEMPTS = 5
_PROBE_RETRY_DELAY = 2.0
_MARKER = f"uni-agent-openyuanrong-tunnel-{uuid.uuid4().hex[:8]}"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.level1,
    pytest.mark.skipif(
        not _HAS_OPENYUANRONG_BACKEND,
        reason="manual: set OPENYUANRONG_SERVER_ADDRESS and OPENYUANRONG_TOKEN against a live cluster",
    ),
    pytest.mark.skipif(not _HAS_OPENYUANRONG_SANDBOX, reason="manual: pip install openyuanrong-sandbox"),
]


@contextmanager
def _local_http_server() -> Iterator[tuple[int, str]]:
    """Serve a unique health body on an ephemeral 127.0.0.1 port (the tunnel upstream)."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib signature
            body = _MARKER.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, name="openyuanrong-tunnel-http", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2) as sock:
                    sock.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                    data = sock.recv(1024)
                if b"200" in data and _MARKER.encode() in data:
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"local reverse-tunnel probe server did not start on port {port}")
        yield port, _MARKER
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _fetch_cmd(url: str) -> str:
    """Fetch ``url`` with the sandbox's Python so the image need not ship curl."""
    script = f"import urllib.request; print(urllib.request.urlopen({url!r}, timeout=30).read().decode())"
    return f"python3 -c {shlex.quote(script)}"


def test_openyuanrong_reverse_tunnel_reaches_local_http_server():
    """Drive ``OpenyuanrongSandbox`` against a live cluster and fetch through the tunnel."""
    # Import inside the test so collection does not require ``uni_agent`` on sys.path
    # (CI still deselects this case via ``-m='cpu and level0'``).
    from uni_agent.sandbox.openyuanrong import OpenyuanrongSandbox

    image = os.getenv("OPENYUANRONG_TEST_IMAGE", "swr.cn-north-4.myhuaweicloud.com/openyuanrong/python:3.12-slim")
    with _local_http_server() as (local_port, marker):
        sandbox = OpenyuanrongSandbox(
            image=image,
            cpu=1000,
            memory=2048,
            idle_timeout=600,
            upstream=f"127.0.0.1:{local_port}",
            proxy_port=_PROXY_PORT,
        )

        async def _run() -> None:
            async with sandbox:
                assert await sandbox.is_alive()
                tunnel_url = sandbox.get_tunnel_url()
                assert f"127.0.0.1:{_PROXY_PORT}" in tunnel_url

                last_error = ""
                for attempt in range(1, _PROBE_ATTEMPTS + 1):
                    result = await sandbox.exec_shell(_fetch_cmd(f"{tunnel_url}/health"), timeout=60)
                    if result.exit_code == 0 and marker in result.stdout:
                        return
                    last_error = (
                        f"attempt {attempt}/{_PROBE_ATTEMPTS}: "
                        f"exit={result.exit_code} stdout={result.stdout!r} stderr={result.stderr!r}"
                    )
                    if attempt < _PROBE_ATTEMPTS:
                        await asyncio.sleep(_PROBE_RETRY_DELAY)
                raise AssertionError(f"sandbox could not reach the local reverse-tunnel server: {last_error}")

        asyncio.run(_run())
