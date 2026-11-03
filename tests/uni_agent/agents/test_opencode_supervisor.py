"""Real pipes and child processes, without requiring an OpenCode install."""

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.level0]
PATH = Path(__file__).resolve().parents[3] / "examples/sidecar/opencode/standalone.py"
spec = importlib.util.spec_from_file_location("opencode_supervisor", PATH)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


@pytest.mark.parametrize("payload", [b"", b'{"url":', b"x" * 4097, b"bad\n"])
def test_invalid_readiness_is_bounded(payload):
    reader, writer = os.pipe()
    with os.fdopen(reader, "rb", buffering=0) as stream:
        try:
            os.write(writer, payload)
            started = time.monotonic()
            if payload == b"bad\n":
                assert supervisor.read_ready(stream, timeout=0.1) == payload
            else:
                with pytest.raises(RuntimeError):
                    supervisor.read_ready(stream, timeout=0.1)
            assert time.monotonic() - started < 2
        finally:
            os.close(writer)


def test_fragmented_readiness_and_eof():
    reader, writer = os.pipe()
    with os.fdopen(reader, "rb", buffering=0) as stream:

        def send():
            for part in [b'{"url":', b'"http://127.0.0.1:1"}', b"\n"]:
                os.write(writer, part)
                time.sleep(0.01)
            os.close(writer)

        thread = threading.Thread(target=send)
        thread.start()
        try:
            assert supervisor.read_ready(stream, timeout=2) == b'{"url":"http://127.0.0.1:1"}\n'
            with pytest.raises(RuntimeError, match="exited"):
                supervisor.read_ready(stream, timeout=2)
        finally:
            thread.join()


def test_stdout_drain_uses_bounded_reads_without_newlines(monkeypatch):
    reader, writer = os.pipe()
    stop = threading.Event()
    sizes = []
    original = os.read

    def read(fd, size):
        sizes.append(size)
        return original(fd, size)

    monkeypatch.setattr(supervisor.os, "read", read)
    with os.fdopen(reader, "rb", buffering=0) as stream:
        thread = threading.Thread(target=supervisor.drain_stdout, args=(stream, stop))
        thread.start()
        try:
            # More than pipe capacity: writer completion proves the reader drains
            # continuously even without a newline or EOF.
            for _ in range(256):
                os.write(writer, b"x" * 4096)
        finally:
            os.close(writer)
            thread.join(timeout=3)
            stop.set()
        assert not thread.is_alive()
    assert sizes and max(sizes) <= 65536


def test_partial_startup_failure_reaps_server_and_does_not_start_client(tmp_path):
    fake = tmp_path / "server.py"
    marker = tmp_path / "client"
    fake.write_text(
        "import sys\nfrom pathlib import Path\n"
        "if 'serve' in sys.argv:\n"
        " print('{', end='', flush=True)\n sys.stdin.read()\n"
        f"else: Path({str(marker)!r}).touch()\n"
    )
    script = (
        "import importlib.util,sys\n"
        f"s=importlib.util.spec_from_file_location('supervisor',{str(PATH)!r})\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        f"m.COMMAND=[sys.executable,{str(fake)!r}]\n"
        "read=m.read_ready;m.read_ready=lambda stream:read(stream,timeout=0.1)\n"
        "sys.argv=['supervisor','run','--standalone','--','prompt']\n"
        "raise SystemExit(m.main())\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "RuntimeError" in result.stderr
    assert not marker.exists()
