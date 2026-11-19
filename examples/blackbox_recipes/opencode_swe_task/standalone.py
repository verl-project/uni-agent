"""Keep v2's private-server lease when an explicit musl loader changes execPath.

The task image supplies /usr/bin/python3. Both CLI processes still run through
the bundled musl loader; their tools inherit the caller's original environment.
"""

from __future__ import annotations

import json
import os
import secrets
import selectors
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

COMMAND = [
    "/opt/opencode/lib/ld.so",
    "--library-path",
    "/opt/opencode/lib",
    "/opt/opencode/libexec/opencode",
]


def read_ready(stream, timeout=30):
    """Read one bounded record under a deadline covering partial writes too."""
    deadline = time.monotonic() + timeout
    record = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(timeout=remaining):
                raise RuntimeError("private server readiness timed out")
            chunk = os.read(stream.fileno(), 4097 - len(record))
            if not chunk:
                raise RuntimeError("private server exited without valid readiness")
            record.extend(chunk)
            newline = record.find(b"\n")
            if 0 <= newline < 4096:
                return bytes(record[: newline + 1])
            if len(record) > 4096:
                raise RuntimeError("private server readiness exceeds limit")


def drain_stdout(stream, stop):
    """Discard output in fixed chunks; neither missing newlines nor EOF is required."""
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while not stop.is_set():
            if selector.select(timeout=0.1) and not os.read(stream.fileno(), 65536):
                return


def main() -> int:
    # Only options before -- belong to the CLI; prompt text is opaque.
    args = sys.argv[1:]
    boundary = args.index("--") if "--" in args else len(args)
    options, prompt = args[:boundary], args[boundary:]
    if "--standalone" not in options:
        os.execv(COMMAND[0], [*COMMAND, *args])
    if any(arg == "--server" or arg.startswith("--server=") for arg in options):
        print("--server and --standalone cannot be combined", file=sys.stderr)
        return 2
    options = [arg for arg in options if arg != "--standalone"]
    env = {**os.environ, "OPENCODE_PASSWORD": secrets.token_urlsafe(32)}
    # The private server treats stdin EOF as loss of its ownership lease. If
    # this supervisor is killed, the OS closes the pipe even without finally.
    server = subprocess.Popen(
        [*COMMAND, "serve", "--stdio", "--port", "0"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None if os.environ.get("OPENCODE_PRINT_LOGS") == "1" else subprocess.DEVNULL,
        start_new_session=True,
    )
    client = None
    drain = None
    stop_drain = threading.Event()
    try:
        line = read_ready(server.stdout)
        url = json.loads(line)["url"]
        address = urlsplit(url)
        if address.scheme != "http" or address.hostname != "127.0.0.1" or not address.port:
            raise RuntimeError("private server did not bind loopback")
        # Server stdout is only a readiness protocol in stdio mode. Drain any
        # later output so the server cannot block on a full pipe.
        drain = threading.Thread(target=drain_stdout, args=(server.stdout, stop_drain), daemon=True)
        drain.start()
        client = subprocess.Popen([*COMMAND, *options, "--server", url, *prompt], env=env)
        return client.wait()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        # Do not include environment, auth or server output in diagnostics.
        print(f"OpenCode private server startup failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if client is not None and client.poll() is None:
            client.terminate()
            try:
                client.wait(timeout=3)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait()
        server.stdin.close()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stop_drain.set()
        if drain is not None:
            drain.join()
        server.stdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
