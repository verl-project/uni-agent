# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recording CLI/health double for shell bootstrap tests; never used by CAPE."""
import json
import os
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
if args == ["--version"]:
    print("pycape 0.1.0")
    raise SystemExit(0)
with open(os.environ["FAKE_CAPE_CALLS"], "a") as stream:
    stream.write(" ".join(args) + "\n")


def option(name):
    return args[args.index(name) + 1]


if args[:2] == ["controller", "init"] or args[:3] == ["user", "token", "create"]:
    if args[:2] == ["controller", "init"]:
        Path(option("--state-dir")).mkdir(mode=0o700)
    target = Path(option("--token-out"))
    with target.open("x") as stream:
        target.chmod(0o600)
        json.dump({"secret": "fixture-" + option("--user")}, stream)
    print("{}")
elif args[:2] == ["controller", "start"]:
    previous_edits = Path(os.environ["FAKE_CAPE_CALLS"]).read_text().count("admin pool edit ")
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", int(option("--port"))), Handler)
    server.timeout = 0.05
    count = 1 + len(os.environ.get("CAPE_ADDITIONAL_POOLS", "shared-pool").split())
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            server.handle_request()
            calls = Path(os.environ["FAKE_CAPE_CALLS"]).read_text()
            if calls.count("admin pool edit ") >= previous_edits + count:
                print("fake controller stopped")
                break
        else:
            raise SystemExit("fixture bootstrap timed out")
    finally:
        server.server_close()
elif args[:2] == ["user", "pools"]:
    print("[]")
elif args[:2] == ["admin", "pool"] or args[:3] == ["user", "token", "resume"]:
    print("{}")
else:
    raise SystemExit("unsupported fixture CLI: " + repr(args[:3]))
