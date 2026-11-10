# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pycape.workspace import WorkspaceConfig
from tests.managed_fixture import EnrollmentHandlerMixin, simulated_gpu_environment


PYCAPE_BIN = os.environ.get("PYCAPE_BIN", "pycape")
RSCAPE_BIN = os.environ.get("RSCAPE_BIN", "rscape")


class RetryHandler(EnrollmentHandlerMixin, BaseHTTPRequestHandler):
    identity_pool = "retry-pool"
    registrations = 0
    heartbeats = 0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.enrollment_response(payload):
            return
        if self.path == "/v1/agent/register":
            type(self).registrations += 1
            self.reply(200, {})
        elif self.path == "/v1/agent/announce":
            self.reply(200, {"workspace_config": WorkspaceConfig(image_size_bytes=64 * 1024 ** 2).to_dict()})
        elif self.path == "/v1/agent/heartbeat":
            if payload.get("state") == "REGISTERING":
                self.reply(200, {"commands": []})
                return
            type(self).heartbeats += 1
            if self.heartbeats == 1:
                self.reply(500, {"error": "temporary"})
            elif self.heartbeats == 2:
                self.reply(404, {"error": "controller restarted"})
            else:
                self.reply(
                    200,
                    {
                        "commands": [],
                        "agent_control": {
                            "authoritative": True,
                            "action": "shutdown",
                            "reason": "allocation_removed",
                            "node_id": "retry-node",
                            "allocation_id": "retry-allocation",
                        },
                    },
                )
        else:
            self.reply(200, {})

    def reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, _format, *_args):
        return


class AgentRetryParityTests(unittest.TestCase):
    def scenario(self, binary):
        RetryHandler.registrations = 0
        RetryHandler.heartbeats = 0
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        server = ThreadingHTTPServer(("127.0.0.1", port), RetryHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run(
                    [
                        binary,
                        "node",
                        "enroll", "--token", "fixture-personal",
                        "--node-token-out", str(Path(temporary) / "agent.json"),
                        "--controller-url",
                        "http://127.0.0.1:{}".format(port),
                        "--node-id",
                        "retry-node",
                        "--pool",
                        "retry-pool",
                        "--allocation-id",
                        "retry-allocation",
                        "--runtime-adapter",
                        "local-process",
                        "--heartbeat-interval",
                        ".01",
                        "--log-dir",
                        temporary,
                        "--state-path",
                        str(Path(temporary) / "state.json"),
                    ],
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertGreaterEqual(RetryHandler.registrations, 2)
        self.assertGreaterEqual(RetryHandler.heartbeats, 3)

    def test_pycape_retries_and_reregisters(self):
        self.scenario(PYCAPE_BIN)

    def test_rscape_retries_and_reregisters(self):
        self.scenario(RSCAPE_BIN)


if __name__ == "__main__":
    unittest.main()
