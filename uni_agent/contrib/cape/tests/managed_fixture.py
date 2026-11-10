# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent CLI fixture; no alternate server or legacy command translation."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import sys

from pycape.credential_file import read_registration


def simulated_gpu_environment(directory, gpu_uuids):
    """Protocol-only CPU tests: simulate discovery, never claim real GPU execution."""
    root = Path(directory) / "simulated-gpu-bin"
    root.mkdir(mode=0o700, exist_ok=True)
    binary = root / "nvidia-smi"
    binary.write_text(
        "#!" + sys.executable + "\nimport sys\n"
        "known = " + repr(list(gpu_uuids)) + "\n"
        "if '--query-gpu=uuid' not in sys.argv: raise SystemExit(1)\n"
        "selected = sys.argv[sys.argv.index('-i')+1].split(',') if '-i' in sys.argv else known\n"
        "if any(g not in known for g in selected): raise SystemExit(1)\n"
        "print('\\n'.join(selected))\n"
    )
    binary.chmod(0o700)
    return dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"])


class EnrollmentHandlerMixin:
    """Identity handshake for fault-injection HTTP doubles, not real controllers."""
    identity_pool = "pool"

    def identity_reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer fixture-personal":
            self.identity_reply(403, {"error": "fixture personal credential required"})
        elif self.path == "/v1/identity/me":
            self.identity_reply(200, {"user_id": "fixture-owner"})
        elif self.path == "/v1/identity/pools":
            self.identity_reply(200, [{"pool_id": "fixture-pool", "name": self.identity_pool,
                                      "config": {"pool_mode": "exclusive", "gpu_slots": 1, "placement": "pack"}}])
        else:
            self.identity_reply(404, {"error": "unknown fixture endpoint"})

    def enrollment_response(self, payload):
        if self.path != "/v1/identity/nodes/enroll":
            return False
        if (self.headers.get("Authorization") != "Bearer fixture-personal"
                or payload.get("owner_user_id") != "fixture-owner" or payload.get("pool") != "fixture-pool"):
            self.identity_reply(403, {"error": "invalid fixture enrollment"})
        else:
            self.identity_reply(200, {"operation_id": payload["operation_id"], "status": "enrolled"})
        return True


def command(binary, *args):
    env = dict(os.environ)
    for key in ("CAPE_TOKEN", "CAPE_ADMIN_TOKEN", "CAPE_JOIN_TOKEN"):
        env.pop(key, None)
    result = subprocess.run([binary, *map(str, args)], env=env, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError("fixture CLI failed: " + result.stderr)
    return json.loads(result.stdout)


class ManagedFixture:
    def __init__(self, binary, directory):
        self.binary = binary
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state = self.root / "controller-state"
        self.secrets = self.root / "secrets"
        self.secrets.mkdir(mode=0o700, exist_ok=True)
        self.admin_file = self.secrets / "admin.json"
        if not self.state.exists():
            command(binary, "controller", "init", "--state-dir", self.state,
                    "--user", "admin", "--token-out", self.admin_file)
        self.admin = read_registration(self.admin_file)["secret"]
        self.url = None

    def start(self, binary=None, *extra, output=None):
        if self.url:
            port = int(self.url.rsplit(":", 1)[1])
        else:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        self.url = "http://127.0.0.1:{}".format(port)
        process = subprocess.Popen([
            binary or self.binary, "controller", "start", "--state-dir", str(self.state),
            "--token-file", str(self.admin_file), "--host", "127.0.0.1", "--port", str(port),
            *extra,
        ], stdout=subprocess.PIPE if output is None else output,
           stderr=subprocess.PIPE if output is None else output, text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError("controller exited: " + (process.communicate()[1] or "see redirected controller log"))
                try:
                    self.request("GET", "/health")
                    return process
                except OSError:
                    time.sleep(0.02)
            raise AssertionError("controller health timeout")
        except BaseException:
            process.terminate()
            process.communicate(timeout=10)
            raise

    def request(self, method, path, body=None, token=None):
        request = urllib.request.Request(
            self.url + path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + (token or self.admin)},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def start_embedded(self, schedule_interval=0.05):
        """Expose the same authority to unit tests that inspect controller internals."""
        from pycape.controller import CAPEController
        from pycape.instance import Instance
        from pycape.journal import SQLiteJournal
        from pycape.managed_auth import ManagedAuthority
        from pycape.service import CAPEControllerService

        self.instance = Instance.open(str(self.state), self.admin)
        self.journal = SQLiteJournal(self.instance.database)
        self.controller = CAPEController.reconcile_from_journal(self.journal, auth=ManagedAuthority(self.instance.store))
        self.service = CAPEControllerService(self.controller, schedule_interval=schedule_interval).start()
        self.url = self.service.url
        return self.service

    def close_embedded(self):
        self.service.stop()
        self.journal.close()
        self.instance.close()

    def pool(self, name, mode="exclusive", slots=1, placement="pack"):
        pools = self.request("GET", "/v1/identity/pools")
        existing = next((p for p in pools if name in (p["name"], p["pool_id"])), None)
        if existing is not None:
            return existing["pool_id"]
        return command(self.binary, "admin", "pool", "create", name,
                       "--pool-mode", mode, "--gpu-slots", slots, "--placement", placement,
                       "--controller-url", self.url, "--token-file", self.admin_file)["pool_id"]

    def user_file(self, name):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in name):
            raise ValueError("invalid fixture username")
        path = self.secrets / ("user-" + name + ".json")
        if not path.exists():
            command(self.binary, "user", "token", "create", "--user", name,
                    "--token-out", path, "--controller-url", self.url)
        return path

    def user_token(self, name):
        return read_registration(self.user_file(name))["secret"]

    def node_credentials(self, node, pool, owner="contributor"):
        self.pool(pool)
        saved = self.secrets / ("node-" + node + ".json")
        args = ["--token-file", str(self.user_file(owner)), "--node-token-out", str(saved)]
        if saved.exists():
            args.append("--resume")
        return args
