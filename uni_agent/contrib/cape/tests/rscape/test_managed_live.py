# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real FUSE/GPU acceptance for persistent identities, restart and node owners."""

import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

from pycape.cli import _request, CapeHTTPError
from pycape.credential_file import read_registration
from pycape.workspace_fs import mount_entries


@unittest.skipUnless(os.environ.get("CAPE_MANAGED_LIVE_TEST") == "1", "requires authorized GPU and real FUSE")
class ManagedLiveTests(unittest.TestCase):
    def command(self, binary, *args, raw=False, timeout=90):
        result = subprocess.run([binary, *args], text=True, capture_output=True, timeout=timeout, env=self.environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip() if raw else json.loads(result.stdout)

    def api(self, method, path, token=None, body=None):
        return _request(method, self.url, path, token=token, payload=body, timeout=30)

    def poll(self, predicate, timeout=45):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(0.05)
        self.fail("managed live condition timed out; evidence=" + str(self.output))

    def start_controller(self, binary):
        stream = open(self.output / "controller.log", "a")
        self.streams.append(stream)
        process = subprocess.Popen([binary, "controller", "start", "--state-dir", str(self.root / "state"),
                                    "--token-file", str(self.admin_file), "--port", str(self.port)],
                                   env=self.environment, stdout=stream, stderr=stream)
        self.controllers.append(process)

        def ready():
            self.assertIsNone(process.poll(), (self.output / "controller.log").read_text())
            try:
                return self.api("GET", "/health")["ok"]
            except Exception:
                return False

        self.poll(ready, 30)
        return process

    def start_node(self, binary, node_id, owner_file):
        stream = open(self.output / (node_id + ".log"), "w")
        self.streams.append(stream)
        process = subprocess.Popen([binary, "node", "enroll", "--controller-url", self.url,
                                    "--pool", "pool", "--node-id", node_id, "--token-file", str(owner_file),
                                    "--node-token-out", str(self.secrets / (node_id + ".json")),
                                    "--cpu-cores", "2", "--memory-gb", "4", "--gpu-uuid", self.gpu,
                                    "--runtime-adapter", "local-process", "--heartbeat-interval", "0.05",
                                    "--state-path", str(self.secrets / (node_id + "-state.json")),
                                    "--log-dir", str(self.root / (node_id + "-logs")), "--runtime-dir", str(self.root / (node_id + "-runtime"))],
                                   env=self.environment, stdout=stream, stderr=stream)
        self.nodes.append(process)

        def ready():
            self.assertIsNone(process.poll(), (self.output / (node_id + ".log")).read_text())
            inventory = self.api("GET", "/v1/identity/nodes", self.admin)
            return any(n["node_id"] == node_id and n["state"] == "ACTIVE" for n in inventory)

        self.poll(ready, 180)
        return process

    def submit(self, client, file, task, program, wait=True, workspace=None, timeout=30):
        args = ["run", "--controller-url", self.url, "--token-file", str(file), "--pool", "pool", "--session-id", task,
                "--gpus", "1", "--runtime-adapter", "local-process", "--isolation-policy", "none",
                "--command-timeout-seconds", str(timeout)]
        if wait:
            args += ["--wait", "--timeout", "60"]
        if workspace:
            args += ["--workspace-id", workspace]
        args += ["--", sys.executable, "-c", program]
        return self.command(client, *args, raw=not wait)

    def capture_failure(self):
        diagnostic = {"root": str(self.root), "agent_state": {}}
        try:
            with sqlite3.connect((self.root / "state/state.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
                state = json.loads(db.execute("SELECT body FROM authority WHERE singleton=1").fetchone()[0])
                diagnostic["nodes"] = {key: {field: node.get(field) for field in ("desired", "announced", "intent_at", "effective_resources", "force")}
                                       for key, node in state.get("nodes", {}).items()}
                diagnostic["events"] = [list(row) for row in db.execute("SELECT timestamp,event_type,entity_id,state,reason FROM events ORDER BY rowid DESC LIMIT 60")]
                latest = {}
                for event, entity, details in db.execute("SELECT event_type,entity_id,details FROM events ORDER BY rowid"):
                    value = json.loads(details)
                    latest[event + ":" + entity] = {key: value.get(key) for key in
                        ("state", "node_id", "execution_session_id", "active_sessions", "active_request_id", "revision", "last_error") if key in value}
                diagnostic["latest"] = latest
            for path in self.secrets.glob("*-state.json"):
                state = json.loads(path.read_text())
                diagnostic["agent_state"][path.name] = {key: state.get(key) for key in ("local_sessions", "running_commands", "closed_workspace_sessions")}
        except Exception as error:
            diagnostic["capture_error"] = type(error).__name__
        (self.output / "failure-state.json").write_text(json.dumps(diagnostic, sort_keys=True) + "\n")

    def test_owner_restart_force_and_workspace_four_way(self):
        python = os.environ["PYCAPE_BIN"]
        rust = os.environ["RSCAPE_BIN"]
        output = Path(os.environ["CAPE_MANAGED_LIVE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        self.gpu = os.environ["CAPE_MANAGED_GPU_UUID"]
        for index, (controller, agent, client) in enumerate(((python, python, rust), (python, rust, python),
                                                            (rust, python, rust), (rust, rust, python))):
            with self.subTest(controller=controller, agent=agent, client=client):
                self.output = output / ("cell-" + str(index))
                self.output.mkdir()
                self.root = Path(tempfile.mkdtemp(prefix="cape-managed-live-"))
                self.secrets = self.root / "secrets"
                self.secrets.mkdir(mode=0o700)
                self.controllers, self.nodes, self.streams = [], [], []
                self.environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
                for key in ("CAPE_TOKEN", "CAPE_ADMIN_TOKEN", "CAPE_JOIN_TOKEN", "CAPE_PROBE_NODE_TOKEN",
                            "CAPE_USER_TOKENS_INLINE", "CAPE_SUBMIT_TOKEN", "CAPE_STATUS_TOKEN", "CAPE_RETIRE_TOKEN"):
                    self.environment.pop(key, None)
                passed = False
                try:
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        self.port = sock.getsockname()[1]
                    self.url = "http://127.0.0.1:" + str(self.port)
                    self.admin_file = self.secrets / "admin.json"
                    initial = self.command(client, "controller", "init", "--state-dir", str(self.root / "state"), "--user", "admin", "--token-out", str(self.admin_file))
                    self.admin = read_registration(self.admin_file)["secret"]
                    server = self.start_controller(controller)
                    self.command(client, "admin", "pool", "create", "pool", "--max-nodes", "1", "--controller-url", self.url, "--token-file", str(self.admin_file))
                    self.api("POST", "/v1/admin/workspace-config", self.admin,
                             {"pool": "pool", "changes": {"image_root": str(self.root / "images"), "image_size_bytes": 128 * 1024 ** 2}})
                    alice_file, second_file, bob_file = (self.secrets / name for name in ("alice.json", "alice-second.json", "bob.json"))
                    for user, file in (("alice", alice_file), ("bob", bob_file)):
                        self.command(client, "user", "token", "create", "--controller-url", self.url, "--user", user, "--token-out", str(file))
                    alice = read_registration(alice_file)["secret"]
                    bob = read_registration(bob_file)["secret"]
                    node = self.start_node(agent, "node-a", alice_file)
                    program = "import os,pathlib,torch; assert 'CAPE_TOKEN' not in os.environ and 'CAPE_ADMIN_TOKEN' not in os.environ; assert torch.cuda.device_count()==1; torch.ones(16,device='cuda').sum().item(); p=pathlib.Path(os.environ['CAPE_WORKSPACE_DIR'])/'counter'; n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); print(n+1)"
                    first = self.submit(client, alice_file, "warm", program)
                    second = self.submit(client, alice_file, "warm", program)
                    self.assertEqual((first["state"], second["state"]), ("COMPLETED", "COMPLETED"))
                    self.assertEqual(first["execution_session_id"], second["execution_session_id"])
                    workspace = first["workspace"]["id"]
                    with self.assertRaises(CapeHTTPError):
                        self.api("GET", "/v1/workspaces/" + workspace, bob)
                    self.command(client, "user", "token", "create", "--controller-url", self.url, "--token-file", str(alice_file), "--token-out", str(second_file))
                    old_id = self.command(client, "user", "whoami", "--controller-url", self.url, "--token-file", str(alice_file))["token_id"]
                    self.command(client, "user", "token", "revoke", old_id, "--controller-url", self.url, "--token-file", str(second_file))
                    alice_second = read_registration(second_file)["secret"]
                    self.command(client, "node", "edit", "node-a", "--cpu-cores", "1", "--controller-url", self.url, "--token-file", str(second_file))
                    self.poll(lambda: any(n["node_id"] == "node-a" and n["state"] == "ACTIVE" and n["resources"]["cpu_cores"] == 1
                                          for n in self.api("GET", "/v1/identity/nodes", alice_second)))
                    reopened = self.submit(client, second_file, "after-edit", program, workspace=workspace)
                    self.assertEqual(reopened["workspace"]["id"], workspace)
                    self.assertEqual(self.api("GET", "/v1/requests/" + reopened["request_id"] + "/logs", alice_second)["stdout"].strip(), "3")
                    marker = self.root / "once.txt"
                    long_program = "import os,pathlib,time,torch; x=torch.ones(1024,device='cuda'); x.sum().item(); p=pathlib.Path(" + repr(str(marker)) + "); f=p.open('a'); f.write('once\\n'); f.flush(); os.fsync(f.fileno()); f.close(); time.sleep(8)"
                    request_id = self.submit(client, bob_file, "restart", long_program, wait=False)
                    self.poll(marker.exists)
                    with self.assertRaises(CapeHTTPError):
                        self.api("GET", "/v1/requests/" + request_id, alice_second)
                    self.command(client, "node", "drain", "node-a", "--controller-url", self.url, "--token-file", str(second_file))
                    server.kill()
                    server.wait(timeout=5)
                    self.start_controller(rust if controller == python else python)
                    self.assertEqual(self.api("GET", "/v1/identity/me", bob)["scheduler_id"], initial["scheduler_id"])
                    done = self.poll(lambda: (lambda r: r if r["state"] in ("COMPLETED", "FAILED", "CANCELLED") else None)(self.api("GET", "/v1/requests/" + request_id, bob)))
                    self.assertEqual(done["state"], "COMPLETED", done)
                    self.assertEqual(marker.read_text().splitlines(), ["once"])
                    self.assertEqual(node.wait(timeout=30), 0, (self.output / "node-a.log").read_text())
                    self.assertEqual(self.api("GET", "/v1/workspaces/" + workspace, alice_second)["state"], "retained")
                    forced_node = self.start_node(agent, "node-b", second_file)
                    force_marker = self.root / "force.txt"
                    force_program = "import pathlib,time,torch; x=torch.ones(1024,device='cuda'); x.sum().item(); pathlib.Path(" + repr(str(force_marker)) + ").write_text('running'); time.sleep(60)"
                    forced_id = self.submit(client, bob_file, "forced", force_program, wait=False, timeout=20)
                    self.poll(force_marker.exists)
                    started = time.monotonic()
                    result = self.command(client, "node", "drain", "node-b", "--force", "--controller-url", self.url, "--token-file", str(second_file))
                    self.assertEqual(set(result), {"node_id", "state"})
                    self.assertEqual(forced_node.wait(timeout=15), 0, (self.output / "node-b.log").read_text())
                    self.assertLess(time.monotonic() - started, 10)
                    self.assertEqual(self.api("GET", "/v1/requests/" + forced_id, bob)["state"], "CANCELLED")
                    self.assertFalse([entry for entry in mount_entries() if entry["target"].startswith(str(self.root) + "/")])
                    images = list((self.root / "images").rglob("workspace.img"))
                    self.assertTrue(images)
                    self.command(client, "admin", "pool", "remove", "pool", "--controller-url", self.url, "--token-file", str(self.admin_file))
                    self.assertTrue(all(image.is_file() for image in images))
                    passed = True
                    print(json.dumps({"cell": index, "controller": controller, "agent": agent, "client": client,
                                      "gpu_uuid": self.gpu, "restart_implementation": rust if controller == python else python,
                                      "exactly_once_marker": True, "workspace_retained": True, "force_cleanup": True,
                                      "gpu_enforcement": "local-process advisory", "result": "passed"}), flush=True)
                finally:
                    if not passed:
                        self.capture_failure()
                    for process in self.nodes + self.controllers:
                        if process.poll() is None:
                            process.send_signal(signal.SIGINT)
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                    for stream in self.streams:
                        stream.close()
                    remaining = [entry for entry in mount_entries() if entry["target"].startswith(str(self.root) + "/")]
                    if passed and not remaining:
                        shutil.rmtree(self.root)
                    else:
                        print(json.dumps({"preserved_fixture": str(self.root), "remaining_mounts": remaining}), flush=True)
