# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from tests.managed_fixture import ManagedFixture


PYCAPE_BIN = os.environ.get("PYCAPE_BIN", "pycape")
RSCAPE_BIN = os.environ.get("RSCAPE_BIN", "rscape")


class AgentControllerInteropTests(unittest.TestCase):
    def free_port(self):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def command(self, binary, *args):
        if getattr(self, "credential", None):
            args = (args[0], "--token-file", str(self.credential), *args[1:])
        return subprocess.run(
            [binary, *args], text=True, capture_output=True
        )

    def scenario(self, controller_binary, agent_binary, client_binary):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ManagedFixture(controller_binary, tmp)
            controller = fixture.start()
            url = fixture.url
            agent = None
            try:
                fixture.pool("pool-a")
                self.credential = fixture.user_file("alice")
                for _ in range(200):
                    probe = self.command(
                        client_binary,
                        "utilization",
                        "--controller-url",
                        url,
                        "--pool",
                        "pool-a",
                    )
                    if probe.returncode == 0:
                        break
                    time.sleep(0.01)
                self.assertEqual(probe.returncode, 0, probe.stderr)

                agent = subprocess.Popen(
                    [
                        agent_binary,
                        "node",
                        "enroll", *fixture.node_credentials("interop-node", "pool-a"),
                        "--controller-url",
                        url,
                        "--node-id",
                        "interop-node",
                        "--pool",
                        "pool-a",
                        "--allocation-id",
                        "interop-allocation",
                        "--cpu-cores",
                        "4",
                        "--memory-gb",
                        "8",
                        "--runtime-adapter",
                        "local-process",
                        "--heartbeat-interval",
                        "0.02",
                        "--log-dir",
                        str(Path(tmp) / "logs"),
                        "--state-path",
                        str(Path(tmp) / "state.json"),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(300):
                    probe = self.command(
                        client_binary,
                        "utilization",
                        "--controller-url",
                        url,
                        "--pool",
                        "pool-a",
                    )
                    if probe.returncode == 0 and json.loads(probe.stdout)["active_nodes"] == 1:
                        break
                    time.sleep(0.01)
                self.assertEqual(json.loads(probe.stdout)["active_nodes"], 1)

                failed = self.command(
                    client_binary,
                    "run",
                    "--controller-url",
                    url,
                    "--pool",
                    "pool-a",
                    "--user",
                    "alice",
                    "--session-id",
                    "interop",
                    "--runtime-adapter",
                    "local-process",
                    "--isolation-policy",
                    "none",
                    "--wait",
                    "--timeout",
                    "10",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                )
                state_text = "<not inspected>"
                log_text = "<not inspected>"
                if failed.returncode != 7:
                    state_text = (Path(tmp) / "state.json").read_text() if (Path(tmp) / "state.json").exists() else "<missing>"
                    log_text = "\n".join(
                        path.name + ":" + path.read_text(errors="replace")
                        for path in sorted((Path(tmp) / "logs").glob("*"))
                    ) if (Path(tmp) / "logs").is_dir() else "<missing>"
                    agent.terminate()
                    agent_stdout, agent_stderr = agent.communicate(timeout=5)
                    agent = None
                else:
                    agent_stdout, agent_stderr = "", ""
                self.assertEqual(
                    failed.returncode,
                    7,
                    "stdout={} stderr={} agent_poll={} agent_stdout={} agent_stderr={} state={} logs={}".format(
                        failed.stdout,
                        failed.stderr,
                        None if agent is None else agent.poll(),
                        agent_stdout,
                        agent_stderr,
                        state_text,
                        log_text,
                    ),
                )
                failed_status = json.loads(failed.stdout)
                self.assertEqual(failed_status["failure_kind"], "workload")

                succeeded = self.command(
                    client_binary,
                    "run",
                    "--controller-url",
                    url,
                    "--pool",
                    "pool-a",
                    "--user",
                    "alice",
                    "--session-id",
                    "interop",
                    "--runtime-adapter",
                    "local-process",
                    "--isolation-policy",
                    "none",
                    "--wait",
                    "--timeout",
                    "10",
                    "--",
                    sys.executable,
                    "-c",
                    "print('interop-ok')",
                )
                self.assertEqual(succeeded.returncode, 0, succeeded.stderr)
                succeeded_status = json.loads(succeeded.stdout)
                self.assertEqual(
                    failed_status["execution_session_id"], succeeded_status["execution_session_id"]
                )
            finally:
                if agent is not None:
                    agent.terminate()
                    agent.communicate(timeout=5)
                controller.terminate()
                controller.communicate(timeout=5)

    def test_pycape_agent_runs_on_rscape_controller(self):
        self.scenario(RSCAPE_BIN, PYCAPE_BIN, RSCAPE_BIN)

    def test_rscape_agent_runs_on_pycape_controller(self):
        self.scenario(PYCAPE_BIN, RSCAPE_BIN, PYCAPE_BIN)


if __name__ == "__main__":
    unittest.main()
