# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import stat
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


STATUS_SCRIPT = Path(__file__).resolve().parent / "manual" / "status.sh"
CAPE_ROOT = STATUS_SCRIPT.parents[3]


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        payload = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *_args):
        pass


class ManualStatusScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmp_path = Path(self.tmp.name)
        self.fake_cape = self.tmp_path / "cape"
        self.fake_cape.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "print(json.dumps(sys.argv[1:]))\n",
            encoding="utf-8",
        )
        self.fake_cape.chmod(self.fake_cape.stat().st_mode | stat.S_IXUSR)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("CAPE_") and key not in ("REPO_ROOT", "BUNDLE")
        }
        self.env["CAPE_STATUS_BIN"] = str(self.fake_cape)

    def run_status(self, *arguments, env=None, check=True):
        return subprocess.run(
            [str(STATUS_SCRIPT), *arguments],
            env=env or self.env,
            check=check,
            text=True,
            capture_output=True,
        )

    def test_forwards_only_read_only_cape_queries(self):
        url = "http://controller.example:18892"
        status = self.run_status(
            "--controller-url", url,
            "--token", "query-token",
            "status", "req-123",
        )
        self.assertEqual(json.loads(status.stdout), [
            "status", "req-123", "--controller-url", url,
            "--token", "query-token",
        ])

        wait = self.run_status(
            "--controller-url", url,
            "wait", "req-456", "--timeout", "12.5",
        )
        self.assertEqual(json.loads(wait.stdout), [
            "wait", "req-456", "--timeout", "12.5",
            "--controller-url", url,
        ])

        utilization = self.run_status(
            "--controller-url", url,
            "--pool", "pool-a",
            "utilization",
        )
        self.assertEqual(json.loads(utilization.stdout), [
            "utilization", "--pool", "pool-a", "--controller-url", url,
        ])

    def test_status_preserves_cape_human_summary_on_stderr(self):
        self.fake_cape.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "sys.stderr.write('QUEUED (pool_empty): no active node is ready in the pool yet; waiting for a node to register\\n')\n"
            "print(json.dumps({'state': 'QUEUED', 'reason': 'pool_empty', 'created_at': 10.0, 'started_at': None, 'ended_at': None, 'updated_at': 10.0}))\n",
            encoding="utf-8",
        )
        self.fake_cape.chmod(self.fake_cape.stat().st_mode | stat.S_IXUSR)

        result = self.run_status(
            "--controller-url", "http://controller.example:18892",
            "status", "req-queued",
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["reason"], "pool_empty")
        self.assertEqual(payload["created_at"], 10.0)
        self.assertIsNone(payload["started_at"])
        self.assertIsNone(payload["ended_at"])
        self.assertIn("QUEUED (pool_empty)", result.stderr)

    def test_help_explains_readiness_and_nonterminal_queue_reasons(self):
        result = self.run_status("--help")

        self.assertIn("active_nodes", result.stdout)
        self.assertIn("node_states", result.stdout)
        self.assertIn("REGISTERING", result.stdout)
        self.assertIn("QUEUED", result.stdout)
        self.assertIn("not failures", result.stdout)
        self.assertIn("started_at", result.stdout)
        self.assertIn("ended_at", result.stdout)
        self.assertIn("never-run", result.stdout)

    def test_rejects_mutating_commands(self):
        for command in ("run", "submit", "node", "controller", "cancel", "admin"):
            with self.subTest(command=command):
                result = self.run_status(command, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn("read-only inspection only", result.stderr)

    def test_controller_env_paths_do_not_select_foreign_installation(self):
        status_bundle = self.tmp_path / "status-bundle"
        controller_env = self.tmp_path / "controller.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export REPO_ROOT=/foreign/controller/repo\n"
            "export CAPE_PROJECT=/foreign/controller/repo/cape\n"
            "export CAPE_VENV=/foreign/controller/venv/aarch64-python3.12\n"
            "export CAPE_BIN=/foreign/controller/venv/aarch64-python3.12/bin/cape\n"
            "export CAPE_STATUS_BIN=/foreign/controller/bin/cape\n"
            "export CAPE_STATUS_BUNDLE=/foreign/controller/status-results\n"
            "export CAPE_STATUS_PYTHON_312=/foreign/controller/bin/python3.12\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-from-env\n"
            "export CAPE_ADMIN_TOKEN=do-not-display-this-token\n"
            "export BUNDLE=/foreign/controller/results/run-a\n",
            encoding="utf-8",
        )

        env = dict(self.env, CAPE_STATUS_BUNDLE=str(status_bundle))
        result = self.run_status(
            "--controller-env", str(controller_env), "info", env=env,
        )
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(fields["controller_url"], "http://controller.example:18892")
        self.assertEqual(fields["pool"], "pool-from-env")
        self.assertEqual(fields["token"], "set")
        self.assertEqual(fields["cape_project"], str(CAPE_ROOT / "pycape"))
        self.assertEqual(fields["cape_bin"], str(self.fake_cape))
        self.assertEqual(fields["bundle"], str(status_bundle))
        self.assertNotIn("/foreign/controller", result.stdout)
        self.assertNotIn("do-not-display-this-token", result.stdout)

    def test_no_token_overrides_controller_environment_credentials(self):
        controller_env = self.tmp_path / "controller-no-token.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n"
            "export CAPE_ADMIN_TOKEN=admin-token\n",
            encoding="utf-8",
        )
        result = self.run_status(
            "--controller-env", str(controller_env),
            "--no-token",
            "status", "req-123",
        )
        arguments = json.loads(result.stdout)
        self.assertNotIn("--token", arguments)
        self.assertEqual(arguments[:2], ["status", "req-123"])

    def test_passed_handoff_does_not_reuse_an_omitted_stale_admin_token(self):
        controller_env = self.tmp_path / "controller-without-token.env"
        controller_env.write_text(
            "export CAPE_CONTROLLER_ENV_VERSION=3\n"
            "export CAPE_URL=http://controller.example:18892\n"
            "export CAPE_POOL=pool-a\n",
            encoding="utf-8",
        )
        env = dict(self.env, CAPE_ADMIN_TOKEN="stale-admin-token")
        result = self.run_status(
            "--controller-env", str(controller_env), "info", env=env,
        )
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(fields["token"], "unset")

    def test_health_does_not_need_cape_installation_or_token(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        url = "http://127.0.0.1:{}".format(server.server_address[1])
        env = dict(self.env)
        env.pop("CAPE_STATUS_BIN")
        result = self.run_status("--controller-url", url, "health", env=env)
        self.assertEqual(json.loads(result.stdout), {"ok": True})
        self.assertEqual(result.stderr, "")

    def test_default_rscape_info_skips_status_venv_and_records_version(self):
        binary_dir = self.tmp_path / "rscape-bin"
        binary_dir.mkdir()
        rscape = binary_dir / "rscape"
        rscape.write_text(
            "#!/usr/bin/env bash\n"
            "[ \"${1:-}\" = --version ] && "
            "{ printf 'rscape 0.1.0\\n'; exit 0; }\n"
            "exit 2\n",
            encoding="utf-8",
        )
        rscape.chmod(rscape.stat().st_mode | stat.S_IXUSR)
        venv_root = self.tmp_path / "must-not-create-status-venv"
        bundle = self.tmp_path / "default-rscape-evidence"
        env = dict(self.env)
        env.pop("CAPE_STATUS_BIN")
        env.update({
            "PATH": str(binary_dir) + os.pathsep + env.get("PATH", ""),
            "CAPE_STATUS_VENV_ROOT": str(venv_root),
            "CAPE_STATUS_VENV": str(venv_root / "status"),
            "CAPE_STATUS_BUNDLE": str(bundle),
        })

        result = self.run_status("info", env=env)
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines())

        self.assertEqual(fields["cape_implementation"], "rscape")
        self.assertEqual(fields["cape_version"], "0.1.0")
        self.assertEqual(fields["cape_bin"], str(rscape))
        self.assertFalse(venv_root.exists())
        self.assertTrue((bundle / "status-client-implementation.txt").is_file())

    def test_tails_controller_and_node_logs(self):
        controller_log = self.tmp_path / "controller.log"
        controller_log.write_text("controller-old\ncontroller-new\n", encoding="utf-8")
        controller_env = dict(self.env, CAPE_STATUS_CONTROLLER_LOG=str(controller_log))
        result = self.run_status("controller-log", "--lines", "1", env=controller_env)
        self.assertEqual(result.stdout, "controller-new\n")

        node_log = self.tmp_path / "agent.log"
        node_log.write_text("node-old\nnode-new\n", encoding="utf-8")
        node_env = dict(self.env, CAPE_STATUS_NODE_LOG=str(node_log))
        result = self.run_status("node-log", "all", "--lines", "1", env=node_env)
        self.assertEqual(result.stdout, "node-new\n")

    def test_tails_exact_runtime_command_and_session_lifecycle_logs(self):
        bundle = self.tmp_path / "bundle"
        log_dir = bundle / "pool" / "node-a" / "agent-logs"
        log_dir.mkdir(parents=True)
        (log_dir / "cmd-000001.log").write_text(
            "exec: bwrap -- true\ncommand-body\n", encoding="utf-8"
        )
        (log_dir / "sess-000001.log").write_text(
            "phase: session-stop\nsession-body\n", encoding="utf-8"
        )
        env = dict(self.env, CAPE_STATUS_BUNDLE=str(bundle))

        command = self.run_status(
            "command-log", "cmd-000001", "node-a", "--lines", "1", env=env
        )
        session = self.run_status(
            "session-log", "sess-000001", "node-a", "--lines", "1", env=env
        )

        self.assertEqual(command.stdout, "command-body\n")
        self.assertEqual(session.stdout, "session-body\n")

    def test_exact_agent_log_directory_is_node_local_and_validates_ids(self):
        log_dir = self.tmp_path / "agent-logs"
        log_dir.mkdir()
        (log_dir / "cmd-safe.log").write_text("safe\n", encoding="utf-8")
        env = dict(self.env, CAPE_STATUS_AGENT_LOG_DIR=str(log_dir))

        result = self.run_status("command-log", "cmd-safe", env=env)
        self.assertEqual(result.stdout, "safe\n")
        rejected = self.run_status(
            "command-log", "cmd-../secret", env=env, check=False
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("unsupported characters", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
