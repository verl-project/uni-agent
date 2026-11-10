# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import (
    CAPEController,
    CAPEControllerService,
    EnforcementMode,
    LocalProcessRuntimeAdapter,
    MemoryJournal,
    MountSpec,
    NodeAgentClient,
    NodeResources,
    NodeState,
    ResourceRequest,
    RuntimeCapabilities,
    SQLiteJournal,
    WorkloadSpec,
)
from pycape.cli import cape_main, _build_cape_parser, _build_run_payload, _node_agent_from_args


from tests.managed_fixture import ManagedFixture
from pycape.identity import new_secret, verifier
import uuid

POOL = "pool-a"


class CapeCliTests(unittest.TestCase):
    def setUp(self):
        # The in-memory HTTP fixtures test CLI transport, not personal authority.
        # Production identity behavior is exercised by ManagedFixture parity tests.
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        environment = mock.patch.dict(os.environ, {"CAPE_SESSION_STATE_DIR": os.path.join(cache.name, "sessions")})
        environment.start()
        self.addCleanup(environment.stop)
        original = CAPEControllerService._http_identity_me
        scheduler_id = str(uuid.uuid4())

        async def identify(service, request):
            if getattr(service.controller.auth, "managed", False):
                return await original(service, request)
            from aiohttp import web
            return web.json_response({"scheduler_id": scheduler_id, "user_id": "alice"})

        patcher = mock.patch.object(CAPEControllerService, "_http_identity_me", identify)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_credentials_fall_back_to_environment_but_granted_tokens_do_not(self):
        env = {name: "synthetic-secret" for name in (
            "CAPE_ADMIN_TOKEN", "CAPE_PROBE_NODE_TOKEN", "CAPE_NODE_TOKEN", "CAPE_USER_TOKENS_INLINE",
            "CAPE_SUBMIT_TOKEN", "CAPE_SUBMIT_USER_TOKENS_INLINE", "CAPE_STATUS_TOKEN",
            "CAPE_RETIRE_TOKEN", "CAPE_CONCURRENCY_STATUS_TOKEN",
        )}
        env.update(CAPE_TOKEN="env-credential", CAPE_JOIN_TOKEN="env-join")
        with mock.patch.dict(os.environ, env):
            parser = _build_cape_parser()
            self.assertEqual(parser.parse_args(["status", "req-1"]).token, "env-credential")
            self.assertEqual(parser.parse_args(["status", "req-1", "--token", "flag"]).token, "flag")
            join = parser.parse_args(["node", "enroll", "--node-token-out", "unused-agent.json", "--node-id", "n", "--pool", "p"])
            self.assertEqual((join.token, join.join_token), ("env-credential", None))
            commands = [["workspace", action, "workspace-id"] for action in
                        ("status", "release", "check", "delete", "grow", "reclaim", "shrink", "recover")]
            commands += [["admin", name, "--pool", "p"] for name in
                         ("get-workspace-config", "set-workspace-config")]
            for command in commands:
                with self.subTest(command=command):
                    self.assertEqual(parser.parse_args(command).token, "env-credential")
                    self.assertEqual(parser.parse_args(command + ["--token", "flag"]).token, "flag")
            # The granted token of admin add-* is never taken from the environment.
            with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
                parser.parse_args(["admin", "add-user-token", "--auth-file", "f", "--user", "u", "--pool", "p"])
            # cape_main drops the credentials so workloads started by this process do not inherit them.
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                try:
                    cape_main(["utilization", "--pool", "p", "--controller-url", "http://127.0.0.1:9"])
                except (SystemExit, Exception):
                    pass
            for name in env:
                self.assertNotIn(name, os.environ)

    def test_controller_liveness_defaults_match_protocol(self):
        args = _build_cape_parser().parse_args(["controller", "start", "--state-dir", "state"])
        self.assertEqual(args.heartbeat_timeout_seconds, 30.0)
        self.assertEqual(args.lost_timeout_seconds, 90.0)

    def test_cape_status_prints_queued_reason_summary_and_preserves_json_stdout(self):
        queued = {
            "request_id": "req-queued",
            "state": "QUEUED",
            "reason": "pool_empty",
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("pycape.cli._request", return_value=queued):
            self.assertEqual(
                cape_main(
                    [
                        "status",
                        "req-queued",
                        "--controller-url",
                        "http://controller.example:8765",
                    ],
                    stdout=stdout,
                    stderr=stderr,
                ),
                0,
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["state"], "QUEUED")
        self.assertEqual(payload["reason"], "pool_empty")
        self.assertEqual(
            payload["status_message"],
            "QUEUED (pool_empty): no active node is ready in the pool yet; waiting for a node to register",
        )
        self.assertEqual(stderr.getvalue().strip(), payload["status_message"])

    def test_node_join_announces_registering_before_probe_and_registers_active_once(self):
        journal = MemoryJournal()
        ctrl = CAPEController(journal=journal)
        service = CAPEControllerService(
            controller=ctrl, schedule_interval=10.0
        ).start()
        capabilities = RuntimeCapabilities(
            adapter="local-process",
            image_formats=("default",),
            isolation_policies=("default", "none"),
        )
        runtime = LocalProcessRuntimeAdapter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                def finish_probe(_args):
                    self.assertIsNone(_args.runtime_adapter)
                    node = ctrl.get_node("node-probing")
                    self.assertEqual(node.state, NodeState.REGISTERING)
                    announced_heartbeat = node.last_heartbeat_at
                    deadline = time.time() + 0.5
                    while (
                        ctrl.get_node("node-probing").last_heartbeat_at
                        <= announced_heartbeat
                        and time.time() < deadline
                    ):
                        time.sleep(0.01)
                    node = ctrl.get_node("node-probing")
                    self.assertEqual(node.state, NodeState.REGISTERING)
                    self.assertGreater(
                        node.last_heartbeat_at, announced_heartbeat
                    )
                    return (
                        capabilities,
                        {"local-process": runtime},
                        {"local-process": capabilities},
                    )

                stdout = io.StringIO()
                with mock.patch(
                    "pycape.cli._runtime_config", side_effect=finish_probe
                ), mock.patch("pycape.node_enrollment.enroll", return_value="probe-epoch"):
                    rc = cape_main(
                        [
                            "node",
                            "enroll", "--node-token-out", "unused-agent.json",
                            "--controller-url",
                            service.url,
                            "--node-id",
                            "node-probing",
                            "--pool",
                            POOL,
                            "--allocation-id",
                            "alloc-probing",
                            "--log-dir",
                            tmp,
                            "--state-path",
                            os.path.join(tmp, "agent-state.json"),
                            "--heartbeat-interval",
                            "0.01",
                            "--once",
                        ],
                        stdout=stdout,
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(ctrl.get_node("node-probing").state, NodeState.ACTIVE)
            self.assertTrue(ctrl.get_node("node-probing").agent_epoch)
            node_events = [
                event.event_type
                for event in journal.list_events()
                if event.entity_id == "node-probing"
            ]
            self.assertEqual(node_events.count("node_announced"), 1)
            self.assertEqual(node_events.count("node_registered"), 1)
            self.assertNotIn("node_refreshed", node_events)
        finally:
            service.stop()

    def test_controller_start_replays_existing_sqlite_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = os.path.join(tmp, "controller-state")
            admin_file = os.path.join(tmp, "admin.json")
            self.assertEqual(cape_main(["controller", "init", "--state-dir", state_dir, "--user", "admin",
                                       "--token-out", admin_file], stdout=io.StringIO()), 0)
            journal_path = os.path.join(state_dir, "state.sqlite3")
            journal = SQLiteJournal(journal_path)
            seed = CAPEController(journal=journal)
            seed.register_node(
                node_id="node-replayed",
                pool=POOL,
                resources=NodeResources(cpu_cores=4, memory_gb=16),
                capabilities=RuntimeCapabilities(adapter="fake"),
                allocation_id="alloc-replayed",
            )
            request = seed.submit_workload(
                WorkloadSpec(
                    pool=POOL,
                    user_id="alice",
                    session_id="replay-task",
                    command=("true",),
                    resources=ResourceRequest(
                        gpus=0, cpu_cores=1, memory_gb=2
                    ),
                )
            )
            seed.schedule_once()
            execution_session_id = request.execution_session_id
            seed.complete_request(request.request_id, exit_code=0)
            journal.close()

            service = mock.Mock()
            service.url = "http://127.0.0.1:18892"
            with mock.patch(
                "pycape.cli.CAPEControllerService"
            ) as service_factory, mock.patch(
                "pycape.cli.time.sleep", side_effect=KeyboardInterrupt
            ):
                service_factory.return_value.start.return_value = service
                self.assertEqual(
                    cape_main(
                        [
                            "controller",
                            "start",
                            "--state-dir", state_dir,
                            "--token-file", admin_file,
                        ],
                        stdout=io.StringIO(),
                    ),
                    0,
                )

            replayed = service_factory.call_args.kwargs["controller"]
            self.assertIn(request.request_id, replayed.requests)
            self.assertIn(execution_session_id, replayed.sessions)
            self.assertEqual(
                replayed.sessions[execution_session_id].state.value, "CACHED_IDLE"
            )
            node = replayed.nodes["node-replayed"]
            self.assertEqual(node.active_sessions, [execution_session_id])
            self.assertEqual(node.used_cpu_cores, 1)
            self.assertEqual(node.used_memory_gb, 2)
            service.stop.assert_called_once_with()

    def test_cape_run_supports_async_id_output_and_bind_payload(self):
        ctrl = CAPEController()
        service = CAPEControllerService(controller=ctrl, schedule_interval=0.05).start()
        try:
            out = io.StringIO()
            rc = cape_main([
                "run",
                "--controller-url", service.url,
                "--pool", POOL,
                "--user", "alice",
                "--session-id", "task-a",
                "--image", "docker://ubuntu:latest",
                "--gpus", "1",
                "--bind", "/host/data:/data:ro",
                "--bind", "/host/workspace-a:/workspace:rw",
                "--workspace-target", "/cape-workspace",
                "--bind", "/host/output-a:/output:rw",
                "--env", "CAPE_RUN_TAG=alice-a",
                "--",
                "python", "train.py",
            ], stdout=out)
            self.assertEqual(rc, 0)
            request_id = out.getvalue().strip()
            self.assertTrue(request_id.startswith("req-"))
            request = ctrl.get_request(request_id)
            self.assertEqual(request.spec.command, ("python", "train.py"))
            self.assertEqual(request.spec.env["CAPE_RUN_TAG"], "alice-a")
            self.assertTrue(request.spec.keep_session_on_failure)
            self.assertEqual(request.spec.mounts, (
                MountSpec("/host/data", "/data", read_only=True),
                MountSpec("/host/workspace-a", "/workspace"),
                MountSpec("/host/output-a", "/output"),
            ))
            for field in ("workspace_path", "workspace_mount_path", "output_path", "output_mount_path"):
                self.assertFalse(hasattr(request.spec, field))
        finally:
            service.stop()

    def test_cape_run_suspend_on_failure_opt_out(self):
        parser = _build_cape_parser()
        default_args = parser.parse_args(
            [
                "run",
                "--pool",
                POOL,
                "--user",
                "alice",
                "--session-id",
                "failure-default",
                "--",
                "false",
            ]
        )
        self.assertTrue(
            _build_run_payload(default_args)["keep_session_on_failure"]
        )
        opt_out_args = parser.parse_args(
            [
                "run",
                "--pool",
                POOL,
                "--user",
                "alice",
                "--session-id",
                "failure-opt-out",
                "--suspend-on-failure",
                "--",
                "false",
            ]
        )
        self.assertFalse(
            _build_run_payload(opt_out_args)["keep_session_on_failure"]
        )

    def test_cape_run_serializes_separate_session_and_command_bounds(self):
        parser = _build_cape_parser()
        args = parser.parse_args(
            [
                "run",
                "--pool", POOL,
                "--user", "alice",
                "--session-id", "bounded-command",
                "--max-duration-seconds", "14400",
                "--command-timeout-seconds", "90",
                "--",
                "true",
            ]
        )
        resources = _build_run_payload(args)["resources"]
        self.assertEqual(resources["max_duration_seconds"], 14400.0)
        self.assertEqual(resources["command_timeout_seconds"], 90.0)

    def test_cape_run_rejects_removed_path_options_and_payload_contains_only_mounts(self):
        parser = _build_cape_parser()
        prefix = [
            "run",
            "--pool", POOL,
            "--user", "alice",
            "--session-id", "logical-task-a",
        ]
        for option in (
            "--workspace-path",
            "--workspace-mount-path",
            "--output-path",
            "--output-mount-path",
        ):
            with self.subTest(option=option):
                with mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(prefix + [option, "/legacy", "--", "echo", "ok"])

        args = parser.parse_args(prefix + [
            "--bind", "/host/workspace-a:/workspace:rw",
            "--bind", "/host/output-a:/output:rw",
            "--cwd", "/workspace",
            "--env", "CAPE_OUTPUT=/output",
            "--",
            "echo", "ok",
        ])
        payload = _build_run_payload(args)
        self.assertEqual(payload["session_id"], "logical-task-a")
        self.assertEqual(payload["mounts"], [
            {"source": "/host/workspace-a", "target": "/workspace", "read_only": False},
            {"source": "/host/output-a", "target": "/output", "read_only": False},
        ])
        self.assertEqual(payload["cwd"], "/workspace")
        self.assertEqual(payload["env"]["CAPE_OUTPUT"], "/output")
        for field in ("workspace_path", "workspace_mount_path", "output_path", "output_mount_path"):
            self.assertNotIn(field, payload)

        with mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([
                    "run",
                    "--pool", POOL,
                    "--user", "alice",
                    "--workspace", "removed-name",
                    "--", "true",
                ])

    def test_cape_async_status_logs_wait_and_cancel_verbs(self):
        ctrl = CAPEController()
        service = CAPEControllerService(controller=ctrl, schedule_interval=0.01).start()
        stop = threading.Event()
        agent = thread = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=4, memory_gb=16, gpu_uuids=()),
                    capabilities=RuntimeCapabilities(adapter="local-process", image_formats=("default",), node_local_cache=True),
                    log_dir=tmp,
                )
                agent.register()

                def heartbeat_loop():
                    while not stop.is_set():
                        agent.heartbeat_once(background=True)
                        time.sleep(0.02)

                thread = threading.Thread(target=heartbeat_loop)
                thread.daemon = True
                thread.start()

                submit_out = io.StringIO()
                rc = cape_main([
                    "run", "--controller-url", service.url, "--pool", POOL,
                    "--user", "alice", "--session-id", "task-a",
                    "--", sys.executable, "-c", "print('cape-log')",
                ], stdout=submit_out)
                self.assertEqual(rc, 0)
                request_id = submit_out.getvalue().strip()

                wait_out = io.StringIO()
                self.assertEqual(cape_main(["wait", request_id, "--controller-url", service.url, "--timeout", "5"], stdout=wait_out), 0)
                wait_payload = json.loads(wait_out.getvalue())
                self.assertEqual(wait_payload["state"], "COMPLETED")
                self.assertIsNotNone(wait_payload["started_at"])
                self.assertIsNotNone(wait_payload["ended_at"])
                self.assertLessEqual(
                    wait_payload["created_at"], wait_payload["started_at"]
                )
                self.assertLessEqual(
                    wait_payload["started_at"], wait_payload["ended_at"]
                )

                status_out = io.StringIO()
                self.assertEqual(cape_main(["status", request_id, "--controller-url", service.url], stdout=status_out), 0)
                status_payload = json.loads(status_out.getvalue())
                self.assertEqual(status_payload["request_id"], request_id)
                self.assertEqual(
                    status_payload["started_at"], wait_payload["started_at"]
                )
                self.assertEqual(
                    status_payload["ended_at"], wait_payload["ended_at"]
                )

                logs_out = io.StringIO()
                logs_err = io.StringIO()
                self.assertEqual(cape_main(["logs", request_id, "--controller-url", service.url], stdout=logs_out, stderr=logs_err), 0)
                self.assertIn("cape-log", logs_out.getvalue())
                self.assertEqual(logs_err.getvalue(), "")
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=5.0)
            if agent is not None:
                agent.join(timeout=5.0)
                with mock.patch.object(agent, "_write_log"), mock.patch.object(agent, "_save_state"):
                    agent._stop_all_local_sessions()
            service.stop()

    def test_cape_wait_transport_timeout_exceeds_server_wait_timeout(self):
        terminal = {
            "request_id": "req-long-wait",
            "state": "COMPLETED",
            "exit_code": 0,
        }
        out = io.StringIO()
        with mock.patch("pycape.cli._request", return_value=terminal) as request:
            self.assertEqual(cape_main([
                "wait", "req-long-wait",
                "--controller-url", "http://controller.example:8765",
                "--timeout", "1800",
            ], stdout=out), 0)

        self.assertEqual(json.loads(out.getvalue()), terminal)
        self.assertEqual(request.call_args[1]["timeout"], 1805.0)
        self.assertEqual(request.call_args[1]["query"], {"timeout": 1800.0})

    def test_cape_run_wait_preserves_terminal_execution_timestamps(self):
        submitted = {"request_id": "req-run-wait", "state": "QUEUED", "session_id": "task-a"}
        terminal = {
            "request_id": "req-run-wait",
            "state": "COMPLETED",
            "created_at": 10.0,
            "started_at": 12.0,
            "ended_at": 15.0,
            "updated_at": 15.0,
            "exit_code": 0,
        }
        out = io.StringIO()
        with mock.patch(
            "pycape.cli._request", side_effect=({"scheduler_id": "fixture", "user_id": "alice"}, submitted, terminal)
        ) as request:
            self.assertEqual(
                cape_main(
                    [
                        "run",
                        "--controller-url",
                        "http://controller.example:8765",
                        "--pool",
                        POOL,
                        "--user",
                        "alice",
                        "--session-id",
                        "task-a",
                        "--wait",
                        "--",
                        "true",
                    ],
                    stdout=out,
                ),
                0,
            )

        self.assertEqual(json.loads(out.getvalue()), terminal)
        self.assertEqual(request.call_count, 3)

    def test_cape_run_wait_can_cancel_on_running_relative_timeout(self):
        submitted = {"request_id": "req-run-timeout", "state": "QUEUED", "session_id": "task-a"}
        waiting = {
            "request_id": "req-run-timeout",
            "state": "RUNNING",
            "created_at": 10.0,
            "started_at": 12.0,
            "ended_at": None,
            "exit_code": None,
            "wait_timed_out": True,
            "wait_timeout_kind": "running",
        }
        cancelled = {
            "request_id": "req-run-timeout",
            "state": "CANCELLED",
            "reason": "test_running_timeout",
            "created_at": 10.0,
            "started_at": 12.0,
            "ended_at": 13.0,
            "exit_code": None,
        }
        out = io.StringIO()
        with mock.patch(
            "pycape.cli._request", side_effect=({"scheduler_id": "fixture", "user_id": "alice"}, submitted, waiting, cancelled)
        ) as request:
            self.assertEqual(
                cape_main(
                    [
                        "run",
                        "--controller-url",
                        "http://controller.example:8765",
                        "--pool",
                        POOL,
                        "--user",
                        "alice",
                        "--session-id",
                        "task-a",
                        "--wait",
                        "--timeout",
                        "1800",
                        "--running-timeout",
                        "60",
                        "--cancel-on-timeout",
                        "--cancel-reason",
                        "test_running_timeout",
                        "--",
                        "true",
                    ],
                    stdout=out,
                ),
                0,
            )

        payload = json.loads(out.getvalue())
        self.assertEqual(payload["state"], "CANCELLED")
        self.assertTrue(payload["wait_timed_out"])
        self.assertEqual(payload["wait_timeout_kind"], "running")
        self.assertEqual(request.call_count, 4)
        self.assertEqual(
            request.call_args_list[2].kwargs["query"],
            {"timeout": 1800.0, "running_timeout": 60.0},
        )
        self.assertEqual(
            request.call_args_list[3].kwargs["payload"],
            {"reason": "test_running_timeout"},
        )

    def test_running_timeout_starts_after_queued_request_runs(self):
        ctrl = CAPEController()
        service = CAPEControllerService(
            controller=ctrl, schedule_interval=0.01
        ).start()
        stop = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient(
                service.url,
                node_id="node-running-timeout",
                pool=POOL,
                resources=NodeResources(cpu_cores=4, memory_gb=16),
                capabilities=RuntimeCapabilities(
                    adapter="local-process",
                    image_formats=("default",),
                    node_local_cache=True,
                ),
                log_dir=tmp,
            )

            def delayed_agent_loop():
                time.sleep(0.2)
                agent.register()
                while not stop.is_set():
                    agent.heartbeat_once(background=True)
                    time.sleep(0.02)

            thread = threading.Thread(target=delayed_agent_loop)
            thread.daemon = True
            thread.start()
            out = io.StringIO()
            before = time.monotonic()
            try:
                rc = cape_main(
                    [
                        "run",
                        "--controller-url",
                        service.url,
                        "--pool",
                        POOL,
                        "--user",
                        "alice",
                        "--session-id",
                        "task-running-timeout",
                        "--wait",
                        "--timeout",
                        "2",
                        "--running-timeout",
                        "0.15",
                        "--cancel-on-timeout",
                        "--cancel-reason",
                        "test_running_timeout",
                        "--",
                        sys.executable,
                        "-c",
                        "import time; time.sleep(1)",
                    ],
                    stdout=out,
                )
                elapsed = time.monotonic() - before
            finally:
                cleanup_started = time.monotonic()
                try:
                    # CANCELLED acknowledges the controller, not process exit.
                    # Keep heartbeats delivering cancellation while workers stop.
                    agent.join(timeout=2.0)
                finally:
                    stop.set()
                    try:
                        thread.join(timeout=2.0)
                        # The final heartbeat may have spawned cleanup workers.
                        agent.join(timeout=2.0)
                    finally:
                        service.stop()
                cleanup_elapsed = time.monotonic() - cleanup_started
            self.assertFalse(thread.is_alive(), "heartbeat survived cleanup ({:.3f}s)".format(cleanup_elapsed))
            with agent._state_lock:
                workers_alive = any(worker.is_alive() for worker in agent._threads)
                processes_remaining = bool(agent.running_processes)
            self.assertFalse(workers_alive, "agent workers survived cleanup ({:.3f}s)".format(cleanup_elapsed))
            self.assertFalse(processes_remaining, "workload processes remain after cancellation")

        payload = json.loads(out.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(payload["state"], "CANCELLED")
        self.assertEqual(payload["reason"], "test_running_timeout")
        self.assertEqual(payload["wait_timeout_kind"], "running")
        self.assertGreaterEqual(payload["started_at"] - payload["created_at"], 0.1)
        self.assertGreaterEqual(elapsed, 0.3)
        self.assertLess(elapsed, 1.5)

    def test_cape_cancel_verb_uses_existing_endpoint(self):
        ctrl = CAPEController()
        service = CAPEControllerService(controller=ctrl, schedule_interval=10.0).start()
        try:
            submit_out = io.StringIO()
            cape_main([
                "run", "--controller-url", service.url, "--pool", POOL,
                "--user", "alice", "--session-id", "task-a",
                "--", sys.executable, "-c", "print('queued')",
            ], stdout=submit_out)
            request_id = submit_out.getvalue().strip()
            cancel_out = io.StringIO()
            self.assertEqual(cape_main(["cancel", request_id, "--controller-url", service.url], stdout=cancel_out), 0)
            self.assertEqual(json.loads(cancel_out.getvalue())["state"], "CANCELLED")
        finally:
            service.stop()

    def test_cape_node_join_builds_apptainer_adapter_with_containment(self):
        parser = _build_cape_parser()
        args = parser.parse_args([
            "node", "enroll", "--node-token-out", "unused-agent.json",
            "--controller-url", "http://127.0.0.1:1",
            "--node-id", "node-a",
            "--pool", POOL,
            "--gpu-uuid", "GPU-a",
            "--runtime-adapter", "apptainer",
            "--apptainer-binary", "/opt/apptainer/bin/apptainer",
            "--apptainer-unsquash",
            "--runtime-cache", "/tmp/cape-cache",
            "--runtime-tmp", "/tmp/cape-tmp",
            "--runtime-dir", "/tmp/cape-run",
            "--once",
        ])
        capabilities = RuntimeCapabilities(
            adapter="apptainer",
            isolation_policies=("default", "image-stateless"),
            gpu_enforcement=EnforcementMode.ADVISORY,
        )
        with mock.patch(
            "pycape.cli.ApptainerRuntimeAdapter.probe",
            return_value=capabilities,
        ):
            agent = _node_agent_from_args(args)
        runtime = agent.runtime_adapters["apptainer"]
        self.assertEqual(agent.capabilities.adapter, "apptainer")
        self.assertEqual(set(agent.runtime_capabilities), {"apptainer"})
        self.assertEqual(
            agent.runtime_capabilities["apptainer"].gpu_enforcement.value,
            "advisory",
        )
        self.assertTrue(runtime.contain)
        self.assertTrue(runtime.unsquash)
        self.assertEqual(runtime.binary, "/opt/apptainer/bin/apptainer")
        self.assertEqual(runtime.cache_dir, "/tmp/cape-cache/apptainer")
        self.assertEqual(runtime.tmp_dir, "/tmp/cape-tmp/apptainer")
        self.assertEqual(runtime.runtime_dir, "/tmp/cape-run/apptainer")
        self.assertEqual(agent.runtime_cache_config, "/tmp/cape-cache")

    def test_cape_node_join_self_exit_defaults_on_with_explicit_opt_out(self):
        parser = _build_cape_parser()
        common = [
            "node",
            "enroll", "--node-token-out", "unused-agent.json",
            "--controller-url",
            "http://127.0.0.1:1",
            "--node-id",
            "node-lifecycle",
            "--pool",
            POOL,
            "--runtime-adapter",
            "local-process",
            "--once",
        ]

        default_agent = _node_agent_from_args(parser.parse_args(common))
        retained_agent = _node_agent_from_args(
            parser.parse_args(common + ["--stay-after-removal"])
        )

        self.assertTrue(default_agent.exit_on_removal)
        self.assertFalse(retained_agent.exit_on_removal)

    def test_long_running_node_join_retries_initial_active_registration(self):
        class FakeAgent(object):
            def __init__(self):
                self.metrics_link = None
                self.retry_intervals = []
                self.run_intervals = []

            def register_until_available(self, interval):
                self.retry_intervals.append(interval)
                return {"node_id": "node-retry", "state": "ACTIVE"}

            def run_forever(self, interval):
                self.run_intervals.append(interval)
                return {"action": "shutdown"}

        agent = FakeAgent()
        stdout = io.StringIO()
        with mock.patch(
            "pycape.cli._request", return_value={"state": "REGISTERING"}
        ), mock.patch("pycape.cli._node_agent_from_args", return_value=agent), mock.patch("pycape.node_enrollment.enroll", return_value="retry-epoch"):
            rc = cape_main(
                [
                    "node",
                    "enroll", "--node-token-out", "unused-agent.json",
                    "--controller-url",
                    "http://controller.invalid",
                    "--node-id",
                    "node-retry",
                    "--pool",
                    POOL,
                    "--heartbeat-interval",
                    "0.25",
                ],
                stdout=stdout,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(agent.retry_intervals, [0.25])
        self.assertEqual(agent.run_intervals, [0.25])
        self.assertEqual(json.loads(stdout.getvalue())["state"], "ACTIVE")

    def test_cape_node_join_builds_bubblewrap_adapter_for_host_policies(self):
        parser = _build_cape_parser()
        args = parser.parse_args([
            "node", "enroll", "--node-token-out", "unused-agent.json",
            "--controller-url", "http://127.0.0.1:1",
            "--node-id", "node-host",
            "--pool", POOL,
            "--gpu-uuid", "GPU-host",
            "--runtime-adapter", "bubblewrap",
            "--bubblewrap-binary", sys.executable,
            "--bubblewrap-system-path", "/usr",
            "--bubblewrap-system-path", "/opt",
            "--runtime-dir", "/tmp/cape-bwrap-run",
            "--once",
        ])

        capabilities = RuntimeCapabilities(
            adapter="bubblewrap",
            image_formats=("host",),
            isolation_policies=("default", "host-stateless"),
            gpu_enforcement=EnforcementMode.ADVISORY,
        )
        with mock.patch(
            "pycape.cli.BubbleWrapRuntimeAdapter.probe",
            return_value=capabilities,
        ):
            agent = _node_agent_from_args(args)

        runtime = agent.runtime_adapters["bubblewrap"]
        self.assertEqual(agent.capabilities.adapter, "bubblewrap")
        self.assertEqual(set(agent.runtime_capabilities), {"bubblewrap"})
        self.assertEqual(
            agent.runtime_capabilities["bubblewrap"].image_formats, ("host",)
        )
        self.assertEqual(runtime.binary, sys.executable)
        self.assertEqual(runtime.runtime_dir, "/tmp/cape-bwrap-run/bubblewrap")
        self.assertEqual(runtime.node_gpu_uuids, ("GPU-host",))
        self.assertEqual(runtime.system_paths, ("/usr", "/opt"))

    def test_cape_node_join_rejects_an_unconfined_bubblewrap_system_path(self):
        parser = _build_cape_parser()
        args = parser.parse_args([
            "node", "enroll", "--node-token-out", "unused-agent.json",
            "--controller-url", "http://127.0.0.1:1",
            "--node-id", "node-host",
            "--pool", POOL,
            "--runtime-adapter", "bubblewrap",
            "--bubblewrap-system-path", "/",
            "--once",
        ])

        with self.assertRaisesRegex(ValueError, "host root"):
            _node_agent_from_args(args)

    def test_cape_node_join_auto_probes_and_registers_every_working_adapter(self):
        parser = _build_cape_parser()
        args = parser.parse_args([
            "node", "enroll", "--node-token-out", "unused-agent.json",
            "--controller-url", "http://127.0.0.1:1",
            "--node-id", "node-multi",
            "--pool", POOL,
            "--runtime-cache", "/tmp/cape-cache",
            "--runtime-tmp", "/tmp/cape-tmp",
            "--runtime-dir", "/tmp/cape-run",
            "--apptainer-probe-image", "docker://example/probe:latest",
            "--podman-probe-image", "docker://example/probe:latest",
            "--once",
        ])
        apptainer = RuntimeCapabilities(
            adapter="apptainer",
            isolation_policies=(
                "default", "image-stateless", "image-stateful"
            ),
            gpu_enforcement=EnforcementMode.ADVISORY,
        )
        bubblewrap = RuntimeCapabilities(
            adapter="bubblewrap",
            image_formats=("host",),
            isolation_policies=(
                "default", "host-stateless", "host-stateful"
            ),
            gpu_enforcement=EnforcementMode.ADVISORY,
        )
        unavailable_podman = RuntimeCapabilities(
            adapter="podman",
            isolation_policies=(),
            nested_container=False,
        )

        with mock.patch(
            "pycape.cli.ApptainerRuntimeAdapter.probe",
            return_value=apptainer,
        ) as apptainer_probe, mock.patch(
            "pycape.cli.BubbleWrapRuntimeAdapter.probe",
            return_value=bubblewrap,
        ) as bubblewrap_probe, mock.patch(
            "pycape.cli.PodmanRuntimeAdapter.probe",
            return_value=unavailable_podman,
        ) as podman_probe:
            agent = _node_agent_from_args(args)

        apptainer_probe.assert_called_once_with(execute=True)
        bubblewrap_probe.assert_called_once_with(execute=True)
        podman_probe.assert_called_once_with(execute=True)
        self.assertEqual(
            set(agent.runtime_adapters),
            {"apptainer", "bubblewrap"},
        )
        self.assertEqual(set(agent.runtime_capabilities), set(agent.runtime_adapters))
        self.assertEqual(agent.capabilities.adapter, "multi")
        self.assertEqual(
            set(agent.capabilities.isolation_policies),
            {
                "default",
                "host-stateless",
                "host-stateful",
                "image-stateless",
                "image-stateful",
            },
        )
        registration = agent._registration_payload()
        self.assertEqual(
            set(registration["runtime_capabilities"]),
            {"apptainer", "bubblewrap"},
        )

    def test_runtime_discovery_defaults_and_image_overrides(self):
        parser = _build_cape_parser()
        base = ["node", "enroll", "--pool", POOL, "--node-token-out", "unused.json"]
        args = parser.parse_args(base)
        self.assertIsNone(args.runtime_adapter)
        self.assertEqual(args.apptainer_probe_image, "docker://docker.io/library/ubuntu:24.04")
        self.assertEqual(args.podman_probe_image, args.apptainer_probe_image)
        custom = parser.parse_args(base + ["--apptainer-probe-image", "/local/probe.sif",
                                          "--podman-probe-image", "docker://registry/probe@sha256:custom"])
        self.assertEqual(custom.apptainer_probe_image, "/local/probe.sif")
        self.assertEqual(custom.podman_probe_image, "docker://registry/probe@sha256:custom")

    def test_auto_discovery_keeps_all_three_healthy_container_adapters(self):
        args = _build_cape_parser().parse_args([
            "node", "enroll", "--pool", POOL, "--node-token-out", "unused.json"])
        with mock.patch("pycape.cli.ApptainerRuntimeAdapter.probe", return_value=RuntimeCapabilities(
                 adapter="apptainer", isolation_policies=("image-stateless",))) as apptainer, \
             mock.patch("pycape.cli.BubbleWrapRuntimeAdapter.probe", return_value=RuntimeCapabilities(
                 adapter="bubblewrap", isolation_policies=("host-stateless",))) as bubblewrap, \
             mock.patch("pycape.cli.PodmanRuntimeAdapter.probe", return_value=RuntimeCapabilities(
                 adapter="podman", isolation_policies=("image-stateless",))) as podman:
            agent = _node_agent_from_args(args)
        self.assertEqual(set(agent.runtime_adapters), {"apptainer", "bubblewrap", "podman"})
        self.assertEqual(set(agent.runtime_capabilities), set(agent.runtime_adapters))
        for probe in (apptainer, bubblewrap, podman):
            probe.assert_called_once_with(execute=True)

    def test_auto_runtime_exception_does_not_suppress_healthy_peers(self):
        args = _build_cape_parser().parse_args([
            "node", "enroll", "--pool", POOL, "--node-token-out", "unused.json"])
        with mock.patch("pycape.cli.ApptainerRuntimeAdapter.probe", side_effect=OSError("probe denied")), \
             mock.patch("pycape.cli.BubbleWrapRuntimeAdapter.probe", return_value=RuntimeCapabilities(
                 adapter="bubblewrap", isolation_policies=("host-stateless",))), \
             mock.patch("pycape.cli.PodmanRuntimeAdapter.probe", return_value=RuntimeCapabilities(
                 adapter="podman", isolation_policies=("image-stateless",))), \
             mock.patch("pycape.cli.LocalProcessRuntimeAdapter") as local:
            agent = _node_agent_from_args(args)
        self.assertEqual(set(agent.runtime_adapters), {"bubblewrap", "podman"})
        local.assert_not_called()

    def test_no_container_runtime_does_not_fall_back_to_local_process(self):
        args = _build_cape_parser().parse_args([
            "node", "enroll", "--pool", POOL, "--node-token-out", "unused.json"])
        unavailable = RuntimeCapabilities(adapter="unavailable", nested_container=False, isolation_policies=())
        with mock.patch("pycape.cli.ApptainerRuntimeAdapter.probe", return_value=unavailable), \
             mock.patch("pycape.cli.BubbleWrapRuntimeAdapter.probe", return_value=unavailable), \
             mock.patch("pycape.cli.PodmanRuntimeAdapter.probe", return_value=unavailable), \
             mock.patch("pycape.cli.LocalProcessRuntimeAdapter") as local:
            with self.assertRaisesRegex(RuntimeError, "no runtime adapter"):
                _node_agent_from_args(args)
        local.assert_not_called()

    def test_explicit_local_process_does_not_probe_containers(self):
        args = _build_cape_parser().parse_args([
            "node", "enroll", "--pool", POOL, "--node-token-out", "unused.json",
            "--runtime-adapter", "local-process"])
        with mock.patch("pycape.cli.ApptainerRuntimeAdapter.probe") as apptainer, \
             mock.patch("pycape.cli.BubbleWrapRuntimeAdapter.probe") as bubblewrap, \
             mock.patch("pycape.cli.PodmanRuntimeAdapter.probe") as podman:
            agent = _node_agent_from_args(args)
        self.assertEqual(set(agent.runtime_adapters), {"local-process"})
        for probe in (apptainer, bubblewrap, podman):
            probe.assert_not_called()

    def test_cape_node_join_repeatable_filter_rejects_a_failed_requested_probe(self):
        parser = _build_cape_parser()
        args = parser.parse_args([
            "node", "enroll", "--node-token-out", "unused-agent.json",
            "--controller-url", "http://127.0.0.1:1",
            "--node-id", "node-filtered",
            "--pool", POOL,
            "--runtime-adapter", "apptainer",
            "--runtime-adapter", "bubblewrap",
            "--once",
        ])
        apptainer = RuntimeCapabilities(
            adapter="apptainer",
            isolation_policies=("default", "image-stateless"),
        )
        unavailable_bubblewrap = RuntimeCapabilities(
            adapter="bubblewrap",
            image_formats=("host",),
            isolation_policies=(),
            nested_container=False,
        )
        with mock.patch(
            "pycape.cli.ApptainerRuntimeAdapter.probe",
            return_value=apptainer,
        ), mock.patch(
            "pycape.cli.BubbleWrapRuntimeAdapter.probe",
            return_value=unavailable_bubblewrap,
        ):
            with self.assertRaisesRegex(RuntimeError, "bubblewrap"):
                _node_agent_from_args(args)

    def test_cape_auth_token_file_supports_owner_status_and_cross_user_denial(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = ManagedFixture(os.environ.get("PYCAPE_BIN", "pycape"), tmp)
            service = fixture.start_embedded()
            fixture.pool(POOL)
            alice_token = fixture.user_token("alice")
            bob_token = fixture.user_token("bob")
            alice_id = fixture.request("GET", "/v1/identity/me", token=alice_token)["user_id"]
            node_token = new_secret()
            fixture.request("POST", "/v1/identity/nodes/enroll", {
                "operation_id": str(uuid.uuid4()), "verifier": verifier(node_token), "node_id": "node-a",
                "pool": fixture.pool(POOL), "allocation_id": "allocation-a", "agent_epoch": "epoch-a",
                "resources": {"cpu_cores": 1, "memory_gb": 1, "gpu_uuids": []},
            }, token=alice_token)
            try:
                submit_out = io.StringIO()
                self.assertEqual(cape_main([
                    "run", "--controller-url", service.url, "--token", alice_token,
                    "--pool", POOL, "--user", "alice", "--session-id", "task-a",
                    "--", sys.executable, "-c", "print('secret')",
                ], stdout=submit_out), 0)
                request_id = submit_out.getvalue().strip()

                owner_out = io.StringIO()
                self.assertEqual(cape_main(["status", request_id, "--controller-url", service.url, "--token", alice_token], stdout=owner_out), 0)
                self.assertEqual(json.loads(owner_out.getvalue())["owner"], alice_id)

                other_err = io.StringIO()
                self.assertEqual(cape_main(["status", request_id, "--controller-url", service.url, "--token", bob_token], stdout=io.StringIO(), stderr=other_err), 1)
                self.assertIn("403", other_err.getvalue())

                node_err = io.StringIO()
                self.assertEqual(cape_main([
                    "run", "--controller-url", service.url, "--token", node_token,
                    "--pool", POOL, "--user", "node-a", "--session-id", "task-a",
                    "--", sys.executable, "-c", "print('bad')",
                ], stdout=io.StringIO(), stderr=node_err), 1)
                self.assertIn("403", node_err.getvalue())
            finally:
                fixture.close_embedded()


if __name__ == "__main__":
    unittest.main()
