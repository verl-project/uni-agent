# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from contextlib import ExitStack
import json
import os
import sqlite3
import sys
import tempfile
import types
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import (
    CAPEController,
    CAPEControllerService,
    ContainerCommandSpec,
    EnforcementMode,
    EventRecord,
    LocalProcessRuntimeAdapter,
    MountSpec,
    NodeAgentClient,
    NodeResources,
    NodeState,
    RequestState,
    ResourceRequest,
    RuntimeCapabilities,
    RuntimeExecution,
    RuntimeResult,
    SchedulerPolicy,
    SQLiteJournal,
    ApptainerRuntimeAdapter,
    BubbleWrapRuntimeAdapter,
    SessionState,
    WorkloadSpec,
)
from pycape.cli import cape_main
from pycape.observability import JsonlExporter, MetricsCollector, WandbExporter
from pycape.profiles import ProfileRegistry
from pycape.service import runtime_capabilities_from_dict
from pycape.testing import FakeClock
from tests.workspace_fixtures import install_synthetic_assignment_workspaces, stop_test_agent_workers


POOL = "pool-a"


def local_spec(command=None, gpus=0, reuse=True, session_id="svc"):
    return WorkloadSpec(
        pool=POOL,
        user_id="alice",
        command=tuple(command or (sys.executable, "-c", "print(36)")),
        resources=ResourceRequest(gpus=gpus, cpu_cores=1, memory_gb=1, max_duration_seconds=100),
        session_id=session_id,
        reuse=reuse,
    )


def register_local_node(ctrl, node_id="node-a", gpus=0, allocation_id=None, cached_images=()):
    return ctrl.register_node(
        node_id=node_id,
        pool=POOL,
        resources=NodeResources(cpu_cores=8, memory_gb=64, gpu_uuids=tuple("{}-gpu-{}".format(node_id, i) for i in range(gpus))),
        capabilities=RuntimeCapabilities(
            adapter="local-process",
            image_formats=("default", "docker"),
            nested_container=True,
            can_bind_mounts=True,
            can_select_gpus=True,
            gpu_enforcement=EnforcementMode.HARD,
            node_local_cache=True,
        ),
        allocation_id=allocation_id,
        cached_images=cached_images,
    )


class StrictServiceAgentTests(unittest.TestCase):
    def setUp(self):
        install_synthetic_assignment_workspaces(self)
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        environment = mock.patch.dict(os.environ, {"CAPE_SESSION_STATE_DIR": os.path.join(cache.name, "sessions")})
        environment.start()
        self.addCleanup(environment.stop)
        original = CAPEControllerService._http_identity_me

        async def identify(service, request):
            if getattr(service.controller.auth, "managed", False):
                return await original(service, request)
            from aiohttp import web
            return web.json_response({"scheduler_id": "unit-{}".format(id(service.controller)), "user_id": "alice"})

        identity = mock.patch.object(CAPEControllerService, "_http_identity_me", identify)
        identity.start()
        self.addCleanup(identity.stop)

    def test_report_transport_failure_retries_small_infrastructure_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient("http://unused", "node", "pool", NodeResources(1, 1), log_dir=tmp)
            sent = []

            def post(path, payload, token=None):
                sent.append(dict(payload))
                if len(sent) <= 2:
                    raise OSError("transport unavailable")
                return {"commands": []}

            with mock.patch.object(agent, "_post", side_effect=post), mock.patch("threading.Event.wait"):
                agent._report("command", 23, "full stdout", "full stderr", "/node/command.log")
            self.assertEqual(sent[0]["stdout"], "full stdout")
            self.assertEqual(sent[1], sent[2])
            self.assertEqual(sent[1]["stdout"], "")
            self.assertEqual(sent[1]["failure_kind"], "infrastructure")
            self.assertNotEqual(sent[1]["exit_code"], 0)
            self.assertEqual(sent[1]["log_path"], "/node/command.log")
            self.assertEqual(json.loads(sent[1]["stderr"])["original_exit_code"], 23)

    def test_sqlite_writer_is_bounded_ordered_batched_and_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "journal.sqlite")
            journal = SQLiteJournal(path, batch_size=64, queue_capacity=1)
            self.assertEqual(journal._thread.name, "cape-journal-writer")
            journal.append(EventRecord(
                0.0, "request_submitted", "req-000001", state="QUEUED"
            ))
            # The durable id is visible to a second SQLite connection before
            # any explicit flush or clean shutdown.
            connection = sqlite3.connect(path)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT event_type FROM events ORDER BY id"
                    ).fetchall(),
                    [("request_submitted",)],
                )
            finally:
                connection.close()
            for index in range(1, 101):
                journal.append(EventRecord(
                    float(index), "request_state", "req-000001", state="RUNNING"
                ))
            journal.close()
            reopened = SQLiteJournal(path)
            try:
                events = reopened.list_events()
                self.assertEqual(len(events), 101)
                self.assertEqual(events[0].event_type, "request_submitted")
                self.assertEqual(events[-1].timestamp, 100.0)
            finally:
                reopened.close()
            self.assertLess(journal.commit_count, 101)

    def test_http_submit_uses_one_durable_commit_then_one_async_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = SQLiteJournal(os.path.join(tmp, "journal.sqlite"))
            ctrl = CAPEController(journal=journal)
            service = CAPEControllerService(
                controller=ctrl,
                schedule_interval=100.0,
            ).start()
            try:
                commits_before = journal.commit_count
                output = io.StringIO()

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
                        "task-a",
                        "--",
                        "true",
                    ],
                    stdout=output,
                )

                self.assertEqual(rc, 0)
                self.assertRegex(output.getvalue().strip(), r"^req-")
                journal.flush()
                events = journal.list_events()
                self.assertEqual(
                    [event.event_type for event in events],
                    ["request_submitted", "logical_session_created", "request_queued"],
                )
                self.assertEqual(journal.commit_count - commits_before, 2)
            finally:
                service.stop()
                journal.close()

    def test_agent_log_directory_creation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "agent-logs")
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-logs",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=1),
                log_dir=log_dir,
            )
            with mock.patch("pycape.agent.os.path.isdir", return_value=False), mock.patch(
                "pycape.agent.os.makedirs"
            ) as makedirs:
                path = agent._log_path("cmd-1")

            self.assertEqual(path, os.path.join(log_dir, "cmd-1.log"))
            makedirs.assert_called_once_with(log_dir, exist_ok=True)

    def test_agent_reports_workload_failure_kind_for_nonzero_command(self):
        class Runtime(object):
            def start_session(self, _session_spec):
                return RuntimeResult(0, "", "")

            def run_command(self, _execution_session_id, _command, cwd=None, env=None):
                return RuntimeResult(9, "", "compile failed")

        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-failure-kind",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=1),
                runtime_adapters={"fake": Runtime()},
                log_dir=tmp,
                state_path=os.path.join(tmp, "state.json"),
            )
            reports = []
            agent._report = lambda *args, **kwargs: reports.append(
                (args, kwargs)
            )
            agent.execute_assignment(
                {
                    "command_id": "cmd-workload-failure",
                    "request_id": "req-workload-failure",
                    "execution_session_id": "sess-workload-failure",
                    "session_spec": {
                        "execution_session_id": "sess-workload-failure",
                        "mounts": [],
                        "runtime_adapter": "fake",
                    },
                    "command_spec": {"command": ["false"], "env": {}},
                }
            )

        self.assertEqual(reports[0][0][1], 9)
        self.assertEqual(reports[0][0][5], "workload")

    def test_agent_reports_infrastructure_for_prepare_start_and_exception(self):
        class Runtime(object):
            def __init__(self, failure_stage):
                self.failure_stage = failure_stage

            def prepare_image(self, image_ref, _cache_config):
                if self.failure_stage == "prepare":
                    return RuntimeResult(127, "", "prepare failed")
                return image_ref

            def start_session(self, _session_spec):
                if self.failure_stage == "start":
                    return RuntimeResult(127, "", "start failed")
                return RuntimeResult(0, "", "")

            def run_command(self, _execution_session_id, _command, cwd=None, env=None):
                if self.failure_stage == "launch":
                    return RuntimeResult(
                        127,
                        "",
                        "failed to launch runtime",
                        failure_kind="infrastructure",
                    )
                raise RuntimeError("agent execution failed")

        for stage in ("prepare", "start", "launch", "exception"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                agent = NodeAgentClient(
                    "http://controller.invalid",
                    node_id="node-infrastructure-" + stage,
                    pool=POOL,
                    resources=NodeResources(cpu_cores=1, memory_gb=1),
                    runtime_adapters={"fake": Runtime(stage)},
                    log_dir=tmp,
                    state_path=os.path.join(tmp, "state.json"),
                )
                reports = []
                agent._report = lambda *args, **kwargs: reports.append(
                    (args, kwargs)
                )
                agent.execute_assignment(
                    {
                        "command_id": "cmd-" + stage,
                        "request_id": "req-" + stage,
                        "execution_session_id": "sess-" + stage,
                        "session_spec": {
                            "execution_session_id": "sess-" + stage,
                            "image": "docker://example/test",
                            "mounts": [],
                            "runtime_adapter": "fake",
                        },
                        "command_spec": {
                            "command": ["true"],
                            "env": {},
                        },
                    }
                )

                self.assertNotEqual(reports[0][0][1], 0)
                self.assertEqual(
                    reports[0][0][5], "infrastructure"
                )

    def test_agent_command_log_records_exact_runtime_launches_for_image_and_host_policies(self):
        class LoggingRuntime(object):
            def __init__(self, start_execution, command_execution):
                self.start_execution = start_execution
                self.command_execution = command_execution
                self.sessions = {}

            def prepare_image(self, image_ref, _cache_config):
                return image_ref

            def start_session(self, session_spec):
                self.sessions[session_spec.execution_session_id] = session_spec
                return RuntimeResult(
                    0,
                    "",
                    "",
                    executions=(self.start_execution,),
                )

            def run_command(self, _execution_session_id, _command, cwd=None, env=None):
                return RuntimeResult(
                    0,
                    "payload\n",
                    "warning\n",
                    executions=(self.command_execution,),
                )

        cases = (
            (
                "apptainer",
                "image-stateless",
                "/images/workload.sif",
                RuntimeExecution(
                    "session-provision",
                    ["apptainer", "overlay", "create", "/state/layer.img"],
                    exit_code=0,
                ),
                RuntimeExecution(
                    "command",
                    [
                        "apptainer",
                        "exec",
                        "--nv",
                        "--pwd",
                        "/workspace",
                        "/images/workload.sif",
                        "python",
                        "train.py",
                    ],
                    cwd=None,
                    env={"APPTAINER_CACHEDIR": "/cache"},
                    exit_code=0,
                ),
                "apptainer exec --nv --pwd /workspace /images/workload.sif python train.py",
            ),
            (
                "bubblewrap",
                "host-stateless",
                None,
                RuntimeExecution(
                    "session-provision", ["/bin/true"], exit_code=0
                ),
                RuntimeExecution(
                    "command",
                    ["bwrap", "--ro-bind", "/", "/", "--", "nvidia-smi", "-L"],
                    env={"CUDA_VISIBLE_DEVICES": "GPU-A"},
                    exit_code=0,
                ),
                "bwrap --ro-bind / / -- nvidia-smi -L",
            ),
            (
                "podman",
                "image-stateless",
                "nvcr.io/nvidia/pytorch:26.06-py3",
                RuntimeExecution(
                    "session-provision", ["/bin/true"], exit_code=0
                ),
                RuntimeExecution(
                    "command",
                    [
                        "podman",
                        "run",
                        "--rm",
                        "nvcr.io/nvidia/pytorch:26.06-py3",
                        "python",
                        "train.py",
                    ],
                    cwd="/workspace",
                    exit_code=0,
                ),
                "podman run --rm nvcr.io/nvidia/pytorch:26.06-py3 python train.py",
            ),
            (
                "local-process",
                "none",
                None,
                RuntimeExecution(
                    "session-provision", ["/bin/true"], exit_code=0
                ),
                RuntimeExecution(
                    "command",
                    ["/usr/bin/python3", "train.py"],
                    cwd="/workspace",
                    exit_code=0,
                ),
                "/usr/bin/python3 train.py",
            ),
        )

        for number, case in enumerate(cases, 1):
            runtime_name, policy, image, start_execution, command_execution, expected_exec = case
            with self.subTest(runtime=runtime_name), tempfile.TemporaryDirectory() as tmp:
                runtime = LoggingRuntime(start_execution, command_execution)
                agent = NodeAgentClient(
                    "http://controller.invalid",
                    node_id="node-logs",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=2, memory_gb=2),
                    runtime_capabilities={
                        runtime_name: RuntimeCapabilities(
                            adapter=runtime_name,
                            isolation_policies=("default", policy),
                        )
                    },
                    runtime_adapters={runtime_name: runtime},
                    log_dir=tmp,
                    state_path=os.path.join(tmp, "state.json"),
                )
                agent._report = lambda *args: None
                command_id = "cmd-log-{}".format(number)
                agent.execute_assignment(
                    {
                        "command_id": command_id,
                        "request_id": "req-log-{}".format(number),
                        "execution_session_id": "sess-log-{}".format(number),
                        "session_spec": {
                            "execution_session_id": "sess-log-{}".format(number),
                            "image": image,
                            "mounts": [],
                            "cwd": "/workspace",
                            "runtime_adapter": runtime_name,
                            "isolation_policy": policy,
                        },
                        "command_spec": {
                            "command": ["python", "train.py"],
                            "cwd": "/workspace",
                            "env": {},
                        },
                    }
                )

                with open(os.path.join(tmp, command_id + ".log")) as fh:
                    logged = fh.read()

                self.assertIn("request req-log-{}".format(number), logged)
                self.assertIn("session sess-log-{}".format(number), logged)
                self.assertIn("runtime-adapter: " + runtime_name, logged)
                self.assertIn("isolation-policy: " + policy, logged)
                self.assertIn("cwd: /workspace", logged)
                self.assertIn("phase: session-provision", logged)
                self.assertIn("exec: " + expected_exec, logged)
                self.assertIn("exit-code: 0", logged)
                self.assertTrue(logged.endswith("payload\nwarning\n"))
                if runtime_name == "apptainer":
                    self.assertNotIn("apptainer instance", logged)

    def test_agent_runtime_command_log_redacts_secret_env_and_option_values(self):
        class SecretRuntime(object):
            def start_session(self, _session_spec):
                return RuntimeResult(0, "", "")

            def run_command(self, _execution_session_id, _command, cwd=None, env=None):
                execution = RuntimeExecution(
                    "command",
                    [
                        "bwrap",
                        "--setenv",
                        "SECRET_TOKEN",
                        "top-secret",
                        "--setenv",
                        "PRIVATE_VALUE",
                        "custom-secret",
                        "--",
                        "tool",
                        "--api-key=also-secret",
                    ],
                    env={
                        "SECRET_TOKEN": "top-secret",
                        "PUBLIC_MODE": "safe",
                        "PRIVATE_VALUE": "custom-secret",
                    },
                    exit_code=0,
                )
                return RuntimeResult(
                    0, "ok\n", "", executions=(execution,)
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = SecretRuntime()
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-secret",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=1),
                capabilities=RuntimeCapabilities(
                    adapter="bubblewrap",
                    isolation_policies=("default", "host-stateless"),
                ),
                runtime_adapters={"bubblewrap": runtime},
                log_dir=tmp,
                state_path=os.path.join(tmp, "state.json"),
                redact_env_keys=("PRIVATE_VALUE",),
            )
            agent._report = lambda *args: None
            agent.execute_assignment(
                {
                    "command_id": "cmd-secret",
                    "request_id": "req-secret",
                    "execution_session_id": "sess-secret",
                    "session_spec": {
                        "execution_session_id": "sess-secret",
                        "mounts": [],
                        "runtime_adapter": "bubblewrap",
                        "isolation_policy": "host-stateless",
                    },
                    "command_spec": {
                        "command": ["tool"],
                        "env": {"SECRET_TOKEN": "top-secret"},
                    },
                }
            )

            with open(os.path.join(tmp, "cmd-secret.log")) as fh:
                logged = fh.read()

        self.assertNotIn("top-secret", logged)
        self.assertNotIn("also-secret", logged)
        self.assertNotIn("custom-secret", logged)
        self.assertIn("SECRET_TOKEN=***", logged)
        self.assertIn("PRIVATE_VALUE=***", logged)
        self.assertIn("--api-key=***", logged)
        self.assertIn("PUBLIC_MODE=safe", logged)

    def test_agent_session_log_records_stop_lifecycle_command(self):
        class LifecycleRuntime(object):
            def __init__(self):
                self.sessions = {}

            def start_session(self, session_spec):
                self.sessions[session_spec.execution_session_id] = session_spec
                return RuntimeResult(0, "", "")

            def run_command(self, _execution_session_id, _command, cwd=None, env=None):
                return RuntimeResult(
                    0,
                    "ok\n",
                    "",
                    argv=["podman", "exec", "sess-life", "true"],
                )

            def stop_session(self, execution_session_id):
                self.sessions.pop(execution_session_id, None)
                return RuntimeResult(
                    0,
                    "",
                    "",
                    executions=(
                        RuntimeExecution(
                            "session-stop",
                            ["podman", "rm", "-f", execution_session_id],
                            exit_code=0,
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as tmp:
            runtime = LifecycleRuntime()
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-life",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=1),
                capabilities=RuntimeCapabilities(adapter="podman"),
                runtime_adapters={"podman": runtime},
                log_dir=tmp,
                state_path=os.path.join(tmp, "state.json"),
            )
            agent._report = lambda *args: None
            agent.execute_assignment(
                {
                    "command_id": "cmd-life",
                    "request_id": "req-life",
                    "execution_session_id": "sess-life",
                    "session_spec": {
                        "execution_session_id": "sess-life",
                        "mounts": [],
                        "runtime_adapter": "podman",
                        "isolation_policy": "image-stateful",
                    },
                    "command_spec": {"command": ["true"], "env": {}},
                }
            )
            agent.stop_session("sess-life")

            with open(os.path.join(tmp, "sess-life.log")) as fh:
                logged = fh.read()

        self.assertIn("runtime-adapter: podman", logged)
        self.assertIn("phase: session-stop", logged)
        self.assertIn("exec: podman rm -f sess-life", logged)

    def test_agent_persists_minimal_cleanup_spec_without_environment(self):
        class StatefulRuntime(object):
            def __init__(self):
                self.sessions = {}

            def start_session(self, session_spec):
                self.sessions[session_spec.execution_session_id] = session_spec
                return RuntimeResult(0, "", "")

            def run_command(self, _execution_session_id, _command, cwd=None, env=None):
                return RuntimeResult(0, "ok", "")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state.json")
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-persist",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=1),
                capabilities=RuntimeCapabilities(adapter="apptainer"),
                runtime_adapters={"apptainer": StatefulRuntime()},
                log_dir=tmp,
                state_path=state_path,
            )
            agent._report = lambda *args: None
            agent.execute_assignment(
                {
                    "command_id": "cmd-persist",
                    "request_id": "req-persist",
                    "execution_session_id": "sess-persist",
                    "session_spec": {
                        "execution_session_id": "sess-persist",
                        "image": "/images/workload.sif",
                        "mounts": [],
                        "env": {"SECRET_TOKEN": "must-not-persist"},
                        "runtime_adapter": "apptainer",
                        "isolation_policy": "image-stateful",
                        "state_layer_id": "layer-persist",
                    },
                    "command_spec": {"command": ["true"], "env": {}},
                }
            )

            with open(state_path) as fh:
                persisted_text = fh.read()
            persisted = json.loads(persisted_text)

        cleanup_spec = persisted["session_contexts"]["sess-persist"][
            "session_spec"
        ]
        self.assertEqual(cleanup_spec["state_layer_id"], "layer-persist")
        self.assertEqual(cleanup_spec["image"], "/images/workload.sif")
        self.assertNotIn("env", cleanup_spec)
        self.assertNotIn("must-not-persist", persisted_text)

    def test_agent_adopts_cleanup_spec_for_session_from_previous_process(self):
        class RestorableRuntime(object):
            def __init__(self):
                self.sessions = {}
                self.restored = []

            def restore_session(self, session_spec):
                self.sessions[session_spec.execution_session_id] = session_spec
                self.restored.append(session_spec)

            def stop_session(self, execution_session_id):
                session_spec = self.sessions.pop(execution_session_id)
                return RuntimeResult(
                    0,
                    "",
                    "",
                    executions=(
                        RuntimeExecution(
                            "session-cleanup",
                            ["cleanup-layer", session_spec.state_layer_id],
                            exit_code=0,
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state.json")
            with open(state_path, "w") as fh:
                json.dump(
                    {
                        "version": 2,
                        "local_sessions": ["sess-recovered"],
                        "session_contexts": {
                            "sess-recovered": {
                                "runtime_adapter": "apptainer",
                                "isolation_policy": "image-stateful",
                            }
                        },
                    },
                    fh,
                )
            runtime = RestorableRuntime()
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-restore",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=1),
                capabilities=RuntimeCapabilities(adapter="apptainer"),
                runtime_adapters={"apptainer": runtime},
                log_dir=tmp,
                state_path=state_path,
            )
            agent.stop_session(
                "sess-recovered",
                runtime_name="apptainer",
                session_spec={
                    "execution_session_id": "sess-recovered",
                    "image": "/images/workload.sif",
                    "mounts": [],
                    "isolation_policy": "image-stateful",
                    "state_layer_id": "layer-recovered",
                },
            )

            with open(state_path) as fh:
                final_state = json.load(fh)
            with open(os.path.join(tmp, "sess-recovered.log")) as fh:
                session_log = fh.read()

        self.assertEqual(len(runtime.restored), 1)
        self.assertEqual(
            runtime.restored[0].state_layer_id, "layer-recovered"
        )
        self.assertEqual(final_state["local_sessions"], [])
        self.assertIn("phase: session-cleanup", session_log)
        self.assertIn("exec: cleanup-layer layer-recovered", session_log)

    def test_concurrent_assignments_serialize_agent_state_snapshots(self):
        class ImmediateRuntime(object):
            def prepare_image(self, image_ref, _cache_config):
                return image_ref

            def start_session(self, _session_spec):
                return RuntimeResult(0, "", "")

            def run_command(self, _execution_session_id, command, cwd=None, env=None):
                return RuntimeResult(0, " ".join(command), "")

            def stop_session(self, _execution_session_id):
                return RuntimeResult(0, "", "")

        class CoordinatedDict(dict):
            def __init__(self, started, proceed, second_inserted, release_second):
                dict.__init__(self)
                self.started = started
                self.proceed = proceed
                self.second_inserted = second_inserted
                self.release_second = release_second
                self.block_once = True

            def __setitem__(self, key, value):
                dict.__setitem__(self, key, value)
                if key == "cmd-2":
                    self.second_inserted.set()
                    self.release_second.wait(2.0)

            def __iter__(self):
                iterator = dict.__iter__(self)
                if not self.block_once:
                    for key in iterator:
                        yield key
                    return
                self.block_once = False
                first = next(iterator)
                self.started.set()
                self.proceed.wait(2.0)
                yield first
                for key in iterator:
                    yield key

        def assignment(number):
            return {
                "command_id": "cmd-{}".format(number),
                "request_id": "req-{}".format(number),
                "execution_session_id": "sess-{}".format(number),
                "session_spec": {
                    "execution_session_id": "sess-{}".format(number),
                    "image": "/images/test.sif",
                    "mounts": [],
                    "runtime_adapter": "fake",
                },
                "command_spec": {"command": ["echo", str(number)]},
            }

        with tempfile.TemporaryDirectory() as tmp:
            started = threading.Event()
            proceed = threading.Event()
            second_inserted = threading.Event()
            release_second = threading.Event()
            reports = []
            errors = []
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-concurrent",
                pool=POOL,
                resources=NodeResources(cpu_cores=4, memory_gb=4),
                capabilities=RuntimeCapabilities(adapter="fake"),
                runtime_adapters={"fake": ImmediateRuntime()},
                log_dir=tmp,
                state_path=os.path.join(tmp, "agent-state.json"),
            )
            agent.running_commands = CoordinatedDict(
                started, proceed, second_inserted, release_second
            )
            agent._report = lambda command_id, *args, **kwargs: reports.append(command_id)

            def execute(item):
                try:
                    agent.execute_assignment(item)
                except BaseException as exc:
                    errors.append(exc)

            first = threading.Thread(target=execute, args=(assignment(1),))
            second = threading.Thread(target=execute, args=(assignment(2),))
            first.start()
            self.assertTrue(started.wait(2.0))
            second.start()
            inserted_without_lock = second_inserted.wait(0.2)
            proceed.set()
            if not inserted_without_lock:
                self.assertTrue(second_inserted.wait(2.0))
            release_second.set()
            first.join(3.0)
            second.join(3.0)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(sorted(reports), ["cmd-1", "cmd-2"])
            self.assertEqual(dict(agent.running_commands), {})
            with open(agent.state_path) as fh:
                persisted = json.load(fh)
            self.assertEqual(persisted["running_commands"], [])

    def test_agent_wire_contract_preserves_isolation_capabilities_and_state_layer(self):
        caps = RuntimeCapabilities(
            adapter="apptainer",
            isolation_policies=("default", "image-stateless", "image-stateful"),
            gpu_isolation_policies=("default", "image-stateless"),
            reduced_isolation_policies=("image-stateful",),
        )
        agent = NodeAgentClient(
            "http://127.0.0.1:1",
            node_id="node-wire",
            pool=POOL,
            resources=NodeResources(cpu_cores=1, memory_gb=1),
            capabilities=caps,
        )

        restored_caps = runtime_capabilities_from_dict(
            agent._capabilities_dict(caps)
        )
        session = agent._session_spec_from_dict({
            "execution_session_id": "sess-wire",
            "isolation_policy": "image-stateful",
            "state_layer_id": "layer-wire",
            "overlay_path": "/runtime/layer-wire",
        })

        self.assertEqual(restored_caps.isolation_policies, caps.isolation_policies)
        self.assertEqual(
            restored_caps.gpu_isolation_policies,
            caps.gpu_isolation_policies,
        )
        self.assertEqual(
            restored_caps.reduced_isolation_policies,
            caps.reduced_isolation_policies,
        )
        self.assertEqual(session.isolation_policy, "image-stateful")
        self.assertEqual(session.state_layer_id, "layer-wire")
        self.assertEqual(session.overlay_path, "/runtime/layer-wire")

    def test_agent_registration_and_execution_preserve_multi_runtime_selection(self):
        runtime_capabilities = {
            "apptainer": RuntimeCapabilities(
                adapter="apptainer",
                isolation_policies=("default", "image-stateless"),
                gpu_enforcement=EnforcementMode.ADVISORY,
            ),
            "bubblewrap": RuntimeCapabilities(
                adapter="bubblewrap",
                image_formats=("host",),
                isolation_policies=("default", "host-stateless"),
                gpu_enforcement=EnforcementMode.ADVISORY,
            ),
        }
        apptainer = ApptainerRuntimeAdapter(
            binary="/runtime/apptainer",
            overlay_supported=False,
        )
        bubblewrap = BubbleWrapRuntimeAdapter(
            binary="/runtime/bwrap",
            overlay_supported=False,
        )
        launches = {"apptainer": [], "bubblewrap": []}

        class ImmediateHandle(object):
            def __init__(self, result):
                self.result = result

            def wait(self, **_kwargs):
                return self.result

            def terminate_tree(self):
                return None

        def capture(adapter_name, adapter):
            def start(execution_session_id, command, cwd=None, env=None, command_id=None):
                launch = adapter.build_run_command(
                    adapter.sessions[execution_session_id],
                    ContainerCommandSpec(command, cwd=cwd, env=env),
                )
                launches[adapter_name].append(list(launch.argv))
                return ImmediateHandle(RuntimeResult(0, adapter_name, ""))

            return start

        apptainer.start_command_process = capture("apptainer", apptainer)
        bubblewrap.start_command_process = capture("bubblewrap", bubblewrap)

        def assignment(number, runtime_adapter, isolation_policy):
            return {
                "command_id": "cmd-{}".format(number),
                "request_id": "req-{}".format(number),
                "execution_session_id": "sess-{}".format(number),
                "session_spec": {
                    "execution_session_id": "sess-{}".format(number),
                    "image": "docker://example/workload:latest",
                    "mounts": [],
                    "runtime_adapter": runtime_adapter,
                    "isolation_policy": isolation_policy,
                },
                "command_spec": {"command": ["true"]},
            }

        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-multi",
                pool=POOL,
                resources=NodeResources(cpu_cores=2, memory_gb=2),
                runtime_capabilities=runtime_capabilities,
                runtime_adapters={
                    "apptainer": apptainer,
                    "bubblewrap": bubblewrap,
                },
                log_dir=tmp,
                state_path=os.path.join(tmp, "agent-state.json"),
            )
            reports = []
            agent._report = lambda command_id, *args: reports.append(command_id)

            registration = agent._registration_payload()
            self.assertEqual(registration["capabilities"]["adapter"], "multi")
            self.assertEqual(
                set(registration["runtime_capabilities"]),
                {"apptainer", "bubblewrap"},
            )

            agent.execute_assignment(
                assignment(1, "bubblewrap", "host-stateless")
            )
            agent.execute_assignment(
                assignment(2, "apptainer", "image-stateless")
            )

        self.assertEqual(reports, ["cmd-1", "cmd-2"])
        self.assertEqual(launches["bubblewrap"][0][0], "/runtime/bwrap")
        self.assertIn("--ro-bind", launches["bubblewrap"][0])
        self.assertEqual(
            launches["apptainer"][0][:2], ["/runtime/apptainer", "exec"]
        )

    def test_http_registration_preserves_per_adapter_capabilities(self):
        ctrl = CAPEController()
        service = CAPEControllerService(
            controller=ctrl, schedule_interval=10.0
        ).start()
        runtime_capabilities = {
            "local-process": RuntimeCapabilities(
                adapter="local-process",
                image_formats=("default",),
                isolation_policies=("default", "none"),
                gpu_enforcement=EnforcementMode.NONE,
            ),
            "bubblewrap": RuntimeCapabilities(
                adapter="bubblewrap",
                image_formats=("host",),
                isolation_policies=("default", "host-stateless"),
                gpu_enforcement=EnforcementMode.ADVISORY,
            ),
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-wire-multi",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=2, memory_gb=4),
                    runtime_capabilities=runtime_capabilities,
                    log_dir=tmp,
                )

                status = agent.register()

            node = ctrl.get_node("node-wire-multi")
            self.assertEqual(node.capabilities.adapter, "multi")
            self.assertEqual(
                set(node.runtime_capabilities),
                {"local-process", "bubblewrap"},
            )
            self.assertEqual(
                set(status["runtime_capabilities"]),
                {"local-process", "bubblewrap"},
            )
        finally:
            service.stop()

    def test_controller_schedules_command_assignments_without_running_inline(self):
        ctrl = CAPEController()
        register_local_node(ctrl)
        first = ctrl.submit_workload(local_spec(command=(sys.executable, "-c", "import time; time.sleep(2)"), session_id="a"))
        second = ctrl.submit_workload(local_spec(command=(sys.executable, "-c", "print(2)"), session_id="b"))
        start = time.time()
        scheduled = ctrl.schedule_once()
        elapsed = time.time() - start

        self.assertEqual(scheduled, 2)
        self.assertLess(elapsed, 0.5)
        self.assertEqual(ctrl.get_request(first.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(second.request_id).state, RequestState.RUNNING)
        self.assertEqual(len(ctrl.command_assignments), 2)
        self.assertEqual(len(ctrl.node_command_queues["node-a"]), 2)

    def test_http_service_and_separate_node_agent_execute_workload(self):
        ctrl = CAPEController()
        service = CAPEControllerService(controller=ctrl, schedule_interval=0.01).start()
        stop = threading.Event()
        try:
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as workers:
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
                workers.callback(stop_test_agent_workers, agent, stop, thread)
                out = io.StringIO()
                rc = cape_main([
                    "run",
                    "--controller-url", service.url,
                    "--pool", POOL,
                    "--user", "alice",
                    "--session-id", "task-a",
                    "--wait",
                    "--timeout", "5",
                    "--",
                    sys.executable,
                    "-c",
                    "print(36)",
                ], stdout=out)
                stop.set()
                agent.join(timeout=2.0)
                payload = json.loads(out.getvalue())
                self.assertEqual(rc, 0)
                self.assertEqual(payload["state"], "COMPLETED")
                self.assertEqual(payload["exit_code"], 0)
                self.assertEqual(payload["log_handle"], "/v1/requests/{}/logs".format(payload["request_id"]))
                logs = ctrl.logs_for_request(payload["request_id"])
                self.assertIn("36", logs["stdout"])
                self.assertTrue(os.path.exists(logs["log_path"]))
        finally:
            stop.set()
            service.stop()

    def test_http_agent_reuses_session_after_workload_failure(self):
        ctrl = CAPEController()
        service = CAPEControllerService(
            controller=ctrl, schedule_interval=0.01
        ).start()
        stop = threading.Event()
        thread = None
        try:
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as workers:
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-failure-reuse",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=4, memory_gb=16),
                    capabilities=RuntimeCapabilities(
                        adapter="local-process",
                        image_formats=("default",),
                        node_local_cache=True,
                    ),
                    runtime_adapters={
                        "local-process": LocalProcessRuntimeAdapter()
                    },
                    log_dir=tmp,
                )
                agent.register()

                def heartbeat_loop():
                    while not stop.is_set():
                        agent.heartbeat_once(background=True)
                        time.sleep(0.02)

                thread = threading.Thread(target=heartbeat_loop)
                thread.start()
                workers.callback(stop_test_agent_workers, agent, stop, thread)

                failed_out = io.StringIO()
                failed_rc = cape_main(
                    [
                        "run",
                        "--controller-url",
                        service.url,
                        "--pool",
                        POOL,
                        "--user",
                        "alice",
                        "--session-id",
                        "failure-reuse",
                        "--wait",
                        "--timeout",
                        "5",
                        "--",
                        sys.executable,
                        "-c",
                        "raise SystemExit(7)",
                    ],
                    stdout=failed_out,
                )
                failed = json.loads(failed_out.getvalue())
                self.assertEqual(failed_rc, 7)
                self.assertEqual(failed["state"], "FAILED")
                self.assertEqual(failed["reason"], "command_failed")
                self.assertEqual(failed["failure_kind"], "workload")
                self.assertTrue(failed["keep_session_on_failure"])
                execution_session_id = failed["execution_session_id"]
                self.assertEqual(
                    ctrl.get_session(execution_session_id).state,
                    SessionState.CACHED_IDLE,
                )
                ctrl.assert_invariants()

                succeeded_out = io.StringIO()
                succeeded_rc = cape_main(
                    [
                        "run",
                        "--controller-url",
                        service.url,
                        "--pool",
                        POOL,
                        "--user",
                        "alice",
                        "--session-id",
                        "failure-reuse",
                        "--wait",
                        "--timeout",
                        "5",
                        "--",
                        sys.executable,
                        "-c",
                        "print('reused')",
                    ],
                    stdout=succeeded_out,
                )
                succeeded = json.loads(succeeded_out.getvalue())
                self.assertEqual(succeeded_rc, 0)
                self.assertEqual(succeeded["state"], "COMPLETED")
                self.assertEqual(succeeded["execution_session_id"], execution_session_id)
                self.assertEqual(succeeded["failure_kind"], "workload")

                opt_out_out = io.StringIO()
                opt_out_rc = cape_main(
                    [
                        "run",
                        "--controller-url",
                        service.url,
                        "--pool",
                        POOL,
                        "--user",
                        "alice",
                        "--session-id",
                        "failure-opt-out",
                        "--suspend-on-failure",
                        "--wait",
                        "--timeout",
                        "5",
                        "--",
                        sys.executable,
                        "-c",
                        "raise SystemExit(3)",
                    ],
                    stdout=opt_out_out,
                )
                opt_out = json.loads(opt_out_out.getvalue())
                self.assertEqual(opt_out_rc, 3)
                self.assertFalse(opt_out["keep_session_on_failure"])
                self.assertIn(
                    ctrl.get_session(opt_out["execution_session_id"]).state,
                    (SessionState.STOPPED, SessionState.SUSPENDED),
                )

                after_opt_out_out = io.StringIO()
                self.assertEqual(
                    cape_main(
                        [
                            "run",
                            "--controller-url",
                            service.url,
                            "--pool",
                            POOL,
                            "--user",
                            "alice",
                            "--session-id",
                            "failure-opt-out",
                            "--wait",
                            "--timeout",
                            "5",
                            "--",
                            sys.executable,
                            "-c",
                            "print('cold')",
                        ],
                        stdout=after_opt_out_out,
                    ),
                    0,
                )
                after_opt_out = json.loads(
                    after_opt_out_out.getvalue()
                )
                self.assertEqual(
                    after_opt_out["execution_session_id"], opt_out["execution_session_id"]
                )
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=2.0)
            if "agent" in locals():
                agent.join(timeout=2.0)
            service.stop()

    def test_node_agent_rejects_cached_identity_without_recovery_metadata(self):
        class RecordingRuntime(object):
            def __init__(self):
                self.sessions = {}
                self.starts = 0

            def start_session(self, session_spec):
                self.starts += 1
                self.sessions[session_spec.execution_session_id] = session_spec
                return RuntimeResult(0, "", "")

            def run_command(self, execution_session_id, command, cwd=None, env=None):
                if execution_session_id not in self.sessions:
                    return RuntimeResult(127, "", "unknown runtime session: " + execution_session_id)
                return RuntimeResult(0, "ok", "")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state.json")
            with open(state_path, "w") as fh:
                json.dump({"version": 2, "local_sessions": ["sess-stale"]}, fh)
            runtime = RecordingRuntime()
            from pycape.workspace_fs import WorkspaceError
            with self.assertRaisesRegex(WorkspaceError, "cached execution metadata is missing"):
                NodeAgentClient("http://127.0.0.1:1", node_id="node-a", pool=POOL,
                    resources=NodeResources(cpu_cores=1, memory_gb=1, gpu_uuids=()),
                    capabilities=RuntimeCapabilities(adapter="apptainer"), runtime_adapters={"apptainer": runtime},
                    log_dir=tmp, state_path=state_path)
            self.assertEqual(runtime.starts, 0)

    def test_http_service_with_sqlite_journal_accepts_threaded_events(self):
        stop = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            journal = SQLiteJournal(os.path.join(tmp, "journal.sqlite"))
            ctrl = CAPEController(journal=journal)
            service = CAPEControllerService(controller=ctrl, schedule_interval=0.01).start()
            try:
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-sqlite",
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
                out = io.StringIO()
                rc = cape_main([
                    "run", "--controller-url", service.url, "--pool", POOL,
                    "--user", "alice", "--session-id", "task-a",
                    "--wait", "--timeout", "5",
                    "--", sys.executable, "-c", "print(49)",
                ], stdout=out)
                stop.set()
                thread.join(timeout=2.0)
                payload = json.loads(out.getvalue())
                self.assertEqual(rc, 0)
                self.assertEqual(payload["state"], "COMPLETED")
                self.assertEqual(payload["exit_code"], 0)
                self.assertGreater(journal.count(), 0)
            finally:
                stop.set()
                service.stop()
                journal.close()

    def test_node_agent_pulls_cancel_and_reaps_running_process(self):
        ctrl = CAPEController()
        service = CAPEControllerService(controller=ctrl, schedule_interval=10.0).start()
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
                req = ctrl.submit_workload(local_spec(command=(sys.executable, "-c", "import time; time.sleep(30)"), session_id="cancel"))
                ctrl.schedule_once()
                agent.heartbeat_once(background=True)
                deadline = time.time() + 5.0
                while not agent.running_processes and time.time() < deadline:
                    time.sleep(0.02)
                self.assertTrue(agent.running_processes)
                ctrl.cancel_request(req.request_id)
                agent.heartbeat_once(background=False)
                agent.join(timeout=3.0)
                self.assertEqual(ctrl.request_status(req.request_id)["state"], "CANCELLED")
                self.assertEqual(agent.running_processes, {})
        finally:
            service.stop()

    def test_node_agent_registers_container_handle_before_waiting(self):
        class BlockingHandle(object):
            def __init__(self):
                self.terminated = threading.Event()

            def wait(self, **_kwargs):
                self.terminated.wait(3.0)
                return RuntimeResult(143, "", "cancelled")

            def terminate_tree(self):
                self.terminated.set()

        class HandleRuntime(object):
            def __init__(self):
                self.sessions = {}
                self.handle = BlockingHandle()

            def start_session(self, session_spec):
                self.sessions[session_spec.execution_session_id] = session_spec
                return RuntimeResult(0, "", "")

            def start_command_process(self, *_args, **_kwargs):
                return self.handle

            def stop_session(self, execution_session_id):
                self.sessions.pop(execution_session_id, None)
                return RuntimeResult(0, "", "")

        runtime = HandleRuntime()
        assignment = {
            "command_id": "cmd-container-cancel",
            "request_id": "req-container-cancel",
            "execution_session_id": "sess-container-cancel",
            "session_spec": {
                "execution_session_id": "sess-container-cancel",
                "image": "docker://example/image:latest",
                "mounts": [],
                "runtime_adapter": "container",
                "isolation_policy": "image-stateless",
            },
            "command_spec": {"command": ["sleep", "300"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-container-cancel",
                pool=POOL,
                resources=NodeResources(cpu_cores=2, memory_gb=2),
                capabilities=RuntimeCapabilities(adapter="container"),
                runtime_adapters={"container": runtime},
                log_dir=tmp,
                state_path=os.path.join(tmp, "agent-state.json"),
            )
            agent._report = lambda *_args, **_kwargs: None
            worker = threading.Thread(
                target=agent.execute_assignment, args=(assignment,)
            )
            worker.start()
            deadline = time.time() + 2.0
            while not agent.running_processes and time.time() < deadline:
                time.sleep(0.01)
            self.assertIn("cmd-container-cancel", agent.running_processes)
            agent.cancel_assignment(assignment)
            worker.join(timeout=3.0)
            self.assertFalse(worker.is_alive())
            self.assertTrue(runtime.handle.terminated.is_set())
            self.assertEqual(agent.running_processes, {})

    def test_profile_resolution_cache_locality_and_status_schema(self):
        registry = ProfileRegistry({
            "torch": {
                "image": "docker://example/torch:latest",
                "default_mounts": [{"source": "/datasets", "target": "/datasets", "read_only": True}],
                "runtime_preference": "local-process",
                "gpu_required": True,
                "env": {"FROM_PROFILE": "1"},
            }
        })
        ctrl = CAPEController(policy=SchedulerPolicy(profile_registry=registry))
        register_local_node(ctrl, node_id="node-cold", gpus=1)
        register_local_node(ctrl, node_id="node-warm", gpus=1, cached_images=("docker://example/torch:latest",))
        workload = WorkloadSpec(
            pool=POOL,
            user_id="alice",
            session_id="task-a",
            command=("echo", "ok"),
            profile="torch",
            resources=ResourceRequest(
                gpus=0,
                ports=(8080,),
                locality=None,
                command_timeout_seconds=30,
            ),
            keep_session_on_failure=False,
        )
        req = ctrl.submit_workload(workload)
        ctrl.schedule_once()
        status = ctrl.request_status(req.request_id)
        session = ctrl.get_session(status["execution_session_id"])

        self.assertEqual(status["node_id"], "node-warm")
        self.assertEqual(status["image"], "docker://example/torch:latest")
        self.assertEqual(status["profile"], "torch")
        self.assertEqual(status["session_id"], "task-a")
        self.assertNotIn("workspace_id", status)
        self.assertEqual(status["ports"], [8080])
        self.assertEqual(status["runtime_adapter"], "local-process")
        self.assertEqual(ctrl.get_request(req.request_id).spec.resources.command_timeout_seconds, 30)
        self.assertFalse(status["keep_session_on_failure"])
        self.assertEqual(status["gpu_uuids"], ["node-warm-gpu-0"])
        self.assertIn(MountSpec("/datasets", "/datasets", True), session.mounts)

    def test_locality_can_target_allocation_and_library_http_drains(self):
        ctrl = CAPEController()
        register_local_node(ctrl, node_id="node-a", gpus=1, allocation_id="alloc-a")
        register_local_node(ctrl, node_id="node-b", gpus=1, allocation_id="alloc-b")
        workload = WorkloadSpec(
            pool=POOL,
            user_id="alice",
            session_id="task-a",
            command=("echo", "ok"),
            resources=ResourceRequest(gpus=1, locality="alloc-b"),
        )
        req = ctrl.submit_workload(workload)
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).node_id, "node-b")

        service = CAPEControllerService(controller=ctrl).start()
        try:
            from pycape.cli import _request
            payload = _request("POST", service.url, "/v1/admin/drain-allocation", payload={"allocation_id": "alloc-b"})
            self.assertEqual(payload["state"], "DRAINING")
            self.assertEqual(ctrl.get_node("node-b").state, NodeState.DRAINING)
        finally:
            service.stop()

    def test_reconciling_session_resolves_to_failed_and_releases_lease_when_agent_reports_missing(self):
        journal = __import__("pycape").MemoryJournal()
        ctrl = CAPEController(journal=journal)
        register_local_node(ctrl, gpus=1)
        req = ctrl.submit_workload(local_spec(gpus=1, session_id="reconcile"))
        ctrl.schedule_once()
        clock = FakeClock()
        replayed = CAPEController.reconcile_from_journal(
            journal,
            clock=clock,
            policy=SchedulerPolicy(
                reconcile_missing_grace_seconds=1,
                reconcile_missing_heartbeats=2,
            ),
        )
        self.assertEqual(replayed.get_request(req.request_id).state, RequestState.RECONCILING)
        self.assertEqual(replayed.active_gpu_lease_count(), 1)

        replayed.node_heartbeat("node-a", status={"authoritative": True, "sessions": []})
        self.assertEqual(
            replayed.get_request(req.request_id).state,
            RequestState.RECONCILING,
        )
        self.assertEqual(replayed.active_gpu_lease_count(), 1)

        clock.advance(1.1)
        replayed.node_heartbeat("node-a", status={"authoritative": True, "sessions": []})
        self.assertEqual(replayed.get_request(req.request_id).state, RequestState.FAILED)
        self.assertEqual(replayed.get_session(replayed.get_request(req.request_id).execution_session_id).state, SessionState.FAILED)
        self.assertEqual(replayed.lease_table, {})

    def test_non_reusable_request_releases_session_after_command_completion(self):
        ctrl = CAPEController()
        register_local_node(ctrl, gpus=1)
        req = ctrl.submit_workload(local_spec(gpus=1, reuse=False, session_id="no-reuse"))
        ctrl.schedule_once()
        command = ctrl.node_heartbeat("node-a", status={"authoritative": True, "sessions": []})["commands"][0]
        ctrl.report_command_result("node-a", command["command_id"], exit_code=0, stdout="ok")
        status = ctrl.request_status(req.request_id)
        self.assertEqual(status["state"], "COMPLETED")
        self.assertEqual(ctrl.get_session(status["execution_session_id"]).state, SessionState.STOPPED)
        self.assertEqual(ctrl.lease_table, {})

    def test_pool_utilization_and_exporters_are_safe(self):
        ctrl = CAPEController()
        register_local_node(ctrl, gpus=2, allocation_id="alloc-a")
        ctrl.submit_workload(local_spec(gpus=1, session_id="util"))
        ctrl.schedule_once()
        summary = ctrl.pool_utilization(pool=POOL)
        self.assertEqual(summary["gpus_total"], 2)
        self.assertEqual(summary["gpus_leased"], 1)
        self.assertEqual(
            summary["node_states"][0]["allocation_id"], "alloc-a"
        )
        self.assertEqual(
            summary["node_states"][0]["active_session_count"], 1
        )
        self.assertEqual(
            len(summary["node_states"][0]["active_sessions"]), 1
        )
        self.assertEqual(
            summary["node_states"][0]["nonterminal_requests"], 1
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            JsonlExporter(path).export("pool", summary)
            self.assertTrue(os.path.exists(path))
        exporter = WandbExporter(token_path="/path/does/not/exist", enabled=True)
        self.assertFalse(exporter.export("pool", summary))
        self.assertEqual(exporter.last_payload[0], "pool")
        self.assertIsNotNone(exporter.last_error)

    def test_pool_utilization_excludes_removed_node_capacity(self):
        ctrl = CAPEController()
        register_local_node(ctrl, gpus=2)
        ctrl.remove_node("node-a")

        summary = ctrl.pool_utilization(pool=POOL)

        self.assertEqual(summary["nodes"], 0)
        self.assertEqual(summary["gpus_total"], 0)
        self.assertEqual(summary["gpus_leased"], 0)
        self.assertEqual(summary["cpu_total"], 0)
        self.assertEqual(summary["cpu_used"], 0)
        self.assertEqual(summary["memory_total"], 0)
        self.assertEqual(summary["memory_used"], 0)

    def test_wandb_exporter_uses_optional_real_module_and_token_file(self):
        class FakeRun(object):
            def __init__(self):
                self.logged = []
            def log(self, payload):
                self.logged.append(payload)
        class FakeWandb(object):
            def __init__(self):
                self.init_kwargs = None
                self.run = FakeRun()
            def init(self, **kwargs):
                self.init_kwargs = kwargs
                return self.run

        fake = FakeWandb()
        old_module = sys.modules.get("wandb")
        old_key = os.environ.get("WANDB_API_KEY")
        sys.modules["wandb"] = fake
        try:
            if "WANDB_API_KEY" in os.environ:
                del os.environ["WANDB_API_KEY"]
            with tempfile.TemporaryDirectory() as tmp:
                token_path = os.path.join(tmp, "wandb")
                with open(token_path, "w") as fh:
                    fh.write("secret-token")
                exporter = WandbExporter(token_path=token_path, enabled=True, project="cape-tests")
                self.assertTrue(exporter.export("pool", {"gpus_total": 2}))
                self.assertEqual(os.environ.get("WANDB_API_KEY"), "secret-token")
                self.assertEqual(fake.init_kwargs["project"], "cape-tests")
                self.assertEqual(fake.run.logged[0]["metric_name"], "pool")
                self.assertEqual(fake.run.logged[0]["gpus_total"], 2)
        finally:
            if old_module is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = old_module
            if old_key is None:
                os.environ.pop("WANDB_API_KEY", None)
            else:
                os.environ["WANDB_API_KEY"] = old_key

    def test_metrics_collector_reads_nvml_when_available(self):
        class Util(object):
            gpu = 37
            memory = 11
        class Mem(object):
            total = 100
            used = 40
            free = 60
        class FakeNvml(object):
            NVML_TEMPERATURE_GPU = 0
            def nvmlInit(self):
                self.initialized = True
            def nvmlDeviceGetCount(self):
                return 1
            def nvmlDeviceGetHandleByIndex(self, index):
                return "handle-{}".format(index)
            def nvmlDeviceGetUUID(self, handle):
                return b"GPU-fake"
            def nvmlDeviceGetName(self, handle):
                return b"Fake GPU"
            def nvmlDeviceGetUtilizationRates(self, handle):
                return Util()
            def nvmlDeviceGetMemoryInfo(self, handle):
                return Mem()
            def nvmlDeviceGetTemperature(self, handle, sensor):
                return 42

        metrics = MetricsCollector(nvml=FakeNvml()).collect_gpu_metrics()
        self.assertEqual(metrics[0]["uuid"], "GPU-fake")
        self.assertEqual(metrics[0]["gpu_utilization_percent"], 37)
        self.assertEqual(metrics[0]["memory_used_bytes"], 40)
        self.assertEqual(metrics[0]["temperature_c"], 42)

    def test_node_agent_status_includes_gpu_metrics_when_collector_reports_them(self):
        class FakeCollector(object):
            def collect_gpu_metrics(self):
                return [{"uuid": "GPU-fake", "gpu_utilization_percent": 5}]

        ctrl = CAPEController()
        service = CAPEControllerService(controller=ctrl, schedule_interval=0.1).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=4, memory_gb=16, gpu_uuids=("GPU-fake",)),
                    capabilities=RuntimeCapabilities(adapter="local-process", image_formats=("default",), node_local_cache=True),
                    log_dir=tmp,
                    metrics_collector=FakeCollector(),
                )
                agent.register()
                agent.heartbeat_once(background=False)
                status = ctrl.node_status("node-a")
                self.assertEqual(status["status"]["gpu_metrics"][0]["uuid"], "GPU-fake")
        finally:
            service.stop()

    def test_round31_gpu_workload_failure_reuses_session_when_configured(self):
        if os.environ.get("PYCAPE_ROUND31_GPU_TEST") != "1":
            self.skipTest("PYCAPE_ROUND31_GPU_TEST is not configured")
        gpu_uuid = os.environ.get("PYCAPE_ROUND31_GPU_UUID")
        if not gpu_uuid:
            self.skipTest("PYCAPE_ROUND31_GPU_UUID is required")

        ctrl = CAPEController()
        service = CAPEControllerService(
            controller=ctrl, schedule_interval=0.01
        ).start()
        stop = threading.Event()
        thread = None
        try:
            with tempfile.TemporaryDirectory() as log_dir:
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-round31-gpu",
                    pool=POOL,
                    resources=NodeResources(
                        cpu_cores=8,
                        memory_gb=64,
                        gpu_uuids=(gpu_uuid,),
                    ),
                    capabilities=RuntimeCapabilities(
                        adapter="local-process",
                        image_formats=("default",),
                        isolation_policies=("default", "none"),
                        gpu_enforcement=EnforcementMode.NONE,
                        node_local_cache=True,
                    ),
                    runtime_adapters={
                        "local-process": LocalProcessRuntimeAdapter()
                    },
                    allocation_id="round31-gpu-live",
                    log_dir=log_dir,
                    metrics_collector=MetricsCollector(enabled=False),
                )
                agent.register()

                def heartbeat_loop():
                    while not stop.is_set():
                        agent.heartbeat_once(background=True)
                        time.sleep(0.02)

                thread = threading.Thread(target=heartbeat_loop)
                thread.start()

                failed_out = io.StringIO()
                self.assertEqual(
                    cape_main(
                        [
                            "run",
                            "--controller-url",
                            service.url,
                            "--pool",
                            POOL,
                            "--user",
                            "alice",
                            "--session-id",
                            "round31-gpu",
                            "--gpus",
                            "1",
                            "--runtime-adapter",
                            "local-process",
                            "--isolation-policy",
                            "none",
                            "--wait",
                            "--timeout",
                            "30",
                            "--",
                            sys.executable,
                            "-c",
                            "raise SystemExit(7)",
                        ],
                        stdout=failed_out,
                    ),
                    7,
                )
                failed = json.loads(failed_out.getvalue())
                execution_session_id = failed["execution_session_id"]
                self.assertEqual(failed["failure_kind"], "workload")
                self.assertEqual(
                    ctrl.get_session(execution_session_id).state,
                    SessionState.CACHED_IDLE,
                )
                self.assertEqual(ctrl.lease_table, {})

                success_out = io.StringIO()
                self.assertEqual(
                    cape_main(
                        [
                            "run",
                            "--controller-url",
                            service.url,
                            "--pool",
                            POOL,
                            "--user",
                            "alice",
                            "--session-id",
                            "round31-gpu",
                            "--gpus",
                            "1",
                            "--runtime-adapter",
                            "local-process",
                            "--isolation-policy",
                            "none",
                            "--wait",
                            "--timeout",
                            "30",
                            "--",
                            sys.executable,
                            "-c",
                            (
                                "import os,torch; "
                                "assert torch.cuda.is_available(); "
                                "assert torch.cuda.device_count()==1; "
                                "x=torch.arange(8,device='cuda').sum(); "
                                "print(os.environ['CUDA_VISIBLE_DEVICES']); "
                                "print(int(x.item()))"
                            ),
                        ],
                        stdout=success_out,
                    ),
                    0,
                )
                succeeded = json.loads(success_out.getvalue())
                self.assertEqual(succeeded["execution_session_id"], execution_session_id)
                self.assertEqual(succeeded["gpu_uuids"], [gpu_uuid])
                logs = ctrl.logs_for_request(succeeded["request_id"])
                self.assertIn(gpu_uuid, logs["stdout"])
                self.assertIn("28", logs["stdout"])
                ctrl.assert_invariants()

                ctrl.remove_allocation("round31-gpu-live", force=True)
                agent.stop_session(execution_session_id)
                self.assertEqual(ctrl.pool_utilization(POOL)["nodes"], 0)
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=2.0)
            if "agent" in locals():
                agent.join(timeout=2.0)
            service.stop()

    def test_service_agent_apptainer_full_stack_runs_selected_gpu_when_configured(self):
        if os.environ.get("PYCAPE_FULLSTACK_APPTAINER_TEST") != "1":
            self.skipTest("PYCAPE_FULLSTACK_APPTAINER_TEST is not configured")
        image = os.environ.get("PYCAPE_FULLSTACK_APPTAINER_TEST_IMAGE") or os.environ.get("PYCAPE_APPTAINER_TEST_IMAGE")
        gpu_uuid = os.environ.get("PYCAPE_APPTAINER_TEST_GPU_UUID")
        if not image or not gpu_uuid:
            self.skipTest("PYCAPE_APPTAINER_TEST_IMAGE and PYCAPE_APPTAINER_TEST_GPU_UUID are required")
        binary = os.environ.get("PYCAPE_APPTAINER_BINARY", "apptainer")
        cache_dir = os.environ.get("PYCAPE_APPTAINER_CACHE_DIR")
        tmp_dir = os.environ.get("PYCAPE_APPTAINER_TMP_DIR")
        for path in (cache_dir, tmp_dir):
            if path and not os.path.isdir(path):
                os.makedirs(path)

        ctrl = CAPEController()
        service = CAPEControllerService(controller=ctrl, schedule_interval=0.01).start()
        stop = threading.Event()
        try:
            with tempfile.TemporaryDirectory() as log_dir:
                runtime = ApptainerRuntimeAdapter(binary=binary, warm_sessions=False, cache_dir=cache_dir, tmp_dir=tmp_dir)
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=8, memory_gb=64, gpu_uuids=(gpu_uuid,)),
                    capabilities=RuntimeCapabilities(
                        adapter="apptainer",
                        image_formats=("default", "docker", "sif", "sandbox", "oci"),
                        nested_container=True,
                        can_bind_mounts=True,
                        can_select_gpus=True,
                        gpu_enforcement=EnforcementMode.ADVISORY,
                        node_local_cache=True,
                    ),
                    runtime_adapters={"apptainer": runtime},
                    runtime_cache_config=cache_dir or "default",
                    log_dir=log_dir,
                    metrics_collector=MetricsCollector(enabled=False),
                )
                agent.register()

                def heartbeat_loop():
                    while not stop.is_set():
                        agent.heartbeat_once(background=True)
                        time.sleep(0.05)

                thread = threading.Thread(target=heartbeat_loop)
                thread.daemon = True
                thread.start()
                out = io.StringIO()
                rc = cape_main([
                    "run",
                    "--controller-url", service.url,
                    "--pool", POOL,
                    "--user", "alice",
                    "--session-id", "task-a",
                    "--image", image,
                    "--gpus", "1",
                    "--wait",
                    "--timeout", "120",
                    "--",
                    "sh", "-lc", "printf '%s' \"$CUDA_VISIBLE_DEVICES\"; test -e /dev/nvidiactl",
                ], stdout=out)
                stop.set()
                agent.join(timeout=5.0)
                payload = json.loads(out.getvalue())
                self.assertEqual(rc, 0)
                self.assertEqual(payload["state"], "COMPLETED")
                self.assertEqual(payload["gpu_uuids"], [gpu_uuid])
                logs = ctrl.logs_for_request(payload["request_id"])
                self.assertEqual(logs["stdout"], gpu_uuid)
                self.assertTrue(os.path.exists(logs["log_path"]))
        finally:
            stop.set()
            service.stop()


if __name__ == "__main__":
    unittest.main()
