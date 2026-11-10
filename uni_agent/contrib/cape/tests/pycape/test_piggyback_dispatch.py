# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import (
    CAPEController,
    CAPEControllerService,
    EnforcementMode,
    NodeAgentClient,
    NodeResources,
    NodeState,
    RequestState,
    ResourceRequest,
    RuntimeCapabilities,
    WorkloadSpec,
)


POOL = "pool-a"


def workload(name, cpu_cores=1):
    return WorkloadSpec(
        pool=POOL,
        user_id="alice",
        command=(sys.executable, "-c", "print({!r})".format(name)),
        resources=ResourceRequest(
            gpus=0,
            cpu_cores=cpu_cores,
            memory_gb=1,
            max_duration_seconds=30,
        ),
        session_id=name,
        reuse=False,
    )


def register_node(
    controller,
    cpu_cores=1,
    allocation_id="alloc-a",
    agent_epoch=None,
):
    return controller.register_node(
        node_id="node-a",
        pool=POOL,
        resources=NodeResources(cpu_cores=cpu_cores, memory_gb=64),
        capabilities=RuntimeCapabilities(
            adapter="local-process",
            image_formats=("default", "docker"),
            isolation_policies=("default", "none"),
            can_select_gpus=True,
            gpu_enforcement=EnforcementMode.HARD,
        ),
        allocation_id=allocation_id,
        agent_epoch=agent_epoch,
    )


def prepare_saturated_controller(cpu_cores=1, successor_count=1):
    controller = CAPEController()
    register_node(controller, cpu_cores=cpu_cores)
    first = controller.submit_workload(
        workload("first", cpu_cores=cpu_cores)
    )
    successors = [
        controller.submit_workload(workload("next-{}".format(index)))
        for index in range(successor_count)
    ]
    controller.schedule_once()
    response = controller.node_heartbeat(
        "node-a",
        status={"authoritative": True, "sessions": []},
        max_commands=1,
    )
    return controller, first, successors, response["commands"][0]


def report_payload(command, accept_commands=True, max_commands=8):
    payload = {
        "node_id": "node-a",
        "command_id": command["command_id"],
        "exit_code": 0,
        "stdout": "done",
        "stderr": "",
    }
    if accept_commands is not None:
        payload["accept_commands"] = bool(accept_commands)
    if max_commands is not None:
        payload["max_commands"] = max_commands
    return payload


def run_commands(payload):
    return [
        command
        for command in payload.get("commands", [])
        if command.get("action", "run") == "run"
    ]


class ControllerReportDispatchTests(unittest.TestCase):
    def test_completion_report_returns_saturated_successor(self):
        controller, first, successors, command = (
            prepare_saturated_controller()
        )
        service = CAPEControllerService(
            controller=controller, schedule_interval=100.0
        ).start()
        try:
            agent = NodeAgentClient(
                service.url,
                node_id="node-a",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=64),
            )

            response = agent._post(
                "/v1/agent/report-command",
                report_payload(command),
            )
        finally:
            service.stop()

        self.assertEqual(response["state"], "COMPLETED")
        self.assertEqual(response["request_id"], first.request_id)
        self.assertEqual(len(run_commands(response)), 1)
        self.assertEqual(
            run_commands(response)[0]["request_id"],
            successors[0].request_id,
        )
        self.assertEqual(
            controller.get_request(successors[0].request_id).state,
            RequestState.RUNNING,
        )
        self.assertEqual(
            controller.node_heartbeat(
                "node-a",
                status={"authoritative": True, "sessions": []},
            )["commands"],
            [],
        )

    def test_report_respects_max_commands_and_leaves_remainder_for_heartbeat(self):
        controller, _, successors, command = prepare_saturated_controller(
            cpu_cores=8, successor_count=6
        )
        service = CAPEControllerService(
            controller=controller, schedule_interval=100.0
        ).start()
        try:
            agent = NodeAgentClient(
                service.url,
                node_id="node-a",
                pool=POOL,
                resources=NodeResources(cpu_cores=8, memory_gb=64),
            )

            response = agent._post(
                "/v1/agent/report-command",
                report_payload(command, max_commands=3),
            )
        finally:
            service.stop()

        report_ids = [
            item["request_id"] for item in run_commands(response)
        ]
        heartbeat_ids = [
            item["request_id"]
            for item in run_commands(
                controller.node_heartbeat(
                    "node-a",
                    status={"authoritative": True, "sessions": []},
                    max_commands=8,
                )
            )
        ]
        self.assertEqual(
            report_ids,
            [item.request_id for item in successors[:3]],
        )
        self.assertEqual(
            heartbeat_ids,
            [item.request_id for item in successors[3:]],
        )
        self.assertEqual(
            len(set(report_ids + heartbeat_ids)),
            len(successors),
        )

    def test_draining_and_removed_nodes_receive_no_report_work(self):
        for terminal_state in (NodeState.DRAINING, NodeState.REMOVED):
            with self.subTest(state=terminal_state.value):
                controller, _, successors, command = (
                    prepare_saturated_controller()
                )
                if terminal_state == NodeState.DRAINING:
                    controller.drain_node("node-a")
                else:
                    controller.remove_allocation("alloc-a", force=True)
                service = CAPEControllerService(
                    controller=controller, schedule_interval=100.0
                ).start()
                try:
                    agent = NodeAgentClient(
                        service.url,
                        node_id="node-a",
                        pool=POOL,
                        resources=NodeResources(
                            cpu_cores=1, memory_gb=64
                        ),
                    )
                    response = agent._post(
                        "/v1/agent/report-command",
                        report_payload(command),
                    )
                finally:
                    service.stop()

                self.assertEqual(run_commands(response), [])
                self.assertTrue(
                    all(
                        command.get("action")
                        in ("cancel", "stop_session")
                        for command in response["commands"]
                    )
                )
                self.assertEqual(
                    controller.get_request(successors[0].request_id).state,
                    RequestState.QUEUED,
                )
                if terminal_state == NodeState.REMOVED:
                    self.assertEqual(
                        response["agent_control"]["action"], "shutdown"
                    )

    def test_legacy_report_without_opt_in_keeps_heartbeat_delivery(self):
        controller, _, successors, command = (
            prepare_saturated_controller()
        )
        service = CAPEControllerService(
            controller=controller, schedule_interval=100.0
        ).start()
        try:
            agent = NodeAgentClient(
                service.url,
                node_id="node-a",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=64),
            )

            response = agent._post(
                "/v1/agent/report-command",
                report_payload(
                    command,
                    accept_commands=None,
                    max_commands=None,
                ),
            )
        finally:
            service.stop()

        self.assertEqual(response["commands"], [])
        heartbeat = controller.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
        )
        self.assertEqual(
            [item["request_id"] for item in run_commands(heartbeat)],
            [successors[0].request_id],
        )

    def test_report_schedule_pass_serializes_with_periodic_scheduler(self):
        class InterleavingController(CAPEController):
            def __init__(self):
                CAPEController.__init__(self)
                self._schedule_lock = threading.Lock()
                self.active_schedule_calls = 0
                self.max_active_schedule_calls = 0

            def schedule_once(self):
                with self._schedule_lock:
                    self.active_schedule_calls += 1
                    self.max_active_schedule_calls = max(
                        self.max_active_schedule_calls,
                        self.active_schedule_calls,
                    )
                try:
                    time.sleep(0.01)
                    return CAPEController.schedule_once(self)
                finally:
                    with self._schedule_lock:
                        self.active_schedule_calls -= 1

        controller = InterleavingController()
        register_node(controller)
        first = controller.submit_workload(workload("first"))
        successor = controller.submit_workload(workload("successor"))
        controller.schedule_once()
        command = controller.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            max_commands=1,
        )["commands"][0]
        service = CAPEControllerService(
            controller=controller, schedule_interval=0.0
        ).start()
        try:
            agent = NodeAgentClient(
                service.url,
                node_id="node-a",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=64),
            )
            response = agent._post(
                "/v1/agent/report-command",
                report_payload(command),
            )
            time.sleep(0.05)
            heartbeat = controller.node_heartbeat(
                "node-a",
                status={"authoritative": True, "sessions": []},
            )
        finally:
            service.stop()

        self.assertEqual(response["request_id"], first.request_id)
        self.assertEqual(
            [item["request_id"] for item in run_commands(response)],
            [successor.request_id],
        )
        self.assertNotIn(
            successor.request_id,
            [item["request_id"] for item in run_commands(heartbeat)],
        )
        self.assertEqual(controller.max_active_schedule_calls, 1)
        self.assertEqual(
            len(
                [
                    assignment
                    for assignment in controller.command_assignments.values()
                    if assignment.request_id == successor.request_id
                ]
            ),
            1,
        )


class AgentReportDispatchTests(unittest.TestCase):
    def make_agent(self, tmp, exit_on_removal=True):
        return NodeAgentClient(
            "http://controller.invalid",
            node_id="node-a",
            pool=POOL,
            resources=NodeResources(cpu_cores=2, memory_gb=4),
            allocation_id="alloc-a",
            log_dir=tmp,
            exit_on_removal=exit_on_removal,
        )

    def test_report_executes_run_cancel_and_stop_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self.make_agent(tmp)
            launched = []
            cancelled = []
            stopped = []
            launch_event = threading.Event()
            posted = []

            def execute(command):
                launched.append(command["command_id"])
                launch_event.set()

            agent.execute_assignment = execute
            agent.cancel_assignment = (
                lambda command: cancelled.append(command["command_id"])
            )
            agent.stop_session = (
                lambda execution_session_id, runtime_name=None, session_spec=None, suspend_version=None:
                stopped.append(execution_session_id)
            )

            def post(path, payload, token=None):
                posted.append((path, dict(payload)))
                return {
                    "commands": [
                        {"action": "cancel", "command_id": "cmd-cancel"},
                        {
                            "action": "stop_session",
                            "execution_session_id": "sess-stop",
                        },
                        {"action": "run", "command_id": "cmd-next"},
                    ]
                }

            agent._post = post

            agent._report("cmd-done", 0, "ok", "", None)
            self.assertTrue(launch_event.wait(1.0))
            agent.join(timeout=1.0)

        self.assertEqual(cancelled, ["cmd-cancel"])
        self.assertEqual(stopped, ["sess-stop"])
        self.assertEqual(launched, ["cmd-next"])
        self.assertEqual(posted[0][0], "/v1/agent/report-command")
        self.assertEqual(posted[0][1]["failure_kind"], "workload")
        self.assertIs(posted[0][1]["accept_commands"], True)
        self.assertEqual(posted[0][1]["max_commands"], 8)

    def test_report_shutdown_suppresses_run_but_allows_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self.make_agent(tmp)
            launched = []
            stopped = []
            agent.execute_assignment = lambda command: launched.append(command)
            agent.stop_session = (
                lambda execution_session_id, runtime_name=None, session_spec=None, suspend_version=None:
                stopped.append(execution_session_id)
            )
            agent._post = lambda *args, **kwargs: {
                "commands": [
                    {
                        "action": "stop_session",
                        "execution_session_id": "sess-stop",
                    },
                    {"action": "run", "command_id": "cmd-must-not-run"},
                ],
                "agent_control": {
                    "authoritative": True,
                    "action": "shutdown",
                    "reason": "allocation_removed",
                    "node_id": "node-a",
                    "allocation_id": "alloc-a",
                },
            }

            response = agent._report("cmd-done", 0, "", "", None)
            pending = agent._take_pending_shutdown_control()

        self.assertEqual(response["agent_control"]["action"], "shutdown")
        self.assertEqual(stopped, ["sess-stop"])
        self.assertEqual(launched, [])
        self.assertEqual(pending["reason"], "allocation_removed")

    def test_report_transport_error_sends_failure_without_spurious_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self.make_agent(tmp)
            launched = []
            agent.execute_assignment = lambda command: launched.append(command)

            reports = []
            def fail(path, payload, **kwargs):
                reports.append(dict(payload))
                if len(reports) == 1:
                    raise aiohttp.ClientConnectionError("controller unavailable")
                return {"commands": []}

            agent._post = fail

            agent._report("cmd-done", 0, "", "", None)
            self.assertEqual(len(reports), 2)
            self.assertEqual(reports[1]["failure_kind"], "infrastructure")
            self.assertNotEqual(reports[1]["exit_code"], 0)

        self.assertEqual(launched, [])
        self.assertIsNone(agent._take_pending_shutdown_control())

    def test_report_tolerates_old_controller_response_without_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = self.make_agent(tmp)
            launched = []
            agent.execute_assignment = lambda command: launched.append(command)
            agent._post = lambda *args, **kwargs: {
                "request_id": "req-done",
                "state": "COMPLETED",
            }

            response = agent._report("cmd-done", 0, "", "", None)

        self.assertEqual(response["state"], "COMPLETED")
        self.assertEqual(launched, [])
        self.assertIsNone(agent._take_pending_shutdown_control())


class LongHeartbeatHandoffTests(unittest.TestCase):
    def test_two_one_capacity_requests_complete_after_only_one_heartbeat(self):
        class CountingAgent(NodeAgentClient):
            def __init__(self, *args, **kwargs):
                NodeAgentClient.__init__(self, *args, **kwargs)
                self.heartbeat_calls = 0

            async def heartbeat_once_async(
                self, background=True, session=None
            ):
                self.heartbeat_calls += 1
                return await NodeAgentClient.heartbeat_once_async(
                    self, background=background, session=session
                )

        controller = CAPEController()
        register_node(controller, agent_epoch="epoch-a")
        first = controller.submit_workload(workload("first"))
        second = controller.submit_workload(workload("second"))
        controller.schedule_once()
        service = CAPEControllerService(
            controller=controller, schedule_interval=100.0
        ).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                agent = CountingAgent(
                    service.url,
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=1, memory_gb=64),
                    allocation_id="alloc-a",
                    agent_epoch="epoch-a",
                    capabilities=RuntimeCapabilities(
                        adapter="local-process",
                        image_formats=("default", "docker"),
                    ),
                    log_dir=tmp,
                )
                agent.register()

                async def exercise():
                    loop_task = asyncio.create_task(
                        agent.run_forever_async(interval=30.0)
                    )
                    deadline = time.monotonic() + 5.0
                    try:
                        while time.monotonic() < deadline:
                            states = (
                                controller.get_request(first.request_id).state,
                                controller.get_request(second.request_id).state,
                            )
                            if states == (
                                RequestState.COMPLETED,
                                RequestState.COMPLETED,
                            ):
                                return
                            await asyncio.sleep(0.01)
                        self.fail(
                            "requests did not complete without a second "
                            "heartbeat: {} / {}".format(*states)
                        )
                    finally:
                        loop_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await loop_task

                asyncio.run(exercise())
                agent.join(timeout=1.0)
        finally:
            service.stop()

        self.assertEqual(agent.heartbeat_calls, 1)
        self.assertEqual(
            controller.get_request(first.request_id).state,
            RequestState.COMPLETED,
        )
        self.assertEqual(
            controller.get_request(second.request_id).state,
            RequestState.COMPLETED,
        )


if __name__ == "__main__":
    unittest.main()
