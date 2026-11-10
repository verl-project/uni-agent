# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pycape.agent as agent_module
from pycape import (
    CAPEController,
    CAPEControllerService,
    MemoryJournal,
    NodeAgentClient,
    NodeResources,
    NodeState,
    RuntimeCapabilities,
    RuntimeResult,
)


POOL = "pool-a"


def shutdown_payload(node_id="node-a", allocation_id="alloc-a"):
    return {
        "node": {
            "node_id": node_id,
            "allocation_id": allocation_id,
            "state": "REMOVED",
        },
        "commands": [],
        "agent_control": {
            "authoritative": True,
            "action": "shutdown",
            "reason": "allocation_removed",
            "node_id": node_id,
            "allocation_id": allocation_id,
        },
    }


class ControllerAgentRetirementTests(unittest.TestCase):
    def register(self, controller, state=NodeState.ACTIVE):
        return controller.register_node(
            node_id="node-a",
            pool=POOL,
            resources=NodeResources(cpu_cores=2, memory_gb=4),
            capabilities=RuntimeCapabilities(adapter="local-process"),
            allocation_id="alloc-a",
            agent_epoch="epoch-a",
            state=state,
        )

    def test_removed_node_heartbeat_has_positive_authoritative_shutdown(self):
        controller = CAPEController()
        self.register(controller)
        controller.remove_allocation("alloc-a")

        response = controller.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            agent_epoch="epoch-a",
        )

        self.assertEqual(response["commands"], [])
        self.assertEqual(
            response["agent_control"],
            shutdown_payload()["agent_control"],
        )

    def test_same_allocation_cannot_reregister_over_removed_tombstone(self):
        controller = CAPEController()
        self.register(controller)
        controller.remove_allocation("alloc-a")

        node = self.register(controller)

        self.assertEqual(node.state, NodeState.REMOVED)
        self.assertEqual(
            controller.node_heartbeat("node-a", agent_epoch="epoch-a")[
                "agent_control"
            ]["reason"],
            "allocation_removed",
        )

    def test_removed_tombstone_ignores_explicit_heartbeat_state(self):
        controller = CAPEController()
        self.register(controller)
        controller.remove_allocation("alloc-a")

        response = controller.node_heartbeat(
            "node-a",
            state=NodeState.REGISTERING,
            status={"authoritative": False, "phase": "runtime_probe"},
            agent_epoch="epoch-a",
        )

        self.assertEqual(controller.get_node("node-a").state, NodeState.REMOVED)
        self.assertEqual(
            response["agent_control"], shutdown_payload()["agent_control"]
        )

    def test_valid_heartbeat_recovers_replayed_unhealthy_node(self):
        controller = CAPEController()
        node = self.register(controller)
        node.state = NodeState.UNHEALTHY

        response = controller.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            agent_epoch="epoch-a",
        )

        self.assertEqual(controller.get_node("node-a").state, NodeState.ACTIVE)
        self.assertNotIn("agent_control", response)

    def test_live_node_recovers_after_real_journal_replay(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        self.register(controller)

        replayed = CAPEController.reconcile_from_journal(journal)
        self.assertEqual(replayed.get_node("node-a").state, NodeState.UNHEALTHY)

        response = replayed.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            agent_epoch="epoch-a",
        )

        self.assertEqual(replayed.get_node("node-a").state, NodeState.ACTIVE)
        self.assertNotIn("agent_control", response)

    def test_removed_allocation_tombstone_and_reason_survive_journal_replay(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        self.register(controller)
        controller.remove_allocation("alloc-a")

        replayed = CAPEController.reconcile_from_journal(journal)
        response = replayed.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            agent_epoch="epoch-a",
        )

        self.assertEqual(replayed.get_node("node-a").state, NodeState.REMOVED)
        self.assertEqual(
            response["agent_control"], shutdown_payload()["agent_control"]
        )
        self.assertEqual(
            self.register(replayed).state,
            NodeState.REMOVED,
        )

    def test_http_heartbeat_serializes_authoritative_shutdown_contract(self):
        controller = CAPEController()
        service = CAPEControllerService(
            controller=controller, schedule_interval=10.0
        ).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                agent = NodeAgentClient(
                    service.url,
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=2, memory_gb=4),
                    allocation_id="alloc-a",
                    agent_epoch="epoch-a",
                    log_dir=tmp,
                )
                agent.register()
                controller.remove_allocation("alloc-a")

                response = agent.heartbeat_once()

            self.assertEqual(
                response["agent_control"],
                shutdown_payload()["agent_control"],
            )
            self.assertEqual(response["commands"], [])
        finally:
            service.stop()

    def test_service_retire_then_fresh_allocation_reregisters(self):
        controller = CAPEController()
        service = CAPEControllerService(
            controller=controller, schedule_interval=10.0
        ).start()
        resources = NodeResources(cpu_cores=2, memory_gb=4)
        try:
            first = NodeAgentClient(
                service.url,
                node_id="node-a",
                pool=POOL,
                resources=resources,
                allocation_id="alloc-retired",
                agent_epoch="epoch-retired",
            )
            self.assertEqual(first.register()["state"], "ACTIVE")
            controller.remove_allocation("alloc-retired")

            stale = NodeAgentClient(
                service.url,
                node_id="node-a",
                pool=POOL,
                resources=resources,
                allocation_id="alloc-retired",
                agent_epoch="epoch-stale-retry",
            )
            stale_response = stale.register()
            self.assertEqual(stale_response["state"], "REMOVED")
            self.assertEqual(
                stale_response["agent_control"]["reason"],
                "allocation_removed",
            )

            fresh = NodeAgentClient(
                service.url,
                node_id="node-a",
                pool=POOL,
                resources=resources,
                allocation_id="alloc-fresh",
                agent_epoch="epoch-fresh",
            )
            fresh_response = fresh.register()

            self.assertEqual(fresh_response["state"], "ACTIVE")
            self.assertNotIn("agent_control", fresh_response)
            self.assertEqual(
                controller.allocation_status("alloc-retired")["state"],
                "REMOVED",
            )
            self.assertEqual(
                controller.allocation_status("alloc-fresh")["state"],
                "ACTIVE",
            )
            utilization = controller.pool_utilization(POOL)
            self.assertEqual(utilization["active_nodes"], 1)
            self.assertEqual(utilization["cpu_schedulable"], 2)
            self.assertEqual(
                utilization["node_states"][0]["allocation_id"],
                "alloc-fresh",
            )
        finally:
            service.stop()


class AgentRetirementClassifierTests(unittest.TestCase):
    def classifier(self):
        return getattr(agent_module, "classify_agent_shutdown")

    def test_classifier_accepts_only_matching_positive_tombstone(self):
        classify = self.classifier()
        payload = shutdown_payload()

        self.assertEqual(
            classify(payload, node_id="node-a", allocation_id="alloc-a"),
            payload["agent_control"],
        )

        invalid = []
        for change in (
            {"authoritative": False},
            {"action": "retry"},
            {"reason": "controller_restart"},
            {"node_id": "node-b"},
            {"allocation_id": "alloc-b"},
        ):
            candidate = shutdown_payload()
            candidate["agent_control"].update(change)
            invalid.append(candidate)
        invalid.extend(
            (
                {"error": "'node-a'"},
                {"status": 404, "error": {"error": "'node-a'"}},
                {"agent_control": "shutdown"},
                {},
            )
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                self.assertIsNone(
                    classify(
                        candidate,
                        node_id="node-a",
                        allocation_id="alloc-a",
                    )
                )


class AgentRetirementLoopTests(unittest.TestCase):
    def scripted_agent(self, responses, exit_on_removal=True):
        http_error = getattr(agent_module, "AgentHTTPError")

        class ScriptedAgent(NodeAgentClient):
            def __init__(self):
                super().__init__(
                    "http://controller.invalid",
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=1, memory_gb=1),
                    allocation_id="alloc-a",
                    exit_on_removal=exit_on_removal,
                )
                self.responses = list(responses(http_error))
                self.register_calls = 0
                self.shutdown_controls = []

            async def register_async(self, session=None):
                self.register_calls += 1
                self._registered = True
                return {"node_id": self.node_id, "state": "ACTIVE"}

            async def heartbeat_once_async(
                self, background=True, session=None
            ):
                item = self.responses.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item

            async def shutdown_async(self, control):
                self.shutdown_controls.append(dict(control))

        return ScriptedAgent()

    def test_transients_and_restart_unknown_retry_until_real_removal(self):
        def responses(http_error):
            return (
                http_error(500, {"error": "temporary"}),
                aiohttp.ClientConnectionError("controller unavailable"),
                asyncio.TimeoutError(),
                http_error(403, {"error": "token refresh window"}),
                http_error(404, {"error": "'node-a'"}),
                {"node": {"state": "ACTIVE"}, "commands": []},
                shutdown_payload(),
            )

        agent = self.scripted_agent(responses)

        control = asyncio.run(agent.run_forever_async(interval=0))

        self.assertEqual(control["reason"], "allocation_removed")
        self.assertEqual(len(agent.shutdown_controls), 1)
        self.assertEqual(agent.register_calls, 2)
        self.assertEqual(agent.responses, [])

    def test_stay_after_removal_never_runs_shutdown(self):
        class StopAfterRemoval(BaseException):
            pass

        def responses(_http_error):
            return (shutdown_payload(), StopAfterRemoval())

        agent = self.scripted_agent(responses, exit_on_removal=False)

        with self.assertRaises(StopAfterRemoval):
            asyncio.run(agent.run_forever_async(interval=0))

        self.assertEqual(agent.shutdown_controls, [])

    def test_stay_after_removal_runs_cleanup_but_never_launches_work(self):
        agent = NodeAgentClient(
            "http://controller.invalid",
            node_id="node-a",
            pool=POOL,
            resources=NodeResources(cpu_cores=1, memory_gb=1),
            allocation_id="alloc-a",
            exit_on_removal=False,
        )
        response = shutdown_payload()
        response["commands"] = [
            {"action": "stop_session", "execution_session_id": "sess-a"},
            {
                "action": "run",
                "command_id": "cmd-must-not-run",
            },
        ]
        stopped = []
        launched = []
        agent._post = lambda *args, **kwargs: response
        agent.stop_session = (
            lambda execution_session_id, runtime_name=None, session_spec=None, suspend_version=None:
            stopped.append(execution_session_id)
        )
        agent.execute_assignment = lambda command: launched.append(command)

        observed = agent.heartbeat_once(background=False)

        self.assertEqual(observed["agent_control"]["action"], "shutdown")
        self.assertEqual(stopped, ["sess-a"])
        self.assertEqual(launched, [])

    def test_initial_registration_transport_failure_retries(self):
        class InitiallyUnavailableAgent(NodeAgentClient):
            def __init__(self):
                super().__init__(
                    "http://controller.invalid",
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=1, memory_gb=1),
                    allocation_id="alloc-a",
                )
                self.register_calls = 0
                self.shutdown_controls = []

            async def register_async(self, session=None):
                self.register_calls += 1
                if self.register_calls == 1:
                    raise aiohttp.ClientConnectionError(
                        "controller restarting"
                    )
                self._registered = True
                return {"node_id": self.node_id, "state": "ACTIVE"}

            async def heartbeat_once_async(
                self, background=True, session=None
            ):
                return shutdown_payload()

            async def shutdown_async(self, control):
                self.shutdown_controls.append(dict(control))

        agent = InitiallyUnavailableAgent()

        control = asyncio.run(agent.run_forever_async(interval=0))

        self.assertEqual(control["reason"], "allocation_removed")
        self.assertEqual(agent.register_calls, 2)
        self.assertEqual(len(agent.shutdown_controls), 1)

    def test_authoritative_report_response_is_consumed_before_next_heartbeat(self):
        class ReportAwareAgent(NodeAgentClient):
            def __init__(self, tmp):
                super().__init__(
                    "http://controller.invalid",
                    node_id="node-a",
                    pool=POOL,
                    resources=NodeResources(cpu_cores=1, memory_gb=1),
                    allocation_id="alloc-a",
                    log_dir=tmp,
                )
                self.shutdown_controls = []

            async def register_async(self, session=None):
                self._registered = True
                return {"node_id": self.node_id, "state": "ACTIVE"}

            async def heartbeat_once_async(
                self, background=True, session=None
            ):
                raise AssertionError(
                    "pending report removal must preempt another heartbeat"
                )

            async def shutdown_async(self, control):
                self.shutdown_controls.append(dict(control))

        with tempfile.TemporaryDirectory() as tmp:
            agent = ReportAwareAgent(tmp)
            agent._post = lambda *args, **kwargs: shutdown_payload()
            agent._report("cmd-a", 0, "", "", None)

            control = asyncio.run(agent.run_forever_async(interval=0))

        self.assertEqual(control["reason"], "allocation_removed")
        self.assertEqual(len(agent.shutdown_controls), 1)


class AgentRetirementCleanupTests(unittest.TestCase):
    def test_shutdown_terminates_processes_cleans_sessions_and_logs_reason(self):
        class FakeHandle(object):
            def __init__(self):
                self.terminated = False

            def terminate_tree(self):
                self.terminated = True

        class FakeRuntime(object):
            def __init__(self):
                self.sessions = {"sess-a": object()}
                self.stopped = []

            def stop_session(self, execution_session_id):
                self.stopped.append(execution_session_id)
                self.sessions.pop(execution_session_id, None)
                return RuntimeResult(0, "", "")

        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime()
            state_path = os.path.join(tmp, "agent-state.json")
            agent = NodeAgentClient(
                "http://controller.invalid",
                node_id="node-a",
                pool=POOL,
                resources=NodeResources(cpu_cores=1, memory_gb=1),
                runtime_adapters={"fake": runtime},
                runtime_capabilities={
                    "fake": RuntimeCapabilities(adapter="fake")
                },
                allocation_id="alloc-a",
                state_path=state_path,
                log_dir=tmp,
            )
            handle = FakeHandle()
            agent.local_sessions.add("sess-a")
            agent.session_contexts["sess-a"] = {
                "runtime_adapter": "fake",
                "request_id": "req-a",
            }
            agent.running_commands["cmd-a"] = {
                "command_id": "cmd-a",
                "execution_session_id": "sess-a",
            }
            agent.running_processes["cmd-a"] = handle
            agent._save_state()
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                asyncio.run(
                    agent.shutdown_async(shutdown_payload()["agent_control"])
                )

            persisted = json.loads(Path(state_path).read_text())

        self.assertTrue(handle.terminated)
        self.assertEqual(runtime.stopped, ["sess-a"])
        self.assertEqual(agent.local_sessions, set())
        self.assertEqual(agent.session_contexts, {})
        self.assertEqual(agent.running_commands, {})
        self.assertEqual(agent.running_processes, {})
        self.assertEqual(persisted["local_sessions"], [])
        self.assertEqual(persisted["running_commands"], [])
        self.assertEqual(persisted["session_contexts"], {})
        self.assertIn("allocation_removed", stderr.getvalue())
        self.assertIn("alloc-a", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
