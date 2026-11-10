# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp
from aiohttp import web

from pycape import CAPEController, CAPEControllerService, NodeAgentClient, NodeResources, RuntimeCapabilities


ROOT = Path(__file__).resolve().parents[2]


class AiohttpTransportTests(unittest.TestCase):
    def test_package_declares_aiohttp_and_python_38_floor(self):
        metadata = (ROOT / "pycape" / "pyproject.toml").read_text()
        self.assertIn('requires-python = ">=3.8"', metadata)
        self.assertIn("dependencies", metadata)
        self.assertIn("aiohttp>=3.8", metadata)

    def test_legacy_http_stacks_are_absent(self):
        service_source = (ROOT / "pycape" / "service.py").read_text()
        agent_source = (ROOT / "pycape" / "agent.py").read_text()
        cli_source = (ROOT / "pycape" / "cli.py").read_text()

        self.assertNotIn("http.server", service_source)
        self.assertNotIn("BaseHTTPRequestHandler", service_source)
        self.assertIn("from aiohttp import web", service_source)
        for source in (agent_source, cli_source):
            self.assertNotIn("urllib", source)
            self.assertNotIn("urlopen", source)
            self.assertIn("aiohttp.ClientSession", source)

    def test_service_exposes_explicit_aiohttp_routes_and_async_scheduler(self):
        service = CAPEControllerService(controller=CAPEController(), schedule_interval=0.01)
        self.assertIsInstance(service.app, web.Application)
        routes = {(route.method, route.resource.canonical) for route in service.app.router.routes()}
        expected = {
            ("GET", "/health"),
            ("POST", "/v1/submit"),
            ("GET", "/v1/requests/{request_id}"),
            ("GET", "/v1/requests/{request_id}/logs"),
            ("GET", "/v1/requests/{request_id}/wait"),
            ("POST", "/v1/requests/{request_id}/cancel"),
            ("GET", "/v1/pools/{pool}/utilization"),
            ("POST", "/v1/agent/announce"),
            ("POST", "/v1/agent/register"),
            ("POST", "/v1/agent/heartbeat"),
            ("POST", "/v1/agent/report-command"),
            ("POST", "/v1/admin/drain-node"),
            ("POST", "/v1/admin/drain-allocation"),
            ("POST", "/v1/admin/remove-allocation"),
        }
        self.assertTrue(expected.issubset(routes))
        self.assertTrue(inspect.iscoroutinefunction(service._scheduler_loop))
        self.assertTrue(inspect.iscoroutinefunction(service._wait_for_request))
        self.assertFalse(hasattr(service, "_scheduler_thread"))

        service.start()
        try:
            self.assertIsInstance(service._scheduler_task, asyncio.Task)
            self.assertTrue(service._scheduler_task.get_loop().is_running())
        finally:
            service.stop()

    def test_slow_scheduler_work_stays_off_the_event_loop(self):
        class SlowSchedulerController(CAPEController):
            def __init__(self):
                CAPEController.__init__(self)
                self.schedule_started = threading.Event()

            def schedule_once(self):
                self.schedule_started.set()
                time.sleep(0.25)
                return CAPEController.schedule_once(self)

        controller = SlowSchedulerController()
        service = CAPEControllerService(controller=controller, schedule_interval=10.0).start()
        try:
            self.assertTrue(controller.schedule_started.wait(1.0))
            health_seconds = asyncio.run(self._measure_health_latency(service.url))
        finally:
            service.stop()

        self.assertLess(health_seconds, 0.15)

    def test_wait_is_event_driven_and_does_not_block_health(self):
        service = CAPEControllerService(controller=CAPEController(), schedule_interval=0.01).start()
        try:
            result = asyncio.run(self._exercise_wait_and_health(service.url))
        finally:
            service.stop()

        self.assertEqual(result["wait_state"], "CANCELLED")
        self.assertTrue(result["health_ok"])
        self.assertLess(result["health_seconds"], 0.2)
        self.assertLess(result["wait_seconds"], 1.0)

    def test_agent_async_heartbeat_offloads_blocking_status_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient(
                "http://127.0.0.1:1",
                "node-a",
                "pool-a",
                NodeResources(cpu_cores=1, memory_gb=1, gpu_uuids=()),
                capabilities=RuntimeCapabilities(adapter="local-process"),
                log_dir=tmp,
            )

            def slow_status():
                time.sleep(0.25)
                return {"authoritative": True, "sessions": []}

            async def fake_post(path, payload, token=None, session=None):
                return {"commands": []}

            agent.collect_status = slow_status
            agent._post_async = fake_post
            loop_delay = asyncio.run(self._measure_heartbeat_loop_delay(agent))

        self.assertTrue(inspect.iscoroutinefunction(agent.heartbeat_once_async))
        self.assertTrue(inspect.iscoroutinefunction(agent.run_forever_async))
        self.assertLess(loop_delay, 0.1)

    def test_agent_runtime_execution_stays_off_the_event_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient(
                "http://127.0.0.1:1",
                "node-a",
                "pool-a",
                NodeResources(cpu_cores=1, memory_gb=1, gpu_uuids=()),
                capabilities=RuntimeCapabilities(adapter="local-process"),
                log_dir=tmp,
            )
            agent.collect_status = lambda: {"authoritative": True, "sessions": []}

            async def fake_post(path, payload, token=None, session=None):
                return {"commands": [{"command_id": "slow-command"}]}

            def slow_assignment(assignment):
                time.sleep(0.25)

            agent._post_async = fake_post
            agent.execute_assignment = slow_assignment
            loop_delay = asyncio.run(self._measure_foreground_assignment_loop_delay(agent))

        self.assertLess(loop_delay, 0.1)

    def test_agent_registration_is_process_local_idempotent_across_run_forever(self):
        class StopHeartbeat(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            agent = NodeAgentClient(
                "http://127.0.0.1:1",
                "node-a",
                "pool-a",
                NodeResources(cpu_cores=1, memory_gb=1, gpu_uuids=()),
                capabilities=RuntimeCapabilities(adapter="local-process"),
                log_dir=tmp,
                agent_epoch="process-a",
            )
            calls = []

            async def fake_post(path, payload, token=None, session=None):
                calls.append((path, payload))
                if path == "/v1/agent/heartbeat":
                    raise StopHeartbeat()
                return {"node_id": "node-a", "state": "ACTIVE"}

            agent._post_async = fake_post

            async def exercise():
                first = await agent.register_async()
                second = await agent.register_async()
                self.assertEqual(first, second)
                with self.assertRaises(StopHeartbeat):
                    await agent.run_forever_async(interval=0)

            asyncio.run(exercise())

        registrations = [
            payload for path, payload in calls
            if path == "/v1/agent/register"
        ]
        self.assertEqual(len(registrations), 1)
        self.assertEqual(registrations[0]["agent_epoch"], "process-a")
        heartbeats = [
            payload for path, payload in calls
            if path == "/v1/agent/heartbeat"
        ]
        self.assertEqual(heartbeats[0]["agent_epoch"], "process-a")

    @staticmethod
    async def _json(response):
        text = await response.text()
        return json.loads(text)

    async def _measure_health_latency(self, url):
        timeout = aiohttp.ClientTimeout(total=2.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            started = time.perf_counter()
            async with session.get(url + "/health") as response:
                self.assertEqual(response.status, 200)
                await self._json(response)
            return time.perf_counter() - started

    async def _exercise_wait_and_health(self, url):
        payload = {
            "pool": "pool-a",
            "user_id": "alice",
            "session_id": "task-a",
            "command": [sys.executable, "-c", "print(1)"],
            "resources": {"gpus": 0, "cpu_cores": 1, "memory_gb": 1},
        }
        timeout = aiohttp.ClientTimeout(total=3.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url + "/v1/submit", json=payload) as response:
                self.assertEqual(response.status, 202)
                submitted = await self._json(response)
            request_id = submitted["request_id"]
            wait_started = time.perf_counter()
            wait_task = asyncio.create_task(
                self._get_json(session, url + "/v1/requests/{}/wait?timeout=2".format(request_id))
            )
            await asyncio.sleep(0.05)
            health_started = time.perf_counter()
            async with session.get(url + "/health") as response:
                health = await self._json(response)
            health_seconds = time.perf_counter() - health_started
            async with session.post(
                url + "/v1/requests/{}/cancel".format(request_id),
                json={"reason": "transport-test"},
            ) as response:
                self.assertEqual(response.status, 200)
                await self._json(response)
            waited = await wait_task
            return {
                "wait_state": waited["state"],
                "health_ok": health["ok"],
                "health_seconds": health_seconds,
                "wait_seconds": time.perf_counter() - wait_started,
            }

    async def _get_json(self, session, url):
        async with session.get(url) as response:
            self.assertEqual(response.status, 200)
            return await self._json(response)

    async def _measure_heartbeat_loop_delay(self, agent):
        task = asyncio.create_task(agent.heartbeat_once_async(background=True))
        started = time.perf_counter()
        await asyncio.sleep(0.02)
        delay = time.perf_counter() - started
        await task
        return delay


    async def _measure_foreground_assignment_loop_delay(self, agent):
        task = asyncio.create_task(agent.heartbeat_once_async(background=False))
        started = time.perf_counter()
        await asyncio.sleep(0.02)
        delay = time.perf_counter() - started
        await task
        return delay


if __name__ == "__main__":
    unittest.main()
