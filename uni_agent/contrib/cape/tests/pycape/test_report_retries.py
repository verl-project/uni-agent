# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import tempfile
import unittest
from unittest import mock

from pycape import NodeAgentClient, NodeResources
from pycape.agent import AgentHTTPError, ReportRetryExhausted


class ReportRetryTests(unittest.TestCase):
    def agent(self, directory):
        return NodeAgentClient("http://unused", "node", "pool", NodeResources(1, 1), log_dir=directory)

    def test_permanent_rejection_returns_without_retry_or_agent_stop(self):
        for status in (400, 401, 403, 404, 409, 429):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                agent = self.agent(directory)
                with mock.patch.object(agent, "_post", side_effect=AgentHTTPError(status, {})) as post, mock.patch.object(agent, "_log_lifecycle") as log:
                    self.assertEqual(agent._report("unknown", 0, "output", "", "/node/log"), {})
                self.assertEqual(post.call_count, 1)
                self.assertIn("reason=permanent_http", log.call_args.args[0])
                self.assertFalse(agent._report_retries_exhausted.is_set())

    def test_oversize_compact_fallback_rejection_does_not_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            with mock.patch.object(agent, "_post", side_effect=AgentHTTPError(413, {})) as post:
                self.assertEqual(agent._report("oversize", 0, "output", "", "/node/log"), {})
            self.assertEqual(post.call_count, 2)
            self.assertFalse(agent._report_retries_exhausted.is_set())

    def test_transient_exhaustion_is_bounded_and_stops_heartbeats(self):
        for error in (OSError("offline"), AgentHTTPError(503, {})):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                agent = self.agent(directory)
                with mock.patch.object(agent, "_post", side_effect=error) as post, mock.patch("threading.Event.wait"):
                    agent._report("pending", 0, "output", "", "/node/log")
                    self.assertEqual(post.call_count, 4)
                    with self.assertRaises(ReportRetryExhausted):
                        agent.heartbeat_once()
                    self.assertEqual(post.call_count, 4)
                self.assertTrue(agent._report_retries_exhausted.is_set())
                with mock.patch.object(agent, "_register_with_retry_async", new=mock.AsyncMock(return_value={})):
                    with self.assertRaisesRegex(RuntimeError, "report_retry_exhausted"):
                        asyncio.run(agent.run_forever_async(interval=0))
                self.assertTrue(agent._shutting_down)

    def test_programming_errors_surface_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.agent(directory)
            with mock.patch.object(agent, "_post", side_effect=TypeError("bad payload")) as post:
                with self.assertRaises(TypeError):
                    agent._report("broken", 0, "", "", None)
                self.assertEqual(post.call_count, 1)
