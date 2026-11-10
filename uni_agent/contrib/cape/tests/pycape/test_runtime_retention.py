# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
from pathlib import Path
import tempfile
import unittest

from pycape import NodeAgentClient, NodeResources, RuntimeCapabilities
from pycape.execution_storage import check_storage

from pycape.runtime import (ApptainerRuntimeAdapter, BubbleWrapRuntimeAdapter,
                            PodmanRuntimeAdapter, ContainerSessionSpec)


class RuntimeRetentionTests(unittest.TestCase):
    def test_only_suspension_controls_wake_heartbeat(self):
        with tempfile.TemporaryDirectory() as root:
            agent = NodeAgentClient("http://unused", "node", "pool", NodeResources(1, 1),
                state_path=str(Path(root) / "state.json"), log_dir=str(Path(root) / "logs"))
            identity = "exec-noop"
            spec = ContainerSessionSpec(identity, isolation_policy="none")
            agent.local_sessions.add(identity)
            agent.session_contexts[identity] = agent._persisted_session_context(
                {"session_spec": {"execution_session_id": identity, "isolation_policy": "none"}},
                "local-process", spec)
            notifications = []
            agent._notify_retention_change = lambda: notifications.append(True)
            agent.stop_session(identity, suspend_version=1)
            self.assertEqual(notifications, [])
            agent.stop_session(identity, suspend_version=2, notify=True)
            self.assertEqual(notifications, [True])
            agent.stop_session(identity, suspend_version=2, notify=True)
            self.assertEqual(notifications, [True])

    def test_agent_retention_survives_restart_and_purge_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = ApptainerRuntimeAdapter(overlay_supported=True)
            spec = ContainerSessionSpec("exec-test", isolation_policy="image-stateful", runtime_dir=root)
            self.assertEqual(runtime.start_session(spec).exit_code, 0)
            marker = Path(spec.overlay_path) / "user-data"
            marker.write_text("preserved")

            def agent(adapter):
                return NodeAgentClient("http://unused", "node", "pool", NodeResources(1, 1),
                    runtime_adapters={"apptainer": adapter}, capabilities=RuntimeCapabilities(adapter="apptainer"),
                    state_path=str(Path(root) / "agent.json"), log_dir=str(Path(root) / "logs"))

            first = agent(runtime)
            first.local_sessions.add("exec-test")
            first.session_contexts["exec-test"] = first._persisted_session_context(
                {"request_id": "req-test", "session_spec": {"execution_session_id": "exec-test", "workspace": None,
                 "isolation_policy": "image-stateful", "runtime_dir": root}}, "apptainer", spec)
            async def suspend_and_observe_wakeup():
                first._heartbeat_loop = asyncio.get_running_loop()
                first._heartbeat_wakeup = asyncio.Event()
                await first._heartbeat_loop.run_in_executor(None,
                    functools.partial(first.stop_session, "exec-test", suspend_version=1, notify=True))
                await asyncio.wait_for(first._heartbeat_wakeup.wait(), timeout=.2)
                first._heartbeat_wakeup.clear()
                await first._heartbeat_loop.run_in_executor(None,
                    functools.partial(first.stop_session, "exec-test", suspend_version=1, notify=True))
                await asyncio.sleep(0)
                self.assertFalse(first._heartbeat_wakeup.is_set(), "duplicate receipt must not spin heartbeats")
            asyncio.run(suspend_and_observe_wakeup())
            self.assertNotIn("exec-test", first.local_sessions)
            self.assertEqual(marker.read_text(), "preserved")
            second = agent(ApptainerRuntimeAdapter(overlay_supported=True))
            self.assertEqual(second.suspended_contexts["exec-test"]["suspend_version"], 1)
            self.assertEqual(second.suspended_contexts["exec-test"]["session_spec"]["runtime_adapter"], "apptainer")
            second.stop_session("exec-test", suspend_version=1)
            self.assertEqual(marker.read_text(), "preserved")
            second.purge_session({"execution_session_id": "exec-test", "suspend_version": 1})
            self.assertFalse(marker.exists())
            self.assertIn("exec-test", second.purged_executions)

    def test_storage_limits_fail_without_deleting_data(self):
        with tempfile.TemporaryDirectory() as root:
            marker = Path(root) / "user-data"
            marker.write_bytes(b"x" * 8192)
            with self.assertRaisesRegex(RuntimeError, "quota"):
                check_storage(root, min_free_bytes=0, max_bytes=1)
            with self.assertRaisesRegex(RuntimeError, "reserve"):
                check_storage(root, min_free_bytes=2 ** 63)
            self.assertEqual(marker.read_bytes(), b"x" * 8192)

    def test_overlay_survives_suspend_and_missing_state_is_not_recreated(self):
        for runtime, policy in ((BubbleWrapRuntimeAdapter(overlay_supported=True), "host-stateful"),
                                (ApptainerRuntimeAdapter(overlay_supported=True), "image-stateful")):
            with self.subTest(runtime=runtime.name), tempfile.TemporaryDirectory() as root:
                spec = ContainerSessionSpec("execution-A", isolation_policy=policy,
                                            runtime_dir=root, state_layer_id="layer-A")
                self.assertEqual(runtime.start_session(spec).exit_code, 0)
                path = Path(spec.overlay_upper if policy == "host-stateful" else spec.overlay_path)
                marker = path / "preserved"
                marker.write_text("state")
                self.assertEqual(runtime.suspend_session("execution-A").exit_code, 0)
                self.assertNotIn("execution-A", runtime.sessions)
                self.assertEqual(marker.read_text(), "state")
                self.assertEqual(runtime.resume_session(spec).exit_code, 0)
                self.assertEqual(marker.read_text(), "state")
                self.assertEqual(runtime.suspend_session("execution-A").exit_code, 0)
                moved = path.with_name(path.name + "-lost")
                path.rename(moved)
                self.assertNotEqual(runtime.resume_session(spec).exit_code, 0)
                self.assertFalse(path.exists(), "resume must never recreate a missing layer")
                self.assertEqual((moved / "preserved").read_text(), "state")

    def test_podman_rejects_stateful_without_silent_downgrade(self):
        runtime = PodmanRuntimeAdapter()
        spec = ContainerSessionSpec("execution-A", isolation_policy="image-stateful")
        with self.assertRaisesRegex(ValueError, "not supported"):
            runtime.start_session(spec)
        self.assertNotEqual(runtime.resume_session(spec).exit_code, 0)
        self.assertFalse(runtime._uses_live_session(ContainerSessionSpec("stateless")))


if __name__ == "__main__":
    unittest.main()
