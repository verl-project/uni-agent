# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import CAPEController, EnforcementMode, NodeResources, ResourceRequest, RuntimeCapabilities, SQLiteJournal, WorkloadSpec


POOL = "pool-a"
PERF_SUITE_VERSION = "v0.6-performance-acceptance"


def workload(index, gpus=0, session_id=None, user_id=None):
    return WorkloadSpec(
        pool=POOL,
        user_id=user_id or "user-{}".format(index % 8),
        session_id=session_id or "perf-{}".format(index),
        command=(sys.executable, "-c", "print({})".format(index)),
        resources=ResourceRequest(gpus=gpus, cpu_cores=1, memory_gb=1, max_duration_seconds=100),
    )


def register_node(ctrl, cpu=512, mem=4096, gpus=0):
    ctrl.register_node(
        "node-a",
        POOL,
        NodeResources(cpu_cores=cpu, memory_gb=mem, gpu_uuids=tuple("GPU-{}".format(i) for i in range(gpus))),
        RuntimeCapabilities(adapter="local-process", gpu_enforcement=EnforcementMode.HARD),
    )


class PerformanceAcceptanceTests(unittest.TestCase):
    def test_suite_version_is_explicit(self):
        self.assertEqual(PERF_SUITE_VERSION, "v0.6-performance-acceptance")

    def test_submit_latency_under_queue_load(self):
        ctrl = CAPEController()
        started = time.time()
        for i in range(1000):
            ctrl.submit_workload(workload(i))
        elapsed = time.time() - started
        self.assertLess(elapsed, float(os.environ.get("PYCAPE_PERF_SUBMIT_1000_MAX_SECONDS", "2.0")))
        self.assertEqual(len(ctrl.requests), 1000)

    def test_scheduling_loop_latency_for_assignment_batch(self):
        ctrl = CAPEController()
        register_node(ctrl, cpu=256)
        for i in range(128):
            ctrl.submit_workload(workload(i))
        started = time.time()
        scheduled = ctrl.schedule_once()
        elapsed = time.time() - started
        self.assertEqual(scheduled, 128)
        self.assertLess(elapsed, float(os.environ.get("PYCAPE_PERF_SCHEDULE_128_MAX_SECONDS", "2.0")))
        self.assertEqual(len(ctrl.command_assignments), 128)

    def test_heartbeat_dequeue_capacity(self):
        ctrl = CAPEController()
        register_node(ctrl, cpu=128)
        for i in range(64):
            ctrl.submit_workload(workload(i))
        ctrl.schedule_once()
        started = time.time()
        response = ctrl.node_heartbeat("node-a", status={"authoritative": True, "sessions": []}, max_commands=64)
        elapsed = time.time() - started
        self.assertEqual(len(response["commands"]), 64)
        self.assertLess(elapsed, float(os.environ.get("PYCAPE_PERF_HEARTBEAT_64_MAX_SECONDS", "1.0")))

    def test_sqlite_batched_write_rate_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = SQLiteJournal(os.path.join(tmp, "events.db"), batch_size=64)
            ctrl = CAPEController(journal=journal)
            register_node(ctrl, cpu=128)
            started = time.time()
            for i in range(64):
                ctrl.submit_workload(workload(i))
            ctrl.schedule_once()
            count = journal.count()
            elapsed = time.time() - started
            self.assertGreaterEqual(count, 64)
            self.assertLess(elapsed, float(os.environ.get("PYCAPE_PERF_SQLITE_64_MAX_SECONDS", "3.0")))
            journal.close()

    def test_warm_reuse_avoids_second_session_and_is_fast(self):
        ctrl = CAPEController()
        register_node(ctrl, cpu=4, gpus=1)
        first = ctrl.submit_workload(workload(1, gpus=1, session_id="warm", user_id="warm-user"))
        ctrl.schedule_once()
        first_session = ctrl.get_request(first.request_id).execution_session_id
        self.assertIsNotNone(first_session)
        ctrl.complete_request(first.request_id, exit_code=0)
        second = ctrl.submit_workload(workload(2, gpus=1, session_id="warm", user_id="warm-user"))
        started = time.time()
        ctrl.schedule_once()
        elapsed = time.time() - started
        self.assertEqual(ctrl.get_request(second.request_id).execution_session_id, first_session)
        self.assertEqual(len(ctrl.sessions), 1)
        self.assertLess(elapsed, float(os.environ.get("PYCAPE_PERF_WARM_REUSE_MAX_SECONDS", "0.5")))


if __name__ == "__main__":
    unittest.main()
