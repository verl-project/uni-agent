# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import (
    CAPEController,
    EnforcementMode,
    EventRecord,
    FakeRuntimeAdapter,
    InProcessNodeAgent,
    JoinToken,
    LocalProcessRuntimeAdapter,
    MemoryJournal,
    MemoryMetricsSink,
    MountSpec,
    NodeResources,
    NodeState,
    RequestRecord,
    RequestState,
    ResourceRequest,
    Role,
    RuntimeCapabilities,
    RuntimeResult,
    SchedulerPolicy,
    SQLiteJournal,
    SessionState,
    TokenAuthority,
    WorkloadSpec,
)
from pycape.service import workload_spec_from_dict
from pycape.metrics import FailingMetricsSink
from pycape.testing import FakeClock


POOL = "pool-a"


class NoReuseLookupTests(unittest.TestCase):
    def test_no_reuse_never_scans_session_history(self):
        class Unscannable(dict):
            def values(self):
                raise AssertionError("session history consulted")

        ctrl = CAPEController()
        ctrl.sessions = Unscannable()
        workload = spec(reuse=False)
        self.assertIsNone(ctrl._find_reusable_session(workload))
        self.assertIsNone(ctrl._find_matching_active_session(workload))


def gpu_ids(node, count):
    return tuple("{}-gpu-{}".format(node, i) for i in range(count))


def spec(user="user-a", gpus=1, cpu=1, mem=1, session_id="task", image=None, profile=None, mounts=(), hard=False, priority=0, runtime=None, privileged=False, isolation="default", reuse=True, keep_session_on_failure=True, max_duration=100, command_timeout=None):
    return WorkloadSpec(
        pool=POOL,
        user_id=user,
        session_id=session_id,
        command=("python", "train.py"),
        image=image,
        profile=profile,
        mounts=mounts,
        reuse=reuse,
        keep_session_on_failure=keep_session_on_failure,
        priority=priority,
        privileged=privileged,
        resources=ResourceRequest(
            gpus=gpus,
            cpu_cores=cpu,
            memory_gb=mem,
            hard_gpu_enforcement=hard,
            runtime_adapter=runtime,
            isolation_policy=isolation,
            max_duration_seconds=max_duration,
            command_timeout_seconds=command_timeout,
        ),
    )


def controller(policy=None, journal=None, metrics=None):
    clock = FakeClock()
    return CAPEController(clock=clock, policy=policy, journal=journal, metrics=metrics), clock


def register_gpu_node(ctrl, node_id="node-a", gpus=4, cpu=64, mem=512, caps=None, runtime_caps=None, deadline=None, drain_before=0, cleanup=0, join_token=None):
    if caps is None and runtime_caps is None:
        caps = RuntimeCapabilities(
            adapter="fake", gpu_enforcement=EnforcementMode.HARD
        )
    return ctrl.register_node(
        node_id=node_id,
        pool=POOL,
        resources=NodeResources(cpu_cores=cpu, memory_gb=mem, gpu_uuids=gpu_ids(node_id, gpus)),
        capabilities=caps,
        runtime_capabilities=runtime_caps,
        allocation_deadline=deadline,
        drain_before_sec=drain_before,
        cleanup_buffer_sec=cleanup,
        join_token=join_token,
    )


class SchedulerCoreTests(unittest.TestCase):

    def multi_runtime_capabilities(self):
        return {
            "local-process": RuntimeCapabilities(
                adapter="local-process",
                image_formats=("default",),
                isolation_policies=("default", "none"),
                gpu_enforcement=EnforcementMode.NONE,
            ),
            "bubblewrap": RuntimeCapabilities(
                adapter="bubblewrap",
                image_formats=("host",),
                isolation_policies=(
                    "default",
                    "host-stateless",
                    "host-stateful",
                ),
                gpu_enforcement=EnforcementMode.ADVISORY,
            ),
            "apptainer": RuntimeCapabilities(
                adapter="apptainer",
                isolation_policies=(
                    "default",
                    "image-stateless",
                    "image-stateful",
                ),
                gpu_enforcement=EnforcementMode.ADVISORY,
            ),
            "podman": RuntimeCapabilities(
                adapter="podman",
                isolation_policies=(
                    "default",
                    "image-stateless",
                    "image-stateful",
                ),
                gpu_enforcement=EnforcementMode.HARD,
            ),
        }

    def test_workload_failure_keep_alive_api_defaults_and_opt_out(self):
        base = {
            "pool": POOL,
            "user_id": "alice",
            "session_id": "failure-policy",
            "command": ["false"],
        }
        self.assertTrue(
            workload_spec_from_dict(base).keep_session_on_failure
        )
        explicit = dict(base, keep_session_on_failure=False)
        self.assertFalse(
            workload_spec_from_dict(explicit).keep_session_on_failure
        )
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            workload_spec_from_dict(
                dict(base, keep_session_on_failure="false")
            )

    def test_workload_failure_keep_alive_survives_journal_replay(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        register_gpu_node(ctrl, gpus=1)
        request = ctrl.submit_workload(
            spec(
                session_id="journal-failure-policy",
                keep_session_on_failure=False,
            )
        )

        replayed = CAPEController.reconcile_from_journal(journal)

        self.assertFalse(
            replayed.get_request(request.request_id)
            .spec.keep_session_on_failure
        )

    def test_workload_failure_kind_and_cached_session_survive_replay(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        register_gpu_node(ctrl, gpus=1)
        request = ctrl.submit_workload(
            spec(session_id="journal-workload-failure")
        )
        ctrl.schedule_once()
        assignment = ctrl.node_heartbeat("node-a")["commands"][0]
        ctrl.report_command_result(
            "node-a",
            assignment["command_id"],
            exit_code=7,
            stderr="compile failed",
            failure_kind="workload",
        )

        replayed = CAPEController.reconcile_from_journal(journal)
        status = replayed.request_status(request.request_id)

        self.assertEqual(status["state"], "FAILED")
        self.assertEqual(status["reason"], "command_failed")
        self.assertEqual(status["failure_kind"], "workload")
        self.assertTrue(status["keep_session_on_failure"])
        self.assertEqual(
            replayed.get_session(status["execution_session_id"]).state,
            SessionState.CACHED_IDLE,
        )
        replayed.assert_invariants()

    def test_multi_runtime_node_routes_policy_and_threads_selected_adapter(self):
        cases = (
            ("none", "local-process"),
            ("host-stateless", "bubblewrap"),
            ("image-stateless", "apptainer"),
        )
        for isolation_policy, expected_adapter in cases:
            with self.subTest(isolation_policy=isolation_policy):
                ctrl, _ = controller()
                register_gpu_node(
                    ctrl,
                    gpus=0,
                    runtime_caps=self.multi_runtime_capabilities(),
                )
                request = ctrl.submit_workload(
                    spec(
                        gpus=0,
                        session_id=isolation_policy,
                        isolation=isolation_policy,
                    )
                )

                ctrl.schedule_once()

                updated = ctrl.get_request(request.request_id)
                session = ctrl.get_session(updated.execution_session_id)
                assignment = ctrl.command_assignments[
                    ctrl.request_commands[request.request_id]
                ]
                self.assertEqual(session.runtime_adapter, expected_adapter)
                self.assertEqual(
                    assignment.session_spec["runtime_adapter"],
                    expected_adapter,
                )

    def test_multi_runtime_image_preference_and_explicit_pin(self):
        ctrl, _ = controller()
        register_gpu_node(
            ctrl,
            gpus=0,
            runtime_caps=self.multi_runtime_capabilities(),
        )
        preferred = ctrl.submit_workload(
            spec(gpus=0, session_id="preferred", isolation="image-stateless")
        )
        ctrl.schedule_once()
        self.assertEqual(
            ctrl.get_session(preferred.execution_session_id).runtime_adapter,
            "apptainer",
        )
        ctrl.complete_request(preferred.request_id)

        pinned = ctrl.submit_workload(
            spec(
                gpus=0,
                session_id="pinned",
                isolation="image-stateless",
                runtime="podman",
            )
        )
        ctrl.schedule_once()
        self.assertEqual(
            ctrl.get_session(pinned.execution_session_id).runtime_adapter,
            "podman",
        )

    def test_image_runtime_preference_is_applied_across_candidate_nodes(self):
        ctrl, _ = controller()
        register_gpu_node(
            ctrl,
            node_id="node-a-podman",
            gpus=0,
            caps=RuntimeCapabilities(
                adapter="podman",
                isolation_policies=("default", "image-stateless"),
            ),
        )
        register_gpu_node(
            ctrl,
            node_id="node-z-apptainer",
            gpus=0,
            caps=RuntimeCapabilities(
                adapter="apptainer",
                isolation_policies=("default", "image-stateless"),
            ),
        )
        request = ctrl.submit_workload(
            spec(gpus=0, isolation="image-stateless")
        )

        ctrl.schedule_once()

        self.assertEqual(request.node_id, "node-z-apptainer")
        self.assertEqual(
            ctrl.get_session(request.execution_session_id).runtime_adapter,
            "apptainer",
        )

    def test_multi_runtime_unsupported_policy_queues_without_lease(self):
        runtime_caps = self.multi_runtime_capabilities()
        runtime_caps["bubblewrap"] = RuntimeCapabilities(
            adapter="bubblewrap",
            image_formats=("host",),
            isolation_policies=("default", "host-stateless"),
        )
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, runtime_caps=runtime_caps)
        request = ctrl.submit_workload(
            spec(
                gpus=1,
                session_id="unsupported",
                isolation="host-stateful",
            )
        )

        ctrl.schedule_once()

        updated = ctrl.get_request(request.request_id)
        self.assertEqual(updated.state, RequestState.QUEUED)
        self.assertEqual(updated.reason, "waiting_for_runtime_capability")
        self.assertEqual(ctrl.lease_table, {})

    def test_gpu_request_requires_runtime_to_prove_the_requested_policy(self):
        runtime_caps = self.multi_runtime_capabilities()
        runtime_caps["bubblewrap"] = RuntimeCapabilities(
            adapter="bubblewrap",
            image_formats=("host",),
            isolation_policies=("default", "host-stateless", "host-stateful"),
            gpu_isolation_policies=("host-stateless",),
            can_select_gpus=True,
            gpu_enforcement=EnforcementMode.ADVISORY,
        )
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, runtime_caps=runtime_caps)
        request = ctrl.submit_workload(
            spec(
                gpus=1,
                session_id="gpu-policy-not-proven",
                isolation="host-stateful",
            )
        )

        ctrl.schedule_once()

        updated = ctrl.get_request(request.request_id)
        self.assertEqual(updated.state, RequestState.QUEUED)
        self.assertEqual(updated.reason, "waiting_for_runtime_capability")
        self.assertEqual(ctrl.lease_table, {})

    def test_cpu_request_can_use_policy_without_gpu_proof(self):
        runtime_caps = self.multi_runtime_capabilities()
        runtime_caps["bubblewrap"] = RuntimeCapabilities(
            adapter="bubblewrap",
            image_formats=("host",),
            isolation_policies=("default", "host-stateless", "host-stateful"),
            gpu_isolation_policies=("host-stateless",),
            can_select_gpus=True,
            gpu_enforcement=EnforcementMode.ADVISORY,
        )
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, runtime_caps=runtime_caps)
        request = ctrl.submit_workload(
            spec(
                gpus=0,
                session_id="cpu-policy",
                isolation="host-stateful",
            )
        )

        ctrl.schedule_once()

        updated = ctrl.get_request(request.request_id)
        self.assertEqual(updated.state, RequestState.RUNNING)
        self.assertEqual(
            ctrl.get_session(updated.execution_session_id).runtime_adapter,
            "bubblewrap",
        )

    def test_legacy_single_runtime_registration_still_routes(self):
        ctrl, _ = controller()
        register_gpu_node(
            ctrl,
            gpus=0,
            caps=RuntimeCapabilities(
                adapter="apptainer",
                isolation_policies=("default", "image-stateless"),
            ),
        )
        request = ctrl.submit_workload(
            spec(gpus=0, isolation="image-stateless")
        )

        ctrl.schedule_once()

        self.assertEqual(
            ctrl.get_session(request.execution_session_id).runtime_adapter,
            "apptainer",
        )

    def test_resource_request_rejects_unknown_isolation_policy(self):
        with self.assertRaisesRegex(ValueError, "isolation_policy"):
            ResourceRequest(isolation_policy="named-instance")

    def test_resource_request_validates_and_resolves_command_timeout(self):
        legacy = ResourceRequest(max_duration_seconds=120)
        self.assertIsNone(legacy.command_timeout_seconds)
        self.assertEqual(legacy.admission_timeout_seconds, 120)
        bounded = ResourceRequest(
            max_duration_seconds=120,
            command_timeout_seconds=10,
        )
        self.assertEqual(bounded.admission_timeout_seconds, 10)
        with self.assertRaisesRegex(ValueError, "command_timeout_seconds"):
            ResourceRequest(command_timeout_seconds=0)

    def test_explicit_isolation_policy_requires_advertised_node_capability(self):
        ctrl, _ = controller()
        register_gpu_node(
            ctrl,
            caps=RuntimeCapabilities(
                adapter="apptainer",
                isolation_policies=("default", "image-stateless"),
            ),
        )
        req = ctrl.submit_workload(
            spec(runtime="apptainer", isolation="image-stateful")
        )

        ctrl.schedule_once()

        self.assertEqual(ctrl.get_request(req.request_id).state, RequestState.QUEUED)
        self.assertEqual(
            ctrl.get_request(req.request_id).reason,
            "waiting_for_runtime_capability",
        )

    def test_explicit_isolation_policy_is_threaded_to_node_assignment(self):
        ctrl, _ = controller()
        register_gpu_node(
            ctrl,
            caps=RuntimeCapabilities(
                adapter="apptainer",
                isolation_policies=(
                    "default",
                    "image-stateless",
                    "image-stateful",
                ),
            ),
        )
        req = ctrl.submit_workload(
            spec(runtime="apptainer", isolation="image-stateful")
        )

        ctrl.schedule_once()

        updated = ctrl.get_request(req.request_id)
        self.assertEqual(updated.state, RequestState.RUNNING)
        session = ctrl.get_session(updated.execution_session_id)
        self.assertEqual(session.isolation_policy, "image-stateful")
        self.assertTrue(session.state_layer_id)
        assignment = ctrl.command_assignments[ctrl.request_commands[req.request_id]]
        self.assertEqual(
            assignment.session_spec["isolation_policy"], "image-stateful"
        )
        self.assertEqual(
            assignment.session_spec["state_layer_id"], session.state_layer_id
        )
        self.assertEqual(
            ctrl.request_status(req.request_id)["isolation_policy"],
            "image-stateful",
        )

    def test_session_reuse_identity_includes_isolation_policy_and_state_layer(self):
        ctrl, _ = controller()
        register_gpu_node(
            ctrl,
            gpus=2,
            caps=RuntimeCapabilities(
                adapter="apptainer",
                isolation_policies=(
                    "default",
                    "image-stateless",
                    "image-stateful",
                ),
            ),
        )
        first = ctrl.submit_workload(
            spec(
                session_id="same",
                runtime="apptainer",
                isolation="image-stateful",
            )
        )
        ctrl.schedule_once()
        first_execution_session_id = ctrl.get_request(first.request_id).execution_session_id
        first_layer_id = ctrl.get_session(first_execution_session_id).state_layer_id
        ctrl.complete_request(first.request_id, exit_code=0)

        same = ctrl.submit_workload(
            spec(
                session_id="same",
                runtime="apptainer",
                isolation="image-stateful",
            )
        )
        ctrl.schedule_once()
        self.assertEqual(
            ctrl.get_request(same.request_id).execution_session_id, first_execution_session_id
        )
        self.assertEqual(
            ctrl.get_session(first_execution_session_id).state_layer_id, first_layer_id
        )
        ctrl.complete_request(same.request_id, exit_code=0)

        changed = ctrl.submit_workload(
            spec(
                session_id="same",
                runtime="apptainer",
                isolation="image-stateless",
            )
        )
        ctrl.schedule_once()
        self.assertNotEqual(
            ctrl.get_request(changed.request_id).execution_session_id, first_execution_session_id
        )

    def test_workload_failure_keeps_reusable_session_cached_by_default(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, cpu=8, mem=64)
        first = ctrl.submit_workload(
            spec(session_id="failure-reuse", gpus=1, cpu=2, mem=4)
        )
        ctrl.schedule_once()
        execution_session_id = first.execution_session_id

        ctrl.complete_request(
            first.request_id,
            exit_code=7,
            stderr="compile failed",
            failure_kind="workload",
        )

        self.assertEqual(first.state, RequestState.FAILED)
        self.assertEqual(first.reason, "command_failed")
        self.assertTrue(first.spec.keep_session_on_failure)
        self.assertEqual(
            ctrl.get_session(execution_session_id).state, SessionState.CACHED_IDLE
        )
        self.assertEqual(ctrl.get_session(execution_session_id).active_request_id, None)
        self.assertEqual(ctrl.lease_table, {})
        ctrl.assert_invariants()

        second = ctrl.submit_workload(
            spec(session_id="failure-reuse", gpus=1, cpu=2, mem=4)
        )
        ctrl.schedule_once()
        self.assertEqual(second.execution_session_id, execution_session_id)

    def test_workload_failure_opt_out_stops_session(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, cpu=8, mem=64)
        first = ctrl.submit_workload(
            spec(
                session_id="failure-opt-out",
                keep_session_on_failure=False,
            )
        )
        ctrl.schedule_once()
        execution_session_id = first.execution_session_id

        ctrl.complete_request(
            first.request_id,
            exit_code=2,
            failure_kind="workload",
        )

        self.assertEqual(first.reason, "command_failed")
        self.assertEqual(
            ctrl.get_session(execution_session_id).state, SessionState.STOPPED
        )
        second = ctrl.submit_workload(
            spec(
                session_id="failure-opt-out",
                keep_session_on_failure=False,
            )
        )
        ctrl.schedule_once()
        self.assertNotEqual(second.execution_session_id, execution_session_id)

    def test_workload_failure_non_reusable_stops_session(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1)
        request = ctrl.submit_workload(
            spec(session_id="failure-no-reuse", reuse=False)
        )
        ctrl.schedule_once()
        execution_session_id = request.execution_session_id

        ctrl.complete_request(
            request.request_id,
            exit_code=4,
            failure_kind="workload",
        )

        self.assertEqual(request.reason, "command_failed")
        self.assertEqual(
            ctrl.get_session(execution_session_id).state, SessionState.STOPPED
        )

    def test_infrastructure_failure_marks_session_failed_and_releases_it(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, cpu=8, mem=64)
        request = ctrl.submit_workload(
            spec(session_id="infrastructure-failure")
        )
        ctrl.schedule_once()
        execution_session_id = request.execution_session_id

        ctrl.complete_request(
            request.request_id,
            exit_code=127,
            stderr="runtime launch failed",
            failure_kind="infrastructure",
        )

        self.assertEqual(request.state, RequestState.FAILED)
        self.assertEqual(request.reason, "infrastructure_failed")
        self.assertEqual(
            ctrl.get_session(execution_session_id).state, SessionState.FAILED
        )
        self.assertEqual(ctrl.get_node("node-a").active_sessions, [])
        self.assertEqual(ctrl.lease_table, {})
        ctrl.assert_invariants()

    def test_failure_kind_rejects_unknown_and_zero_exit_infrastructure(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1)
        request = ctrl.submit_workload(spec(session_id="invalid-kind"))
        ctrl.schedule_once()
        with self.assertRaisesRegex(ValueError, "failure_kind"):
            ctrl.complete_request(
                request.request_id,
                exit_code=1,
                failure_kind="unknown",
            )
        with self.assertRaisesRegex(ValueError, "nonzero"):
            ctrl.complete_request(
                request.request_id,
                exit_code=0,
                failure_kind="infrastructure",
            )

    def test_in_process_node_agent_registers_heartbeats_and_reports_status(self):
        ctrl, _ = controller()
        agent = InProcessNodeAgent(
            ctrl,
            node_id="node-agent-a",
            pool=POOL,
            resources=NodeResources(cpu_cores=8, memory_gb=64, gpu_uuids=gpu_ids("node-agent-a", 1)),
            capabilities=RuntimeCapabilities(adapter="fake", gpu_enforcement=EnforcementMode.HARD),
        )
        node = agent.register()
        self.assertEqual(node.node_id, "node-agent-a")
        self.assertEqual(agent.heartbeat().node_id, "node-agent-a")
        status = agent.collect_status()
        self.assertEqual(status["node_id"], "node-agent-a")
        self.assertEqual(status["available_cpu"], 8)

    def test_sqlite_journal_accepts_appends_from_service_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = SQLiteJournal(os.path.join(tmp, "journal.sqlite"), batch_size=3)
            errors = []

            def append_events(worker):
                try:
                    for i in range(8):
                        journal.append(EventRecord(
                            float(worker * 100 + i),
                            "thread_event",
                            "worker-{}-{}".format(worker, i),
                            state="OK",
                            details={"worker": worker, "index": i},
                        ))
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=append_events, args=(i,)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            try:
                self.assertEqual(errors, [])
                self.assertEqual(journal.count("thread_event"), 32)
                self.assertEqual(len(journal.list_events()), 32)
            finally:
                journal.close()

    def test_submit_without_nodes_is_queued_without_lease(self):
        ctrl, _ = controller()
        req = ctrl.submit_workload(spec())
        self.assertEqual(req.state, RequestState.QUEUED)
        self.assertEqual(req.reason, "pool_empty")
        self.assertEqual(req.gpu_uuids, ())
        self.assertIsNone(req.execution_session_id)
        status = ctrl.request_status(req.request_id)
        self.assertEqual(status["state"], "QUEUED")
        self.assertEqual(status["reason"], "pool_empty")
        self.assertEqual(
            status["status_message"],
            "QUEUED (pool_empty): no active node is ready in the pool yet; waiting for a node to register",
        )
        self.assertEqual(ctrl.schedule_once(), 0)
        ctrl.assert_invariants()

    def test_request_execution_timestamps_survive_terminal_status_and_replay(self):
        journal = MemoryJournal()
        ctrl, clock = controller(journal=journal)
        req = ctrl.submit_workload(spec(gpus=0, session_id="timestamps"))
        queued = ctrl.request_status(req.request_id)
        self.assertIsNone(queued["started_at"])
        self.assertIsNone(queued["ended_at"])

        register_gpu_node(ctrl, gpus=0)
        clock.advance(2)
        ctrl.schedule_once()
        running = ctrl.request_status(req.request_id)
        self.assertEqual(running["started_at"], 2.0)
        self.assertIsNone(running["ended_at"])

        clock.advance(1)
        ctrl.schedule_once()
        self.assertEqual(ctrl.request_status(req.request_id)["started_at"], 2.0)

        clock.advance(3)
        ctrl.complete_request(req.request_id, exit_code=0)
        terminal = ctrl.request_status(req.request_id)
        self.assertEqual(terminal["started_at"], 2.0)
        self.assertEqual(terminal["ended_at"], 6.0)
        self.assertEqual(terminal["updated_at"], 6.0)
        self.assertLessEqual(terminal["created_at"], terminal["leased_at"])
        self.assertLessEqual(terminal["leased_at"], terminal["started_at"])
        self.assertLessEqual(terminal["started_at"], terminal["ended_at"])

        replayed = CAPEController.reconcile_from_journal(
            journal, clock=FakeClock(100)
        )
        replayed_terminal = replayed.request_status(req.request_id)
        self.assertEqual(replayed_terminal["started_at"], 2.0)
        self.assertEqual(replayed_terminal["ended_at"], 6.0)

    def test_never_run_cancelled_and_failed_requests_keep_null_started_at(self):
        ctrl, clock = controller()
        cancelled = ctrl.submit_workload(
            spec(gpus=0, session_id="cancelled-before-run")
        )
        clock.advance(4)
        ctrl.cancel_request(cancelled.request_id)
        cancelled_status = ctrl.request_status(cancelled.request_id)
        self.assertIsNone(cancelled_status["started_at"])
        self.assertEqual(cancelled_status["ended_at"], 4.0)

        failed = ctrl.submit_workload(
            spec(gpus=0, session_id="failed-before-run")
        )
        clock.advance(2)
        ctrl._fail_running_request(
            failed,
            "pre_run_failure",
            RuntimeResult(1, "", "failed before execution"),
        )
        failed_status = ctrl.request_status(failed.request_id)
        self.assertEqual(failed_status["state"], "FAILED")
        self.assertIsNone(failed_status["started_at"])
        self.assertEqual(failed_status["ended_at"], 6.0)

    def test_infeasible_request_is_terminal_without_a_started_at(self):
        ctrl, _ = controller(
            policy=SchedulerPolicy(max_queued_requests_per_user=0)
        )
        request = ctrl.submit_workload(spec(gpus=0))
        status = ctrl.request_status(request.request_id)
        self.assertEqual(status["state"], "INFEASIBLE")
        self.assertIsNone(status["started_at"])
        self.assertEqual(status["ended_at"], status["created_at"])

    def test_dynamic_node_registration_re_evaluates_queued_request(self):
        ctrl, _ = controller()
        req = ctrl.submit_workload(spec())
        register_gpu_node(ctrl)
        self.assertEqual(ctrl.schedule_once(), 1)
        updated = ctrl.get_request(req.request_id)
        self.assertEqual(updated.state, RequestState.RUNNING)
        self.assertEqual(len(updated.gpu_uuids), 1)
        self.assertEqual(ctrl.active_gpu_lease_count(), 1)
        ctrl.assert_invariants()

    def test_whole_gpu_leases_are_unique(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=4)
        requests = [ctrl.submit_workload(spec(user="user-{}".format(i), session_id="task-{}".format(i))) for i in range(4)]
        self.assertEqual(ctrl.schedule_once(), 4)
        leased = []
        for req in requests:
            leased.extend(ctrl.get_request(req.request_id).gpu_uuids)
        self.assertEqual(len(leased), len(set(leased)))
        self.assertEqual(len(ctrl.lease_table), 4)
        ctrl.assert_invariants()

    def test_load_spreads_equivalent_one_gpu_sessions(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, node_id="node-a", gpus=2)
        register_gpu_node(ctrl, node_id="node-b", gpus=2)
        a = ctrl.submit_workload(spec(user="u1", session_id="a"))
        b = ctrl.submit_workload(spec(user="u2", session_id="b"))
        ctrl.schedule_once()
        self.assertNotEqual(ctrl.get_request(a.request_id).node_id, ctrl.get_request(b.request_id).node_id)

    def test_multi_gpu_request_uses_one_node(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, node_id="node-a", gpus=2)
        register_gpu_node(ctrl, node_id="node-b", gpus=4)
        req = ctrl.submit_workload(spec(gpus=4, session_id="big"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).node_id, "node-b")
        self.assertEqual(len(ctrl.get_request(req.request_id).gpu_uuids), 4)

    def test_warm_session_reuse_keeps_execution_session_id(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl)
        first = ctrl.submit_workload(spec(session_id="same"))
        ctrl.schedule_once()
        first_session = ctrl.get_request(first.request_id).execution_session_id
        ctrl.complete_request(first.request_id, exit_code=0)
        cached = ctrl.get_session(first_session)
        self.assertEqual(cached.state, SessionState.CACHED_IDLE)
        self.assertEqual(cached.gpu_uuids, ())
        self.assertEqual(cached.gpu_count, 1)
        self.assertEqual(ctrl.active_gpu_lease_count(), 0)
        second = ctrl.submit_workload(spec(session_id="same"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(second.request_id).execution_session_id, first_session)
        self.assertEqual(ctrl.get_session(first_session).state, SessionState.RUNNING_COMMAND)
        self.assertEqual(ctrl.active_gpu_lease_count(), 1)
        self.assertEqual(len(ctrl.sessions), 1)

    def test_same_reuse_identity_waits_for_its_active_session(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, node_id="node-a", gpus=1)
        register_gpu_node(ctrl, node_id="node-b", gpus=1)
        first = ctrl.submit_workload(
            spec(user="alice", session_id="same")
        )
        ctrl.schedule_once()
        first_session = ctrl.get_request(first.request_id).execution_session_id

        second = ctrl.submit_workload(
            spec(user="alice", session_id="same")
        )
        ctrl.schedule_once()

        self.assertEqual(ctrl.get_request(second.request_id).state, RequestState.QUEUED)
        self.assertEqual(ctrl.get_request(second.request_id).reason, "waiting_for_session")
        self.assertEqual(len(ctrl.sessions), 1)

        ctrl.complete_request(first.request_id, exit_code=0)
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(second.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(second.request_id).execution_session_id, first_session)
        self.assertEqual(len(ctrl.sessions), 1)

    def test_reuse_rejected_when_image_changes(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=2)
        first = ctrl.submit_workload(spec(session_id="same", image="docker://image-a"))
        ctrl.schedule_once()
        first_session = ctrl.get_request(first.request_id).execution_session_id
        ctrl.complete_request(first.request_id, exit_code=0)
        second = ctrl.submit_workload(spec(session_id="same", image="docker://image-b"))
        ctrl.schedule_once()
        self.assertNotEqual(ctrl.get_request(second.request_id).execution_session_id, first_session)
        self.assertEqual(len(ctrl.sessions), 2)

    def test_fairness_prefers_user_without_active_gpu_session(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=2)
        running = ctrl.submit_workload(spec(user="alice", session_id="first"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(running.request_id).state, RequestState.RUNNING)
        alice_second = ctrl.submit_workload(spec(user="alice", session_id="second"))
        bob_first = ctrl.submit_workload(spec(user="bob", session_id="first"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(bob_first.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(alice_second.request_id).state, RequestState.QUEUED)

    def test_active_gpu_quota_blocks_new_gpu_session(self):
        policy = SchedulerPolicy(max_active_gpu_sessions_per_user=1)
        ctrl, _ = controller(policy=policy)
        register_gpu_node(ctrl, gpus=2)
        first = ctrl.submit_workload(spec(user="alice", session_id="first"))
        ctrl.schedule_once()
        second = ctrl.submit_workload(spec(user="alice", session_id="second"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(second.request_id).state, RequestState.QUEUED)
        self.assertEqual(ctrl.get_request(second.request_id).reason, "quota_exceeded")

    def test_repeated_waiting_reason_does_not_emit_duplicate_journal_events(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        register_gpu_node(ctrl, gpus=1)
        first = ctrl.submit_workload(
            spec(user="alice", session_id="running")
        )
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(first.request_id).state, RequestState.RUNNING)

        waiting = ctrl.submit_workload(
            spec(user="bob", session_id="waiting")
        )
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(waiting.request_id).reason, "waiting_for_gpu")
        event_count = journal.count()

        ctrl.schedule_once()
        ctrl.schedule_once()

        self.assertEqual(
            journal.count(),
            event_count,
            [
                (event.event_type, event.entity_id, event.state, event.reason)
                for event in journal.list_events()[event_count:]
            ],
        )


    def test_cpu_only_request_is_not_blocked_by_active_gpu_quota(self):
        policy = SchedulerPolicy(max_active_gpu_sessions_per_user=1)
        ctrl, _ = controller(policy=policy)
        register_gpu_node(ctrl, gpus=1)
        gpu_req = ctrl.submit_workload(spec(user="alice", session_id="gpu", gpus=1))
        ctrl.schedule_once()
        cpu_req = ctrl.submit_workload(spec(user="alice", session_id="cpu", gpus=0))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(gpu_req.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(cpu_req.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(cpu_req.request_id).gpu_uuids, ())

    def test_warm_reuse_wins_under_active_gpu_quota(self):
        policy = SchedulerPolicy(max_active_gpu_sessions_per_user=1)
        ctrl, _ = controller(policy=policy)
        register_gpu_node(ctrl, gpus=1)
        first = ctrl.submit_workload(spec(user="alice", session_id="same", gpus=1))
        ctrl.schedule_once()
        execution_session_id = ctrl.get_request(first.request_id).execution_session_id
        ctrl.complete_request(first.request_id, exit_code=0)
        second = ctrl.submit_workload(spec(user="alice", session_id="same", gpus=1))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(second.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(second.request_id).execution_session_id, execution_session_id)

    def test_active_gpu_quota_counts_gpu_leases_not_sessions(self):
        policy = SchedulerPolicy(max_active_gpu_sessions_per_user=2)
        ctrl, _ = controller(policy=policy)
        register_gpu_node(ctrl, gpus=4)
        first = ctrl.submit_workload(spec(user="alice", session_id="two-gpu", gpus=2))
        ctrl.schedule_once()
        second = ctrl.submit_workload(spec(user="alice", session_id="one-more", gpus=1))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(first.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(second.request_id).state, RequestState.QUEUED)
        self.assertEqual(ctrl.get_request(second.request_id).reason, "quota_exceeded")

    def test_deadline_drain_blocks_new_sessions(self):
        ctrl, clock = controller()
        register_gpu_node(ctrl, deadline=100.0, drain_before=20, cleanup=5)
        clock.advance(85)
        req = ctrl.submit_workload(spec(session_id="late"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_node("node-a").state, NodeState.DRAINING)
        self.assertEqual(ctrl.get_request(req.request_id).reason, "node_draining")

    def test_lifetime_longer_than_safe_deadline_is_not_placed(self):
        ctrl, clock = controller()
        register_gpu_node(ctrl, deadline=150.0, drain_before=0, cleanup=10)
        clock.advance(60)
        req = ctrl.submit_workload(spec(session_id="too-long"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).reason, "allocation_expiring")

    def test_command_timeout_is_admission_operand_not_session_lifetime(self):
        ctrl, clock = controller()
        register_gpu_node(ctrl, deadline=150.0, drain_before=0, cleanup=10)
        clock.advance(60)
        accepted = ctrl.submit_workload(
            spec(
                session_id="short-command-long-session",
                max_duration=100,
                command_timeout=10,
            )
        )
        ctrl.schedule_once()
        self.assertEqual(accepted.state, RequestState.RUNNING)
        self.assertEqual(
            ctrl.get_session(accepted.execution_session_id).max_duration_seconds,
            100,
        )

        other, other_clock = controller()
        register_gpu_node(other, deadline=150.0, drain_before=0, cleanup=10)
        other_clock.advance(60)
        rejected = other.submit_workload(
            spec(
                session_id="long-command",
                max_duration=100,
                command_timeout=90,
            )
        )
        other.schedule_once()
        self.assertEqual(rejected.reason, "allocation_expiring")

    def test_node_without_deadline_never_drains_or_blocks_for_deadline(self):
        ctrl, clock = controller()
        node = register_gpu_node(ctrl, deadline=None, drain_before=20, cleanup=10)
        clock.advance(1000000)
        ctrl.refresh_deadlines()
        request = ctrl.submit_workload(
            spec(session_id="unbounded", max_duration=100000)
        )
        ctrl.schedule_once()
        self.assertIsNone(node.allocation_deadline)
        self.assertEqual(node.state, NodeState.ACTIVE)
        self.assertEqual(request.state, RequestState.RUNNING)

    def test_explicit_drain_prevents_placement(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl)
        ctrl.drain_node("node-a")
        req = ctrl.submit_workload(spec(session_id="drained"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).reason, "node_draining")

    def test_explicit_drain_stops_cached_idle_sessions(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, cpu=8, mem=64)
        request = ctrl.submit_workload(
            spec(
                session_id="cached-before-drain",
                gpus=1,
                cpu=2,
                mem=4,
            )
        )
        ctrl.schedule_once()
        assignment = ctrl.node_heartbeat("node-a")["commands"][0]
        ctrl.report_command_result(
            "node-a", assignment["command_id"], exit_code=0
        )
        execution_session_id = request.execution_session_id

        self.assertEqual(
            ctrl.get_session(execution_session_id).state,
            SessionState.CACHED_IDLE,
        )
        self.assertEqual(ctrl.get_node("node-a").active_sessions, [execution_session_id])

        ctrl.drain_node("node-a")

        node = ctrl.get_node("node-a")
        self.assertEqual(node.state, NodeState.DRAINING)
        self.assertEqual(node.active_sessions, [])
        self.assertEqual(node.used_cpu_cores, 0)
        self.assertEqual(node.used_memory_gb, 0)
        self.assertEqual(
            ctrl.get_session(execution_session_id).state,
            SessionState.STOPPED,
        )
        self.assertEqual(
            ctrl.node_heartbeat("node-a")["commands"],
            [{"action": "stop_session", "execution_session_id": execution_session_id, "suspend_version": 1}],
        )
        ctrl.assert_invariants()

    def test_running_session_is_not_preempted_but_cannot_cache_after_drain(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, cpu=8, mem=64)
        request = ctrl.submit_workload(
            spec(
                session_id="running-during-drain",
                gpus=1,
                cpu=2,
                mem=4,
            )
        )
        ctrl.schedule_once()
        assignment = ctrl.node_heartbeat("node-a")["commands"][0]
        execution_session_id = request.execution_session_id

        ctrl.drain_node("node-a")

        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertEqual(
            ctrl.get_session(execution_session_id).state,
            SessionState.RUNNING_COMMAND,
        )
        self.assertEqual(
            ctrl.get_node("node-a").active_sessions,
            [execution_session_id],
        )
        self.assertEqual(ctrl.node_control_queues["node-a"], [])

        ctrl.report_command_result(
            "node-a", assignment["command_id"], exit_code=0
        )

        node = ctrl.get_node("node-a")
        self.assertEqual(request.state, RequestState.COMPLETED)
        self.assertEqual(
            ctrl.get_session(execution_session_id).state,
            SessionState.STOPPED,
        )
        self.assertEqual(node.active_sessions, [])
        self.assertEqual(node.used_cpu_cores, 0)
        self.assertEqual(node.used_memory_gb, 0)
        self.assertEqual(ctrl.lease_table, {})
        self.assertEqual(
            ctrl.node_heartbeat("node-a")["commands"],
            [{"action": "stop_session", "execution_session_id": execution_session_id, "suspend_version": 1}],
        )
        ctrl.assert_invariants()

    def test_deadline_drain_stops_cached_idle_sessions(self):
        ctrl, clock = controller()
        register_gpu_node(
            ctrl,
            gpus=1,
            deadline=200.0,
            drain_before=20,
            cleanup=5,
        )
        request = ctrl.submit_workload(
            spec(session_id="cached-before-deadline-drain")
        )
        ctrl.schedule_once()
        assignment = ctrl.node_heartbeat("node-a")["commands"][0]
        ctrl.report_command_result(
            "node-a", assignment["command_id"], exit_code=0
        )
        execution_session_id = request.execution_session_id
        clock.advance(180)

        ctrl.refresh_deadlines()

        self.assertEqual(
            ctrl.get_node("node-a").state,
            NodeState.DRAINING,
        )
        self.assertEqual(
            ctrl.get_session(execution_session_id).state,
            SessionState.STOPPED,
        )
        self.assertEqual(ctrl.get_node("node-a").active_sessions, [])
        self.assertEqual(
            ctrl.node_heartbeat("node-a")["commands"],
            [{"action": "stop_session", "execution_session_id": execution_session_id, "suspend_version": 1}],
        )
        ctrl.assert_invariants()

    def test_heartbeat_timeout_and_loss_release_lease_safely(self):
        policy = SchedulerPolicy(heartbeat_timeout_seconds=10, lost_timeout_seconds=20)
        ctrl, clock = controller(policy=policy)
        register_gpu_node(ctrl)
        req = ctrl.submit_workload(spec(session_id="running"))
        ctrl.schedule_once()
        leased_gpu = ctrl.get_request(req.request_id).gpu_uuids[0]
        clock.advance(11)
        ctrl.check_heartbeats()
        self.assertEqual(ctrl.get_node("node-a").state, NodeState.UNHEALTHY)
        clock.advance(10)
        ctrl.check_heartbeats()
        self.assertEqual(ctrl.get_node("node-a").state, NodeState.LOST)
        lost = ctrl.get_request(req.request_id)
        self.assertEqual(lost.state, RequestState.LOST)
        self.assertEqual(lost.failure_kind, "infrastructure")
        self.assertIsNone(lost.exit_code)
        self.assertNotIn(leased_gpu, ctrl.lease_table)
        self.assertEqual(ctrl.get_node("node-a").active_sessions, [])
        self.assertIsNotNone(lost.started_at)
        self.assertEqual(lost.ended_at, lost.updated_at)
        self.assertLessEqual(lost.started_at, lost.ended_at)

    def test_cancel_queued_and_running_requests(self):
        ctrl, _ = controller()
        queued = ctrl.submit_workload(spec(session_id="q"))
        ctrl.cancel_request(queued.request_id)
        queued_cancelled = ctrl.get_request(queued.request_id)
        self.assertEqual(queued_cancelled.state, RequestState.CANCELLED)
        self.assertIsNone(queued_cancelled.started_at)
        self.assertEqual(queued_cancelled.ended_at, queued_cancelled.updated_at)
        register_gpu_node(ctrl)
        running = ctrl.submit_workload(spec(session_id="r"))
        ctrl.schedule_once()
        execution_session_id = ctrl.get_request(running.request_id).execution_session_id
        ctrl.cancel_request(running.request_id)
        running_cancelled = ctrl.get_request(running.request_id)
        self.assertIsNotNone(running_cancelled.started_at)
        self.assertEqual(
            running_cancelled.ended_at, running_cancelled.updated_at
        )
        self.assertEqual(ctrl.get_request(running.request_id).state, RequestState.CANCELLED)
        self.assertEqual(ctrl.get_session(execution_session_id).state, SessionState.STOPPED)
        self.assertEqual(ctrl.lease_table, {})

    def test_cpu_only_request_consumes_no_gpu_lease(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl)
        req = ctrl.submit_workload(spec(gpus=0, session_id="cpu"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).state, RequestState.RUNNING)
        self.assertEqual(ctrl.get_request(req.request_id).gpu_uuids, ())
        self.assertEqual(ctrl.lease_table, {})

    def test_hard_enforcement_request_rejects_advisory_node(self):
        ctrl, _ = controller()
        caps = RuntimeCapabilities(adapter="fake", gpu_enforcement=EnforcementMode.ADVISORY)
        register_gpu_node(ctrl, caps=caps)
        req = ctrl.submit_workload(spec(hard=True, session_id="hard"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).state, RequestState.QUEUED)
        self.assertEqual(ctrl.get_request(req.request_id).reason, "waiting_for_runtime_capability")

    def test_default_image_is_part_of_execution_session_identity(self):
        policy = SchedulerPolicy(default_image="docker://ubuntu:latest")
        ctrl, _ = controller(policy=policy)
        register_gpu_node(ctrl)
        req = ctrl.submit_workload(spec(session_id="default-image", image=None, profile=None))
        ctrl.schedule_once()
        session = ctrl.get_session(ctrl.get_request(req.request_id).execution_session_id)
        self.assertEqual(session.image_identity, "docker://ubuntu:latest")

    def test_bind_mount_requires_capability(self):
        ctrl, _ = controller()
        caps = RuntimeCapabilities(adapter="fake", can_bind_mounts=False)
        register_gpu_node(ctrl, caps=caps)
        mount = MountSpec("/workspace", "/workspace")
        workload = spec(mounts=(mount,), session_id="mount")
        workload.workspace["target"] = "/cape-workspace"
        req = ctrl.submit_workload(workload)
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).reason, "waiting_for_runtime_capability")

    def test_idle_session_expiration_releases_resources(self):
        policy = SchedulerPolicy(idle_session_timeout_seconds=10)
        ctrl, clock = controller(policy=policy)
        register_gpu_node(ctrl)
        req = ctrl.submit_workload(spec(session_id="idle"))
        ctrl.schedule_once()
        execution_session_id = ctrl.get_request(req.request_id).execution_session_id
        ctrl.complete_request(req.request_id, exit_code=0)
        self.assertEqual(ctrl.get_session(execution_session_id).state, SessionState.CACHED_IDLE)
        self.assertEqual(ctrl.lease_table, {})
        clock.advance(10)
        expired = ctrl.expire_idle_sessions()
        self.assertEqual(expired, [execution_session_id])
        self.assertEqual(ctrl.lease_table, {})

    def test_idle_session_expiration_tells_remote_agent_to_retain_state_layer(self):
        policy = SchedulerPolicy(idle_session_timeout_seconds=10)
        ctrl, clock = controller(policy=policy)
        register_gpu_node(
            ctrl,
            caps=RuntimeCapabilities(
                adapter="fake",
                isolation_policies=("default", "image-stateful"),
            ),
        )
        req = ctrl.submit_workload(
            spec(
                session_id="overlay-idle",
                isolation="image-stateful",
            )
        )
        ctrl.schedule_once()
        assignment = ctrl.node_heartbeat("node-a")["commands"][0]
        ctrl.report_command_result(
            "node-a", assignment["command_id"], exit_code=0
        )
        execution_session_id = ctrl.get_request(req.request_id).execution_session_id

        clock.advance(10)
        ctrl.expire_idle_sessions()
        controls = ctrl.node_heartbeat("node-a")["commands"]

        self.assertEqual(
            controls,
            [{"action": "suspend_session", "execution_session_id": execution_session_id, "suspend_version": 1}],
        )

    def test_node_reregistration_preserves_cached_sessions_for_forced_cleanup(self):
        ctrl, _ = controller()
        resources = NodeResources(
            cpu_cores=8,
            memory_gb=64,
            gpu_uuids=gpu_ids("node-a", 1),
        )
        capabilities = RuntimeCapabilities(
            adapter="fake",
            gpu_enforcement=EnforcementMode.HARD,
        )
        ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
        )
        request = ctrl.submit_workload(
            spec(gpus=1, cpu=2, mem=4, session_id="survive-reregister")
        )
        ctrl.schedule_once()
        execution_session_id = ctrl.get_request(request.request_id).execution_session_id
        assignment = ctrl.node_heartbeat("node-a")["commands"][0]
        ctrl.report_command_result(
            "node-a", assignment["command_id"], exit_code=0
        )

        node = ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
        )

        self.assertEqual(node.active_sessions, [execution_session_id])
        self.assertEqual(node.used_cpu_cores, 2)
        self.assertEqual(node.used_memory_gb, 4)
        ctrl.remove_allocation("alloc-a", force=True)
        controls = ctrl.node_heartbeat("node-a")["commands"]
        self.assertEqual(
            controls,
            [{"action": "stop_session", "execution_session_id": execution_session_id, "suspend_version": 1}],
        )
        self.assertEqual(ctrl.get_session(execution_session_id).state, SessionState.STOPPED)
        ctrl.assert_invariants()

    def test_node_reregistration_reconciles_inflight_sessions(self):
        policy = SchedulerPolicy(
            reconcile_missing_grace_seconds=2,
            reconcile_missing_heartbeats=2,
        )
        ctrl, clock = controller(policy=policy)
        resources = NodeResources(
            cpu_cores=8,
            memory_gb=64,
            gpu_uuids=gpu_ids("node-a", 1),
        )
        capabilities = RuntimeCapabilities(
            adapter="fake",
            gpu_enforcement=EnforcementMode.HARD,
        )
        ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
            agent_epoch="epoch-a",
        )
        request = ctrl.submit_workload(
            spec(gpus=1, session_id="running-during-reregister")
        )
        ctrl.schedule_once()
        ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
            agent_epoch="epoch-b",
        )

        session = ctrl.get_session(request.execution_session_id)
        self.assertEqual(session.state, SessionState.RECONCILING)
        self.assertEqual(request.state, RequestState.RECONCILING)
        self.assertEqual(request.reason, "node_reregistered")

        ctrl.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            agent_epoch="epoch-b",
        )

        self.assertEqual(session.state, SessionState.RECONCILING)
        self.assertEqual(request.state, RequestState.RECONCILING)

        clock.advance(2.1)
        ctrl.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            agent_epoch="epoch-b",
        )

        self.assertEqual(session.state, SessionState.FAILED)
        self.assertEqual(request.state, RequestState.FAILED)
        self.assertEqual(request.reason, "reconcile_missing")
        self.assertEqual(ctrl.lease_table, {})
        ctrl.assert_invariants()

    def test_same_epoch_registration_retry_does_not_reconcile_dispatched_request(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        resources = NodeResources(
            cpu_cores=8,
            memory_gb=64,
            gpu_uuids=gpu_ids("node-a", 1),
        )
        capabilities = RuntimeCapabilities(
            adapter="fake",
            gpu_enforcement=EnforcementMode.HARD,
        )
        request = ctrl.submit_workload(
            spec(gpus=1, session_id="queued-before-retry")
        )
        self.assertEqual(request.state, RequestState.QUEUED)
        self.assertEqual(request.reason, "pool_empty")
        ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
            agent_epoch="process-a",
        )
        ctrl.schedule_once()

        refreshed = ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
            agent_epoch="process-a",
        )

        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertIsNone(request.reason)
        self.assertEqual(
            ctrl.get_session(request.execution_session_id).state,
            SessionState.RUNNING_COMMAND,
        )
        self.assertEqual(refreshed.agent_epoch, "process-a")
        self.assertEqual(
            [
                event.event_type
                for event in journal.list_events()
                if event.entity_id == "node-a"
            ],
            ["node_registered", "node_refreshed"],
        )

        heartbeat = ctrl.node_heartbeat(
            "node-a",
            status={"authoritative": True, "sessions": []},
            agent_epoch="process-a",
        )
        self.assertEqual(len(heartbeat["commands"]), 1)
        self.assertEqual(request.state, RequestState.RUNNING)
        ctrl.report_command_result(
            "node-a", heartbeat["commands"][0]["command_id"], exit_code=0
        )
        self.assertEqual(request.state, RequestState.COMPLETED)
        self.assertNotIn(
            "reconcile_missing",
            [event.reason for event in journal.list_events()],
        )

    def test_registering_node_is_visible_but_not_schedulable_until_active(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        request = ctrl.submit_workload(
            spec(gpus=1, session_id="wait-for-probe")
        )
        self.assertEqual(request.reason, "pool_empty")
        resources = NodeResources(
            cpu_cores=8,
            memory_gb=64,
            gpu_uuids=gpu_ids("node-a", 1),
        )
        capabilities = RuntimeCapabilities(
            adapter="fake",
            gpu_enforcement=EnforcementMode.HARD,
        )

        ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=RuntimeCapabilities(
                adapter="probing",
                image_formats=(),
                isolation_policies=(),
                nested_container=False,
                can_bind_mounts=False,
                can_select_gpus=False,
                gpu_enforcement=EnforcementMode.NONE,
                can_enforce_cpu_memory=False,
                node_local_cache=False,
                supports_warm_sessions=False,
            ),
            allocation_id="alloc-a",
            agent_epoch="process-a",
            state=NodeState.REGISTERING,
        )
        ctrl.schedule_once()

        self.assertEqual(request.state, RequestState.QUEUED)
        self.assertEqual(request.reason, "node_probing")
        summary = ctrl.pool_utilization(pool=POOL)
        self.assertEqual(summary["nodes"], 1)
        self.assertEqual(summary["active_nodes"], 0)
        self.assertEqual(summary["registering_nodes"], 1)
        self.assertEqual(summary["gpus_schedulable"], 0)
        self.assertEqual(
            summary["node_states"],
            [{
                "node_id": "node-a",
                "allocation_id": "alloc-a",
                "allocation_deadline": None,
                "state": "REGISTERING",
                "gpus_total": 1,
                "gpus_leased": 0,
                "gpus_available": 0,
                "cpu_total": 8,
                "cpu_available": 0,
                "memory_total": 64,
                "memory_available": 0,
                "last_heartbeat_at": 0.0,
                "active_sessions": [],
                "active_session_count": 0,
                "nonterminal_requests": 0,
            }],
        )

        ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
            agent_epoch="process-a",
        )
        ctrl.schedule_once()

        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertEqual(
            [event.event_type for event in journal.list_events()].count(
                "node_registered"
            ),
            1,
        )
        self.assertIn(
            "node_announced",
            [event.event_type for event in journal.list_events()],
        )

    def test_stale_agent_epoch_cannot_heartbeat_replacement_node(self):
        ctrl, _ = controller()
        resources = NodeResources(cpu_cores=2, memory_gb=4)
        capabilities = RuntimeCapabilities(adapter="fake")
        ctrl.register_node(
            "node-a",
            POOL,
            resources,
            capabilities=capabilities,
            agent_epoch="old-process",
        )
        ctrl.register_node(
            "node-a",
            POOL,
            resources,
            capabilities=capabilities,
            agent_epoch="new-process",
        )

        with self.assertRaisesRegex(PermissionError, "agent epoch"):
            ctrl.node_heartbeat(
                "node-a",
                status={"authoritative": True, "sessions": []},
                agent_epoch="old-process",
            )

    def test_late_successful_report_cannot_reverse_terminal_failure(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1)
        request = ctrl.submit_workload(
            spec(gpus=1, session_id="terminal-is-monotonic")
        )
        ctrl.schedule_once()
        command = ctrl.node_heartbeat("node-a")["commands"][0]
        request.state = RequestState.FAILED
        request.reason = "reconcile_missing"

        ctrl.report_command_result(
            "node-a", command["command_id"], exit_code=0, stdout="late"
        )

        self.assertEqual(request.state, RequestState.FAILED)
        self.assertEqual(request.reason, "reconcile_missing")

    def test_node_reregistration_rejects_conflicting_live_resources(self):
        ctrl, _ = controller()
        resources = NodeResources(
            cpu_cores=8,
            memory_gb=64,
            gpu_uuids=gpu_ids("node-a", 1),
        )
        capabilities = RuntimeCapabilities(
            adapter="fake",
            gpu_enforcement=EnforcementMode.HARD,
        )
        ctrl.register_node(
            node_id="node-a",
            pool=POOL,
            resources=resources,
            capabilities=capabilities,
            allocation_id="alloc-a",
        )
        request = ctrl.submit_workload(
            spec(gpus=1, cpu=2, mem=4, session_id="retained-resources")
        )
        ctrl.schedule_once()
        ctrl.complete_request(request.request_id, exit_code=0)

        with self.assertRaisesRegex(ValueError, "cannot change allocation"):
            ctrl.register_node(
                node_id="node-a",
                pool=POOL,
                resources=resources,
                capabilities=capabilities,
                allocation_id="alloc-b",
            )
        with self.assertRaisesRegex(ValueError, "insufficient CPU"):
            ctrl.register_node(
                node_id="node-a",
                pool=POOL,
                resources=NodeResources(
                    cpu_cores=1,
                    memory_gb=64,
                    gpu_uuids=gpu_ids("node-a", 1),
                ),
                capabilities=capabilities,
                allocation_id="alloc-a",
            )

        node = ctrl.get_node("node-a")
        self.assertEqual(node.active_sessions, [request.execution_session_id])
        self.assertEqual(node.allocation_id, "alloc-a")
        ctrl.assert_invariants()

    def test_cached_idle_session_releases_gpu_without_destroying_cache(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, cpu=2, mem=2)
        first = ctrl.submit_workload(spec(user="alice", session_id="alice-session", gpus=1))
        ctrl.schedule_once()
        first_session = ctrl.get_request(first.request_id).execution_session_id
        ctrl.complete_request(first.request_id, exit_code=0)
        self.assertEqual(ctrl.get_session(first_session).state, SessionState.CACHED_IDLE)
        self.assertEqual(ctrl.get_session(first_session).gpu_uuids, ())
        self.assertEqual(ctrl.active_gpu_lease_count(), 0)

        second = ctrl.submit_workload(spec(user="bob", session_id="bob-session", gpus=1))
        self.assertEqual(ctrl.schedule_once(), 1)

        updated = ctrl.get_request(second.request_id)
        self.assertEqual(ctrl.get_session(first_session).state, SessionState.CACHED_IDLE)
        self.assertEqual(updated.state, RequestState.RUNNING)
        self.assertEqual(updated.gpu_uuids, ("node-a-gpu-0",))
        self.assertEqual(ctrl.lease_table, {"node-a-gpu-0": updated.execution_session_id})
        ctrl.assert_invariants()

    def test_cached_idle_session_is_evicted_for_cpu_memory_pressure(self):
        ctrl, _ = controller()
        register_gpu_node(ctrl, gpus=1, cpu=1, mem=1)
        first = ctrl.submit_workload(spec(user="alice", session_id="alice-session", gpus=1))
        ctrl.schedule_once()
        first_session = ctrl.get_request(first.request_id).execution_session_id
        ctrl.complete_request(first.request_id, exit_code=0)
        self.assertEqual(ctrl.get_session(first_session).state, SessionState.CACHED_IDLE)
        self.assertEqual(ctrl.active_gpu_lease_count(), 0)

        second = ctrl.submit_workload(spec(user="bob", session_id="bob-session", gpus=1))
        self.assertEqual(ctrl.schedule_once(), 0)
        self.assertEqual(ctrl.get_session(first_session).state, SessionState.SUSPENDING)
        self.assertEqual(ctrl.get_node("node-a").used_cpu_cores, 1)
        workspace = ctrl.workspaces[ctrl.sessions[first_session].workspace["id"]]
        workspace["state"] = "retained"
        ctrl.node_heartbeat("node-a", status={"authoritative": True, "sessions": [], "suspended_sessions": {first_session: 1}})
        self.assertEqual(ctrl.schedule_once(), 1)
        updated = ctrl.get_request(second.request_id)
        self.assertEqual(ctrl.get_session(first_session).state, SessionState.SUSPENDED)
        self.assertEqual(updated.state, RequestState.RUNNING)
        self.assertEqual(ctrl.lease_table, {"node-a-gpu-0": updated.execution_session_id})
        ctrl.assert_invariants()


class AuthJournalRuntimeTests(unittest.TestCase):
    def setUp(self):
        from tests.workspace_fixtures import install_embedded_workspace_double
        install_embedded_workspace_double(self)

    def test_join_token_scope_and_limits(self):
        clock = FakeClock()
        auth = TokenAuthority(clock=clock)
        auth.add_join_token(JoinToken("token", pool_id=POOL, allocation_id="alloc-a", expires_at=10, max_nodes=1, allowed_node_prefixes=("node-",), allowed_runtime_configs=("local",)))
        auth.validate_join_token("token", POOL, "node-1", "local")
        with self.assertRaises(PermissionError):
            auth.validate_join_token("token", POOL, "node-2", "local")
        clock.advance(10)
        with self.assertRaises(PermissionError):
            auth.validate_join_token("token", POOL, "node-3", "local")

    def test_user_token_roles(self):
        auth = TokenAuthority(clock=FakeClock())
        auth.add_user_token("submit", "alice", POOL, (Role.SUBMITTER,))
        self.assertEqual(auth.authenticate_user("submit", POOL, Role.SUBMITTER), "alice")
        with self.assertRaises(PermissionError):
            auth.authenticate_user("submit", POOL, Role.ADMIN)

    def test_sqlite_journal_persists_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = os.path.join(tmp, "cape.db")
            journal = SQLiteJournal(journal_path)
            ctrl, _ = controller(journal=journal)
            register_gpu_node(ctrl)
            req = ctrl.submit_workload(spec(session_id="journal"))
            ctrl.schedule_once()
            self.assertGreaterEqual(journal.count(), 1)
            event_types = [event.event_type for event in journal.list_events()]
            self.assertIn("request_submitted", event_types)
            self.assertIn("request_state", event_types)
            journal.close()
            reopened = SQLiteJournal(journal_path)
            self.assertEqual(reopened.count(), len(event_types))
            reopened.close()

    def test_heartbeat_without_state_change_does_not_journal_each_heartbeat(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        register_gpu_node(ctrl)
        before = journal.count()
        for _ in range(5):
            ctrl.heartbeat("node-a")
        self.assertEqual(journal.count(), before)

    def test_metrics_exporter_failure_does_not_break_scheduling(self):
        ctrl, _ = controller(metrics=FailingMetricsSink())
        register_gpu_node(ctrl)
        req = ctrl.submit_workload(spec(session_id="metrics"))
        ctrl.schedule_once()
        self.assertEqual(ctrl.get_request(req.request_id).state, RequestState.RUNNING)

    def test_memory_metrics_records_events(self):
        sink = MemoryMetricsSink()
        ctrl, _ = controller(metrics=sink)
        register_gpu_node(ctrl)
        req = ctrl.submit_workload(spec(session_id="metrics-memory"))
        ctrl.schedule_once()
        self.assertTrue(any(event.entity_id == req.request_id for event in sink.events))


    def test_controller_executes_request_through_registered_runtime_adapter(self):
        runtime = FakeRuntimeAdapter()
        ctrl, _ = controller()
        ctrl.runtime_adapters["fake"] = runtime
        register_gpu_node(ctrl, gpus=1)
        req = ctrl.submit_workload(spec(session_id="runtime", gpus=1))
        ctrl.schedule_once()
        updated = ctrl.get_request(req.request_id)
        self.assertEqual(updated.state, RequestState.COMPLETED)
        self.assertEqual(updated.exit_code, 0)
        self.assertIn("python train.py", updated.stdout)
        self.assertIn("run_command:" + updated.execution_session_id, runtime.calls)

    def test_controller_passes_prepared_image_and_effective_mounts_to_runtime(self):
        class CapturingRuntime(object):
            def __init__(self):
                self.started = None
            def prepare_image(self, image_ref, cache_config="default"):
                return "/prepared/base.sif"
            def start_session(self, session_spec):
                self.started = session_spec
                return RuntimeResult(0, "", "")
            def run_command(self, execution_session_id, command, cwd=None, env=None):
                return RuntimeResult(0, "ok", "")
            def stop_session(self, execution_session_id):
                return RuntimeResult(0, "", "")

        runtime = CapturingRuntime()
        ctrl, _ = controller()
        ctrl.runtime_adapters["fake"] = runtime
        register_gpu_node(ctrl, gpus=1)
        workload = WorkloadSpec(
            pool=POOL,
            user_id="alice",
            session_id="task-a",
            command=("sh", "-lc", "echo ok"),
            resources=ResourceRequest(gpus=1, max_duration_seconds=100),
            workspace={"target": "/cape-workspace"},
            image="docker://example/base:latest",
            cwd="/workspace",
            mounts=(
                MountSpec("/host/workspace", "/workspace"),
                MountSpec("/host/output", "/output"),
            ),
            log_path="/host/output/run.log",
            artifact_paths=("/host/output/result.txt",),
        )
        req = ctrl.submit_workload(workload)
        ctrl.schedule_once()
        updated = ctrl.get_request(req.request_id)
        self.assertEqual(updated.state, RequestState.COMPLETED)
        self.assertEqual(runtime.started.image, "/prepared/base.sif")
        self.assertIn(MountSpec("/host/workspace", "/workspace"), runtime.started.mounts)
        self.assertIn(MountSpec("/host/output", "/output"), runtime.started.mounts)
        self.assertEqual(updated.log_path, "/host/output/run.log")
        self.assertEqual(updated.artifact_paths, ("/host/output/result.txt",))

    def test_runtime_nonzero_marks_request_failed_and_keeps_session_cached(self):
        class FailingRuntime(object):
            def prepare_image(self, image_ref, cache_config="default"):
                pass
            def start_session(self, execution_session_id):
                pass
            def run_command(self, execution_session_id, command, cwd=None, env=None):
                return RuntimeResult(17, "", "boom")
            def stop_session(self, execution_session_id):
                pass
        ctrl, _ = controller()
        ctrl.runtime_adapters["fake"] = FailingRuntime()
        register_gpu_node(ctrl, gpus=1)
        req = ctrl.submit_workload(spec(session_id="runtime-fail", gpus=1))
        ctrl.schedule_once()
        updated = ctrl.get_request(req.request_id)
        self.assertEqual(updated.state, RequestState.FAILED)
        self.assertEqual(updated.exit_code, 17)
        self.assertEqual(updated.reason, "command_failed")
        self.assertEqual(
            ctrl.get_session(updated.execution_session_id).state,
            SessionState.CACHED_IDLE,
        )
        self.assertEqual(ctrl.lease_table, {})

    def test_runtime_infrastructure_result_marks_session_failed(self):
        class FailingRuntime(object):
            def prepare_image(self, image_ref, cache_config="default"):
                return image_ref

            def start_session(self, _session_spec):
                return RuntimeResult(0, "", "")

            def run_command(self, execution_session_id, command, cwd=None, env=None):
                return RuntimeResult(
                    127,
                    "",
                    "failed to launch runtime",
                    failure_kind="infrastructure",
                )

            def stop_session(self, execution_session_id):
                pass

        ctrl, _ = controller()
        ctrl.runtime_adapters["fake"] = FailingRuntime()
        register_gpu_node(ctrl, gpus=1)
        request = ctrl.submit_workload(
            spec(session_id="runtime-infrastructure", gpus=1)
        )

        ctrl.schedule_once()

        updated = ctrl.get_request(request.request_id)
        self.assertEqual(updated.state, RequestState.FAILED)
        self.assertEqual(updated.reason, "infrastructure_failed")
        self.assertEqual(
            ctrl.get_session(updated.execution_session_id).state,
            SessionState.FAILED,
        )
        self.assertEqual(ctrl.lease_table, {})

    def test_auth_enforces_submit_read_and_cancel_ownership(self):
        auth = TokenAuthority(clock=FakeClock())
        auth.add_user_token("alice", "alice", POOL, (Role.SUBMITTER,))
        auth.add_user_token("bob", "bob", POOL, (Role.SUBMITTER,))
        auth.add_user_token("viewer", "viewer", POOL, (Role.VIEWER,))
        ctrl, _ = controller()
        ctrl.auth = auth
        req = ctrl.submit_workload(spec(user="alice", session_id="auth", gpus=0), token="alice")
        with self.assertRaises(PermissionError):
            ctrl.submit_workload(spec(user="alice", session_id="bad", gpus=0), token="bob")
        with self.assertRaises(PermissionError):
            ctrl.get_request(req.request_id, token="bob")
        self.assertEqual(ctrl.get_request(req.request_id, token="alice").request_id, req.request_id)
        self.assertEqual(ctrl.get_request(req.request_id, token="viewer").request_id, req.request_id)
        with self.assertRaises(PermissionError):
            ctrl.cancel_request(req.request_id, token="bob")
        ctrl.cancel_request(req.request_id, token="alice")
        self.assertEqual(ctrl.get_request(req.request_id, token="alice").state, RequestState.CANCELLED)

    def test_node_agent_token_required_for_heartbeat_when_auth_configured(self):
        auth = TokenAuthority(clock=FakeClock())
        auth.add_join_token(JoinToken("join-a", pool_id=POOL, allocation_id="alloc-a", allowed_node_prefixes=("node-",)))
        auth.add_join_token(JoinToken("join-b", pool_id=POOL, allocation_id="alloc-b", allowed_node_prefixes=("node-",)))
        auth.add_user_token("node-a-token", "node-a", POOL, (Role.NODE_AGENT,))
        auth.add_user_token("node-b-token", "node-b", POOL, (Role.NODE_AGENT,))
        ctrl, _ = controller()
        ctrl.auth = auth
        with self.assertRaises(PermissionError):
            register_gpu_node(ctrl)
        register_gpu_node(ctrl, node_id="node-a", join_token="join-a")
        register_gpu_node(ctrl, node_id="node-b", join_token="join-b")
        with self.assertRaises(PermissionError):
            ctrl.heartbeat("node-a")
        with self.assertRaises(PermissionError):
            ctrl.heartbeat("node-a", token="node-b-token")
        self.assertEqual(ctrl.heartbeat("node-a", token="node-a-token").node_id, "node-a")
        self.assertEqual(ctrl.get_node("node-a", token="node-a-token").node_id, "node-a")
        with self.assertRaises(PermissionError):
            ctrl.get_node("node-b", token="node-a-token")
        with self.assertRaises(PermissionError):
            ctrl.submit_workload(spec(user="node-a", session_id="bad", gpus=0), token="node-a-token")

    def test_new_workload_schema_and_service_parser_use_only_mounts(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        ctrl.submit_workload(spec(
            user="alice",
            gpus=0,
            session_id="mount-schema",
            mounts=(MountSpec("/host/data", "/data", read_only=True),),
        ))
        submitted = next(event for event in journal.records if event.event_type == "request_submitted")
        serialized = submitted.details["spec"]
        self.assertEqual(serialized["session_id"], "mount-schema")
        self.assertNotIn("workspace_id", serialized)
        self.assertEqual(serialized["mounts"], [
            {"source": "/host/data", "target": "/data", "read_only": True},
        ])
        for field in ("workspace_path", "workspace_mount_path", "output_path", "output_mount_path"):
            self.assertNotIn(field, serialized)

        parsed = workload_spec_from_dict({
            "pool": POOL,
            "user_id": "alice",
            "session_id": "logical-task",
            "command": ["echo", "ok"],
            "mounts": [{"source": "/host/data", "target": "/data", "read_only": True}],
            "workspace_path": "/legacy/workspace",
            "workspace_mount_path": "/workspace",
            "output_path": "/legacy/output",
            "output_mount_path": "/output",
        })
        self.assertEqual(parsed.mounts, (MountSpec("/host/data", "/data", read_only=True),))
        self.assertEqual(parsed.session_id, "logical-task")
        for field in ("workspace_path", "workspace_mount_path", "output_path", "output_mount_path"):
            self.assertFalse(hasattr(parsed, field))

        with self.assertRaisesRegex(ValueError, "obsolete"):
            workload_spec_from_dict({
                "pool": POOL,
                "user_id": "alice",
                "session_id": "new-task",
                "workspace_id": "old-task",
                "command": ["echo", "conflict"],
            })

    def test_reconcile_from_legacy_journal_folds_path_fields_into_mounts(self):
        seed_journal = MemoryJournal()
        seed, _ = controller(journal=seed_journal)
        seed.submit_workload(spec(user="alice", gpus=0, session_id="legacy-seed"))
        submitted = next(event for event in seed_journal.records if event.event_type == "request_submitted")
        details = dict(submitted.details)
        legacy_spec = dict(details["spec"])
        legacy_spec.pop("keep_session_on_failure", None)
        legacy_spec["workspace_id"] = legacy_spec.pop("session_id")
        legacy_spec["mounts"] = [
            {"source": "/host/data", "target": "/data", "read_only": True},
            {"source": "/host/workspace", "target": "/workspace", "read_only": False},
        ]
        legacy_spec.update({
            "workspace_path": "/host/workspace",
            "workspace_mount_path": "/workspace",
            "output_path": "/host/output",
            "output_mount_path": "/output",
        })
        details["request_id"] = "req-legacy"
        details["sequence"] = 77
        details["spec"] = legacy_spec

        legacy_journal = MemoryJournal()
        legacy_journal.append(EventRecord(
            1.0,
            "request_submitted",
            "req-legacy",
            state=RequestState.SUBMITTED.value,
            details=details,
        ))
        with self.assertRaisesRegex(ValueError, "obsolete"):
            CAPEController.reconcile_from_journal(legacy_journal, clock=FakeClock())

    def test_reconcile_legacy_workspace_identity_for_request_and_session(self):
        current = MemoryJournal()
        seeded, _ = controller(journal=current)
        register_gpu_node(seeded, gpus=1)
        request = seeded.submit_workload(
            spec(gpus=1, session_id="legacy-identity")
        )
        seeded.schedule_once()

        legacy = MemoryJournal()
        for event in current.records:
            details = copy.deepcopy(event.details)
            if "session_id" in details:
                details["workspace_id"] = details.pop("session_id")
            if "spec" in details and "session_id" in details["spec"]:
                details["spec"]["workspace_id"] = details["spec"].pop("session_id")
            legacy.append(
                EventRecord(
                    event.timestamp,
                    event.event_type,
                    event.entity_id,
                    state=event.state,
                    reason=event.reason,
                    details=details,
                )
            )

        with self.assertRaisesRegex(ValueError, "obsolete"):
            CAPEController.reconcile_from_journal(legacy, clock=FakeClock())

    def test_reconcile_from_journal_rebuilds_queue_sessions_and_leases(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        register_gpu_node(ctrl, gpus=2)
        req = ctrl.submit_workload(spec(session_id="replay", gpus=1))
        ctrl.schedule_once()
        replayed = CAPEController.reconcile_from_journal(journal, clock=FakeClock())
        self.assertIn(req.request_id, replayed.requests)
        self.assertEqual(replayed.get_request(req.request_id).state, RequestState.RECONCILING)
        self.assertEqual(replayed.get_request(req.request_id).reason, "controller_restart")
        execution_session_id = replayed.get_request(req.request_id).execution_session_id
        self.assertEqual(replayed.get_session(execution_session_id).state, SessionState.RECONCILING)
        self.assertEqual(replayed.active_gpu_lease_count(), 1)
        replayed.assert_invariants()

    def test_reconcile_from_journal_keeps_cached_idle_gpu_free(self):
        journal = MemoryJournal()
        ctrl, _ = controller(journal=journal)
        register_gpu_node(ctrl, gpus=1)
        req = ctrl.submit_workload(spec(session_id="cached-replay", gpus=1))
        ctrl.schedule_once()
        execution_session_id = ctrl.get_request(req.request_id).execution_session_id
        ctrl.complete_request(req.request_id, exit_code=0)
        replayed = CAPEController.reconcile_from_journal(journal, clock=FakeClock())
        session = replayed.get_session(execution_session_id)
        self.assertEqual(session.state, SessionState.CACHED_IDLE)
        self.assertEqual(session.gpu_count, 1)
        self.assertEqual(session.gpu_uuids, ())
        self.assertEqual(replayed.active_gpu_lease_count(), 0)
        self.assertEqual(replayed.lease_table, {})
        replayed.assert_invariants()

    def test_reconcile_from_journal_grants_nodes_a_fresh_heartbeat_window(self):
        journal = MemoryJournal()
        seeded, _ = controller(journal=journal)
        register_gpu_node(seeded, gpus=1)
        request = seeded.submit_workload(
            spec(gpus=1, session_id="cached-before-restart")
        )
        seeded.schedule_once()
        execution_session_id = request.execution_session_id
        assignment = seeded.node_heartbeat("node-a")["commands"][0]
        seeded.report_command_result(
            "node-a", assignment["command_id"], exit_code=0
        )

        recovery_clock = FakeClock(1000)
        replayed = CAPEController.reconcile_from_journal(
            journal, clock=recovery_clock
        )

        node = replayed.get_node("node-a")
        self.assertEqual(node.state, NodeState.UNHEALTHY)
        self.assertEqual(node.last_heartbeat_at, 1000)
        replayed.check_heartbeats()
        recovery_clock.advance(89)
        replayed.check_heartbeats()
        self.assertEqual(
            replayed.get_session(execution_session_id).state,
            SessionState.CACHED_IDLE,
        )

        register_gpu_node(replayed, gpus=1)
        self.assertEqual(
            replayed.get_node("node-a").active_sessions,
            [execution_session_id],
        )
        replayed.assert_invariants()

    def test_authoritative_heartbeat_cleans_agent_session_lost_by_controller(self):
        policy = SchedulerPolicy(
            heartbeat_timeout_seconds=5, lost_timeout_seconds=10
        )
        ctrl, clock = controller(policy=policy)
        capabilities = RuntimeCapabilities(
            adapter="fake",
            isolation_policies=("default", "image-stateful"),
        )
        register_gpu_node(ctrl, gpus=1, caps=capabilities)
        request = ctrl.submit_workload(
            spec(
                gpus=1,
                session_id="agent-retained-after-loss",
                isolation="image-stateful",
            )
        )
        ctrl.schedule_once()
        execution_session_id = request.execution_session_id
        assignment = ctrl.node_heartbeat("node-a")["commands"][0]
        ctrl.report_command_result(
            "node-a", assignment["command_id"], exit_code=0
        )
        clock.advance(10)
        ctrl.check_heartbeats()
        self.assertEqual(
            ctrl.get_session(execution_session_id).state, SessionState.LOST
        )

        register_gpu_node(ctrl, gpus=1, caps=capabilities)
        commands = ctrl.node_heartbeat(
            "node-a",
            status={
                "authoritative": True,
                "sessions": [
                    {"execution_session_id": execution_session_id, "state": "CACHED_IDLE"}
                ],
            },
        )["commands"]

        self.assertEqual(len(commands), 1)
        cleanup = commands[0]
        self.assertEqual(cleanup["action"], "stop_session")
        self.assertEqual(cleanup["execution_session_id"], execution_session_id)
        self.assertEqual(cleanup["runtime_adapter"], "fake")
        self.assertEqual(
            cleanup["session_spec"]["isolation_policy"],
            "image-stateful",
        )
        self.assertTrue(cleanup["session_spec"]["state_layer_id"])
        self.assertNotIn("env", cleanup["session_spec"])
        ctrl.assert_invariants()

    def test_max_lifetime_expiration_stops_session_and_marks_request_expired(self):
        ctrl, clock = controller()
        register_gpu_node(ctrl, gpus=1)
        req = ctrl.submit_workload(spec(session_id="lifetime", gpus=1))
        ctrl.schedule_once()
        execution_session_id = ctrl.get_request(req.request_id).execution_session_id
        clock.advance(100)
        expired = ctrl.expire_max_lifetime_sessions()
        self.assertEqual(expired, [execution_session_id])
        self.assertEqual(ctrl.get_request(req.request_id).state, RequestState.EXPIRED)
        expired_request = ctrl.get_request(req.request_id)
        self.assertIsNotNone(expired_request.started_at)
        self.assertEqual(
            expired_request.ended_at, expired_request.updated_at
        )
        self.assertLessEqual(
            expired_request.started_at, expired_request.ended_at
        )
        self.assertEqual(ctrl.get_session(execution_session_id).state, SessionState.STOPPED)
        self.assertEqual(ctrl.lease_table, {})

    def test_max_queued_requests_per_user_is_enforced(self):
        policy = SchedulerPolicy(max_queued_requests_per_user=1)
        ctrl, _ = controller(policy=policy)
        first = ctrl.submit_workload(spec(user="alice", session_id="q1", gpus=0))
        second = ctrl.submit_workload(spec(user="alice", session_id="q2", gpus=0))
        self.assertEqual(first.state, RequestState.QUEUED)
        self.assertEqual(second.state, RequestState.INFEASIBLE)
        self.assertEqual(second.reason, "quota_exceeded")

    def test_workload_spec_hash_supports_env_mounts_and_output_metadata(self):
        workload = spec(gpus=0)
        custom = WorkloadSpec(
            pool=POOL,
            user_id="alice",
            session_id="task-a",
            command=("echo", "ok"),
            env={"A": "1"},
            mounts=(MountSpec("/workspace-host", "/workspace"), MountSpec("/out", "/output")),
            log_path="/logs/run.log",
            artifact_paths=("/out/file",),
            resources=ResourceRequest(gpus=0),
        )
        self.assertIsInstance(hash(workload), int)
        self.assertIsInstance(hash(custom), int)
        req = RequestRecord("req-x", custom, RequestState.QUEUED, None, 0, 0, 1)
        self.assertEqual(req.log_path, "/logs/run.log")
        self.assertEqual(req.artifact_paths, ("/out/file",))
        self.assertEqual(custom.effective_mounts(), (
            MountSpec("/workspace-host", "/workspace"),
            MountSpec("/out", "/output"),
        ))
        self.assertEqual(custom.effective_mounts(), custom.mounts)

    def test_sqlite_journal_batches_and_flushes_events_on_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal_path = os.path.join(tmp, "batched.db")
            journal = SQLiteJournal(journal_path, batch_size=10)
            ctrl, _ = controller(journal=journal)
            register_gpu_node(ctrl)
            req = ctrl.submit_workload(spec(session_id="batch"))
            ctrl.schedule_once()
            self.assertGreater(journal.count(), 0)
            reopened = SQLiteJournal(journal_path)
            self.assertEqual(reopened.count(), journal.count())
            reopened.close()
            journal.close()

    def test_fake_runtime_adapter_contract(self):
        runtime = FakeRuntimeAdapter()
        runtime.prepare_image("docker://ubuntu:latest", "local")
        runtime.start_session("sess-1")
        result = runtime.run_command("sess-1", ("echo", "ok"))
        runtime.stop_session("sess-1")
        runtime.cleanup_session("sess-1")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("run_command:sess-1", runtime.calls)

    def test_local_process_runtime_runs_cpu_command(self):
        runtime = LocalProcessRuntimeAdapter()
        result = runtime.run_command("local", (sys.executable, "-c", "print('cape-ok')"))
        self.assertEqual(result.exit_code, 0)
        self.assertIn("cape-ok", result.stdout)

    def test_async_submit_wrapper_returns_quickly(self):
        ctrl, _ = controller()
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            record = loop.run_until_complete(ctrl.submit_workload_async(spec(session_id="async")))
        finally:
            loop.close()
        self.assertEqual(record.state, RequestState.QUEUED)
        self.assertEqual(record.reason, "pool_empty")


if __name__ == "__main__":
    unittest.main()
