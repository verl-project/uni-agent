# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tempfile
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import (
    CAPEController,
    CAPEControllerService,
    EnforcementMode,
    MemoryJournal,
    NodeAgentClient,
    NodeResources,
    NodeState,
    RequestState,
    ResourceRequest,
    RuntimeCapabilities,
    SchedulerPolicy,
    WorkloadSpec,
)
from pycape.cli import _build_cape_parser, cape_main
from pycape.service import workload_spec_from_dict


GPU_0 = "GPU-00000000-0000-0000-0000-000000000000"
GPU_1 = "GPU-11111111-1111-1111-1111-111111111111"
SHARED_POOL = "shared-pool"
EXCLUSIVE_POOL = "exclusive-pool"


def capabilities():
    return RuntimeCapabilities(
        adapter="fake",
        gpu_enforcement=EnforcementMode.HARD,
    )


def register(
    controller,
    pool,
    node_id,
    mode,
    gpu_slots=1,
    placement="pack",
    gpu_uuids=(GPU_0,),
    allocation_id=None,
):
    return controller.register_node(
        node_id=node_id,
        pool=pool,
        resources=NodeResources(
            cpu_cores=64,
            memory_gb=512,
            gpu_uuids=gpu_uuids,
        ),
        capabilities=capabilities(),
        allocation_id=allocation_id or "alloc-" + node_id,
        pool_mode=mode,
        gpu_slots=gpu_slots,
        placement=placement,
    )


def workload(pool, name, gpus=1, reuse=False):
    return WorkloadSpec(
        pool=pool,
        user_id="user-" + name,
        session_id=name,
        command=("true",),
        reuse=reuse,
        resources=ResourceRequest(
            gpus=gpus,
            cpu_cores=1,
            memory_gb=1,
            max_duration_seconds=60,
        ),
    )


def submit(controller, pool, name, gpus=1, reuse=False):
    return controller.submit_workload(
        workload(pool, name, gpus=gpus, reuse=reuse)
    )


def complete(controller, request):
    return controller.complete_request(request.request_id, exit_code=0)


class PoolDeclarationTests(unittest.TestCase):
    def test_first_join_establishes_pool_config_and_matching_join_is_accepted(self):
        controller = CAPEController()

        first = register(
            controller,
            SHARED_POOL,
            "shared-a",
            "shared",
            gpu_slots=4,
            placement="spread",
        )
        second = register(
            controller,
            SHARED_POOL,
            "shared-b",
            "shared",
            gpu_slots=4,
            placement="spread",
            gpu_uuids=(GPU_1,),
        )

        config = controller.pool_configs[SHARED_POOL]
        self.assertEqual(config.pool_mode, "shared")
        self.assertEqual(config.gpu_slots, 4)
        self.assertEqual(config.placement, "spread")
        for node in (first, second):
            self.assertEqual(node.pool_mode, "shared")
            self.assertEqual(node.gpu_slots, 4)
            self.assertEqual(node.placement, "spread")

    def test_conflicting_pool_join_is_rejected_before_registration_mutates_state(self):
        controller = CAPEController()
        register(
            controller,
            SHARED_POOL,
            "shared-a",
            "shared",
            gpu_slots=4,
            placement="pack",
        )

        conflicts = (
            ("exclusive", 1, "pack"),
            ("shared", 2, "pack"),
            ("shared", 4, "spread"),
        )
        for index, (mode, slots, placement) in enumerate(conflicts):
            node_id = "conflict-{}".format(index)
            with self.subTest(mode=mode, slots=slots, placement=placement):
                with self.assertRaisesRegex(
                    ValueError, "pool.*configuration"
                ):
                    register(
                        controller,
                        SHARED_POOL,
                        node_id,
                        mode,
                        gpu_slots=slots,
                        placement=placement,
                        gpu_uuids=(GPU_1,),
                    )
                self.assertNotIn(node_id, controller.nodes)
                self.assertNotIn("alloc-" + node_id, controller.allocations)

    def test_invalid_slot_and_placement_declarations_are_rejected(self):
        controller = CAPEController()
        cases = (
            ("exclusive", 2, "pack"),
            ("shared", 0, "pack"),
            ("shared", 4, "random"),
        )
        for index, (mode, slots, placement) in enumerate(cases):
            with self.subTest(mode=mode, slots=slots, placement=placement):
                with self.assertRaises(ValueError):
                    register(
                        controller,
                        "pool-{}".format(index),
                        "node-{}".format(index),
                        mode,
                        gpu_slots=slots,
                        placement=placement,
                    )

    def test_registration_payload_carries_the_same_declaration_for_announce_and_register(self):
        parser = _build_cape_parser()
        args = parser.parse_args(
            [
                "node",
                "enroll", "--node-token-out", "unused-agent.json",
                "--node-id",
                "shared-node",
                "--pool",
                SHARED_POOL,
                "--pool-mode",
                "shared",
                "--gpu-slots",
                "4",
                "--pool-placement",
                "spread",
            ]
        )
        from pycape.cli import _node_announcement_payload

        announcement = _node_announcement_payload(args, "epoch-a")
        self.assertEqual(announcement["pool_mode"], "shared")
        self.assertEqual(announcement["gpu_slots"], 4)
        self.assertEqual(announcement["placement"], "spread")

        agent = NodeAgentClient(
            "http://controller.invalid",
            node_id="shared-node",
            pool=SHARED_POOL,
            resources=NodeResources(4, 16, (GPU_0,)),
            pool_mode="shared",
            gpu_slots=4,
            placement="spread",
        )
        registration = agent._registration_payload()
        self.assertEqual(
            {
                key: registration[key]
                for key in ("pool_mode", "gpu_slots", "placement")
            },
            {
                key: announcement[key]
                for key in ("pool_mode", "gpu_slots", "placement")
            },
        )


class PhysicalGpuLedgerTests(unittest.TestCase):
    def dual_controller(self, policy=None, shared_slots=4, placement="pack"):
        controller = CAPEController(policy=policy)
        register(
            controller,
            EXCLUSIVE_POOL,
            "exclusive-node",
            "exclusive",
            gpu_slots=1,
        )
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=shared_slots,
            placement=placement,
        )
        return controller

    def test_exclusive_holder_blocks_every_shared_view_of_the_uuid(self):
        controller = self.dual_controller()
        exclusive = submit(controller, EXCLUSIVE_POOL, "exclusive")
        controller.schedule_once()
        shared = submit(controller, SHARED_POOL, "shared")
        controller.schedule_once()

        self.assertEqual(exclusive.state, RequestState.RUNNING)
        self.assertEqual(shared.state, RequestState.QUEUED)
        entry = controller.gpu_ledger[GPU_0]
        self.assertEqual(entry.state.value, "EXCLUSIVE")
        self.assertEqual(entry.exclusive_execution_session_id, exclusive.execution_session_id)
        self.assertEqual(entry.shared_refcount, 0)
        self.assertEqual(
            controller.pool_utilization(SHARED_POOL)["gpus_available"], 0
        )

        complete(controller, exclusive)
        controller.schedule_once()
        self.assertEqual(shared.state, RequestState.RUNNING)

    def test_shared_holders_block_the_exclusive_view_until_the_last_release(self):
        controller = self.dual_controller(shared_slots=2)
        shared_a = submit(controller, SHARED_POOL, "shared-a")
        shared_b = submit(controller, SHARED_POOL, "shared-b")
        controller.schedule_once()
        exclusive = submit(controller, EXCLUSIVE_POOL, "exclusive")
        controller.schedule_once()

        entry = controller.gpu_ledger[GPU_0]
        self.assertEqual(entry.state.value, "SHARED")
        self.assertEqual(entry.shared_refcount, 2)
        self.assertEqual(exclusive.state, RequestState.QUEUED)

        complete(controller, shared_a)
        controller.schedule_once()
        self.assertEqual(entry.shared_refcount, 1)
        self.assertEqual(exclusive.state, RequestState.QUEUED)
        self.assertEqual(shared_b.state, RequestState.RUNNING)

        complete(controller, shared_b)
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.RUNNING)
        self.assertEqual(entry.state.value, "EXCLUSIVE")

    def test_shared_refcount_and_dispatch_cotenant_status_are_exact(self):
        controller = CAPEController()
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=4,
        )
        requests = [
            submit(controller, SHARED_POOL, "shared-{}".format(index))
            for index in range(5)
        ]

        controller.schedule_once()

        self.assertEqual(
            [request.state for request in requests],
            [RequestState.RUNNING] * 4 + [RequestState.QUEUED],
        )
        entry = controller.gpu_ledger[GPU_0]
        self.assertEqual(entry.shared_refcount, 4)
        self.assertEqual(len(set(entry.shared_execution_session_ids)), 4)
        for index, request in enumerate(requests[:4]):
            status = controller.request_status(request.request_id)
            self.assertEqual(status["gpu_mode"], "shared")
            self.assertEqual(status["gpu_cotenant_count"], index)
            self.assertEqual(
                status["gpu_cotenant_counts"], {GPU_0: index}
            )

        complete(controller, requests[1])
        complete(controller, requests[3])
        self.assertEqual(entry.shared_refcount, 2)
        controller.schedule_once()
        self.assertEqual(requests[4].state, RequestState.RUNNING)
        self.assertEqual(entry.shared_refcount, 3)
        self.assertEqual(
            controller.request_status(requests[4].request_id)[
                "gpu_cotenant_count"
            ],
            2,
        )
        controller.assert_invariants()

    def test_concurrent_completion_reports_release_each_shared_holder_once(self):
        controller = CAPEController()
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
        )
        requests = [
            submit(controller, SHARED_POOL, "concurrent-{}".format(index))
            for index in range(2)
        ]
        controller.schedule_once()
        commands = controller.node_heartbeat(
            "shared-node",
            status={"authoritative": True, "sessions": []},
            max_commands=2,
        )["commands"]
        service = CAPEControllerService(
            controller=controller, schedule_interval=100
        ).start()
        errors = []
        barrier = threading.Barrier(2)

        def report(command):
            try:
                agent = NodeAgentClient(
                    service.url,
                    node_id="shared-node",
                    pool=SHARED_POOL,
                    resources=NodeResources(64, 512, (GPU_0,)),
                    pool_mode="shared",
                    gpu_slots=2,
                )
                barrier.wait()
                agent._post(
                    "/v1/agent/report-command",
                    {
                        "node_id": "shared-node",
                        "command_id": command["command_id"],
                        "exit_code": 0,
                        "accept_commands": True,
                        "max_commands": 8,
                    },
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=report, args=(command,))
            for command in commands
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        finally:
            service.stop()

        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertTrue(
            all(request.state == RequestState.COMPLETED for request in requests)
        )
        entry = controller.gpu_ledger[GPU_0]
        self.assertEqual(entry.state.value, "FREE")
        self.assertEqual(entry.shared_refcount, 0)
        controller.assert_invariants()

    def test_shared_multi_gpu_request_takes_one_slot_on_each_distinct_uuid(self):
        controller = CAPEController()
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
            gpu_uuids=(GPU_0, GPU_1),
        )
        first = submit(controller, SHARED_POOL, "multi-a", gpus=2)
        second = submit(controller, SHARED_POOL, "multi-b", gpus=2)
        third = submit(controller, SHARED_POOL, "multi-c", gpus=2)

        controller.schedule_once()

        self.assertEqual(first.gpu_uuids, (GPU_0, GPU_1))
        self.assertEqual(second.gpu_uuids, (GPU_0, GPU_1))
        self.assertEqual(len(set(first.gpu_uuids)), 2)
        self.assertEqual(len(set(second.gpu_uuids)), 2)
        self.assertEqual(third.state, RequestState.QUEUED)
        self.assertEqual(controller.gpu_ledger[GPU_0].shared_refcount, 2)
        self.assertEqual(controller.gpu_ledger[GPU_1].shared_refcount, 2)

    def test_pack_and_spread_rank_shared_slots_by_current_refcount(self):
        expected = {
            "pack": ((GPU_0,), (GPU_0,), (GPU_1,)),
            "spread": ((GPU_0,), (GPU_1,), (GPU_0,)),
        }
        for placement, expected_gpus in expected.items():
            with self.subTest(placement=placement):
                controller = CAPEController()
                register(
                    controller,
                    SHARED_POOL,
                    "shared-node",
                    "shared",
                    gpu_slots=2,
                    placement=placement,
                    gpu_uuids=(GPU_0, GPU_1),
                )
                requests = [
                    submit(
                        controller,
                        SHARED_POOL,
                        "{}-{}".format(placement, index),
                    )
                    for index in range(3)
                ]
                controller.schedule_once()
                self.assertEqual(
                    tuple(request.gpu_uuids for request in requests),
                    expected_gpus,
                )

    def test_pack_and_spread_apply_across_nodes_in_the_pool(self):
        expected = {
            "pack": ((GPU_0,), (GPU_0,)),
            "spread": ((GPU_0,), (GPU_1,)),
        }
        for placement, expected_gpus in expected.items():
            with self.subTest(placement=placement):
                controller = CAPEController()
                register(
                    controller,
                    SHARED_POOL,
                    "node-a",
                    "shared",
                    gpu_slots=2,
                    placement=placement,
                    gpu_uuids=(GPU_0,),
                )
                register(
                    controller,
                    SHARED_POOL,
                    "node-b",
                    "shared",
                    gpu_slots=2,
                    placement=placement,
                    gpu_uuids=(GPU_1,),
                )
                requests = []
                for index in range(2):
                    requests.append(
                        submit(
                            controller,
                            SHARED_POOL,
                            "{}-cross-node-{}".format(
                                placement, index
                            ),
                        )
                    )
                    controller.schedule_once()
                self.assertEqual(
                    tuple(request.gpu_uuids for request in requests),
                    expected_gpus,
                )

    def test_cached_idle_shared_session_releases_its_physical_slot(self):
        controller = CAPEController()
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
        )
        request = submit(controller, SHARED_POOL, "warm", reuse=True)
        controller.schedule_once()
        complete(controller, request)

        session = controller.sessions[request.execution_session_id]
        self.assertEqual(session.state.value, "CACHED_IDLE")
        self.assertEqual(session.gpu_uuids, ())
        self.assertEqual(controller.gpu_ledger[GPU_0].state.value, "FREE")
        self.assertEqual(controller.lease_table, {})

    def test_utilization_preserves_pool_views_and_deduplicates_physical_uuid(self):
        controller = self.dual_controller(shared_slots=4)
        requests = [
            submit(controller, SHARED_POOL, "util-{}".format(index))
            for index in range(2)
        ]
        controller.schedule_once()

        shared = controller.pool_utilization(SHARED_POOL)
        exclusive = controller.pool_utilization(EXCLUSIVE_POOL)
        global_summary = controller.pool_utilization()

        self.assertEqual(shared["gpus_total"], 1)
        self.assertEqual(shared["gpus_leased"], 1)
        self.assertEqual(shared["gpu_slots_total"], 4)
        self.assertEqual(shared["gpu_slots_leased"], 2)
        self.assertEqual(shared["gpu_slots_available"], 2)
        self.assertEqual(exclusive["gpus_total"], 1)
        self.assertEqual(exclusive["gpus_available"], 0)
        self.assertEqual(global_summary["gpus_total"], 2)
        self.assertEqual(global_summary["physical"]["gpus_total"], 1)
        self.assertEqual(global_summary["physical"]["gpus_shared"], 1)
        self.assertEqual(
            global_summary["physical"]["shared_slots_leased"], 2
        )
        self.assertEqual(
            global_summary["physical"]["gpus"][0],
            {
                "gpu_uuid": GPU_0,
                "state": "SHARED",
                "shared_refcount": 2,
            },
        )
        self.assertTrue(
            all(request.state == RequestState.RUNNING for request in requests)
        )

    def test_journal_replay_restores_pool_config_effective_status_and_ledger(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=3,
            placement="spread",
        )
        requests = [
            submit(controller, SHARED_POOL, "replay-{}".format(index))
            for index in range(2)
        ]
        controller.schedule_once()

        replayed = CAPEController.reconcile_from_journal(journal)

        config = replayed.pool_configs[SHARED_POOL]
        self.assertEqual(
            (config.pool_mode, config.gpu_slots, config.placement),
            ("shared", 3, "spread"),
        )
        entry = replayed.gpu_ledger[GPU_0]
        self.assertEqual(entry.state.value, "SHARED")
        self.assertEqual(entry.shared_refcount, 2)
        self.assertEqual(
            replayed.request_status(requests[1].request_id)[
                "gpu_cotenant_count"
            ],
            1,
        )
        replayed.assert_invariants()

    def test_shared_completion_report_piggybacks_the_freed_slot_successor(self):
        controller = CAPEController()
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
        )
        requests = [
            submit(controller, SHARED_POOL, "piggyback-{}".format(index))
            for index in range(3)
        ]
        controller.schedule_once()
        commands = controller.node_heartbeat(
            "shared-node",
            status={"authoritative": True, "sessions": []},
            max_commands=2,
        )["commands"]
        service = CAPEControllerService(
            controller=controller, schedule_interval=100
        ).start()
        try:
            agent = NodeAgentClient(
                service.url,
                node_id="shared-node",
                pool=SHARED_POOL,
                resources=NodeResources(64, 512, (GPU_0,)),
                pool_mode="shared",
                gpu_slots=2,
            )
            response = agent._post(
                "/v1/agent/report-command",
                {
                    "node_id": "shared-node",
                    "command_id": commands[0]["command_id"],
                    "exit_code": 0,
                    "accept_commands": True,
                    "max_commands": 8,
                },
            )
        finally:
            service.stop()

        run_commands = [
            command
            for command in response["commands"]
            if command.get("action", "run") == "run"
        ]
        self.assertEqual(
            [command["request_id"] for command in run_commands],
            [requests[2].request_id],
        )
        self.assertEqual(requests[2].state, RequestState.RUNNING)
        self.assertEqual(controller.gpu_ledger[GPU_0].shared_refcount, 2)


class ContentionPolicyTests(unittest.TestCase):
    def controller(self, policy_name, ratio=2, overrides=None, slots=2):
        policy = SchedulerPolicy(
            gpu_contention_policy=policy_name,
            gpu_shared_ratio=ratio,
            gpu_share_overrides=overrides,
        )
        controller = CAPEController(policy=policy)
        register(
            controller,
            EXCLUSIVE_POOL,
            "exclusive-node",
            "exclusive",
        )
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=slots,
        )
        return controller

    def test_barrier_ratio_admits_n_shared_then_naturally_drains_for_exclusive(self):
        controller = self.controller("barrier-ratio", ratio=2, slots=2)
        exclusive = submit(controller, EXCLUSIVE_POOL, "exclusive")
        shared = [
            submit(controller, SHARED_POOL, "shared-{}".format(index))
            for index in range(4)
        ]

        controller.schedule_once()

        self.assertEqual(exclusive.state, RequestState.QUEUED)
        self.assertEqual(
            [request.state for request in shared],
            [RequestState.RUNNING, RequestState.RUNNING]
            + [RequestState.QUEUED] * 2,
        )
        complete(controller, shared[0])
        controller.schedule_once()
        self.assertEqual(shared[1].state, RequestState.RUNNING)
        self.assertEqual(shared[2].state, RequestState.QUEUED)
        self.assertEqual(exclusive.state, RequestState.QUEUED)

        complete(controller, shared[1])
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.RUNNING)
        self.assertEqual(
            [request.state for request in shared[2:]],
            [RequestState.QUEUED, RequestState.QUEUED],
        )

        complete(controller, exclusive)
        controller.schedule_once()
        self.assertEqual(
            [request.state for request in shared[2:]],
            [RequestState.RUNNING, RequestState.RUNNING],
        )

    def test_exclusive_first_stops_new_shared_admissions_without_preemption(self):
        controller = self.controller("exclusive-first", ratio=4, slots=2)
        running_shared = submit(controller, SHARED_POOL, "running")
        controller.schedule_once()
        exclusive = submit(controller, EXCLUSIVE_POOL, "exclusive")
        waiting_shared = submit(controller, SHARED_POOL, "waiting")

        controller.schedule_once()

        self.assertEqual(running_shared.state, RequestState.RUNNING)
        self.assertEqual(exclusive.state, RequestState.QUEUED)
        self.assertEqual(waiting_shared.state, RequestState.QUEUED)

        complete(controller, running_shared)
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.RUNNING)
        self.assertEqual(waiting_shared.state, RequestState.QUEUED)

    def test_ratio_only_replenishes_while_shared_is_active_then_yields_when_free(self):
        controller = self.controller("ratio-only", ratio=1, slots=2)
        first = submit(controller, SHARED_POOL, "first")
        controller.schedule_once()
        exclusive = submit(controller, EXCLUSIVE_POOL, "exclusive")
        second = submit(controller, SHARED_POOL, "second")
        third = submit(controller, SHARED_POOL, "third")

        controller.schedule_once()
        self.assertEqual(second.state, RequestState.RUNNING)
        self.assertEqual(third.state, RequestState.QUEUED)

        complete(controller, first)
        controller.schedule_once()
        self.assertEqual(third.state, RequestState.RUNNING)
        self.assertEqual(exclusive.state, RequestState.QUEUED)

        complete(controller, second)
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.QUEUED)
        self.assertEqual(third.state, RequestState.RUNNING)

        complete(controller, third)
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.RUNNING)

    def test_each_mode_is_work_conserving_when_only_one_queue_has_work(self):
        for policy_name in (
            "barrier-ratio",
            "ratio-only",
            "exclusive-first",
        ):
            with self.subTest(policy=policy_name):
                controller = self.controller(
                    policy_name, ratio=1, slots=1
                )
                shared = [
                    submit(
                        controller,
                        SHARED_POOL,
                        "{}-shared-{}".format(policy_name, index),
                    )
                    for index in range(3)
                ]
                for request in shared:
                    controller.schedule_once()
                    self.assertEqual(request.state, RequestState.RUNNING)
                    complete(controller, request)

                exclusive = submit(
                    controller,
                    EXCLUSIVE_POOL,
                    policy_name + "-exclusive",
                )
                controller.schedule_once()
                self.assertEqual(exclusive.state, RequestState.RUNNING)

    def test_per_gpu_override_replaces_controller_default(self):
        controller = self.controller(
            "ratio-only",
            ratio=99,
            slots=2,
            overrides={
                GPU_0: {
                    "policy": "exclusive-first",
                    "ratio": 1,
                }
            },
        )
        running_shared = submit(controller, SHARED_POOL, "override-running")
        controller.schedule_once()
        exclusive = submit(controller, EXCLUSIVE_POOL, "override-exclusive")
        waiting_shared = submit(
            controller, SHARED_POOL, "override-waiting"
        )

        controller.schedule_once()

        self.assertEqual(running_shared.state, RequestState.RUNNING)
        self.assertEqual(exclusive.state, RequestState.QUEUED)
        self.assertEqual(waiting_shared.state, RequestState.QUEUED)
        self.assertEqual(
            controller.contention_policy_for_gpu(GPU_0),
            ("exclusive-first", 1),
        )

    def test_shared_membership_can_retire_without_preempting_or_stopping_exclusive_view(self):
        controller = self.controller("exclusive-first", ratio=1, slots=2)
        running = [
            submit(controller, SHARED_POOL, "retire-{}".format(index))
            for index in range(2)
        ]
        waiting_shared = submit(controller, SHARED_POOL, "retire-waiting")
        controller.schedule_once()
        exclusive = submit(controller, EXCLUSIVE_POOL, "retire-exclusive")

        controller.drain_node("shared-node")
        controller.schedule_once()
        self.assertTrue(
            all(request.state == RequestState.RUNNING for request in running)
        )
        self.assertEqual(waiting_shared.state, RequestState.QUEUED)
        self.assertEqual(exclusive.state, RequestState.QUEUED)

        complete(controller, running[0])
        controller.schedule_once()
        self.assertEqual(running[1].state, RequestState.RUNNING)
        self.assertEqual(waiting_shared.state, RequestState.QUEUED)

        complete(controller, running[1])
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.RUNNING)
        self.assertEqual(waiting_shared.state, RequestState.QUEUED)
        controller.remove_node("shared-node")
        self.assertEqual(
            controller.nodes["shared-node"].state, NodeState.REMOVED
        )
        self.assertEqual(
            controller.nodes["exclusive-node"].state, NodeState.ACTIVE
        )

    def test_lost_shared_holder_releases_exclusive_capacity(self):
        clock = mock.Mock(return_value=0.0)
        policy = SchedulerPolicy(
            heartbeat_timeout_seconds=5,
            lost_timeout_seconds=10,
        )
        controller = CAPEController(clock=clock, policy=policy)
        register(
            controller,
            EXCLUSIVE_POOL,
            "exclusive-node",
            "exclusive",
        )
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
        )
        shared = submit(controller, SHARED_POOL, "lost-shared")
        controller.schedule_once()

        clock.return_value = 11.0
        controller.heartbeat("exclusive-node")
        controller.check_heartbeats()

        self.assertEqual(
            controller.nodes["shared-node"].state, NodeState.LOST
        )
        self.assertEqual(
            controller.sessions[shared.execution_session_id].state.value, "LOST"
        )
        self.assertNotIn(
            shared.execution_session_id,
            controller.gpu_ledger[GPU_0].shared_execution_session_ids,
        )
        exclusive = submit(controller, EXCLUSIVE_POOL, "lost-exclusive")
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.RUNNING)
        controller.assert_invariants()

    def test_replay_preserves_released_lost_physical_lease(self):
        clock = mock.Mock(return_value=0.0)
        journal = MemoryJournal()
        policy = SchedulerPolicy(
            heartbeat_timeout_seconds=5,
            lost_timeout_seconds=10,
        )
        controller = CAPEController(
            clock=clock,
            journal=journal,
            policy=policy,
        )
        register(
            controller,
            EXCLUSIVE_POOL,
            "exclusive-node",
            "exclusive",
        )
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
        )
        shared = submit(controller, SHARED_POOL, "replay-lost")
        controller.schedule_once()
        clock.return_value = 11.0
        controller.heartbeat("exclusive-node")
        controller.check_heartbeats()

        replayed = CAPEController.reconcile_from_journal(
            journal,
            clock=mock.Mock(return_value=12.0),
            policy=policy,
        )

        self.assertNotIn(
            shared.execution_session_id,
            replayed.gpu_ledger[GPU_0].shared_execution_session_ids,
        )
        self.assertNotIn(
            shared.execution_session_id,
            replayed.nodes["shared-node"].active_sessions,
        )
        replayed.assert_invariants()
        self.assertEqual(
            replayed.gpu_ledger[GPU_0].state.value, "FREE"
        )

    def test_recovered_lost_holder_releases_after_cleanup_confirmation(self):
        clock = mock.Mock(return_value=0.0)
        policy = SchedulerPolicy(
            heartbeat_timeout_seconds=5,
            lost_timeout_seconds=10,
        )
        controller = CAPEController(clock=clock, policy=policy)
        register(
            controller,
            EXCLUSIVE_POOL,
            "exclusive-node",
            "exclusive",
        )
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
        )
        shared = submit(controller, SHARED_POOL, "recover-lost")
        controller.schedule_once()
        controller.node_heartbeat(
            "shared-node",
            status={"authoritative": True, "sessions": []},
        )

        clock.return_value = 11.0
        controller.heartbeat("exclusive-node")
        controller.check_heartbeats()
        register(
            controller,
            SHARED_POOL,
            "shared-node",
            "shared",
            gpu_slots=2,
        )
        cleanup = controller.node_heartbeat(
            "shared-node",
            status={
                "authoritative": True,
                "sessions": [
                    {
                        "execution_session_id": shared.execution_session_id,
                        "state": "RUNNING_COMMAND",
                    }
                ],
            },
        )["commands"]

        self.assertEqual(
            [
                command["execution_session_id"]
                for command in cleanup
                if command["action"] == "stop_session"
            ],
            [shared.execution_session_id],
        )
        self.assertNotIn(
            shared.execution_session_id,
            controller.gpu_ledger[GPU_0].shared_execution_session_ids,
        )

        controller.node_heartbeat(
            "shared-node",
            status={"authoritative": True, "sessions": []},
        )
        self.assertEqual(
            controller.gpu_ledger[GPU_0].state.value, "FREE"
        )
        self.assertNotIn(
            shared.execution_session_id,
            controller.nodes["shared-node"].active_sessions,
        )
        controller.assert_invariants()


class RequestAndControllerCliTests(unittest.TestCase):
    def test_request_gpu_mode_is_rejected_by_cli_and_service(self):
        parser = _build_cape_parser()
        prefix = [
            "run",
            "--pool",
            SHARED_POOL,
            "--user",
            "alice",
            "--session-id",
            "task-a",
        ]
        for option in ("--gpu-mode", "--gpu-mode=whole"):
            with self.subTest(option=option):
                arguments = prefix + [option]
                if option == "--gpu-mode":
                    arguments.append("whole")
                arguments.extend(["--", "true"])
                with mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(arguments)

        payload = {
            "pool": SHARED_POOL,
            "user_id": "alice",
            "session_id": "task-a",
            "command": ["true"],
            "resources": {"gpus": 1, "gpu_mode": "whole"},
        }
        with self.assertRaisesRegex(ValueError, "gpu_mode.*pool"):
            workload_spec_from_dict(payload)

    def test_controller_cli_builds_default_and_override_contention_policy(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        state = str(Path(temporary.name) / "state")
        credential = str(Path(temporary.name) / "admin.json")
        self.assertEqual(cape_main(["controller", "init", "--state-dir", state, "--user", "admin", "--token-out", credential], stdout=io.StringIO()), 0)
        service = mock.Mock()
        service.url = "http://127.0.0.1:18892"
        stdout = io.StringIO()
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
                        "--state-dir", state, "--token-file", credential,
                        "--gpu-contention-policy",
                        "ratio-only",
                        "--gpu-shared-ratio",
                        "7",
                        "--gpu-share-override",
                        GPU_0 + "=exclusive-first:1",
                    ],
                    stdout=stdout,
                ),
                0,
            )

        controller = service_factory.call_args.kwargs["controller"]
        self.assertEqual(
            controller.contention_policy_for_gpu(GPU_1),
            ("ratio-only", 7),
        )
        self.assertEqual(
            controller.contention_policy_for_gpu(GPU_0),
            ("exclusive-first", 1),
        )
        service.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
