# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pycape import (
    CAPEController,
    CAPEControllerService,
    EnforcementMode,
    MemoryJournal,
    NodeAgentClient,
    NodeResources,
    RequestState,
    ResourceRequest,
    Role,
    RuntimeCapabilities,
    SessionState,
    TokenAuthority,
    WorkloadSpec,
)
from pycape.agent import AgentHTTPError
from pycape.cli import _request, CapeHTTPError


POOL = "shared-pool"
OTHER_POOL = "exclusive-pool"
GPU = "GPU-00000000-0000-0000-0000-000000000000"


def capabilities():
    return RuntimeCapabilities(
        adapter="fake",
        gpu_enforcement=EnforcementMode.HARD,
    )


def register(
    controller,
    node_id,
    slots,
    pool=POOL,
    mode="shared",
    allocation_id=None,
):
    return controller.register_node(
        node_id=node_id,
        pool=pool,
        resources=NodeResources(8, 64, (GPU,)),
        capabilities=capabilities(),
        allocation_id=allocation_id or "alloc-" + node_id,
        pool_mode=mode,
        gpu_slots=slots,
        placement="pack",
    )


def workload(pool, name, reuse=False):
    return WorkloadSpec(
        pool=pool,
        user_id="user-" + name,
        session_id=name,
        command=("true",),
        reuse=reuse,
        resources=ResourceRequest(
            gpus=1,
            cpu_cores=1,
            memory_gb=1,
            max_duration_seconds=60,
        ),
    )


def config_tuple(controller, pool=POOL):
    config = controller.pool_configs[pool]
    return (config.pool_mode, config.gpu_slots, config.placement)


def conflict_payload(context):
    payload = getattr(context.exception, "payload", None)
    if payload is None:
        raise AssertionError("pool configuration error has no structured payload")
    return payload


class EmptyPoolImplicitReplacementTests(unittest.TestCase):
    def test_exact_repro_retires_k1_rejoins_k4_and_third_k1_join_fails(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        register(
            controller,
            "shared-node",
            1,
            allocation_id="alloc-k1",
        )
        controller.remove_allocation("alloc-k1")

        rejoined = register(
            controller,
            "shared-node",
            4,
            allocation_id="alloc-k4",
        )

        self.assertEqual(rejoined.gpu_slots, 4)
        self.assertEqual(config_tuple(controller), ("shared", 4, "pack"))
        replacements = [
            event
            for event in journal.list_events()
            if event.event_type == "pool_config_replaced"
        ]
        self.assertEqual(len(replacements), 1)
        self.assertEqual(
            replacements[0].details,
            {
                "joining_node_id": "shared-node",
                "new_config": {
                    "gpu_slots": 4,
                    "placement": "pack",
                    "pool_mode": "shared",
                },
                "old_config": {
                    "gpu_slots": 1,
                    "placement": "pack",
                    "pool_mode": "shared",
                },
                "pool": POOL,
            },
        )

        with self.assertRaises(ValueError) as raised:
            register(controller, "third-node", 1)
        payload = conflict_payload(raised)
        self.assertEqual(
            payload["blockers"]["member_node_ids"], ["shared-node"]
        )
        self.assertIn("drain all members", payload["hint"])
        self.assertIn("pycape admin pool edit", payload["hint"])

    def test_live_member_blocks_replacement_and_is_named(self):
        controller = CAPEController()
        register(controller, "live-node", 1)

        with self.assertRaises(ValueError) as raised:
            register(controller, "mismatch-node", 4)

        self.assertEqual(
            conflict_payload(raised)["blockers"],
            {
                "member_node_ids": ["live-node"],
                "request_ids": [],
                "execution_session_ids": [],
            },
        )
        self.assertNotIn("mismatch-node", controller.nodes)

    def test_queued_request_blocks_replacement_and_is_named(self):
        controller = CAPEController()
        register(controller, "old-node", 1)
        controller.remove_allocation("alloc-old-node")
        request = controller.submit_workload(workload(POOL, "queued"))
        self.assertEqual(request.state, RequestState.QUEUED)

        with self.assertRaises(ValueError) as raised:
            register(controller, "new-node", 4)

        self.assertEqual(
            conflict_payload(raised)["blockers"],
            {
                "member_node_ids": [],
                "request_ids": [request.request_id],
                "execution_session_ids": [],
            },
        )

    def test_cached_idle_session_blocks_replacement_and_is_named(self):
        controller = CAPEController()
        node = register(controller, "cached-node", 1)
        request = controller.submit_workload(
            workload(POOL, "cached", reuse=True)
        )
        controller.schedule_once()
        controller.complete_request(request.request_id, exit_code=0)
        session = controller.sessions[request.execution_session_id]
        self.assertEqual(session.state, SessionState.CACHED_IDLE)

        # Model a controller-recovery edge where durable cached state outlives
        # visible membership. The reconfiguration predicate must still see it.
        node.state = type(node.state).REMOVED
        node.active_sessions = []

        with self.assertRaises(ValueError) as raised:
            register(controller, "new-node", 4)

        self.assertEqual(
            conflict_payload(raised)["blockers"],
            {
                "member_node_ids": [],
                "request_ids": [],
                "execution_session_ids": [session.execution_session_id],
            },
        )

    def test_join_conflict_http_payload_names_every_blocker(self):
        controller = CAPEController()
        register(controller, "live-node", 1)
        queued = controller.submit_workload(workload(POOL, "queued"))
        service = CAPEControllerService(
            controller=controller, schedule_interval=100
        ).start()
        try:
            client = NodeAgentClient(
                service.url,
                node_id="mismatch-node",
                pool=POOL,
                resources=NodeResources(8, 64, (GPU,)),
                capabilities=capabilities(),
                pool_mode="shared",
                gpu_slots=4,
            )
            with self.assertRaises(AgentHTTPError) as raised:
                client._post(
                    "/v1/agent/announce",
                    client._registration_payload(),
                )
        finally:
            service.stop()

        self.assertEqual(raised.exception.status, 400)
        self.assertEqual(
            raised.exception.payload["blockers"],
            {
                "member_node_ids": ["live-node"],
                "request_ids": [queued.request_id],
                "execution_session_ids": [queued.execution_session_id],
            },
        )


class AdministratorPoolConfigTests(unittest.TestCase):
    def authority(self):
        authority = TokenAuthority()
        authority.add_user_token(
            "admin-token", "pool-admin", POOL, (Role.ADMIN,)
        )
        authority.add_user_token(
            "submitter-token", "submitter", POOL, (Role.SUBMITTER,)
        )
        authority.add_user_token(
            "viewer-token", "viewer", POOL, (Role.VIEWER,)
        )
        authority.add_user_token(
            "operator-token", "operator", POOL, (Role.OPERATOR,)
        )
        authority.add_user_token(
            "node-token", "node", POOL, (Role.NODE_AGENT,)
        )
        return authority

    def test_admin_can_predeclare_and_first_join_must_match(self):
        controller = CAPEController()
        configured = controller.set_pool_config(
            POOL,
            pool_mode="shared",
            gpu_slots=4,
            placement="spread",
        )
        self.assertEqual(
            (
                configured.pool_mode,
                configured.gpu_slots,
                configured.placement,
            ),
            ("shared", 4, "spread"),
        )

        with self.assertRaises(ValueError) as raised:
            register(controller, "wrong-first-node", 1)
        payload = conflict_payload(raised)
        self.assertTrue(payload["requires_matching_join"])
        self.assertEqual(
            payload["blockers"],
            {
                "member_node_ids": [],
                "request_ids": [],
                "execution_session_ids": [],
            },
        )

        controller.register_node(
            node_id="matching-first-node",
            pool=POOL,
            resources=NodeResources(8, 64, (GPU,)),
            capabilities=capabilities(),
            allocation_id="alloc-matching",
            pool_mode="shared",
            gpu_slots=4,
            placement="spread",
        )
        controller.remove_allocation("alloc-matching")
        register(controller, "later-node", 2)
        self.assertEqual(config_tuple(controller), ("shared", 2, "pack"))

    def test_admin_set_on_empty_pool_records_actor_without_token(self):
        journal = MemoryJournal()
        controller = CAPEController(
            journal=journal, auth=self.authority()
        )

        controller.set_pool_config(
            POOL,
            pool_mode="shared",
            gpu_slots=3,
            placement="spread",
            token="admin-token",
        )

        event = [
            event
            for event in journal.list_events()
            if event.event_type == "pool_config_set"
        ][0]
        self.assertEqual(event.details["actor"], "pool-admin")
        self.assertEqual(
            event.details["new_config"],
            {
                "pool_mode": "shared",
                "gpu_slots": 3,
                "placement": "spread",
            },
        )
        self.assertNotIn("admin-token", json.dumps(event.details))

    def test_non_admin_roles_are_rejected(self):
        controller = CAPEController(auth=self.authority())

        for token in (
            "submitter-token",
            "viewer-token",
            "operator-token",
            "node-token",
        ):
            with self.subTest(token=token):
                with self.assertRaises(PermissionError):
                    controller.set_pool_config(
                        POOL,
                        pool_mode="shared",
                        gpu_slots=4,
                        placement="pack",
                        token=token,
                    )
        self.assertNotIn(POOL, controller.pool_configs)

    def test_admin_rejection_payload_names_live_request_and_session_blockers(self):
        controller = CAPEController(auth=self.authority())
        controller.auth = None
        node = register(controller, "live-node", 1)
        cached_request = controller.submit_workload(
            workload(POOL, "cached", reuse=True)
        )
        controller.schedule_once()
        controller.complete_request(cached_request.request_id, exit_code=0)
        queued = controller.submit_workload(workload(POOL, "queued"))
        controller.auth = self.authority()

        with self.assertRaises(ValueError) as raised:
            controller.set_pool_config(
                POOL,
                pool_mode="shared",
                gpu_slots=4,
                placement="pack",
                token="admin-token",
            )

        payload = conflict_payload(raised)
        self.assertEqual(payload["blockers"]["member_node_ids"], [node.node_id])
        self.assertEqual(payload["blockers"]["request_ids"], [queued.request_id])
        self.assertEqual(
            payload["blockers"]["execution_session_ids"],
            [cached_request.execution_session_id],
        )

    def test_admin_http_sets_config_and_returns_empty_pool_status(self):
        controller = CAPEController(auth=self.authority())
        service = CAPEControllerService(
            controller=controller, schedule_interval=100
        ).start()
        output = io.StringIO()
        try:
            payload = _request("POST", service.url, "/v1/admin/set-pool-config",
                               token="admin-token", payload={"pool": POOL, "pool_mode": "shared",
                                                             "gpu_slots": 4, "placement": "spread"})
        finally:
            service.stop()

        self.assertEqual(
            payload["pool_config"],
            {
                "gpu_slots": 4,
                "placement": "spread",
                "pool_mode": "shared",
            },
        )
        self.assertTrue(payload["requires_matching_join"])
        self.assertEqual(payload["nodes"], 0)
        self.assertEqual(payload["gpu_slots_total"], 0)

    def test_admin_http_conflict_is_structured_4xx(self):
        controller = CAPEController()
        register(controller, "live-node", 1)
        controller.auth = self.authority()
        service = CAPEControllerService(
            controller=controller, schedule_interval=100
        ).start()
        stderr = io.StringIO()
        try:
            with self.assertRaises(CapeHTTPError) as caught:
                _request("POST", service.url, "/v1/admin/set-pool-config",
                         token="admin-token", payload={"pool": POOL, "pool_mode": "shared",
                                                       "gpu_slots": 4, "placement": "pack"})
        finally:
            service.stop()

        self.assertEqual(caught.exception.status, 400)
        payload = {"error": caught.exception.payload}
        self.assertEqual(
            payload["error"]["blockers"]["member_node_ids"], ["live-node"]
        )


class PoolConfigReplayAndLedgerTests(unittest.TestCase):
    def test_empty_pool_config_is_visible_without_inventing_capacity(self):
        controller = CAPEController()
        controller.set_pool_config(
            POOL,
            pool_mode="shared",
            gpu_slots=4,
            placement="spread",
        )

        summary = controller.pool_utilization(POOL)

        self.assertEqual(summary["nodes"], 0)
        self.assertEqual(summary["gpu_slots_total"], 0)
        self.assertEqual(
            summary["pool_config"],
            {
                "pool_mode": "shared",
                "gpu_slots": 4,
                "placement": "spread",
            },
        )
        self.assertTrue(summary["requires_matching_join"])

    def test_replaced_config_wins_over_removed_tombstones_on_replay(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        register(controller, "old-node", 1)
        controller.remove_allocation("alloc-old-node")
        register(controller, "new-node", 4)
        controller.remove_allocation("alloc-new-node")

        replayed = CAPEController.reconcile_from_journal(journal)

        self.assertEqual(config_tuple(replayed), ("shared", 4, "pack"))
        self.assertEqual(
            replayed.pool_utilization(POOL)["pool_config"]["gpu_slots"], 4
        )
        replayed.assert_invariants()

    def test_normally_established_empty_pool_remains_visible_on_replay(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        register(controller, "only-node", 2)
        controller.remove_allocation("alloc-only-node")

        replayed = CAPEController.reconcile_from_journal(journal)

        self.assertEqual(config_tuple(replayed), ("shared", 2, "pack"))
        summary = replayed.pool_utilization(POOL)
        self.assertEqual(summary["nodes"], 0)
        self.assertEqual(summary["gpu_slots_total"], 0)
        self.assertEqual(summary["pool_config"]["gpu_slots"], 2)
        self.assertFalse(summary["requires_matching_join"])

    def test_admin_predeclaration_pin_survives_replay(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        controller.set_pool_config(
            POOL,
            pool_mode="shared",
            gpu_slots=4,
            placement="pack",
        )

        replayed = CAPEController.reconcile_from_journal(journal)

        with self.assertRaises(ValueError) as raised:
            register(replayed, "wrong-node", 1)
        self.assertTrue(
            conflict_payload(raised)["requires_matching_join"]
        )

    def test_replacement_does_not_touch_other_pool_physical_lease(self):
        journal = MemoryJournal()
        controller = CAPEController(journal=journal)
        register(
            controller,
            "exclusive-node",
            1,
            pool=OTHER_POOL,
            mode="exclusive",
        )
        register(controller, "shared-old", 1)
        controller.remove_allocation("alloc-shared-old")
        exclusive = controller.submit_workload(
            workload(OTHER_POOL, "exclusive")
        )
        controller.schedule_once()
        self.assertEqual(exclusive.state, RequestState.RUNNING)
        entry = controller.gpu_ledger[GPU]
        holder = entry.exclusive_execution_session_id

        register(controller, "shared-new", 4)

        self.assertIs(controller.gpu_ledger[GPU], entry)
        self.assertEqual(entry.state.value, "EXCLUSIVE")
        self.assertEqual(entry.exclusive_execution_session_id, holder)
        self.assertEqual(entry.shared_refcount, 0)
        self.assertEqual(config_tuple(controller), ("shared", 4, "pack"))
        self.assertEqual(
            config_tuple(controller, OTHER_POOL),
            ("exclusive", 1, "pack"),
        )

        replayed = CAPEController.reconcile_from_journal(journal)
        self.assertEqual(
            replayed.gpu_ledger[GPU].state.value, "EXCLUSIVE"
        )
        self.assertEqual(config_tuple(replayed), ("shared", 4, "pack"))
        self.assertEqual(
            config_tuple(replayed, OTHER_POOL),
            ("exclusive", 1, "pack"),
        )
        replayed.assert_invariants()


if __name__ == "__main__":
    unittest.main()
