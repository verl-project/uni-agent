# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lost dispatch-response regression. CPU-only, deterministic clock, no real agent.

Run from cape/: python -m unittest \
    tests.pycape.test_assignment_delivery_reconciliation -v

The regression must fail until a live missing assignment converges safely.
Do not hide it with expectedFailure or treat phantom RUNNING as a passing result.
"""
import json
import unittest

from pycape import RequestState, SchedulerPolicy, SessionState
from pycape.models import CommandState
from tests.pycape.test_pycape import POOL, controller, register_gpu_node, spec


class AssignmentDeliveryReconciliationTests(unittest.TestCase):
    def scenario(self):
        ctrl, clock = controller(policy=SchedulerPolicy(
            reconcile_missing_grace_seconds=2, reconcile_missing_heartbeats=2))
        register_gpu_node(ctrl, gpus=1)
        request = ctrl.submit_workload(spec(reuse=False, session_id='lost-response'))
        ctrl.schedule_once()
        self.assertEqual(request.state, RequestState.RUNNING)
        return ctrl, clock, request

    def test_dropped_assignment_converges_without_replay_in_active_and_draining_nodes(self):
        for draining in (False, True):
            with self.subTest(draining=draining):
                ctrl, clock, request = self.scenario()
                response = ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})
                assignments = [c for c in response['commands'] if c.get('action', 'run') == 'run']
                self.assertEqual(len(assignments), 1)
                self.assertEqual(assignments[0]['request_id'], request.request_id)
                # Intentionally discard the response; no worker ever receives/runs it.
                if draining:
                    ctrl.drain_node('node-a')
                observations = []
                for _ in range(6):
                    clock.advance(1)
                    reply = ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})
                    self.assertFalse(any(c.get('action', 'run') == 'run' for c in reply['commands']),
                                     'uncertain commands must not be blindly replayed')
                    observations.append({'time': clock(), 'request_state': request.state.value,
                                         'running_requests': ctrl.pool_utilization(POOL)['running_requests']})
                utilization = ctrl.pool_utilization(POOL)
                self.assertEqual(
                    (request.state in (RequestState.FAILED, RequestState.LOST),
                     utilization['running_requests'], utilization['gpus_leased']),
                    (True, 0, 0),
                    'missing assignment retained phantom execution/capacity: ' + json.dumps(observations))
                ctrl.assert_invariants()

    def test_pending_not_yet_delivered_is_not_a_missing_assignment(self):
        ctrl, clock, request = self.scenario()
        for _ in range(6):
            clock.advance(1)
            ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []}, max_commands=0)
        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertEqual(ctrl.pool_utilization(POOL)['gpus_leased'], 1)
        reply = ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})
        self.assertTrue(any(c.get('request_id') == request.request_id for c in reply['commands']))

    def test_restored_missing_assignment_closes_command_as_well_as_request(self):
        ctrl, clock, request = self.scenario()
        command = ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})['commands'][0]
        ctrl.sessions[request.execution_session_id].state = SessionState.RECONCILING
        request.state = RequestState.RECONCILING
        ctrl.drain_node('node-a')
        for _ in range(6):
            clock.advance(1)
            ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})
        self.assertEqual(request.state, RequestState.FAILED)
        self.assertEqual(request.reason, 'reconcile_missing')
        self.assertEqual(ctrl.command_assignments[command['command_id']].state, CommandState.FAILED)
        self.assertEqual(ctrl.pool_utilization(POOL)['gpus_leased'], 0)

    def test_reported_running_session_remains_live(self):
        ctrl, clock, request = self.scenario()
        ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})
        for _ in range(6):
            clock.advance(1)
            ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': [{
                'execution_session_id': request.execution_session_id, 'state': 'RUNNING_COMMAND',
                'active_request_id': request.request_id}]})
        self.assertEqual(request.state, RequestState.RUNNING)
        self.assertEqual(ctrl.pool_utilization(POOL)['gpus_leased'], 1)
        ctrl.assert_invariants()

    def test_reused_idle_session_missing_command_and_pending_report_are_distinct(self):
        for reporting in (False, True):
            with self.subTest(reporting=reporting):
                ctrl, clock, request = self.scenario()
                command = ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})['commands'][0]
                for _ in range(6):
                    clock.advance(1)
                    ctrl.node_heartbeat('node-a', status={'authoritative': True,
                        'sessions': [{'execution_session_id': request.execution_session_id, 'state': 'CACHED_IDLE'}],
                        'command_ids': [command['command_id']] if reporting else []})
                if reporting:
                    self.assertEqual(request.state, RequestState.RUNNING)
                    self.assertEqual(ctrl.pool_utilization(POOL)['gpus_leased'], 1)
                else:
                    self.assertEqual(request.state, RequestState.FAILED)
                    self.assertEqual(request.reason, 'assignment_missing')
                    self.assertEqual(ctrl.pool_utilization(POOL)['gpus_leased'], 0)

    def test_missing_observations_reset_and_late_success_cannot_revive(self):
        ctrl, clock, request = self.scenario()
        command = ctrl.node_heartbeat('node-a', status={'authoritative': True, 'sessions': []})['commands'][0]
        empty = {'authoritative': True, 'sessions': [], 'command_ids': []}
        # An old/single empty snapshot must not fail a just-dispatched command.
        ctrl.node_heartbeat('node-a', status=empty)
        clock.advance(1)
        ctrl.node_heartbeat('node-a', status=empty)
        self.assertEqual(request.state, RequestState.RUNNING)
        for barrier in ({'authoritative': False, 'sessions': []},
                        {'authoritative': True},
                        {'authoritative': True, 'sessions': [], 'command_ids': 'invalid'},
                        {'authoritative': True, 'sessions': [], 'command_ids': [command['command_id']]}):
            ctrl.node_heartbeat('node-a', status=barrier)
            clock.advance(3)
            ctrl.node_heartbeat('node-a', status=empty)
            self.assertEqual(request.state, RequestState.RUNNING)
        clock.advance(3)
        ctrl.node_heartbeat('node-a', status=empty)
        self.assertEqual(request.reason, 'assignment_missing')
        self.assertEqual(ctrl.command_assignments[command['command_id']].state, CommandState.FAILED)
        ended = request.ended_at
        ctrl.report_command_result('node-a', command['command_id'], exit_code=0)
        self.assertEqual((request.state, request.reason, request.ended_at),
                         (RequestState.FAILED, 'assignment_missing', ended))
        self.assertEqual(ctrl.command_assignments[command['command_id']].state, CommandState.FAILED)
        successor = ctrl.submit_workload(spec(reuse=False, session_id='successor'))
        ctrl.schedule_once()
        self.assertEqual(successor.state, RequestState.RUNNING)
        ctrl.assert_invariants()


if __name__ == '__main__':
    unittest.main()
