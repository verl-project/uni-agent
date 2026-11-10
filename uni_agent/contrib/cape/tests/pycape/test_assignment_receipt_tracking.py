# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from unittest import mock

from pycape import NodeAgentClient, NodeResources


class AssignmentReceiptTrackingTests(unittest.TestCase):
    def test_batch_is_visible_before_response_processing_or_worker_start(self):
        with tempfile.TemporaryDirectory() as root:
            agent = NodeAgentClient('http://unused', 'node', 'pool', NodeResources(1, 1), log_dir=root)
            agent.metrics_collector = None
            payload = {'commands': [{'command_id': 'c1', 'execution_session_id': 's1', 'request_id': 'r1'},
                                    {'command_id': 'c2', 'execution_session_id': 's2', 'request_id': 'r2'}]}
            def slow_response(_):
                status = agent.collect_status()
                self.assertEqual(set(status['command_ids']), {'c1', 'c2'})
                self.assertEqual({s['execution_session_id'] for s in status['sessions']}, {'s1', 's2'})
            with mock.patch.object(agent, '_post', return_value=payload), \
                 mock.patch.object(agent, '_workspace_response', side_effect=slow_response), \
                 mock.patch.object(agent, '_dispatch_response_commands'):
                agent.heartbeat_once()

    def test_pending_result_stays_visible_until_report_ack(self):
        with tempfile.TemporaryDirectory() as root:
            agent = NodeAgentClient('http://unused', 'node', 'pool', NodeResources(1, 1), log_dir=root)
            agent.metrics_collector = None
            agent._pending_command_reports.add('completed-but-unreported')
            self.assertIn('completed-but-unreported', agent.collect_status()['command_ids'])
            with mock.patch.object(agent, '_post', return_value={}):
                agent._report('completed-but-unreported', 0, '', '', None)
            self.assertNotIn('completed-but-unreported', agent.collect_status()['command_ids'])

    def test_response_error_removes_only_new_reservations(self):
        with tempfile.TemporaryDirectory() as root:
            agent = NodeAgentClient('http://unused', 'node', 'pool', NodeResources(1, 1), log_dir=root)
            agent.metrics_collector = None
            old = {'command_id': 'old', 'execution_session_id': 'old-session', 'request_id': 'old-request'}
            agent._reserve_response_commands({'commands': [old]})
            new = {'command_id': 'new', 'execution_session_id': 'new-session', 'request_id': 'new-request'}
            with mock.patch.object(agent, '_workspace_response', side_effect=OSError('injected metadata failure')):
                with self.assertRaises(OSError):
                    agent._prepare_received_response({'commands': [old, new]})
            self.assertEqual(agent.collect_status()['command_ids'], ['old'])

    def test_reservation_failure_and_closed_session_do_not_advertise_new_work(self):
        with tempfile.TemporaryDirectory() as root:
            agent = NodeAgentClient('http://unused', 'node', 'pool', NodeResources(1, 1), log_dir=root)
            command = {'command_id': 'new', 'execution_session_id': 'session', 'request_id': 'request'}
            with mock.patch.object(agent, '_save_state', side_effect=OSError('injected disk failure')):
                with self.assertRaises(OSError):
                    agent._reserve_response_commands({'commands': [command]})
            self.assertEqual(agent.running_commands, {})
            agent._closed_workspace_sessions.add('session')
            agent._reserve_response_commands({'commands': [command]})
            self.assertEqual(agent.running_commands, {})


if __name__ == '__main__':
    unittest.main()
