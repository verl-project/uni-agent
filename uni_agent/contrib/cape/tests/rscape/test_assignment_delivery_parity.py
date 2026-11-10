# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real paired controllers, simulated agent transport, no GPU or workload execution."""
import subprocess
import tempfile
import time
import unittest

from tests.rscape import test_functional_parity as parity


class AssignmentDeliveryParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parity.FunctionalParityTests.setUpClass()

    def wait_retired(self, helper, url):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            value = helper.request('GET', url, '/v1/pools/delivery/utilization')
            if value['nodes'] == 0:
                return value
            time.sleep(.02)
        self.fail('drain did not converge: ' + repr(value))

    def test_restart_then_drain_missing_assignment_has_no_live_command_blocker(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as root:
                helper = parity.FunctionalParityTests()
                options = ('--reconcile-missing-grace-seconds', '0', '--reconcile-missing-heartbeats', '2')
                controller, url, _ = helper.start_controller(binary, root, *options)
                try:
                    helper.register(url, 'node-a', 'delivery')
                    fixture = helper.fixtures[url]
                    submitted = helper.request('POST', url, '/v1/submit', {
                        'pool': 'delivery', 'user_id': 'alice', 'session_id': 'lost-before-restart', 'reuse': False,
                        'command': ['true'], 'resources': {'gpus': 1, 'cpu_cores': 1, 'memory_gb': 1,
                        'runtime_adapter': 'local-process', 'isolation_policy': 'none'}})
                    helper.wait_state(url, submitted['request_id'], 'RUNNING', timeout=5)
                    empty = {'node_id': 'node-a', 'status': {'authoritative': True, 'sessions': [], 'command_ids': []}}
                    reply = helper.request('POST', url, '/v1/agent/heartbeat', empty)
                    self.assertTrue(any(c.get('request_id') == submitted['request_id'] for c in reply['commands']))
                    helper.stop_controller(controller)
                    controller = fixture.start(binary, *options)
                    result = subprocess.run([binary, 'node', 'drain', 'node-a', '--controller-url', url,
                        '--token-file', str(fixture.user_file('contributor'))], capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for _ in range(4):
                        reply = helper.request('POST', url, '/v1/agent/heartbeat', empty)
                        self.assertFalse(any(c.get('action', 'run') == 'run' for c in reply['commands']))
                    status = helper.request('GET', url, '/v1/requests/' + submitted['request_id'])
                    self.assertEqual((status['state'], status['reason']), ('FAILED', 'reconcile_missing'))
                    utilization = self.wait_retired(helper, url)
                    self.assertEqual((utilization['nodes'], utilization['running_requests'], utilization['gpus_leased']), (0, 0, 0))
                finally:
                    helper.stop_controller(controller)

    def test_lost_delivery_terminates_without_replay_and_unblocks_capacity(self):
        for binary in (parity.PYCAPE_BIN, parity.RSCAPE_BIN):
            for through_report in (False, True):
                for draining in (False, True):
                    with self.subTest(binary=binary, through_report=through_report, draining=draining), tempfile.TemporaryDirectory() as root:
                        helper = parity.FunctionalParityTests()
                        controller, url, _ = helper.start_controller(binary, root,
                            '--reconcile-missing-grace-seconds', '0', '--reconcile-missing-heartbeats', '2')
                        try:
                            helper.register(url, 'node-a', 'delivery')
                            fixture = helper.fixtures[url]
                            def submit(task):
                                return helper.request('POST', url, '/v1/submit', {
                                    'pool': 'delivery', 'user_id': 'alice', 'session_id': task, 'reuse': False,
                                    'command': ['true'], 'resources': {'gpus': 1, 'cpu_cores': 1,
                                    'memory_gb': 1, 'runtime_adapter': 'local-process', 'isolation_policy': 'none'}})
                            def heartbeat(status):
                                return helper.request('POST', url, '/v1/agent/heartbeat',
                                    {'node_id': 'node-a', 'status': status, 'max_commands': 8})
                            empty = {'authoritative': True, 'sessions': [], 'command_ids': []}
                            first = submit('first')
                            helper.wait_state(url, first['request_id'], 'RUNNING', timeout=5)
                            response = heartbeat(empty)
                            command = next(c for c in response['commands'] if c.get('action', 'run') == 'run')
                            target = first
                            if through_report:
                                target = submit('second')
                                response = helper.request('POST', url, '/v1/agent/report-command', {
                                    'node_id': 'node-a', 'command_id': command['command_id'], 'exit_code': 0,
                                    'failure_kind': 'workload', 'accept_commands': True, 'max_commands': 8})
                                command = next(c for c in response['commands'] if c.get('request_id') == target['request_id'])
                            # Drop this response instead of executing its command.
                            if draining:
                                result = subprocess.run([binary, 'node', 'drain', 'node-a',
                                    '--controller-url', url, '--token-file', str(fixture.user_file('contributor'))],
                                    text=True, capture_output=True, timeout=10)
                                self.assertEqual(result.returncode, 0, result.stderr)
                            for _ in range(3):
                                response = heartbeat(empty)
                                self.assertFalse(any(c.get('action', 'run') == 'run' for c in response['commands']))
                            state = helper.request('GET', url, '/v1/requests/' + target['request_id'])
                            self.assertEqual((state['state'], state['reason'], state['failure_kind']),
                                             ('FAILED', 'assignment_missing', 'infrastructure'))
                            util = helper.request('GET', url, '/v1/pools/delivery/utilization')
                            self.assertEqual((util['running_requests'], util['gpus_leased']), (0, 0))
                            if draining:
                                self.wait_retired(helper, url)
                            if not draining:
                                helper.request('POST', url, '/v1/agent/report-command', {
                                    'node_id': 'node-a', 'command_id': command['command_id'],
                                    'exit_code': 0, 'failure_kind': 'workload'})
                                state = helper.request('GET', url, '/v1/requests/' + target['request_id'])
                                self.assertEqual((state['state'], state['reason']), ('FAILED', 'assignment_missing'))
                                successor = submit('successor')
                                helper.wait_state(url, successor['request_id'], 'RUNNING', timeout=5)
                                response = heartbeat(empty)
                                self.assertTrue(any(c.get('request_id') == successor['request_id'] for c in response['commands']))
                        finally:
                            helper.stop_controller(controller)
