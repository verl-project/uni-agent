# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Queued-to-running timeout: real CLI/controller wait path, with protocol and real-agent variants."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import urllib.error
import uuid

from pycape.identity import new_secret, verifier
from pycape.workspace import WorkspaceConfig
from tests.managed_fixture import ManagedFixture


class RunningTimeoutReproMixin:
    binary_variable = None
    binary_default = None

    def test_running_timeout_after_queue_protocol(self):
        """Simulated agent remains running; no FUSE, GPU or workload execution."""
        self.reproduce(real_agent=False)

    @unittest.skipUnless(os.environ.get('CAPE_RUNNING_TIMEOUT_LIVE') == '1',
                         'set CAPE_RUNNING_TIMEOUT_LIVE=1 on a FUSE-capable host')
    def test_running_timeout_after_queue_real_agent(self):
        """Actual local-process agent and sleep 1, with normal workspace probing."""
        self.reproduce(real_agent=True)

    def reproduce(self, real_agent):
        selected = os.environ.get(self.binary_variable, self.binary_default)
        binary = shutil.which(selected)
        if not binary:
            if self.binary_variable in os.environ:
                self.fail('explicit binary is unavailable: ' + selected)
            self.skipTest('binary unavailable: ' + selected)
        with tempfile.TemporaryDirectory(prefix='cape-queued-running-timeout-') as directory:
            root = Path(directory)
            fixture = ManagedFixture(binary, root)
            processes = []
            timeline = []
            with (root / 'controller.log').open('w+') as controller_log, (root / 'agent.log').open('w+') as agent_log:
                try:
                    controller = fixture.start(binary, '--schedule-interval', '0.01', output=controller_log)
                    processes.append(controller)
                    pool = fixture.pool('queued-running-timeout')
                    user = fixture.user_file('alice')
                    owner = fixture.user_token('contributor')
                    node_args = fixture.node_credentials('queued-running-timeout-node', 'queued-running-timeout') if real_agent else None
                    secret, epoch = new_secret(), uuid.uuid4().hex
                    resources = {'cpu_cores': 1, 'memory_gb': 1, 'gpu_uuids': []}
                    if not real_agent:
                        fixture.request('POST', '/v1/identity/nodes/enroll', {
                            'operation_id': str(uuid.uuid4()), 'verifier': verifier(secret),
                            'node_id': 'queued-running-timeout-node', 'pool': pool, 'allocation_id': 'queued-running-timeout-allocation',
                            'agent_epoch': epoch, 'resources': resources}, owner)
                    env = {k: v for k, v in os.environ.items() if k not in ('CAPE_TOKEN', 'CAPE_TOKEN_FILE')}
                    before = time.monotonic()
                    client = subprocess.Popen([
                        binary, 'run', '--controller-url', fixture.url, '--token-file', str(user),
                        '--pool', 'queued-running-timeout', '--session-id', 'queued-running-timeout', '--gpus', '0',
                        '--runtime-adapter', 'local-process', '--isolation-policy', 'none',
                        '--wait', '--timeout', '2', '--running-timeout', '0.15',
                        '--cancel-on-timeout', '--cancel-reason', 'test_running_timeout',
                        '--', '/bin/sleep', '1'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    processes.append(client)
                    queued_at = None
                    registered = False
                    agent = None
                    while client.poll() is None and time.monotonic() - before < 12:
                        try:
                            status = fixture.request('GET', '/v1/requests/req-000001')
                        except urllib.error.HTTPError as error:
                            if error.code != 404:
                                raise
                        else:
                            timeline.append({'elapsed': time.monotonic() - before,
                                             'state': status['state'], 'started_at': status.get('started_at')})
                            if status['state'] == 'QUEUED' and queued_at is None:
                                queued_at = time.monotonic()
                        if queued_at is not None and not registered and time.monotonic() - queued_at >= .2:
                            if real_agent:
                                agent = subprocess.Popen([
                                    binary, 'node', 'enroll', '--controller-url', fixture.url,
                                    '--pool', 'queued-running-timeout', '--node-id', 'queued-running-timeout-node', *node_args,
                                    '--cpu-cores', '1', '--memory-gb', '1', '--runtime-adapter', 'local-process',
                                    '--heartbeat-interval', '.02', '--state-path', str(root / 'agent.json'),
                                    '--log-dir', str(root / 'commands'), '--runtime-dir', str(root / 'runtime')],
                                    env=env, stdout=agent_log, stderr=agent_log)
                                processes.append(agent)
                            else:
                                capability = {'adapter': 'local-process', 'image_formats': ['default'],
                                              'isolation_policies': ['default', 'none'],
                                              'gpu_isolation_policies': ['default', 'none'],
                                              'can_select_gpus': True, 'gpu_enforcement': 'none'}
                                fixture.request('POST', '/v1/agent/register', {
                                    'node_id': 'queued-running-timeout-node', 'pool': pool, 'agent_epoch': epoch,
                                    'allocation_id': 'queued-running-timeout-allocation', 'resources': resources,
                                    'capabilities': capability, 'runtime_capabilities': {'local-process': capability},
                                    'workspace_config': {'state': 'applied', 'applied_version': 0,
                                                         'config': WorkspaceConfig().to_dict()}}, secret)
                                fixture.request('POST', '/v1/agent/heartbeat', {
                                    'node_id': 'queued-running-timeout-node', 'agent_epoch': epoch, 'max_commands': 8}, secret)
                            registered = True
                        if agent is not None and agent.poll() is not None:
                            raise AssertionError('real agent exited before wait completed')
                        time.sleep(.02)
                    stdout, stderr = client.communicate(timeout=3)
                    elapsed = time.monotonic() - before
                    self.assertEqual(client.returncode, 0, stderr)
                    payload = json.loads(stdout)
                    self.assertIsNotNone(queued_at)
                    self.assertEqual(payload['state'], 'CANCELLED')
                    self.assertEqual(payload['reason'], 'test_running_timeout')
                    self.assertTrue(payload['wait_timed_out'])
                    self.assertEqual(payload['wait_timeout_kind'], 'running')
                    self.assertGreaterEqual(payload['started_at'] - payload['created_at'], .1)
                    self.assertGreaterEqual(elapsed, .3)
                    self.assertLess(elapsed, 1.5)
                    print(json.dumps({'implementation': self.binary_default, 'real_agent': real_agent,
                                      'elapsed': elapsed, 'wait_timeout_kind': payload['wait_timeout_kind'],
                                      'timeline': timeline}), flush=True)
                except Exception as error:
                    controller_log.flush(); agent_log.flush()
                    controller_log.seek(0); agent_log.seek(0)
                    raise AssertionError('{}\ntimeline={}\ncontroller={}\nagent={}'.format(
                        error, json.dumps(timeline), controller_log.read()[-4000:], agent_log.read()[-4000:])) from error
                finally:
                    for process in reversed(processes):
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.communicate(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate(timeout=5)
