# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only cooperative-lock tests; optional real Bubblewrap bind check.

Run: python -m unittest tests.test_gpu_timing_lock -v
Set CAPE_GPU_TIMING_LOCK_BIND_TEST=1 to require real Bubblewrap execution.
Synthetic UUIDs identify locks only: no CUDA discovery, kernel or GPU isolation
claim. Mount-construction tests cover Apptainer, Podman and Bubblewrap; the optional
bind test runs two separate CAPE-built Bubblewrap sessions, not a scheduler lease.
"""
import inspect
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import unittest

from pycape import (ApptainerRuntimeAdapter, BubbleWrapRuntimeAdapter,
                    PodmanRuntimeAdapter, ContainerSessionSpec, ContainerCommandSpec, MountSpec)
from pycape.runtime import leased_gpu_visibility
from tests import gpu_timing_lock_example as example

GPU_A = 'GPU-00000000-0000-0000-0000-000000000001'
GPU_B = 'GPU-00000000-0000-0000-0000-000000000002'
WORKER = inspect.getsource(example) + '''
import sys
try:
    fd = acquire(sys.argv[1], os.environ.get('CUDA_VISIBLE_DEVICES'),
                 lambda: print('blocked', flush=True))
except ValueError as error:
    print(str(error), file=sys.stderr)
    raise SystemExit(2)
with os.fdopen(fd, 'r+'):
    print('acquired', flush=True)
    sys.stdin.readline()
print('released', flush=True)
'''


class GPUTimingLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cape-gpu-timing-lock-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.processes = []
        self.addCleanup(self.stop_workers)

    def stop_workers(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)

    def start(self, uuid, directory=None, sandbox=False):
        directory = directory or self.root
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=leased_gpu_visibility([uuid], {}))
        args = [sys.executable, '-u', '-c', WORKER, str(directory)]
        if sandbox:
            session = ContainerSessionSpec('lock-' + str(len(self.processes)),
                isolation_policy='host-stateless', mounts=(MountSpec(str(directory), '/runtime/gpulock'),))
            # CPU-only fixture: synthetic lock key, injected inside the sandbox.
            command = ContainerCommandSpec(('/usr/bin/env', 'CUDA_VISIBLE_DEVICES=' + uuid,
                '/usr/bin/python3', '-u', '-c', WORKER, '/runtime/gpulock'))
            launch = BubbleWrapRuntimeAdapter().build_run_command(session, command)
            args, env = launch.argv, dict(os.environ, **launch.env)
        process = subprocess.Popen(args, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def event(self, process, expected):
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            self.assertTrue(selector.select(timeout=10), 'worker event timeout: ' + expected)
        line = process.stdout.readline().strip()
        if not line:
            self.fail('worker exited without event: ' + process.stderr.read())
        self.assertEqual(line, expected)

    def release(self, process):
        process.stdin.write('\n')
        process.stdin.flush()
        self.event(process, 'released')
        self.assertEqual(process.wait(timeout=5), 0)

    def check_same_gpu(self, sandbox=False):
        dev = self.start(GPU_A, sandbox=sandbox)
        self.event(dev, 'acquired')
        formal = self.start(GPU_A, sandbox=sandbox)
        self.event(formal, 'blocked')
        self.assertIsNone(formal.poll())
        inode = (self.root / example.lock_name(GPU_A)).stat().st_ino
        self.release(dev)
        self.event(formal, 'acquired')
        self.release(formal)
        self.assertEqual((self.root / example.lock_name(GPU_A)).stat().st_ino, inode)

    def test_same_gpu_dev_and_formal_commands_are_mutually_exclusive(self):
        self.check_same_gpu()

    def test_different_gpu_locks_can_be_held_simultaneously(self):
        first = self.start(GPU_A)
        self.event(first, 'acquired')
        second = self.start(GPU_B)
        self.event(second, 'acquired')
        self.release(second)
        self.release(first)

    def test_holder_exit_releases_lock_without_deleting_file(self):
        holder = self.start(GPU_A)
        self.event(holder, 'acquired')
        waiter = self.start(GPU_A)
        self.event(waiter, 'blocked')
        holder.kill()
        holder.wait(timeout=5)
        self.event(waiter, 'acquired')
        self.release(waiter)
        self.assertTrue((self.root / example.lock_name(GPU_A)).is_file())

    def test_separate_directories_do_not_coordinate_the_same_uuid(self):
        other = self.root / 'not-shared'
        other.mkdir()
        first = self.start(GPU_A)
        self.event(first, 'acquired')
        second = self.start(GPU_A, directory=other)
        self.event(second, 'acquired')
        self.release(second)
        self.release(first)

    def test_missing_empty_numeric_and_multigpu_masks_have_no_fallback(self):
        for mask in (None, '', ' ', '\t', '0', 'GPU-other', GPU_A + ',' + GPU_B, '../escape'):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                example.acquire(self.root, mask)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(example.lock_name(' ' + GPU_A + ' '), GPU_A + '.lock')

    def test_all_container_adapters_preserve_shared_readwrite_bind(self):
        for adapter, policy in ((ApptainerRuntimeAdapter(), 'image-stateless'),
                                (PodmanRuntimeAdapter(), 'image-stateless'),
                                (BubbleWrapRuntimeAdapter(), 'host-stateless')):
            for role in ('dev', 'formal'):
                with self.subTest(adapter=adapter.name, role=role):
                    session = ContainerSessionSpec(role, image='docker://ubuntu:24.04',
                        isolation_policy=policy, mounts=(MountSpec(str(self.root), '/runtime/gpulock'),))
                    argv = adapter.build_run_command(session, ContainerCommandSpec(('true',))).argv
                    if adapter.name == 'bubblewrap':
                        self.assertTrue(any(argv[i:i + 3] == ['--bind', str(self.root), '/runtime/gpulock']
                                            for i in range(len(argv))))
                    else:
                        self.assertIn(str(self.root) + ':/runtime/gpulock:rw', argv)

    @unittest.skipUnless(os.environ.get('CAPE_GPU_TIMING_LOCK_BIND_TEST') == '1',
                         'set CAPE_GPU_TIMING_LOCK_BIND_TEST=1 with functional Bubblewrap')
    def test_real_bubblewrap_sessions_share_bound_lock(self):
        self.check_same_gpu(sandbox=True)


if __name__ == '__main__':
    unittest.main()
