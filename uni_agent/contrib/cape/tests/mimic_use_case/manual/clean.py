#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover an orphan controller and preflight a bounded manual test reset."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import sys
import time


def inside(path, root):
    return path == root or root in path.parents


def processes():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] != 'Z':
                yield proc, args, stat[19]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def controller_state(args):
    for i, arg in enumerate(args):
        if Path(arg).name not in ('rscape', 'pycape'):
            continue
        if args[i + 1:i + 3] != ['controller', 'start']:
            continue
        for j in range(i + 3, len(args)):
            if args[j].startswith('--state-dir='):
                return args[j].split('=', 1)[1]
            if args[j] == '--state-dir' and j + 1 < len(args):
                return args[j + 1]
    return None


def recover_controller(state):
    for proc, args, start in processes():
        value = controller_state(args)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = proc / 'cwd' / path
        if path.resolve() != state:
            continue
        # pidfd pins this process, never a reused numeric PID or its whole group.
        fd = os.pidfd_open(int(proc.name))
        try:
            current = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            if current != start:
                raise RuntimeError('controller identity changed; retry cleanup')
            print(f'[clean] SIGTERM matching controller pid={proc.name}', flush=True)
            signal.pidfd_send_signal(fd, signal.SIGTERM)
            import select
            if not select.select([fd], [], [], 30)[0]:
                raise RuntimeError(f'controller pid={proc.name} did not exit; files preserved')
        finally:
            os.close(fd)


def open_holders(targets):
    holders = set()
    for proc, _, _ in processes():
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    value = os.readlink(fd).removesuffix(' (deleted)')
                    if value.startswith('/') and any(inside(Path(value), p) for p in targets):
                        holders.add(int(proc.name))
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            raise RuntimeError(f'cannot inspect same-user pid={proc.name}; files preserved')
    return sorted(holders)


def clean(root, logs, reports, bundle, state):
    if int(os.readlink('/proc/self')) != os.getpid():
        raise RuntimeError('/proc does not match this PID namespace; run clean on the owning node')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('cleanup requires Python 3.9+ with Linux pidfd support')
    root = root.resolve()
    if root == Path('/') or root == Path.home() or not (root / 'Makefile').is_file():
        raise RuntimeError('not a bounded manual test directory')
    targets = [p.resolve() for p in (logs, reports, bundle)]
    for target in targets:
        if target == root or not inside(target, root):
            raise RuntimeError(f'refusing cleanup outside a child of {root}: {target}')
    if len(set(targets)) != len(targets) or any(inside(a, b) for a in targets for b in targets if a != b):
        raise RuntimeError('cleanup targets must be distinct, non-overlapping directories')
    state = state.resolve()
    if not inside(state, targets[2]):
        raise RuntimeError('controller state is outside the test bundle; files preserved')
    for proc, args, _ in processes():
        if not any(Path(arg).name in ('rscape', 'pycape') and args[i + 1:i + 3] == ['node', 'enroll']
                   for i, arg in enumerate(args)):
            continue
        if any(arg.startswith('/') and any(inside(Path(arg).resolve(), target) for target in targets)
               for arg in args):
            raise RuntimeError(f'node pid={proc.name} still uses this test directory; '
                               'drain/exit that node before cleaning; files preserved')
    recover_controller(state)
    targets.extend(p.resolve() for p in root.glob('*.out'))
    if any(target == root or not inside(target, root) for target in targets):
        raise RuntimeError('output symlink escapes the test directory; files preserved')
    for row in Path('/proc/self/mountinfo').read_text().splitlines():
        value = row.split()[4]
        for escaped, decoded in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
            value = value.replace(escaped, decoded)
        if any(inside(Path(value), target) for target in targets):
            raise RuntimeError(f'mount remains at {value}; files preserved')
    for _ in range(30):
        holders = open_holders(targets)
        nfs = [p for target in targets if target.is_dir() for p in target.rglob('.nfs*')]
        if not holders and not nfs:
            break
        time.sleep(.1)
    if holders or nfs:
        raise RuntimeError(f'open files remain (local pids={holders}, NFS files={len(nfs)}); '
                           'close their holders on the owning host; files preserved')
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    print('[clean] removed test logs, reports and state/credentials; unrelated processes preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'logs', 'reports', 'bundle', 'state'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        clean(args.root, args.logs, args.reports, args.bundle, args.state)
    except (OSError, RuntimeError) as exc:
        print(f'[clean] {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
