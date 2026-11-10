# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU cooperative-lock example, not CAPE policy or GPU exclusivity.

The deployment must provision a common node-local directory and compatible file
permissions. Never unlink live lock files: all holders must lock the same inode.
Multiple GPU workloads need a separate ordered multi-lock protocol.
"""
import fcntl
import os
from pathlib import Path
import re


def lock_name(visibility):
    value = (visibility or '').strip()
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value):
        raise ValueError('timing lock requires exactly one full GPU UUID; no numeric/empty fallback')
    return value + '.lock'


def acquire(directory, visibility, on_blocked=lambda: None):
    """Return a locked descriptor; caller closes it to release (also on process exit)."""
    path = Path(directory) / lock_name(visibility)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o660)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            on_blocked()
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise
