# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission guards for retained layers; never reclaim data to make room."""

import os
import stat
from .credential_file import _private_parent


def check_storage(root, min_free_bytes=512 * 1024 ** 2, max_bytes=0):
    if min_free_bytes < 0 or max_bytes < 0:
        raise ValueError("execution storage limits must be nonnegative")
    os.makedirs(root, mode=0o700, exist_ok=True)
    with _private_parent(os.path.join(root, ".state-owner")):
        pass
    free = os.statvfs(root)
    if free.f_bavail * free.f_frsize < min_free_bytes:
        raise RuntimeError("execution state storage reserve exhausted; existing state preserved")
    if max_bytes:
        device = os.lstat(root).st_dev
        pending, seen, total = [root], set(), 0
        while pending:
            path = pending.pop()
            info = os.lstat(path)
            if info.st_dev != device:
                continue  # Mounted merged views are not retained backing allocation.
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += info.st_blocks * 512
            if total >= max_bytes:
                raise RuntimeError("execution state storage quota exhausted; explicit purge required")
            if stat.S_ISDIR(info.st_mode):
                with os.scandir(path) as entries:
                    pending.extend(entry.path for entry in entries)
