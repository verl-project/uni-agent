# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure maintenance policy and deterministic inode-aware target calculation."""

from .workspace import MIN_IMAGE_SIZE_BYTES

class MaintenancePolicy:
    DEFAULTS = {
        "auto_grow": False, "grow_threshold_percent": 80, "grow_target_percent": 60,
        "grow_min_bytes": 1024 ** 3, "grow_cooldown_seconds": 60,
        "reclaim_on_release": False, "reclaim_min_allocated_bytes": 64 * 1024 ** 2,
        "reclaim_cooldown_seconds": 300, "timeout_seconds": 300,
        "reclaim_idle_seconds": 60, "reclaim_start_interval_seconds": 30,
        "reclaim_max_candidates": 32, "reclaim_retry_backoff_seconds": 300,
        "reclaim_low_yield_backoff_seconds": 3600, "reclaim_nice": 10,
        "backing_reserve_bytes": 512 * 1024 ** 2,
    }

    def __init__(self, values=None):
        values = {} if values is None else values
        if not isinstance(values, dict) or set(values) - set(self.DEFAULTS):
            raise ValueError("invalid workspace maintenance policy fields")
        self.values = dict(self.DEFAULTS, **values)
        for key, value in self.values.items():
            if type(self.DEFAULTS[key]) is bool:
                if type(value) is not bool:
                    raise ValueError(key + " must be boolean")
            elif type(value) is not int or not 0 <= value < 2 ** 63:
                raise ValueError(key + " must be a nonnegative bounded integer")
        if not 1 <= self.values["grow_target_percent"] < self.values["grow_threshold_percent"] < 100:
            raise ValueError("growth target must be below the utilization threshold, both within 1..99")
        if self.values["grow_min_bytes"] < MIN_IMAGE_SIZE_BYTES or not 1 <= self.values["timeout_seconds"] <= 3600:
            raise ValueError("invalid maintenance growth increment or timeout")
        if not 1 <= self.values["reclaim_max_candidates"] <= 1024 or not 0 <= self.values["reclaim_nice"] <= 19:
            raise ValueError("invalid reclaim candidate limit or priority")

    def to_dict(self):
        return dict(self.values)

def automatic_grow_target(record, config, now):
    policy = MaintenancePolicy(config.maintenance).to_dict()
    if not policy["auto_grow"] or record.get("state") != "mounted" or record.get("operation", {}).get("state") == "pending":
        return None
    if now - record.get("last_maintenance_at", -1e30) < policy["grow_cooldown_seconds"]:
        return None
    raw_space = record.get("space")
    if raw_space is None:
        return None
    if not isinstance(raw_space, dict):
        raise ValueError("invalid capacity telemetry")
    space = raw_space.get("internal")
    if space is None:
        return None
    if not isinstance(space, dict) or any(type(space.get(key)) is not int or not 0 <= space[key] < 2 ** 63
                                          for key in ("total_bytes", "available_bytes", "total_inodes", "free_inodes")):
        raise ValueError("invalid capacity telemetry")
    total, free = space.get("total_bytes", 0), space.get("available_bytes", 0)
    inodes, free_inodes = space.get("total_inodes", 0), space.get("free_inodes", 0)
    if free > total or free_inodes > inodes:
        raise ValueError("invalid capacity telemetry")
    threshold, desired = policy["grow_threshold_percent"], policy["grow_target_percent"]
    byte_pressure = total > 0 and (total - free) * 100 >= total * threshold
    inode_pressure = inodes > 0 and (inodes - free_inodes) * 100 >= inodes * threshold
    if not byte_pressure and not inode_pressure:
        return None
    current = record.get("current_image_size_bytes", record["config"]["image_size_bytes"])
    target = current + policy["grow_min_bytes"]
    if byte_pressure:
        target = max(target, ((total - free) * 100 + desired - 1) // desired + current - total)
    if inode_pressure:
        geometry = record.get("geometry") or {}
        if any(type(geometry.get(key)) is not int or geometry[key] <= 0 for key in ("blocks_per_group", "block_size")):
            raise ValueError("inode-aware growth requires recorded filesystem geometry")
        group_bytes = geometry.get("blocks_per_group", 0) * geometry.get("block_size", 0)
        if not group_bytes:
            raise ValueError("inode-aware growth requires recorded filesystem geometry")
        groups = (current + group_bytes - 1) // group_bytes
        per_group = inodes // groups
        if not per_group:
            raise ValueError("invalid inode geometry")
        wanted_inodes = ((inodes - free_inodes) * 100 + desired - 1) // desired
        target = max(target, max(groups + 1, (wanted_inodes + per_group - 1) // per_group) * group_bytes)
    unit = 1024 ** 2
    target = ((target + unit - 1) // unit) * unit
    if target > config.max_image_size_bytes:
        raise ValueError("automatic growth exceeds max_image_size_bytes")
    return {"target_bytes": target, "reason": "inodes" if inode_pressure else "bytes", "policy": policy}
