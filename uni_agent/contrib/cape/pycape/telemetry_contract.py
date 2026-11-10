# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport-independent telemetry contracts; no GPU or network initialization."""

import copy
import math
import time
from dataclasses import dataclass


def identifier(value):
    if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256
            or any(ord(c) < 33 or ord(c) == 127 for c in value)):
        raise ValueError("invalid telemetry identifier")
    return value


def finite(value, minimum, maximum):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("invalid telemetry number")
    return value


@dataclass(frozen=True)
class CollectionConfig:
    enabled: bool = False
    basic: bool = True
    profiling: bool = False
    interval_seconds: float = 5.0
    stale_periods: int = 3

    def __post_init__(self):
        if any(type(value) is not bool for value in (self.enabled, self.basic, self.profiling)):
            raise ValueError("telemetry switches must be booleans")
        finite(self.interval_seconds, 1, 3600)
        object.__setattr__(self, "interval_seconds", float(self.interval_seconds))
        if type(self.stale_periods) is not int or not 1 <= self.stale_periods <= 120:
            raise ValueError("invalid telemetry stale periods")

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("unknown telemetry configuration field")
        return cls(**value)


class RegistrationSet:
    """Single-thread-owned registry scoped to one host and scheduler.

    Epoch ordering comes from the existing scheduler, not anonymous callers.
    register() is idempotent for one incarnation and refuses an implicit replacement.
    A confirmed replacement must explicitly retire the prior registration first.
    """

    def __init__(self, scheduler_id, host_id, ttl_seconds=30, capacity=4096, clock=time.monotonic):
        self.scheduler_id = identifier(scheduler_id)
        self.host_id = identifier(host_id)
        self.ttl_seconds = finite(ttl_seconds, 1, 3600)
        if type(capacity) is not int or not 1 <= capacity <= 65536:
            raise ValueError("invalid registration capacity")
        self.capacity = capacity
        self.clock = clock
        self._entries = {}

    def _now(self):
        return finite(self.clock(), 0, 1e20)

    def expire(self):
        now = self._now()
        expired = [key for key, entry in self._entries.items() if entry["deadline"] <= now]
        for key in expired:
            del self._entries[key]
        return len(expired)

    def register(self, scheduler_id, node_id, epoch, pool_id, gpu_uuids):
        if identifier(scheduler_id) != self.scheduler_id:
            raise ValueError("telemetry scheduler mismatch")
        node_id, epoch, pool_id = map(identifier, (node_id, epoch, pool_id))
        if not isinstance(gpu_uuids, list) or len(gpu_uuids) > 1024:
            raise ValueError("invalid telemetry GPU list")
        gpus = frozenset(identifier(gpu) for gpu in gpu_uuids)
        self.expire()
        previous = self._entries.get(node_id)
        if previous and previous["epoch"] != epoch:
            raise ValueError("telemetry registration incarnation conflict")
        if not previous and len(self._entries) >= self.capacity:
            raise ValueError("telemetry registration limit reached")
        self._entries[node_id] = dict(epoch=epoch, pool_id=pool_id, gpus=gpus,
                                      deadline=self._now() + self.ttl_seconds)

    def remove(self, node_id, epoch):
        self.expire()
        entry = self._entries.get(node_id)
        if not entry or entry["epoch"] != epoch:
            return False
        del self._entries[node_id]
        return True

    def gpu_uuids(self, pool_id=None, node_id=None):
        self.expire()
        return sorted({gpu for key, entry in self._entries.items()
                       if (node_id is None or key == node_id)
                       and (pool_id is None or entry["pool_id"] == pool_id)
                       for gpu in entry["gpus"]})


class LatestSnapshots:
    """Latest only; freshness uses receipt monotonic time, not wall-clock differences."""

    def __init__(self, capacity=1024, clock=time.monotonic):
        if type(capacity) is not int or not 1 <= capacity <= 65536:
            raise ValueError("invalid snapshot capacity")
        self.capacity = capacity
        self.clock = clock
        self._values = {}

    def retain(self, gpu_uuids):
        allowed = set(gpu_uuids)
        self._values = {gpu: value for gpu, value in self._values.items() if gpu in allowed}

    def put(self, gpu_uuid, sampled_at, values):
        gpu_uuid = identifier(gpu_uuid)
        finite(sampled_at, 0, 1e20)
        if not isinstance(values, dict) or len(values) > 64:
            raise ValueError("invalid telemetry values")
        normalized = {}
        for name, value in values.items():
            identifier(name)
            normalized[name] = None if value is None else finite(value, -1e20, 1e20)
        previous = self._values.get(gpu_uuid)
        if previous and sampled_at <= previous["sampled_at"]:
            return False  # Replayed data must not renew freshness.
        if not previous and len(self._values) >= self.capacity:
            raise ValueError("telemetry snapshot limit reached")
        self._values[gpu_uuid] = dict(sampled_at=sampled_at, values=normalized,
                                      received_at=finite(self.clock(), 0, 1e20))
        return True

    def get(self, gpu_uuid, stale_after):
        finite(stale_after, 0.001, 432000)
        entry = self._values.get(gpu_uuid)
        if entry is None:
            return {"gpu_uuid": gpu_uuid, "state": "unavailable", "sampled_at": None, "values": {}}
        age = finite(self.clock(), 0, 1e20) - entry["received_at"]
        stale = age < 0 or age >= stale_after
        return {"gpu_uuid": gpu_uuid, "state": "stale" if stale else "ready",
                "sampled_at": entry["sampled_at"],
                "values": {} if stale else copy.deepcopy(entry["values"])}
