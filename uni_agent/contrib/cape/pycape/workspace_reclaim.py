# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap reclaim admission signals, separate from exclusive filesystem work."""

MAX_COUNTER = 2 ** 63 - 1


def maintenance_host_key():
    import hashlib
    from pathlib import Path
    import uuid
    try:
        boot = str(uuid.UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
        return hashlib.sha256(boot.encode("ascii")).hexdigest()
    except (OSError, ValueError):
        return None


def _counter(value):
    return type(value) is int and 0 <= value <= MAX_COUNTER


def reclaim_sample(allocated_bytes, geometry, image_generation, usage_generation):
    fields = ("block_size", "blocks", "free_blocks", "size_bytes")
    if not isinstance(geometry, dict) or not all(_counter(geometry.get(k)) for k in fields):
        raise ValueError("invalid reclaim geometry")
    if (geometry["block_size"] == 0 or geometry["blocks"] == 0
            or geometry["free_blocks"] > geometry["blocks"]
            or geometry["blocks"] * geometry["block_size"] != geometry["size_bytes"]):
        raise ValueError("inconsistent reclaim geometry")
    if not all(_counter(v) for v in (allocated_bytes, image_generation, usage_generation)):
        raise ValueError("invalid reclaim sample counter")
    return {"allocated_bytes": allocated_bytes,
            "used_bytes": (geometry["blocks"] - geometry["free_blocks"]) * geometry["block_size"],
            "logical_bytes": geometry["size_bytes"], "image_generation": image_generation,
            "usage_generation": usage_generation}


def reclaim_signal(record):
    reference, sample = record.get("reclaim_reference"), record.get("reclaim_sample")
    if reference is None:
        return None, "reference_missing"
    fields = ("allocated_bytes", "used_bytes", "logical_bytes", "image_generation", "usage_generation")
    for value in (reference, sample):
        if (not isinstance(value, dict) or not all(_counter(value.get(k)) for k in fields)
                or value["logical_bytes"] == 0 or value["used_bytes"] > value["logical_bytes"]):
            return None, "invalid_sample"
    usage, generation = record.get("usage_generation"), record.get("generation", 0)
    logical = record.get("current_image_size_bytes", (record.get("config") or {}).get("image_size_bytes"))
    if (not _counter(usage) or usage == 0 or not _counter(generation)
            or sample["usage_generation"] != usage or reference["usage_generation"] > usage
            or reference["image_generation"] != generation or sample["image_generation"] != generation
            or reference["logical_bytes"] != logical or sample["logical_bytes"] != logical):
        return None, "stale_sample"
    # A screening signal, not a block-map measurement of reclaimable storage.
    delta = ((sample["allocated_bytes"] - reference["allocated_bytes"])
             - (sample["used_bytes"] - reference["used_bytes"]))
    return min(sample["allocated_bytes"], max(0, delta)), None


def reclaim_candidate(record, policy, now):
    if record.get("auto_reclaim", "inherit") != "inherit":
        return None, "workspace_disabled"
    if not policy["reclaim_on_release"]:
        return None, "disabled"
    if record.get("state") != "retained" or record.get("maintenance_deferred"):
        return None, "not_retained"
    if record.get("operation", {}).get("state") == "pending":
        return None, "operation_pending"
    signal, error = reclaim_signal(record)
    if error:
        return None, error
    usage = record["usage_generation"]
    if record.get("reclaim_success_generation") == usage:
        return None, "generation_completed"
    if signal <= 0 or signal < policy["reclaim_min_allocated_bytes"]:
        return None, "below_relative_threshold"
    released_at = record.get("reclaim_release_observed_at", now)
    last_success = record.get("reclaim_last_success_at", -1e30)
    not_before = max(released_at + policy["reclaim_idle_seconds"],
                     last_success + policy["reclaim_cooldown_seconds"],
                     record.get("reclaim_retry_after", -1e30),
                     record.get("reclaim_low_yield_until", -1e30))
    return {"usage_generation": usage, "estimated_bytes": signal,
            "first_seen_at": released_at, "not_before": not_before}, (
                "ready" if now >= not_before else "waiting_idle_or_backoff")
