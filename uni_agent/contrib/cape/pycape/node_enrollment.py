# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import uuid
from pathlib import Path

from .credential_file import prepare_operation, read_registration, lock_registration
from .identity_nodes import resources


class EnrollmentUncertain(Exception):
    def __init__(self, path, operation_id):
        super().__init__("enrollment result unknown; retain the credential file and resume")
        self.path = path
        self.operation_id = operation_id


def validate_host_resources(declared):
    declared = resources(declared)
    cpus = len(os.sched_getaffinity(0))
    with open("/proc/meminfo") as stream:
        memory = next(int(line.split()[1]) for line in stream if line.startswith("MemTotal:")) / (1024 ** 2)
    # Container /proc/meminfo may describe the host rather than the job's limit.
    candidates = {Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")}
    try:
        with open("/proc/self/cgroup") as stream:
            for line in stream:
                _, controllers, relative = line.strip().split(":", 2)
                if controllers == "":
                    base, name = Path("/sys/fs/cgroup"), "memory.max"
                elif "memory" in controllers.split(","):
                    base, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
                else:
                    continue
                path = base / relative.lstrip("/")
                while path == base or base in path.parents:
                    candidates.add(path / name)
                    if path == base:
                        break
                    path = path.parent
    except (OSError, ValueError):
        pass
    for path in candidates:
        try:
            limit = int(path.read_text().strip())
            if limit > 0:
                memory = min(memory, limit / (1024 ** 3))
        except (OSError, ValueError):
            pass
    if declared["cpu_cores"] > cpus or declared["memory_gb"] > memory:
        raise ValueError("declared CPU/memory exceeds detected host capacity")
    gpus = declared["gpu_uuids"]
    if gpus:
        probe = subprocess.run(["nvidia-smi", "-i", ",".join(gpus), "--query-gpu=uuid", "--format=csv,noheader"],
                               check=True, capture_output=True, text=True, timeout=15)
        if sorted(line.strip() for line in probe.stdout.splitlines() if line.strip()) != gpus:
            raise ValueError("declared GPUs differ from detected physical devices")
    return declared


def enroll(args, request, url):
    declared = validate_host_resources({"cpu_cores": args.cpu_cores, "memory_gb": args.memory_gb, "gpu_uuids": args.gpu_uuid})
    actor = request("GET", url, "/v1/identity/me", token=args.token)
    pools = request("GET", url, "/v1/identity/pools", token=args.token)
    selected_pool = next((p for p in pools if args.pool in (p["pool_id"], p["name"])), None)
    if selected_pool is None:
        raise ValueError("unknown pool")
    pool = selected_pool["pool_id"]
    policy = selected_pool.get("config", {"pool_mode": "exclusive", "gpu_slots": 1, "placement": "pack"})
    args.pool_mode, args.gpu_slots, args.pool_placement = policy["pool_mode"], policy["gpu_slots"], policy["placement"]
    if args.resume:
        envelope = read_registration(args.node_token_out)
        body = envelope["registration"]
        if body.get("owner_user_id") != actor["user_id"]:
            raise ValueError("saved enrollment target identity mismatch or missing binding")
        if envelope["controller_url"] != url or body.get("resources") != declared or body.get("pool") != pool:
            raise ValueError("enrollment recovery differs from saved operation")
        if args.node_id and args.node_id != body.get("node_id"):
            raise ValueError("enrollment node differs from saved operation")
        if args.allocation_id and args.allocation_id != body.get("allocation_id"):
            raise ValueError("enrollment allocation differs from saved operation")
    else:
        envelope = prepare_operation(args.node_token_out, url, {
            "node_id": args.node_id or "node-" + uuid.uuid4().hex,
            "pool": pool, "agent_epoch": uuid.uuid4().hex,
            "allocation_id": args.allocation_id or "alloc-" + uuid.uuid4().hex, "resources": declared, "owner_user_id": actor["user_id"]})
        body = envelope["registration"]
    args._enrollment_lock = lock_registration(args.node_token_out)
    try:
        result = request("POST", url, "/v1/identity/nodes/enroll", payload=body, token=args.token)
        if result.get("operation_id") != body["operation_id"] or result.get("status") != "enrolled":
            raise ValueError("invalid acknowledgement")
    except Exception:
        raise EnrollmentUncertain(args.node_token_out, body["operation_id"]) from None
    args.node_id, args.pool, args.allocation_id = body["node_id"], body["pool"], body["allocation_id"]
    args.token, args.join_token = envelope["secret"], None
    return body["agent_epoch"]
