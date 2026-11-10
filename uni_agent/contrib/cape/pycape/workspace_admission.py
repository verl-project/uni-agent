# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-owned maintenance admission; candidates never acquire image fences."""

import copy
import uuid

from .workspace_policy import MaintenancePolicy
from .models import NodeState, REQUEST_TERMINAL_STATES


def host_key(controller, node_id):
    node = controller.nodes.get(node_id)
    value = node.workspace_status.get("maintenance_host_key") if node else None
    return value if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value) else None


def overlaps(left, right):
    return left is None or right is None or left == right


def granted_host(controller, record):
    operation = record.get("operation", {})
    node = controller.nodes.get(record["node_id"])
    if node is None or operation.get("grant_epoch") != getattr(node, "agent_epoch", None):
        return None
    return operation.get("grant_host_key")


def dispatch(controller, node_id, ready=False):
    node = controller.nodes[node_id]
    if node.state == NodeState.REMOVED:
        return []
    key = host_key(controller, node_id)
    now = controller.clock()
    capable = node.workspace_status.get("maintenance_protocol") == 1
    pending = []
    for record in controller.workspaces.values():
        operation = record.get("operation", {})
        if operation.get("state") == "pending":
            pending.append(record)
    # A missing flag is an older/uncertain grant, not permission to run concurrently.
    granted = [r for r in pending if r["operation"].get("dispatch_granted", True)]
    own = [r for r in granted if r["node_id"] == node_id]
    if own:
        return copy.deepcopy(own[:1])
    if any(overlaps(key, granted_host(controller, r)) for r in granted):
        return []
    eligible = []
    for record in pending:
        owner = controller.nodes.get(record["node_id"])
        if owner is None or owner.state not in (NodeState.ACTIVE, NodeState.DRAINING):
            continue
        legacy_safe = record["operation"]["action"] in ("check", "delete") and not record.get("maintenance")
        if owner.workspace_status.get("maintenance_protocol") != 1 and not legacy_safe:
            continue
        session = controller.sessions.get(record.get("execution_session_id"))
        if session is not None and session.active_request_id:
            continue
        if not overlaps(key, host_key(controller, record["node_id"])):
            continue
        eligible.append(record)
    eligible.sort(key=lambda r: (r["operation"].get("requested_at", 0), r["id"]))
    selected = eligible[0] if eligible else None
    if selected is not None:
        legacy_safe = selected["operation"]["action"] in ("check", "delete") and not selected.get("maintenance")
        if selected["node_id"] != node_id or not (capable or legacy_safe):
            return []
    else:
        if not ready or not capable or node.state != NodeState.ACTIVE:
            return []
        pools = {n.pool for nid, n in controller.nodes.items()
                 if n.state not in (NodeState.REMOVED, NodeState.LOST) and overlaps(key, host_key(controller, nid))}
        if any(r.spec.pool in pools and r.state not in REQUEST_TERMINAL_STATES for r in controller.requests.values()):
            return []
        candidates = []
        interval = 0
        window = 1024
        for record in controller.workspaces.values():
            if not overlaps(key, host_key(controller, record["node_id"])):
                continue
            owner = controller.nodes.get(record["node_id"])
            if owner is None or owner.state != NodeState.ACTIVE or owner.workspace_status.get("maintenance_protocol") != 1:
                continue
            controller._automatic_workspace_operation(record, owner)
            policy = MaintenancePolicy(controller.workspace_configs.get(record["pool"], record["node_id"]).maintenance).to_dict()
            config = controller.workspace_configs.get(record["pool"], record["node_id"])
            if owner.workspace_status.get("state") != "applied" or owner.workspace_status.get("applied_version") != config.version:
                continue
            interval = max(interval, policy["reclaim_start_interval_seconds"])
            window = min(window, policy["reclaim_max_candidates"])
            candidate = record.get("reclaim_candidate")
            session = controller.sessions.get(record.get("execution_session_id"))
            if candidate and candidate["not_before"] <= now and not (session and session.active_request_id):
                candidates.append(record)
                candidates.sort(key=lambda r: (r["reclaim_candidate"]["first_seen_at"], r["id"]))
            del candidates[window:]
        if any(overlaps(key, r.get("reclaim_attempt_host_key")) and
               now < r.get("reclaim_last_attempt_at", -1e30) + interval for r in controller.workspaces.values()):
            return []
        candidates.sort(key=lambda r: (r["reclaim_candidate"]["first_seen_at"], r["id"]))
        candidates = candidates[:window]
        if not candidates or candidates[0]["node_id"] != node_id:
            return []
        selected = candidates[0]
        config = controller.workspace_configs.get(selected["pool"], node_id)
        policy = MaintenancePolicy(config.maintenance).to_dict()
        selected["operation"] = {"id": uuid.uuid4().hex, "action": "reclaim", "state": "pending",
            "sequence": controller._advance_workspace_operation(selected), "source": "automatic",
            "boundary": False, "requested_at": now, "phase": "requested", "target_bytes": None,
            "policy": policy, "policy_version": config.version, "max_image_size_bytes": config.max_image_size_bytes,
            "usage_generation": selected["usage_generation"]}
        selected.update(reclaim_last_attempt_at=now, reclaim_attempt_host_key=key, reclaim_candidate=None,
                        reclaim_status="granted")
    selected["operation"].update(dispatch_granted=True, grant_host_key=key,
                                 grant_epoch=getattr(node, "agent_epoch", None))
    controller._emit("workspace_state", selected["id"], selected["state"], "maintenance_granted", copy.deepcopy(selected))
    return [copy.deepcopy(selected)]


def complete(record, now):
    operation = record.get("operation", {})
    if operation.get("action") != "reclaim":
        return
    policy = MaintenancePolicy(operation.get("policy")).to_dict()
    if operation.get("state") == "completed":
        record.update(reclaim_last_success_at=now, reclaim_success_generation=record.get("usage_generation"))
        reclaimed = operation.get("reclaimed_bytes", 0)
        if not isinstance(reclaimed, int) or reclaimed < policy["reclaim_min_allocated_bytes"]:
            record["reclaim_low_yield_until"] = now + policy["reclaim_low_yield_backoff_seconds"]
    elif operation.get("state") == "failed":
        record["reclaim_retry_after"] = now + policy["reclaim_retry_backoff_seconds"]
