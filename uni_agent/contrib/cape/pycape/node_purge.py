# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Administrative retirement is not proof of remote process or image cleanup."""

import copy

from .identity import _admin, _audit
from .models import (NodeState, SessionState, CommandState, RequestState, AllocationState,
                     REQUEST_TERMINAL_STATES, SESSION_TERMINAL_STATES)


def purge(controller, token, node_id, reason):
    store = controller.auth.store
    store.require_admin(token)
    if not isinstance(reason, str) or not reason.strip() or len(reason.encode("utf-8")) > 1024:
        raise ValueError("purge requires a nonblank reason of at most 1024 UTF-8 bytes")
    with store._transaction() as state:
        actor = _admin(state, token, store.clock())
        intent = state.get("nodes", {}).get(node_id)
        if intent is None:
            raise ValueError("unknown node")
        if "purge" not in intent:
            node = controller.nodes.get(node_id)
            if node is None or node.state != NodeState.LOST:
                raise ValueError("node purge requires LOST state")
            intent["purge"] = {"node_id": node_id, "state": "REMOVED", "purged": True,
                "cleanup_confirmed": False, "reason": reason.strip(), "purged_at": store.clock(),
                "actor_user_id": actor["user_id"],
                "quarantined_gpu_uuids": sorted(set(node.resources.gpu_uuids)
                    | set(intent.get("effective_resources", intent["resources"])["gpu_uuids"]))}
            intent.update(desired="removed", force=False)
            intent.pop("pending_resources", None)
            _audit(state, actor["user_id"], "node.purge", node_id, store.clock())
        result = copy.deepcopy(intent["purge"])
    # Identity intent is durable first. Replay reconciles a crash before projection.
    from .managed_lifecycle import reconcile
    reconcile(controller)
    controller.journal.flush()
    return result


def release_quarantine(controller, token, node_id, reason, confirm_stopped):
    store = controller.auth.store
    store.require_admin(token)
    if confirm_stopped is not True:
        raise ValueError("quarantine release requires explicit confirm_stopped=true")
    if not isinstance(reason, str) or not reason.strip() or len(reason.encode("utf-8")) > 1024:
        raise ValueError("quarantine release requires a nonblank reason of at most 1024 UTF-8 bytes")
    with store._transaction() as state:
        actor = _admin(state, token, store.clock())
        intent = state.get("nodes", {}).get(node_id)
        if intent is None:
            raise ValueError("unknown node")
        if "purge" not in intent:
            raise ValueError("quarantine release requires a purged node")
        if "quarantine_release" not in intent:
            intent["quarantine_release"] = {"actor_user_id": actor["user_id"],
                "released_at": store.clock(), "reason": reason.strip(), "confirmed_stopped": True,
                "confirmation_source": "administrator"}
            _audit(state, actor["user_id"], "node.release-quarantine", node_id, store.clock())
        receipt = copy.deepcopy(intent["quarantine_release"])
        scope = set(intent["purge"]["quarantined_gpu_uuids"])
    from .managed_lifecycle import reconcile
    reconcile(controller)
    controller.journal.flush()
    return {"node_id": node_id, "state": "REMOVED", "released": True, "cleanup_confirmed": False,
            "release_receipt": receipt, "released_source_gpu_uuids": sorted(scope),
            "still_quarantined_gpu_uuids": sorted(scope & controller.node_purge_quarantine),
            "unquarantined_gpu_uuids": sorted(scope - controller.node_purge_quarantine)}


def project(controller, node_id, intent):
    reason = "node_purged"
    for request in controller.requests.values():
        if request.state in REQUEST_TERMINAL_STATES:
            continue
        workspace = controller.workspaces.get(request.spec.workspace["id"], {})
        if request.node_id != node_id and not (request.state == RequestState.QUEUED and workspace.get("node_id") == node_id):
            continue
        terminal = RequestState.FAILED if request.state == RequestState.QUEUED else RequestState.LOST
        controller._mark_request_terminal(request, terminal)
        request.reason = "workspace_node_unavailable" if terminal == RequestState.FAILED else reason
        request.failure_kind = "infrastructure"
        controller._emit("request_state", request.request_id, terminal.value, request.reason, controller._request_details(request))
    for command in controller.command_assignments.values():
        if command.node_id == node_id and command.state in (CommandState.PENDING, CommandState.RUNNING):
            command.state = CommandState.FAILED
            command.updated_at = controller.clock()
            command.result = {"reason": reason, "failure_kind": "infrastructure", "exit_code": 127}
            controller._emit("command_state", command.command_id, command.state.value, reason,
                             controller.command_assignment_details(command))
    for session in controller.sessions.values():
        if session.node_id == node_id and session.state not in SESSION_TERMINAL_STATES:
            controller._release_session(session.execution_session_id)
            session.state = SessionState.LOST
            session.active_request_id = None
            controller._emit("session_state", session.execution_session_id, session.state.value, reason,
                             controller._session_details(session))
    for record in controller.workspaces.values():
        if record["node_id"] == node_id and not record.get("node_retired", False):
            record.update(node_retired=True, cleanup_confirmed=False)
            controller._emit("workspace_state", record["id"], record["state"], reason, copy.deepcopy(record))
    node = controller.nodes.get(node_id)
    if node is not None and node.state != NodeState.REMOVED:
        node.state = NodeState.REMOVED
        controller._emit("node_state", node_id, node.state.value, reason, controller._node_details(node))
    allocation = controller.allocations.get(intent["allocation_id"])
    if allocation and allocation.state != AllocationState.REMOVED and all(
            n in controller.nodes and controller.nodes[n].state == NodeState.REMOVED for n in allocation.node_ids):
        allocation.state = AllocationState.REMOVED
        allocation.reason = reason
        allocation.updated_at = controller.clock()
        controller._emit("allocation_state", allocation.allocation_id, allocation.state.value, reason,
                         controller.allocation_status(allocation.allocation_id))
