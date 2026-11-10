# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist intent before drain; retire only after fresh agent cleanup proof."""

from .identity import _authenticate, _admin, _name, _audit
from .identity_nodes import owned_node, authenticate_agent, resources
from .models import NodeState, NodeResources, SessionState, CommandState, RequestState, AllocationState, PoolConfig, REQUEST_TERMINAL_STATES


def _clean(controller, node_id, after=0):
    node = controller.nodes.get(node_id)
    if node is None or node.active_sessions:
        return False
    assignments = [c for c in controller.command_assignments.values() if c.node_id == node_id]
    if any(c.state in (CommandState.PENDING, CommandState.RUNNING) for c in assignments):
        return False
    latest = max((c.updated_at or c.created_at for c in assignments), default=0)
    status = getattr(node, "last_status", {})
    received = controller._managed_seen.get(node_id, 0)
    return (received > max(after, latest) and controller.clock() - received < controller.policy.heartbeat_timeout_seconds
            and status.get("authoritative") is True and status.get("sessions") == []
            and not status.get("running_commands") and not status.get("command_ids"))


def node_action(controller, token, node_id, action, force=False, allow_agent=False):
    if action not in ("drain", "exit") or type(force) is not bool or (action == "exit" and force):
        raise ValueError("invalid node action")
    store = controller.auth.store
    try:
        owned_node(store, token, node_id)
        agent = False
    except PermissionError:
        if not allow_agent or force:
            raise
        authenticate_agent(store, token, node_id=node_id)
        agent = True
    record = store.snapshot()["nodes"][node_id]
    node = controller.nodes.get(node_id)
    if record["desired"] == "removed":
        return {"node_id": node_id, "state": "REMOVED"}
    if action == "exit" and (node is not None or record.get("announced", False)) and not _clean(controller, node_id):
        raise ValueError("node is busy or cleanup is unconfirmed; use drain")
    with store._transaction() as state:
        if not agent:
            actor = _authenticate(state, token, store.clock())
            if not actor["is_admin"] and state["nodes"][node_id]["owner_user_id"] != actor["user_id"]:
                raise PermissionError("node not accessible")
        current = state["nodes"][node_id]
        if current["desired"] != "draining" or (force and not current.get("force")):
            current.update(desired="draining", force=force, intent_at=controller.clock())
            current.pop("pending_resources", None)
        if action == "exit":
            # Already-known fresh emptiness is sufficient for serialized exit;
            # drain instead requires a heartbeat after its intent.
            current["intent_at"] = 0
        _audit(state, "node:" + node_id if agent else actor["user_id"], "node." + action + (".force" if force else ""), node_id, controller.clock())
    reconcile(controller)
    node = controller.nodes.get(node_id)
    return {"node_id": node_id, "state": node.state.value if node else store.snapshot()["nodes"][node_id]["desired"].upper()}


def edit_node(controller, token, node_id, changes):
    store = controller.auth.store
    record = owned_node(store, token, node_id)
    if not isinstance(changes, dict) or set(changes) - {"cpu_cores", "memory_gb", "gpu_uuids"}:
        raise ValueError("invalid node changes")
    if record["desired"] not in ("active", "reconfigure"):
        raise ValueError("retiring nodes cannot be edited")
    old = record.get("effective_resources", record["resources"])
    changed = resources(dict(old, **changes))
    if changed["cpu_cores"] > old["cpu_cores"] or changed["memory_gb"] > old["memory_gb"] or not set(changed["gpu_uuids"]).issubset(old["gpu_uuids"]):
        raise ValueError("node edit supports reductions only; enroll a new contribution to add resources")
    if record["desired"] == "reconfigure" and record["pending_resources"] != changed:
        raise ValueError("node reconfiguration is already pending")
    if record["desired"] == "active" and changed == old:
        node = controller.nodes.get(node_id)
        return {"node_id": node_id, "state": node.state.value if node else "enrolled", "desired": "active"}
    with store._transaction() as state:
        actor = _authenticate(state, token, store.clock())
        current = state["nodes"][node_id]
        if not actor["is_admin"] and current["owner_user_id"] != actor["user_id"]:
            raise PermissionError("node not accessible")
        if current["desired"] != "reconfigure":
            current.update(desired="reconfigure", pending_resources=changed, force=False, intent_at=controller.clock())
        _audit(state, actor["user_id"], "node.edit", node_id, controller.clock())
    reconcile(controller)
    node = controller.nodes.get(node_id)
    return {"node_id": node_id, "state": node.state.value if node else "enrolled", "desired": store.snapshot()["nodes"][node_id]["desired"]}


def pool_action(controller, token, pool_id, action, force=False):
    store = controller.auth.store
    store.require_admin(token)
    if action not in ("drain", "remove") or type(force) is not bool or (action == "remove" and force):
        raise ValueError("invalid pool action")
    state = store.snapshot()
    if pool_id not in state["pools"]:
        raise ValueError("unknown pool")
    active = [n for n in state.get("nodes", {}).values() if n["pool"] == pool_id and n["desired"] != "removed"]
    if action == "remove" and active:
        raise ValueError("pool is not empty; drain it before removal")
    with store._transaction() as state:
        actor = _admin(state, token, store.clock())
        state["pools"][pool_id]["draining"] = True
        for node in state.get("nodes", {}).values():
            if node["pool"] == pool_id and node["desired"] != "removed":
                if node["desired"] != "draining" or (force and not node.get("force")):
                    node.update(desired="draining", force=force, intent_at=controller.clock())
                    node.pop("pending_resources", None)
        _audit(state, actor["user_id"], "pool." + action + (".force" if force else ""), pool_id, controller.clock())
    reconcile(controller)
    if action == "remove":
        controller.journal.flush()
        with store._transaction() as state:
            _admin(state, token, store.clock())
            del state["pools"][pool_id]
        return {"pool_id": pool_id, "state": "removed"}
    return {"pool_id": pool_id, "state": "draining" if store.snapshot()["pools"][pool_id]["draining"] else "unavailable"}


def edit_pool(controller, token, pool_id, changes):
    store = controller.auth.store
    store.require_admin(token)
    if not isinstance(changes, dict) or set(changes) - {"name", "config", "max_nodes"}:
        raise ValueError("invalid pool changes")
    state = store.snapshot()
    current = state["pools"][pool_id]
    if current["draining"]:
        raise ValueError("cannot edit a draining pool")
    changed_config = changes.get("config", current.get("config", {}))
    if not isinstance(changed_config, dict) or set(changed_config) - {"pool_mode", "gpu_slots", "placement"}:
        raise ValueError("invalid pool configuration")
    config = PoolConfig(**changed_config).to_dict()
    if config["gpu_slots"] > 2 ** 63 - 1:
        raise ValueError("GPU slot count is too large")
    active = sum(n["pool"] == pool_id and n["desired"] != "removed" for n in state.get("nodes", {}).values())
    if config != current.get("config", PoolConfig().to_dict()) and (active or controller._pool_config_has_blockers(controller._pool_config_blockers(pool_id))):
        raise ValueError("pool configuration requires an empty pool")
    limit = changes.get("max_nodes", current.get("max_nodes"))
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 2 ** 63 - 1 or limit < active):
        raise ValueError("invalid node limit or limit below enrolled capacity")
    name = _name(changes.get("name", current["name"]))
    if any(key != pool_id and p["name"] == name for key, p in state["pools"].items()):
        raise ValueError("pool name already exists")
    with store._transaction() as state:
        actor = _admin(state, token, store.clock())
        state["pools"][pool_id].update(name=name, config=config, max_nodes=limit)
        _audit(state, actor["user_id"], "pool.edit", pool_id, controller.clock())
    controller.pool_configs[pool_id] = PoolConfig(**config)
    controller._pool_config_requires_matching_join[pool_id] = True
    return dict(pool_id=pool_id, name=name, config=config, max_nodes=limit)


def reconcile(controller):
    if not getattr(controller.auth, "managed", False):
        return
    store = controller.auth.store
    state = store.snapshot()
    for pool, config in state["pools"].items():
        controller.pool_configs[pool] = PoolConfig(**config.get("config", {}))
        controller._pool_config_requires_matching_join[pool] = True
    retired = []
    reconfigured = []
    controller.node_purge_quarantine = {gpu for node in state.get("nodes", {}).values()
        if "quarantine_release" not in node
        for gpu in node.get("purge", {}).get("quarantined_gpu_uuids", [])}
    for request in list(controller.requests.values()):
        if request.state == RequestState.QUEUED and state["pools"].get(request.spec.pool, {}).get("draining"):
            request.reason = "pool_draining"
            request.failure_kind = "workload"
            controller._mark_request_terminal(request, RequestState.FAILED)
            controller._emit("request_state", request.request_id, request.state.value, request.reason, controller._request_details(request))
    for node_id, intent in state.get("nodes", {}).items():
        if "purge" in intent:
            if node_id not in controller._projected_node_purges:
                from .node_purge import project
                project(controller, node_id, intent)
                controller._projected_node_purges.add(node_id)
            continue
        if intent["desired"] == "active":
            continue
        node = controller.nodes.get(node_id)
        if node is None:
            if intent["desired"] == "reconfigure" and not intent.get("announced", False):
                reconfigured.append(node_id)
                continue
            if intent["desired"] != "removed" and not intent.get("announced", False):
                retired.append(node_id)
            continue
        if node.state == NodeState.REMOVED:
            if intent["desired"] != "removed":
                retired.append(node_id)
            continue
        if node.state not in (NodeState.DRAINING, NodeState.LOST):
            controller._transition_node_to_draining(node, "managed_drain")
        for execution_session_id in list(node.active_sessions):
            session = controller.sessions.get(execution_session_id)
            if (session is not None and session.state in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE)
                    and not any(command.execution_session_id == execution_session_id and command.state in (CommandState.PENDING, CommandState.RUNNING)
                                for command in controller.command_assignments.values())):
                controller.stop_session(execution_session_id, reason="managed_drain", enforce_auth=False)
        if intent.get("force"):
            for request in list(controller.requests.values()):
                if request.node_id == node_id and request.state not in REQUEST_TERMINAL_STATES:
                    controller.cancel_request(request.request_id, reason="cancelled", enforce_auth=False)
        for request in list(controller.requests.values()):
            workspace = controller.workspaces.get(request.spec.workspace["id"])
            if intent["desired"] != "reconfigure" and request.state == RequestState.QUEUED and workspace and workspace["node_id"] == node_id:
                request.reason = "workspace_node_unavailable"
                request.failure_kind = "infrastructure"
                controller._mark_request_terminal(request, RequestState.FAILED)
                controller._emit("request_state", request.request_id, request.state.value, request.reason, controller._request_details(request))
        if intent["desired"] == "removed" or _clean(controller, node_id, intent["intent_at"]):
            if intent["desired"] == "reconfigure":
                node.resources = NodeResources(**intent["pending_resources"])
                node.state = NodeState.ACTIVE
                controller._emit("node_state", node_id, node.state.value, "managed_reconfigured", controller._node_details(node))
                reconfigured.append(node_id)
                continue
            node.state = NodeState.REMOVED
            controller._emit("node_state", node_id, node.state.value, "managed_drain_complete", controller._node_details(node))
            retired.append(node_id)
            allocation = controller.allocations.get(node.allocation_id)
            if allocation and all(controller.nodes[n].state == NodeState.REMOVED for n in allocation.node_ids):
                allocation.state = AllocationState.REMOVED
                allocation.reason = "allocation_removed"
                allocation.updated_at = controller.clock()
                controller._emit("allocation_state", allocation.allocation_id, allocation.state.value,
                                 allocation.reason, controller.allocation_status(allocation.allocation_id))
    if retired:
        # Never commit retirement proof before the matching request/node events.
        # Flush outside the authority transaction: both writers share one DB.
        controller.journal.flush()
        with store._transaction() as current:
            for node_id in retired:
                current["nodes"][node_id]["desired"] = "removed"
                _audit(current, "controller", "node.retired", node_id, controller.clock())
    if reconfigured:
        controller.journal.flush()
        with store._transaction() as current:
            for node_id in reconfigured:
                node = current["nodes"][node_id]
                node.update(desired="active", effective_resources=node.pop("pending_resources"))
                _audit(current, "controller", "node.reconfigured", node_id, controller.clock())
    state = store.snapshot()
    complete = [pool for pool, config in state["pools"].items() if config["draining"] and not any(
        n["pool"] == pool and n["desired"] != "removed" for n in state.get("nodes", {}).values())]
    if complete:
        controller.journal.flush()
        with store._transaction() as current:
            for pool in complete:
                current["pools"][pool]["draining"] = False
                _audit(current, "controller", "pool.drained", pool, controller.clock())
