# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Website projection of the native persistent authority; no second store."""
import copy
import uuid
from .identity import new_secret, verifier, _authenticate, _audit, _name

FEATURES = ["identity", "groups", "sessions", "sso-sessions", "resources", "node-enrollment", "node-controls"]


def capabilities(controller):
    return {"management_api": 2, "managed": getattr(controller.auth, "managed", False) is True,
            "features": FEATURES, "identity_model": "persistent-groups"}


def public_user(state, uid, admin=False):
    user = copy.deepcopy(state["users"][uid])
    user.update(id=uid, platform_admin=admin)
    user["grants"] = {pid: ["submit"] for pid in state["pools"] if admin or any(
        state["groups"][g]["all_pools"] or pid in state["groups"][g]["pools"] for g in user["groups"])}
    return user


def public_token(key, value):
    return dict(id=key, **{k: v for k, v in value.items() if k != "verifier"})


def inventory(controller, state):
    pools = []
    for pid, pool in state["pools"].items():
        summary = controller.pool_utilization(pid)
        for node in summary.get("node_states", []):
            node.pop("active_sessions", None)
        nodes = []
        for key, declared in state.get("nodes", {}).items():
            if declared["pool"] != pid or declared["desired"] == "removed":
                continue
            node = controller.nodes.get(key)
            resources = declared.get("effective_resources", declared["resources"])
            nodes.append(dict(id=key, state=node.state.value if node else "enrolled", **resources))
        pools.append(dict(id=pid, **pool, utilization=summary, nodes=nodes))
    return {"pools": pools, "timestamp": controller.clock()}


def call(controller, action, token, body):
    if not isinstance(body, dict):
        raise ValueError("management body must be an object")
    if getattr(controller.auth, "managed", False) is not True:
        raise ValueError("persistent identity is not configured")
    store = controller.auth.store
    if action.startswith("sso.") or action.startswith("session."):
        from .management_sso import call as sso_call
        return sso_call(store, action, token, body)
    actor = store.authenticate(token)
    state = store.snapshot()
    uid = actor["user_id"]
    if action == "me":
        credential = {"id": actor["token_id"], "kind": actor["token_kind"]}
        if actor["token_kind"] == "session":
            credential["auth_method"] = "sso"
        return {"user": public_user(state, uid, actor["is_admin"]), "credential": credential,
                "scheduler_id": state["scheduler_id"]}
    if action == "resources":
        return inventory(controller, state)
    if action == "pools.list":
        return {"pools": [dict(id=key, **value) for key, value in state["pools"].items()]}
    if action == "tokens.list":
        return {"tokens": [public_token(key, row) for key, row in state["tokens"].items()
                           if row["user_id"] == uid and row["kind"] == "regular"]}
    if action == "tokens.create":
        # Replay returns metadata only; the original secret is never persisted.
        operation = body.get("operation_id")
        if not isinstance(operation, str) or not 1 <= len(operation) <= 128:
            raise ValueError("operation_id required")
        name = body.get("name", "personal")
        _name(name)
        expiry = body.get("expires_at")
        import math
        if expiry is not None and (type(expiry) not in (int, float) or not math.isfinite(expiry)):
            raise ValueError("invalid expiry")
        params = {"name": name, "expires_at": float(expiry) if expiry is not None else None}
        with store._transaction() as current:
            actor = _authenticate(current, token, store.clock())
            key = "web-token:" + actor["user_id"] + ":" + operation
            previous = current["operations"].get(key)
            if previous:
                if previous["params"] != params:
                    raise ValueError("operation parameters differ")
                return dict(previous["result"], already_issued=True)
            if expiry is not None and expiry <= store.clock():
                raise ValueError("expiry must be in the future")
            secret, tid = new_secret(), str(uuid.uuid4())
            row = dict(user_id=actor["user_id"], name=name, kind="regular", verifier=verifier(secret),
                       expires_at=params["expires_at"], revoked=False, created_at=store.clock())
            current["tokens"][tid] = row
            result = public_token(tid, row)
            current["operations"][key] = {"owner_user_id":actor["user_id"], "params":params, "result":result}
            _audit(current, actor["user_id"], "token.create", tid, store.clock())
            return dict(result, token=secret)
    if action == "tokens.revoke":
        store.revoke_token(token, body["token_id"])
        return {"revoked": True}
    if action == "nodes.list":
        return {"nodes": [dict(id=key, **{k:v for k,v in row.items() if k != "verifier"})
                          for key,row in state.get("nodes", {}).items()
                          if actor["is_admin"] or row["owner_user_id"] == uid]}
    if action == "nodes.enroll":
        from .identity_nodes import enroll
        return enroll(store, token, body)
    if action == "nodes.action":
        from .managed_lifecycle import node_action, edit_node
        if body.get("action") == "edit":
            if type(body.get("force", False)) is not bool or body.get("force", False):
                raise ValueError("force applies to drain, not edit")
            return edit_node(controller, token, body["node_id"], body["changes"])
        return node_action(controller, token, body["node_id"], body["action"], body.get("force", False))
    store.require_admin(token)
    if action == "users.list":
        return {"users": [public_user(state, key) for key in state["users"]]}
    if action == "users.update":
        if set(body) - {"operation_id", "user_id", "groups", "disabled"}:
            raise ValueError("use group membership, not legacy grants or platform_admin")
        store.edit_user(token, body["user_id"], body.get("groups"), body.get("disabled"))
    elif action == "groups.list":
        return {"groups": state["groups"]}
    elif action == "groups.set":
        store.set_group(token, body["name"], body.get("all_pools", False), body.get("pools", []))
    elif action == "groups.delete":
        store.delete_group(token, body["name"])
    elif action == "pools.create":
        return {"id": store.create_pool(token, body["name"], body.get("config"), body.get("max_nodes"))}
    elif action == "pools.edit":
        from .managed_lifecycle import edit_pool
        return edit_pool(controller, token, controller.auth.pool_id(body["pool"]), body["changes"])
    elif action == "pools.drain" or action == "pools.remove":
        from .managed_lifecycle import pool_action
        return pool_action(controller, token, controller.auth.pool_id(body["pool"]), action.split(".")[1], body.get("force", False))
    elif action == "audit.list":
        return {"items": store.audit_events(token, body.get("limit", 100))}
    else:
        raise ValueError("unsupported management v2 action; use identity groups and node enrollment")
    return {"updated": True}
