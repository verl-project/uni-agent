# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped node enrollment, without join credentials or approval workflows."""

import copy
import math
import secrets
import uuid


def resources(value):
    if not isinstance(value, dict) or set(value) != {"cpu_cores", "memory_gb", "gpu_uuids"}:
        raise ValueError("invalid resource declaration")
    result = {}
    for key, limit in (("cpu_cores", 100000), ("memory_gb", 10000000)):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 < number <= limit:
            raise ValueError("invalid resource capacity")
        result[key] = float(number)
    gpus = value["gpu_uuids"]
    if not isinstance(gpus, list) or len(gpus) > 64 or any(not isinstance(g, str) or not g.startswith("GPU-") for g in gpus):
        raise ValueError("invalid GPU UUIDs")
    canonical = ["GPU-" + str(uuid.UUID(g[4:])) for g in gpus]
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate GPU UUID")
    result["gpu_uuids"] = sorted(canonical)
    return result


def enroll(store, bearer, body):
    from .identity import _authenticate, _name, _VERIFIER, _audit
    required = {"operation_id", "verifier", "node_id", "pool", "agent_epoch", "allocation_id", "resources"}
    if not isinstance(body, dict) or not required.issubset(body) or set(body) - required - {"owner_user_id"}:
        raise ValueError("invalid enrollment parameters")
    operation = str(uuid.UUID(body["operation_id"]))
    if operation != body["operation_id"] or not isinstance(body["verifier"], str) or not _VERIFIER.fullmatch(body["verifier"]):
        raise ValueError("invalid enrollment operation")
    for key in ("node_id", "agent_epoch", "allocation_id"):
        _name(body[key])
    params = dict(body, resources=resources(body["resources"]))
    with store._transaction() as state:
        actor = _authenticate(state, bearer, store.clock())
        if "owner_user_id" in body and body["owner_user_id"] != actor["user_id"]:
            raise PermissionError("node enrollment target identity mismatch")
        key = "node:" + operation
        previous = state["operations"].get(key)
        if previous:
            if previous["owner_user_id"] != actor["user_id"]:
                raise PermissionError("enrollment belongs to another identity")
            if params != previous["params"]:
                raise ValueError("operation parameters differ")
            return copy.deepcopy(previous["result"])
        pool = state["pools"].get(body["pool"])
        if pool is None or pool["draining"]:
            raise ValueError("pool does not allow enrollment")
        nodes = state.setdefault("nodes", {})
        if pool.get("max_nodes") is not None and sum(n["pool"] == body["pool"] and n["desired"] != "removed" for n in nodes.values()) >= pool["max_nodes"]:
            raise ValueError("pool node limit reached")
        if body["node_id"] in nodes:
            raise ValueError("node identity already enrolled; use the saved operation to recover")
        if any(node["verifier"] == body["verifier"] or node["allocation_id"] == body["allocation_id"] for node in nodes.values()):
            raise ValueError("node credential or allocation already registered")
        from .management_sso import contains_verifier
        if contains_verifier(state, body["verifier"]):
            raise ValueError("SSO credential cannot be reused as node credential")
        if any(token["verifier"] == body["verifier"] for token in state["tokens"].values()):
            raise ValueError("personal credential cannot be reused as node credential")
        nodes[body["node_id"]] = dict(params, effective_resources=copy.deepcopy(params["resources"]), owner_user_id=actor["user_id"], desired="active", announced=False, intent_at=store.clock())
        result = {"node_id": body["node_id"], "pool": body["pool"], "allocation_id": body["allocation_id"],
                  "agent_epoch": body["agent_epoch"], "operation_id": operation, "status": "enrolled"}
        state["operations"][key] = {"owner_user_id": actor["user_id"], "params": params, "result": result}
        _audit(state, actor["user_id"], "node.enroll", body["node_id"], store.clock())
        return copy.deepcopy(result)


def authenticate_agent(store, secret, pool=None, node_id=None, epoch=None):
    from .identity import verifier
    digest = verifier(secret)
    for key, node in store.snapshot().get("nodes", {}).items():
        if secrets.compare_digest(node["verifier"], digest):
            if "purge" in node:
                break
            if ((pool is not None and pool != node["pool"]) or (node_id is not None and node_id != key)
                    or (epoch is not None and epoch != node["agent_epoch"])):
                break
            return copy.deepcopy(node)
    raise PermissionError("invalid node credential")


def registration(store, secret, node_id, pool, epoch, allocation_id, declared):
    if not epoch:
        raise PermissionError("node epoch required")
    node = authenticate_agent(store, secret, pool, node_id, epoch)
    if allocation_id != node["allocation_id"] or resources(declared) != node["resources"]:
        raise PermissionError("node registration differs from enrolled scope")
    if node["desired"] == "removed" and not node.get("announced", False):
        raise PermissionError("node enrollment is retired")
    if not node.get("announced", False):
        from .identity import _audit
        with store._transaction() as state:
            state["nodes"][node_id]["announced"] = True
            _audit(state, "node:" + node_id, "node.registration-started", node_id, store.clock())
    return node


def owned_node(store, bearer, node_id):
    actor = store.authenticate(bearer)
    node = store.snapshot().get("nodes", {}).get(node_id)
    if node is None or (not actor["is_admin"] and node["owner_user_id"] != actor["user_id"]):
        raise PermissionError("node not accessible")
    return node
