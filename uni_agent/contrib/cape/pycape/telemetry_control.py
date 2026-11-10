# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable desired configuration, not a telemetry sample database."""

import copy
from dataclasses import asdict

from .telemetry_contract import CollectionConfig, identifier


MAX_VERSION = 2 ** 53 - 1


def initial_state():
    return {"version": 0, "default": asdict(CollectionConfig()), "overrides": {}, "bindings": {}, "receipts": {}}


def validate_state(value):
    required = {"version", "default", "overrides", "bindings", "receipts"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"locations"}:
        raise ValueError("invalid telemetry state")
    if type(value["version"]) is not int or not 0 <= value["version"] <= MAX_VERSION:
        raise ValueError("invalid telemetry version")
    CollectionConfig.parse(value["default"])
    for name in ("overrides", "bindings", "receipts"):
        if not isinstance(value[name], dict) or len(value[name]) > 65536:
            raise ValueError("invalid telemetry records")
    for service, config in value["overrides"].items():
        identifier(service)
        CollectionConfig.parse(config)
    for node, service in value["bindings"].items():
        identifier(node)
        identifier(service)
    for node, receipt in value["receipts"].items():
        identifier(node)
        if (not isinstance(receipt, dict) or set(receipt) != {"version", "state", "service_id"}
                or type(receipt["version"]) is not int or not 0 <= receipt["version"] <= value["version"]
                or receipt["state"] not in ("pending", "applied", "error")
                or receipt["service_id"] != value["bindings"].get(node)):
            raise ValueError("invalid telemetry receipt")
    locations = value.get("locations", {})
    if not isinstance(locations, dict) or len(locations) > 65536:
        raise ValueError("invalid telemetry locations")
    for key, location in locations.items():
        identifier(key)
        if not isinstance(location, dict) or set(location) != {"url", "incarnation", "expires_at"}:
            raise ValueError("invalid telemetry location")
        _location_url(location["url"])
        identifier(location["incarnation"])
        from .telemetry_contract import finite
        finite(location["expires_at"], 0, 1e20)


def _location_url(value):
    from urllib.parse import urlsplit
    if not isinstance(value, str) or len(value) > 4096 or any(ord(c) < 33 for c in value):
        raise ValueError("invalid service URL")
    url = urlsplit(value)
    if url.scheme != "http" or not url.hostname or url.username is not None or url.password is not None or url.query or url.fragment or url.path not in ("", "/"):
        raise ValueError("invalid service URL")
    url.port


def public_config(store, service_id):
    identifier(service_id)
    state = store.snapshot()
    telemetry = state.get("telemetry", initial_state())
    if service_id not in telemetry["bindings"].values():
        raise ValueError("unknown telemetry service")
    config = telemetry["overrides"].get(service_id, telemetry["default"])
    location = copy.deepcopy(telemetry.get("locations", {}).get(service_id))
    return {"scheduler_id": state["scheduler_id"], "service_id": service_id, "version": telemetry["version"],
            "config": config, "location": location,
            "lease_remaining_seconds": max(0, location["expires_at"] - store.clock()) if location else 0}


def _view(state, node_id=None):
    telemetry = state.get("telemetry", initial_state())
    validate_state(telemetry)
    service_id = None
    if node_id is not None:
        identifier(node_id)
        if node_id not in state.get("nodes", {}):
            raise ValueError("unknown node")
        service_id = telemetry["bindings"].get(node_id)
        if service_id is None:
            raise ValueError("node has not registered a metrics service")
    config = telemetry["overrides"].get(service_id, telemetry["default"])
    receipts = {key: value for key, value in telemetry["receipts"].items()
                if service_id is None or value["service_id"] == service_id}
    return copy.deepcopy({"scheduler_id": state["scheduler_id"], "service_id": service_id,
                          "version": telemetry["version"], "config": config, "receipts": receipts})


def get_config(store, token, node_id=None, admin=False):
    if admin:
        store.require_admin(token)
    else:
        store.authenticate(token)
    return _view(store.snapshot(), node_id)


def set_config(store, token, changes, node_id=None, expected_version=None):
    from .identity import _admin, _audit
    store.require_admin(token)
    if not isinstance(changes, dict) or not changes:
        raise ValueError("telemetry changes must be a nonempty object")
    if expected_version is not None and (type(expected_version) is not int or not 0 <= expected_version <= MAX_VERSION):
        raise ValueError("invalid expected telemetry version")
    with store._transaction() as state:
        actor = _admin(state, token, store.clock())
        current = _view(state, node_id)
        if expected_version is not None and current["version"] != expected_version:
            raise ValueError("telemetry configuration version conflict")
        config = asdict(CollectionConfig.parse(dict(current["config"], **changes)))
        if config != current["config"]:
            telemetry = state.setdefault("telemetry", initial_state())
            if telemetry["version"] >= MAX_VERSION:
                raise ValueError("telemetry version exhausted")
            telemetry["version"] += 1
            if current["service_id"] is None:
                telemetry["default"] = config
            else:
                telemetry["overrides"][current["service_id"]] = config
            _audit(state, actor["user_id"], "metric.configure", current["service_id"] or state["scheduler_id"], store.clock())
        return _view(state, node_id)


def poll_control(store, token, body):
    from .identity import verifier
    from .identity_nodes import authenticate_agent
    if isinstance(body, dict) and set(body) == {"node_id", "agent_epoch", "discover"} and body["discover"] is True:
        node = authenticate_agent(store, token, node_id=body["node_id"], epoch=body["agent_epoch"])
        if node["desired"] == "removed":
            raise PermissionError("invalid node credential")
        return {"scheduler_id": store.snapshot()["scheduler_id"]}
    required = {"node_id", "agent_epoch", "service_id"}
    if (not isinstance(body, dict) or not required <= set(body)
            or set(body) - required - {"applied_version", "state", "service_url", "incarnation"}):
        raise ValueError("invalid telemetry control request")
    for field in required:
        identifier(body[field])
    node = authenticate_agent(store, token, node_id=body["node_id"], epoch=body["agent_epoch"])
    if node["desired"] == "removed":
        raise PermissionError("retired agent cannot configure telemetry")
    version = body.get("applied_version", 0)
    status = body.get("state", "pending")
    if ("service_url" in body) != ("incarnation" in body):
        raise ValueError("service claim requires URL and incarnation")
    if "service_url" in body:
        _location_url(body["service_url"])
        identifier(body["incarnation"])
    if type(version) is not int or not 0 <= version <= MAX_VERSION or status not in ("pending", "applied", "error"):
        raise ValueError("invalid telemetry acknowledgement")
    snapshot = store.snapshot()
    receipt = {"version": version, "state": status, "service_id": body["service_id"]}
    telemetry = snapshot.get("telemetry", {})
    if ("service_url" not in body and telemetry.get("bindings", {}).get(body["node_id"]) == body["service_id"]
            and telemetry.get("receipts", {}).get(body["node_id"]) == receipt):
        desired = _view(snapshot, body["node_id"])
        return {key: desired[key] for key in ("scheduler_id", "service_id", "version", "config")}
    with store._transaction() as state:
        current_node = state["nodes"][body["node_id"]]
        if (current_node["verifier"] != verifier(token) or current_node["agent_epoch"] != body["agent_epoch"]
                or current_node["desired"] == "removed" or "purge" in current_node):
            raise PermissionError("invalid node credential")
        telemetry = state.setdefault("telemetry", initial_state())
        service_id = body["service_id"]
        previous = telemetry["bindings"].get(body["node_id"])
        if previous is not None and previous != service_id:
            raise ValueError("telemetry service binding conflict")
        if version > telemetry["version"]:
            raise ValueError("future telemetry acknowledgement")
        telemetry["bindings"][body["node_id"]] = service_id
        receipt = telemetry["receipts"].get(body["node_id"])
        if receipt is None or version >= receipt["version"]:
            telemetry["receipts"][body["node_id"]] = {"version": version, "state": status, "service_id": service_id}
        if "service_url" in body:
            locations = telemetry.setdefault("locations", {})
            location = locations.get(service_id)
            if location is None or location["expires_at"] <= store.clock() or location["incarnation"] == body["incarnation"]:
                locations[service_id] = {"url": body["service_url"], "incarnation": body["incarnation"], "expires_at": store.clock() + 30}
        desired = _view(state, body["node_id"])
        result = {key: desired[key] for key in ("scheduler_id", "service_id", "version", "config")}
        result["location"] = copy.deepcopy(telemetry.get("locations", {}).get(service_id))
        return result
