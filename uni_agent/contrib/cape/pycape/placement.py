# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .models import EnforcementMode, NodeState


RUNTIME_POLICY_PREFERENCES = {
    "default": ("apptainer", "podman", "bubblewrap", "local-process"),
    "none": ("local-process",),
    "host-stateless": ("bubblewrap",),
    "host-stateful": ("bubblewrap",),
    "image-stateless": ("apptainer", "podman"),
    "image-stateful": ("apptainer", "podman"),
}


def runtime_preference_rank(isolation_policy, adapter):
    preferences = RUNTIME_POLICY_PREFERENCES.get(isolation_policy, ())
    try:
        return preferences.index(adapter)
    except ValueError:
        return len(preferences)


class PlacementResult(object):
    def __init__(self, node_id, gpu_uuids=(), reason=None, runtime_adapter=None):
        self.node_id = node_id
        self.gpu_uuids = tuple(gpu_uuids)
        self.reason = reason
        self.runtime_adapter = runtime_adapter

    @property
    def placed(self):
        return self.node_id is not None


def _runtime_capability_map(caps, runtime_capabilities=None):
    if runtime_capabilities:
        return dict(runtime_capabilities)
    if caps is None or caps.adapter == "multi":
        return {}
    return {caps.adapter: caps}


def _candidate_runtime_capabilities(
    spec, caps, runtime_capabilities=None, runtime_adapter=None
):
    capabilities_by_adapter = _runtime_capability_map(
        caps, runtime_capabilities
    )
    pinned_adapter = runtime_adapter or spec.resources.runtime_adapter
    if pinned_adapter:
        capability = capabilities_by_adapter.get(pinned_adapter)
        if capability is None:
            return []
        candidates = [(pinned_adapter, capability)]
    else:
        candidates = list(capabilities_by_adapter.items())
    candidates = [
        (adapter, capability)
        for adapter, capability in candidates
        if spec.resources.isolation_policy in capability.isolation_policies
    ]
    return sorted(
        candidates,
        key=lambda item: (
            runtime_preference_rank(spec.resources.isolation_policy, item[0]),
            item[0],
        ),
    )


def _capability_reason_for_adapter(spec, caps):
    if not caps.nested_container:
        return "waiting_for_runtime_capability"
    if spec.effective_mounts() and not caps.can_bind_mounts:
        return "waiting_for_runtime_capability"
    if spec.resources.gpus > 0:
        if not caps.can_select_gpus:
            return "waiting_for_runtime_capability"
        if (
            spec.resources.isolation_policy
            not in caps.gpu_isolation_policies
        ):
            return "waiting_for_runtime_capability"
    if spec.resources.hard_gpu_enforcement and caps.gpu_enforcement != EnforcementMode.HARD:
        return "waiting_for_runtime_capability"
    if spec.privileged and not caps.privileged_allowed:
        return "no_capable_node"
    return None


def resolve_runtime_capability(
    spec, caps, runtime_capabilities=None, runtime_adapter=None
):
    for adapter, capability in _candidate_runtime_capabilities(
        spec,
        caps,
        runtime_capabilities=runtime_capabilities,
        runtime_adapter=runtime_adapter,
    ):
        if _capability_reason_for_adapter(spec, capability) is None:
            return adapter, capability
    return None


def capability_reason(
    spec, caps, runtime_capabilities=None, runtime_adapter=None
):
    candidates = _candidate_runtime_capabilities(
        spec,
        caps,
        runtime_capabilities=runtime_capabilities,
        runtime_adapter=runtime_adapter,
    )
    if not candidates:
        return "waiting_for_runtime_capability"
    reasons = [
        _capability_reason_for_adapter(spec, capability)
        for _, capability in candidates
    ]
    if any(reason is None for reason in reasons):
        return None
    if reasons and all(reason == "no_capable_node" for reason in reasons):
        return "no_capable_node"
    return "waiting_for_runtime_capability"


def rank_nodes_for_request(nodes, request, free_gpu_counts):
    if request.gpus > 1:
        return sorted(
            nodes,
            key=lambda n: (
                free_gpu_counts[n.node_id] - request.gpus,
                len(n.active_sessions),
                n.used_cpu_cores,
                n.node_id,
            ),
        )
    if request.gpus == 1:
        return sorted(
            nodes,
            key=lambda n: (
                len(n.active_sessions),
                n.used_cpu_cores,
                -(free_gpu_counts[n.node_id]),
                n.node_id,
            ),
        )
    return sorted(nodes, key=lambda n: (len(n.active_sessions), n.used_cpu_cores, n.node_id))


def node_can_accept_lifetime(node, now, max_duration_seconds):
    remaining = node.remaining_safe_seconds(now)
    if remaining is None:
        return True
    return max_duration_seconds <= remaining


def node_is_placement_eligible(node, now):
    if node.state != NodeState.ACTIVE:
        return False
    if node.should_drain(now):
        return False
    return True
