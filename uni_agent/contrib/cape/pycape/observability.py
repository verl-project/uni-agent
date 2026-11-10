# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

from .models import NodeState, REQUEST_TERMINAL_STATES, SessionState


REQUEST_REASON_MESSAGES = {
    "node_purged": "an administrator retired the lost node without remote cleanup confirmation",
    "pool_empty": "no active node is ready in the pool yet; waiting for a node to register",
    "permission_revoked": "current account or permission groups no longer allow this queued request",
    "pool_draining": "pool is draining; this unassigned request will not be dispatched",
    "workspace_node_unavailable": "the workspace node is retiring; its retained image has not been migrated or deleted",
    "authority_unavailable": "authorization state is unavailable; assignment is paused",
    "node_probing": "a node has joined but is still probing runtime capabilities; waiting for it to become active",
    "no_capable_node": "nodes are present, but none can satisfy the request",
    "waiting_for_gpu": "waiting for enough leased-GPU capacity",
    "waiting_for_cpu_or_memory": "waiting for enough CPU or memory capacity",
    "waiting_for_runtime_capability": "waiting for a node with the required runtime capability",
    "waiting_for_image_cache_or_prepare": "waiting for image preparation or a suitable cached image",
    "waiting_for_session": "waiting for the matching reusable session to become idle",
    "retained_node_unavailable": "waiting for the original node that holds this execution's retained state",
    "retained_state_unavailable": "retained execution state cannot be confirmed; explicit recovery or rebuild is required",
    "forced_suspend": "active work was cancelled by an explicit forced suspension",
    "workspace_maintenance_failed": "workspace maintenance failed; recover or release the workspace before retrying",
    "allocation_expiring": "matching allocation capacity is too close to its deadline",
    "node_draining": "matching nodes are draining and no longer accept new work",
    "quota_exceeded": "waiting for user or pool quota capacity",
    "pool_session_limit": "waiting for the pool's active-session limit",
    "fairness_delay": "queued behind other work under the fairness policy",
    "node_reregistered": "the assigned node restarted and CAPE is reconciling its session",
    "controller_restart": "the controller restarted and CAPE is reconciling the request",
    "reconcile_missing": "the assigned session was absent after the reconciliation grace period",
    "command_failed": "the workload command exited unsuccessfully",
    "infrastructure_failed": "the node agent or runtime infrastructure failed",
    "command_status_unknown": "the node reported an idle session but the command result is unknown",
    "assignment_missing": "the agent no longer reports the dispatched command; execution outcome is unknown and it was not automatically retried",
    "cancelled": "the request was cancelled before normal completion",
    "node_lost": "the assigned node stopped heartbeating and exceeded the loss timeout",
    "max_lifetime": "the reusable session reached its declared maximum lifetime",
    "allocation_removed": "the allocation was removed before the request could complete",
    "node_removed": "the assigned node was removed before the request could complete",
    "allocation_drain": "the allocation is draining and no longer accepts new work",
    "operator_drain": "an operator placed the assigned capacity into drain mode",
    "stopped": "the request or its session was stopped",
}


SESSION_REASON_MESSAGES = {
    "node_purged": "an administrator retired the lost node without remote cleanup confirmation",
    "operator_suspend": "the owner requested suspension while preserving filesystem state",
    "suspend_requested": "suspension is waiting for the active command to finish",
    "suspended": "the agent confirmed retained state and released execution resources",
    "retained_after_stop": "the stopped execution's filesystem state is retained for reuse",
    "purge_requested": "explicit execution-layer deletion is awaiting agent confirmation",
    "purged": "the execution layer was explicitly deleted; workspace data is separate",
    "forced_suspend": "active work was cancelled before suspension",
    "command_failed": "the workload command exited unsuccessfully",
    "infrastructure_failed": "the node agent or runtime infrastructure failed",
    "node_lost": "the assigned node exceeded the loss timeout",
    "idle_timeout": "the cached session exceeded its idle timeout",
    "max_lifetime": "the session reached its declared maximum lifetime",
    "stopped": "the session was stopped",
    "evict_for_queued_work": "the cached session was reclaimed for queued work",
    "allocation_drain": "the allocation entered drain mode",
    "allocation_removed": "the allocation was removed",
    "node_removed": "the node was removed",
    "operator_drain": "an operator placed the node into drain mode",
    "allocation_expiring": "the declared allocation deadline entered its advisory drain window",
    "reconcile_missing": "the session was absent from authoritative node status after reconciliation",
    "reconciled": "authoritative node status reconciled the session",
    "node_reregistered": "the node re-registered and the session requires reconciliation",
    "lost_session_cleanup_confirmed": "the recovered agent confirmed cleanup of a previously lost session",
}


NODE_REASON_MESSAGES = {
    "node_purged": "an administrator retired the lost node without remote cleanup confirmation",
    "heartbeat_timeout": "the node missed heartbeats and is temporarily unhealthy",
    "heartbeat_lost": "the node exceeded the loss timeout",
    "allocation_expiring": "the declared allocation deadline entered its advisory drain window",
    "allocation_deadline": "the declared allocation deadline was reached",
    "allocation_drain": "the allocation entered drain mode",
    "allocation_removed": "the allocation was removed",
    "node_removed": "the node was removed",
    "operator_drain": "an operator placed the node into drain mode",
    "remove_pending_sessions": "node removal is waiting for active sessions",
}


REQUEST_STATE_MESSAGES = {
    "SUBMITTED": "the request was accepted",
    "QUEUED": "the request is waiting for schedulable capacity",
    "INFEASIBLE": "the request cannot currently be admitted",
    "LEASED": "resources have been leased",
    "PREPARING_IMAGE": "the runtime is preparing the image",
    "STARTING_SESSION": "the runtime is starting the session",
    "RUNNING": "the workload is executing",
    "RECONCILING": "CAPE is reconciling state with the node",
    "COMPLETED": "the workload completed successfully",
    "FAILED": "the request failed",
    "CANCELLED": "the request was cancelled",
    "EXPIRED": "the request expired",
    "LOST": "the assigned node or session was lost",
}


def request_status_message(state, reason=None):
    state = str(state or "UNKNOWN")
    if reason:
        message = REQUEST_REASON_MESSAGES.get(str(reason))
        if message is None:
            message = "request state reason: {}".format(reason)
        return "{} ({}): {}".format(state, reason, message)
    return "{}: {}".format(
        state,
        REQUEST_STATE_MESSAGES.get(
            state, "request state is {}".format(state)
        ),
    )


class PoolUtilizationSummary(object):
    def __init__(self, pool, nodes, gpus_total, gpus_leased, cpu_total, cpu_used, memory_total, memory_used, running_requests, queued_requests, active_nodes=0, registering_nodes=0, gpus_schedulable=0, gpus_available=0, cpu_schedulable=0, memory_schedulable=0, node_states=(), pool_mode=None, gpu_slots_total=0, gpu_slots_leased=0, gpu_slots_available=0, physical=None, pool_config=None, requires_matching_join=False):
        self.pool = pool
        self.nodes = nodes
        self.gpus_total = gpus_total
        self.gpus_leased = gpus_leased
        self.cpu_total = cpu_total
        self.cpu_used = cpu_used
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.running_requests = running_requests
        self.queued_requests = queued_requests
        self.active_nodes = active_nodes
        self.registering_nodes = registering_nodes
        self.gpus_schedulable = gpus_schedulable
        self.gpus_available = gpus_available
        self.cpu_schedulable = cpu_schedulable
        self.memory_schedulable = memory_schedulable
        self.node_states = list(node_states)
        self.pool_mode = pool_mode
        self.gpu_slots_total = gpu_slots_total
        self.gpu_slots_leased = gpu_slots_leased
        self.gpu_slots_available = gpu_slots_available
        self.physical = dict(physical or {})
        self.pool_config = (
            dict(pool_config) if pool_config is not None else None
        )
        self.requires_matching_join = bool(requires_matching_join)

    def to_dict(self):
        return {
            "pool": self.pool,
            "nodes": self.nodes,
            "gpus_total": self.gpus_total,
            "gpus_leased": self.gpus_leased,
            "cpu_total": self.cpu_total,
            "cpu_used": self.cpu_used,
            "memory_total": self.memory_total,
            "memory_used": self.memory_used,
            "running_requests": self.running_requests,
            "queued_requests": self.queued_requests,
            "active_nodes": self.active_nodes,
            "registering_nodes": self.registering_nodes,
            "gpus_schedulable": self.gpus_schedulable,
            "gpus_available": self.gpus_available,
            "cpu_schedulable": self.cpu_schedulable,
            "memory_schedulable": self.memory_schedulable,
            "node_states": list(self.node_states),
            "pool_mode": self.pool_mode,
            "gpu_slots_total": self.gpu_slots_total,
            "gpu_slots_leased": self.gpu_slots_leased,
            "gpu_slots_available": self.gpu_slots_available,
            "physical": dict(self.physical),
            "pool_config": (
                dict(self.pool_config)
                if self.pool_config is not None
                else None
            ),
            "requires_matching_join": self.requires_matching_join,
        }


def summarize_pool(controller, pool=None):
    established_pool_config = (
        controller.pool_configs.get(pool)
        if pool is not None
        else None
    )
    nodes = [
        node
        for node in controller.nodes.values()
        if pool is None or node.pool == pool
    ]
    visible_nodes = [
        node for node in nodes if node.state != NodeState.REMOVED
    ]
    active_nodes = [
        node for node in visible_nodes if node.state == NodeState.ACTIVE
    ]
    registering_nodes = [
        node
        for node in visible_nodes
        if node.state == NodeState.REGISTERING
    ]
    requests = [
        request
        for request in controller.requests.values()
        if pool is None or request.spec.pool == pool
    ]
    relevant_gpu_uuids = {
        gpu_uuid
        for node in visible_nodes
        for gpu_uuid in node.resources.gpu_uuids
    }

    def holder_is_in_view(execution_session_id):
        session = controller.sessions.get(execution_session_id)
        return bool(
            session is not None
            and (pool is None or session.pool == pool)
        )

    gpus_leased = sum(
        1
        for gpu_uuid in relevant_gpu_uuids
        if any(
            holder_is_in_view(execution_session_id)
            for execution_session_id in controller.gpu_ledger.get(
                gpu_uuid
            ).holder_execution_session_ids
        )
    )
    gpu_slots_leased = sum(
        1
        for gpu_uuid in relevant_gpu_uuids
        for execution_session_id in controller.gpu_ledger.get(
            gpu_uuid
        ).holder_execution_session_ids
        if holder_is_in_view(execution_session_id)
    )

    def available_slots(node, gpu_uuid):
        if node.state != NodeState.ACTIVE:
            return 0
        if not controller._gpu_admission_allowed(node, gpu_uuid):
            return 0
        if node.pool_mode == "exclusive":
            return 1
        entry = controller.gpu_ledger[gpu_uuid]
        return max(0, node.gpu_slots - entry.shared_refcount)

    physical_gpus = []
    for gpu_uuid in sorted(relevant_gpu_uuids):
        entry = controller.gpu_ledger.get(gpu_uuid)
        state = entry.state.value if entry is not None else "FREE"
        if gpu_uuid in controller.node_purge_quarantine:
            state = "QUARANTINED"
        physical_gpus.append(
            {
                "gpu_uuid": gpu_uuid,
                "state": state,
                "shared_refcount": (
                    entry.shared_refcount if entry is not None else 0
                ),
            }
        )
    physical = {
        "gpus_total": len(relevant_gpu_uuids),
        "gpus_free": sum(
            item["state"] == "FREE" for item in physical_gpus
        ),
        "gpus_exclusive": sum(
            item["state"] == "EXCLUSIVE" for item in physical_gpus
        ),
        "gpus_shared": sum(
            item["state"] == "SHARED" for item in physical_gpus
        ),
        "exclusive_leases": sum(
            item["state"] == "EXCLUSIVE" for item in physical_gpus
        ),
        "shared_slots_leased": sum(
            item["shared_refcount"] for item in physical_gpus
        ),
        "gpus": physical_gpus,
    }
    physical["lease_count"] = (
        physical["exclusive_leases"]
        + physical["shared_slots_leased"]
    )
    node_states = []
    for node in sorted(visible_nodes, key=lambda item: item.node_id):
        active_sessions = list(node.active_sessions)
        nonterminal_requests = sum(
            1
            for request in requests
            if request.node_id == node.node_id
            and request.state not in REQUEST_TERMINAL_STATES
        )
        node_execution_session_ids = set(node.active_sessions)
        leased = sum(
            1
            for gpu_uuid in node.resources.gpu_uuids
            if any(
                execution_session_id in node_execution_session_ids
                for execution_session_id in controller.gpu_ledger[
                    gpu_uuid
                ].holder_execution_session_ids
            )
        )
        schedulable = node.state == NodeState.ACTIVE
        node_gpus_available = sum(
            1
            for gpu_uuid in node.resources.gpu_uuids
            if available_slots(node, gpu_uuid) > 0
        )
        node_states.append({
            "node_id": node.node_id,
            "allocation_id": node.allocation_id,
            "allocation_deadline": node.allocation_deadline,
            "state": node.state.value,
            "gpus_total": len(node.resources.gpu_uuids),
            "gpus_leased": leased,
            "gpus_available": (
                node_gpus_available if schedulable else 0
            ),
            "cpu_total": node.resources.cpu_cores,
            "cpu_available": node.available_cpu() if schedulable else 0,
            "memory_total": node.resources.memory_gb,
            "memory_available": (
                node.available_memory() if schedulable else 0
            ),
            "last_heartbeat_at": node.last_heartbeat_at,
            "active_sessions": active_sessions,
            "active_session_count": len(active_sessions),
            "nonterminal_requests": nonterminal_requests,
        })
    return PoolUtilizationSummary(
        pool=pool or "*",
        nodes=len(visible_nodes),
        gpus_total=sum(
            len(node.resources.gpu_uuids) for node in visible_nodes
        ),
        gpus_leased=gpus_leased,
        cpu_total=sum(node.resources.cpu_cores for node in visible_nodes),
        cpu_used=sum(node.used_cpu_cores for node in visible_nodes),
        memory_total=sum(node.resources.memory_gb for node in visible_nodes),
        memory_used=sum(node.used_memory_gb for node in visible_nodes),
        running_requests=len([
            request for request in requests
            if request.state.value == "RUNNING"
        ]),
        queued_requests=len([
            request for request in requests
            if request.state.value == "QUEUED"
        ]),
        active_nodes=len(active_nodes),
        registering_nodes=len(registering_nodes),
        gpus_schedulable=sum(
            sum(gpu not in controller.node_purge_quarantine for gpu in node.resources.gpu_uuids) for node in active_nodes
        ),
        gpus_available=sum(
            sum(
                1
                for gpu_uuid in node.resources.gpu_uuids
                if available_slots(node, gpu_uuid) > 0
            )
            for node in active_nodes
        ),
        cpu_schedulable=sum(
            node.available_cpu() for node in active_nodes
        ),
        memory_schedulable=sum(
            node.available_memory() for node in active_nodes
        ),
        node_states=node_states,
        pool_mode=(
            controller.pool_configs[pool].pool_mode
            if pool in controller.pool_configs
            else (
                next(iter({
                    node.pool_mode for node in visible_nodes
                }))
                if visible_nodes
                and len({
                    node.pool_mode for node in visible_nodes
                })
                == 1
                else "mixed"
            )
        ),
        gpu_slots_total=sum(
            len(node.resources.gpu_uuids) * node.gpu_slots
            for node in visible_nodes
        ),
        gpu_slots_leased=gpu_slots_leased,
        gpu_slots_available=sum(
            available_slots(node, gpu_uuid)
            for node in active_nodes
            for gpu_uuid in node.resources.gpu_uuids
        ),
        physical=physical,
        pool_config=(
            established_pool_config.to_dict()
            if established_pool_config is not None
            else None
        ),
        requires_matching_join=(
            controller.pool_config_requires_matching_join(pool)
            if pool is not None
            else False
        ),
    )


class NoopExporter(object):
    def export(self, name, payload):
        return None


class JsonlExporter(object):
    def __init__(self, path):
        self.path = path

    def export(self, name, payload):
        parent = os.path.dirname(self.path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"time": time.time(), "name": name, "payload": payload}, sort_keys=True) + "\n")


class WandbExporter(object):
    def __init__(self, token_path=None, enabled=False, project="pycape", entity=None, run_name=None, init_kwargs=None):
        self.token_path = token_path
        self.enabled = enabled
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.init_kwargs = dict(init_kwargs or {})
        self.last_payload = None
        self.last_error = None
        self._run = None

    def export(self, name, payload):
        self.last_payload = (name, payload)
        if not self.enabled:
            return None
        try:
            token = self._read_token()
            if token and "WANDB_API_KEY" not in os.environ:
                os.environ["WANDB_API_KEY"] = token
            wandb = __import__("wandb")
            run = self._ensure_run(wandb)
            record = self._log_record(name, payload)
            if hasattr(run, "log"):
                run.log(record)
            elif hasattr(wandb, "log"):
                wandb.log(record)
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def _read_token(self):
        if not self.token_path:
            return None
        with open(self.token_path) as fh:
            return fh.read().strip()

    def _ensure_run(self, wandb):
        if self._run is not None:
            return self._run
        kwargs = {"project": self.project}
        if self.entity:
            kwargs["entity"] = self.entity
        if self.run_name:
            kwargs["name"] = self.run_name
        kwargs.update(self.init_kwargs)
        if hasattr(wandb, "init"):
            self._run = wandb.init(**kwargs)
        else:
            self._run = wandb
        return self._run

    def _log_record(self, name, payload):
        if isinstance(payload, dict):
            record = dict(payload)
        else:
            record = {"value": payload}
        record["metric_name"] = name
        record["timestamp"] = time.time()
        return record


class MetricsCollector(object):
    def __init__(self, enabled=True, nvml=None, device_uuids=None):
        self.enabled = enabled
        self.nvml = nvml
        self.device_uuids = set(device_uuids or ())
        self.last_error = None

    def collect_gpu_metrics(self):
        if not self.enabled:
            return []
        try:
            nvml = self.nvml or __import__("pynvml")
            if hasattr(nvml, "nvmlInit"):
                nvml.nvmlInit()
            count = nvml.nvmlDeviceGetCount()
            metrics = []
            for index in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                record = self._record_for_handle(nvml, handle, index)
                if self.device_uuids and record.get("uuid") not in self.device_uuids:
                    continue
                metrics.append(record)
            self.last_error = None
            return metrics
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def _record_for_handle(self, nvml, handle, index):
        uuid = _decode_nvml(nvml.nvmlDeviceGetUUID(handle))
        name = _decode_nvml(nvml.nvmlDeviceGetName(handle)) if hasattr(nvml, "nvmlDeviceGetName") else None
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle) if hasattr(nvml, "nvmlDeviceGetUtilizationRates") else None
        memory = nvml.nvmlDeviceGetMemoryInfo(handle) if hasattr(nvml, "nvmlDeviceGetMemoryInfo") else None
        record = {
            "index": index,
            "uuid": uuid,
            "name": name,
            "timestamp": time.time(),
        }
        if utilization is not None:
            record["gpu_utilization_percent"] = getattr(utilization, "gpu", None)
            record["memory_utilization_percent"] = getattr(utilization, "memory", None)
        if memory is not None:
            record["memory_total_bytes"] = getattr(memory, "total", None)
            record["memory_used_bytes"] = getattr(memory, "used", None)
            record["memory_free_bytes"] = getattr(memory, "free", None)
        if hasattr(nvml, "nvmlDeviceGetTemperature") and hasattr(nvml, "NVML_TEMPERATURE_GPU"):
            try:
                record["temperature_c"] = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
        return record


def _decode_nvml(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
