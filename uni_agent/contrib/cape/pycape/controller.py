# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import time
import uuid

from .journal import EventRecord, MemoryJournal
from .metrics import NoopMetricsSink
from .workspace import WorkspaceConfig, WorkspaceConfigStore, workspace_bind_conflict
from .workspace_fs import WorkspaceManager
from .workspace_policy import MaintenancePolicy, automatic_grow_target
from .workspace_reclaim import reclaim_candidate
from .models import (
    AllocationRecord,
    AllocationState,
    CommandAssignment,
    COMMAND_FAILURE_KINDS,
    CommandState,
    EnforcementMode,
    GPU_CONTENTION_POLICIES,
    GpuLeaseState,
    MountSpec,
    NodeRecord,
    NodeResources,
    NodeState,
    PhysicalGpuLedgerEntry,
    PoolConfig,
    REQUEST_TERMINAL_STATES,
    RequestRecord,
    RequestState,
    ResourceRequest,
    Role,
    RuntimeCapabilities,
    SESSION_TERMINAL_STATES,
    STATEFUL_ISOLATION_POLICIES,
    SessionRecord,
    SessionState,
    WorkloadSpec,
)
from .placement import PlacementResult, capability_reason, node_can_accept_lifetime, node_is_placement_eligible, rank_nodes_for_request, resolve_runtime_capability, runtime_preference_rank
from .observability import request_status_message, summarize_pool
from .runtime import ContainerSessionSpec, RuntimeResult


class PoolConfigConflictError(ValueError):
    def __init__(
        self,
        message,
        pool,
        established_config,
        requested_config,
        blockers,
        requires_matching_join=False,
    ):
        hint = (
            "drain all members and resolve queued/live work, then use "
            "pycape admin pool edit"
        )
        ValueError.__init__(self, "{}; {}".format(message, hint))
        self.payload = {
            "error": str(self),
            "pool": pool,
            "established_config": (
                established_config.to_dict()
                if established_config is not None
                else None
            ),
            "requested_config": requested_config.to_dict(),
            "blockers": {
                "member_node_ids": list(
                    blockers["member_node_ids"]
                ),
                "request_ids": list(blockers["request_ids"]),
                "execution_session_ids": list(blockers["execution_session_ids"]),
            },
            "requires_matching_join": bool(requires_matching_join),
            "hint": hint,
        }


class SchedulerPolicy(object):
    def __init__(
        self,
        max_active_gpu_sessions_per_user=None,
        max_queued_requests_per_user=None,
        max_active_sessions_per_pool=None,
        allow_privileged=False,
        heartbeat_timeout_seconds=30.0,
        lost_timeout_seconds=90.0,
        idle_session_timeout_seconds=3600.0,
        reconcile_missing_grace_seconds=5.0,
        reconcile_missing_heartbeats=2,
        default_image="docker://ubuntu:latest",
        profile_registry=None,
        gpu_contention_policy="barrier-ratio",
        gpu_shared_ratio=4,
        gpu_share_overrides=None,
    ):
        # Kept for API compatibility. The unit is now active GPU leases, not sessions.
        self.max_active_gpu_sessions_per_user = max_active_gpu_sessions_per_user
        self.max_queued_requests_per_user = max_queued_requests_per_user
        self.max_active_sessions_per_pool = max_active_sessions_per_pool
        self.allow_privileged = allow_privileged
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.lost_timeout_seconds = lost_timeout_seconds
        self.idle_session_timeout_seconds = idle_session_timeout_seconds
        self.reconcile_missing_grace_seconds = max(
            0.0, float(reconcile_missing_grace_seconds)
        )
        self.reconcile_missing_heartbeats = max(
            1, int(reconcile_missing_heartbeats)
        )
        self.default_image = default_image
        self.profile_registry = profile_registry
        if gpu_contention_policy not in GPU_CONTENTION_POLICIES:
            raise ValueError(
                "gpu_contention_policy must be one of: "
                + ", ".join(GPU_CONTENTION_POLICIES)
            )
        if (
            isinstance(gpu_shared_ratio, bool)
            or not isinstance(gpu_shared_ratio, int)
            or gpu_shared_ratio <= 0
        ):
            raise ValueError("gpu_shared_ratio must be a positive integer")
        self.gpu_contention_policy = gpu_contention_policy
        self.gpu_shared_ratio = gpu_shared_ratio
        self.gpu_share_overrides = {}
        for gpu_uuid, override in dict(
            gpu_share_overrides or {}
        ).items():
            if not gpu_uuid:
                raise ValueError("GPU override UUID must not be empty")
            if isinstance(override, dict):
                override_policy = override.get(
                    "policy", gpu_contention_policy
                )
                override_ratio = override.get("ratio", gpu_shared_ratio)
            elif isinstance(override, (tuple, list)) and len(override) == 2:
                override_policy, override_ratio = override
            else:
                raise ValueError(
                    "GPU override must contain policy and ratio"
                )
            if override_policy not in GPU_CONTENTION_POLICIES:
                raise ValueError(
                    "GPU override policy must be one of: "
                    + ", ".join(GPU_CONTENTION_POLICIES)
                )
            if (
                isinstance(override_ratio, bool)
                or not isinstance(override_ratio, int)
                or override_ratio <= 0
            ):
                raise ValueError(
                    "GPU override ratio must be a positive integer"
                )
            self.gpu_share_overrides[str(gpu_uuid)] = (
                override_policy,
                override_ratio,
            )


class CAPEController(object):
    def __init__(self, journal=None, clock=None, policy=None, metrics=None, runtime_adapters=None, auth=None):
        self.clock = clock or time.time
        self.policy = policy or SchedulerPolicy()
        self.journal = journal or MemoryJournal()
        self.metrics = metrics or NoopMetricsSink()
        self.runtime_adapters = dict(runtime_adapters or {})
        self.auth = auth
        self.requests = {}
        self._managed_seen = {}
        self.sessions = {}
        self.logical_sessions = {}
        self.nodes = {}
        self.node_purge_quarantine = set()
        self._projected_node_purges = set()
        self.pool_configs = {}
        self.workspace_configs = WorkspaceConfigStore()
        self.workspaces = {}
        self._embedded_workspaces = {}
        self._explicit_pool_configs = {}
        self._pool_config_requires_matching_join = {}
        self.gpu_ledger = {}
        self._gpu_shared_admissions = {}
        self.allocations = {}
        self.command_assignments = {}
        self.node_command_queues = {}
        self.node_cancel_queues = {}
        self.node_control_queues = {}
        self.request_commands = {}
        self._started_runtime_sessions = set()
        self._request_seq = 0
        self._session_seq = 0
        self._node_seq = 0
        self._command_seq = 0
        self._reconcile_missing_observations = {}
        self._lost_cleanup_dispatched = set()

    @property
    def lease_table(self):
        """Compatibility projection for callers that predate shared leases.

        Scheduler decisions never consult this one-owner view. For a shared
        GPU it exposes the oldest current holder so old exclusive-only
        diagnostics retain their membership/empty-table behavior.
        """
        projected = {}
        for gpu_uuid, entry in self.gpu_ledger.items():
            holders = entry.holder_execution_session_ids
            if holders:
                projected[gpu_uuid] = holders[0]
        return projected

    def contention_policy_for_gpu(self, gpu_uuid):
        return self.policy.gpu_share_overrides.get(
            gpu_uuid,
            (
                self.policy.gpu_contention_policy,
                self.policy.gpu_shared_ratio,
            ),
        )

    @classmethod
    def reconcile_from_journal(cls, journal, clock=None, policy=None, metrics=None, runtime_adapters=None, auth=None):
        ctrl = cls(journal=journal, clock=clock, policy=policy, metrics=metrics, runtime_adapters=runtime_adapters, auth=auth)
        for event in journal.list_events():
            ctrl._replay_event(event)
        ctrl._mark_inflight_reconciling_after_replay()
        ctrl._rebuild_indexes_after_replay()
        if getattr(auth, "managed", False):
            from .managed_lifecycle import reconcile
            reconcile(ctrl)
        return ctrl

    async def submit_workload_async(self, spec, token=None):
        await asyncio.sleep(0)
        return self.submit_workload(spec, token=token)

    def submit_workload(self, spec, token=None):
        spec = self._resolve_profile(spec)
        self._authorize_submit(spec, token)
        if spec.resources.runtime_adapter != "local-process" and any(workspace_bind_conflict(mount.target, spec.workspace["target"]) for mount in spec.mounts):
            raise ValueError("managed workspace target conflicts with an explicit bind; select --workspace-target")
        if spec.workspace["id"] is not None:
            workspace = self.workspaces.get(spec.workspace["id"])
            if workspace is None:
                raise ValueError("workspace does not exist")
            if workspace["pool"] != spec.pool or workspace["user_id"] != spec.user_id:
                raise PermissionError("workspace belongs to another user or pool")
            owner = self.sessions.get(workspace["execution_session_id"])
            reserved_empty = owner is not None and owner.empty_retained and owner.matches_reuse_identity(spec, spec.image_identity(self.policy.default_image), owner.runtime_adapter)
            if workspace["state"] in ("deleted", "deleting", "needs_check", "needs_recovery") or (workspace["state"] == "absent" and not reserved_empty):
                raise ValueError("workspace is unavailable; recover it or explicitly choose another workspace")
        if spec.privileged and not self.policy.allow_privileged:
            raise PermissionError("privileged workloads are not allowed by policy")
        self._request_seq += 1
        request_id = "req-{:06d}".format(self._request_seq)
        now = self.clock()
        queued_quota_exceeded = self._queued_quota_exceeded(spec.user_id)
        state = RequestState.INFEASIBLE if queued_quota_exceeded else RequestState.QUEUED
        reason = "quota_exceeded" if queued_quota_exceeded else self._initial_queue_reason(spec)
        record = RequestRecord(
            request_id=request_id,
            spec=spec,
            state=state,
            reason=reason,
            created_at=now,
            updated_at=now,
            sequence=self._request_seq,
            ended_at=now if state == RequestState.INFEASIBLE else None,
        )
        if getattr(self.auth, "managed", False):
            record.admission_token_id = spec._admission_token_id
        self.requests[request_id] = record
        identity = (spec.user_id, spec.session_id)
        created_logical = identity not in self.logical_sessions
        if identity not in self.logical_sessions:
            logical = {"session_id": spec.session_id, "user_id": spec.user_id,
                       "created_at": now, "state": "alive"}
            self.logical_sessions[identity] = logical
        self._emit("request_submitted", request_id, RequestState.SUBMITTED.value, None, self._request_details(record))
        if created_logical:
            self._emit("logical_session_created", spec.session_id, "alive", None, self.logical_sessions[identity])
        if state == RequestState.INFEASIBLE:
            self._emit("request_state", request_id, state.value, reason, self._request_details(record))
        else:
            self._emit("request_queued", request_id, state.value, reason, self._request_details(record))
        return record

    def register_node(
        self,
        node_id,
        pool,
        resources,
        capabilities=None,
        runtime_capabilities=None,
        allocation_id=None,
        allocation_deadline=None,
        drain_before_sec=0,
        cleanup_buffer_sec=0,
        runtime_cache_config="default",
        cached_images=(),
        join_token=None,
        agent_epoch=None,
        recovering=False,
        state=NodeState.ACTIVE,
        pool_mode="exclusive",
        gpu_slots=1,
        placement="pack",
        workspace_status=None,
        agent_token=None,
    ):
        state = NodeState(state)
        if state not in (NodeState.REGISTERING, NodeState.ACTIVE):
            raise ValueError(
                "node registration state must be REGISTERING or ACTIVE"
            )
        agent_epoch = agent_epoch or None
        if getattr(self.auth, "managed", False):
            from .identity_nodes import registration
            enrollment = registration(self.auth.store, agent_token, node_id, pool, agent_epoch, allocation_id,
                                      {"cpu_cores": resources.cpu_cores, "memory_gb": resources.memory_gb, "gpu_uuids": list(resources.gpu_uuids)})
            if enrollment["desired"] == "removed" and node_id not in self.nodes:
                raise PermissionError("node enrollment is retired")
            if "effective_resources" in enrollment:
                resources = NodeResources(**enrollment["effective_resources"])
        elif self.auth is not None:
            if join_token is None:
                raise PermissionError("join token is required when auth is configured")
            self.auth.validate_join_token(join_token, pool, node_id, runtime_cache_config)
        requested_pool_config = PoolConfig(
            pool_mode=pool_mode,
            gpu_slots=gpu_slots,
            placement=placement,
        )
        if getattr(self.auth, "managed", False):
            expected = PoolConfig(**self.auth.store.snapshot()["pools"][pool].get("config", {}))
            if requested_pool_config != expected:
                raise PermissionError("node cannot change administrator pool configuration")
            self.pool_configs[pool] = expected
            self._pool_config_requires_matching_join[pool] = True
        established_pool_config = self.pool_configs.get(pool)
        pool_config_mismatch = bool(
            established_pool_config is not None
            and established_pool_config != requested_pool_config
        )
        pool_config_blockers = (
            self._pool_config_blockers(pool)
            if pool_config_mismatch
            else self._empty_pool_config_blockers()
        )
        requires_matching_join = bool(
            self._pool_config_requires_matching_join.get(pool, False)
        )
        if (
            pool_config_mismatch
            and (
                self._pool_config_has_blockers(pool_config_blockers)
                or requires_matching_join
            )
        ):
            raise self._join_pool_config_conflict(
                pool,
                established_pool_config,
                requested_pool_config,
                pool_config_blockers,
                requires_matching_join=requires_matching_join,
            )
        now = self.clock()
        previous_node = self.nodes.get(node_id)
        if previous_node is not None and previous_node.pool != pool:
            raise ValueError(
                "node {!r} already belongs to pool {!r}".format(
                    node_id, previous_node.pool
                )
            )
        previous_epoch = (
            getattr(previous_node, "agent_epoch", None)
            if previous_node is not None
            else None
        )
        same_allocation = bool(
            previous_node is not None
            and previous_node.allocation_id == allocation_id
        )
        if (
            previous_node is not None
            and same_allocation
            and previous_node.state == NodeState.REMOVED
        ):
            if pool_config_mismatch:
                raise self._join_pool_config_conflict(
                    pool,
                    established_pool_config,
                    requested_pool_config,
                    pool_config_blockers,
                    requires_matching_join=requires_matching_join,
                )
            return previous_node
        epoch_matches = bool(
            previous_node is not None
            and previous_epoch is not None
            and agent_epoch is not None
            and previous_epoch == agent_epoch
        )
        legacy_recent_retry = bool(
            previous_node is not None
            and previous_epoch is None
            and agent_epoch is None
            and same_allocation
            and previous_node.state
            in (NodeState.REGISTERING, NodeState.ACTIVE)
            and now - previous_node.last_heartbeat_at
            < self.policy.heartbeat_timeout_seconds
        )
        same_incarnation = bool(
            same_allocation and (epoch_matches or legacy_recent_retry)
        )
        epoch_changed = bool(
            previous_node is not None
            and previous_epoch != agent_epoch
            and (previous_epoch is not None or agent_epoch is not None)
        )
        heartbeat_gap_expired = bool(
            previous_node is not None
            and now - previous_node.last_heartbeat_at
            >= self.policy.heartbeat_timeout_seconds
        )
        genuine_recovery = bool(
            previous_node is not None
            and (
                recovering
                or previous_node.state
                in (NodeState.UNHEALTHY, NodeState.LOST)
                or not same_allocation
                or epoch_changed
                or heartbeat_gap_expired
            )
        )
        if (
            state == NodeState.REGISTERING
            and previous_node is not None
            and same_incarnation
            and previous_node.state != NodeState.REGISTERING
        ):
            previous_node.last_heartbeat_at = now
            return previous_node
        retained_sessions = sorted(
            (
                session
                for session in self.sessions.values()
                if session.node_id == node_id
                and session.state not in (SessionState.SUSPENDED, SessionState.PURGING)
                and (
                    session.state not in SESSION_TERMINAL_STATES
                    or self._session_holds_uncertain_gpu_lease(session)
                )
            ),
            key=lambda session: session.execution_session_id,
        )
        caps = capabilities
        if (
            state == NodeState.REGISTERING
            and caps is None
            and not runtime_capabilities
        ):
            caps = RuntimeCapabilities(
                adapter="probing",
                image_formats=(),
                isolation_policies=(),
                nested_container=False,
                can_bind_mounts=False,
                can_select_gpus=False,
                gpu_enforcement=EnforcementMode.NONE,
                can_enforce_cpu_memory=False,
                node_local_cache=False,
                supports_warm_sessions=False,
            )
        elif caps is None and not runtime_capabilities:
            caps = RuntimeCapabilities()
        registered_at = now
        if (
            previous_node is not None
            and same_incarnation
            and previous_node.state == state
        ):
            registered_at = previous_node.registered_at
        node = NodeRecord(
            node_id=node_id,
            pool=pool,
            resources=resources,
            capabilities=caps,
            runtime_capabilities=runtime_capabilities,
            state=state,
            registered_at=registered_at,
            last_heartbeat_at=now,
            allocation_id=allocation_id,
            allocation_deadline=allocation_deadline,
            drain_before_sec=drain_before_sec,
            cleanup_buffer_sec=cleanup_buffer_sec,
            runtime_cache_config=runtime_cache_config,
            cached_images=cached_images,
            used_cpu_cores=sum(
                session.cpu_cores for session in retained_sessions
            ),
            used_memory_gb=sum(
                session.memory_gb for session in retained_sessions
            ),
            active_sessions=(
                session.execution_session_id for session in retained_sessions
            ),
            agent_epoch=agent_epoch,
            pool_mode=requested_pool_config.pool_mode,
            gpu_slots=requested_pool_config.gpu_slots,
            placement=requested_pool_config.placement,
            # Direct state-machine registration is a trusted inventory declaration.
            # HTTP registration supplies an explicit unavailable value for legacy agents.
            workspace_status=workspace_status if workspace_status is not None else {
                "protocol": 1, "state": "applied", "error": None,
                "applied_version": self.workspace_configs.get(pool, node_id).version,
                "config": self.workspace_configs.get(pool, node_id).to_dict(),
            },
        )
        if previous_node is not None:
            node.last_status = getattr(previous_node, "last_status", {})
        self._validate_node_reregistration(
            node, previous_node, retained_sessions
        )
        existing_allocation = (
            self.allocations.get(allocation_id) if allocation_id else None
        )
        if existing_allocation is not None and existing_allocation.pool != pool:
            raise ValueError(
                "allocation {!r} belongs to pool {!r}, not {!r}".format(
                    allocation_id, existing_allocation.pool, pool
                )
            )
        if state == NodeState.ACTIVE:
            if (
                previous_node is not None
                and same_incarnation
                and previous_node.state
                in (NodeState.DRAINING, NodeState.REMOVING)
            ):
                node.state = previous_node.state
            elif node.should_drain(now):
                node.state = NodeState.DRAINING
        if established_pool_config is None:
            self.pool_configs[pool] = requested_pool_config
        elif pool_config_mismatch:
            self.pool_configs[pool] = requested_pool_config
            self._explicit_pool_configs[pool] = requested_pool_config
            self._pool_config_requires_matching_join[pool] = False
            self._emit(
                "pool_config_replaced",
                pool,
                None,
                "empty_pool_reestablished",
                {
                    "pool": pool,
                    "old_config": established_pool_config.to_dict(),
                    "new_config": requested_pool_config.to_dict(),
                    "joining_node_id": node_id,
                },
            )
        self.nodes[node_id] = node
        for gpu_uuid in node.resources.gpu_uuids:
            self.gpu_ledger.setdefault(
                gpu_uuid, PhysicalGpuLedgerEntry(gpu_uuid)
            )
            self._gpu_shared_admissions.setdefault(gpu_uuid, 0)
        self.node_command_queues.setdefault(node_id, [])
        self.node_cancel_queues.setdefault(node_id, [])
        self.node_control_queues.setdefault(node_id, [])
        if (
            previous_node is not None
            and previous_node.allocation_id
            and previous_node.allocation_id != allocation_id
        ):
            previous_allocation = self.allocations.get(
                previous_node.allocation_id
            )
            if (
                previous_allocation is not None
                and node_id in previous_allocation.node_ids
            ):
                previous_allocation.node_ids.remove(node_id)
                previous_allocation.updated_at = now
        if allocation_id:
            allocation = existing_allocation
            if allocation is None:
                allocation = AllocationRecord(allocation_id, pool, state=AllocationState.ACTIVE, node_ids=(), deadline=allocation_deadline, created_at=now, updated_at=now)
                self.allocations[allocation_id] = allocation
            if node_id not in allocation.node_ids:
                allocation.node_ids.append(node_id)
            allocation.deadline = allocation_deadline or allocation.deadline
            allocation.updated_at = now
        if state == NodeState.REGISTERING:
            event_type = (
                "node_announcement_refreshed"
                if previous_node is not None and same_incarnation
                else "node_announced"
            )
        elif previous_node is None or previous_node.state == NodeState.REGISTERING:
            event_type = "node_registered"
        elif genuine_recovery:
            event_type = "node_recovered"
        else:
            event_type = "node_refreshed"
        self._emit(
            event_type,
            node_id,
            node.state.value,
            None,
            self._node_details(node),
        )
        self._consume_pool_config_matching_join(
            pool, requested_pool_config
        )
        if previous_node is not None and genuine_recovery:
            self._mark_reregistered_sessions_reconciling(retained_sessions)
        return node

    @staticmethod
    def _empty_pool_config_blockers():
        return {
            "member_node_ids": [],
            "request_ids": [],
            "execution_session_ids": [],
        }

    @staticmethod
    def _pool_config_has_blockers(blockers):
        return any(blockers.values())

    def _pool_config_blockers(self, pool):
        return {
            "member_node_ids": sorted(
                node.node_id
                for node in self.nodes.values()
                if node.pool == pool and node.state != NodeState.REMOVED
            ),
            "request_ids": sorted(
                request.request_id
                for request in self.requests.values()
                if request.spec.pool == pool
                and request.state not in REQUEST_TERMINAL_STATES
            ),
            "execution_session_ids": sorted(
                session.execution_session_id
                for session in self.sessions.values()
                if session.pool == pool
                and (
                    session.state not in SESSION_TERMINAL_STATES | {SessionState.SUSPENDED, SessionState.PURGING}
                    or self._session_holds_uncertain_gpu_lease(session)
                )
            ),
        }

    def _join_pool_config_conflict(
        self,
        pool,
        established_config,
        requested_config,
        blockers,
        requires_matching_join=False,
    ):
        return PoolConfigConflictError(
            "pool {!r} configuration is {}, but join declared {}".format(
                pool,
                established_config,
                requested_config,
            ),
            pool,
            established_config,
            requested_config,
            blockers,
            requires_matching_join=requires_matching_join,
        )

    def _consume_pool_config_matching_join(self, pool, joined_config):
        if (
            self._pool_config_requires_matching_join.get(pool, False)
            and self.pool_configs.get(pool) == joined_config
        ):
            self._pool_config_requires_matching_join[pool] = False

    def pool_config_requires_matching_join(self, pool):
        return bool(
            self._pool_config_requires_matching_join.get(pool, False)
        )

    def _authorize_pool_admin(self, pool, token):
        if self.auth is None:
            return "local-admin"
        return self.auth.authenticate_user(token, pool, Role.ADMIN)

    def set_pool_config(
        self,
        pool,
        pool_mode,
        gpu_slots,
        placement,
        token=None,
    ):
        if getattr(self.auth, "managed", False):
            from .managed_lifecycle import edit_pool
            pool = self.auth.pool_id(pool)
            edit_pool(self, token, pool, {"config": {"pool_mode": pool_mode, "gpu_slots": gpu_slots, "placement": placement}})
            return self.pool_configs[pool]
        if not pool:
            raise ValueError("pool is required")
        actor = self._authorize_pool_admin(pool, token)
        requested_config = PoolConfig(
            pool_mode=pool_mode,
            gpu_slots=gpu_slots,
            placement=placement,
        )
        blockers = self._pool_config_blockers(pool)
        established_config = self.pool_configs.get(pool)
        if self._pool_config_has_blockers(blockers):
            raise PoolConfigConflictError(
                (
                    "pool {!r} cannot set configuration {} while it is "
                    "not reconfigurable-empty (members={}, requests={}, "
                    "sessions={})"
                ).format(
                    pool,
                    requested_config,
                    blockers["member_node_ids"],
                    blockers["request_ids"],
                    blockers["execution_session_ids"],
                ),
                pool,
                established_config,
                requested_config,
                blockers,
            )
        self.pool_configs[pool] = requested_config
        self._explicit_pool_configs[pool] = requested_config
        self._pool_config_requires_matching_join[pool] = True
        self._emit(
            "pool_config_set",
            pool,
            None,
            "operator_set",
            {
                "pool": pool,
                "old_config": (
                    established_config.to_dict()
                    if established_config is not None
                    else None
                ),
                "new_config": requested_config.to_dict(),
                "actor": actor,
            },
        )
        return requested_config

    def get_workspace_config(self, pool, node_id=None, token=None):
        if getattr(self.auth, "managed", False):
            pool = self.auth.pool_id(pool)
        self._authorize_pool_admin(pool, token)
        desired = self.workspace_configs.get(pool, node_id)
        if node_id in self.nodes and self.nodes[node_id].pool != pool:
            raise ValueError("workspace node override belongs to another pool")
        nodes = []
        for node in sorted(self.nodes.values(), key=lambda value: value.node_id):
            if node.pool != pool or (node_id is not None and node.node_id != node_id):
                continue
            reported = node.workspace_status
            applied = reported.get("applied_version")
            ready = reported.get("state") == "applied" and applied == desired.version
            nodes.append({"node_id": node.node_id, "desired_version": desired.version,
                          "applied_version": applied, "state": "applied" if ready else ("failed" if reported.get("error") else "pending"),
                          "error": reported.get("error"), "node_state": node.state.value,
                          "agent_epoch": node.agent_epoch, "last_heartbeat_at": node.last_heartbeat_at,
                          "applied_config": reported.get("config")})
        return {"pool": pool, "node_id": node_id, "desired_config": desired.to_dict(),
                "overrides": self.workspace_configs.record(pool), "nodes": nodes,
                "maintenance_defaults": MaintenancePolicy().to_dict()}

    def set_workspace_config(self, pool, changes, node_id=None, token=None):
        if getattr(self.auth, "managed", False):
            pool = self.auth.pool_id(pool)
        actor = self._authorize_pool_admin(pool, token)
        self.get_workspace_config(pool, node_id, token)
        candidate = copy.deepcopy(self.workspace_configs)
        record = candidate.update(pool, changes, node_id)
        self._emit("workspace_config_set", pool, None, "operator_set",
                   {"pool": pool, "new_config": record, "actor": actor})
        self.workspace_configs = candidate
        return self.get_workspace_config(pool, node_id, token)

    def workspace_status(self, workspace_id, token=None):
        workspace = self.workspaces.get(workspace_id)
        if workspace is None:
            raise KeyError("workspace does not exist")
        if getattr(self.auth, "managed", False):
            self.auth.require_owner(token, workspace["user_id"])
        elif self.auth is not None:
            pool = workspace["pool"]
            if not (self.auth.user_has_role(token, pool, Role.VIEWER) or self.auth.user_has_role(token, pool, Role.ADMIN)):
                user = self.auth.authenticate_user(token, pool, Role.SUBMITTER)
                if user != workspace["user_id"]:
                    raise PermissionError("cannot inspect another user's workspace")
        result = copy.deepcopy(workspace)
        result.setdefault("auto_reclaim", "inherit")
        policy = MaintenancePolicy(self.workspace_configs.get(workspace["pool"], workspace["node_id"]).maintenance)
        result["auto_reclaim_enabled"] = result["auto_reclaim"] == "inherit" and policy.to_dict()["reclaim_on_release"]
        return result

    def _reserve_workspace(self, spec, node_id, execution_session_id):
        identity = spec.workspace["id"]
        if identity is None:
            identity = "ws-" + uuid.uuid4().hex
            record = {"id": identity, "pool": spec.pool, "user_id": spec.user_id, "node_id": node_id,
                      "config": self.workspace_configs.get(spec.pool, node_id).to_dict(), "revision": 0,
                      "path": None, "last_error": None}
        else:
            record = copy.deepcopy(self.workspaces[identity])
            if record["state"] != "retained" or record["node_id"] != node_id or record.get("operation", {}).get("state") == "pending":
                raise ValueError("workspace is unavailable on the selected node")
        record.update(state="pending", execution_session_id=execution_session_id, target=spec.workspace["target"])
        self._emit("workspace_state", identity, "pending", None, copy.deepcopy(record))
        self.workspaces[identity] = record
        return dict(copy.deepcopy(record), create=spec.workspace["id"] is None)

    def workspace_operation(self, workspace_id, action, token=None, target_bytes=None, operation_id=None, auto_reclaim=None):
        self.workspace_status(workspace_id, token)
        record = copy.deepcopy(self.workspaces[workspace_id])
        if getattr(self.auth, "managed", False):
            self.auth.require_owner(token, record["user_id"])
        elif self.auth is not None and not self.auth.user_has_role(token, record["pool"], Role.ADMIN):
            user = self.auth.authenticate_user(token, record["pool"], Role.SUBMITTER)
            if user != record["user_id"]:
                raise PermissionError("cannot modify another user's workspace")
        if action == "set":
            if auto_reclaim not in ("off", "inherit") or target_bytes is not None or operation_id is not None:
                raise ValueError("set requires auto_reclaim off/inherit and no maintenance parameters")
            record["auto_reclaim"] = auto_reclaim
            record["reclaim_candidate"] = None
            if auto_reclaim == "off":
                record["reclaim_status"] = "workspace_disabled"
            else:
                record["reclaim_status"] = "awaiting_evaluation"
            self._emit("workspace_state", workspace_id, record["state"], "auto_reclaim_set", record)
            self.workspaces[workspace_id] = record
            return self.workspace_status(workspace_id, token)
        if auto_reclaim is not None:
            raise ValueError("auto_reclaim applies only to set")
        if action == "release":
            session = self.sessions.get(record["execution_session_id"])
            if session is not None and session.active_request_id:
                raise ValueError("cancel or finish the active command before releasing its workspace")
            if session is not None and record["state"] not in ("retained", "deleted"):
                self.stop_session(session.execution_session_id, reason="workspace_release", enforce_auth=False)
            return self.workspace_status(workspace_id, token)
        if action not in ("check", "delete", "grow", "reclaim", "shrink", "recover"):
            raise ValueError("unknown workspace operation")
        if action == "delete" and any(s.execution_session_id != record["execution_session_id"]
                and s.state != SessionState.PURGED and (s.workspace or {}).get("id") == workspace_id
                for s in self.sessions.values()):
            raise ValueError("workspace is referenced by another execution; purge that execution first")
        previous = record.get("operation", {})
        if operation_id is not None:
            if previous.get("id") != operation_id or previous.get("action") != action or previous.get("target_bytes") != target_bytes:
                raise ValueError("operation id is stale or parameters differ")
            return record
        if previous.get("state") == "pending":
            if previous.get("action") != action or previous.get("target_bytes") != target_bytes:
                raise ValueError("another workspace operation is pending")
            return record
        modifying = action in ("grow", "reclaim", "shrink", "recover")
        if modifying:
            node = self.nodes.get(record["node_id"])
            if node is None or node.workspace_status.get("maintenance_protocol") != 1:
                raise ValueError("node does not advertise verified workspace maintenance support")
            desired = self.workspace_configs.get(record["pool"], record["node_id"])
            if node.workspace_status.get("state") != "applied" or node.workspace_status.get("applied_version") != desired.version:
                raise ValueError("workspace maintenance configuration is not yet applied on node")
            if action != "recover":
                from .workspace_maintenance import validate_request
                try:
                    validate_request(action, target_bytes, self.workspace_configs.get(record["pool"], record["node_id"]).max_image_size_bytes)
                except RuntimeError as error:
                    raise ValueError(str(error))
                current = record.get("current_image_size_bytes", record["config"]["image_size_bytes"])
                if (action == "grow" and target_bytes <= current) or (action == "shrink" and target_bytes >= current):
                    raise ValueError("target must change capacity in the requested direction")
            elif target_bytes is not None:
                raise ValueError("recovery does not accept target capacity")
        elif target_bytes is not None:
            raise ValueError("this operation does not accept target capacity")
        owner = self.sessions.get(record["execution_session_id"])
        boundary = ((action == "grow" and record["state"] == "mounted")
                    or (action == "recover" and owner is not None and owner.state not in SESSION_TERMINAL_STATES | {SessionState.SUSPENDED}))
        if not boundary and record["state"] not in ("retained", "needs_check", "deleted", "maintenance", "needs_recovery"):
            raise ValueError("workspace must be released before offline maintenance")
        session = self.sessions.get(record["execution_session_id"])
        if not boundary and session is not None and session.state not in SESSION_TERMINAL_STATES | {SessionState.SUSPENDED}:
            raise ValueError("workspace still has a session owner")
        sequence = self._advance_workspace_operation(record)
        record["operation"] = {"id": uuid.uuid4().hex, "action": action, "state": "pending", "sequence": sequence,
                               "dispatch_granted": False, "requested_at": self.clock()}
        if modifying:
            if action != "recover" and record["state"] != "retained" and not boundary:
                raise ValueError("modifying maintenance requires a clean retained workspace")
            config = self.workspace_configs.get(record["pool"], record["node_id"])
            record["operation"].update(target_bytes=target_bytes, policy=config.maintenance or {}, source="explicit",
                                       boundary=boundary, requested_at=self.clock(), phase="waiting_boundary" if boundary else "requested")
            record["operation"]["max_image_size_bytes"] = config.max_image_size_bytes
            record["operation"]["policy_version"] = config.version
        self._emit("workspace_state", workspace_id, record["state"], "maintenance_requested", record)
        self.workspaces[workspace_id] = record
        return copy.deepcopy(record)

    def workspace_operations(self, node_id, ready=False):
        from .workspace_admission import dispatch
        return dispatch(self, node_id, ready)

    @staticmethod
    def _advance_workspace_operation(record):
        sequence = record.get("operation_sequence", 0)
        if type(sequence) is not int or not 0 <= sequence < 2 ** 63 - 1:
            raise ValueError("invalid/exhausted workspace operation sequence")
        record["operation_sequence"] = sequence + 1
        return sequence + 1

    def _automatic_workspace_operation(self, record, node):
        if node.workspace_status.get("maintenance_protocol") != 1 or record.get("operation", {}).get("state") == "pending":
            return
        config = self.workspace_configs.get(record["pool"], record["node_id"])
        if node.workspace_status.get("state") != "applied" or node.workspace_status.get("applied_version") != config.version:
            return
        policy = MaintenancePolicy(config.maintenance).to_dict()
        now = self.clock()
        try:
            growth = automatic_grow_target(record, config, now)
        except ValueError as error:
            record.update(maintenance_warning=str(error), last_maintenance_at=now)
            return
        action, target = ("grow", growth["target_bytes"]) if growth else (None, None)
        if action is None:
            if record.get("state") == "retained" and record.get("reclaim_release_generation") != record.get("usage_generation"):
                record.update(reclaim_release_generation=record.get("usage_generation"), reclaim_release_observed_at=now)
            candidate, reason = reclaim_candidate(record, policy, now)
            record["reclaim_candidate"] = candidate
            record["reclaim_status"] = reason
        if action:
            try:
                sequence = self._advance_workspace_operation(record)
            except ValueError as error:
                record["maintenance_warning"] = str(error)
                return
            record["operation"] = {"id": uuid.uuid4().hex, "action": action, "state": "pending", "sequence": sequence,
                                   "dispatch_granted": False,
                                   "target_bytes": target, "policy": policy, "source": "automatic",
                                   "boundary": record["state"] == "mounted", "requested_at": now,
                                   "phase": "waiting_boundary" if record["state"] == "mounted" else "requested",
                                   "max_image_size_bytes": config.max_image_size_bytes,
                                   "policy_version": config.version,
                                   "trigger": growth["reason"] if growth else "release"}
            record["last_maintenance_at"] = now

    def _workspace_blocks_session(self, session):
        if session.suspend_requested or session.state in (SessionState.SUSPENDING, SessionState.PURGING):
            return True
        workspace = self.workspaces.get((session.workspace or {}).get("id"), {})
        node = self.nodes.get(session.node_id)
        return (workspace.get("operation", {}).get("state") == "pending"
                or (workspace.get("operation", {}).get("boundary") and workspace["operation"].get("state") == "failed")
                or workspace.get("state") in ("maintenance", "needs_recovery")
                or (bool(workspace.get("maintenance")) and (node is None or node.workspace_status.get("maintenance_protocol") != 1)))

    def _failed_workspace_for_request(self, spec):
        candidates = []
        if spec.workspace["id"] is not None:
            candidates.append(self.workspaces.get(spec.workspace["id"], {}))
        if spec.reuse:
            for session in self.sessions.values():
                if session.state not in SESSION_TERMINAL_STATES and session.matches_reuse_identity(spec, spec.image_identity(self.policy.default_image), session.runtime_adapter):
                    candidates.append(self.workspaces.get((session.workspace or {}).get("id"), {}))
        for workspace in candidates:
            if workspace.get("node_retired"):
                return "workspace_node_unavailable"
            operation = workspace.get("operation", {})
            if operation.get("state") == "failed" and (operation.get("boundary") or workspace.get("state") in ("maintenance", "needs_recovery")):
                return operation.get("error") or workspace.get("last_error") or "workspace maintenance requires recovery"
        return None

    def apply_workspace_updates(self, node_id, updates, token=None, agent_epoch=None):
        node = self.nodes[node_id]
        self._authorize_node_agent(token, node.pool, node_id)
        if updates and not agent_epoch:
            raise PermissionError("workspace updates require the current agent epoch")
        self._validate_agent_epoch(node, agent_epoch)
        if not isinstance(updates, list) or len(updates) > 64:
            raise ValueError("workspace updates must be an array of at most 64 entries")
        acknowledgements = {}
        for update in updates:
            if not isinstance(update, dict):
                raise ValueError("workspace update must be an object")
            identity = update.get("workspace_id")
            if not isinstance(identity, str):
                raise ValueError("workspace update requires workspace_id")
            record = self.workspaces.get(identity)
            if record is None:
                continue
            if record["node_id"] != node_id or record["pool"] != node.pool:
                raise PermissionError("workspace update belongs to another node or pool")
            revision = update.get("revision")
            if type(revision) is not int or revision < 0 or revision >= 2 ** 63:
                raise ValueError("invalid workspace revision")
            if not isinstance(update.get("execution_session_id"), str):
                raise ValueError("workspace update requires execution_session_id")
            operation = record.get("operation", {})
            operation_result = update.get("operation_result")
            progress = update.get("operation_progress")
            if isinstance(progress, dict) and progress.get("id") != operation.get("id"):
                continue
            maintenance = isinstance(operation_result, dict) and operation.get("state") == "pending" and operation_result.get("id") == operation.get("id")
            if revision <= record["revision"] and not maintenance and not (revision == record["revision"] and record["state"] == "pending" and update.get("state") in ("absent", "needs_check", "retained")):
                if not operation_result and revision == record["revision"] and update.get("execution_session_id") == record["execution_session_id"] and update.get("user_id") == record["user_id"] and WorkspaceConfig.from_dict(update.get("config")).to_dict() == WorkspaceConfig.from_dict(record["config"]).to_dict():
                    record["space"] = copy.deepcopy(update.get("space"))
                    previous_operation = copy.deepcopy(record.get("operation"))
                    self._automatic_workspace_operation(record, node)
                    if record.get("operation") != previous_operation:
                        self._emit("workspace_state", identity, record["state"], "automatic_maintenance_requested", copy.deepcopy(record))
                acknowledgements[identity] = {"revision": record["revision"], "execution_session_id": update["execution_session_id"], "operation_id": (operation_result or {}).get("id"), "telemetry_id": update.get("telemetry_id")}
                if "telemetry_id" not in update:
                    acknowledgements[identity].pop("telemetry_id", None)
                continue
            if update.get("execution_session_id") != record["execution_session_id"] or update.get("user_id") != record["user_id"]:
                continue
            if WorkspaceConfig.from_dict(update.get("config")).to_dict() != WorkspaceConfig.from_dict(record["config"]).to_dict():
                raise ValueError("workspace creation configuration must not change")
            state = update.get("state")
            if state not in ("creating", "mounting", "mounted", "retained", "needs_check", "deleting", "deleted", "absent", "maintenance", "needs_recovery"):
                raise ValueError("invalid workspace lifecycle state")
            if record["state"] == "deleted" and state != "deleted":
                raise ValueError("deleted workspaces must not be resurrected")
            updated = copy.deepcopy(record)
            if isinstance(progress, dict) and operation.get("state") == "pending":
                updated["operation"]["phase"] = progress.get("phase", "applying")
            if maintenance:
                if revision < record["revision"] or operation_result.get("state") not in ("completed", "failed"):
                    raise ValueError("invalid workspace operation result")
                updated["operation"] = dict(operation, **operation_result)
                updated["last_maintenance_at"] = self.clock()
            updated.update(state=state, revision=revision, path=update.get("path"), last_error=update.get("last_error"))
            updated["space"] = copy.deepcopy(update.get("space"))
            updated["timings"] = copy.deepcopy(update.get("timings"))
            for field in ("current_image_size_bytes", "geometry", "generation", "maintenance", "usage_generation",
                          "reclaim_reference", "reclaim_sample", "reclaim_sample_error"):
                if field in update:
                    updated[field] = copy.deepcopy(update[field])
            updated["maintenance_deferred"] = bool(update.get("maintenance_deferred", False))
            if maintenance:
                from .workspace_admission import complete
                complete(updated, self.clock())
            if not maintenance:
                self._automatic_workspace_operation(updated, node)
            self._emit("workspace_state", identity, state, None, updated)
            self.workspaces[identity] = updated
            session = self.sessions.get(updated["execution_session_id"])
            if maintenance and session is not None:
                session.last_used_at = self.clock()
            if state == "retained" and session is not None and session.state in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE):
                self.stop_session(session.execution_session_id, reason="workspace_unmounted", enforce_auth=False)
            acknowledgements[identity] = {"revision": revision, "execution_session_id": update["execution_session_id"], "operation_id": (operation_result or {}).get("id"), "telemetry_id": update.get("telemetry_id")}
            if "telemetry_id" not in update:
                acknowledgements[identity].pop("telemetry_id", None)
        return acknowledgements

    @staticmethod
    def _validate_node_reregistration(node, previous_node, sessions):
        if not sessions:
            return
        if previous_node is not None:
            if previous_node.pool != node.pool:
                raise ValueError(
                    "node {!r} cannot change pool while sessions remain".format(
                        node.node_id
                    )
                )
            if previous_node.allocation_id != node.allocation_id:
                raise ValueError(
                    "node {!r} cannot change allocation while sessions remain".format(
                        node.node_id
                    )
                )
        if any(session.pool != node.pool for session in sessions):
            raise ValueError(
                "node {!r} has retained sessions from another pool".format(
                    node.node_id
                )
            )
        if node.used_cpu_cores > node.resources.cpu_cores:
            raise ValueError(
                "node {!r} re-registration has insufficient CPU for retained sessions".format(
                    node.node_id
                )
            )
        if node.used_memory_gb > node.resources.memory_gb:
            raise ValueError(
                "node {!r} re-registration has insufficient memory for retained sessions".format(
                    node.node_id
                )
            )
        available_gpus = set(node.resources.gpu_uuids)
        for session in sessions:
            if session.gpu_count > len(available_gpus):
                raise ValueError(
                    "node {!r} re-registration has insufficient GPUs for session {!r}".format(
                        node.node_id, session.execution_session_id
                    )
                )
            if not set(session.gpu_uuids).issubset(available_gpus):
                raise ValueError(
                    "node {!r} re-registration omits a leased GPU for session {!r}".format(
                        node.node_id, session.execution_session_id
                    )
                )

    def _mark_reregistered_sessions_reconciling(self, sessions):
        for session in sessions:
            if session.state in (
                SessionState.CACHED_IDLE,
                SessionState.WARM_IDLE,
                SessionState.LOST,
                SessionState.RECONCILING,
                SessionState.SUSPENDING,
                SessionState.SUSPENDED,
                SessionState.PURGING,
            ):
                continue
            session.state = SessionState.RECONCILING
            self._reconcile_missing_observations.pop(
                session.execution_session_id, None
            )
            self._emit(
                "session_state",
                session.execution_session_id,
                session.state.value,
                "node_reregistered",
                self._session_details(session),
            )
            request_id = session.active_request_id
            request = self.requests.get(request_id)
            if request is None or request.state in REQUEST_TERMINAL_STATES:
                continue
            request.state = RequestState.RECONCILING
            request.reason = "node_reregistered"
            request.updated_at = self.clock()
            self._emit(
                "request_state",
                request.request_id,
                request.state.value,
                request.reason,
                self._request_details(request),
            )

    def heartbeat(
        self, node_id, state=None, token=None, agent_epoch=None
    ):
        node = self.nodes[node_id]
        self._authorize_node_agent(token, node.pool, node.node_id)
        self._validate_agent_epoch(node, agent_epoch)
        old_state = node.state
        if getattr(self.auth, "managed", False):
            if state not in (None, NodeState.REGISTERING, NodeState.UNHEALTHY):
                raise PermissionError("agent cannot set lifecycle state")
            if old_state == NodeState.DRAINING:
                state = None
        node.last_heartbeat_at = self.clock()
        if old_state == NodeState.REMOVED:
            return node
        if state is not None:
            node.state = state
        elif old_state == NodeState.UNHEALTHY:
            allocation = (
                self.allocations.get(node.allocation_id)
                if node.allocation_id
                else None
            )
            if allocation is None or allocation.state == AllocationState.ACTIVE:
                node.state = NodeState.ACTIVE
        if old_state != node.state:
            reason = (
                "heartbeat_recovered"
                if old_state == NodeState.UNHEALTHY
                and node.state == NodeState.ACTIVE
                else None
            )
            self._emit(
                "node_state",
                node_id,
                node.state.value,
                reason,
                self._node_details(node),
            )
        return node

    def _validate_agent_epoch(self, node, agent_epoch):
        if getattr(self.auth, "managed", False) and not agent_epoch:
            raise PermissionError("node epoch required")
        if (
            agent_epoch is not None
            and getattr(node, "agent_epoch", None) is not None
            and agent_epoch != node.agent_epoch
        ):
            raise PermissionError("agent epoch does not match current node process")

    def node_heartbeat(
        self,
        node_id,
        state=None,
        status=None,
        token=None,
        max_commands=8,
        agent_epoch=None,
    ):
        node = self.heartbeat(
            node_id,
            state=state,
            token=token,
            agent_epoch=agent_epoch,
        )
        workspace_updates = (status or {}).get("workspace_updates", [])
        workspace_acks = self.apply_workspace_updates(node_id, workspace_updates, token, agent_epoch)
        workspace_response = {"workspace_acks": workspace_acks,
                              "workspace_operations": [],
                              "workspace_config": self.workspace_configs.get(node.pool, node_id).to_dict()}
        control = self.agent_control(node.node_id)
        if control is not None:
            return {
                "node": self.node_status(node.node_id, token=token),
                "commands": self.dequeue_node_response_commands(
                    node_id,
                    max_commands=max_commands,
                    allow_assignments=False,
                ),
                "agent_control": control,
                **workspace_response,
            }
        if isinstance(status, dict):
            node.last_status = status
            if getattr(self.auth, "managed", False):
                self._managed_seen[node_id] = self.clock()
            if isinstance(status.get("workspace_config"), dict):
                node.workspace_status = dict(status["workspace_config"])
        self._resolve_reconciling_from_node_status(node_id, status or {})
        workspace_response["workspace_operations"] = self.workspace_operations(node_id, (status or {}).get("reclaim_ready") is True)
        return {
            "node": self.node_status(node.node_id, token=token),
            **workspace_response,
            "commands": self.dequeue_node_response_commands(
                node_id,
                max_commands=max_commands,
            ),
        }

    def agent_control(self, node_id):
        node = self.nodes.get(node_id)
        if node is None or node.state != NodeState.REMOVED:
            return None
        reason = "node_removed"
        allocation = (
            self.allocations.get(node.allocation_id)
            if node.allocation_id
            else None
        )
        if allocation is not None and allocation.state == AllocationState.REMOVED:
            reason = "allocation_removed"
        return {
            "authoritative": True,
            "action": "shutdown",
            "reason": reason,
            "node_id": node.node_id,
            "allocation_id": node.allocation_id,
        }

    def dequeue_node_response_commands(
        self,
        node_id,
        max_commands=8,
        allow_assignments=True,
    ):
        control = self.agent_control(node_id)
        payloads = []
        if control is not None:
            self.node_command_queues.setdefault(node_id, [])[:] = []
        elif allow_assignments:
            commands = self._dequeue_node_commands(
                node_id,
                max_commands=max_commands,
            )
            payloads.extend(
                self.command_assignment_details(command)
                for command in commands
            )
        payloads.extend(self._dequeue_node_cancel_commands(node_id))
        payloads.extend(self._dequeue_node_control_commands(node_id))
        return payloads

    def _dequeue_node_commands(self, node_id, max_commands=8):
        queue = self.node_command_queues.setdefault(node_id, [])
        commands = []
        while queue and len(commands) < max_commands:
            command_id = queue.pop(0)
            assignment = self.command_assignments.get(command_id)
            if assignment is None or assignment.state != CommandState.PENDING:
                continue
            assignment.state = CommandState.RUNNING
            assignment.updated_at = self.clock()
            commands.append(assignment)
            self._emit("command_state", command_id, assignment.state.value, None, self.command_assignment_details(assignment))
        return commands


    def _dequeue_node_cancel_commands(self, node_id):
        queue = self.node_cancel_queues.setdefault(node_id, [])
        payloads = []
        while queue:
            command_id = queue.pop(0)
            assignment = self.command_assignments.get(command_id)
            if assignment is None:
                continue
            payload = self.command_assignment_details(assignment)
            payload["action"] = "cancel"
            payloads.append(payload)
        return payloads

    def _dequeue_node_control_commands(self, node_id):
        queue = self.node_control_queues.setdefault(node_id, [])
        payloads = list(queue)
        del queue[:]
        for payload in payloads:
            if payload.get("action") != "stop_session":
                continue
            execution_session_id = payload.get("execution_session_id")
            session = self.sessions.get(execution_session_id)
            if (
                session is not None
                and session.node_id == node_id
                and self._session_holds_uncertain_gpu_lease(session)
            ):
                self._lost_cleanup_dispatched.add(execution_session_id)
        return payloads

    def report_command_result(self, node_id, command_id, exit_code=0, stdout="", stderr="", failure_kind="workload", log_path=None, artifact_paths=(), token=None, agent_epoch=None):
        if failure_kind not in COMMAND_FAILURE_KINDS:
            raise ValueError(
                "failure_kind must be one of: "
                + ", ".join(COMMAND_FAILURE_KINDS)
            )
        if failure_kind == "infrastructure" and exit_code == 0:
            raise ValueError(
                "infrastructure failure_kind requires a nonzero exit_code"
            )
        node = self.nodes[node_id]
        self._authorize_node_agent(token, node.pool, node.node_id)
        self._validate_agent_epoch(node, agent_epoch)
        assignment = self.command_assignments[command_id]
        if assignment.node_id != node_id:
            raise PermissionError("command is not assigned to this node")
        request = self.requests[assignment.request_id]
        if request.reason == "assignment_missing" and request.state in REQUEST_TERMINAL_STATES:
            return self._request_status_record(request)
        if request.state == RequestState.CANCELLED or assignment.state == CommandState.CANCELLED:
            cancel_reason = (
                assignment.result.get("reason")
                or request.reason
                or "cancelled"
            )
            assignment.state = CommandState.CANCELLED
            assignment.updated_at = self.clock()
            assignment.log_path = log_path or assignment.log_path
            assignment.result = {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "failure_kind": failure_kind, "artifact_paths": list(artifact_paths or ()), "reason": cancel_reason}
            self._emit("command_state", command_id, assignment.state.value, cancel_reason, self.command_assignment_details(assignment))
            return self._request_status_record(request)
        assignment.state = CommandState.COMPLETED if exit_code == 0 else CommandState.FAILED
        assignment.updated_at = self.clock()
        assignment.log_path = log_path or assignment.log_path
        assignment.result = {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "failure_kind": failure_kind, "artifact_paths": list(artifact_paths or ())}
        if log_path:
            request.log_path = log_path
        if artifact_paths:
            request.artifact_paths = tuple(artifact_paths)
        if request.state in REQUEST_TERMINAL_STATES:
            self._emit(
                "command_state",
                command_id,
                assignment.state.value,
                "request_already_terminal",
                self.command_assignment_details(assignment),
            )
            return self._request_status_record(request)
        if request.state in (RequestState.RUNNING, RequestState.RECONCILING):
            # A scoped terminal report resolves reconciliation and must still
            # execute the normal session/lease cleanup path.
            request.state = RequestState.RUNNING
            self.complete_request(
                request.request_id,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                failure_kind=failure_kind,
            )
        else:
            request.exit_code = exit_code
            request.failure_kind = failure_kind
            request.stdout = stdout
            request.stderr = stderr
            self._mark_request_terminal(
                request,
                RequestState.COMPLETED
                if exit_code == 0
                else RequestState.FAILED,
            )
            request.reason = (
                None
                if exit_code == 0
                else (
                    "command_failed"
                    if failure_kind == "workload"
                    else "infrastructure_failed"
                )
            )
            self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
        self._emit("command_state", command_id, assignment.state.value, None, self.command_assignment_details(assignment))
        return self._request_status_record(request)

    def cancel_command(self, command_id, reason="cancelled"):
        assignment = self.command_assignments.get(command_id)
        if assignment is None:
            return None
        if assignment.state == CommandState.RUNNING:
            queue = self.node_cancel_queues.setdefault(assignment.node_id, [])
            if command_id not in queue:
                queue.append(command_id)
        was_dispatched = assignment.result.get("was_dispatched", True) if assignment.state == CommandState.CANCELLED else assignment.state != CommandState.PENDING
        assignment.state = CommandState.CANCELLED
        assignment.updated_at = self.clock()
        assignment.result = {"reason": reason, "was_dispatched": was_dispatched}
        self._emit("command_state", command_id, assignment.state.value, reason, self.command_assignment_details(assignment))
        return assignment

    def refresh_deadlines(self):
        now = self.clock()
        for node in self.nodes.values():
            if node.state == NodeState.ACTIVE and node.should_drain(now):
                self._transition_node_to_draining(
                    node, "allocation_expiring"
                )
            if node.allocation_deadline is not None and now >= node.allocation_deadline and node.state not in (NodeState.REMOVED, NodeState.LOST):
                node.state = NodeState.REMOVING
                self._emit("node_state", node.node_id, node.state.value, "allocation_deadline", self._node_details(node))

    def check_heartbeats(self):
        now = self.clock()
        for node in self.nodes.values():
            if node.state in (NodeState.REMOVED, NodeState.LOST):
                continue
            age = now - node.last_heartbeat_at
            if age >= self.policy.lost_timeout_seconds:
                node.state = NodeState.LOST
                self._mark_sessions_lost_for_node(node.node_id)
                self._emit("node_state", node.node_id, node.state.value, "heartbeat_lost", self._node_details(node))
            elif age >= self.policy.heartbeat_timeout_seconds and node.state == NodeState.ACTIVE:
                node.state = NodeState.UNHEALTHY
                self._emit("node_state", node.node_id, node.state.value, "heartbeat_timeout", self._node_details(node))

    def schedule_once(self):
        if getattr(self.auth, "managed", False):
            from .managed_lifecycle import reconcile
            reconcile(self)
        self.refresh_deadlines()
        self.expire_max_lifetime_sessions()
        scheduled = 0
        for request in self._ordered_queue():
            if request.state != RequestState.QUEUED:
                continue
            if getattr(self.auth, "managed", False) and not self.auth.store.pending_allowed(
                    request.spec.user_id, request.spec.pool, getattr(request, "admission_token_id", None)):
                request.reason = "permission_revoked"
                request.failure_kind = "workload"
                self._mark_request_terminal(request, RequestState.FAILED)
                self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
                continue
            maintenance_error = self._failed_workspace_for_request(request.spec)
            if maintenance_error:
                request.reason = "workspace_node_unavailable" if maintenance_error == "workspace_node_unavailable" else "workspace_maintenance_failed"
                request.failure_kind = "infrastructure"
                request.stderr = str(maintenance_error)
                self._mark_request_terminal(request, RequestState.FAILED)
                self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
                continue
            if self._resume_retained_execution(request):
                if request.execution_session_id is not None and request.state != RequestState.QUEUED:
                    scheduled += 1
                continue
            reused = self._find_reusable_session(request.spec)
            if reused is not None:
                gpu_uuids = self._choose_reuse_gpus(reused, request)
                if gpu_uuids is None:
                    self._set_queued_reason(request, "waiting_for_gpu")
                    continue
                self._lease_gpus_to_session(reused.execution_session_id, gpu_uuids)
                self._start_request_on_session(request, reused, gpu_uuids, reused.node_id)
                scheduled += 1
                continue
            if self._find_matching_active_session(request.spec) is not None:
                self._set_queued_reason(request, "waiting_for_session")
                continue
            if self._would_exceed_active_gpu_quota(request.spec):
                self._set_queued_reason(request, "quota_exceeded")
                continue
            if self._would_exceed_pool_session_cap(request.spec):
                self._set_queued_reason(request, "pool_session_limit")
                continue
            placement = self._choose_placement(request)
            while not placement.placed and placement.reason == "waiting_for_cpu_or_memory":
                evicted_execution_session_id = self._evict_cached_idle_session_for_request(request.spec)
                if evicted_execution_session_id is None:
                    break
                placement = self._choose_placement(request)
            if not placement.placed:
                self._set_queued_reason(
                    request,
                    placement.reason or "no_capable_node",
                )
                continue
            session = self._create_session(
                request,
                placement.node_id,
                placement.gpu_uuids,
                runtime_adapter=placement.runtime_adapter,
            )
            self._start_request_on_session(request, session, placement.gpu_uuids, placement.node_id)
            scheduled += 1
        return scheduled

    def _set_queued_reason(self, request, reason):
        if request.reason == reason:
            return False
        request.reason = reason
        request.updated_at = self.clock()
        self._emit(
            "request_state",
            request.request_id,
            request.state.value,
            request.reason,
            self._request_details(request),
        )
        return True

    def _mark_request_running(self, request):
        now = self.clock()
        request.state = RequestState.RUNNING
        if request.started_at is None:
            request.started_at = now
        request.updated_at = now

    def _mark_request_terminal(self, request, state, now=None):
        now = self.clock() if now is None else now
        request.state = state
        request.updated_at = now
        if request.ended_at is None:
            request.ended_at = now

    def complete_request(
        self,
        request_id,
        exit_code=0,
        stdout="",
        stderr="",
        failure_kind="workload",
    ):
        if failure_kind not in COMMAND_FAILURE_KINDS:
            raise ValueError(
                "failure_kind must be one of: "
                + ", ".join(COMMAND_FAILURE_KINDS)
            )
        if failure_kind == "infrastructure" and exit_code == 0:
            raise ValueError(
                "infrastructure failure_kind requires a nonzero exit_code"
            )
        request = self.requests[request_id]
        if request.state != RequestState.RUNNING:
            raise ValueError("only running requests can complete")
        request.exit_code = exit_code
        request.failure_kind = failure_kind
        request.stdout = stdout
        request.stderr = stderr
        self._mark_request_terminal(
            request,
            RequestState.COMPLETED if exit_code == 0 else RequestState.FAILED,
        )
        request.reason = (
            None
            if exit_code == 0
            else (
                "command_failed"
                if failure_kind == "workload"
                else "infrastructure_failed"
            )
        )
        if request.execution_session_id:
            session = self.sessions[request.execution_session_id]
            session.active_request_id = None
            session.last_used_at = self.clock()
            node = self.nodes.get(session.node_id)
            node_is_draining = (
                node is not None
                and node.state
                in (
                    NodeState.DRAINING,
                    NodeState.REMOVING,
                    NodeState.REMOVED,
                )
            )
            session_unhealthy = failure_kind == "infrastructure"
            keep_after_outcome = (
                exit_code == 0
                or request.spec.keep_session_on_failure
            )
            if (
                request.spec.reuse
                and not node_is_draining
                and not session_unhealthy
                and keep_after_outcome
            ):
                self._release_gpu_lease(session.execution_session_id)
                session.state = SessionState.CACHED_IDLE
                self._emit("session_state", session.execution_session_id, session.state.value, None, self._session_details(session))
                if session.suspend_requested:
                    self.suspend_session(session.execution_session_id, enforce_auth=False)
            elif not session_unhealthy:
                session.state = SessionState.STOPPED
                self._cleanup_runtime_session(session.execution_session_id)
                self._release_session(session.execution_session_id)
                completion_reason = (
                    "command_complete_node_draining"
                    if node_is_draining
                    else (
                        "command_failed_suspend_session"
                        if exit_code != 0
                        else "command_complete_no_reuse"
                    )
                )
                self._emit("session_state", session.execution_session_id, session.state.value, completion_reason, self._session_details(session))
            else:
                session.state = SessionState.FAILED
                self._cleanup_runtime_session(session.execution_session_id)
                self._release_session(session.execution_session_id)
                self._emit("session_state", session.execution_session_id, session.state.value, request.reason, self._session_details(session))
        self._emit("request_state", request_id, request.state.value, request.reason, self._request_details(request))
        return request

    def cancel_request(self, request_id, reason="cancelled", token=None, enforce_auth=True):
        request = self.requests[request_id]
        if enforce_auth:
            self._authorize_request_cancel(request, token)
        if request.state in REQUEST_TERMINAL_STATES:
            return request
        self._mark_request_terminal(request, RequestState.CANCELLED)
        request.reason = reason
        command_id = self.request_commands.get(request_id)
        if command_id:
            self.cancel_command(command_id, reason=reason)
        if request.execution_session_id:
            self.stop_session(request.execution_session_id, reason=reason, enforce_auth=False)
        self._emit("request_state", request_id, request.state.value, reason, self._request_details(request))
        return request

    def stop_session(self, execution_session_id, reason="stopped", token=None, enforce_auth=True):
        session = self.sessions[execution_session_id]
        if enforce_auth:
            self._authorize_session_operator(session, token)
        session.state = SessionState.STOPPED
        session.active_request_id = None
        session.last_used_at = self.clock()
        self._cleanup_runtime_session(execution_session_id)
        self._release_session(execution_session_id)
        self._emit("session_state", execution_session_id, session.state.value, reason, self._session_details(session))
        return session

    def expire_idle_sessions(self):
        now = self.clock()
        expired = []
        for session in list(self.sessions.values()):
            if self.workspaces.get((session.workspace or {}).get("id"), {}).get("operation", {}).get("state") == "pending":
                continue
            if session.state in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE) and now - session.last_used_at >= self.policy.idle_session_timeout_seconds:
                self.suspend_session(session.execution_session_id, reason="idle_timeout", enforce_auth=False)
                expired.append(session.execution_session_id)
        return expired

    def suspend_session(self, execution_session_id, reason="operator_suspend", token=None, enforce_auth=True):
        session = self.sessions[execution_session_id]
        if enforce_auth:
            self._authorize_session_operator(session, token)
        if session.state in (SessionState.SUSPENDING, SessionState.SUSPENDED):
            return session
        if session.state not in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE):
            raise ValueError("execution session is busy; wait for its command before suspension")
        session.state = SessionState.SUSPENDING
        session.suspend_requested = False
        session.suspend_version = getattr(session, "suspend_version", 0) + 1
        session.last_used_at = self.clock()
        self._emit("session_state", execution_session_id, session.state.value, reason, self._session_details(session))
        cleaned = self._cleanup_runtime_session(execution_session_id, preserve=True)
        if cleaned and session.runtime_adapter in self.runtime_adapters:
            self._finish_suspension(session)
        return session

    def _finish_suspension(self, session):
        if session.state != SessionState.SUSPENDING:
            return
        self._release_session(session.execution_session_id)
        session.state = SessionState.SUSPENDED
        self._emit("session_state", session.execution_session_id, session.state.value, "suspended", self._session_details(session))

    def _suspend_unstarted(self, session):
        self._release_session(session.execution_session_id)
        session.empty_retained = True
        session.suspend_requested = False
        session.resume_required = False
        session.state = SessionState.SUSPENDED
        workspace = self.workspaces[session.workspace["id"]]
        workspace["state"] = "absent" if session.workspace.get("create") else "retained"
        self._emit("workspace_state", workspace["id"], workspace["state"], None, copy.deepcopy(workspace))
        self._emit("session_state", session.execution_session_id, session.state.value, "suspended", self._session_details(session))

    def expire_max_lifetime_sessions(self):
        now = self.clock()
        expired = []
        for session in list(self.sessions.values()):
            if session.state in SESSION_TERMINAL_STATES or session.state in (SessionState.SUSPENDING, SessionState.SUSPENDED):
                continue
            if now - session.activated_at < session.max_duration_seconds:
                continue
            active_request_id = session.active_request_id
            if active_request_id and active_request_id in self.requests:
                request = self.requests[active_request_id]
                self._mark_request_terminal(
                    request, RequestState.EXPIRED, now=now
                )
                request.reason = "max_lifetime"
                request.exit_code = None
                self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
            self.stop_session(session.execution_session_id, reason="max_lifetime", enforce_auth=False)
            expired.append(session.execution_session_id)
        return expired

    def drain_node(self, node_id, reason="operator_drain", token=None):
        if getattr(self.auth, "managed", False):
            from .managed_lifecycle import node_action
            node_action(self, token, node_id, "drain", allow_agent=True)
            return self.nodes[node_id]
        self._authorize_operator_for_node(node_id, token)
        node = self.nodes[node_id]
        return self._transition_node_to_draining(node, reason)

    def _transition_node_to_draining(self, node, reason):
        node.state = NodeState.DRAINING
        self._emit(
            "node_state",
            node.node_id,
            node.state.value,
            reason,
            self._node_details(node),
        )
        for execution_session_id in list(node.active_sessions):
            session = self.sessions.get(execution_session_id)
            if (
                session is not None
                and session.state
                in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE)
                and not (getattr(self.auth, "managed", False) and any(
                    command.execution_session_id == execution_session_id and command.state in (CommandState.PENDING, CommandState.RUNNING)
                    for command in self.command_assignments.values()))
            ):
                self.stop_session(
                    execution_session_id,
                    reason=reason,
                    enforce_auth=False,
                )
        return node

    def remove_node(self, node_id, force=False, token=None):
        if getattr(self.auth, "managed", False):
            from .managed_lifecycle import node_action
            node_action(self, token, node_id, "drain" if force else "exit", force=force)
            return self.nodes[node_id]
        self._authorize_operator_for_node(node_id, token)
        node = self.nodes[node_id]
        if node.active_sessions and not force:
            node.state = NodeState.REMOVING
            self._emit("node_state", node_id, node.state.value, "remove_pending_sessions", self._node_details(node))
            return node
        for execution_session_id in list(node.active_sessions):
            if execution_session_id in self.sessions:
                self.stop_session(execution_session_id, reason="node_removed", enforce_auth=False)
        node.state = NodeState.REMOVED
        self._emit("node_state", node_id, node.state.value, "node_removed", self._node_details(node))
        return node

    def get_request(self, request_id, token=None):
        request = self.requests[request_id]
        self._authorize_request_read(request, token)
        return request

    def get_session(self, execution_session_id, token=None):
        session = self.sessions[execution_session_id]
        self._authorize_session_read(session, token)
        return session

    def get_node(self, node_id, token=None):
        node = self.nodes[node_id]
        self._authorize_node_read(node, token)
        return node

    def active_gpu_lease_count(self, user_id=None):
        count = 0
        for entry in self.gpu_ledger.values():
            for execution_session_id in entry.holder_execution_session_ids:
                session = self.sessions.get(execution_session_id)
                if session is None:
                    continue
                if user_id is not None and session.user_id != user_id:
                    continue
                count += 1
        return count

    def assert_invariants(self):
        for pool, config in self.pool_configs.items():
            for node in self.nodes.values():
                if (
                    node.pool != pool
                    or node.state == NodeState.REMOVED
                ):
                    continue
                node_config = PoolConfig(
                    node.pool_mode, node.gpu_slots, node.placement
                )
                if node_config != config:
                    raise AssertionError(
                        "node pool configuration disagrees with pool"
                    )
        for gpu_uuid, entry in self.gpu_ledger.items():
            if (
                entry.exclusive_execution_session_id is not None
                and entry.shared_execution_session_ids
            ):
                raise AssertionError(
                    "GPU has exclusive and shared holders: " + gpu_uuid
                )
            if len(set(entry.shared_execution_session_ids)) != len(
                entry.shared_execution_session_ids
            ):
                raise AssertionError(
                    "duplicate shared GPU holder: " + gpu_uuid
                )
            if entry.shared_execution_session_ids:
                if entry.shared_capacity is None:
                    raise AssertionError(
                        "shared GPU is missing its capacity: " + gpu_uuid
                    )
                if entry.shared_refcount > entry.shared_capacity:
                    raise AssertionError(
                        "shared GPU exceeds capacity: " + gpu_uuid
                    )
            elif entry.shared_capacity is not None:
                raise AssertionError(
                    "free/exclusive GPU retained shared capacity: "
                    + gpu_uuid
                )
            for execution_session_id in entry.holder_execution_session_ids:
                if execution_session_id not in self.sessions:
                    raise AssertionError(
                        "GPU ledger points to unknown session"
                    )
                session = self.sessions[execution_session_id]
                uncertain_lease = (
                    self._session_holds_uncertain_gpu_lease(session)
                )
                if gpu_uuid not in session.gpu_uuids:
                    raise AssertionError(
                        "GPU ledger disagrees with session GPU set"
                    )
                if (
                    session.state in SESSION_TERMINAL_STATES
                    and not uncertain_lease
                ) or session.state in (
                    SessionState.CACHED_IDLE,
                    SessionState.WARM_IDLE,
                ):
                    raise AssertionError(
                        "non-running session retains GPU lease"
                    )
                if (
                    entry.state == GpuLeaseState.EXCLUSIVE
                    and session.gpu_mode != "exclusive"
                ):
                    raise AssertionError(
                        "exclusive ledger holder has shared mode"
                    )
                if (
                    entry.state == GpuLeaseState.SHARED
                    and session.gpu_mode != "shared"
                ):
                    raise AssertionError(
                        "shared ledger holder has exclusive mode"
                    )
        for session in self.sessions.values():
            if session.state in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE) and session.active_request_id is not None:
                raise AssertionError("cached idle session has active request")
            should_hold = (
                self._session_holds_uncertain_gpu_lease(session)
                or (
                    session.state not in SESSION_TERMINAL_STATES
                    and session.state
                    not in (
                        SessionState.CACHED_IDLE,
                        SessionState.WARM_IDLE,
                    )
                )
            )
            for gpu_uuid in session.gpu_uuids:
                entry = self.gpu_ledger.get(gpu_uuid)
                if (
                    should_hold
                    and (
                        entry is None
                        or session.execution_session_id
                        not in entry.holder_execution_session_ids
                    )
                ):
                    raise AssertionError(
                        "session GPU is absent from physical ledger"
                    )
        for request in self.requests.values():
            if request.state == RequestState.QUEUED and (request.execution_session_id or request.gpu_uuids):
                raise AssertionError("queued request holds hardware lease")
            if request.state == RequestState.RUNNING:
                if request.execution_session_id is None:
                    raise AssertionError("running request missing session")
                session = self.sessions[request.execution_session_id]
                if session.active_request_id != request.request_id:
                    raise AssertionError("running request not linked from session")
        for node in self.nodes.values():
            seen_sessions = set()
            for execution_session_id in node.active_sessions:
                if execution_session_id in seen_sessions:
                    raise AssertionError("duplicate session on node: " + execution_session_id)
                seen_sessions.add(execution_session_id)
                if execution_session_id not in self.sessions:
                    raise AssertionError("node points to unknown session")
                if self.sessions[execution_session_id].node_id != node.node_id:
                    raise AssertionError("node/session ownership mismatch")

    def _ordered_queue(self):
        queued = [r for r in self.requests.values() if r.state == RequestState.QUEUED]
        return sorted(
            queued,
            key=lambda r: (
                -r.spec.priority,
                self.active_gpu_lease_count(r.spec.user_id),
                r.sequence,
            ),
        )

    def _initial_queue_reason(self, spec):
        pool_nodes = [
            node for node in self.nodes.values() if node.pool == spec.pool
        ]
        if not any(node.state == NodeState.ACTIVE for node in pool_nodes):
            if any(
                node.state == NodeState.REGISTERING for node in pool_nodes
            ):
                return "node_probing"
            return "pool_empty"
        if spec.resources.gpus > 0:
            return "waiting_for_gpu"
        return "waiting_for_cpu_or_memory"

    def _queued_quota_exceeded(self, user_id):
        if self.policy.max_queued_requests_per_user is None:
            return False
        queued = sum(1 for r in self.requests.values() if r.spec.user_id == user_id and r.state == RequestState.QUEUED)
        return queued >= self.policy.max_queued_requests_per_user

    def _active_gpu_cap(self):
        if self.policy.max_active_gpu_sessions_per_user is None:
            return 10 ** 12
        return self.policy.max_active_gpu_sessions_per_user

    def _would_exceed_active_gpu_quota(self, spec):
        if spec.resources.gpus == 0:
            return False
        return self.active_gpu_lease_count(spec.user_id) + spec.resources.gpus > self._active_gpu_cap()

    def _would_exceed_pool_session_cap(self, spec):
        cap = self.policy.max_active_sessions_per_pool
        if cap is None:
            return False
        active = 0
        for session in self.sessions.values():
            if session.pool == spec.pool and session.state not in SESSION_TERMINAL_STATES | {SessionState.SUSPENDED, SessionState.PURGING}:
                active += 1
        return active + 1 > cap

    def _find_reusable_session(self, spec):
        if not spec.reuse:
            return None
        image_identity = spec.image_identity(self.policy.default_image)
        for session in self.sessions.values():
            if self._workspace_blocks_session(session):
                continue
            if session.reusable_for(spec, image_identity, session.runtime_adapter):
                node = self.nodes.get(session.node_id)
                if (
                    node is not None
                    and node.state == NodeState.ACTIVE
                    and capability_reason(
                        spec,
                        node.capabilities,
                        node.runtime_capabilities,
                        runtime_adapter=session.runtime_adapter,
                    )
                    is None
                ):
                    return session
        return None

    def _resume_retained_execution(self, request):
        spec = request.spec
        if not spec.reuse:
            return False
        for session in self.sessions.values():
            if session.state not in (SessionState.SUSPENDING, SessionState.SUSPENDED, SessionState.STOPPED, SessionState.FAILED, SessionState.LOST):
                continue
            if not session.matches_reuse_identity(spec, spec.image_identity(self.policy.default_image), session.runtime_adapter):
                continue
            if session.state == SessionState.SUSPENDING:
                self._set_queued_reason(request, "waiting_for_session")
                return True
            node = self.nodes.get(session.node_id)
            if node is None or not node_is_placement_eligible(node, self.clock()):
                self._set_queued_reason(request, "retained_node_unavailable")
                return True
            workspace = self.workspaces.get((session.workspace or {}).get("id"))
            if (workspace is None or workspace.get("state") in ("deleted", "needs_check", "needs_recovery")
                    or (workspace.get("state") == "absent" and not session.empty_retained)
                    or (session.state in (SessionState.FAILED, SessionState.LOST) and workspace.get("state") != "retained")):
                self._mark_request_terminal(request, RequestState.FAILED)
                request.reason, request.failure_kind = "retained_state_unavailable", "infrastructure"
                self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
                return True
            if (workspace["state"] != "retained" and not session.empty_retained) or workspace.get("operation", {}).get("state") == "pending":
                self._set_queued_reason(request, "waiting_for_session")
                return True
            if not session.empty_retained and session.runtime_adapter not in self.runtime_adapters:
                snapshot = node.last_status or {}
                retained = snapshot.get("suspended_sessions")
                items = snapshot.get("sessions")
                valid = (snapshot.get("authoritative") is True and isinstance(items, list)
                         and all(isinstance(item, dict) and isinstance(item.get("execution_session_id"), str) for item in items))
                present = valid and any(item["execution_session_id"] == session.execution_session_id for item in items)
                receipt = retained.get(session.execution_session_id) if isinstance(retained, dict) else None
                if valid and isinstance(retained, dict) and not present and (receipt is None or type(receipt) is not int or receipt > session.suspend_version):
                    self._mark_request_terminal(request, RequestState.FAILED)
                    request.reason, request.failure_kind = "retained_state_unavailable", "infrastructure"
                    request.stderr = "agent cannot confirm retained execution metadata; explicit rebuild required"
                    self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
                    return True
                if not valid or receipt != session.suspend_version or present:
                    command = {"action": "stop_session", "execution_session_id": session.execution_session_id,
                               "suspend_version": session.suspend_version}
                    queue = self.node_control_queues.setdefault(node.node_id, [])
                    if command not in queue:
                        queue.append(command)
                    self._set_queued_reason(request, "waiting_for_session")
                    return True
            if self._would_exceed_active_gpu_quota(request.spec):
                self._set_queued_reason(request, "quota_exceeded")
                return True
            if self._would_exceed_pool_session_cap(request.spec):
                self._set_queued_reason(request, "pool_session_limit")
                return True
            if node.available_cpu() < session.cpu_cores or node.available_memory() < session.memory_gb:
                if node.resources.cpu_cores >= session.cpu_cores and node.resources.memory_gb >= session.memory_gb:
                    self._evict_cached_idle_session_for_request(request.spec, node_id=node.node_id)
                self._set_queued_reason(request, "waiting_for_cpu_or_memory")
                return True
            gpu_uuids = self._choose_reuse_gpus(session, request)
            if gpu_uuids is None:
                self._set_queued_reason(request, "waiting_for_gpu")
                return True
            workspace.update(state="pending", execution_session_id=session.execution_session_id)
            create = session.empty_retained and session.workspace.get("create", False)
            session.workspace = dict(copy.deepcopy(workspace), create=create)
            session.resume_required = not session.empty_retained
            session.empty_retained = False
            session.activated_at = self.clock()
            self._emit("workspace_state", workspace["id"], "pending", "execution_resume", copy.deepcopy(workspace))
            node.used_cpu_cores += session.cpu_cores
            node.used_memory_gb += session.memory_gb
            node.active_sessions.append(session.execution_session_id)
            self._lease_gpus_to_session(session.execution_session_id, gpu_uuids)
            self._start_request_on_session(request, session, gpu_uuids, node.node_id)
            return True
        return False

    def _find_matching_active_session(self, spec):
        if not spec.reuse:
            return None
        image_identity = spec.image_identity(self.policy.default_image)
        for session in self.sessions.values():
            if (session.state not in SESSION_TERMINAL_STATES and self._workspace_blocks_session(session)
                    and session.matches_reuse_identity(spec, image_identity, session.runtime_adapter)):
                return session
            if session.state in SESSION_TERMINAL_STATES or session.state in (
                SessionState.CACHED_IDLE,
                SessionState.WARM_IDLE,
            ):
                continue
            if not session.matches_reuse_identity(
                spec,
                image_identity,
                session.runtime_adapter,
            ):
                continue
            node = self.nodes.get(session.node_id)
            if (
                node is not None
                and node.state == NodeState.ACTIVE
                and capability_reason(
                    spec,
                    node.capabilities,
                    node.runtime_capabilities,
                    runtime_adapter=session.runtime_adapter,
                )
                is None
            ):
                return session
        return None

    def _choose_placement(self, request):
        spec = request.spec
        workspace = self.workspaces.get(spec.workspace["id"]) if spec.workspace["id"] is not None else None
        if workspace is not None and (workspace["state"] != "retained" or workspace.get("operation", {}).get("state") == "pending"):
            return PlacementResult(None, reason="waiting_for_session")
        now = self.clock()
        candidates = []
        resolved_adapters = {}
        candidate_gpus = {}
        any_pool_nodes = False
        any_runtime_block = False
        any_gpu_shortage = False
        any_cpu_mem_shortage = False
        any_deadline_block = False
        any_draining = False
        any_registering = False
        any_active = False
        for node in self.nodes.values():
            if node.pool != spec.pool:
                continue
            if workspace is not None and workspace["node_id"] != node.node_id:
                continue
            if workspace is not None and workspace.get("maintenance") and node.workspace_status.get("maintenance_protocol") != 1:
                any_runtime_block = True
                continue
            any_pool_nodes = True
            if spec.resources.locality and spec.resources.locality not in (node.node_id, node.allocation_id):
                continue
            if node.state == NodeState.REGISTERING:
                any_registering = True
            if node.state == NodeState.ACTIVE:
                any_active = True
            if not node_is_placement_eligible(node, now):
                if node.state == NodeState.DRAINING:
                    any_draining = True
                continue
            if workspace is None:
                desired = self.workspace_configs.get(spec.pool, node.node_id)
                reported = node.workspace_status
                if reported.get("state") != "applied" or reported.get("applied_version") != desired.version or reported.get("config") != desired.to_dict():
                    any_runtime_block = True
                    continue
            if not node_can_accept_lifetime(node, now, spec.resources.admission_timeout_seconds):
                any_deadline_block = True
                continue
            reason = capability_reason(
                spec, node.capabilities, node.runtime_capabilities
            )
            if reason is not None:
                any_runtime_block = True
                continue
            if node.available_cpu() < spec.resources.cpu_cores or node.available_memory() < spec.resources.memory_gb:
                any_cpu_mem_shortage = True
                continue
            available = self._gpu_candidates_for_node(node, spec)
            if len(available) < spec.resources.gpus:
                any_gpu_shortage = True
                continue
            resolved_runtime = resolve_runtime_capability(
                spec, node.capabilities, node.runtime_capabilities
            )
            if resolved_runtime is None:
                any_runtime_block = True
                continue
            resolved_adapters[node.node_id] = resolved_runtime[0]
            candidate_gpus[node.node_id] = available
            candidates.append(node)
        if not candidates:
            if not any_pool_nodes:
                return PlacementResult(None, reason="pool_empty")
            if any_registering and not any_active:
                return PlacementResult(None, reason="node_probing")
            if any_deadline_block:
                return PlacementResult(None, reason="allocation_expiring")
            if any_draining:
                return PlacementResult(None, reason="node_draining")
            if any_runtime_block:
                return PlacementResult(None, reason="waiting_for_runtime_capability")
            if any_gpu_shortage:
                return PlacementResult(None, reason="waiting_for_gpu")
            if any_cpu_mem_shortage:
                return PlacementResult(None, reason="waiting_for_cpu_or_memory")
            return PlacementResult(None, reason="no_capable_node")
        free_counts = {
            node.node_id: len(candidate_gpus[node.node_id])
            for node in candidates
        }
        image_identity = spec.image_identity(self.policy.default_image)
        pool_config = self.pool_configs.get(spec.pool)
        if (
            pool_config is not None
            and pool_config.pool_mode == "shared"
            and spec.resources.gpus > 0
        ):
            def shared_node_key(node):
                selected = candidate_gpus[node.node_id][
                    : spec.resources.gpus
                ]
                refcounts = tuple(
                    self.gpu_ledger[gpu_uuid].shared_refcount
                    for gpu_uuid in selected
                )
                if pool_config.placement == "pack":
                    placement_key = (
                        -sum(count > 0 for count in refcounts),
                        -sum(refcounts),
                        tuple(-count for count in refcounts),
                    )
                else:
                    placement_key = (
                        max(refcounts),
                        sum(refcounts),
                        refcounts,
                    )
                return (
                    placement_key,
                    runtime_preference_rank(
                        spec.resources.isolation_policy,
                        resolved_adapters[node.node_id],
                    ),
                    0 if image_identity in node.cached_images else 1,
                    len(node.active_sessions),
                    node.used_cpu_cores,
                    node.node_id,
                )

            node = min(candidates, key=shared_node_key)
        else:
            ranked = rank_nodes_for_request(
                candidates, spec.resources, free_counts
            )
            ranked = sorted(
                enumerate(ranked),
                key=lambda item: (
                    runtime_preference_rank(
                        spec.resources.isolation_policy,
                        resolved_adapters[item[1].node_id],
                    ),
                    0
                    if image_identity in item[1].cached_images
                    else 1,
                    item[0],
                ),
            )
            node = ranked[0][1]
        gpu_uuids = tuple(
            candidate_gpus[node.node_id][: spec.resources.gpus]
        )
        runtime_adapter = resolved_adapters[node.node_id]
        return PlacementResult(
            node.node_id,
            gpu_uuids=gpu_uuids,
            runtime_adapter=runtime_adapter,
        )

    def _evict_cached_idle_session_for_request(self, spec, node_id=None):
        now = self.clock()
        image_identity = spec.image_identity(self.policy.default_image)
        candidates = []
        for session in self.sessions.values():
            if node_id is not None and session.node_id != node_id:
                continue
            if session.state not in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE):
                continue
            if self.workspaces.get((session.workspace or {}).get("id"), {}).get("operation", {}).get("state") == "pending":
                continue
            if session.pool != spec.pool:
                continue
            if session.reusable_for(spec, image_identity, session.runtime_adapter):
                continue
            node = self.nodes.get(session.node_id)
            if node is None or not node_is_placement_eligible(node, now):
                continue
            if not node_can_accept_lifetime(node, now, spec.resources.admission_timeout_seconds):
                continue
            if capability_reason(
                spec, node.capabilities, node.runtime_capabilities
            ) is not None:
                continue
            if node_id is None and node.available_cpu() + session.cpu_cores < spec.resources.cpu_cores:
                continue
            if node_id is None and node.available_memory() + session.memory_gb < spec.resources.memory_gb:
                continue
            available_gpus = self._gpu_candidates_for_node(node, spec)
            if len(available_gpus) < spec.resources.gpus:
                continue
            candidates.append(session)
        if not candidates:
            return None
        session = sorted(candidates, key=lambda item: (item.last_used_at, item.created_at, item.execution_session_id))[0]
        self.suspend_session(session.execution_session_id, reason="evict_for_queued_work", enforce_auth=False)
        return session.execution_session_id

    def _request_can_target_gpu(self, request, gpu_uuid):
        if request.state != RequestState.QUEUED:
            return False
        spec = request.spec
        now = self.clock()
        for node in self.nodes.values():
            if node.pool != spec.pool:
                continue
            if gpu_uuid not in node.resources.gpu_uuids:
                continue
            if len(node.resources.gpu_uuids) < spec.resources.gpus:
                continue
            if (
                spec.resources.locality
                and spec.resources.locality
                not in (node.node_id, node.allocation_id)
            ):
                continue
            if not node_is_placement_eligible(node, now):
                continue
            if not node_can_accept_lifetime(
                node, now, spec.resources.admission_timeout_seconds
            ):
                continue
            if (
                node.available_cpu() < spec.resources.cpu_cores
                or node.available_memory() < spec.resources.memory_gb
            ):
                continue
            if (
                capability_reason(
                    spec, node.capabilities, node.runtime_capabilities
                )
                is not None
            ):
                continue
            return True
        return False

    def _has_queued_gpu_mode_demand(self, gpu_uuid, pool_mode):
        for request in self.requests.values():
            config = self.pool_configs.get(request.spec.pool)
            if config is None or config.pool_mode != pool_mode:
                continue
            if request.spec.resources.gpus <= 0:
                continue
            if self._request_can_target_gpu(request, gpu_uuid):
                return True
        return False

    def _gpu_admission_allowed(self, node, gpu_uuid):
        if gpu_uuid in self.node_purge_quarantine:
            return False
        entry = self.gpu_ledger.setdefault(
            gpu_uuid, PhysicalGpuLedgerEntry(gpu_uuid)
        )
        if node.pool_mode == "exclusive":
            if entry.state != GpuLeaseState.FREE:
                return False
            if not self._has_queued_gpu_mode_demand(
                gpu_uuid, "shared"
            ):
                return True
            policy, ratio = self.contention_policy_for_gpu(gpu_uuid)
            if policy == "exclusive-first":
                return True
            return self._gpu_shared_admissions.get(gpu_uuid, 0) >= ratio

        if entry.state == GpuLeaseState.EXCLUSIVE:
            return False
        if (
            entry.shared_capacity is not None
            and entry.shared_capacity != node.gpu_slots
        ):
            return False
        if entry.shared_refcount >= node.gpu_slots:
            return False
        if not self._has_queued_gpu_mode_demand(
            gpu_uuid, "exclusive"
        ):
            return True
        policy, ratio = self.contention_policy_for_gpu(gpu_uuid)
        if policy == "exclusive-first":
            return False
        shared_admissions = self._gpu_shared_admissions.get(gpu_uuid, 0)
        if policy == "barrier-ratio":
            return shared_admissions < ratio
        return (
            entry.state == GpuLeaseState.SHARED
            or shared_admissions < ratio
        )

    def _gpu_candidates_for_node(self, node, spec):
        if spec.resources.gpus <= 0:
            return []
        candidates = [
            gpu_uuid
            for gpu_uuid in node.resources.gpu_uuids
            if self._gpu_admission_allowed(node, gpu_uuid)
        ]
        if node.pool_mode != "shared":
            return candidates
        if node.placement == "pack":
            return sorted(
                candidates,
                key=lambda gpu_uuid: (
                    0
                    if self.gpu_ledger[gpu_uuid].shared_refcount > 0
                    else 1,
                    -self.gpu_ledger[gpu_uuid].shared_refcount,
                    gpu_uuid,
                ),
            )
        return sorted(
            candidates,
            key=lambda gpu_uuid: (
                self.gpu_ledger[gpu_uuid].shared_refcount,
                gpu_uuid,
            ),
        )

    def _choose_reuse_gpus(self, session, request):
        spec = request.spec
        node = self.nodes.get(session.node_id)
        if node is None or not node_is_placement_eligible(node, self.clock()):
            return None
        if not node_can_accept_lifetime(node, self.clock(), spec.resources.admission_timeout_seconds):
            return None
        if capability_reason(
            spec,
            node.capabilities,
            node.runtime_capabilities,
            runtime_adapter=session.runtime_adapter,
        ) is not None:
            return None
        if spec.resources.gpus == 0:
            return ()
        available = self._gpu_candidates_for_node(node, spec)
        if len(available) < spec.resources.gpus:
            return None
        return tuple(available[: spec.resources.gpus])

    def _record_gpu_contention_state(self, gpu_uuid, reason):
        self._emit(
            "gpu_contention_state",
            gpu_uuid,
            None,
            reason,
            {
                "gpu_uuid": gpu_uuid,
                "shared_admissions": self._gpu_shared_admissions.get(
                    gpu_uuid, 0
                ),
            },
        )

    def _validate_gpu_grant(
        self, execution_session_id, gpu_uuid, pool_mode, gpu_slots
    ):
        entry = self.gpu_ledger.setdefault(
            gpu_uuid, PhysicalGpuLedgerEntry(gpu_uuid)
        )
        if execution_session_id in entry.holder_execution_session_ids:
            return
        if pool_mode == "exclusive":
            if entry.state != GpuLeaseState.FREE:
                raise RuntimeError("GPU already leased: " + gpu_uuid)
            return
        if entry.state == GpuLeaseState.EXCLUSIVE:
            raise RuntimeError("GPU exclusively leased: " + gpu_uuid)
        if (
            entry.shared_capacity is not None
            and entry.shared_capacity != gpu_slots
        ):
            raise RuntimeError(
                "GPU shared capacity conflicts across pools: " + gpu_uuid
            )
        if entry.shared_refcount >= gpu_slots:
            raise RuntimeError("GPU shared slots exhausted: " + gpu_uuid)

    def _grant_gpu_leases(
        self,
        execution_session_id,
        gpu_uuids,
        pool_mode,
        gpu_slots,
        count_admission=True,
    ):
        gpu_uuids = tuple(gpu_uuids)
        if len(set(gpu_uuids)) != len(gpu_uuids):
            raise RuntimeError(
                "one request cannot take multiple slots on one GPU"
            )
        session = self.sessions[execution_session_id]
        node = self.nodes.get(session.node_id)
        if node is not None:
            missing = set(gpu_uuids) - set(node.resources.gpu_uuids)
            if missing:
                raise RuntimeError(
                    "session GPU is not advertised by its node: "
                    + ", ".join(sorted(missing))
                )
        for gpu_uuid in gpu_uuids:
            self._validate_gpu_grant(
                execution_session_id, gpu_uuid, pool_mode, gpu_slots
            )
        cotenant_counts = {
            gpu_uuid: len(
                self.gpu_ledger[gpu_uuid].holder_execution_session_ids
            )
            for gpu_uuid in gpu_uuids
        }
        for gpu_uuid in gpu_uuids:
            entry = self.gpu_ledger[gpu_uuid]
            if execution_session_id in entry.holder_execution_session_ids:
                continue
            if pool_mode == "exclusive":
                entry.exclusive_execution_session_id = execution_session_id
            else:
                if entry.shared_capacity is None:
                    entry.shared_capacity = gpu_slots
                entry.shared_execution_session_ids.append(execution_session_id)
                if count_admission:
                    self._gpu_shared_admissions[gpu_uuid] = (
                        self._gpu_shared_admissions.get(gpu_uuid, 0) + 1
                    )
                    self._record_gpu_contention_state(
                        gpu_uuid, "shared_admitted"
                    )
        session.gpu_uuids = gpu_uuids
        session.gpu_count = len(gpu_uuids)
        session.gpu_mode = pool_mode
        session.gpu_slots = gpu_slots
        session.gpu_cotenant_counts = cotenant_counts
        session.gpu_cotenant_count = (
            max(cotenant_counts.values()) if cotenant_counts else 0
        )

    def _lease_gpus_to_session(self, execution_session_id, gpu_uuids):
        session = self.sessions[execution_session_id]
        gpu_uuids = tuple(gpu_uuids)
        node = self.nodes.get(session.node_id)
        if node is None:
            raise RuntimeError("session node is unavailable")
        self._release_gpu_lease(execution_session_id)
        self._grant_gpu_leases(
            execution_session_id,
            gpu_uuids,
            node.pool_mode,
            node.gpu_slots,
        )
        adapter_capabilities = (
            node.runtime_capabilities.get(session.runtime_adapter)
        )
        session.enforcement_mode = (
            adapter_capabilities.gpu_enforcement
            if adapter_capabilities is not None and gpu_uuids
            else EnforcementMode.NONE
        )

    def _release_gpu_lease(self, execution_session_id):
        session = self.sessions.get(execution_session_id)
        for gpu_uuid, entry in self.gpu_ledger.items():
            if entry.exclusive_execution_session_id == execution_session_id:
                entry.exclusive_execution_session_id = None
                self._gpu_shared_admissions[gpu_uuid] = 0
                self._record_gpu_contention_state(
                    gpu_uuid, "exclusive_released"
                )
            if execution_session_id in entry.shared_execution_session_ids:
                entry.shared_execution_session_ids.remove(execution_session_id)
                if not entry.shared_execution_session_ids:
                    entry.shared_capacity = None
        if session is not None:
            session.gpu_uuids = ()
            session.enforcement_mode = EnforcementMode.NONE

    def _create_session(
        self, request, node_id, gpu_uuids, runtime_adapter=None
    ):
        self._session_seq += 1
        execution_session_id = "exec-" + uuid.uuid4().hex
        spec = request.spec
        node = self.nodes[node_id]
        image_identity = spec.image_identity(self.policy.default_image)
        resolved_runtime = resolve_runtime_capability(
            spec,
            node.capabilities,
            node.runtime_capabilities,
            runtime_adapter=runtime_adapter,
        )
        if resolved_runtime is None:
            raise RuntimeError("node has no compatible runtime adapter")
        runtime_adapter, adapter_capabilities = resolved_runtime
        enforcement = (
            adapter_capabilities.gpu_enforcement
            if gpu_uuids
            else EnforcementMode.NONE
        )
        state_layer_id = None
        if spec.resources.isolation_policy in STATEFUL_ISOLATION_POLICIES:
            state_layer_id = execution_session_id
        workspace = self._reserve_workspace(spec, node_id, execution_session_id)
        session = SessionRecord(
            workspace=workspace,
            execution_session_id=execution_session_id,
            pool=spec.pool,
            user_id=spec.user_id,
            session_id=spec.session_id,
            node_id=node_id,
            runtime_adapter=runtime_adapter,
            image_identity=image_identity,
            mounts=spec.effective_mounts(),
            privileged=spec.privileged,
            gpu_uuids=tuple(gpu_uuids),
            cpu_cores=spec.resources.cpu_cores,
            memory_gb=spec.resources.memory_gb,
            enforcement_mode=enforcement,
            state=SessionState.CREATING,
            created_at=self.clock(),
            last_used_at=self.clock(),
            max_duration_seconds=spec.resources.max_duration_seconds,
            isolation_policy=spec.resources.isolation_policy,
            state_layer_id=state_layer_id,
            gpu_mode=node.pool_mode,
            gpu_slots=node.gpu_slots,
        )
        self.sessions[execution_session_id] = session
        node.used_cpu_cores += spec.resources.cpu_cores
        node.used_memory_gb += spec.resources.memory_gb
        node.active_sessions.append(execution_session_id)
        self._grant_gpu_leases(
            execution_session_id,
            gpu_uuids,
            node.pool_mode,
            node.gpu_slots,
        )
        self._emit("session_state", execution_session_id, SessionState.CREATING.value, None, self._session_details(session))
        return session

    def _start_request_on_session(self, request, session, gpu_uuids, node_id):
        leased_at = self.clock()
        request.state = RequestState.LEASED
        if request.leased_at is None:
            request.leased_at = leased_at
        request.reason = None
        request.execution_session_id = session.execution_session_id
        request.node_id = node_id
        request.gpu_uuids = tuple(gpu_uuids)
        request.gpu_mode = session.gpu_mode
        request.gpu_cotenant_count = session.gpu_cotenant_count
        request.gpu_cotenant_counts = dict(
            session.gpu_cotenant_counts
        )
        request.updated_at = self.clock()
        self._emit("request_state", request.request_id, RequestState.LEASED.value, None, self._request_details(request))
        request.state = RequestState.PREPARING_IMAGE
        self._emit("request_state", request.request_id, RequestState.PREPARING_IMAGE.value, None, self._request_details(request))
        request.state = RequestState.STARTING_SESSION
        session.state = SessionState.RUNNING_COMMAND
        session.active_request_id = request.request_id
        session.last_used_at = self.clock()
        self._emit("request_state", request.request_id, RequestState.STARTING_SESSION.value, None, self._request_details(request))
        self._emit("session_state", session.execution_session_id, session.state.value, None, self._session_details(session))
        self._mark_request_running(request)
        self._emit("request_state", request.request_id, RequestState.RUNNING.value, None, self._request_details(request))
        if not self._execute_request_if_runtime_available(request, session):
            self._assign_command_to_node(request, session)

    def _execute_request_if_runtime_available(self, request, session):
        runtime = self.runtime_adapters.get(session.runtime_adapter)
        if runtime is None:
            return False
        try:
            workspace = session.workspace
            manager = self._embedded_workspaces.get(session.execution_session_id)
            if manager is None:
                manager = WorkspaceManager(WorkspaceConfig.from_dict(workspace["config"]), session.node_id, session.pool)
                self._embedded_workspaces[session.execution_session_id] = manager
            handle = manager.prepare(workspace["id"], session.user_id, session.execution_session_id, create=workspace.get("create", False))
            self._record_embedded_workspace(session, manager)
            visible = handle.path if session.runtime_adapter == "local-process" else workspace["target"]
            image = request.spec.image_identity(self.policy.default_image)
            prepared_image = image
            if hasattr(runtime, "prepare_image"):
                prepared = runtime.prepare_image(image, self.nodes[session.node_id].runtime_cache_config)
                if isinstance(prepared, RuntimeResult):
                    if prepared.exit_code != 0:
                        self._fail_running_request(request, "image_prepare_failed", prepared)
                        return True
                elif prepared:
                    prepared_image = prepared
            if session.execution_session_id not in self._started_runtime_sessions and hasattr(runtime, "start_session"):
                start_result = self._runtime_start_session(runtime, session, request, prepared_image)
                if isinstance(start_result, RuntimeResult) and start_result.exit_code != 0:
                    self._fail_running_request(request, "runtime_start_failed", start_result)
                    return True
                self._started_runtime_sessions.add(session.execution_session_id)
            manager.verify(handle)
            result = runtime.run_command(session.execution_session_id, request.spec.command, cwd=request.spec.cwd or visible,
                                         env=dict(request.spec.env, CAPE_WORKSPACE_DIR=visible))
            manager.verify(handle)
            if not isinstance(result, RuntimeResult):
                result = RuntimeResult(0, "", "")
            self.complete_request(
                request.request_id,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                failure_kind=result.failure_kind or "workload",
            )
            return True
        except Exception as exc:
            self._fail_running_request(request, "runtime_launch_exception", RuntimeResult(127, "", str(exc)))
            return True

    def _runtime_start_session(self, runtime, session, request, image=None):
        container_spec = self._container_session_spec(session, request, image=image)
        manager = self._embedded_workspaces[session.execution_session_id]
        handle = manager.handles[session.workspace["id"]]
        visible = handle.path if session.runtime_adapter == "local-process" else session.workspace["target"]
        if session.runtime_adapter != "local-process":
            container_spec.mounts = (MountSpec(handle.path, visible),) + tuple(container_spec.mounts)
        if container_spec.cwd is None:
            container_spec.cwd = visible
        container_spec.env = dict(container_spec.env, CAPE_WORKSPACE_DIR=visible)
        if session.suspend_version:
            retained = getattr(runtime, "retained_sessions", {}).get(session.execution_session_id)
            if retained is None:
                return RuntimeResult(127, "", "retained execution metadata is missing")
            for field in ("overlay_path", "overlay_upper", "overlay_work"):
                setattr(container_spec, field, getattr(retained, field, None))
            return runtime.resume_session(container_spec)
        return runtime.start_session(container_spec)

    def _record_embedded_workspace(self, session, manager):
        workspace = session.workspace
        update = manager.inspect(workspace["id"], session.user_id)
        record = dict(self.workspaces[workspace["id"]], state=update["state"], revision=update["revision"],
                      path=update["path"], last_error=update.get("last_error"))
        self.workspaces[workspace["id"]] = record
        self._emit("workspace_state", workspace["id"], record["state"], None, record)

    def _container_session_spec(self, session, request, image=None):
        node = self.nodes[session.node_id]
        cache_dir = node.runtime_cache_config if node.runtime_cache_config and node.runtime_cache_config != "default" else None
        return ContainerSessionSpec(
            execution_session_id=session.execution_session_id,
            image=image or session.image_identity,
            mounts=session.mounts,
            cwd=request.spec.cwd,
            env=request.spec.env,
            gpu_uuids=session.gpu_uuids,
            privileged=session.privileged,
            cache_dir=cache_dir,
            isolation_policy=session.isolation_policy,
            state_layer_id=session.state_layer_id,
        )

    def _assign_command_to_node(self, request, session):
        if request.request_id in self.request_commands:
            return self.command_assignments[self.request_commands[request.request_id]]
        self._command_seq += 1
        command_id = "cmd-{:06d}".format(self._command_seq)
        session_spec = self._container_session_spec(session, request)
        session_spec_details = self._container_session_spec_details(session_spec)
        session_spec_details["runtime_adapter"] = session.runtime_adapter
        session_spec_details["workspace"] = copy.deepcopy(session.workspace)
        session_spec_details["suspend_version"] = session.suspend_version
        session_spec_details["resume_retained"] = session.resume_required
        assignment = CommandAssignment(
            command_id=command_id,
            request_id=request.request_id,
            execution_session_id=session.execution_session_id,
            node_id=session.node_id,
            session_spec=session_spec_details,
            command_spec={"command": list(request.spec.command), "cwd": request.spec.cwd, "env": dict(request.spec.env)},
            state=CommandState.PENDING,
            created_at=self.clock(),
            log_path=request.log_path,
        )
        self.command_assignments[command_id] = assignment
        self.request_commands[request.request_id] = command_id
        self.node_command_queues.setdefault(session.node_id, []).append(command_id)
        self._emit("command_state", command_id, assignment.state.value, None, self.command_assignment_details(assignment))
        return assignment

    def _container_session_spec_details(self, spec):
        return {
            "execution_session_id": spec.execution_session_id,
            "image": spec.image,
            "mounts": [self._mount_details(m) for m in spec.mounts],
            "cwd": spec.cwd,
            "env": dict(spec.env),
            "gpu_uuids": list(spec.gpu_uuids),
            "privileged": spec.privileged,
            "cache_dir": spec.cache_dir,
            "tmp_dir": spec.tmp_dir,
            "runtime_dir": spec.runtime_dir,
            "oci_bundle": spec.oci_bundle,
            "isolation_policy": spec.isolation_policy,
            "state_layer_id": spec.state_layer_id,
            "overlay_path": spec.overlay_path,
            "overlay_upper": spec.overlay_upper,
            "overlay_work": spec.overlay_work,
        }

    def _cleanup_session_spec_details(self, session):
        node = self.nodes.get(session.node_id)
        cache_dir = None
        if (
            node is not None
            and node.runtime_cache_config
            and node.runtime_cache_config != "default"
        ):
            cache_dir = node.runtime_cache_config
        details = self._container_session_spec_details(
            ContainerSessionSpec(
                execution_session_id=session.execution_session_id,
                image=session.image_identity,
                mounts=session.mounts,
                gpu_uuids=(),
                privileged=session.privileged,
                cache_dir=cache_dir,
                isolation_policy=session.isolation_policy,
                state_layer_id=session.state_layer_id,
            )
        )
        details.pop("env", None)
        details["workspace"] = copy.deepcopy(session.workspace)
        return details

    def _fail_running_request(self, request, reason, result):
        request.exit_code = result.exit_code
        request.failure_kind = "infrastructure"
        request.stdout = result.stdout
        request.stderr = result.stderr
        self._mark_request_terminal(request, RequestState.FAILED)
        request.reason = reason
        if request.execution_session_id and request.execution_session_id in self.sessions:
            session = self.sessions[request.execution_session_id]
            session.active_request_id = None
            session.state = SessionState.FAILED
            self._cleanup_runtime_session(session.execution_session_id)
            self._release_session(session.execution_session_id)
            self._emit("session_state", session.execution_session_id, session.state.value, reason, self._session_details(session))
        self._emit("request_state", request.request_id, request.state.value, reason, self._request_details(request))

    def _cleanup_runtime_session(self, execution_session_id, preserve=False):
        session = self.sessions.get(execution_session_id)
        if session is None:
            self._started_runtime_sessions.discard(execution_session_id)
            return
        if not preserve:
            session.suspend_version += 1
        runtime = self.runtime_adapters.get(session.runtime_adapter)
        if runtime is not None and execution_session_id in self._started_runtime_sessions and hasattr(runtime, "stop_session"):
            try:
                result = (runtime.suspend_session(execution_session_id) if preserve
                          else runtime.stop_session(execution_session_id))
                if isinstance(result, RuntimeResult) and result.exit_code:
                    raise RuntimeError(result.stderr or "runtime cleanup failed")
            except Exception:
                return False
        elif runtime is None and session.node_id in self.nodes:
            control = {
                "action": "suspend_session" if preserve else "stop_session",
                "execution_session_id": execution_session_id,
            }
            control["suspend_version"] = session.suspend_version
            self.node_control_queues.setdefault(session.node_id, []).append(control)
        self._started_runtime_sessions.discard(execution_session_id)
        manager = self._embedded_workspaces.get(execution_session_id)
        if manager is not None:
            try:
                manager.release(session.workspace["id"])
                self._embedded_workspaces.pop(execution_session_id, None)
            finally:
                self._record_embedded_workspace(session, manager)
        return True

    def _release_session(self, execution_session_id):
        session = self.sessions.get(execution_session_id)
        if session is None:
            return
        node = self.nodes.get(session.node_id)
        if node is not None:
            node.used_cpu_cores = max(0, node.used_cpu_cores - session.cpu_cores)
            node.used_memory_gb = max(0, node.used_memory_gb - session.memory_gb)
            if execution_session_id in node.active_sessions:
                node.active_sessions.remove(execution_session_id)
        self._release_gpu_lease(execution_session_id)

    @staticmethod
    def _session_holds_uncertain_gpu_lease(session):
        return bool(
            session.state == SessionState.LOST and session.gpu_uuids
        )

    def _mark_sessions_lost_for_node(self, node_id):
        for session in self.sessions.values():
            if session.node_id != node_id or session.state in (SessionState.STOPPED, SessionState.FAILED, SessionState.LOST):
                continue
            session.state = SessionState.LOST
            if session.active_request_id and session.active_request_id in self.requests:
                request = self.requests[session.active_request_id]
                self._mark_request_terminal(request, RequestState.LOST)
                request.reason = "node_lost"
                request.failure_kind = "infrastructure"
                self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
            self._release_session(session.execution_session_id)
            self._emit("session_state", session.execution_session_id, session.state.value, "node_lost", self._session_details(session))

    def _emit(self, event_type, entity_id, state, reason, details=None):
        event = EventRecord(
            timestamp=self.clock(),
            event_type=event_type,
            entity_id=entity_id,
            state=state,
            reason=reason,
            details=details or {},
        )
        self.journal.append(event)
        try:
            self.metrics.record_event(event)
        except Exception:
            pass

    def _authorize_submit(self, spec, token):
        if self.auth is None:
            return
        if getattr(self.auth, "managed", False):
            actor = self.auth.store.authenticate(token)
            if spec.user_id not in (actor["user_id"], actor["name"]):
                raise PermissionError("cannot submit for another identity")
            spec.user_id = actor["user_id"]
            spec.pool = self.auth.pool_id(spec.pool)
            spec._admission_token_id = actor["token_id"]
        user_id = self.auth.authenticate_user(token, spec.pool, Role.SUBMITTER)
        if user_id != spec.user_id and not self.auth.user_has_role(token, spec.pool, Role.ADMIN):
            raise PermissionError("token user cannot submit for another user")

    def _authorize_request_read(self, request, token):
        if getattr(self.auth, "managed", False):
            self.auth.require_owner(token, request.spec.user_id)
            return
        if self.auth is None:
            return
        if self.auth.user_has_role(token, request.spec.pool, Role.VIEWER) or self.auth.user_has_role(token, request.spec.pool, Role.OPERATOR):
            return
        user_id = self.auth.authenticate_user(token, request.spec.pool, Role.SUBMITTER)
        if user_id != request.spec.user_id:
            raise PermissionError("cannot inspect another user's request")

    def _authorize_request_cancel(self, request, token):
        if getattr(self.auth, "managed", False):
            self.auth.require_owner(token, request.spec.user_id)
            return
        if self.auth is None:
            return
        if self.auth.user_has_role(token, request.spec.pool, Role.OPERATOR):
            return
        user_id = self.auth.authenticate_user(token, request.spec.pool, Role.SUBMITTER)
        if user_id != request.spec.user_id:
            raise PermissionError("cannot cancel another user's request")

    def _authorize_session_read(self, session, token):
        if getattr(self.auth, "managed", False):
            self.auth.require_owner(token, session.user_id)
            return
        if self.auth is None:
            return
        if self.auth.user_has_role(token, session.pool, Role.VIEWER) or self.auth.user_has_role(token, session.pool, Role.OPERATOR):
            return
        user_id = self.auth.authenticate_user(token, session.pool, Role.SUBMITTER)
        if user_id != session.user_id:
            raise PermissionError("cannot inspect another user's session")

    def _authorize_session_operator(self, session, token):
        if getattr(self.auth, "managed", False):
            self.auth.require_owner(token, session.user_id)
            return
        if self.auth is None:
            return
        if self.auth.user_has_role(token, session.pool, Role.OPERATOR):
            return
        user_id = self.auth.authenticate_user(token, session.pool, Role.SUBMITTER)
        if user_id != session.user_id:
            raise PermissionError("cannot stop another user's session")

    def _authorize_operator_for_node(self, node_id, token):
        if self.auth is None:
            return
        if getattr(self.auth, "managed", False):
            from .identity_nodes import owned_node, authenticate_agent
            try:
                owned_node(self.auth.store, token, node_id)
            except PermissionError:
                authenticate_agent(self.auth.store, token, node_id=node_id)
            return
        node = self.nodes[node_id]
        if self.auth.user_has_role(token, node.pool, Role.OPERATOR):
            return
        user_id = self.auth.authenticate_user(token, node.pool, Role.NODE_AGENT)
        if user_id != node_id and not self.auth.user_has_role(token, node.pool, Role.ADMIN):
            raise PermissionError("node-agent token cannot drain another node")

    def _authorize_node_read(self, node, token):
        if self.auth is None:
            return
        if self.auth.user_has_role(token, node.pool, Role.VIEWER) or self.auth.user_has_role(token, node.pool, Role.OPERATOR):
            return
        user_id = self.auth.authenticate_user(token, node.pool, Role.NODE_AGENT)
        if user_id != node.node_id and not self.auth.user_has_role(token, node.pool, Role.ADMIN):
            raise PermissionError("node-agent token cannot inspect another node")

    def _authorize_node_agent(self, token, pool, node_id=None):
        if self.auth is None:
            return
        user_id = self.auth.authenticate_user(token, pool, Role.NODE_AGENT)
        if node_id is not None and user_id != node_id and not self.auth.user_has_role(token, pool, Role.ADMIN):
            raise PermissionError("node-agent token cannot operate on another node")

    def _request_details(self, request):
        return {
            **({"admission_token_id": request.admission_token_id} if hasattr(request, "admission_token_id") else {}),
            "request_id": request.request_id,
            "spec": self._spec_details(request.spec),
            "state": request.state.value,
            "reason": request.reason,
            "created_at": request.created_at,
            "updated_at": request.updated_at,
            "leased_at": request.leased_at,
            "started_at": request.started_at,
            "ended_at": request.ended_at,
            "sequence": request.sequence,
            "execution_session_id": request.execution_session_id,
            "node_id": request.node_id,
            "gpu_uuids": list(request.gpu_uuids),
            "gpu_mode": request.gpu_mode,
            "gpu_cotenant_count": request.gpu_cotenant_count,
            "gpu_cotenant_counts": dict(
                request.gpu_cotenant_counts
            ),
            "exit_code": request.exit_code,
            "failure_kind": request.failure_kind,
            "stdout": request.stdout,
            "stderr": request.stderr,
            "log_path": request.log_path,
            "artifact_paths": list(request.artifact_paths),
        }

    def _session_details(self, session):
        return {
            "suspend_version": session.suspend_version,
            "suspend_requested": session.suspend_requested,
            "resume_required": session.resume_required,
            "activated_at": session.activated_at,
            "empty_retained": session.empty_retained,
            "execution_session_id": session.execution_session_id,
            "pool": session.pool,
            "user_id": session.user_id,
            "session_id": session.session_id,
            "node_id": session.node_id,
            "runtime_adapter": session.runtime_adapter,
            "image_identity": session.image_identity,
            "mounts": [self._mount_details(m) for m in session.mounts],
            "privileged": session.privileged,
            "gpu_uuids": list(session.gpu_uuids),
            "gpu_count": session.gpu_count,
            "gpu_mode": session.gpu_mode,
            "gpu_slots": session.gpu_slots,
            "gpu_cotenant_count": session.gpu_cotenant_count,
            "gpu_cotenant_counts": dict(
                session.gpu_cotenant_counts
            ),
            "cpu_cores": session.cpu_cores,
            "memory_gb": session.memory_gb,
            "enforcement_mode": session.enforcement_mode.value,
            "state": session.state.value,
            "created_at": session.created_at,
            "last_used_at": session.last_used_at,
            "max_duration_seconds": session.max_duration_seconds,
            "active_request_id": session.active_request_id,
            "isolation_policy": session.isolation_policy,
            "state_layer_id": session.state_layer_id,
            "workspace": copy.deepcopy(session.workspace),
        }

    def _node_details(self, node):
        return {
            "node_id": node.node_id,
            "pool": node.pool,
            "pool_mode": node.pool_mode,
            "gpu_slots": node.gpu_slots,
            "placement": node.placement,
            "resources": self._resources_details(node.resources),
            "capabilities": self._capability_details(node.capabilities),
            "runtime_capabilities": {
                adapter: self._capability_details(capabilities)
                for adapter, capabilities in sorted(
                    node.runtime_capabilities.items()
                )
            },
            "state": node.state.value,
            "agent_epoch": node.agent_epoch,
            "registered_at": node.registered_at,
            "last_heartbeat_at": node.last_heartbeat_at,
            "allocation_id": node.allocation_id,
            "allocation_deadline": node.allocation_deadline,
            "drain_before_sec": node.drain_before_sec,
            "cleanup_buffer_sec": node.cleanup_buffer_sec,
            "runtime_cache_config": node.runtime_cache_config,
            "cached_images": list(node.cached_images),
            "used_cpu_cores": node.used_cpu_cores,
            "used_memory_gb": node.used_memory_gb,
            "active_sessions": list(node.active_sessions),
            "status": getattr(node, "last_status", {}),
            "workspace_status": dict(node.workspace_status),
            "workspace_config": self.workspace_configs.get(node.pool, node.node_id).to_dict(),
        }

    def _spec_details(self, spec):
        return {
            "pool": spec.pool,
            "user_id": spec.user_id,
            "session_id": spec.session_id,
            "command": list(spec.command),
            "resources": self._resource_request_details(spec.resources),
            "cwd": spec.cwd,
            "env": dict(spec.env),
            "image": spec.image,
            "profile": getattr(spec, "resolved_profile", spec.profile),
            "mounts": [self._mount_details(m) for m in spec.mounts],
            "reuse": spec.reuse,
            "keep_session_on_failure": spec.keep_session_on_failure,
            "priority": spec.priority,
            "privileged": spec.privileged,
            "log_path": spec.log_path,
            "artifact_paths": list(spec.artifact_paths),
            "workspace": dict(spec.workspace),
        }

    def _mount_details(self, mount):
        return {"source": mount.source, "target": mount.target, "read_only": mount.read_only}

    def _resource_request_details(self, resources):
        return {
            "gpus": resources.gpus,
            "cpu_cores": resources.cpu_cores,
            "memory_gb": resources.memory_gb,
            "max_duration_seconds": resources.max_duration_seconds,
            "command_timeout_seconds": resources.command_timeout_seconds,
            "hard_gpu_enforcement": resources.hard_gpu_enforcement,
            "runtime_adapter": resources.runtime_adapter,
            "ports": list(resources.ports),
            "locality": resources.locality,
            "isolation_policy": resources.isolation_policy,
        }

    def _resources_details(self, resources):
        return {"cpu_cores": resources.cpu_cores, "memory_gb": resources.memory_gb, "gpu_uuids": list(resources.gpu_uuids)}

    def _capability_details(self, caps):
        return {
            "adapter": caps.adapter,
            "image_formats": list(caps.image_formats),
            "isolation_policies": list(caps.isolation_policies),
            "nested_container": caps.nested_container,
            "can_bind_mounts": caps.can_bind_mounts,
            "can_select_gpus": caps.can_select_gpus,
            "gpu_enforcement": caps.gpu_enforcement.value,
            "can_enforce_cpu_memory": caps.can_enforce_cpu_memory,
            "node_local_cache": caps.node_local_cache,
            "supports_warm_sessions": caps.supports_warm_sessions,
            "privileged_allowed": caps.privileged_allowed,
            "gpu_isolation_policies": list(caps.gpu_isolation_policies),
            "reduced_isolation_policies": list(
                caps.reduced_isolation_policies
            ),
        }

    def command_assignment_details(self, assignment):
        return {
            "command_id": assignment.command_id,
            "request_id": assignment.request_id,
            "execution_session_id": assignment.execution_session_id,
            "node_id": assignment.node_id,
            "session_spec": dict(assignment.session_spec),
            "command_spec": dict(assignment.command_spec),
            "state": assignment.state.value if hasattr(assignment.state, "value") else assignment.state,
            "action": "run",
            "created_at": assignment.created_at,
            "updated_at": assignment.updated_at,
            "log_path": assignment.log_path,
            "result": dict(assignment.result),
        }

    def request_status(self, request_id, token=None):
        request = self.get_request(request_id, token=token)
        return self._request_status_record(request)

    def _request_status_record(self, request):
        session = self.sessions.get(request.execution_session_id) if request.execution_session_id else None
        node = self.nodes.get(request.node_id) if request.node_id else None
        command_id = self.request_commands.get(request.request_id)
        return {
            "request_id": request.request_id,
            "state": request.state.value,
            "reason": request.reason,
            "status_message": request_status_message(
                request.state.value, request.reason
            ),
            "owner": request.spec.user_id,
            "pool": request.spec.pool,
            "session_id": request.spec.session_id,
            "node_id": request.node_id,
            "allocation_id": node.allocation_id if node is not None else None,
            "execution_session_id": request.execution_session_id,
            "command_id": command_id,
            "workspace": copy.deepcopy(self.workspaces.get(session.workspace["id"])) if session is not None and session.workspace is not None else None,
            "failure_kind": request.failure_kind,
            "keep_session_on_failure": (
                request.spec.keep_session_on_failure
            ),
            "gpu_uuids": list(request.gpu_uuids),
            "gpu_mode": request.gpu_mode,
            "gpu_cotenant_count": request.gpu_cotenant_count,
            "gpu_cotenant_counts": dict(
                request.gpu_cotenant_counts
            ),
            "runtime_adapter": session.runtime_adapter if session is not None else request.spec.resources.runtime_adapter,
            "isolation_policy": session.isolation_policy if session is not None else request.spec.resources.isolation_policy,
            "enforcement_mode": session.enforcement_mode.value if session is not None else EnforcementMode.NONE.value,
            "image": session.image_identity if session is not None else request.spec.image_identity(self.policy.default_image),
            "profile": getattr(request.spec, "resolved_profile", request.spec.profile),
            "created_at": request.created_at,
            "updated_at": request.updated_at,
            "leased_at": request.leased_at,
            "started_at": request.started_at,
            "ended_at": request.ended_at,
            "exit_code": request.exit_code,
            "log_path": request.log_path,
            "log_handle": "/v1/requests/{}/logs".format(request.request_id),
            "artifact_paths": list(request.artifact_paths),
            "deadline": node.allocation_deadline if node is not None else None,
            "ports": list(request.spec.resources.ports),
            "locality": request.spec.resources.locality,
        }

    def node_status(self, node_id, token=None):
        node = self.get_node(node_id, token=token)
        return self._node_details(node)

    def logs_for_request(self, request_id, token=None):
        request = self.get_request(request_id, token=token)
        return {"request_id": request_id, "stdout": request.stdout, "stderr": request.stderr, "log_path": request.log_path}

    def allocation_status(self, allocation_id):
        allocation = self.allocations[allocation_id]
        return {
            "allocation_id": allocation.allocation_id,
            "pool": allocation.pool,
            "state": allocation.state.value if hasattr(allocation.state, "value") else allocation.state,
            "node_ids": list(allocation.node_ids),
            "deadline": allocation.deadline,
            "created_at": allocation.created_at,
            "updated_at": allocation.updated_at,
            "reason": allocation.reason,
        }

    def pool_utilization(self, pool=None):
        if pool is not None and getattr(self.auth, "managed", False):
            pool = self.auth.pool_id(pool)
        return summarize_pool(self, pool=pool).to_dict()

    def drain_allocation(self, allocation_id, reason="allocation_drain", token=None):
        if getattr(self.auth, "managed", False):
            self.auth.store.require_admin(token)
        allocation = self.allocations[allocation_id]
        allocation.state = AllocationState.DRAINING
        allocation.reason = reason
        allocation.updated_at = self.clock()
        for node_id in list(allocation.node_ids):
            if node_id in self.nodes:
                self.drain_node(node_id, reason=reason, token=token)
        self._emit("allocation_state", allocation_id, allocation.state.value, reason, self.allocation_status(allocation_id))
        return allocation

    def remove_allocation(self, allocation_id, force=False, token=None):
        if getattr(self.auth, "managed", False):
            self.auth.store.require_admin(token)
            raise ValueError("persistent allocations retire through node drain/exit; direct removal is not allowed")
        allocation = self.allocations[allocation_id]
        allocation.state = AllocationState.REMOVING
        allocation.updated_at = self.clock()
        for node_id in list(allocation.node_ids):
            if node_id in self.nodes:
                self.remove_node(node_id, force=force, token=token)
        allocation.state = AllocationState.REMOVED
        allocation.reason = "allocation_removed"
        allocation.updated_at = self.clock()
        self._emit("allocation_state", allocation_id, allocation.state.value, allocation.reason, self.allocation_status(allocation_id))
        return allocation

    def _resolve_profile(self, spec):
        registry = getattr(self.policy, "profile_registry", None)
        if registry is None or not spec.profile:
            return spec
        return registry.resolve(spec)

    def _inflight_assignment_for_session(self, session):
        request_id = session.active_request_id
        command_id = self.request_commands.get(request_id)
        assignment = self.command_assignments.get(command_id)
        if (
            assignment is None
            or assignment.execution_session_id != session.execution_session_id
            or assignment.state
            not in (CommandState.PENDING, CommandState.RUNNING)
        ):
            return None
        return assignment

    def _reconcile_dispatch_is_in_grace(self, session, now):
        assignment = self._inflight_assignment_for_session(session)
        if assignment is None:
            return False
        return (
            now - assignment.updated_at
            < self.policy.reconcile_missing_grace_seconds
        )

    def _fail_unreported_command(self, request_id, reason):
        command = self.command_assignments.get(self.request_commands.get(request_id))
        if command is not None and command.state in (CommandState.PENDING, CommandState.RUNNING):
            command.state = CommandState.FAILED
            command.updated_at = self.clock()
            command.result = {"reason": reason, "failure_kind": "infrastructure", "exit_code": 127}
            self._emit("command_state", command.command_id, command.state.value, reason,
                       self.command_assignment_details(command))

    def _resolve_missing_live_assignments(self, node_id, status):
        for key in list(self._reconcile_missing_observations):
            if key.startswith("live:"):
                assignment = self.command_assignments.get(key[5:])
                if assignment is None or assignment.state != CommandState.RUNNING:
                    self._reconcile_missing_observations.pop(key, None)
        items = status.get("sessions") if isinstance(status, dict) else None
        authoritative = (isinstance(status, dict) and status.get("authoritative") is True
                         and isinstance(items, list)
                         and all(isinstance(item, dict) and isinstance(item.get("execution_session_id"), str)
                                 and item["execution_session_id"] for item in items))
        command_ids = status.get("command_ids") if isinstance(status, dict) else None
        if isinstance(status, dict) and "command_ids" in status:
            authoritative = authoritative and isinstance(command_ids, list) and all(
                isinstance(value, str) and value for value in command_ids)
        reported = {item["execution_session_id"]: item for item in items} if authoritative else {}
        accepted = set(command_ids or []) if authoritative else set()
        for session in list(self.sessions.values()):
            if session.node_id != node_id or session.state != SessionState.RUNNING_COMMAND:
                continue
            command = self._inflight_assignment_for_session(session)
            observation_key = "live:" + command.command_id if command is not None else None
            item = reported.get(session.execution_session_id)
            protected = item is not None and (command_ids is None or item.get("state") not in
                                             ("CACHED_IDLE", "WARM_IDLE", "IDLE"))
            if (not authoritative or protected or command is None
                    or command.state != CommandState.RUNNING or command.command_id in accepted):
                self._reconcile_missing_observations.pop(observation_key, None)
                continue
            now = self.clock()
            observation = self._reconcile_missing_observations.setdefault(
                observation_key, {"first_missing_at": now, "count": 0})
            observation["count"] += 1
            if (self._reconcile_dispatch_is_in_grace(session, now)
                    or observation["count"] < self.policy.reconcile_missing_heartbeats
                    or now - observation["first_missing_at"] < self.policy.reconcile_missing_grace_seconds):
                continue
            request = self.requests[command.request_id]
            if request.state in REQUEST_TERMINAL_STATES:
                continue
            message = "dispatched assignment missing from authoritative agent state; outcome unknown; not automatically replayed"
            command.state = CommandState.FAILED
            command.updated_at = now
            command.result = {"reason": "assignment_missing", "failure_kind": "infrastructure",
                              "exit_code": 127, "stderr": message}
            self._mark_request_terminal(request, RequestState.FAILED, now=now)
            request.reason = "assignment_missing"
            request.failure_kind = "infrastructure"
            request.exit_code = 127
            request.stderr = message
            self._emit("command_state", command.command_id, command.state.value,
                       request.reason, self.command_assignment_details(command))
            self._emit("request_state", request.request_id, request.state.value,
                       request.reason, self._request_details(request))
            self._reconcile_missing_observations.pop(observation_key, None)
            self.stop_session(session.execution_session_id, reason="assignment_missing", enforce_auth=False)

    def _resolve_reconciling_from_node_status(self, node_id, status):
        if isinstance(status, dict) and status.get("authoritative") is True:
            suspended = status.get("suspended_sessions", {})
            items = status.get("sessions")
            valid_snapshot = (isinstance(items, list) and all(isinstance(item, dict)
                and isinstance(item.get("execution_session_id"), str) and item["execution_session_id"] for item in items))
            reported_active = {item["execution_session_id"] for item in items} if valid_snapshot else set()
            for session in self.sessions.values():
                workspace = self.workspaces.get((session.workspace or {}).get("id"), {})
                receipt = suspended.get(session.execution_session_id) if isinstance(suspended, dict) else None
                confirmed = (type(receipt) is int and receipt == session.suspend_version
                             and session.execution_session_id not in reported_active and valid_snapshot
                             and workspace.get("state") == "retained")
                if session.node_id == node_id and session.state in (SessionState.STOPPED, SessionState.FAILED, SessionState.LOST) and confirmed:
                    if session.execution_session_id in self.nodes[node_id].active_sessions:
                        self._release_session(session.execution_session_id)
                    else:
                        self._release_gpu_lease(session.execution_session_id)
                    session.state = SessionState.SUSPENDED
                    self._emit("session_state", session.execution_session_id, session.state.value, "retained_after_stop", self._session_details(session))
                    continue
                if session.node_id == node_id and session.state == SessionState.PURGING:
                    purged = status.get("purged_executions", [])
                    if valid_snapshot and session.execution_session_id not in reported_active and isinstance(purged, list) and session.execution_session_id in purged:
                        session.state = SessionState.PURGED
                        self._emit("session_state", session.execution_session_id, session.state.value, "purged", self._session_details(session))
                    else:
                        command = {"action": "purge_session", "execution_session_id": session.execution_session_id,
                                   "suspend_version": session.suspend_version}
                        queue = self.node_control_queues.setdefault(node_id, [])
                        if command not in queue:
                            queue.append(command)
                    continue
                if session.node_id != node_id or session.state != SessionState.SUSPENDING:
                    continue
                version = getattr(session, "suspend_version", 0)
                if confirmed:
                    self._finish_suspension(session)
                else:
                    command = {"action": "suspend_session", "execution_session_id": session.execution_session_id,
                               "suspend_version": version}
                    queue = self.node_control_queues.setdefault(node_id, [])
                    if command not in queue:
                        queue.append(command)
        self._resolve_missing_live_assignments(node_id, status)
        sessions = status.get("sessions", []) if isinstance(status, dict) else []
        if not isinstance(sessions, list) or any(not isinstance(item, dict) for item in sessions):
            return
        reported = {}
        for item in sessions:
            execution_session_id = item.get("execution_session_id")
            if execution_session_id:
                reported[execution_session_id] = item
        authoritative = bool(status.get("authoritative")) if isinstance(status, dict) else False
        if authoritative:
            self._resolve_lost_gpu_leases_from_node_status(
                node_id, reported
            )
        for session in list(self.sessions.values()):
            if session.node_id != node_id or session.state != SessionState.RECONCILING:
                continue
            item = reported.get(session.execution_session_id)
            if item is None and not authoritative:
                continue
            now = self.clock()
            if item is None:
                observation = self._reconcile_missing_observations.setdefault(
                    session.execution_session_id,
                    {"first_missing_at": now, "count": 0},
                )
                observation["count"] += 1
                if self._reconcile_dispatch_is_in_grace(session, now):
                    continue
                if (
                    observation["count"]
                    < self.policy.reconcile_missing_heartbeats
                    or now - observation["first_missing_at"]
                    < self.policy.reconcile_missing_grace_seconds
                ):
                    continue
            else:
                self._reconcile_missing_observations.pop(
                    session.execution_session_id, None
                )
            state = (item or {}).get("state", "missing")
            active_request_id = (item or {}).get("active_request_id") or session.active_request_id
            if state in ("RUNNING", "RUNNING_COMMAND"):
                session.state = SessionState.RUNNING_COMMAND
                session.active_request_id = active_request_id
                if (
                    active_request_id in self.requests
                    and self.requests[active_request_id].state
                    not in REQUEST_TERMINAL_STATES
                ):
                    request = self.requests[active_request_id]
                    self._mark_request_running(request)
                    request.reason = None
                    self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
                self._emit("session_state", session.execution_session_id, session.state.value, "reconciled", self._session_details(session))
            elif state in ("CACHED_IDLE", "WARM_IDLE", "IDLE"):
                self._release_gpu_lease(session.execution_session_id)
                session.state = SessionState.CACHED_IDLE
                session.active_request_id = None
                session.last_used_at = self.clock()
                self._emit("session_state", session.execution_session_id, session.state.value, "reconciled", self._session_details(session))
                if active_request_id in self.requests and self.requests[active_request_id].state == RequestState.RECONCILING:
                    request = self.requests[active_request_id]
                    self._mark_request_terminal(request, RequestState.LOST)
                    request.reason = "command_status_unknown"
                    self._fail_unreported_command(request.request_id, request.reason)
                    self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
            else:
                self._reconcile_missing_observations.pop(
                    session.execution_session_id, None
                )
                session.state = SessionState.FAILED
                session.active_request_id = None
                self._release_session(session.execution_session_id)
                self._emit("session_state", session.execution_session_id, session.state.value, "reconcile_missing", self._session_details(session))
                if (
                    active_request_id in self.requests
                    and self.requests[active_request_id].state
                    not in REQUEST_TERMINAL_STATES
                ):
                    request = self.requests[active_request_id]
                    self._mark_request_terminal(request, RequestState.FAILED)
                    request.reason = "reconcile_missing"
                    self._fail_unreported_command(request.request_id, request.reason)
                    self._emit("request_state", request.request_id, request.state.value, request.reason, self._request_details(request))
        if authoritative:
            expected_sessions = {
                session.execution_session_id
                for session in self.sessions.values()
                if session.node_id == node_id
                and session.state not in SESSION_TERMINAL_STATES
            }
            control_queue = self.node_control_queues.setdefault(node_id, [])
            already_queued = {
                command.get("execution_session_id")
                for command in control_queue
                if command.get("action") == "stop_session"
            }
            for execution_session_id in sorted(reported):
                if (
                    not execution_session_id
                    or execution_session_id in expected_sessions
                ):
                    continue
                if execution_session_id not in already_queued:
                    self._queue_stop_session_control(
                        node_id, execution_session_id
                    )
                    already_queued.add(execution_session_id)

    def _queue_stop_session_control(self, node_id, execution_session_id):
        control_queue = self.node_control_queues.setdefault(node_id, [])
        if any(
            command.get("action") == "stop_session"
            and command.get("execution_session_id") == execution_session_id
            for command in control_queue
        ):
            return False
        command = {
            "action": "stop_session",
            "execution_session_id": execution_session_id,
        }
        session = self.sessions.get(execution_session_id)
        if session is not None and session.node_id == node_id:
            command["suspend_version"] = max(1, session.suspend_version)
            command["runtime_adapter"] = session.runtime_adapter
            command["session_spec"] = (
                self._cleanup_session_spec_details(session)
            )
        control_queue.append(command)
        return True

    def _resolve_lost_gpu_leases_from_node_status(
        self, node_id, reported
    ):
        for session in list(self.sessions.values()):
            if (
                session.node_id != node_id
                or not self._session_holds_uncertain_gpu_lease(session)
            ):
                continue
            execution_session_id = session.execution_session_id
            if (
                execution_session_id not in reported
                and execution_session_id in self._lost_cleanup_dispatched
            ):
                self._lost_cleanup_dispatched.discard(execution_session_id)
                session.active_request_id = None
                self._release_session(execution_session_id)
                self._emit(
                    "session_state",
                    execution_session_id,
                    session.state.value,
                    "lost_session_cleanup_confirmed",
                    self._session_details(session),
                )
                continue
            if execution_session_id in reported:
                self._lost_cleanup_dispatched.discard(execution_session_id)
            self._queue_stop_session_control(node_id, execution_session_id)

    def _replay_event(self, event):
        details = event.details or {}
        for body in (details, details.get("spec", {})):
            if any(key in body for key in ("task_id", "session_key", "workspace_id")):
                raise ValueError("obsolete journal identity schema; initialize a new controller")
        if event.event_type == "logical_session_created":
            self.logical_sessions[(details["user_id"], details["session_id"])] = copy.deepcopy(details)
        elif event.event_type == "workspace_state":
            self.workspaces[event.entity_id] = copy.deepcopy(details)
        elif event.event_type == "workspace_config_set":
            self.workspace_configs.restore(details.get("pool", event.entity_id), details["new_config"])
        elif (
            event.event_type
            in ("pool_config_set", "pool_config_replaced")
            and "new_config" in details
        ):
            pool = details.get("pool", event.entity_id)
            config = PoolConfig(**details["new_config"])
            self._explicit_pool_configs[pool] = config
            self.pool_configs[pool] = config
            self._pool_config_requires_matching_join[pool] = (
                event.event_type == "pool_config_set"
            )
        elif event.event_type == "request_submitted" and "spec" in details:
            record = self._request_from_details(details)
            self.logical_sessions.setdefault((record.spec.user_id, record.spec.session_id), {
                "user_id": record.spec.user_id, "session_id": record.spec.session_id,
                "state": "alive", "created_at": record.created_at})
            self.requests[record.request_id] = record
            self._request_seq = max(self._request_seq, record.sequence)
        elif event.event_type in ("request_queued", "request_state") and event.entity_id in self.requests:
            self._update_request_from_details(self.requests[event.entity_id], details, event)
        elif event.event_type in (
            "node_announced",
            "node_announcement_refreshed",
            "node_registered",
            "node_refreshed",
            "node_recovered",
        ) and "resources" in details:
            node = self._node_from_details(details)
            self.nodes[node.node_id] = node
            self._consume_pool_config_matching_join(
                node.pool,
                PoolConfig(
                    node.pool_mode, node.gpu_slots, node.placement
                ),
            )
        elif event.event_type == "node_state" and event.entity_id in self.nodes:
            self._update_node_from_details(self.nodes[event.entity_id], details, event)
        elif event.event_type == "allocation_state" and "allocation_id" in details:
            self.allocations[event.entity_id] = AllocationRecord(
                allocation_id=details["allocation_id"],
                pool=details["pool"],
                state=AllocationState(
                    event.state
                    or details.get("state", AllocationState.ACTIVE.value)
                ),
                node_ids=tuple(details.get("node_ids", ())),
                deadline=details.get("deadline"),
                created_at=details.get("created_at", event.timestamp),
                updated_at=details.get("updated_at", event.timestamp),
                reason=event.reason or details.get("reason"),
            )
        elif event.event_type == "session_state" and "execution_session_id" in details:
            session = self._session_from_details(details)
            self.sessions[session.execution_session_id] = session
            try:
                self._session_seq = max(self._session_seq, int(session.execution_session_id.split("-")[-1]))
            except Exception:
                pass
        elif event.event_type == "command_state" and "command_id" in details:
            assignment = CommandAssignment(
                command_id=details["command_id"],
                request_id=details["request_id"],
                execution_session_id=details["execution_session_id"],
                node_id=details["node_id"],
                session_spec=dict(details.get("session_spec", {})),
                command_spec=dict(details.get("command_spec", {})),
                state=CommandState(
                    event.state
                    or details.get("state", CommandState.PENDING.value)
                ),
                created_at=details.get("created_at", event.timestamp),
                updated_at=details.get("updated_at", event.timestamp),
                log_path=details.get("log_path"),
                result=dict(details.get("result", {})),
            )
            self.command_assignments[assignment.command_id] = assignment
            self.request_commands[assignment.request_id] = assignment.command_id
            try:
                self._command_seq = max(
                    self._command_seq,
                    int(assignment.command_id.split("-")[-1]),
                )
            except Exception:
                pass
        elif event.event_type == "gpu_contention_state":
            gpu_uuid = details.get("gpu_uuid", event.entity_id)
            self._gpu_shared_admissions[gpu_uuid] = int(
                details.get("shared_admissions", 0)
            )

    def _mark_inflight_reconciling_after_replay(self):
        recovery_started_at = self.clock()
        for assignment in self.command_assignments.values():
            if assignment.state in (CommandState.PENDING, CommandState.RUNNING):
                assignment.updated_at = recovery_started_at
        for node in self.nodes.values():
            if node.state in (NodeState.REMOVED, NodeState.LOST):
                continue
            node.last_heartbeat_at = recovery_started_at
            if node.state not in (NodeState.DRAINING, NodeState.REMOVING):
                node.state = NodeState.UNHEALTHY
        for request in self.requests.values():
            if request.state in REQUEST_TERMINAL_STATES or request.state == RequestState.QUEUED:
                continue
            request.state = RequestState.RECONCILING
            request.reason = "controller_restart"
        for session in self.sessions.values():
            if session.state in SESSION_TERMINAL_STATES or session.state in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE, SessionState.SUSPENDING, SessionState.SUSPENDED, SessionState.PURGING):
                continue
            session.state = SessionState.RECONCILING

    def _rebuild_indexes_after_replay(self):
        derived_pool_configs = {}
        removed_pool_configs = {}
        self.gpu_ledger = {}
        for node in self.nodes.values():
            config = PoolConfig(
                node.pool_mode, node.gpu_slots, node.placement
            )
            if node.state == NodeState.REMOVED:
                removed_pool_configs.setdefault(node.pool, set()).add(
                    config
                )
            else:
                established = derived_pool_configs.get(node.pool)
                if established is not None and established != config:
                    raise ValueError(
                        "journal contains conflicting configuration for pool "
                        + node.pool
                    )
                derived_pool_configs[node.pool] = config
            for gpu_uuid in node.resources.gpu_uuids:
                self.gpu_ledger.setdefault(
                    gpu_uuid, PhysicalGpuLedgerEntry(gpu_uuid)
                )
                self._gpu_shared_admissions.setdefault(gpu_uuid, 0)
            if node.allocation_id:
                allocation = self.allocations.get(node.allocation_id)
                if allocation is None:
                    allocation = AllocationRecord(
                        node.allocation_id,
                        node.pool,
                        state=AllocationState.ACTIVE,
                        node_ids=(),
                        deadline=node.allocation_deadline,
                        created_at=node.registered_at,
                        updated_at=node.last_heartbeat_at,
                    )
                    self.allocations[node.allocation_id] = allocation
                if node.node_id not in allocation.node_ids:
                    allocation.node_ids.append(node.node_id)
            node.used_cpu_cores = 0
            node.used_memory_gb = 0
            node.active_sessions = []
        for pool, historical_configs in removed_pool_configs.items():
            if (
                pool in derived_pool_configs
                or pool in self._explicit_pool_configs
            ):
                continue
            if len(historical_configs) != 1:
                raise ValueError(
                    "journal contains conflicting historical "
                    "configuration for pool " + pool
                )
            derived_pool_configs[pool] = next(
                iter(historical_configs)
            )
        for pool, explicit_config in self._explicit_pool_configs.items():
            derived_config = derived_pool_configs.get(pool)
            if (
                derived_config is not None
                and derived_config != explicit_config
            ):
                raise ValueError(
                    "journal contains a live node that conflicts with "
                    "the explicit configuration for pool " + pool
                )
            derived_pool_configs[pool] = explicit_config
        self.pool_configs = derived_pool_configs
        for session in self.sessions.values():
            if (
                session.state in SESSION_TERMINAL_STATES or session.state in (SessionState.SUSPENDED, SessionState.PURGING)
            ) and (
                not self._session_holds_uncertain_gpu_lease(session)
            ):
                continue
            node = self.nodes.get(session.node_id)
            if node is not None:
                node.used_cpu_cores += session.cpu_cores
                node.used_memory_gb += session.memory_gb
                if session.execution_session_id not in node.active_sessions:
                    node.active_sessions.append(session.execution_session_id)
            if session.state not in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE):
                self._grant_gpu_leases(
                    session.execution_session_id,
                    session.gpu_uuids,
                    session.gpu_mode,
                    session.gpu_slots,
                    count_admission=False,
                )
        for request in self.requests.values():
            try:
                self._request_seq = max(self._request_seq, int(request.request_id.split("-")[-1]))
            except Exception:
                pass

    def _request_from_details(self, details):
        spec = self._spec_from_details(details["spec"])
        record = RequestRecord(
            request_id=details["request_id"],
            spec=spec,
            state=RequestState(details.get("state", RequestState.SUBMITTED.value)),
            reason=details.get("reason"),
            created_at=details.get("created_at", 0.0),
            updated_at=details.get("updated_at", 0.0),
            sequence=details.get("sequence", 0),
            execution_session_id=details.get("execution_session_id"),
            node_id=details.get("node_id"),
            gpu_uuids=tuple(details.get("gpu_uuids", ())),
            exit_code=details.get("exit_code"),
            failure_kind=details.get("failure_kind"),
            stdout=details.get("stdout", ""),
            stderr=details.get("stderr", ""),
            log_path=details.get("log_path"),
            artifact_paths=tuple(details.get("artifact_paths", ())),
            gpu_mode=details.get("gpu_mode"),
            gpu_cotenant_count=details.get(
                "gpu_cotenant_count", 0
            ),
            gpu_cotenant_counts=details.get(
                "gpu_cotenant_counts", {}
            ),
            started_at=details.get("started_at"),
            ended_at=details.get("ended_at"),
            leased_at=details.get("leased_at"),
        )
        if "admission_token_id" in details:
            record.admission_token_id = details["admission_token_id"]
        return record

    def _update_request_from_details(self, request, details, event):
        if event.state:
            request.state = RequestState(event.state)
        request.reason = event.reason
        request.updated_at = details.get("updated_at", event.timestamp)
        if request.leased_at is None and details.get("leased_at") is not None:
            request.leased_at = details["leased_at"]
        if request.started_at is None and details.get("started_at") is not None:
            request.started_at = details["started_at"]
        if request.ended_at is None and details.get("ended_at") is not None:
            request.ended_at = details["ended_at"]
        request.execution_session_id = details.get("execution_session_id", request.execution_session_id)
        request.node_id = details.get("node_id", request.node_id)
        request.gpu_uuids = tuple(details.get("gpu_uuids", request.gpu_uuids))
        request.gpu_mode = details.get(
            "gpu_mode", request.gpu_mode
        )
        request.gpu_cotenant_count = int(
            details.get(
                "gpu_cotenant_count", request.gpu_cotenant_count
            )
            or 0
        )
        request.gpu_cotenant_counts = dict(
            details.get(
                "gpu_cotenant_counts", request.gpu_cotenant_counts
            )
        )
        request.exit_code = details.get("exit_code", request.exit_code)
        request.failure_kind = details.get(
            "failure_kind", request.failure_kind
        )
        request.stdout = details.get("stdout", request.stdout)
        request.stderr = details.get("stderr", request.stderr)
        request.log_path = details.get("log_path", request.log_path)
        request.artifact_paths = tuple(details.get("artifact_paths", request.artifact_paths))

    def _spec_mounts_from_details(self, details):
        mounts = [
            MountSpec(m["source"], m["target"], m.get("read_only", False))
            for m in details.get("mounts", ())
        ]
        legacy_pairs = (
            ("workspace_path", "workspace_mount_path"),
            ("output_path", "output_mount_path"),
        )
        for source_key, target_key in legacy_pairs:
            source = details.get(source_key)
            if source:
                mounts.append(MountSpec(source, details.get(target_key) or source))

        deduped = []
        seen = set()
        for mount in mounts:
            key = (mount.source, mount.target, mount.read_only)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(mount)
        return tuple(deduped)

    def _spec_from_details(self, details):
        resource = details.get("resources", {})
        image = details.get("image")
        profile = details.get("profile")
        spec = WorkloadSpec(
            pool=details["pool"],
            user_id=details["user_id"],
            session_id=details["session_id"],
            command=tuple(details["command"]),
            workspace=details.get("workspace"),
            resources=ResourceRequest(
                gpus=resource.get("gpus", 0),
                cpu_cores=resource.get("cpu_cores", 1),
                memory_gb=resource.get("memory_gb", 1),
                max_duration_seconds=resource.get("max_duration_seconds", 3600),
                command_timeout_seconds=resource.get("command_timeout_seconds"),
                hard_gpu_enforcement=resource.get("hard_gpu_enforcement", False),
                runtime_adapter=resource.get("runtime_adapter"),
                ports=tuple(resource.get("ports", ())),
                locality=resource.get("locality"),
                isolation_policy=resource.get("isolation_policy", "default"),
            ),
            cwd=details.get("cwd"),
            env=details.get("env", {}),
            image=image,
            profile=None if image and profile else profile,
            mounts=self._spec_mounts_from_details(details),
            reuse=details.get("reuse", True),
            keep_session_on_failure=details.get(
                "keep_session_on_failure", True
            ),
            priority=details.get("priority", 0),
            privileged=details.get("privileged", False),
            log_path=details.get("log_path"),
            artifact_paths=tuple(details.get("artifact_paths", ())),
        )
        if image and profile:
            spec.resolved_profile = profile
        return spec

    def _node_from_details(self, details):
        node = NodeRecord(
            node_id=details["node_id"],
            pool=details["pool"],
            resources=self._resources_from_details(details["resources"]),
            capabilities=self._capability_from_details(details["capabilities"]),
            runtime_capabilities={
                adapter: self._capability_from_details(capabilities)
                for adapter, capabilities in details.get(
                    "runtime_capabilities", {}
                ).items()
            },
            state=NodeState(details.get("state", NodeState.ACTIVE.value)),
            registered_at=details.get("registered_at", 0.0),
            last_heartbeat_at=details.get("last_heartbeat_at", 0.0),
            allocation_id=details.get("allocation_id"),
            allocation_deadline=details.get("allocation_deadline"),
            drain_before_sec=details.get("drain_before_sec", 0),
            cleanup_buffer_sec=details.get("cleanup_buffer_sec", 0),
            runtime_cache_config=details.get("runtime_cache_config", "default"),
            cached_images=tuple(details.get("cached_images", ())),
            used_cpu_cores=details.get("used_cpu_cores", 0),
            used_memory_gb=details.get("used_memory_gb", 0),
            active_sessions=details.get("active_sessions", ()),
            agent_epoch=details.get("agent_epoch"),
            pool_mode=details.get("pool_mode", "exclusive"),
            gpu_slots=details.get("gpu_slots", 1),
            placement=details.get("placement", "pack"),
        )
        node.last_status = details.get("status", {})
        node.workspace_status = dict(details.get("workspace_status") or {"protocol": 0, "state": "unavailable", "applied_version": None})
        return node

    def _update_node_from_details(self, node, details, event):
        node.state = NodeState(event.state)
        node.last_heartbeat_at = details.get("last_heartbeat_at", event.timestamp)
        node.active_sessions = list(details.get("active_sessions", node.active_sessions))
        node.agent_epoch = details.get("agent_epoch", node.agent_epoch)

    def _session_from_details(self, details):
        state_value = details.get("state", SessionState.CREATING.value)
        state = SessionState.CACHED_IDLE if state_value in ("WARM_IDLE", "IDLE") else SessionState(state_value)
        raw_gpu_uuids = tuple(details.get("gpu_uuids", ()))
        gpu_count = details.get("gpu_count", len(raw_gpu_uuids))
        gpu_uuids = () if state == SessionState.CACHED_IDLE else raw_gpu_uuids
        enforcement_mode = EnforcementMode.NONE if state == SessionState.CACHED_IDLE else EnforcementMode(details.get("enforcement_mode", EnforcementMode.NONE.value))
        return SessionRecord(
            execution_session_id=details["execution_session_id"],
            pool=details["pool"],
            user_id=details["user_id"],
            session_id=details["session_id"],
            workspace=details.get("workspace"),
            node_id=details["node_id"],
            runtime_adapter=details["runtime_adapter"],
            image_identity=details["image_identity"],
            mounts=tuple(MountSpec(m["source"], m["target"], m.get("read_only", False)) for m in details.get("mounts", ())),
            privileged=details.get("privileged", False),
            gpu_uuids=gpu_uuids,
            cpu_cores=details.get("cpu_cores", 1),
            memory_gb=details.get("memory_gb", 1),
            enforcement_mode=enforcement_mode,
            state=state,
            created_at=details.get("created_at", 0.0),
            last_used_at=details.get("last_used_at", 0.0),
            max_duration_seconds=details.get("max_duration_seconds", 3600),
            active_request_id=details.get("active_request_id"),
            gpu_count=gpu_count,
            isolation_policy=details.get("isolation_policy", "default"),
            state_layer_id=details.get("state_layer_id"),
            suspend_version=details.get("suspend_version", 0),
            suspend_requested=details.get("suspend_requested", False),
            resume_required=details.get("resume_required", False),
            activated_at=details.get("activated_at", details.get("created_at", 0.0)),
            empty_retained=details.get("empty_retained", False),
            gpu_mode=details.get("gpu_mode", "exclusive"),
            gpu_slots=details.get("gpu_slots", 1),
            gpu_cotenant_count=details.get(
                "gpu_cotenant_count", 0
            ),
            gpu_cotenant_counts=details.get(
                "gpu_cotenant_counts", {}
            ),
        )

    def _resources_from_details(self, details):
        return NodeResources(details.get("cpu_cores", 0), details.get("memory_gb", 0), tuple(details.get("gpu_uuids", ())))

    def _capability_from_details(self, details):
        return RuntimeCapabilities(
            adapter=details.get("adapter", "fake"),
            image_formats=tuple(details.get("image_formats", ("default", "docker", "sif", "oci"))),
            isolation_policies=tuple(details.get("isolation_policies", ("default",))),
            nested_container=details.get("nested_container", True),
            can_bind_mounts=details.get("can_bind_mounts", True),
            can_select_gpus=details.get("can_select_gpus", True),
            gpu_enforcement=EnforcementMode(details.get("gpu_enforcement", EnforcementMode.HARD.value)),
            can_enforce_cpu_memory=details.get("can_enforce_cpu_memory", True),
            node_local_cache=details.get("node_local_cache", True),
            supports_warm_sessions=details.get("supports_warm_sessions", True),
            privileged_allowed=details.get("privileged_allowed", False),
            gpu_isolation_policies=(
                tuple(details["gpu_isolation_policies"])
                if "gpu_isolation_policies" in details
                else None
            ),
            reduced_isolation_policies=tuple(
                details.get("reduced_isolation_policies", ())
            ),
        )
