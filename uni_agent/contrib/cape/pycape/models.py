# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from .workspace import workspace_selection


ISOLATION_POLICIES = (
    "default",
    "none",
    "host-stateless",
    "host-stateful",
    "image-stateless",
    "image-stateful",
)

STATEFUL_ISOLATION_POLICIES = ("host-stateful", "image-stateful")

POOL_MODES = ("exclusive", "shared")
POOL_PLACEMENTS = ("pack", "spread")
GPU_CONTENTION_POLICIES = (
    "barrier-ratio",
    "ratio-only",
    "exclusive-first",
)

COMMAND_FAILURE_KINDS = ("workload", "infrastructure")


class RequestState(str, Enum):
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    INFEASIBLE = "INFEASIBLE"
    LEASED = "LEASED"
    PREPARING_IMAGE = "PREPARING_IMAGE"
    STARTING_SESSION = "STARTING_SESSION"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    LOST = "LOST"
    RECONCILING = "RECONCILING"


class SessionState(str, Enum):
    CREATING = "CREATING"
    WARM_IDLE = "WARM_IDLE"
    CACHED_IDLE = "CACHED_IDLE"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    PURGING = "PURGING"
    PURGED = "PURGED"
    RUNNING_COMMAND = "RUNNING_COMMAND"
    DRAINING = "DRAINING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    LOST = "LOST"
    RECONCILING = "RECONCILING"


class NodeState(str, Enum):
    REGISTERING = "REGISTERING"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    UNHEALTHY = "UNHEALTHY"
    REMOVING = "REMOVING"
    REMOVED = "REMOVED"
    LOST = "LOST"


class EnforcementMode(str, Enum):
    HARD = "hard"
    ADVISORY = "advisory"
    NONE = "none"


class CommandState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AllocationState(str, Enum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    REMOVING = "REMOVING"
    REMOVED = "REMOVED"
    LOST = "LOST"


class GpuLeaseState(str, Enum):
    FREE = "FREE"
    EXCLUSIVE = "EXCLUSIVE"
    SHARED = "SHARED"


class Role(str, Enum):
    SUBMITTER = "submitter"
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    NODE_AGENT = "node-agent"


REQUEST_TERMINAL_STATES = {
    RequestState.COMPLETED,
    RequestState.FAILED,
    RequestState.CANCELLED,
    RequestState.EXPIRED,
    RequestState.LOST,
    RequestState.INFEASIBLE,
}

SESSION_TERMINAL_STATES = {SessionState.STOPPED, SessionState.FAILED, SessionState.LOST, SessionState.PURGED}

NODE_NON_PLACEMENT_STATES = {
    NodeState.DRAINING,
    NodeState.UNHEALTHY,
    NodeState.REMOVING,
    NodeState.REMOVED,
    NodeState.LOST,
}


def _hashable(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_hashable(item) for item in value))
    return value


class ValueObject(object):
    _fields = ()

    def __eq__(self, other):
        return type(self) is type(other) and all(getattr(self, field) == getattr(other, field) for field in self._fields)

    def __hash__(self):
        return hash(tuple(_hashable(getattr(self, field)) for field in self._fields))

    def __repr__(self):
        fields = ", ".join("{}={!r}".format(field, getattr(self, field)) for field in self._fields)
        return "{}({})".format(type(self).__name__, fields)


class MountSpec(ValueObject):
    _fields = ("source", "target", "read_only")

    def __init__(self, source, target, read_only=False):
        if not source or not target:
            raise ValueError("mount source and target are required")
        self.source = source
        self.target = target
        self.read_only = read_only


class PoolConfig(ValueObject):
    _fields = ("pool_mode", "gpu_slots", "placement")

    def __init__(
        self,
        pool_mode="exclusive",
        gpu_slots=1,
        placement="pack",
    ):
        if pool_mode not in POOL_MODES:
            raise ValueError(
                "pool_mode must be one of: " + ", ".join(POOL_MODES)
            )
        if (
            isinstance(gpu_slots, bool)
            or not isinstance(gpu_slots, int)
            or gpu_slots <= 0
        ):
            raise ValueError("gpu_slots must be a positive integer")
        if pool_mode == "exclusive" and gpu_slots != 1:
            raise ValueError("exclusive pools require gpu_slots=1")
        if placement not in POOL_PLACEMENTS:
            raise ValueError(
                "placement must be one of: "
                + ", ".join(POOL_PLACEMENTS)
            )
        self.pool_mode = pool_mode
        self.gpu_slots = gpu_slots
        self.placement = placement

    def to_dict(self):
        return {
            "pool_mode": self.pool_mode,
            "gpu_slots": self.gpu_slots,
            "placement": self.placement,
        }


class PhysicalGpuLedgerEntry(object):
    def __init__(self, gpu_uuid):
        if not gpu_uuid:
            raise ValueError("gpu_uuid is required")
        self.gpu_uuid = gpu_uuid
        self.exclusive_execution_session_id = None
        self.shared_execution_session_ids = []
        self.shared_capacity = None

    @property
    def state(self):
        if self.exclusive_execution_session_id is not None:
            return GpuLeaseState.EXCLUSIVE
        if self.shared_execution_session_ids:
            return GpuLeaseState.SHARED
        return GpuLeaseState.FREE

    @property
    def shared_refcount(self):
        return len(self.shared_execution_session_ids)

    @property
    def holder_execution_session_ids(self):
        if self.exclusive_execution_session_id is not None:
            return (self.exclusive_execution_session_id,)
        return tuple(self.shared_execution_session_ids)


class ResourceRequest(ValueObject):
    _fields = ("gpus", "cpu_cores", "memory_gb", "max_duration_seconds", "command_timeout_seconds", "hard_gpu_enforcement", "runtime_adapter", "ports", "locality", "isolation_policy")

    def __init__(self, gpus=0, cpu_cores=1, memory_gb=1, max_duration_seconds=3600, command_timeout_seconds=None, hard_gpu_enforcement=False, runtime_adapter=None, ports=(), locality=None, isolation_policy="default"):
        if gpus < 0:
            raise ValueError("gpus must be non-negative")
        if cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if command_timeout_seconds is not None and command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if isolation_policy not in ISOLATION_POLICIES:
            raise ValueError(
                "isolation_policy must be one of: "
                + ", ".join(ISOLATION_POLICIES)
            )
        self.gpus = gpus
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.max_duration_seconds = max_duration_seconds
        self.command_timeout_seconds = command_timeout_seconds
        self.hard_gpu_enforcement = hard_gpu_enforcement
        self.runtime_adapter = runtime_adapter
        self.ports = tuple(ports)
        self.locality = locality
        self.isolation_policy = isolation_policy

    @property
    def admission_timeout_seconds(self):
        if self.command_timeout_seconds is not None:
            return self.command_timeout_seconds
        return self.max_duration_seconds


class WorkloadSpec(ValueObject):
    _fields = (
        "pool",
        "user_id",
        "session_id",
        "command",
        "resources",
        "cwd",
        "env",
        "image",
        "profile",
        "mounts",
        "reuse",
        "keep_session_on_failure",
        "priority",
        "privileged",
        "log_path",
        "artifact_paths",
        "workspace",
    )

    def __init__(
        self,
        pool,
        user_id,
        session_id,
        command,
        resources=None,
        cwd=None,
        env=None,
        image=None,
        profile=None,
        mounts=(),
        reuse=True,
        keep_session_on_failure=True,
        priority=0,
        privileged=False,
        log_path=None,
        artifact_paths=(),
        workspace=None,
    ):
        self.pool = pool
        self.user_id = user_id
        self.session_id = session_id
        self.command = tuple(command)
        self.resources = resources or ResourceRequest()
        self.cwd = cwd
        self.env = dict(env or {})
        self.image = image
        self.profile = profile
        self.mounts = tuple(mounts)
        self.reuse = reuse
        self.keep_session_on_failure = keep_session_on_failure
        self.priority = priority
        self.privileged = privileged
        self.log_path = log_path
        self.artifact_paths = tuple(artifact_paths)
        self.workspace = workspace_selection(workspace)
        self._validate()

    def _validate(self):
        if not self.pool:
            raise ValueError("pool is required")
        if not self.user_id:
            raise ValueError("user_id is required")
        from .session_identity import validate_session_id
        validate_session_id(self.session_id)
        if not self.command:
            raise ValueError("command must not be empty")
        if any(part == "" for part in self.command):
            raise ValueError("command entries must not be empty")
        if self.image and self.profile:
            raise ValueError("specify either image or profile, not both")
        if not isinstance(self.keep_session_on_failure, bool):
            raise ValueError("keep_session_on_failure must be boolean")

    def image_identity(self, default_image):
        if self.image:
            return self.image
        if self.profile:
            return "profile:" + self.profile
        return default_image

    def effective_mounts(self):
        return self.mounts


class RuntimeCapabilities(ValueObject):
    _fields = (
        "adapter",
        "image_formats",
        "isolation_policies",
        "nested_container",
        "can_bind_mounts",
        "can_select_gpus",
        "gpu_enforcement",
        "can_enforce_cpu_memory",
        "node_local_cache",
        "supports_warm_sessions",
        "privileged_allowed",
        "gpu_isolation_policies",
        "reduced_isolation_policies",
    )

    def __init__(
        self,
        adapter="fake",
        image_formats=("default", "docker", "sif", "oci"),
        isolation_policies=("default",),
        nested_container=True,
        can_bind_mounts=True,
        can_select_gpus=True,
        gpu_enforcement=EnforcementMode.HARD,
        can_enforce_cpu_memory=True,
        node_local_cache=True,
        supports_warm_sessions=True,
        privileged_allowed=False,
        gpu_isolation_policies=None,
        reduced_isolation_policies=(),
    ):
        self.adapter = adapter
        self.image_formats = tuple(image_formats)
        isolation_policies = tuple(isolation_policies)
        unknown_policies = set(isolation_policies) - set(ISOLATION_POLICIES)
        if unknown_policies:
            raise ValueError(
                "unknown runtime isolation policies: "
                + ", ".join(sorted(unknown_policies))
            )
        self.isolation_policies = isolation_policies
        if gpu_isolation_policies is None:
            gpu_isolation_policies = (
                isolation_policies if can_select_gpus else ()
            )
        gpu_isolation_policies = tuple(gpu_isolation_policies)
        reduced_isolation_policies = tuple(reduced_isolation_policies)
        for label, policies in (
            ("GPU", gpu_isolation_policies),
            ("reduced-isolation", reduced_isolation_policies),
        ):
            unknown = set(policies) - set(ISOLATION_POLICIES)
            if unknown:
                raise ValueError(
                    "unknown {} policies: {}".format(
                        label, ", ".join(sorted(unknown))
                    )
                )
            unsupported = set(policies) - set(isolation_policies)
            if unsupported:
                raise ValueError(
                    "{} policies are not advertised isolation policies: {}".format(
                        label, ", ".join(sorted(unsupported))
                    )
                )
        self.nested_container = nested_container
        self.can_bind_mounts = can_bind_mounts
        self.can_select_gpus = bool(can_select_gpus and gpu_isolation_policies)
        self.gpu_enforcement = gpu_enforcement
        self.can_enforce_cpu_memory = can_enforce_cpu_memory
        self.node_local_cache = node_local_cache
        self.supports_warm_sessions = supports_warm_sessions
        self.privileged_allowed = privileged_allowed
        self.gpu_isolation_policies = gpu_isolation_policies
        self.reduced_isolation_policies = reduced_isolation_policies


def normalize_runtime_capabilities(capabilities=None, runtime_capabilities=None):
    normalized = {}
    for adapter, adapter_capabilities in dict(runtime_capabilities or {}).items():
        if not isinstance(adapter_capabilities, RuntimeCapabilities):
            raise TypeError("runtime capability entries must be RuntimeCapabilities")
        if adapter != adapter_capabilities.adapter:
            raise ValueError(
                "runtime capability key {!r} does not match adapter {!r}".format(
                    adapter, adapter_capabilities.adapter
                )
            )
        normalized[adapter] = adapter_capabilities
    if capabilities is not None:
        if not isinstance(capabilities, RuntimeCapabilities):
            raise TypeError("capabilities must be RuntimeCapabilities")
        if capabilities.adapter != "multi":
            normalized.setdefault(capabilities.adapter, capabilities)
    if not normalized:
        fallback = capabilities or RuntimeCapabilities()
        normalized[fallback.adapter] = fallback
    return normalized


def aggregate_runtime_capabilities(runtime_capabilities):
    runtime_capabilities = normalize_runtime_capabilities(
        runtime_capabilities=runtime_capabilities
    )
    if len(runtime_capabilities) == 1:
        return next(iter(runtime_capabilities.values()))

    image_formats = set()
    isolation_policies = set()
    gpu_isolation_policies = set()
    reduced_isolation_policies = set()
    for capabilities in runtime_capabilities.values():
        image_formats.update(capabilities.image_formats)
        isolation_policies.update(capabilities.isolation_policies)
        gpu_isolation_policies.update(capabilities.gpu_isolation_policies)
        reduced_isolation_policies.update(
            capabilities.reduced_isolation_policies
        )

    enforcement_rank = {
        EnforcementMode.NONE: 0,
        EnforcementMode.ADVISORY: 1,
        EnforcementMode.HARD: 2,
    }
    gpu_enforcement = max(
        (capabilities.gpu_enforcement for capabilities in runtime_capabilities.values()),
        key=lambda mode: enforcement_rank[mode],
    )
    return RuntimeCapabilities(
        adapter="multi",
        image_formats=tuple(sorted(image_formats)),
        isolation_policies=tuple(
            policy for policy in ISOLATION_POLICIES if policy in isolation_policies
        ),
        nested_container=any(
            capabilities.nested_container
            for capabilities in runtime_capabilities.values()
        ),
        can_bind_mounts=any(
            capabilities.can_bind_mounts
            for capabilities in runtime_capabilities.values()
        ),
        can_select_gpus=any(
            capabilities.can_select_gpus
            for capabilities in runtime_capabilities.values()
        ),
        gpu_enforcement=gpu_enforcement,
        can_enforce_cpu_memory=any(
            capabilities.can_enforce_cpu_memory
            for capabilities in runtime_capabilities.values()
        ),
        node_local_cache=any(
            capabilities.node_local_cache
            for capabilities in runtime_capabilities.values()
        ),
        supports_warm_sessions=any(
            capabilities.supports_warm_sessions
            for capabilities in runtime_capabilities.values()
        ),
        privileged_allowed=any(
            capabilities.privileged_allowed
            for capabilities in runtime_capabilities.values()
        ),
        gpu_isolation_policies=tuple(
            policy
            for policy in ISOLATION_POLICIES
            if policy in gpu_isolation_policies
        ),
        reduced_isolation_policies=tuple(
            policy
            for policy in ISOLATION_POLICIES
            if policy in reduced_isolation_policies
        ),
    )


class NodeResources(ValueObject):
    _fields = ("cpu_cores", "memory_gb", "gpu_uuids")

    def __init__(self, cpu_cores, memory_gb, gpu_uuids=()):
        if cpu_cores < 0:
            raise ValueError("cpu_cores must be non-negative")
        if memory_gb < 0:
            raise ValueError("memory_gb must be non-negative")
        gpu_uuids = tuple(gpu_uuids)
        if len(set(gpu_uuids)) != len(gpu_uuids):
            raise ValueError("gpu_uuids must be unique")
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.gpu_uuids = gpu_uuids


class RequestRecord(object):
    def __init__(self, request_id, spec, state, reason, created_at, updated_at, sequence, execution_session_id=None, node_id=None, gpu_uuids=(), exit_code=None, stdout="", stderr="", log_path=None, artifact_paths=None, gpu_mode=None, gpu_cotenant_count=0, gpu_cotenant_counts=None, started_at=None, ended_at=None, failure_kind=None, leased_at=None):
        self.request_id = request_id
        self.spec = spec
        self.state = state
        self.reason = reason
        self.created_at = created_at
        self.updated_at = updated_at
        self.started_at = started_at
        self.leased_at = leased_at
        self.ended_at = ended_at
        self.sequence = sequence
        self.execution_session_id = execution_session_id
        self.node_id = node_id
        self.gpu_uuids = tuple(gpu_uuids)
        self.exit_code = exit_code
        self.failure_kind = failure_kind
        self.stdout = stdout
        self.stderr = stderr
        self.log_path = log_path if log_path is not None else spec.log_path
        self.artifact_paths = tuple(spec.artifact_paths if artifact_paths is None else artifact_paths)
        self.gpu_mode = gpu_mode
        self.gpu_cotenant_count = int(gpu_cotenant_count or 0)
        self.gpu_cotenant_counts = dict(gpu_cotenant_counts or {})


class CommandAssignment(object):
    def __init__(self, command_id, request_id, execution_session_id, node_id, session_spec, command_spec, state, created_at, updated_at=None, log_path=None, result=None):
        self.command_id = command_id
        self.request_id = request_id
        self.execution_session_id = execution_session_id
        self.node_id = node_id
        self.session_spec = dict(session_spec or {})
        self.command_spec = dict(command_spec or {})
        self.state = state
        self.created_at = created_at
        self.updated_at = updated_at if updated_at is not None else created_at
        self.log_path = log_path
        self.result = dict(result or {})


class AllocationRecord(object):
    def __init__(self, allocation_id, pool, state=AllocationState.ACTIVE, node_ids=(), deadline=None, created_at=0.0, updated_at=0.0, reason=None):
        self.allocation_id = allocation_id
        self.pool = pool
        self.state = state
        self.node_ids = list(node_ids or [])
        self.deadline = deadline
        self.created_at = created_at
        self.updated_at = updated_at
        self.reason = reason


class SessionRecord(object):
    def __init__(
        self,
        execution_session_id,
        pool,
        user_id,
        session_id,
        node_id,
        runtime_adapter,
        image_identity,
        mounts,
        privileged,
        gpu_uuids,
        cpu_cores,
        memory_gb,
        enforcement_mode,
        state,
        created_at,
        last_used_at,
        max_duration_seconds=3600,
        active_request_id=None,
        gpu_count=None,
        isolation_policy="default",
        state_layer_id=None,
        suspend_version=0,
        suspend_requested=False,
        resume_required=False,
        activated_at=None,
        empty_retained=False,
        gpu_mode="exclusive",
        gpu_slots=1,
        gpu_cotenant_count=0,
        gpu_cotenant_counts=None,
        workspace=None,
    ):
        self.execution_session_id = execution_session_id
        self.pool = pool
        self.user_id = user_id
        self.session_id = session_id
        self.node_id = node_id
        self.runtime_adapter = runtime_adapter
        self.image_identity = image_identity
        self.mounts = tuple(mounts)
        self.privileged = privileged
        self.gpu_uuids = tuple(gpu_uuids)
        self.gpu_count = len(self.gpu_uuids) if gpu_count is None else gpu_count
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.enforcement_mode = enforcement_mode
        self.state = state
        self.created_at = created_at
        self.last_used_at = last_used_at
        self.max_duration_seconds = max_duration_seconds
        self.active_request_id = active_request_id
        self.isolation_policy = isolation_policy
        self.state_layer_id = state_layer_id
        self.suspend_version = suspend_version
        self.suspend_requested = suspend_requested
        self.resume_required = resume_required
        self.activated_at = created_at if activated_at is None else activated_at
        self.empty_retained = empty_retained
        self.gpu_mode = gpu_mode
        self.gpu_slots = gpu_slots
        self.gpu_cotenant_count = int(gpu_cotenant_count or 0)
        self.gpu_cotenant_counts = dict(gpu_cotenant_counts or {})
        self.workspace = dict(workspace) if workspace is not None else None

    def reusable_for(self, spec, image_identity, runtime_adapter):
        if self.state not in (SessionState.CACHED_IDLE, SessionState.WARM_IDLE):
            return False
        return self.matches_reuse_identity(spec, image_identity, runtime_adapter)

    def matches_reuse_identity(self, spec, image_identity, runtime_adapter):
        if not spec.reuse:
            return False
        if self.workspace is not None:
            if self.workspace.get("target", "/workspace") != spec.workspace["target"]:
                return False
            if spec.workspace["id"] is not None and spec.workspace["id"] != self.workspace["id"]:
                return False
        elif spec.workspace["id"] is not None:
            return False
        if self.user_id != spec.user_id or self.session_id != spec.session_id:
            return False
        if self.pool != spec.pool:
            return False
        if self.image_identity != image_identity or self.runtime_adapter != runtime_adapter:
            return False
        if spec.resources.runtime_adapter and spec.resources.runtime_adapter != self.runtime_adapter:
            return False
        if self.isolation_policy != spec.resources.isolation_policy:
            return False
        if self.isolation_policy in STATEFUL_ISOLATION_POLICIES and not self.state_layer_id:
            return False
        if self.mounts != spec.effective_mounts() or self.privileged != spec.privileged:
            return False
        if self.gpu_count != spec.resources.gpus:
            return False
        if spec.resources.ports:
            return False
        if self.cpu_cores < spec.resources.cpu_cores or self.memory_gb < spec.resources.memory_gb:
            return False
        return True


class NodeRecord(object):
    def __init__(
        self,
        node_id,
        pool,
        resources,
        capabilities,
        state,
        registered_at,
        last_heartbeat_at,
        allocation_id=None,
        allocation_deadline=None,
        drain_before_sec=0,
        cleanup_buffer_sec=0,
        runtime_cache_config="default",
        cached_images=(),
        used_cpu_cores=0,
        used_memory_gb=0,
        active_sessions=None,
        runtime_capabilities=None,
        agent_epoch=None,
        pool_mode="exclusive",
        gpu_slots=1,
        placement="pack",
        workspace_status=None,
    ):
        self.node_id = node_id
        self.pool = pool
        self.resources = resources
        self.runtime_capabilities = normalize_runtime_capabilities(
            capabilities=capabilities,
            runtime_capabilities=runtime_capabilities,
        )
        self.capabilities = aggregate_runtime_capabilities(
            self.runtime_capabilities
        )
        self.state = state
        self.registered_at = registered_at
        self.last_heartbeat_at = last_heartbeat_at
        self.allocation_id = allocation_id
        self.allocation_deadline = allocation_deadline
        self.drain_before_sec = drain_before_sec
        self.cleanup_buffer_sec = cleanup_buffer_sec
        self.runtime_cache_config = runtime_cache_config
        self.cached_images = tuple(cached_images or ())
        self.used_cpu_cores = used_cpu_cores
        self.used_memory_gb = used_memory_gb
        self.active_sessions = list(active_sessions or [])
        self.agent_epoch = agent_epoch
        config = PoolConfig(
            pool_mode=pool_mode,
            gpu_slots=gpu_slots,
            placement=placement,
        )
        self.pool_mode = config.pool_mode
        self.gpu_slots = config.gpu_slots
        self.placement = config.placement
        self.workspace_status = dict(workspace_status or {})

    def remaining_safe_seconds(self, now):
        if self.allocation_deadline is None:
            return None
        return max(0.0, self.allocation_deadline - now - self.cleanup_buffer_sec)

    def should_drain(self, now):
        if self.allocation_deadline is None:
            return False
        return now >= self.allocation_deadline - self.drain_before_sec

    def available_cpu(self):
        return self.resources.cpu_cores - self.used_cpu_cores

    def available_memory(self):
        return self.resources.memory_gb - self.used_memory_gb


class LeaseCandidate(ValueObject):
    _fields = ("node_id", "gpu_uuids", "reason")

    def __init__(self, node_id, gpu_uuids=(), reason=None):
        self.node_id = node_id
        self.gpu_uuids = tuple(gpu_uuids)
        self.reason = reason
