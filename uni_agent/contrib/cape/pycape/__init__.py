# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CAPE: Container-Aware Placement Engine."""

from .auth import JoinToken, Role, TokenAuthority
from .controller import CAPEController, SchedulerPolicy
from .agent import NodeAgentClient, local_node_resources, local_runtime_capabilities
from .journal import EventRecord, MemoryJournal, SQLiteJournal
from .metrics import MemoryMetricsSink, NoopMetricsSink
from .node_agent import InProcessNodeAgent
from .observability import JsonlExporter, MetricsCollector, NoopExporter, PoolUtilizationSummary, WandbExporter, summarize_pool
from .profiles import EnvironmentProfile, ProfileRegistry
from .service import CAPEControllerService
from .models import (
    EnforcementMode,
    GpuLeaseState,
    MountSpec,
    NodeRecord,
    NodeResources,
    NodeState,
    PhysicalGpuLedgerEntry,
    PoolConfig,
    RequestRecord,
    RequestState,
    ResourceRequest,
    RuntimeCapabilities,
    SessionRecord,
    SessionState,
    WorkloadSpec,
)
from .runtime import (
    ApptainerRuntimeAdapter,
    BubbleWrapRuntimeAdapter,
    ContainerCommandSpec,
    ContainerRuntimeAdapter,
    ContainerSessionSpec,
    FakeRuntimeAdapter,
    LocalProcessRuntimeAdapter,
    ManagedProcess,
    PodmanRuntimeAdapter,
    RuntimeLaunch,
    RuntimeExecution,
    RuntimeResult,
    terminate_process_tree,
)

__all__ = [
    "CAPEController",
    "terminate_process_tree",
    "RuntimeLaunch",
    "RuntimeExecution",
    "PodmanRuntimeAdapter",
    "ManagedProcess",
    "ContainerSessionSpec",
    "ContainerRuntimeAdapter",
    "ContainerCommandSpec",
    "ApptainerRuntimeAdapter",
    "BubbleWrapRuntimeAdapter",
    "EnforcementMode",
    "GpuLeaseState",
    "EventRecord",
    "FakeRuntimeAdapter",
    "InProcessNodeAgent",
    "JoinToken",
    "LocalProcessRuntimeAdapter",
    "MemoryJournal",
    "MemoryMetricsSink",
    "MountSpec",
    "NodeRecord",
    "NodeResources",
    "NodeState",
    "PhysicalGpuLedgerEntry",
    "PoolConfig",
    "NoopMetricsSink",
    "RequestRecord",
    "RequestState",
    "ResourceRequest",
    "Role",
    "RuntimeCapabilities",
    "RuntimeResult",
    "SchedulerPolicy",
    "SessionRecord",
    "SessionState",
    "SQLiteJournal",
    "TokenAuthority",
    "WorkloadSpec",
    "CAPEControllerService",
    "EnvironmentProfile",
    "JsonlExporter",
    "MetricsCollector",
    "NodeAgentClient",
    "NoopExporter",
    "PoolUtilizationSummary",
    "ProfileRegistry",
    "WandbExporter",
    "local_node_resources",
    "local_runtime_capabilities",
    "summarize_pool",
]
