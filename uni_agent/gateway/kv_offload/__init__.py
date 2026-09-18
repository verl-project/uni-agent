"""Out-of-tree vLLM KV offload integrations.

Imports are lazy so installing uni-agent does not require importing vLLM in
Gateway-only processes.
"""

__all__ = [
    "DynamicPriorityInputs",
    "PriorityCPUOffloadingManager",
    "PriorityCachePolicy",
    "PriorityDecision",
    "PriorityGPUOffloadingSpec",
    "TrajectoryAwareCachePolicy",
    "compute_dynamic_priority",
]


def __getattr__(name: str):
    if name in {"DynamicPriorityInputs", "PriorityDecision", "compute_dynamic_priority"}:
        from uni_agent.gateway.kv_offload import hints

        return getattr(hints, name)
    if name == "TrajectoryAwareCachePolicy":
        from uni_agent.gateway.kv_offload.trajectory_policy import TrajectoryAwareCachePolicy

        return TrajectoryAwareCachePolicy
    if name in {"PriorityCPUOffloadingManager", "PriorityCachePolicy", "PriorityGPUOffloadingSpec"}:
        from uni_agent.gateway.kv_offload import priority_policy

        return getattr(priority_policy, name)
    raise AttributeError(name)
