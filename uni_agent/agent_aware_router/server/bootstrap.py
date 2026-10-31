"""Explicit, process-local registration; also usable as a Ray setup hook."""


def _load_metadata_replica():
    from .http_server import KVMetadataReplica

    return KVMetadataReplica


def register_vllm_metadata_replica() -> None:
    """Install before LLMServerManager creation in the process creating replicas.

    Keep the backend name ``vllm`` so verl's worker and actor-name lookups
    continue to work. PD replicas use a different factory and are unaffected.
    """
    from verl.workers.rollout import replica

    # verl exposes registration but no loader accessor. Inspect the loader
    # without invoking it (which would eagerly import vLLM in every worker).
    current = replica.RolloutReplicaRegistry._registry.get("vllm")
    if current is _load_metadata_replica:
        return
    if current is not replica._load_vllm:
        raise RuntimeError("Cannot install KV metadata replica: another vllm replica loader is registered")
    replica.RolloutReplicaRegistry.register("vllm", _load_metadata_replica)
