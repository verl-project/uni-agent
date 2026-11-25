"""Optional metadata access for verl's vLLM servers.

Call ``register_vllm_metadata_replica`` before creating replicas.
"""


def _load_metadata_replica():
    from .http_server import KVMetadataReplica

    return KVMetadataReplica


def register_vllm_metadata_replica() -> None:
    """Install before LLMServerManager creation in the process creating replicas.

    Keep the backend name ``vllm`` so verl's worker and actor-name lookups
    continue to work. This adapter checks ``RolloutReplicaRegistry._registry``
    for verl's default ``replica._load_vllm`` loader, then its replica sets
    ``vLLMReplica.server_class`` to the metadata-enabled actor. PD replicas
    use a different factory.
    """
    from verl.workers.rollout import replica

    # verl exposes registration but no loader accessor. Inspect the loader
    # without invoking it (which would eagerly import vLLM in every worker).
    current = replica.RolloutReplicaRegistry._registry.get("vllm")
    if current is _load_metadata_replica:
        return
    if current is not replica._load_vllm:
        raise RuntimeError(f"Cannot install KV metadata replica: another vllm replica loader is active: {current!r}")
    replica.RolloutReplicaRegistry.register("vllm", _load_metadata_replica)
