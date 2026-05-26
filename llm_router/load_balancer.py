"""Ray-remote wrapper around a RouterPolicy."""
import ray

from llm_router.policy import LegacyStickyPolicy, RouterPolicy, RuleBasedPolicy

_POLICY_BUILDERS: dict[str, type[RouterPolicy]] = {
    "legacy_sticky": LegacyStickyPolicy,
    "rule_based": RuleBasedPolicy,
}


def _build_policy(
    server_ids: list[str],
    policy_name: str,
    routing_cache_size: int = 10000,
    hit_threshold: int = 1,
    gpu_hit_threshold: int | None = None,
    cpu_hit_threshold: int | None = None,
    load_threshold: int = 1024,
    max_prefix_entries_per_server: int = 8192,
) -> RouterPolicy:
    if policy_name not in _POLICY_BUILDERS:
        raise ValueError(f"unknown policy: {policy_name!r}")
    cls = _POLICY_BUILDERS[policy_name]
    if cls is RuleBasedPolicy:
        return cls(
            server_ids=server_ids,
            max_cache_size=routing_cache_size,
            hit_threshold=hit_threshold,
            gpu_hit_threshold=gpu_hit_threshold,
            cpu_hit_threshold=cpu_hit_threshold,
            load_threshold=load_threshold,
            max_prefix_entries_per_server=max_prefix_entries_per_server,
        )
    return cls(server_ids=server_ids, max_cache_size=routing_cache_size)


@ray.remote
class LoadBalancer:
    """Single actor that owns the RouterPolicy state for the entire job."""

    def __init__(
        self,
        server_ids: list[str],
        policy_name: str = "legacy_sticky",
        routing_cache_size: int = 10000,
        hit_threshold: int = 1,
        gpu_hit_threshold: int | None = None,
        cpu_hit_threshold: int | None = None,
        load_threshold: int = 1024,
        max_prefix_entries_per_server: int = 8192,
        server_aliases: dict[str, str | list[str]] | None = None,
    ):
        self._server_ids = list(server_ids)
        self._server_aliases = self._normalize_aliases(server_aliases or {})
        self._policy = _build_policy(
            server_ids=server_ids,
            policy_name=policy_name,
            routing_cache_size=routing_cache_size,
            hit_threshold=hit_threshold,
            gpu_hit_threshold=gpu_hit_threshold,
            cpu_hit_threshold=cpu_hit_threshold,
            load_threshold=load_threshold,
            max_prefix_entries_per_server=max_prefix_entries_per_server,
        )

    def acquire_server(self, request_id: str, **kwargs) -> str:
        return self._policy.acquire_server(request_id, **kwargs)

    def release_server(self, server_id: str) -> None:
        self._policy.release_server(server_id)

    def report_prefixes(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]],
        *,
        tier: str = "gpu",
    ) -> None:
        for resolved_id in self._resolve_server_ids(server_id):
            self._policy.report_prefixes(resolved_id, prefix_signatures, tier=tier)

    def _normalize_aliases(
        self,
        aliases: dict[str, str | list[str]],
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {sid: [sid] for sid in self._server_ids}
        for alias, targets in aliases.items():
            if isinstance(targets, str):
                target_list = [targets]
            else:
                target_list = [str(t) for t in targets]
            normalized[str(alias)] = [
                target for target in target_list if target in self._server_ids
            ]
        return normalized

    def _resolve_server_ids(self, server_id: str) -> list[str]:
        resolved = self._server_aliases.get(str(server_id), [])
        if not resolved:
            # Let the policy raise its normal Invalid server_id error.
            return [server_id]
        return resolved
