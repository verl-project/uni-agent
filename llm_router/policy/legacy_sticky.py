"""LegacyStickyPolicy: behavior-preserving port of verl's GlobalRequestLoadBalancer.

This is the baseline policy used when context-aware scheduling is disabled. It
guarantees the same routing decisions as the original verl implementation so
that switching the manager via `agent_loop_manager_class` FQN is a no-op.
"""
from cachetools import LRUCache

from llm_router.policy.base import RouterPolicy

DEFAULT_CACHE_SIZE = 10000


class LegacyStickyPolicy(RouterPolicy):
    """Multi-turn sticky session + least-in-flight first-time assignment."""

    def __init__(self, server_ids: list[str], max_cache_size: int = DEFAULT_CACHE_SIZE):
        super().__init__(server_ids=server_ids)
        self._inflight: dict[str, int] = {sid: 0 for sid in self.server_ids}
        self._cache: LRUCache = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str, **_) -> str:
        if request_id in self._cache:
            server_id = self._cache[request_id]
            self._inflight[server_id] += 1
            return server_id
        server_id = min(self._inflight, key=self._inflight.get)
        self._cache[request_id] = server_id
        self._inflight[server_id] += 1
        return server_id

    def release_server(self, server_id: str) -> None:
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id: {server_id}")
        if self._inflight[server_id] <= 0:
            raise ValueError(f"Release with no in-flight on server {server_id}")
        self._inflight[server_id] -= 1

    def report_prefixes(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]],
        *,
        tier: str = "gpu",
    ) -> None:
        """No-op. Legacy sticky binding does not use prefix locations.

        Validates `server_id` to surface config errors early — silent no-ops
        on bogus ids would hide trainer misconfiguration.
        """
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id: {server_id}")
