"""RuleBasedPolicy: RFC §5.3 two-stage routing rule.

Routing decision per request:

1. **Fast path (session affinity)**: the primary chosen by consistent
   hashing on `session_id` is tested against three conditions:
   - reachable (not flagged unhealthy);
   - `L_gpu(primary) >= gpu_hit_threshold` (primary has the prefix);
   - `wait(primary) < load_threshold` (primary not overloaded).
   If all three hold, return primary in O(1).

2. **Rule 1 (slow path, GPU-hit + load gate)**: scan all replicas; the
   candidate set is `{r : L_gpu(r) >= gpu_hit_threshold AND wait(r) <
   load_threshold}`. Among candidates, return the one with the largest
   `L_gpu(r)`. Tie-break by smaller `wait(r)`, then by server id.

3. **Rule 2 (Mooncake local CPU hit + load gate)**: if no GPU candidate
   exists, scan Mooncake CPU placement hints. The candidate set is
   `{r : L_cpu(r) >= cpu_hit_threshold AND wait(r) < load_threshold}`.
   Among candidates, return the one with the largest `L_cpu(r)`.

4. **Rule 3 (fallback, least loaded)**: if the tiered rules yield no candidate,
   return `argmin wait(r)` over all replicas.

`L_gpu(r)` and `L_cpu(r)` are maintained as `max` of prefix lengths
previously reported for each `(weight_version, prefix_hash)` on replica r
and tier — see `report_prefixes()`. The per-replica index is an LRU capped
by `max_prefix_entries_per_server`.

Worker-side prefix_signatures and report calls come from verl's
AsyncLLMServerManager (`verl/verl/experimental/agent_loop/agent_loop.py`,
lines 287 and 360 respectively). LLMRouter does not need to add new
hooks — it just needs to honor the contract on the LB side.
"""
from cachetools import LRUCache

from llm_router.policy.base import RouterPolicy

DEFAULT_CACHE_SIZE = 10000
DEFAULT_HIT_THRESHOLD = 1
DEFAULT_LOAD_THRESHOLD = 1024
DEFAULT_MAX_PREFIX_ENTRIES = 8192
GPU_TIER = "gpu"
CPU_TIER = "cpu"
VALID_TIERS = {GPU_TIER, CPU_TIER}


class RuleBasedPolicy(RouterPolicy):
    """Tiered routing rule: GPU hit, Mooncake CPU hit, least-loaded."""

    def __init__(
        self,
        server_ids: list[str],
        max_cache_size: int = DEFAULT_CACHE_SIZE,
        hit_threshold: int = DEFAULT_HIT_THRESHOLD,
        gpu_hit_threshold: int | None = None,
        cpu_hit_threshold: int | None = None,
        load_threshold: int = DEFAULT_LOAD_THRESHOLD,
        max_prefix_entries_per_server: int = DEFAULT_MAX_PREFIX_ENTRIES,
    ):
        super().__init__(server_ids=server_ids)
        self._inflight: dict[str, int] = {sid: 0 for sid in self.server_ids}
        self._session_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        default_hit = max(0, int(hit_threshold))
        self._gpu_hit_threshold = max(
            0,
            int(default_hit if gpu_hit_threshold is None else gpu_hit_threshold),
        )
        self._cpu_hit_threshold = max(
            0,
            int(default_hit if cpu_hit_threshold is None else cpu_hit_threshold),
        )
        # Backward-compatible alias for older tests/debuggers that inspect it.
        self._hit_threshold = self._gpu_hit_threshold
        self._load_threshold = max(0, int(load_threshold))
        # Per-server prefix index: server_id → LRU[(version, prefix_hash) → max_observed_len]
        self._prefix_locations: dict[str, LRUCache] = {
            sid: LRUCache(maxsize=max_prefix_entries_per_server)
            for sid in self.server_ids
        }
        self._cpu_prefix_locations: dict[str, LRUCache] = {
            sid: LRUCache(maxsize=max_prefix_entries_per_server)
            for sid in self.server_ids
        }
        self._tier_locations = {
            GPU_TIER: self._prefix_locations,
            CPU_TIER: self._cpu_prefix_locations,
        }

    # ---- ABC contract ----

    def acquire_server(
        self,
        request_id: str,
        *,
        session_id: str | None = None,
        prefix_signatures: list[tuple[str, str, int]] | None = None,
        **kwargs,
    ) -> str:
        session_id = session_id or request_id

        # Fast path: session consistent hash → check primary.
        primary = self._session_to_server.get(session_id)
        if primary is None:
            primary = self._consistent_hash_server(session_id)
            self._session_to_server[session_id] = primary

        primary_hit = self._prefix_hit_len(primary, prefix_signatures, tier=GPU_TIER)
        if (
            primary_hit >= self._gpu_hit_threshold
            and self._inflight[primary] < self._load_threshold
        ):
            self._inflight[primary] += 1
            return primary

        # Slow path: GPU-hit → Mooncake local-CPU hit → least loaded.
        server_id = (
            self._preferred_server(prefix_signatures, tier=GPU_TIER)
            or self._preferred_server(prefix_signatures, tier=CPU_TIER)
            or self._least_loaded_server()
        )
        self._session_to_server[session_id] = server_id
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
        tier: str = GPU_TIER,
    ) -> None:
        tier = self._normalize_tier(tier)
        locations = self._tier_locations[tier]
        if server_id not in locations:
            raise ValueError(f"Invalid server_id: {server_id}")
        index = locations[server_id]
        for version, prefix_hash, prefix_len in prefix_signatures:
            key = (str(version), str(prefix_hash))
            current = int(index.get(key, 0))
            new_len = int(prefix_len)
            # Record the max observed length per (version, hash) — shorter
            # observations should not overwrite a longer hit.
            if new_len > current:
                index[key] = new_len

    # ---- Internal helpers ----

    def _consistent_hash_server(self, session_id: str) -> str:
        import hashlib

        digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="big") % len(self.server_ids)
        return self.server_ids[idx]

    def _prefix_hit_len(
        self,
        server_id: str,
        prefix_signatures: list[tuple[str, str, int]] | None,
        *,
        tier: str = GPU_TIER,
    ) -> int:
        if not prefix_signatures:
            return 0
        index = self._tier_locations[self._normalize_tier(tier)].get(server_id, {})
        hit_len = 0
        for version, prefix_hash, prefix_len in prefix_signatures:
            cached = int(index.get((str(version), str(prefix_hash)), 0))
            hit_len = max(hit_len, min(cached, int(prefix_len)))
        return hit_len

    def _preferred_server(
        self,
        prefix_signatures: list[tuple[str, str, int]] | None,
        *,
        tier: str,
    ) -> str | None:
        """Pick tier-hit + load-gate candidate with largest hit."""
        threshold = self._threshold_for_tier(tier)
        candidates = []
        for server_id in self.server_ids:
            hit_len = self._prefix_hit_len(server_id, prefix_signatures, tier=tier)
            wait = self._inflight[server_id]
            if hit_len >= threshold and wait < self._load_threshold:
                candidates.append((hit_len, -wait, server_id))
        if not candidates:
            return None
        # max() picks largest hit_len, then largest -wait (= smallest wait),
        # then largest server_id (lexicographic). server_id tiebreak is
        # stable but arbitrary; deterministic for tests.
        return max(candidates)[2]

    def _least_loaded_server(self) -> str:
        return min(self._inflight, key=self._inflight.get)

    def _normalize_tier(self, tier: str) -> str:
        normalized = str(tier or GPU_TIER).lower()
        if normalized not in VALID_TIERS:
            raise ValueError(f"Invalid prefix tier: {tier!r}")
        return normalized

    def _threshold_for_tier(self, tier: str) -> int:
        tier = self._normalize_tier(tier)
        if tier == CPU_TIER:
            return self._cpu_hit_threshold
        return self._gpu_hit_threshold
