"""Trajectory-aware CPU KV eviction policy for Gateway-managed vLLM offloading.

The engine owns transfer safety through ``ref_cnt`` and ``protected``. This
policy only chooses among blocks that vLLM has already declared evictable.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


@dataclass
class _BlockMeta:
    trajectory_id: str | None = None
    lease_until: float = 0.0
    reuse_probability: float = 0.0
    touch_seq: int = 0
    position: int = 0


class TrajectoryAwareCachePolicy(CachePolicy):
    """Lease-aware, chain-preserving replacement policy.

    Blocks without uni-agent hints behave like LRU entries. Within a hinted
    trajectory, suffix blocks are evicted first so the remaining CPU data stays
    a useful contiguous prefix.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self._blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self._meta: dict[OffloadKey, _BlockMeta] = {}
        self._seq = 0

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self._blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self._blocks[key] = block
        self._meta[key] = _BlockMeta(touch_seq=self._seq)

    def remove(self, key: OffloadKey) -> None:
        self._blocks.pop(key, None)
        self._meta.pop(key, None)

    @staticmethod
    def _hint(req_context: ReqContext) -> dict[str, Any]:
        params = getattr(req_context, "kv_transfer_params", None) or {}
        hint = params.get("uni_agent", {})
        return hint if isinstance(hint, dict) and hint.get("schema_version") == 1 else {}

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        touched = list(keys)
        hint = self._hint(req_context)
        trajectory_id = hint.get("trajectory_id")
        lease_until = float(hint.get("lease_until", 0.0))
        reuse_probability = min(1.0, max(0.0, float(hint.get("reuse_probability", 0.0))))
        self._seq += 1
        for position, key in enumerate(touched):
            block = self._blocks.get(key)
            if block is None:
                continue
            self._blocks.move_to_end(key)
            meta = self._meta[key]
            meta.touch_seq = self._seq
            meta.position = position
            if trajectory_id is not None:
                meta.trajectory_id = str(trajectory_id)
                meta.lease_until = lease_until
                meta.reuse_probability = reuse_probability

    def clear(self) -> None:
        self._blocks.clear()
        self._meta.clear()

    def _eviction_rank(self, key: OffloadKey, now: float) -> tuple:
        meta = self._meta[key]
        leased = meta.lease_until > now
        # Lower tuple = cheaper to discard. Negative position makes a chain's
        # suffix leave before its prefix; touch_seq provides the LRU fallback.
        return (
            leased,
            meta.reuse_probability if leased else 0.0,
            meta.touch_seq,
            meta.trajectory_id or "",
            -meta.position,
        )

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        now = time.time()
        candidates = [
            key
            for key, block in self._blocks.items()
            if block.ref_cnt == 0 and key not in protected
        ]
        if len(candidates) < n:
            return None
        selected = sorted(candidates, key=lambda key: self._eviction_rank(key, now))[:n]
        result = [(key, self._blocks[key]) for key in selected]
        for key in selected:
            self.remove(key)
        return result
