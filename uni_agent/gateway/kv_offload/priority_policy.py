"""Priority-aware OOT GPU-to-CPU KV offloading for vLLM v1.

The classes in this module replace scheduler-side admission and CPU eviction
only. vLLM's native ``CPUOffloadingSpec`` still owns block sizing and creates
the native GPU/CPU transfer handlers.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

logger = logging.getLogger("uni_agent.gateway.priority_kv")
logger.setLevel(getattr(logging, os.getenv("PRIORITY_KV_LOG_LEVEL", "INFO").upper(), logging.INFO))


def _event(name: str, level: int = logging.INFO, **fields: Any) -> None:
    if logger.isEnabledFor(level):
        logger.log(level, "[PKV][%s] %s", name, " ".join(f"{key}={value!r}" for key, value in fields.items()))


@dataclass
class _Hint:
    priority: int = 0
    lease_until: float = 0.0
    trajectory_id: str = ""
    touch_seq: int = 0

    def expired(self, now: float | None = None) -> bool:
        return self.lease_until > 0 and self.lease_until <= (time.time() if now is None else now)


def _parse_hint(req_context: ReqContext) -> tuple[_Hint, bool]:
    params = getattr(req_context, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return _Hint(), False
    raw = params.get("agent_hint", params)
    if not isinstance(raw, dict) or "kv_priority" not in raw:
        return _Hint(), False
    try:
        priority = int(raw.get("kv_priority", 0))
        lease_until = float(raw.get("lease_until", 0.0) or 0.0)
    except (TypeError, ValueError):
        return _Hint(), False
    priority = max(0, min(100, priority))
    return _Hint(
        priority=priority,
        lease_until=lease_until,
        trajectory_id=str(raw.get("trajectory_id", "")),
    ), True


class PriorityCachePolicy(CachePolicy):
    """Evict expired, then low-priority, then least-recently-used blocks."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.meta: dict[OffloadKey, _Hint] = {}
        self._seq = 0
        self._current_hint: _Hint | None = None
        self.evict_calls = 0
        self.evicted_blocks = 0
        self.evict_failures = 0
        _event("POLICY_INIT", capacity=cache_capacity)

    def set_request_hint(self, hint: _Hint | None) -> None:
        self._current_hint = hint

    def _tick(self) -> int:
        self._seq += 1
        return self._seq

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        hint = self._current_hint or _Hint()
        self.blocks[key] = block
        self.meta[key] = _Hint(
            priority=hint.priority,
            lease_until=hint.lease_until,
            trajectory_id=hint.trajectory_id,
            touch_seq=self._tick(),
        )
        _event("CPU_INSERT", logging.DEBUG, key=repr(key), priority=hint.priority, trajectory_id=hint.trajectory_id)

    def remove(self, key: OffloadKey) -> None:
        self.blocks.pop(key, None)
        self.meta.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        now = time.time()
        for key in keys:
            if key not in self.blocks:
                continue
            self.blocks.move_to_end(key)
            meta = self.meta[key]
            meta.touch_seq = self._tick()
            hint = self._current_hint
            if hint is None:
                continue
            if meta.lease_until > now:
                # A shared hot prefix must not be downgraded by a lower-value
                # request. Its old value disappears naturally when the lease
                # expires, after which the next touch replaces it.
                meta.priority = max(meta.priority, hint.priority)
                meta.lease_until = max(meta.lease_until, hint.lease_until)
                if not meta.trajectory_id:
                    meta.trajectory_id = hint.trajectory_id
            else:
                meta.priority = hint.priority
                meta.lease_until = hint.lease_until
                meta.trajectory_id = hint.trajectory_id

    def clear(self) -> None:
        self.blocks.clear()
        self.meta.clear()

    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        self.evict_calls += 1
        if n == 0:
            return []
        now = time.time()
        candidates = [
            (key, block)
            for key, block in self.blocks.items()
            if getattr(block, "ref_cnt", 0) == 0 and key not in protected
        ]
        _event("CPU_EVICT_BEGIN", requested=n, candidates=len(candidates), protected=len(protected))
        if len(candidates) < n:
            self.evict_failures += 1
            _event("CPU_EVICT_FAIL", requested=n, candidates=len(candidates))
            return None

        def rank(item: tuple[OffloadKey, BlockStatus]) -> tuple[int, int, int]:
            meta = self.meta.get(item[0], _Hint())
            return (0 if meta.expired(now) else 1, meta.priority, meta.touch_seq)

        victims = sorted(candidates, key=rank)[:n]
        for key, block in victims:
            meta = self.meta.get(key, _Hint())
            _event(
                "CPU_EVICT_VICTIM",
                logging.DEBUG,
                key=repr(key),
                block_id=getattr(block, "block_id", None),
                priority=meta.priority,
                expired=meta.expired(now),
                trajectory_id=meta.trajectory_id,
            )
        for key, _ in victims:
            self.remove(key)
        self.evicted_blocks += len(victims)
        _event("CPU_EVICT_DONE", evicted=len(victims), evicted_total=self.evicted_blocks)
        return victims

    def snapshot(self) -> dict[str, int]:
        now = time.time()
        return {
            "entries": len(self.blocks),
            "evictable": sum(getattr(block, "ref_cnt", 0) == 0 for block in self.blocks.values()),
            "expired": sum(meta.expired(now) for meta in self.meta.values()),
            "evict_calls": self.evict_calls,
            "evicted_blocks": self.evicted_blocks,
            "evict_failures": self.evict_failures,
        }


class PriorityCPUOffloadingManager(CPUOffloadingManager):
    """Native vLLM CPU manager with hint-based admission and eviction."""

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        min_offload_priority: int = 0,
        admit_without_hint: bool = True,
    ) -> None:
        super().__init__(
            num_blocks=num_blocks,
            cache_policy="lru",
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        del cache_policy
        self.min_offload_priority = int(min_offload_priority)
        if not 0 <= self.min_offload_priority <= 100:
            raise ValueError("min_offload_priority must be between 0 and 100")
        self.admit_without_hint = bool(admit_without_hint)
        self._policy = PriorityCachePolicy(cache_capacity=num_blocks)
        self.accepted_store_calls = 0
        self.rejected_store_calls = 0
        self._stats = {
            "lookup_hit": 0,
            "lookup_pending": 0,
            "lookup_miss": 0,
            "load_prepare": 0,
            "load_complete": 0,
            "store_complete": 0,
        }
        _event(
            "MANAGER_INIT",
            num_blocks=num_blocks,
            min_offload_priority=self.min_offload_priority,
            admit_without_hint=self.admit_without_hint,
        )

    def lookup(self, key: OffloadKey, req_context: ReqContext):
        result = super().lookup(key, req_context)
        state = "hit" if result is True else "pending" if result is None else "miss"
        self._stats[f"lookup_{state}"] += 1
        _event(
            "LOOKUP",
            req_id=getattr(req_context, "req_id", ""),
            state=state,
            cpu_cache=self._policy.snapshot(),
        )
        return result

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        keys_list = list(keys)
        self._stats["load_prepare"] += 1
        _event("LOAD_PREPARE", req_id=getattr(req_context, "req_id", ""), keys=len(keys_list))
        return super().prepare_load(keys_list, req_context)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        hint, present = _parse_hint(req_context)
        self._policy.set_request_hint(hint if present else None)
        return super().touch(list(keys), req_context)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        keys_list = list(keys)
        output = super().complete_load(keys_list, req_context)
        self._stats["load_complete"] += 1
        _event(
            "LOAD_COMPLETE",
            req_id=getattr(req_context, "req_id", ""),
            keys=len(keys_list),
            cpu_cache=self._policy.snapshot(),
        )
        return output

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext):
        keys_list = list(keys)
        hint, present = _parse_hint(req_context)
        admitted = (present or self.admit_without_hint) and hint.priority >= self.min_offload_priority
        _event(
            "ADMISSION",
            req_id=getattr(req_context, "req_id", ""),
            admit=admitted,
            priority=hint.priority,
            lease_until=hint.lease_until,
            trajectory_id=hint.trajectory_id,
            requested_keys=len(keys_list),
        )
        if not admitted:
            self.rejected_store_calls += 1
            return super().prepare_store([], req_context)
        self.accepted_store_calls += 1
        self._policy.set_request_hint(hint)
        return super().prepare_store(keys_list, req_context)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        output = super().complete_store(keys, req_context, success=success)
        self._stats["store_complete"] += 1
        _event(
            "STORE_COMPLETE",
            req_id=getattr(req_context, "req_id", ""),
            keys=len(keys),
            success=success,
            cpu_cache=self._policy.snapshot(),
        )
        return output

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "manager": {
                **self._stats,
                "store_admit": self.accepted_store_calls,
                "store_reject": self.rejected_store_calls,
            },
            "policy": self._policy.snapshot(),
            "min_offload_priority": self.min_offload_priority,
            "admit_without_hint": self.admit_without_hint,
        }


class PriorityGPUOffloadingSpec(CPUOffloadingSpec):
    """OOT spec retaining vLLM's native GPU/CPU transfer implementation."""

    def __init__(self, vllm_config, kv_cache_config) -> None:
        super().__init__(vllm_config, kv_cache_config)
        self.min_offload_priority = int(self.extra_config.get("min_offload_priority", 0))
        self.admit_without_hint = bool(self.extra_config.get("admit_without_hint", True))

    def get_manager(self):
        if self._manager is None:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = kv_events_config is not None and kv_events_config.enable_kv_cache_events
            self._manager = PriorityCPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                enable_events=enable_events,
                store_threshold=int(self.extra_config.get("store_threshold", 0)),
                max_tracker_size=int(self.extra_config.get("max_tracker_size", 64_000)),
                min_offload_priority=self.min_offload_priority,
                admit_without_hint=self.admit_without_hint,
            )
            _event("MANAGER_CREATED", manager_type=type(self._manager).__name__, num_blocks=self.num_blocks)
        return self._manager
