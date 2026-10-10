"""Priority-aware OOT GPU-to-CPU KV offloading for vLLM v1.

The classes in this module replace scheduler-side admission and CPU eviction
only. Targets vLLM 0.23.0, as pinned in requirements-test.txt. vLLM's native
``CPUOffloadingSpec`` still owns block sizing and GPU/CPU transfer handlers.
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

from uni_agent.gateway.agent_hint import AgentRuntimeHint
from uni_agent.gateway.kv_offload.hints import AgentHintConfig, compute_dynamic_priority

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
        return self.lease_until > 0 and self.lease_until <= (time.monotonic() if now is None else now)


def _parse_hint(req_context: ReqContext) -> AgentRuntimeHint | None:
    params = getattr(req_context, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    return AgentRuntimeHint.from_dict(params.get("agent_hint"))


class _IndexedHeap:
    """One record per key; updates and removals take O(log N).

    Only ranks are compared, so opaque keys never need ordering support.
    Updating in place prevents stale records from accumulating after touches.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[tuple[int | float, ...], OffloadKey]] = []
        self._positions: dict[OffloadKey, int] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def peek(self) -> tuple[tuple[int | float, ...], OffloadKey]:
        return self._entries[0]

    def _swap(self, left: int, right: int) -> None:
        entries = self._entries
        entries[left], entries[right] = entries[right], entries[left]
        self._positions[entries[left][1]] = left
        self._positions[entries[right][1]] = right

    def _sift_up(self, index: int) -> int:
        while index:
            parent = (index - 1) // 2
            if self._entries[parent][0] <= self._entries[index][0]:
                break
            self._swap(index, parent)
            index = parent
        return index

    def _sift_down(self, index: int) -> None:
        size = len(self._entries)
        while (child := 2 * index + 1) < size:
            if child + 1 < size and self._entries[child + 1][0] < self._entries[child][0]:
                child += 1
            if self._entries[index][0] <= self._entries[child][0]:
                break
            self._swap(index, child)
            index = child

    def update(self, key: OffloadKey, rank: tuple[int | float, ...]) -> None:
        index = self._positions.get(key)
        if index is None:
            index = len(self._entries)
            self._entries.append((rank, key))
            self._positions[key] = index
        else:
            if self._entries[index][0] == rank:
                return
            self._entries[index] = (rank, key)
        self._sift_down(self._sift_up(index))

    def remove(self, key: OffloadKey) -> None:
        index = self._positions.pop(key, None)
        if index is None:
            return
        last = self._entries.pop()
        if index < len(self._entries):
            self._entries[index] = last
            self._positions[last[1]] = index
            self._sift_down(self._sift_up(index))

    def clear(self) -> None:
        self._entries.clear()
        self._positions.clear()


class PriorityCachePolicy(CachePolicy):
    """Evict expired, then low-priority, then least-recently-used blocks.

    Reverse touch order favors prefix heads when priority and expiry tie,
    matching native vLLM LRU. Block-local ranking does not guarantee a
    contiguous prefix across different priorities or protected blocks.
    """

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.meta: dict[OffloadKey, _Hint] = {}
        self._eviction_index = _IndexedHeap()
        self._lease_index = _IndexedHeap()
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

    def _index(self, key: OffloadKey, now: float) -> None:
        meta = self.meta[key]
        self._eviction_index.update(key, (0 if meta.expired(now) else 1, meta.priority, meta.touch_seq))
        if meta.lease_until > now:
            self._lease_index.update(key, (meta.lease_until,))
        else:
            self._lease_index.remove(key)

    def _expire_leases(self, now: float) -> None:
        # Each deadline is processed once unless a later touch renews it.
        while self._lease_index:
            (deadline,), key = self._lease_index.peek()
            if deadline > now:
                break
            self._lease_index.remove(key)
            meta = self.meta[key]
            self._eviction_index.update(key, (0, meta.priority, meta.touch_seq))

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        hint = self._current_hint or _Hint()
        self.blocks[key] = block
        self.meta[key] = _Hint(
            priority=hint.priority,
            lease_until=hint.lease_until,
            trajectory_id=hint.trajectory_id,
            touch_seq=self._tick(),
        )
        self._index(key, time.monotonic())
        _event("CPU_INSERT", logging.DEBUG, key=repr(key), priority=hint.priority, trajectory_id=hint.trajectory_id)

    def remove(self, key: OffloadKey) -> None:
        self._eviction_index.remove(key)
        self._lease_index.remove(key)
        self.blocks.pop(key, None)
        self.meta.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        now = time.monotonic()
        # Native vLLM LRU marks prefix heads most recent within each touch.
        for key in reversed(list(keys)):
            if key not in self.blocks:
                continue
            self.blocks.move_to_end(key)
            meta = self.meta[key]
            meta.touch_seq = self._tick()
            hint = self._current_hint
            if hint is not None and hint.lease_until > now:
                if meta.lease_until > now and hint.priority < meta.priority:
                    # A lower priority touch updates LRU only; it cannot
                    # renew the old priority/deadline claim.
                    pass
                elif meta.lease_until > now and hint.priority == meta.priority:
                    meta.lease_until = max(meta.lease_until, hint.lease_until)
                    if not meta.trajectory_id:
                        meta.trajectory_id = hint.trajectory_id
                else:
                    # An upgrade takes its own deadline. Expired claims
                    # are replaced without inheriting their old lease.
                    meta.priority = hint.priority
                    meta.lease_until = hint.lease_until
                    meta.trajectory_id = hint.trajectory_id
            self._index(key, now)

    def clear(self) -> None:
        self.blocks.clear()
        self.meta.clear()
        self._eviction_index.clear()
        self._lease_index.clear()

    def evict(
        self,
        n: int,
        protected: set[OffloadKey],
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        self.evict_calls += 1
        if n == 0:
            return []
        now = time.monotonic()
        self._expire_leases(now)
        _event("CPU_EVICT_BEGIN", requested=n, protected=len(protected))
        victims: list[tuple[OffloadKey, BlockStatus]] = []
        skipped: list[tuple[tuple[int | float, ...], OffloadKey]] = []
        # The native manager changes ref counts outside this policy. Check
        # them at selection time and restore busy/protected entries afterward.
        while self._eviction_index and len(victims) < n:
            rank, key = self._eviction_index.peek()
            self._eviction_index.remove(key)
            block = self.blocks[key]
            if key not in protected and block.ref_cnt == 0:
                victims.append((key, block))
            else:
                skipped.append((rank, key))
        for rank, key in skipped:
            self._eviction_index.update(key, rank)
        if len(victims) < n:
            # Selection is atomic: retain every block and lease on failure.
            for key, _ in victims:
                meta = self.meta[key]
                self._eviction_index.update(key, (0 if meta.expired(now) else 1, meta.priority, meta.touch_seq))
            self.evict_failures += 1
            _event("CPU_EVICT_FAIL", requested=n, candidates=len(victims))
            return None
        for key, block in victims:
            meta = self.meta[key]
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
        now = time.monotonic()
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
        agent_hint_config: AgentHintConfig | None = None,
    ) -> None:
        if agent_hint_config is not None and not isinstance(agent_hint_config, AgentHintConfig):
            raise ValueError("agent_hint_config must be an AgentHintConfig")
        self.agent_hint_config = agent_hint_config or AgentHintConfig()
        super().__init__(
            num_blocks=num_blocks,
            cache_policy="lru",
            enable_events=enable_events,
            store_threshold=store_threshold,
            max_tracker_size=max_tracker_size,
        )
        del cache_policy
        self.min_offload_priority = self.agent_hint_config.min_offload_priority
        self.admit_without_hint = self.agent_hint_config.admit_without_hint
        self._request_hints: dict[str, _Hint | None] = {}
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

    def _request_hint(self, req_context: ReqContext) -> _Hint | None:
        """Compute once per engine request, including callers without lifecycle hooks."""
        req_id = req_context.req_id
        if req_id not in self._request_hints:
            runtime = _parse_hint(req_context)
            self._request_hints[req_id] = (
                _Hint(
                    priority=compute_dynamic_priority(runtime).priority,
                    lease_until=time.monotonic() + self.agent_hint_config.active_lease_seconds,
                    trajectory_id=runtime.trajectory_id,
                )
                if runtime is not None
                else None
            )
        return self._request_hints[req_id]

    def on_new_request(self, req_context: ReqContext):
        self._request_hint(req_context)
        return super().on_new_request(req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._request_hints.pop(req_context.req_id, None)
        return super().on_request_finished(req_context)

    def lookup(self, key: OffloadKey, req_context: ReqContext):
        result = super().lookup(key, req_context)
        state = "hit" if result is True else "pending" if result is None else "miss"
        self._stats[f"lookup_{state}"] += 1
        _event(
            "LOOKUP",
            logging.DEBUG,
            req_id=getattr(req_context, "req_id", ""),
            state=state,
        )
        return result

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        keys_list = list(keys)
        self._stats["load_prepare"] += 1
        _event("LOAD_PREPARE", req_id=getattr(req_context, "req_id", ""), keys=len(keys_list))
        return super().prepare_load(keys_list, req_context)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        self._policy.set_request_hint(self._request_hint(req_context))
        try:
            return super().touch(list(keys), req_context)
        finally:
            self._policy.set_request_hint(None)

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        keys_list = list(keys)
        output = super().complete_load(keys_list, req_context)
        self._stats["load_complete"] += 1
        _event(
            "LOAD_COMPLETE",
            req_id=getattr(req_context, "req_id", ""),
            keys=len(keys_list),
        )
        return output

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext):
        keys_list = list(keys)
        hint = self._request_hint(req_context)
        present = hint is not None and not hint.expired()
        if not present:
            hint = _Hint()
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
            self._policy.set_request_hint(None)
            return super().prepare_store([], req_context)
        self.accepted_store_calls += 1
        self._policy.set_request_hint(hint)
        try:
            return super().prepare_store(keys_list, req_context)
        finally:
            self._policy.set_request_hint(None)

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
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        legacy = {
            "priority_mode",
            "priority",
            "tool_priority",
            "active_lease_seconds",
            "min_offload_priority",
            "admit_without_hint",
        }
        if legacy.intersection(extra_config):
            raise ValueError("Put policy settings in agent_hint_config; static priority options have been removed")
        self.agent_hint_config = AgentHintConfig.from_dict(extra_config.get("agent_hint_config", {}))
        super().__init__(vllm_config, kv_cache_config)

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
                agent_hint_config=self.agent_hint_config,
            )
            _event("MANAGER_CREATED", manager_type=type(self._manager).__name__, num_blocks=self.num_blocks)
        return self._manager
