"""PD decode-node scheduling strategies (uni-agent side, one-shot select).

Each strategy is a plain class with the uni-agent selector interface:
``select(req, candidates) -> DecodeCandidate``. ``SelectorRouter`` exposes
these policies through verl's standard request-load-balancer protocol.
The cache-aware and hash strategies are ported from the private repo's
``pd_routing.py`` (token radix tree + FBI/murmur hash), keeping the original
selection semantics and maintaining local select/release load estimates.

A strategy that keeps cross-request state may additionally implement the
optional ``reset()`` hook. verl calls it when the decode replicas' prefix cache
was invalidated (weight update / sleep-wake / explicit KV-cache clear), so the
state must not keep routing by prefix ownership that no longer exists. Only
``CacheAwareDecodeSelector`` needs it today.
"""

from __future__ import annotations

import bisect
import logging
import random
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class DecodeCandidate:
    """A decode server exposed to the uni-agent selector implementations."""

    key: str
    handle: Any


# ── FBI / murmur hash (ported from pd_routing.py) ──────────────────────────


def _murmur_hash_64a(key: bytes, seed: int = 4193360111) -> int:
    mask = (1 << 64) - 1
    multiplier = 0xC6A4A7935BD1E995
    shift = 47
    value = (seed ^ (len(key) * multiplier)) & mask
    chunk_end = len(key) - len(key) % 8
    for offset in range(0, chunk_end, 8):
        chunk = int.from_bytes(key[offset : offset + 8], "little")
        chunk = (chunk * multiplier) & mask
        chunk ^= chunk >> shift
        chunk = (chunk * multiplier) & mask
        value ^= chunk
        value = (value * multiplier) & mask
    remainder = key[chunk_end:]
    for index, byte in enumerate(remainder):
        value ^= byte << (index * 8)
    if remainder:
        value = (value * multiplier) & mask
    value ^= value >> shift
    value = (value * multiplier) & mask
    value ^= value >> shift
    return value & mask


def _murmur_rehash_64a(key: int) -> int:
    return _murmur_hash_64a(key.to_bytes(8, "little"))


def _furc_get_bit(key: bytes, index: int, hash_cache: list[int]) -> bool:
    order = index >> 6
    while len(hash_cache) <= order:
        hash_cache.append(_murmur_hash_64a(key) if not hash_cache else _murmur_rehash_64a(hash_cache[-1]))
    return bool((hash_cache[order] >> (index & 0x3F)) & 1)


def _furc_hash(key: bytes, modulus: int) -> int:
    if modulus <= 1:
        return 0
    depth = 0
    while modulus > 1 << depth:
        depth += 1
    index = depth
    hash_cache: list[int] = []
    for _ in range(32):
        while not _furc_get_bit(key, index, hash_cache):
            if depth == 0:
                return 0
            depth -= 1
            index = depth
        index += 23
        value = 1
        for _ in range(max(0, depth - 1)):
            value = (value << 1) | _furc_get_bit(key, index, hash_cache)
            index += 23
        if value < modulus:
            return value
    return 0


def _fbi_hash(key: str) -> int:
    furc_result = _furc_hash(key.encode(), (1 << 23) - 1)
    return _murmur_hash_64a(furc_result.to_bytes(4, "little"))


# ── Token radix tree (ported from pd_routing.py) ──────────────────────────


@dataclass(eq=False)
class _RadixNode:
    text: tuple[int, ...]
    parent: _RadixNode | None
    children: dict[int, _RadixNode] = field(default_factory=dict)
    tenant_access: dict[int, int] = field(default_factory=dict)


class _TokenRadixTree:
    """Compressed, multi-tenant radix tree adapted from vLLM Router's Tree."""

    def __init__(self, num_tenants: int):
        self._epoch = 0
        self.root = _RadixNode((), None, tenant_access={tenant: 0 for tenant in range(num_tenants)})
        self.tenant_sizes = [0] * num_tenants

    @staticmethod
    def _shared_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
        return next(
            (index for index, pair in enumerate(zip(left, right, strict=False)) if pair[0] != pair[1]),
            min(len(left), len(right)),
        )

    def _next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    def insert(self, token_ids: Sequence[int], tenant: int) -> None:
        tokens = tuple(token_ids)
        epoch = self._next_epoch()
        if not tokens:
            self.root.tenant_access[tenant] = epoch
            return

        parent = self.root
        offset = 0
        while offset < len(tokens):
            first_token = tokens[offset]
            child = parent.children.get(first_token)
            if child is None:
                segment = tokens[offset:]
                parent.children[first_token] = _RadixNode(segment, parent, tenant_access={tenant: epoch})
                self.tenant_sizes[tenant] += len(segment)
                return

            shared = self._shared_prefix_len(tokens[offset:], child.text)
            if shared < len(child.text):
                prefix = child.text[:shared]
                suffix = child.text[shared:]
                split = _RadixNode(prefix, parent, {suffix[0]: child}, dict(child.tenant_access))
                parent.children[first_token] = split
                child.text = suffix
                child.parent = split

                if tenant not in split.tenant_access:
                    split.tenant_access[tenant] = 0
                    self.tenant_sizes[tenant] += len(prefix)

                offset += shared
                if offset == len(tokens):
                    split.tenant_access[tenant] = epoch
                    return

                segment = tokens[offset:]
                split.children[segment[0]] = _RadixNode(segment, split, tenant_access={tenant: epoch})
                self.tenant_sizes[tenant] += len(segment)
                return

            if tenant not in child.tenant_access:
                child.tenant_access[tenant] = 0
                self.tenant_sizes[tenant] += len(child.text)
            offset += shared
            parent = child

        parent.tenant_access[tenant] = epoch

    def prefix_match(self, token_ids: Sequence[int]) -> tuple[int, int]:
        node = self.root
        offset = 0
        while offset < len(token_ids):
            child = node.children.get(token_ids[offset])
            if child is None:
                break
            shared = self._shared_prefix_len(token_ids[offset:], child.text)
            offset += shared
            node = child
            if shared < len(child.text):
                break
        return max(node.tenant_access, key=node.tenant_access.get), offset

    def evict_tenant_by_size(self, tenant: int, max_size: int) -> None:
        while self.tenant_sizes[tenant] > max_size:
            leaves: list[_RadixNode] = []
            stack = list(self.root.children.values())
            while stack:
                node = stack.pop()
                stack.extend(node.children.values())
                if tenant in node.tenant_access and not any(
                    tenant in child.tenant_access for child in node.children.values()
                ):
                    leaves.append(node)
            if not leaves:
                return

            leaf = min(leaves, key=lambda node: node.tenant_access[tenant])
            del leaf.tenant_access[tenant]
            self.tenant_sizes[tenant] = max(0, self.tenant_sizes[tenant] - len(leaf.text))
            if not leaf.tenant_access and not leaf.children and leaf.parent is not None:
                del leaf.parent.children[leaf.text[0]]


# ── Strategies ────────────────────────────────────────────────────────────


class LeastLoadDecodeSelector:
    """Power-of-two choices over instance-local in-flight counters.

    ``select`` increments the chosen candidate's local count; the caller uses
    ``release(index)`` to decrement it on completion. No load RPC is required.
    Changing candidate identities resets the counts, even when the candidate
    count is unchanged, so array indices cannot retain another server's load.
    """

    def __init__(self, sample_size: int = 2) -> None:
        self._sample_size = max(1, int(sample_size))
        self._candidate_keys: tuple[str, ...] = ()
        self._pending_requests: list[int] = []

    def select(self, req: dict, candidates: list[DecodeCandidate]) -> DecodeCandidate:
        if not candidates:
            raise ValueError("no decode candidates available")
        candidate_keys = tuple(candidate.key for candidate in candidates)
        if candidate_keys != self._candidate_keys:
            self._candidate_keys = candidate_keys
            self._pending_requests = [0] * len(candidates)
        if len(candidates) == 1:
            self._pending_requests[0] += 1
            return candidates[0]
        sampled = random.sample(range(len(candidates)), min(self._sample_size, len(candidates)))
        chosen = min(sampled, key=self._pending_requests.__getitem__)
        self._pending_requests[chosen] += 1
        return candidates[chosen]

    def release(self, index: int) -> None:
        if 0 <= index < len(self._pending_requests) and self._pending_requests[index] > 0:
            self._pending_requests[index] -= 1


class CacheAwareDecodeSelector:
    """Token-prefix radix tree affinity with a least-load fallback."""

    def __init__(
        self,
        # Defaults aligned with the reference cache_aware policy
        # (vLLM Router): 0.5 / 32 / 1.1. The old 0.3 / 64 / 1.5 pair made the
        # balance gate unreachable at max_concurrent_sessions=96 (a 3591:1
        # skew was never corrected).
        cache_threshold: float = 0.5,
        balance_abs_threshold: int = 32,
        balance_rel_threshold: float = 1.1,
        eviction_interval_secs: int = 120,
        max_tree_size: int = 2**26,
    ) -> None:
        self._cache_threshold = float(cache_threshold)
        self._balance_abs_threshold = int(balance_abs_threshold)
        self._balance_rel_threshold = float(balance_rel_threshold)
        self._eviction_interval_secs = int(eviction_interval_secs)
        self._max_tree_size = int(max_tree_size)
        self._tree: _TokenRadixTree | None = None
        self._num_tenants = 0
        self._candidate_keys: tuple[str, ...] = ()
        self._pending_requests: list[int] = []
        self._last_eviction = time.monotonic()

    def select(self, req: dict, candidates: list[DecodeCandidate]) -> DecodeCandidate:
        if not candidates:
            raise ValueError("no decode candidates available")
        prompt_ids = req.get("prompt_ids") if isinstance(req, dict) else None
        if not prompt_ids:
            raise ValueError("cache_aware requires prompt_ids")

        candidate_keys = tuple(candidate.key for candidate in candidates)
        if candidate_keys != self._candidate_keys:
            self._candidate_keys = candidate_keys
            self._pending_requests = [0] * len(candidates)
            self._tree = None
        if self._tree is None:
            self._tree = _TokenRadixTree(len(candidates))
            self._num_tenants = len(candidates)
        self._evict_cache_if_due()

        candidate_loads = self._pending_requests
        min_load = min(candidate_loads)
        max_load = max(candidate_loads)
        imbalanced = (
            max_load - min_load > self._balance_abs_threshold and max_load > min_load * self._balance_rel_threshold
        )

        if imbalanced:
            selected = min(range(len(candidates)), key=candidate_loads.__getitem__)
        else:
            matched_tenant, matched_tokens = self._tree.prefix_match(prompt_ids)
            match_rate = matched_tokens / len(prompt_ids) if prompt_ids else 0.0
            if match_rate > self._cache_threshold and matched_tenant < len(candidates):
                selected = matched_tenant
            else:
                selected = min(range(len(candidates)), key=candidate_loads.__getitem__)

        self._tree.insert(prompt_ids, selected)
        self._pending_requests[selected] += 1
        return candidates[selected]

    def release(self, index: int) -> None:
        if 0 <= index < len(self._pending_requests) and self._pending_requests[index] > 0:
            self._pending_requests[index] -= 1

    def reset(self) -> None:
        """Drop cached prefix affinity (rebuilt lazily on the next ``select``).

        Mirrors the old ``DecodePeerSelector.clear_cache()``, which the PD
        wiring called on weight update / wake_up / sleep / clear_kv_cache.
        """
        # In-flight requests still release their counters after cache invalidation.
        self._tree = None
        self._num_tenants = 0
        self._last_eviction = time.monotonic()

    def _evict_cache_if_due(self) -> None:
        if self._tree is None:
            return
        interval = self._eviction_interval_secs
        now = time.monotonic()
        if interval <= 0 or now - self._last_eviction < interval:
            return
        for tenant in range(self._num_tenants):
            self._tree.evict_tenant_by_size(tenant, self._max_tree_size)
        self._last_eviction = now


class RandomDecodeSelector:
    def select(self, req: dict, candidates: list[DecodeCandidate]) -> DecodeCandidate:
        if not candidates:
            raise ValueError("no decode candidates available")
        return random.choice(candidates)


class ConsistentHashDecodeSelector:
    """Virtual-node consistent hashing over ``routing_key`` with FBI hash."""

    def __init__(self, virtual_nodes: int = 160) -> None:
        self._virtual_nodes = max(1, int(virtual_nodes))
        self._ring_keys: tuple[str, ...] | None = None
        self._ring: list[tuple[int, int]] = []
        self._points: list[int] = []

    def select(self, req: dict, candidates: list[DecodeCandidate]) -> DecodeCandidate:
        if not candidates:
            raise ValueError("no decode candidates available")
        routing_key = (req or {}).get("routing_key") or (req or {}).get("request_id") or ""
        self._ensure_ring(candidates)
        position = bisect.bisect_left(self._points, _fbi_hash(str(routing_key)))
        if position == len(self._ring):
            position = 0
        index = self._ring[position][1]
        return candidates[index]

    def _ensure_ring(self, candidates: list[DecodeCandidate]) -> None:
        keys = tuple(c.key for c in candidates)
        if keys == self._ring_keys:
            return
        ring = sorted(
            (_fbi_hash(f"{key}:{virtual_node}"), index)
            for index, key in enumerate(keys)
            for virtual_node in range(self._virtual_nodes)
        )
        self._ring = ring
        self._points = [point for point, _ in ring]
        self._ring_keys = keys


class RendezvousHashDecodeSelector:
    """Highest-random-weight (rendezvous) hashing with FBI hash."""

    def select(self, req: dict, candidates: list[DecodeCandidate]) -> DecodeCandidate:
        if not candidates:
            raise ValueError("no decode candidates available")
        routing_key = (req or {}).get("routing_key") or (req or {}).get("request_id") or ""
        return max(candidates, key=lambda c: _fbi_hash(f"{routing_key}:{c.key}"))


# ---- per-replica pressure snapshots (used by the overload guard) ----


@dataclass
class _PressureSnapshot:
    """Latest relevant ``/metrics`` readings for one decode replica."""

    ts: float
    kv_usage: float | None = None
    running: float | None = None
    wait_capacity: float | None = None


def _metric_value(line: str) -> float | None:
    try:
        return float(line.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return None


class _MetricsPressureCache:
    """Poll decode replicas' ``/metrics`` on a daemon thread and cache snapshots.

    A candidate key is its HTTP base address, such as ``http://host:port``.
    The metrics endpoint uses the same address, so polling needs no additional
    metrics RPC in the companion verl integration.

    Read ``vllm:kv_cache_usage_perc``, ``vllm:num_requests_running`` and
    ``vllm:num_requests_waiting_by_reason{reason="capacity"}``.
    Failed fetches retain the previous snapshot until it becomes stale.
    ``snapshot()`` only reads the cache, with no network I/O on the event loop.
    """

    def __init__(
        self,
        interval_secs: float = 1.0,
        timeout_secs: float = 0.15,
        stale_secs: float = 3.0,
    ) -> None:
        self._interval = max(0.05, float(interval_secs))
        self._timeout = max(0.01, float(timeout_secs))
        self._stale = max(self._interval, float(stale_secs))
        self._lock = threading.Lock()
        self._snapshots: dict[str, _PressureSnapshot] = {}
        self._keys: tuple[str, ...] = ()
        self._thread: threading.Thread | None = None

    def observe(self, keys: Sequence[str]) -> None:
        """Update the keys to poll, starting the thread on the first call."""
        keys = tuple(keys)
        if keys == self._keys:
            return
        self._keys = keys
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="pd-decode-pressure-cache", daemon=True)
            self._thread.start()

    def snapshot(self, key: str) -> _PressureSnapshot | None:
        with self._lock:
            snap = self._snapshots.get(key)
        if snap is None or (time.monotonic() - snap.ts) > self._stale:
            return None
        return snap

    def _loop(self) -> None:
        while True:
            for key in self._keys:
                snap = self._fetch(key)
                if snap is not None:
                    with self._lock:
                        self._snapshots[key] = snap
            time.sleep(self._interval)

    def _fetch(self, key: str) -> _PressureSnapshot | None:
        url = key.rstrip("/") + "/metrics"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                body = resp.read(200_000).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - Network errors or sleeping engines yield no fresh snapshot.
            return None
        kv_usage = running = wait_capacity = None
        for line in body.splitlines():
            if not line.startswith("vllm:"):
                continue
            if line.startswith("vllm:kv_cache_usage_perc"):
                kv_usage = _metric_value(line)
            elif line.startswith("vllm:num_requests_running"):
                running = _metric_value(line)
            elif line.startswith("vllm:num_requests_waiting_by_reason") and 'reason="capacity"' in line:
                wait_capacity = _metric_value(line)
        if kv_usage is None and running is None and wait_capacity is None:
            return None
        return _PressureSnapshot(
            ts=time.monotonic(),
            kv_usage=kv_usage,
            running=running,
            wait_capacity=wait_capacity,
        )


class SessionStickyDecodeSelector:
    """Session-scoped sticky routing with a light overload guard.

    Why this exists
    ---------------
    ``CacheAwareDecodeSelector`` keys affinity on the *prompt prefix*. In this
    workload every session shares ~96-100% of its prompt (system prompt + tool
    definitions), so the radix tree hands out a single owner for all sessions and
    traffic collapses onto one decode replica (measured: 4.7:1 request skew and a
    92.7% KV peak while the other two replicas sat idle). Keying affinity on the
    *session* keeps the useful part (cross-turn KV reuse) without the collapse:

    * ``routing_key`` (session id) -> rendezvous hash -> one replica per session,
      stable across all of that session's turns (prefix reuse where it matters);
    * different sessions hash evenly across replicas, so a shared system-prefix
      can no longer drag every session onto the same replica;
    * if the sticky owner is overloaded, fall back to the least loaded candidate.

    Overload is decided by explicit hard thresholds only (KV watermark, queue
    depth, in-flight prompt tokens). The soft scores are used exclusively to rank
    fallback candidates, so the guard cannot silently swallow the sticky path the
    way the previous token-budgeted variant did (its budget was ~10x below the real in-flight
    token volume, so every replica looked hot on every request).

    Parameters are sized for the SWE-agent workload: prompts are ~18-30k tokens
    and each decode replica holds ~20-60 concurrent requests, i.e. 0.5-1.8M
    in-flight prompt tokens, hence the 2M default budget.
    """

    def __init__(
        self,
        kv_hard_stop: float = 0.95,
        wait_budget: int = 32,
        inflight_token_budget: int = 2_000_000,
        kv_high_watermark: float = 0.8,
        snapshot_interval_secs: float = 1.0,
        snapshot_timeout_secs: float = 0.15,
        snapshot_stale_secs: float = 3.0,
        log_interval_secs: float = 60.0,
        pressure_cache: Any = None,
    ) -> None:
        self._kv_hard_stop = float(kv_hard_stop)
        self._kv_high_watermark = max(1e-6, float(kv_high_watermark))
        self._wait_budget = max(1, int(wait_budget))
        self._inflight_token_budget = max(1, int(inflight_token_budget))
        self._log_interval_secs = float(log_interval_secs)
        self._pressure = (
            pressure_cache
            if pressure_cache is not None
            else _MetricsPressureCache(
                interval_secs=snapshot_interval_secs,
                timeout_secs=snapshot_timeout_secs,
                stale_secs=snapshot_stale_secs,
            )
        )
        self._candidate_keys: tuple[str, ...] = ()
        self._pending_tokens: list[int] = []
        self._inflight: list[deque[int]] = []
        self._request_weights: dict[str, tuple[int, int]] = {}
        self._last_log = 0.0
        self.stats = {"sticky": 0, "rebalance": 0, "overload_all": 0, "no_key": 0}

    # -- PDDecodeSelector protocol ----------------------------------------

    def select(self, req: dict, candidates: list[DecodeCandidate]) -> DecodeCandidate:
        if not candidates:
            raise ValueError("no decode candidates available")

        payload = req if isinstance(req, dict) else {}
        reservation_id = payload.get("reservation_id")
        if reservation_id is not None and reservation_id in self._request_weights:
            raise ValueError(f"reservation {reservation_id!r} is already active")
        candidate_keys = tuple(candidate.key for candidate in candidates)
        if candidate_keys != self._candidate_keys:
            if self._request_weights:
                raise RuntimeError("drain active reservations before changing decode candidates")
            self._candidate_keys = candidate_keys
            self._pending_tokens = [0] * len(candidates)
            self._inflight = [deque() for _ in candidates]
        self._pressure.observe(candidate_keys)

        prompt_ids = payload.get("prompt_ids") or []
        tokens = len(prompt_ids)
        sticky_key = payload.get("routing_key") or payload.get("request_id")

        scores, hot = self._pressure_scores()
        available = [i for i in range(len(candidates)) if not hot[i]]
        pool = available or list(range(len(candidates)))
        if not available:
            self.stats["overload_all"] += 1

        if sticky_key:
            owner = self._sticky_owner(str(sticky_key), candidate_keys)
            if owner in available:
                selected = owner
                self.stats["sticky"] += 1
            else:
                selected = min(pool, key=lambda i: (self._pending_tokens[i], scores[i], i))
                self.stats["rebalance"] += 1
        else:
            # No session identity (non-agent callers): stay on token load only.
            selected = min(pool, key=lambda i: (self._pending_tokens[i], scores[i], i))
            self.stats["no_key"] += 1

        self._pending_tokens[selected] += tokens
        if reservation_id is not None:
            self._request_weights[reservation_id] = (selected, tokens)
        elif tokens:
            # Legacy direct callers without identity retain index/FIFO release.
            # SelectorRouter always supplies identity and uses release_request.
            self._inflight[selected].append(tokens)
        self._maybe_log(scores, hot)
        return candidates[selected]

    def release(self, index: int) -> None:
        """Legacy FIFO release, only for direct selects without reservation_id."""
        if 0 <= index < len(self._inflight) and self._inflight[index]:
            weight = self._inflight[index].popleft()
            self._pending_tokens[index] = max(0, self._pending_tokens[index] - weight)

    def release_request(self, index: int, reservation_id: str) -> None:
        """Return exactly this request's prompt weight, independent of completion order."""
        reservation = self._request_weights.get(reservation_id)
        if reservation is None:
            return
        owner, weight = reservation
        if owner != index:
            raise ValueError(f"reservation {reservation_id!r} belongs to candidate {owner}, not {index}")
        del self._request_weights[reservation_id]
        self._pending_tokens[index] -= weight

    def reset(self) -> None:
        """Called by verl on weight update / sleep / wake / cache clear.

        There is nothing to drop: the sticky mapping is a pure function of
        ``routing_key`` and the candidate set, so a session simply keeps the
        replica it was already using (and rebuilds its KV there instead of
        ping-ponging between replicas). In-flight token accounting must survive,
        because those requests are still running and will call ``release``.
        """
        return None

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _sticky_owner(sticky_key: str, candidate_keys: tuple[str, ...]) -> int:
        """Rendezvous (highest-random-weight) hashing over the session id."""
        return max(
            range(len(candidate_keys)),
            key=lambda i: _fbi_hash(f"{sticky_key}:{candidate_keys[i]}"),
        )

    def _pressure_scores(self) -> tuple[list[float], list[bool]]:
        """Per-replica soft score (for ranking) and hard overload flag."""
        n = len(self._candidate_keys)
        scores: list[float] = [0.0] * n
        hot: list[bool] = [False] * n
        for i, key in enumerate(self._candidate_keys):
            score = self._pending_tokens[i] / self._inflight_token_budget
            if self._pending_tokens[i] >= self._inflight_token_budget:
                hot[i] = True
            snap = self._pressure.snapshot(key)
            if snap is not None:
                if snap.kv_usage is not None:
                    score = max(score, snap.kv_usage / self._kv_high_watermark)
                    if snap.kv_usage >= self._kv_hard_stop:
                        hot[i] = True
                if snap.wait_capacity is not None:
                    score = max(score, snap.wait_capacity / self._wait_budget)
                    if snap.wait_capacity >= self._wait_budget:
                        hot[i] = True
            scores[i] = score
        return scores, hot

    def _maybe_log(self, scores: list[float], hot: list[bool]) -> None:
        if self._log_interval_secs <= 0 or not any(hot):
            return
        now = time.monotonic()
        if now - self._last_log < self._log_interval_secs:
            return
        self._last_log = now
        logger.warning(
            "[session_sticky] overload guard engaged: scores=%s pending_tokens=%s hot=%s stats=%s",
            [round(s, 2) for s in scores],
            list(self._pending_tokens),
            hot,
            dict(self.stats),
        )


__all__ = [
    "LeastLoadDecodeSelector",
    "CacheAwareDecodeSelector",
    "SessionStickyDecodeSelector",
    "RandomDecodeSelector",
    "ConsistentHashDecodeSelector",
    "RendezvousHashDecodeSelector",
]
