# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KVCAwareBalancer — orchestration shell for the KV-cache-aware router.

A pure framework shell: it wires Config / Strategy / collectors, manages their
lifecycle, and delegates each request to ``route()``. It contains no routing
algorithm. veRL loads this class from ``router_class`` in the external YAML
and wraps it with
``ray.remote(...)`` at runtime, so this is a plain class — directly
constructible and unit-testable. It satisfies veRL's ``RequestLoadBalancer``
Protocol via structural subtyping.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import ray
from omegaconf import DictConfig, OmegaConf

from .collectors import CollectorManager
from .config import KVCAwareConfig
from .store import DataStore
from .strategies import (
    ReplicaInfo,
    StrategyRegistry,
    route,
)

logger = logging.getLogger(__name__)


class KVCAwareBalancer:
    """Pure-framework router shell. See module docstring."""

    # Emit a route() latency aggregate every N route() calls to bound log volume.
    _ROUTE_LOG_EVERY = 64

    def __init__(
        self,
        servers: dict[str, Any],
        config: Optional[dict] = None,
        provider_factory: Callable[..., Any] = None,
    ) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        # Provider construction seam: verl instantiates this class with
        # ``(servers, yaml_config)`` only, so the default is the real manager;
        # unit tests inject a fake through this kwarg instead of monkey-patching
        # class attributes (which ray.remote's by-value serialization leaks).
        if provider_factory is None:
            provider_factory = CollectorManager
        self._provider_factory = provider_factory

        self._servers: dict[str, Any] = dict(servers)
        self._config = self._init_config(config)
        logger.info(f"KVCAwareBalancer, config={self._config}")

        primary_cfg = self._config.strategy
        self._strategy = StrategyRegistry.get(type(primary_cfg)).from_config(primary_cfg)
        self._strategy_summary = str(self._strategy)

        # Balancer-side in-flight counters (server_id → count), mirroring verl's
        # ``GlobalRequestLoadBalancer._inflight_requests`` — answers the #7115
        # protocol's ``get_total_inflight()`` RPC without depending on collector
        # configuration (the store's INFLIGHT_COUNT is the collector-fed twin).
        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        max_num_seqs = self._resolve_max_num_seqs()
        max_num_batched_tokens = self._resolve_max_num_batched_tokens()
        if hasattr(self._strategy, "set_capacity"):
            self._strategy.set_capacity(max_num_seqs, max_num_batched_tokens)
        logger.info(f"KVCAwareBalancer: max_num_seqs={max_num_seqs}, max_num_batched_tokens={max_num_batched_tokens}")

        self._route_calls = 0
        # route() latency profiling — cumulative stats flushed every _ROUTE_LOG_EVERY calls.
        # The flush trigger derives from _route_calls % _ROUTE_LOG_EVERY (a separate
        # since-flush counter would just track it in lockstep).
        self._route_time_total_s = 0.0
        self._route_time_max_ms = 0.0
        self._callbacks: dict[str, list[Callable]] = {
            "on_acquire": [],
            "on_release": [],
            "on_servers_removed": [],
        }
        self._store = DataStore()
        self._init_provider()

    def _fetch_rollout_cfg(self) -> Any | None:
        """Return the first server's rollout config, or None when unavailable.

        All replicas share the same rollout config, so the first successful RPC
        wins and the loop exits right away. ``None`` when no server exposes
        ``get_rollout_config`` or every RPC fails.
        """
        for handle in self._servers.values():
            if not hasattr(handle, "get_rollout_config"):
                continue
            try:
                return ray.get(handle.get_rollout_config.remote())
            except Exception as e:  # noqa: BLE001
                logger.warning(f"get_rollout_config failed on {type(handle).__name__}: {e}")
                continue
        return None

    @staticmethod
    def _read_nested(obj: Any, *keys: str) -> Any:
        """Walk ``obj`` down ``keys``, accepting mappings or plain objects."""
        current = obj
        for key in keys:
            if current is None:
                return None
            if isinstance(current, dict | DictConfig):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    def _fetch_router_override(self) -> dict | None:
        """Read the runtime router override from the rollout config, if any.

        VeRL configs carry the knob overrides under
        ``rollout.custom.agent_framework.router`` (a flat mapping of field name
        → value, e.g. ``load_threshold`` / ``http_interval``). Returns a plain
        dict, or ``None`` when no rollout config is available or the node is
        absent / empty.
        """
        rollout_cfg = self._fetch_rollout_cfg()
        if rollout_cfg is None:
            return None
        router = self._read_nested(rollout_cfg, "custom", "agent_framework", "router")
        if not router:
            return None
        if isinstance(router, DictConfig):
            return OmegaConf.to_container(router, resolve=True)
        return dict(router)

    def _resolve_rollout_config_int(self, attr: str, default: int) -> int:
        """Read a positive int ``attr`` from the first server's rollout config.

        Fail-closed to ``default`` when no rollout config is available, the RPC
        fails, or the value is missing / non-positive. Shared by the
        ``max_num_seqs`` (per-step request cap) and ``max_num_batched_tokens``
        (per-step token budget) resolvers.
        """
        rollout_cfg = self._fetch_rollout_cfg()
        if rollout_cfg is None:
            return default
        value = getattr(rollout_cfg, attr, default)
        if value is None or value <= 0:
            logger.warning(f"server returned non-positive {attr}={value}; using default={default}")
            return default
        return value

    def _resolve_max_num_seqs(self, default: int = 256) -> int:
        """Per-step sequence cap — denominator for the in-flight request load."""
        return self._resolve_rollout_config_int("max_num_seqs", default)

    def _resolve_max_num_batched_tokens(self, default: int = 2048) -> int:
        """Per-step token budget — denominator for the in-flight token load."""
        return self._resolve_rollout_config_int("max_num_batched_tokens", default)

    def _init_config(self, config) -> KVCAwareConfig:
        _config = KVCAwareConfig.from_config(config)
        _config.apply_override(self._fetch_router_override())
        return _config

    def _init_provider(self) -> None:
        """Resolve per-server endpoints from Ray actor handles, then start collectors.

        Non-actor handles (plain strings in unit tests) have no
        ``get_server_address``; discovery is skipped and collectors fall back
        to configured/default endpoints.
        """
        collection_names = sorted(self._strategy.collection_names)
        server_addresses: dict[str, str] = {}
        kv_event_endpoints: dict[str, list[str]] = {}
        addr_futures = []
        ep_futures = []
        active_replicas = []
        for replica_id, handle in self._servers.items():
            if not hasattr(handle, "get_server_address"):
                logger.warning(
                    f"server '{replica_id}' handle has no get_server_address remote "
                    f"(type={type(handle).__name__}); skipping dynamic endpoint discovery",
                )
                continue
            active_replicas.append(replica_id)
            addr_futures.append(handle.get_server_address.remote())
            ep_futures.append(handle.get_kv_events_endpoints.remote())

        if active_replicas:
            ips_ports = ray.get(addr_futures)
            endpoints_list = ray.get(ep_futures)
            for replica_id, (ip, port), endpoints in zip(active_replicas, ips_ports, endpoints_list, strict=False):
                server_addresses[replica_id] = f"{ip}:{port}"
                if endpoints is None:
                    continue
                # Pad verl's [sub, replay] to the [sub, replay, publisher, topic] ZMQTransport expects.
                if len(endpoints) == 2:
                    endpoints = [*endpoints, "zmq", "kv-events"]
                kv_event_endpoints[replica_id] = endpoints
        self._provider = self._provider_factory(
            self._config.collector,
            collection_names,
            server_addresses=server_addresses,
            kv_event_endpoints=kv_event_endpoints,
            balancer_handler=self,
        )
        self._provider.start()

    # ── Callback registry (opt-in hook points for statistic collectors) ──

    def register_call_back(self, event: str, fn: Callable) -> None:
        """Append ``fn`` to the listeners for ``event``.

        Opt-in hook points: ``on_acquire`` / ``on_release`` / ``on_servers_removed``.
        """
        self._callbacks.setdefault(event, []).append(fn)

    def un_register_call_back(self, event: str, fn: Callable) -> None:
        """Remove ``fn`` from ``event``'s callback list (idempotent)."""
        lst = self._callbacks.get(event, [])
        if fn in lst:
            lst.remove(fn)

    def _fire(self, event: str, *args: Any) -> None:
        """Invoke every registered callback for ``event``; errors are swallowed."""
        for fn in self._callbacks.get(event, []):
            try:
                fn(*args)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"callback {event} failed: {type(exc).__name__}: {exc}")

    def get_all_servers(self) -> list[str]:
        """List all active server ids."""
        return list(self._servers.keys())

    def get_status(self) -> dict:
        """Construction + routing snapshot for debugging."""
        return {
            "servers": list(self._servers.keys()),
            "provider": type(self._provider).__name__,
            "strategies": [{"type": type(self._strategy).__name__}],  # legacy key; single strategy
            "route_calls": self._route_calls,
            "sticky_size": self._store.sticky_status()["size"],
            "total_inflight": self.get_total_inflight(),
        }

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server after a request completes; fires ``on_release``.

        ``prompt_ids`` (the completing request's input token ids) mirror the value
        passed at ``acquire_server`` time; ``len(prompt_ids)`` is forwarded as the
        same ``prompt_len`` int that ``on_acquire`` produced, so the in-flight
        token gauge is decremented by exactly what acquire added. ``request_id``
        lets the inflight parser attribute the release to the right request (e.g.
        to subtract its turn from the in-flight turn sum). Both default for
        callers that do not track them (the token gauge simply stays unchanged on
        release).
        """
        if self._inflight.get(server_id, 0) > 0:
            self._inflight[server_id] -= 1
        self._fire("on_release", server_id, request_id)

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Delegate to ``route()`` for a best-first ranking, return ``(top, handle)``.

        Raises ``RuntimeError`` if no replica is available. ``request_id`` is
        forwarded so strategies can short-circuit to a sticky-bound replica;
        ``on_acquire`` then refreshes the binding.
        """
        replicas = [ReplicaInfo(replica_id=sid) for sid in self._servers]
        self._route_calls += 1
        t0 = time.perf_counter()
        ranking = route(
            self._strategy,
            prompt_ids,
            self._store,
            replicas,
            request_id,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        self._route_time_total_s += dt_ms / 1000.0
        self._route_time_max_ms = max(self._route_time_max_ms, dt_ms)
        if not ranking:
            raise RuntimeError("no available replica to route to")
        server_id = ranking[0]
        self._inflight[server_id] = self._inflight.get(server_id, 0) + 1
        self._fire("on_acquire", request_id, server_id, prompt_ids)
        logger.info(
            f"request={request_id} routed to server={server_id} (ranking={ranking}, pool={list(self._servers)}, "
            f"route={dt_ms:.2f}ms, strategy=[{self._strategy_summary}])"
        )
        if self._route_calls % self._ROUTE_LOG_EVERY == 0:
            logger.debug(
                f"route-stats: calls={self._ROUTE_LOG_EVERY} total={self._route_time_total_s:.3f}s "
                f"mean={self._route_time_total_s * 1000 / max(self._route_calls, 1):.2f}ms "
                f"max={self._route_time_max_ms:.2f}ms (flushed every {self._ROUTE_LOG_EVERY} calls) "
                f"strategy=[{self._strategy_summary}]"
            )
        return server_id, self._servers[server_id]

    # ── verl #7115 protocol: field declaration + trainer-facing controls ──

    def require_acquire_fields(self) -> list[str]:
        """``generate()`` kwargs this router consumes at acquire time.

        verl's ``LLMServerClient`` serializes only the declared fields into the
        ``acquire_server`` RPC; prefix-hash routing needs the prompt tokens.
        """
        return ["prompt_ids"]

    def require_release_fields(self) -> list[str]:
        """Identity fields this router consumes at release time.

        Only ``"request_id"`` is supported by verl today; the prompt length is
        recovered from acquire-time bookkeeping (see ``release_server``).
        """
        return ["request_id"]

    def clear_sticky_cache(self) -> dict:
        """Drop every sticky binding so returning sessions re-route (verl #7115).

        Called by trainers on training/rollout phase switches and by fully-async
        rebalancing. Prefix-hash checkpoints are deliberately kept — they are
        replica-agnostic (shared across replicas, append-only), so re-routed
        requests still reuse them.

        Returns:
            ``{"cleared_entries": int, "server_loads": dict[str, int]}``.
        """
        cleared = self._store.clear_sticky_bindings()
        loads = dict(self._inflight)
        logger.info(f"clear_sticky_cache: cleared {cleared} binding(s); server_loads={loads}")
        return {"cleared_entries": cleared, "server_loads": loads}

    def get_total_inflight(self) -> int:
        """Total in-flight requests across the pool (verl #7115 drain polling)."""
        return sum(self._inflight.values())

    def add_servers(self, servers: dict[str, Any]) -> None:
        """Bulk-add servers to the pool (provider is keyed by init-time addresses, untouched here)."""
        for sid, handle in servers.items():
            self._servers[sid] = handle
            self._inflight.setdefault(sid, 0)

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers; fires ``on_servers_removed`` to invalidate sticky bindings."""
        for sid in server_ids:
            self._servers.pop(sid, None)
            # Mirror verl's built-in: drop the counter row; a later release for
            # an in-flight request is tolerated by release_server's floor guard.
            self._inflight.pop(sid, None)
        if server_ids:
            self._fire("on_servers_removed", server_ids)
