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

"""Balancer orchestration for agent-aware routing."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import httpx
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
    """Route requests and manage collector-backed state."""

    _ROUTE_LOG_EVERY = 64
    _MAX_DISCOVERY_WORKERS = 8
    _DISCOVERY_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        servers: dict[str, Any],
        config: dict[str, Any] | DictConfig | None = None,
        provider_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not servers:
            raise ValueError("servers must be non-empty")
        if provider_factory is None:
            provider_factory = CollectorManager
        self._provider_factory = provider_factory

        self._servers: dict[str, Any] = dict(servers)
        rollout_config = self._fetch_rollout_config()
        self._config = self._init_config(config, rollout_config)
        logger.info("KVCAwareBalancer, config=%s", self._config)

        primary_cfg = self._config.strategy
        self._strategy = StrategyRegistry.get(type(primary_cfg)).from_config(primary_cfg)
        self._strategy_summary = str(self._strategy)

        self._inflight: dict[str, int] = {sid: 0 for sid in self._servers}
        max_num_seqs = self._rollout_config_int(rollout_config, "max_num_seqs", 256)
        max_num_batched_tokens = self._rollout_config_int(rollout_config, "max_num_batched_tokens", 2048)
        if hasattr(self._strategy, "set_capacity"):
            self._strategy.set_capacity(max_num_seqs, max_num_batched_tokens)
        logger.info(
            "KVCAwareBalancer: max_num_seqs=%s, max_num_batched_tokens=%s",
            max_num_seqs,
            max_num_batched_tokens,
        )

        self._route_calls = 0
        self._route_time_total_ms = 0.0
        self._route_time_max_ms = 0.0
        self._callbacks: dict[str, list[Callable[..., None]]] = {
            "on_acquire": [],
            "on_release": [],
            "on_servers_removed": [],
        }
        self._store = DataStore()
        self._init_provider()

    def _fetch_rollout_config(self) -> Any | None:
        """Return the first available rollout config."""
        for handle in self._servers.values():
            if not hasattr(handle, "get_rollout_config"):
                continue
            try:
                return ray.get(handle.get_rollout_config.remote())
            except Exception as e:  # noqa: BLE001
                logger.warning("get_rollout_config failed on %s: %s", type(handle).__name__, e)
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

    @classmethod
    def _router_override(cls, rollout_config: Any | None) -> dict[str, Any] | None:
        if rollout_config is None:
            return None
        router = cls._read_nested(rollout_config, "custom", "agent_framework", "router")
        if not router:
            return None
        if isinstance(router, DictConfig):
            return OmegaConf.to_container(router, resolve=True)
        return dict(router)

    @staticmethod
    def _rollout_config_int(rollout_config: Any | None, name: str, default: int) -> int:
        if rollout_config is None:
            return default
        if isinstance(rollout_config, dict | DictConfig):
            value = rollout_config.get(name, default)
        else:
            value = getattr(rollout_config, name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            logger.warning("server returned invalid %s=%r; using default=%s", name, value, default)
            return default
        return value

    def _init_config(self, config: dict[str, Any] | DictConfig | None, rollout_config: Any | None) -> KVCAwareConfig:
        _config = KVCAwareConfig.from_config(config)
        _config.apply_override(self._router_override(rollout_config))
        return _config

    def _init_provider(self) -> None:
        """Resolve per-server endpoints from vLLM HTTP servers, then start collectors.

        Non-actor handles (plain strings in unit tests) have no
        ``get_server_address``; discovery is skipped and collectors fall back
        to configured/default endpoints.
        """
        collection_names = sorted(self._strategy.collection_names)
        address_futures = []
        replica_ids = []
        for replica_id, handle in self._servers.items():
            if not hasattr(handle, "get_server_address"):
                logger.warning(
                    "server '%s' handle has no get_server_address remote (type=%s); "
                    "skipping dynamic endpoint discovery",
                    replica_id,
                    type(handle).__name__,
                )
                continue
            replica_ids.append(replica_id)
            address_futures.append(handle.get_server_address.remote())

        server_addresses: dict[str, str] = {}
        if replica_ids:
            addresses = ray.get(address_futures)
            server_addresses = {
                replica_id: f"{ip}:{port}" for replica_id, (ip, port) in zip(replica_ids, addresses, strict=True)
            }
        kv_event_endpoints = self._discover_kv_event_endpoints(server_addresses)
        self._provider = self._provider_factory(
            self._config.collector,
            collection_names,
            server_addresses=server_addresses,
            kv_event_endpoints=kv_event_endpoints,
            balancer_handler=self,
        )
        self._provider.start()

    def _discover_kv_event_endpoints(self, server_addresses: dict[str, str]) -> dict[str, list[str]]:
        if not server_addresses:
            return {}

        endpoints: dict[str, list[str]] = {}
        max_workers = min(len(server_addresses), self._MAX_DISCOVERY_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kv-event-discovery") as executor:
            futures = {
                executor.submit(self._fetch_kv_event_endpoints, address): replica_id
                for replica_id, address in server_addresses.items()
            }
            for future in as_completed(futures):
                replica_id = futures[future]
                try:
                    source = future.result()
                except httpx.HTTPError as e:
                    logger.warning("failed to discover KV-event endpoint from server '%s': %s", replica_id, e)
                    continue
                if source is not None:
                    endpoints[replica_id] = source
        return endpoints

    @staticmethod
    def _collector_address(endpoint: Any) -> str:
        """Convert a vLLM TCP URI into the address expected by ZMQTransport."""
        if endpoint is None:
            return ""
        if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
            raise RuntimeError(f"unsupported KV-event endpoint: {endpoint!r}")
        return endpoint.removeprefix("tcp://")

    @classmethod
    def _parse_kv_event_sources(cls, sources: Any) -> list[str] | None:
        """Convert vLLM's rank-keyed response into the current single-DP shape."""
        if not isinstance(sources, dict):
            raise RuntimeError(f"vLLM returned invalid KV-event sources: {sources!r}")
        if not sources:
            return None
        if len(sources) > 1:
            raise RuntimeError(
                "agent-aware-router does not support multiple DP ranks per rollout server yet: "
                f"reported ranks={sorted(sources)}"
            )

        rank, config = next(iter(sources.items()))
        if not isinstance(config, dict):
            raise RuntimeError(f"vLLM returned invalid KV-event config for DP rank {rank}: {config!r}")
        publisher = config.get("publisher")
        if publisher != "zmq":
            raise RuntimeError(f"unsupported KV-event publisher for DP rank {rank}: {publisher!r}")
        topic = config.get("topic")
        if not isinstance(topic, str):
            raise RuntimeError(f"invalid KV-event topic for DP rank {rank}: {topic!r}")
        return [
            cls._collector_address(config.get("endpoint")),
            cls._collector_address(config.get("replay_endpoint")),
            publisher,
            topic,
        ]

    @classmethod
    def _fetch_kv_event_endpoints(cls, server_address: str) -> list[str] | None:
        """Query the endpoint-discovery API provided by vLLM #55844."""
        with httpx.Client(timeout=cls._DISCOVERY_TIMEOUT_SECONDS, trust_env=False) as client:
            response = client.get(f"http://{server_address}/kv_event_sources")
            response.raise_for_status()
            return cls._parse_kv_event_sources(response.json())

    def register_call_back(self, event: str, fn: Callable[..., None]) -> None:
        """Register an event callback."""
        self._callbacks.setdefault(event, []).append(fn)

    def un_register_call_back(self, event: str, fn: Callable[..., None]) -> None:
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
                logger.warning("callback %s failed: %s: %s", event, type(exc).__name__, exc)

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
        """Release a server and notify statistic collectors."""
        if self._inflight.get(server_id, 0) > 0:
            self._inflight[server_id] -= 1
        self._fire("on_release", server_id, request_id)

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Return the highest-ranked server id and handle."""
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
        self._route_time_total_ms += dt_ms
        self._route_time_max_ms = max(self._route_time_max_ms, dt_ms)
        if not ranking:
            raise RuntimeError("no available replica to route to")
        server_id = ranking[0]
        self._inflight[server_id] = self._inflight.get(server_id, 0) + 1
        self._fire("on_acquire", request_id, server_id, prompt_ids)
        logger.debug(
            "request=%s routed to server=%s (ranking=%s, pool=%s, route=%.2fms, strategy=[%s])",
            request_id,
            server_id,
            ranking,
            list(self._servers),
            dt_ms,
            self._strategy_summary,
        )
        if self._route_calls % self._ROUTE_LOG_EVERY == 0:
            logger.info(
                "route-stats: calls=%s total=%.3fs mean=%.2fms max=%.2fms strategy=[%s]",
                self._route_calls,
                self._route_time_total_ms / 1000,
                self._route_time_total_ms / self._route_calls,
                self._route_time_max_ms,
                self._strategy_summary,
            )
        return server_id, self._servers[server_id]

    def require_acquire_fields(self) -> list[str]:
        """Declare fields forwarded by verl during acquire."""
        return ["prompt_ids"]

    def require_release_fields(self) -> list[str]:
        """Declare fields forwarded by verl during release."""
        return ["request_id"]

    def clear_sticky_cache(self) -> dict:
        """Clear sticky bindings while preserving prefix checkpoints."""
        cleared = self._store.clear_sticky_bindings()
        loads = dict(self._inflight)
        logger.info("clear_sticky_cache: cleared %s binding(s); server_loads=%s", cleared, loads)
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
            self._inflight.pop(sid, None)
        if server_ids:
            self._fire("on_servers_removed", server_ids)
