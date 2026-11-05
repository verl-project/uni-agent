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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import httpx
import ray
from omegaconf import DictConfig, OmegaConf

from .collectors import CollectorManager
from .collectors.transport.lifecycle import LifecycleControlReceiver
from .config import KVCAwareConfig
from .store import DataStore
from .strategies import (
    ReplicaInfo,
    StrategyRegistry,
    route,
)
from .types.cache_lifecycle import CacheLifecycleState
from .utils.endpoints import format_host_port

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
        self._lifecycle_lock = threading.RLock()
        self._provider_lock = threading.RLock()
        self._lifecycle_states = {}
        self._lifecycle_sources = {}
        self._lifecycle_registrations = {}
        self._lifecycle_handshakes = {}
        self._pending_servers = set()
        self._lifecycle_receiver = None
        self._provider = None
        self._store = DataStore()
        self._init_cache_registry()
        try:
            self._register_cache_lifecycles(self._servers)
            self._init_provider()
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.warning("Could not finish failed balancer construction cleanup", exc_info=True)
            raise

    def update_cache_lifecycle(self, server_id, value, *, registration=False, cache_metadata=None):
        state = CacheLifecycleState.from_dict(value)
        with self._lifecycle_lock:
            if server_id not in self._servers:
                return {"applied": False, "reason": "server_removed"}
            accepted = self._store.update_cache_lifecycle(
                server_id, state, registration=registration, cache_metadata=cache_metadata
            )
            current = self._store.cache_lifecycle_state(server_id)
            if current is not None and current.instance_id == state.instance_id:
                # An equal revision is valid for the same engine, but a rejected
                # handshake must never borrow another engine's serving state.
                self._lifecycle_states[server_id] = current
            return {"applied": accepted, "revision": self._lifecycle_states.get(server_id, state).revision}

    def _receive_cache_lifecycle(self, server_id, value, *, cache_metadata=None):
        with self._lifecycle_lock:
            handle = self._servers.get(server_id)
            if handle is None or not hasattr(handle, "register_cache_lifecycle_sink"):
                return {"applied": False, "reason": "lifecycle_sink_removed"}
            pending = self._lifecycle_handshakes.get(server_id)
            if pending is not None:
                state = CacheLifecycleState.from_dict(value)
                previous = pending.get(state.instance_id)
                if previous is None or (state.epoch >= previous[0].epoch and state.revision > previous[0].revision):
                    metadata = cache_metadata
                    if metadata is None and previous is not None:
                        metadata = previous[1]
                    pending[state.instance_id] = state, metadata
                elif previous[1] is None and cache_metadata is not None:
                    pending[state.instance_id] = previous[0], cache_metadata
                return {"applied": False, "reason": "lifecycle_handshake_pending"}
            if server_id not in self._lifecycle_states:
                return {"applied": False, "reason": "lifecycle_handshake_pending"}
            return self.update_cache_lifecycle(server_id, value, cache_metadata=cache_metadata)

    def _register_cache_lifecycles(self, servers):
        handles = {sid: handle for sid, handle in servers.items() if hasattr(handle, "register_cache_lifecycle_sink")}
        if not handles:
            return
        created = self._lifecycle_receiver is None
        if created:
            self._lifecycle_receiver = LifecycleControlReceiver(
                self._receive_cache_lifecycle,
                ray.util.get_node_ip_address(),
                registration_lock=self._lifecycle_lock,
            )
        registrations = {}
        try:
            for sid, handle in handles.items():
                with self._lifecycle_lock:
                    self._lifecycle_handshakes[sid] = {}
                    before = self._store.cache_lifecycle_state(sid)
                    before_instance = before.instance_id if before is not None else None
                    descriptor = self._lifecycle_receiver.descriptor(sid)
                    registrations[sid] = descriptor
                    self._lifecycle_registrations[sid] = descriptor
                response = ray.get(
                    handle.register_cache_lifecycle_sink.remote(descriptor),
                    timeout=self._DISCOVERY_TIMEOUT_SECONDS,
                )
                state = CacheLifecycleState.from_dict(response["state"])
                cache_metadata = response.get("cache_metadata")
                source = self._parse_kv_event_sources(response["sources"])
                with self._lifecycle_lock:
                    # The engine may publish after taking the RPC snapshot but
                    # before it reaches this thread. Revisions remain monotonic
                    # when this producer replaces its engine; registration
                    # capabilities exclude notifications from former handles.
                    pending = self._lifecycle_handshakes.pop(sid)
                    candidates = [(state, cache_metadata), *pending.values()]
                    selected, selected_metadata = max(candidates, key=lambda value: value[0].revision)
                    if selected_metadata is None:
                        selected_metadata = next(
                            (
                                metadata
                                for candidate, metadata in candidates
                                if candidate.instance_id == selected.instance_id and metadata is not None
                            ),
                            None,
                        )
                    state, cache_metadata = selected, selected_metadata
                    current = self._store.cache_lifecycle_state(sid)
                    if (
                        current is not None
                        and current.instance_id != state.instance_id
                        and current.instance_id != before_instance
                    ):
                        raise RuntimeError(f"cache lifecycle handshake superseded for server {sid!r}")
                    self.update_cache_lifecycle(sid, state.to_dict(), registration=True, cache_metadata=cache_metadata)
                    current = self._store.cache_lifecycle_state(sid)
                    if current is None or current.instance_id != state.instance_id:
                        raise RuntimeError(f"cache lifecycle handshake rejected for server {sid!r}")
                    self._lifecycle_sources[sid] = source
        except Exception:
            with self._lifecycle_lock:
                for sid in handles:
                    self._lifecycle_handshakes.pop(sid, None)
                    self._lifecycle_receiver.revoke(sid)
            # A failed constructor/registration must not leave dead control
            # targets attached to servers that registered successfully.
            try:
                self._detach_cache_lifecycles(handles, registrations)
            except Exception:
                logger.warning("Could not detach failed cache lifecycle registration", exc_info=True)
            if created:
                self._lifecycle_receiver.close()
                self._lifecycle_receiver = None
            raise  # Never silently route with an unacknowledged lifecycle handshake.

    def _detach_cache_lifecycles(self, servers, registrations):
        futures, error = [], None
        for sid, handle in servers.items():
            descriptor = registrations.get(sid)
            if descriptor is not None and hasattr(handle, "unregister_cache_lifecycle_sink"):
                try:
                    futures.append(
                        handle.unregister_cache_lifecycle_sink.remote(
                            descriptor["token"], generation=descriptor["generation"]
                        )
                    )
                except Exception as exc:
                    error = error or exc
        if futures:
            try:
                ray.get(futures, timeout=self._DISCOVERY_TIMEOUT_SECONDS)
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    def close(self):
        with self._provider_lock:
            error = None
            try:
                if self._provider is not None:
                    self._provider.stop()
            except Exception as exc:
                error = exc
            receiver = self._lifecycle_receiver
            if receiver is not None:
                with self._lifecycle_lock:
                    servers = dict(self._servers)
                    registrations = dict(self._lifecycle_registrations)
                    for sid in servers:
                        receiver.revoke(sid)
                detached = False
                try:
                    self._detach_cache_lifecycles(servers, registrations)
                    detached = True
                except Exception as exc:
                    logger.warning("Could not detach cache lifecycle sinks", exc_info=True)
                    error = error or exc
                try:
                    receiver.close()
                except Exception as exc:
                    logger.warning("Could not close cache lifecycle receiver", exc_info=True)
                    error = error or exc
                else:
                    if detached:
                        self._lifecycle_receiver = None
                        self._lifecycle_registrations.clear()
            if error is not None:
                raise error

    def _init_cache_registry(self) -> None:
        """Install the shared layout and bind each replica's own hash profile."""

        def query(handle):
            if not hasattr(handle, "get_kv_cache_metadata"):
                return {"status": "unsupported", "reason": "metadata_method_unavailable"}
            try:
                return ray.get(handle.get_kv_cache_metadata.remote(), timeout=self._DISCOVERY_TIMEOUT_SECONDS)
            except Exception as exc:
                return {"status": "error", "reason": f"metadata_query_failed:{type(exc).__name__}"}

        with ThreadPoolExecutor(max_workers=min(self._MAX_DISCOVERY_WORKERS, len(self._servers))) as executor:
            values = executor.map(query, self._servers.values())
            metadata_by_node = dict(zip(self._servers, values, strict=True))
        for replica_id, metadata in metadata_by_node.items():
            state = self._store.configure_cache_registry(metadata, replica_metadata=metadata_by_node)
            if state.status in {"registered", "disabled"}:
                return
            logger.warning("Cache registry unavailable (source replica %s): %s", replica_id, state.reason)

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

    def _init_provider(self, *, refresh=False, reset_kv_nodes=()) -> None:
        """Resolve per-server endpoints from vLLM HTTP servers, then start collectors.

        Non-actor handles (plain strings in unit tests) have no
        ``get_server_address``; discovery is skipped and collectors fall back
        to configured/default endpoints.
        """
        with self._lifecycle_lock:
            servers = dict(self._servers)
            lifecycle_sources = dict(self._lifecycle_sources)
        collection_names = sorted(self._strategy.collection_names)
        address_futures = []
        replica_ids = []
        for replica_id, handle in servers.items():
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
            addresses = ray.get(address_futures, timeout=self._DISCOVERY_TIMEOUT_SECONDS)
            server_addresses = {
                replica_id: format_host_port(ip, port)
                for replica_id, (ip, port) in zip(replica_ids, addresses, strict=True)
            }
        native_addresses = {sid: address for sid, address in server_addresses.items() if sid not in lifecycle_sources}
        kv_event_endpoints = self._discover_kv_event_endpoints(native_addresses)
        kv_event_endpoints.update({sid: source for sid, source in lifecycle_sources.items() if source})
        if refresh:
            self._provider.update_sources(server_addresses, kv_event_endpoints, reset_kv_nodes=reset_kv_nodes)
            return
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
            "cache_lifecycles": {sid: state.to_dict() for sid, state in self._lifecycle_states.items()},
        }

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server and notify statistic collectors."""
        with self._lifecycle_lock:
            if self._inflight.get(server_id, 0) > 0:
                self._inflight[server_id] -= 1
                if not self._inflight[server_id] and server_id not in self._servers:
                    self._inflight.pop(server_id)
            self._fire("on_release", server_id, request_id)

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Return the highest-ranked server id and handle."""
        with self._lifecycle_lock:
            return self._acquire_server_locked(request_id, prompt_ids)

    def _acquire_server_locked(self, request_id, prompt_ids):
        replicas = [
            ReplicaInfo(replica_id=sid)
            for sid in self._servers
            if sid not in self._pending_servers
            and (sid not in self._lifecycle_states or self._lifecycle_states[sid].serving)
        ]
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
        """Add servers and refresh network collectors while routing stays available."""
        with self._provider_lock:
            with self._lifecycle_lock:
                reset_nodes = {
                    sid
                    for sid, handle in servers.items()
                    if sid in self._pending_servers or self._servers.get(sid) is not handle
                }
                self._pending_servers.update(servers)
                for sid, handle in servers.items():
                    previous = self._servers.get(sid)
                    if previous is not None and previous is not handle:
                        descriptor = self._lifecycle_registrations.pop(sid, None)
                        if self._lifecycle_receiver is not None:
                            self._lifecycle_receiver.revoke(sid)
                        if descriptor is not None and hasattr(previous, "unregister_cache_lifecycle_sink"):
                            previous.unregister_cache_lifecycle_sink.remote(
                                descriptor["token"], generation=descriptor["generation"]
                            )
                        self._store.remove_cache_node(sid)
                        self._lifecycle_states.pop(sid, None)
                        self._lifecycle_sources.pop(sid, None)
                        self._lifecycle_handshakes.pop(sid, None)
                    self._servers[sid] = handle
                    self._inflight.setdefault(sid, 0)
            for sid in reset_nodes:
                handle = servers[sid]
                metadata = {"status": "unsupported", "reason": "metadata_method_unavailable"}
                if hasattr(handle, "get_kv_cache_metadata"):
                    try:
                        metadata = ray.get(
                            handle.get_kv_cache_metadata.remote(), timeout=self._DISCOVERY_TIMEOUT_SECONDS
                        )
                    except Exception as exc:
                        metadata = {"status": "error", "reason": f"metadata_query_failed:{type(exc).__name__}"}
                self._store.configure_replica_cache_metadata(sid, metadata)
            self._register_cache_lifecycles(servers)
            if reset_nodes or any(hasattr(handle, "get_server_address") for handle in servers.values()):
                self._init_provider(refresh=True, reset_kv_nodes=reset_nodes)
            with self._lifecycle_lock:
                self._pending_servers.difference_update(servers)

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers; fires ``on_servers_removed`` to invalidate sticky bindings."""
        with self._provider_lock:
            with self._lifecycle_lock:
                for sid in server_ids:
                    handle = self._servers.pop(sid, None)
                    descriptor = self._lifecycle_registrations.pop(sid, None)
                    if self._lifecycle_receiver is not None:
                        self._lifecycle_receiver.revoke(sid)
                    if (
                        handle is not None
                        and descriptor is not None
                        and hasattr(handle, "unregister_cache_lifecycle_sink")
                    ):
                        handle.unregister_cache_lifecycle_sink.remote(
                            descriptor["token"], generation=descriptor["generation"]
                        )
                    if not self._inflight.get(sid, 0):
                        self._inflight.pop(sid, None)
                    self._store.remove_cache_node(sid)
                    self._lifecycle_states.pop(sid, None)
                    self._lifecycle_sources.pop(sid, None)
                    self._lifecycle_handshakes.pop(sid, None)
                    self._pending_servers.discard(sid)
                if server_ids:
                    self._fire("on_servers_removed", server_ids)
            if server_ids:
                self._init_provider(refresh=True)
