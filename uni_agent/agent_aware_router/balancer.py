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
from uuid import uuid4

import httpx
import ray
from omegaconf import DictConfig, OmegaConf

from uni_agent.events import (
    ADMISSION_ROUTER_OBSERVATION_SUBSCRIPTION,
    GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
    REPLICA_CAPACITY_CHANGED,
    ROUTE_COMMITTED,
    SESSION_CLOSED,
    SESSION_OPENED,
    DeliveryMode,
    DirectAck,
    DirectAckStatus,
    DirectEventEndpoint,
    DirectStateBatch,
    EventPublisher,
    LocalEventBus,
    Scope,
    SnapshotAndCursor,
    SubscriptionSpec,
)
from uni_agent.gateway.admission import AdmissionSignalsProjector

from .collectors import CollectorManager
from .config import KVCAwareConfig
from .router_state import RouterInflightStateProjector, RouterStateMode, RouterStickyStateProjector
from .session_state import GatewaySessionStateProjector
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
        router_state_mode: str | RouterStateMode | None = None,
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
        self._router_state_mode = self._resolve_router_state_mode(router_state_mode, rollout_config)
        self._router_sticky_projector: RouterStickyStateProjector | None = None
        self._router_sticky_update_handler = None
        self._router_sticky_update_observer = None
        self._router_inflight_projector: RouterInflightStateProjector | None = None
        self._router_inflight_update_handler = None
        self._router_inflight_update_observer = None
        self._router_event_bus: LocalEventBus | None = None
        self._router_event_publisher: EventPublisher | None = None
        self._admission_signals_projector: AdmissionSignalsProjector | None = None
        self._capacity_versions: dict[str, int] = {}
        self._replica_epochs: dict[str, str] = {}
        self._router_observation_failures = 0
        self._init_router_state_runtime()
        self._direct_event_bus: LocalEventBus | None = None
        self._direct_session_projector: GatewaySessionStateProjector | None = None
        self._direct_session_endpoint: DirectEventEndpoint | None = None
        self._init_provider()

    def _init_router_state_runtime(self) -> None:
        if self._router_state_mode is RouterStateMode.LEGACY:
            return
        projector = RouterStickyStateProjector(self._router_state_mode, self._store)
        self._router_sticky_projector = projector
        if self._router_state_mode is RouterStateMode.SHADOW:
            self._router_sticky_update_observer = projector.observe
        else:
            self._router_sticky_update_handler = projector.apply
        inflight = RouterInflightStateProjector(self._router_state_mode, self._store)
        self._router_inflight_projector = inflight
        if self._router_state_mode is RouterStateMode.SHADOW:
            self._router_inflight_update_observer = inflight.observe
        else:
            self._router_inflight_update_handler = inflight.apply

        bus = LocalEventBus()
        admission = AdmissionSignalsProjector()
        bus.subscribe(
            SubscriptionSpec(
                subscription_id=ADMISSION_ROUTER_OBSERVATION_SUBSCRIPTION,
                event_types=(ROUTE_COMMITTED, REPLICA_CAPACITY_CHANGED),
                scope=Scope.LOCAL,
                delivery=DeliveryMode.INLINE,
            ),
            admission.apply,
        )
        self._router_event_bus = bus
        self._router_event_publisher = EventPublisher(
            bus,
            run_id=f"router-run-{uuid4().hex}",
            producer_id=f"router:{','.join(sorted(self._servers))}",
        )
        self._admission_signals_projector = admission
        self._capacity_versions = {replica_id: 0 for replica_id in self._servers}
        self._replica_epochs = {replica_id: uuid4().hex for replica_id in self._servers}

    def _ensure_direct_session_endpoint(self) -> DirectEventEndpoint:
        endpoint = self._direct_session_endpoint
        if endpoint is not None:
            return endpoint
        bus = LocalEventBus()
        projector = GatewaySessionStateProjector()

        def apply_session_event(event) -> None:
            projector.apply(event)
            if self._admission_signals_projector is not None:
                self._admission_signals_projector.apply(event)

        def install_snapshot(snapshot) -> None:
            projector.install_snapshot(snapshot)
            if self._admission_signals_projector is not None:
                self._admission_signals_projector.install_gateway_snapshot(snapshot)

        bus.subscribe(
            SubscriptionSpec(
                subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
                event_types=(SESSION_OPENED, SESSION_CLOSED),
                scope=Scope.DIRECT,
                delivery=DeliveryMode.STATE_SYNC,
            ),
            apply_session_event,
        )
        endpoint = DirectEventEndpoint(bus, snapshot_installer=install_snapshot)
        self._direct_event_bus = bus
        self._direct_session_projector = projector
        self._direct_session_endpoint = endpoint
        return endpoint

    def install_direct_snapshot(self, request: dict[str, Any]) -> dict[str, Any]:
        """Install one authoritative Gateway session snapshot in shadow state."""
        try:
            parsed = SnapshotAndCursor.from_dict(request)
        except (KeyError, TypeError, ValueError) as exc:
            return self._invalid_direct_request_ack(request, exc)
        return self._ensure_direct_session_endpoint().install_snapshot(parsed).to_dict()

    def receive_direct_events(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply one contiguous Direct batch and return an applied ACK."""
        try:
            parsed = DirectStateBatch.from_dict(batch)
        except (KeyError, TypeError, ValueError) as exc:
            return self._invalid_direct_request_ack(batch, exc)
        return self._ensure_direct_session_endpoint().receive_events(parsed).to_dict()

    @staticmethod
    def _invalid_direct_request_ack(request: dict[str, Any], error: Exception) -> dict[str, Any]:
        return DirectAck(
            subscription_id=str(request.get("subscription_id") or "invalid"),
            stream_epoch=str(request.get("stream_epoch") or "invalid"),
            contiguous_cursor=0,
            status=DirectAckStatus.RESYNC_REQUIRED,
            reason=f"invalid Direct DTO: {type(error).__name__}",
        ).to_dict()

    def get_direct_session_shadow_status(self) -> dict[str, Any]:
        """Return shadow state and transport health without affecting routing."""
        if self._direct_session_projector is None or self._direct_session_endpoint is None:
            return {"enabled": False, "mode": "shadow", "current": [], "terminal": [], "streams": []}
        return {
            "enabled": True,
            **self._direct_session_projector.status(),
            "streams": list(self._direct_session_endpoint.status()),
        }

    def get_router_state_status(self) -> dict[str, Any]:
        """Return Router input ownership and observe-only admission state."""
        if self._router_sticky_projector is None:
            return {
                "mode": RouterStateMode.LEGACY.value,
                "enabled": False,
                "sticky": {"commit_owner": "legacy"},
                "inflight": {"commit_owner": "legacy"},
                "admission": {"enabled": False},
                "observation_failures": 0,
            }
        admission = self._admission_signals_projector
        inflight = self._router_inflight_projector
        return {
            "mode": self._router_state_mode.value,
            "enabled": True,
            "sticky": self._router_sticky_projector.status(),
            "inflight": inflight.status() if inflight is not None else {"commit_owner": "legacy"},
            "admission": admission.status() if admission is not None else {"enabled": False},
            "observation_failures": self._router_observation_failures,
        }

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

    def _resolve_router_state_mode(
        self,
        explicit: str | RouterStateMode | None,
        rollout_config: Any | None,
    ) -> RouterStateMode:
        value = explicit
        if value is None:
            value = self._read_nested(
                rollout_config,
                "custom",
                "agent_framework",
                "collectors",
                "router_state",
                "mode",
            )
        if value is None:
            return RouterStateMode.LEGACY
        if isinstance(value, RouterStateMode):
            return value
        if not isinstance(value, str):
            raise ValueError(f"collectors.router_state.mode must be a string, got {type(value).__name__}")
        try:
            return RouterStateMode(value)
        except ValueError as exc:
            raise ValueError(
                f"collectors.router_state.mode must be one of legacy, shadow, or projector, got {value!r}"
            ) from exc

    def _publish_router_fact(self, event_type: str, payload: dict[str, Any]) -> None:
        publisher = self._router_event_publisher
        if publisher is None:
            return
        try:
            receipt = publisher.publish(event_type, payload)
        except Exception:
            self._router_observation_failures += 1
            logger.exception("Failed to publish Router fact %s", event_type)
            return
        self._router_observation_failures += receipt.failed + receipt.dropped

    def _publish_capacity_change(
        self,
        replica_id: str,
        *,
        reason: str,
        route_request_id: str | None = None,
        health: str = "healthy",
    ) -> None:
        publisher = self._router_event_publisher
        if publisher is None:
            return
        version = self._capacity_versions.get(replica_id, 0) + 1
        self._capacity_versions[replica_id] = version
        if route_request_id is not None:
            self._publish_router_fact(
                ROUTE_COMMITTED,
                {
                    "request_id": route_request_id,
                    "replica_id": replica_id,
                    "ledger_version": version,
                    "status": "committed",
                },
            )
        self._publish_router_fact(
            REPLICA_CAPACITY_CHANGED,
            {
                "replica_id": replica_id,
                "replica_epoch": self._replica_epochs[replica_id],
                "ledger_version": version,
                "reason": reason,
                "health": health,
                "replica_inflight": self._inflight.get(replica_id, 0),
                "total_inflight": self.get_total_inflight(),
            },
        )

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
        released = self._inflight.get(server_id, 0) > 0
        if released:
            self._inflight[server_id] -= 1
        self._fire("on_release", server_id, request_id)
        if released and self._router_event_publisher is not None:
            self._publish_capacity_change(server_id, reason="release")

    def _unhealthy_router_projector(self, *, committing: bool) -> str | None:
        """Name the first unhealthy projector that must block route expansion.

        Only projector-mode failures fail closed: a shadow projector's state is
        comparison-only and its faults must not change routing availability.
        """
        if self._router_state_mode is not RouterStateMode.PROJECTOR:
            return None
        projectors = (
            ("sticky", self._router_sticky_projector),
            ("inflight", self._router_inflight_projector),
        )
        for name, projector in projectors:
            if projector is not None and not projector.healthy:
                if committing:
                    return f"Router {name} Projector failed while committing the route"
                return f"Router {name} Projector is unhealthy"
        return None

    def acquire_server(self, request_id: str, prompt_ids: list[int] | None = None) -> tuple[str, Any]:
        """Return the highest-ranked server id and handle."""
        reason = self._unhealthy_router_projector(committing=False)
        if reason is not None:
            raise RuntimeError(reason)
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
        reason = self._unhealthy_router_projector(committing=True)
        if reason is not None:
            raise RuntimeError(reason)
        if self._router_event_publisher is not None:
            self._publish_capacity_change(server_id, reason="acquire", route_request_id=request_id)
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
        projector = self._router_sticky_projector
        if projector is None:
            cleared = self._store.clear_sticky_bindings()
        elif self._router_state_mode is RouterStateMode.PROJECTOR:
            cleared = projector.clear(commit=True)
        else:
            cleared = self._store.clear_sticky_bindings()
            projector.clear(commit=False)
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
            if self._router_event_publisher is not None:
                self._capacity_versions.setdefault(sid, 0)
                self._replica_epochs[sid] = uuid4().hex
                self._publish_capacity_change(sid, reason="server_added")

    def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers; fires ``on_servers_removed`` to invalidate sticky bindings."""
        removed = []
        for sid in server_ids:
            if sid in self._servers:
                removed.append(sid)
            self._servers.pop(sid, None)
            self._inflight.pop(sid, None)
        if server_ids:
            self._fire("on_servers_removed", server_ids)
            if self._router_event_publisher is not None:
                for server_id in removed:
                    self._publish_capacity_change(server_id, reason="server_removed", health="removed")
