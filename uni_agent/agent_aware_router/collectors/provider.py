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

"""CollectorManager — constructs and runs data collectors.

Owns collector construction and lifecycle (start/stop). Strategies read metrics
from ``DataStore`` (the unified facade over ``PerReplicaStore`` / ``KVCacheStore``),
not from the manager.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..config.collector import CollectorConfig
from .collector import Collector, get_collector

logger = logging.getLogger(__name__)


class CollectorManager:
    """Lifecycle manager for data collectors.

    Args:
        collectors_config: ``CollectorConfig`` — connection tuning parameters.
        collection_names: List of collection names to initialize (e.g.
            ``["vllm_metrics", "vllm_zmq"]``).
        server_addresses: ``{replica_id: ip:port}`` for HTTP transport.
        kv_event_endpoints: ``{replica_id: [sub_addr, replay_addr]}`` for ZMQ transport.
        balancer_handler: The Balancer, forwarded to ``get_collector`` so the
            ``sticky_stat`` / ``inflight_stat`` collectors can build a
            ``CallbackTransport`` that registers its callbacks. Ignored by the
            network collectors.
    """

    def __init__(
        self,
        collectors_config: CollectorConfig,
        collection_names: list[str],
        server_addresses: dict[str, str] | None = None,
        kv_event_endpoints: dict[str, list[str]] | None = None,
        balancer_handler=None,
        metadata_servers: dict[str, Any] | None = None,
    ) -> None:
        self._collection_names = list(collection_names)
        if metadata_servers is not None:
            self._collection_names.insert(0, "vllm_metadata")
        self._collectors: list[Collector] = [
            get_collector(
                name,
                collectors_config,
                server_addresses=server_addresses,
                kv_event_endpoints=kv_event_endpoints,
                balancer_handler=balancer_handler,
                **({"metadata_servers": metadata_servers} if name == "vllm_metadata" else {}),
            )
            for name in self._collection_names
        ]
        self._pending_replays: dict[str, int] = {}
        self._server_addresses = dict(server_addresses or {})
        self._kv_event_endpoints = dict(kv_event_endpoints or {})
        self._network_lock = threading.RLock()
        self._metadata_collector = next(
            (
                collector
                for name, collector in zip(self._collection_names, self._collectors, strict=True)
                if name == "vllm_metadata"
            ),
            None,
        )
        if self._metadata_collector is not None:
            self._metadata_collector._on_metadata_replay = self._request_replay
            self._metadata_collector._transport._on_endpoints = self._update_endpoints
            self._metadata_collector._transport._requires_kv_events = "vllm_zmq" in self._collection_names

    def _request_replay(self, replica_id: str, revision: int) -> None:
        with self._network_lock:
            self._pending_replays[replica_id] = revision
            self._sync_endpoints(replica_id)

    def _update_endpoints(self, replica_id: str, address: str, kv_endpoint: list[str] | None) -> None:
        with self._network_lock:
            self._server_addresses.setdefault(replica_id, address)
            if kv_endpoint is not None:
                self._kv_event_endpoints.setdefault(replica_id, kv_endpoint)
            self._sync_endpoints(replica_id)

    def _sync_endpoints(self, replica_id: str) -> None:
        address = self._server_addresses.get(replica_id)
        endpoint = self._kv_event_endpoints.get(replica_id)
        for name, collector in zip(self._collection_names, self._collectors, strict=True):
            runner = collector._loop_runner
            if runner is None or not runner.loop.is_running():
                continue
            if name == "vllm_metrics" and address is not None:
                runner.loop.call_soon_threadsafe(collector._transport.register_endpoints, {replica_id: address})
            elif name == "vllm_zmq" and endpoint is not None:
                runner.loop.call_soon_threadsafe(collector._transport.register_endpoints, {replica_id: endpoint})
                revision = self._pending_replays.pop(replica_id, None)
                if revision is not None:
                    runner.loop.call_soon_threadsafe(collector._transport.request_replay, replica_id, revision)

    def get_status(self) -> list[dict]:
        """Return each collector's diagnostics in configured order."""
        return [
            {"name": name, **collector.get_status()}
            for name, collector in zip(self._collection_names, self._collectors, strict=True)
        ]

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all collectors, rolling back this attempt if one fails."""
        for collector in self._collectors:
            collector._ensure_startable()
        attempted: list[Collector] = []
        try:
            for collector in self._collectors:
                attempted.append(collector)
                collector.start()
                if collector is self._metadata_collector:
                    collector._transport.wait_initial(5.0)
            with self._network_lock:
                for replica in self._server_addresses.keys() | self._kv_event_endpoints.keys():
                    self._sync_endpoints(replica)
        except Exception:
            for collector in reversed(attempted):
                try:
                    collector.stop()
                except Exception:
                    logger.exception("Failed to roll back collector %s", type(collector).__name__)
            raise

    def stop(self) -> None:
        """Attempt all cleanup; retry timeouts, replace collectors with terminal failure."""
        errors: list[Exception] = []
        for collector in self._collectors:
            try:
                collector.stop()
            except Exception as exc:
                errors.append(exc)
                logger.exception("Failed to stop collector %s", type(collector).__name__)
        if errors:
            raise RuntimeError(f"Failed to stop {len(errors)} collector(s)") from errors[0]
