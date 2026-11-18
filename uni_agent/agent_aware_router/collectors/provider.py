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
    ) -> None:
        self._collection_names = list(collection_names)
        self._collectors: list[Collector] = [
            get_collector(
                name,
                collectors_config,
                server_addresses=server_addresses,
                kv_event_endpoints=kv_event_endpoints,
                balancer_handler=balancer_handler,
            )
            for name in self._collection_names
        ]

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
