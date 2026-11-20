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

"""Parser abstract base and collector-local update types.

``KVCacheUpdate`` is the shared domain contract, re-exported here for parser
consumers. Metrics and sticky updates remain local to the collector pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ...types import InflightUpdate
from ...types.kv_cache import KVCacheUpdate


@dataclass
class MetricsUpdate:
    """Structured update command for PerReplicaStore.

    Attributes:
        replica_id: Target endpoint identifier.
        metrics: Dict of canonical_key → value.
        is_delta: When ``False`` (default, ``VLLMMetricsParser``) the values
            are absolute gauges applied via ``refresh`` (merge overwrite). When
            ``True`` the values are signed deltas applied via ``incr``.
            Inflight accounting is derived inside DataStore, not by a parser.
        request_id: Optional routing request id carried with the update
            (``None`` when the update isn't request-scoped).
    """

    replica_id: str
    metrics: dict[str, Any]
    is_delta: bool = False
    request_id: str | None = None


@dataclass
class StickyUpdate:
    """Structured update command for the per-request store (sticky binding).

    Emitted by ``StickyParser`` from the Balancer's ``on_acquire`` /
    ``on_servers_removed`` callbacks (packed into a ``StatisticEvent`` by
    ``CallbackTransport``). Distinct from ``KVCacheUpdate`` (per-replica
    block) and ``MetricsUpdate`` (per-replica gauge) because sticky is a
    per-request dimension with LRU recency semantics — refresh/merge can't
    express ``put`` / ``invalidate_replica``.

    Attributes:
        action: ``"put"`` / ``"invalidate"`` / ``"invalidate_replica"``.
        request_id: Bound request id (for ``put`` / ``invalidate``).
        replica_id: Bound replica id (for ``put``); ignored on invalidate_replica.
        replica_ids: Replicas to clear (for ``invalidate_replica``).
    """

    action: str
    request_id: str | None = None
    replica_id: str | None = None
    replica_ids: tuple[str, ...] = ()


class Parser(ABC):
    """Abstract base for data parsers.

    Subclasses implement ``parse()`` with their backend-specific parsing logic,
    returning a structured update the application sink applies to the store — a
    single ``KVCacheUpdate`` / ``MetricsUpdate`` / ``StickyUpdate`` /
    ``InflightUpdate``, a ``list`` of them, or ``None`` on
    failure/skip.
    """

    @abstractmethod
    def parse(
        self, raw_data: bytes | str | Any, replica_id: str
    ) -> KVCacheUpdate | MetricsUpdate | StickyUpdate | InflightUpdate | list | None:
        """Parse raw data and return a structured update (or list of them).

        Args:
            raw_data: Raw payload — ``bytes`` (from ZMQ) / ``str`` (from HTTP
                response text) for network transports, or a ``StatisticEvent``
                for the callback transport (sticky / inflight parsers).
            replica_id: Source endpoint identifier (empty string for the callback
                transport — the event carries its own request/replica ids).

        Returns:
            A single update, a list of updates (applied in order), or ``None``
            for skipped input.
        """
