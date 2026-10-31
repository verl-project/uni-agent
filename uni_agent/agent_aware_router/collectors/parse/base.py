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

"""Parser abstract base and its update return types.

A ``Parser`` turns a raw transport payload (bytes/str) into a structured
update that the ``Collector`` applies to the store.  The update types
(``KVCacheUpdate``, ``MetricsUpdate``) live here, next to the base, because
they are the parser's output contract — the layer that both produces and
consumes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...types import Layer


@dataclass
class KVCacheUpdate:
    """Mutable accumulator for KVCacheStore updates, built by the parser.

    The parser dispatches each event to a handler that calls ``add``/``remove``
    to fold blocks in; the collector then reads ``add_blocks``/``remove_blocks``
    to write the store.

    Attributes:
        node_id: Target endpoint identifier.
        add_blocks: layer → block hashes accumulated to add.
        remove_blocks: layer → block hashes accumulated to remove.
        clear_all: If True, clear all blocks for this node.
        block_size: Block size learned from the first BlockStored (None until then).
    """

    node_id: str
    add_blocks: dict[Layer, list[str]] = field(default_factory=dict)
    remove_blocks: dict[Layer, list[str]] = field(default_factory=dict)
    clear_all: bool = False
    block_size: int | None = None

    def add(self, layer: Layer, block_hashes: list[str]) -> None:
        """Fold stored blocks into ``add_blocks`` under ``layer``."""
        self.add_blocks.setdefault(layer, []).extend(block_hashes)

    def remove(self, layer: Layer, block_hashes: list[str]) -> None:
        """Fold removed blocks into ``remove_blocks`` under ``layer``."""
        self.remove_blocks.setdefault(layer, []).extend(block_hashes)

    def clear(self) -> None:
        """Mark the replica for a full block clear."""
        self.clear_all = True

    def set_block_size(self, size: int) -> None:
        """Set the learned block size (first BlockStored wins; set by the parser)."""
        self.block_size = size


@dataclass
class MetricsUpdate:
    """Structured update command for PerReplicaStore.

    Attributes:
        node_id: Target endpoint identifier.
        metrics: Dict of canonical_key → value.
        is_delta: When ``False`` (default, ``VLLMMetricsParser``) the values
            are absolute gauges applied via ``refresh`` (merge overwrite). When
            ``True`` (``InflightParser``) the values are fully-netted signed
            deltas applied via ``incr`` — the parser owns the netting (turn /
            token folding, stale re-acquire recovery); the store owns the
            running counters.
        request_id: Optional routing request id carried with the update
            (``None`` when the update isn't request-scoped).
    """

    node_id: str
    metrics: dict[str, Any]
    is_delta: bool = False
    request_id: str | None = None


@dataclass
class InflightBlockUpdate:
    """Structured command for the router's held-block book (``KVCacheStore``).

    Emitted by ``InflightParser`` from the Balancer's ``on_acquire`` /
    ``on_release`` callbacks. The parser owns the acquire/release protocol
    decisions (when to pin, when a re-acquire must first unwind a stale pin);
    the collector applies each command mechanically — pin/unpin sets for
    ``unpin`` / ``stale_unpin`` come from the per-request marker row
    (``KVCacheStore.pop_marker``), which is the ground truth of what the
    dispatch pinned.

    Attributes:
        node_id: Replica the request landed on — the ``pin`` target;
            informational for ``unpin`` / ``stale_unpin`` (whose replica comes
            from the marker row and may be a previous one).
        request_id: The routing request id the command is scoped to.
        action: ``"pin"`` / ``"unpin"`` / ``"stale_unpin"``.
        block_hashes: Resolved full-block prefix hash chain — ``pin`` only.
        mark_resident: ``pin`` only — also record the blocks resident
            (``KVCacheStore.record_dispatch_blocks``) so ``gpu_hit`` sees the
            prefix immediately instead of trailing the kv-event stream.
    """

    node_id: str
    request_id: str
    action: str
    block_hashes: list[str] = field(default_factory=list)
    mark_resident: bool = False


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
    returning a structured update the ``Collector`` applies to the store — a
    single ``KVCacheUpdate`` / ``MetricsUpdate`` / ``StickyUpdate``, a
    ``list`` of them (mixed with ``InflightBlockUpdate``), or ``None`` on
    failure/skip.
    """

    @abstractmethod
    def parse(
        self, raw_data: bytes | str | Any, node_id: str
    ) -> KVCacheUpdate | MetricsUpdate | StickyUpdate | InflightBlockUpdate | list | None:
        """Parse raw data and return a structured update (or list of them).

        Args:
            raw_data: Raw payload — ``bytes`` (from ZMQ) / ``str`` (from HTTP
                response text) for network transports, or a ``StatisticEvent``
                for the callback transport (sticky / inflight parsers).
            node_id: Source endpoint identifier (empty string for the callback
                transport — the event carries its own request/replica ids).

        Returns:
            A single update, a list of updates (applied in order), or ``None``
            if parsing fails or the payload type is not handled.
        """
