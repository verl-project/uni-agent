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

"""Narrow store contracts for scoring and post-write observation."""

from __future__ import annotations

from typing import Any, Protocol

from ..types import Layer


class MetricsReader(Protocol):
    def get_metric(self, node_id: str, key: str) -> Any: ...

    def get_metrics(self, node_id: str) -> dict[str, Any]: ...

    def get_metric_node_ids(self) -> list[str]: ...


class KVReader(Protocol):
    def get_block_size(self) -> int | None: ...

    def get_layer_prefix_hit_rate(
        self,
        node_id: str,
        hash_strs: list[str],
        layer: Layer = Layer.GPU,
    ) -> float: ...

    def kv_cache_load(self, node_id: str) -> float: ...

    def per_replica_block_counts(self) -> dict[str, int]: ...

    def inflight_block_count(self, replica_id: str) -> int: ...


class RequestReader(Protocol):
    def get_sticky_binding(self, request_id: str) -> str | None: ...


class RoutingRequests(RequestReader, Protocol):
    """Scoring may memoize prefix hashes, but cannot mutate bindings or bookings."""

    def get_prefix_cache(self, request_id: str) -> dict[str, Any] | None: ...

    def set_prefix_cache(self, request_id: str, value: dict[str, Any]) -> None: ...


class RoutingState(Protocol):
    @property
    def metrics(self) -> MetricsReader: ...

    @property
    def kv(self) -> KVReader: ...

    @property
    def requests(self) -> RoutingRequests: ...
