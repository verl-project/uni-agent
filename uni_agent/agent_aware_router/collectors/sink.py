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

"""Typed application boundary and context adapter for the pure inflight parser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .parse import InflightBlockUpdate, KVCacheUpdate, MetricsUpdate, Parser, StickyUpdate
from .parse.basic.inflight import InflightParser
from .transport.callback import StatisticEvent


@dataclass(frozen=True)
class InflightContext:
    turn: int = 0
    stale: dict[str, Any] | None = None
    block_size: int = 0
    booked_len: int = 0


class UpdateSink(Protocol):
    """Apply typed updates in parser order; own all store-side bookkeeping."""

    def apply_kv(self, update: KVCacheUpdate) -> None: ...

    def apply_metrics(self, update: MetricsUpdate) -> None: ...

    def apply_sticky(self, update: StickyUpdate) -> None: ...

    def apply_inflight_blocks(self, update: InflightBlockUpdate) -> None: ...

    def prepare_inflight(self, event: StatisticEvent) -> InflightContext: ...


class ContextualInflightParser(Parser):
    """Supply application-owned context while retaining the pure parser's decisions."""

    def __init__(self, parser: InflightParser, sink: UpdateSink) -> None:
        self._parser = parser
        self._sink = sink

    def parse(self, raw_data: Any, node_id: str) -> list | None:
        if not isinstance(raw_data, StatisticEvent) or raw_data.replica_id is None:
            return None
        if raw_data.event not in ("on_acquire", "on_release"):
            return None
        context = self._sink.prepare_inflight(raw_data)
        return self._parser.parse(
            raw_data,
            node_id,
            turn=context.turn,
            stale=context.stale,
            block_size=context.block_size,
            booked_len=context.booked_len,
        )
