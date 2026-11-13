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

"""Typed application boundary; parsers never request store context."""

from typing import Protocol

from ..types import InflightUpdate
from .parse import KVCacheUpdate, MetricsUpdate, StickyUpdate


class UpdateSink(Protocol):
    """Apply decoded updates; own all store-side bookkeeping."""

    def apply_kv(self, update: KVCacheUpdate) -> None: ...

    def apply_metrics(self, update: MetricsUpdate) -> None: ...

    def apply_sticky(self, update: StickyUpdate) -> None: ...

    def apply_inflight(self, update: InflightUpdate) -> None: ...
