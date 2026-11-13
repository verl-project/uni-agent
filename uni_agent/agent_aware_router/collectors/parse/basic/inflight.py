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

"""Decode callback payloads into inflight facts; all accounting belongs to DataStore."""

from __future__ import annotations

from typing import Any

from ....types import InflightUpdate
from ...transport.callback import StatisticEvent
from .. import Parser


class InflightParser(Parser):
    """Normalize supported callbacks without reading or deriving store state."""

    def parse(self, raw_data: bytes | str | Any, node_id: str) -> InflightUpdate | None:
        if not isinstance(raw_data, StatisticEvent) or raw_data.replica_id is None:
            return None
        if raw_data.event == "on_acquire":
            return InflightUpdate("acquire", raw_data.replica_id, raw_data.request_id, raw_data.prompt_ids)
        if raw_data.event == "on_release":
            return InflightUpdate("release", raw_data.replica_id, raw_data.request_id)
        return None
