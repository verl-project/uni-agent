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

"""Decode native KV messages without keeping cache identity state."""

from __future__ import annotations

import logging

import msgpack

from ....collectors.parse import Parser
from ....collectors.parse.vllm.kv_event import decode_kv_events
from ....types.kv_cache import InvalidKVEventPayload, KVCacheUpdate

logger = logging.getLogger(__name__)


class VLLMKVParser(Parser):
    """Normalize native batches into one ordered event list.

    Transport supplies raw payloads. Cache layout, parent hashes and replica
    residency are interpreted by the store.
    """

    def parse(self, raw_data: bytes | str, replica_id: str) -> KVCacheUpdate | None:
        """Return the ordered update, or None for unsupported input types.

        Raises:
            InvalidKVEventPayload: The batch cannot be applied safely. The transport
                must recover before advancing its replay cursor.
        """
        if isinstance(raw_data, str):
            logger.debug("VLLMKVParser received string data, expected bytes — skipping")
            return None
        try:
            raw = msgpack.unpackb(raw_data, raw=False)
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"Unexpected msgpack format (type={type(raw).__name__})")
            batches = raw if isinstance(raw[0], list) else [raw]
            events = [event for batch in batches for event in decode_kv_events(batch, default_replica_id=replica_id)]
            return KVCacheUpdate(replica_id=replica_id, events=events)
        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            logger.warning(
                "Invalid KV event payload from %s (%s, %d bytes)", replica_id, type(exc).__name__, len(raw_data)
            )
            raise InvalidKVEventPayload("Invalid KV event payload") from None
