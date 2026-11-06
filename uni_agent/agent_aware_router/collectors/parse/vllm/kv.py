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
from ....collectors.transport.base import TransportMessage
from ....types.kv_cache import KVCacheUpdate

logger = logging.getLogger(__name__)


class VLLMKVParser(Parser):
    """Normalize native array and mapping batches into one ordered event list.

    Transport metadata arrives in ``TransportMessage``. Cache layout, parent hashes
    and replica residency are interpreted by the store.
    """

    def parse(self, message: TransportMessage, node_id: str) -> KVCacheUpdate:
        """Translate protocol status and decode one ordered native event batch."""
        if not isinstance(message, TransportMessage):
            raise TypeError("VLLMKVParser requires a TransportMessage")
        if message.status == "new_stream":
            return KVCacheUpdate(node_id=node_id, sequence=message.sequence, reset_stream=True)
        if message.status in {"no_continuity", "unreachable"}:
            return KVCacheUpdate(
                node_id=node_id, sequence=message.sequence, stream_error=f"event_stream_{message.status}"
            )

        try:
            raw = msgpack.unpackb(message.payload, raw=False)
            if not isinstance(raw, list | dict) or not raw:
                raise ValueError(f"Unexpected msgpack format (type={type(raw).__name__})")
            # A native batch is [timestamp, events, ...] or a mapping. Older
            # callers may encode a list of batches; flatten it in wire order.
            batches = raw if isinstance(raw, list) and isinstance(raw[0], list | dict) else [raw]
            events = [event for batch in batches for event in decode_kv_events(batch, default_node_id=node_id)]
            return KVCacheUpdate(node_id=node_id, events=events, sequence=message.sequence)
        except (msgpack.UnpackException, ValueError, TypeError) as exc:
            payload = message.payload
            preview = bytes(payload[:32]).hex() if isinstance(payload, bytes | bytearray) else str(payload)[:64]
            logger.warning("Failed to parse KV payload from node %s: %r (head=%s)", node_id, exc, preview)
            return KVCacheUpdate(node_id=node_id, sequence=message.sequence, stream_error="invalid_kv_event_payload")
