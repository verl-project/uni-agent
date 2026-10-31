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

"""Serialized cache invalidation and Clear accounting."""

import asyncio
import uuid
from dataclasses import replace

import msgpack

from ..collectors.transport.lifecycle import notify_lifecycle
from ..types.cache_lifecycle import CacheLifecycleState


def is_clear(event):
    tag = event.get("type") if isinstance(event, dict) else event[0]
    return tag == 2 or str(tag).lower().replace("_", "") in {"clear", "allblockscleared"}


class CacheLifecycle:
    def __init__(self, notifier=notify_lifecycle, failure_callback=None):
        self.state = CacheLifecycleState(uuid.uuid4().hex)
        self.operation_lock = asyncio.Lock()
        self.sinks = {}
        self._notifications = {}
        self.notifier = notifier
        self.failure_callback = failure_callback
        self._recovery_blocked = False
        self.failed = False

    def bind(self, sink):
        self.sinks[sink["token"]] = dict(sink)
        return self.state.to_dict()

    async def unbind(self, token):
        self.sinks.pop(token, None)
        pending = tuple(self._notifications.get(token, ()))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _change(self, **values):
        self.state = replace(self.state, revision=self.state.revision + 1, **values)

    async def publish(self):
        snapshot = self.state.to_dict()
        try:
            for sink in list(self.sinks.values()):
                token = sink["token"]
                if token not in self.sinks:
                    continue
                task = asyncio.create_task(self.notifier(sink, snapshot))
                pending = self._notifications.setdefault(token, set())
                pending.add(task)
                try:
                    await task
                finally:
                    pending.discard(task)
                    if not pending:
                        self._notifications.pop(token, None)
        except Exception:
            self.failed = True
            if self.failure_callback is not None:
                self.failure_callback()
            self._recovery_blocked = True
            self._change(serving=False, index_state="unknown", reason="lifecycle_notification_failed")
            raise

    async def begin(self, reason: str, expected_clears: int = 0):
        values = {"serving": False, "reason": reason}
        if expected_clears:
            values.update(
                epoch=self.state.epoch + 1,
                index_state="unknown" if self._recovery_blocked else "awaiting_clear",
                expected_clears=self.state.expected_clears + expected_clears,
            )
        self._change(**values)
        await self.publish()

    async def finish(self, serving: bool):
        if serving and self.failed:
            raise RuntimeError("failed cache lifecycle must be repaired before opening routing")
        self._change(serving=serving, reason=self.state.reason if self.state.index_state != "ready" else None)
        await self.publish()

    async def fail(self, reason: str):
        self.failed = True
        if self.failure_callback is not None:
            self.failure_callback()
        self._recovery_blocked = True
        self._change(serving=False, index_state="unknown", reason=reason)
        # Keep the original operation error if notification also fails.
        try:
            await self.publish()
        except Exception:
            pass

    async def stream_gap(self, reason):
        self._recovery_blocked = True
        self._change(index_state="unknown", reason=reason)
        await self.publish()

    async def new_instance(self):
        self.failed = False
        self._recovery_blocked = False
        self.state = CacheLifecycleState(uuid.uuid4().hex, serving=False)
        await self.publish()

    async def accept(self, payload: bytes) -> bytes | None:
        """Track native Clears and forward events from the current cache epoch."""
        batch = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        events = batch["events"] if isinstance(batch, dict) else batch[1]
        last_clear = None
        for i, event in enumerate(events):
            if not is_clear(event):
                continue
            last_clear = i
            observed = self.state.observed_clears + 1
            if observed > self.state.expected_clears:
                # A Clear outside managed operations starts an unknown epoch.
                self._recovery_blocked = True
                self._change(
                    epoch=self.state.epoch + 1,
                    index_state="unknown",
                    reason="unmanaged_clear",
                    observed_clears=observed,
                )
            else:
                self._change(observed_clears=observed)
        if last_clear is not None:
            if not self._recovery_blocked and self.state.observed_clears == self.state.expected_clears:
                self._change(index_state="ready", reason=None)
            await self.publish()
        if self.state.index_state != "ready":
            return None
        if last_clear is not None:
            # Forward the final Clear and subsequent events in order.
            if isinstance(batch, dict):
                return msgpack.packb({**batch, "events": events[last_clear:]}, use_bin_type=True)
            return msgpack.packb([batch[0], events[last_clear:], *batch[2:]], use_bin_type=True)
        return payload
