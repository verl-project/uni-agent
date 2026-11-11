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

from uni_agent.vllm_plugins.lifecycle import RESET_BOUNDARY_KEY

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
        self.failed = False
        self.cache_metadata = None
        self._recovery_nonce = None
        self._recovery_ready = asyncio.Event()

    def bind(self, sink):
        self.sinks[sink["token"]] = dict(sink)
        return self.state.to_dict()

    async def unbind(self, token, generation=None):
        sink = self.sinks.get(token)
        if sink is None or (generation is not None and sink.get("generation") != generation):
            return
        self.sinks.pop(token)
        pending = tuple(self._notifications.get((token, sink.get("generation")), ()))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _change(self, **values):
        self.state = replace(self.state, revision=self.state.revision + 1, **values)

    async def publish(self):
        snapshot = self.state.to_dict()
        metadata = self.cache_metadata
        try:
            for sink in list(self.sinks.values()):
                token = sink["token"]
                if self.sinks.get(token) != sink:
                    continue
                notification = (
                    self.notifier(sink, snapshot, cache_metadata=metadata)
                    if metadata is not None
                    else self.notifier(sink, snapshot)
                )
                task = asyncio.create_task(notification)
                notification_key = token, sink.get("generation")
                pending = self._notifications.setdefault(notification_key, set())
                pending.add(task)
                try:
                    await task
                except Exception:
                    if self.sinks.get(token) == sink:
                        raise
                finally:
                    pending.discard(task)
                    if not pending:
                        self._notifications.pop(notification_key, None)
        except Exception:
            self.failed = True
            if self.failure_callback is not None:
                self.failure_callback()
            self._change(serving=False, index_state="unknown", reason="lifecycle_notification_failed")
            raise

    async def begin(self, reason: str, expected_clears: int = 0):
        values = {"serving": False, "reason": reason}
        if expected_clears:
            values.update(
                epoch=self.state.epoch + 1,
                index_state="unknown" if self.state.index_state == "unknown" else "awaiting_clear",
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
        self._recovery_nonce = None
        self.failed = True
        if self.failure_callback is not None:
            self.failure_callback()
        self._change(serving=False, index_state="unknown", reason=reason)
        # Keep the original operation error if notification also fails.
        try:
            await self.publish()
        except Exception:
            pass

    async def stream_gap(self, reason):
        self._change(index_state="unknown", reason=reason)
        await self.publish()

    async def new_instance(self, cache_metadata=None):
        self.failed = False
        self.cache_metadata = cache_metadata
        self._recovery_nonce = None
        self.state = CacheLifecycleState(uuid.uuid4().hex, revision=self.state.revision + 1, serving=False)
        await self.publish()

    async def prepare_recovery(self, nonce):
        """Ignore old Clear counts until a checked native reset's exact marker."""
        self._recovery_nonce = nonce
        self._recovery_ready = asyncio.Event()
        self._change(
            serving=False,
            epoch=self.state.epoch + 1,
            index_state="awaiting_clear",
            expected_clears=1,
            observed_clears=0,
            reason="cache_recovery",
        )
        await self.publish()

    async def wait_for_recovery(self, timeout):
        try:
            await asyncio.wait_for(self._recovery_ready.wait(), timeout)
        except BaseException:
            self._recovery_nonce = None
            raise

    async def accept(self, payload: bytes) -> bytes | None:
        """Track native Clears and forward events from the current cache epoch."""
        batch = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        events = batch["events"] if isinstance(batch, dict) else batch[1]
        if self._recovery_nonce is not None:
            extension = batch if isinstance(batch, dict) else batch[-1] if len(batch) > 3 else None
            marker = extension.get(RESET_BOUNDARY_KEY) if isinstance(extension, dict) else None
            if (
                not isinstance(marker, dict)
                or type(marker.get("schema_version")) is not int
                or marker.get("schema_version") != 1
                or marker.get("nonce") != self._recovery_nonce
            ):
                return None
            clears = [i for i, event in enumerate(events) if is_clear(event)]
            if not clears:
                return None
            completion = self._recovery_ready
            self._recovery_nonce = None
            self.failed = False
            self._change(index_state="ready", expected_clears=1, observed_clears=1, reason=None)
            await self.publish()
            completion.set()
            if isinstance(batch, dict):
                forwarded = {**batch, "events": events[clears[-1] :]}
                forwarded.pop(RESET_BOUNDARY_KEY, None)
                return msgpack.packb(forwarded, use_bin_type=True)
            return msgpack.packb([batch[0], events[clears[-1] :], *batch[2:-1]], use_bin_type=True)
        last_clear = None
        for i, event in enumerate(events):
            if not is_clear(event):
                continue
            last_clear = i
            observed = self.state.observed_clears + 1
            if observed > self.state.expected_clears:
                # A Clear outside managed operations starts an unknown epoch.
                self._change(
                    epoch=self.state.epoch + 1,
                    index_state="unknown",
                    reason="unmanaged_clear",
                    observed_clears=observed,
                )
            else:
                self._change(observed_clears=observed)
        if last_clear is not None:
            if self.state.index_state != "unknown" and self.state.observed_clears == self.state.expected_clears:
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
