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

"""Cache lifecycle integration for verl's vLLM server."""

import asyncio
from contextlib import asynccontextmanager

import httpx

from verl.workers.rollout.replica import RolloutMode

from .cache_lifecycle import CacheLifecycle
from .kv_event_bridge import KVEventBridge


class CacheLifecycleServerMixin:
    def _close_cache_admission(self):
        self._submission_paused = True
        self._rejecting = True
        self._resume_event.clear()

    def _cache_control(self):
        if not hasattr(self, "_cache_lifecycle"):
            self._cache_lifecycle = CacheLifecycle(failure_callback=self._close_cache_admission)
            self._cache_bridge_lock = asyncio.Lock()
        return self._cache_lifecycle

    async def _native_cache_source(self):
        host, port = self.get_server_address()
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            response = await client.get(f"http://{host}:{port}/kv_event_sources")
            if response.status_code == 404:
                return host, None
            response.raise_for_status()
        sources = response.json()
        if not sources:
            return host, None
        if len(sources) != 1:
            raise RuntimeError("cache lifecycle bridge requires exactly one DP event source")
        source = next(iter(sources.values()))
        if source.get("publisher") != "zmq":
            raise RuntimeError("cache lifecycle bridge requires native ZMQ events")
        return host, source

    async def _ensure_cache_bridge(self):
        control = self._cache_control()
        async with self._cache_bridge_lock:
            engine = getattr(self, "engine", None)
            if engine is None:
                raise RuntimeError("cache lifecycle engine is not ready")
            if getattr(self, "_cache_bridge_engine", None) is engine:
                return control
            if hasattr(self, "_cache_bridge"):
                # Replace the native subscription before publishing the new instance.
                paused = self._submission_paused
                await control.fail("engine_replaced")
                host, source = await self._native_cache_source()
                await self._cache_bridge.rebind(source)
                self._cache_bridge_engine = engine
                await control.finish(not paused)
                if not paused:
                    self._submission_paused = self._rejecting = False
                    self._resume_event.set()
                return control
            host, source = await self._native_cache_source()
            bridge = KVEventBridge(control, host)
            try:
                await bridge.start(source)
            except BaseException:
                await bridge.close()
                raise
            self._cache_bridge = bridge
            self._cache_bridge_engine = engine
            return control

    async def launch_server(self, *args, **kwargs):
        result = await super().launch_server(*args, **kwargs)
        if self.node_rank == 0:
            # Start event collection before lifecycle operations.
            await self._ensure_cache_bridge()
        return result

    async def register_cache_lifecycle_sink(self, sink):
        control = await self._ensure_cache_bridge()
        state = control.bind(sink)
        return {"state": state, "sources": self._cache_bridge.sources()}

    async def unregister_cache_lifecycle_sink(self, token):
        control = self._cache_control()
        async with self._cache_bridge_lock:
            await control.unbind(token)

    async def _checked_prefix_reset(self):
        result = await self.engine.reset_prefix_cache(reset_connector=True)
        if result is not True:
            raise RuntimeError("engine did not confirm successful prefix cache reset")

    @asynccontextmanager
    async def _cache_operation(self, reason, clears):
        control = await self._ensure_cache_bridge()
        async with control.operation_lock:
            paused = self._submission_paused
            rejecting = self._rejecting
            if not paused:
                self._cache_gate_before_pause = False
            self._close_cache_admission()
            try:
                await control.begin(reason, clears)
                yield paused, rejecting
            except BaseException:
                self._close_cache_admission()
                await control.fail(f"{reason}_failed")
                raise

    async def _finish_cache_operation(self, serving, rejecting=False):
        await self._cache_control().finish(serving)
        if serving:
            self._submission_paused = False
            self._rejecting = rejecting
            self._resume_event.set()

    async def _finish_cache_wake(self, paused):
        serving = not await self.engine.is_sleeping() and not getattr(self, "_cache_gate_before_pause", paused)
        await self._finish_cache_operation(serving)

    async def clear_kv_cache(self):
        if self.node_rank != 0:
            return await super().clear_kv_cache()

        async with self._cache_operation("clear_kv_cache", 1) as (paused, rejecting):
            while self._admitting:
                await asyncio.sleep(0.01)
            await self.engine.wait_for_requests_to_drain()
            await self._checked_prefix_reset()
            await self.engine.reset_mm_cache()
            await self.engine.reset_encoder_cache()
            await self._finish_cache_operation(not paused, rejecting)

    async def abort_all_requests(self, reset_prefix_cache=True, reject_request=False, abort_only=False):
        if abort_only:
            return await super().abort_all_requests(
                reset_prefix_cache=reset_prefix_cache, reject_request=reject_request, abort_only=True
            )
        if self.node_rank != 0:
            return await super().abort_all_requests(reset_prefix_cache, reject_request)
        async with self._cache_operation("abort_all_requests", int(reset_prefix_cache)) as (paused, _):
            if not paused:
                self._cache_gate_before_pause = True
            result = await super().abort_all_requests(reset_prefix_cache, reject_request)
            await self._finish_cache_operation(False)
            return result

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine or self.rollout_mode == RolloutMode.STANDALONE:
            return await super().sleep()
        async with self._cache_operation("sleep", 1):
            result = await super().sleep()
            await self._finish_cache_operation(False)
            return result

    async def wake_up(self, tags=None):
        mode = self.rollout_mode
        if self.node_rank != 0 or mode == RolloutMode.STANDALONE:
            return await super().wake_up(tags)

        async with self._cache_operation("wake_up", 1) as (paused, _):
            await self.engine.wake_up(
                tags=(tags or self._get_wake_up_tags()) if mode == RolloutMode.HYBRID else self._get_wake_up_tags()
            )
            await self._checked_prefix_reset()
            await self._finish_cache_wake(paused)

    async def release_kv_cache(self):
        if self.node_rank != 0 or not self.config.free_cache_engine or self.rollout_mode == RolloutMode.COLOCATED:
            return await super().release_kv_cache()
        async with self._cache_operation("release_kv_cache", 1):
            result = await super().release_kv_cache()
            await self._finish_cache_operation(False)
            return result

    async def resume_kv_cache(self):
        if self.node_rank != 0 or not self.config.free_cache_engine or self.rollout_mode == RolloutMode.COLOCATED:
            return await super().resume_kv_cache()

        async with self._cache_operation("resume_kv_cache", 1) as (paused, _):
            await self.engine.wake_up(tags=["kv_cache"])
            await self._checked_prefix_reset()
            await self._finish_cache_wake(paused)

    async def open_submission_gate(self):
        if self.node_rank != 0:
            return await super().open_submission_gate()
        control = await self._ensure_cache_bridge()
        async with control.operation_lock:
            if await self.engine.is_sleeping():
                raise RuntimeError("cannot open routing while engine allocations are asleep")
            await control.finish(True)
            await super().open_submission_gate()
            self._cache_gate_before_pause = False
