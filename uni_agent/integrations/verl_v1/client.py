# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Own dispatched Ray requests without replacing verl's client implementation."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import ray

from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


class GenerationCancellationError(RuntimeError):
    """Remote cancellation was not confirmed; training must not resume."""


async def _settle(task: asyncio.Task):
    """Preserve bounded cleanup through repeated cancellation of its caller."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


class CancellableLLMServerClient(LLMServerClient):
    """Keep the native routing contract and acknowledge request cancellation."""

    def __init__(self, config, load_balancer_handle=None, *, cancellation_timeout=5.0, **kwargs):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        if not math.isfinite(cancellation_timeout) or cancellation_timeout <= 0:
            raise ValueError("cancellation_timeout must be finite and positive")
        self.cancellation_timeout = cancellation_timeout

    async def _acquire_owned(self, request_id, **kwargs):
        pending = asyncio.create_task(self._acquire_server(request_id, **kwargs))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # The router may already have allocated a slot. Do not cancel the
            # local acquisition future and lose the late allocation identity.
            def release_late(done):
                if done.cancelled():
                    return
                try:
                    server_id, _ = done.result()
                except Exception:
                    return
                try:
                    self._release_server(server_id, request_id=request_id)
                except Exception:
                    logger.warning("Late routing reservation could not be released for %s", request_id)

            async def release_acquisition():
                try:
                    server_id, _ = await asyncio.wait_for(asyncio.shield(pending), timeout=self.cancellation_timeout)
                except asyncio.TimeoutError:
                    pending.add_done_callback(release_late)
                    raise
                self._release_server(server_id, request_id=request_id)

            cleanup = asyncio.create_task(release_acquisition())
            try:
                await _settle(cleanup)
            except Exception as error:
                raise GenerationCancellationError("Routing cancellation was not confirmed") from error
            raise

    async def _cancel_dispatched(self, reference):
        async def acknowledge():
            ray.cancel(reference)
            try:
                # A new ray.get observes remote termination, not the cancelled
                # local Future previously used by ObjectRef.__await__.
                await asyncio.to_thread(ray.get, reference, timeout=self.cancellation_timeout)
            except ray.exceptions.TaskCancelledError:
                return

        try:
            await _settle(asyncio.create_task(acknowledge()))
        except Exception as error:
            raise GenerationCancellationError("Generation cancellation was not confirmed") from error

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        **kwargs,
    ) -> TokenOutput:
        """Dispatch one native generation and retain its cancellation authority.

        Argument forwarding derives from verl V1 at 2bc6ac46. Routing,
        request IDs, output metadata, and full-async resumption remain native.
        An unacknowledged cancellation deliberately retains the reservation.
        """
        server_id, server = await self._acquire_owned(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            **kwargs,
        )
        release = True
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            processor_output = kwargs.pop("mm_processor_output", None)
            if self.config.actor_rollout_ref.rollout.name == "sglang":
                video_data = processor_output
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority} if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm" else {}
            )
            reference = server.generate.remote(
                request_id=self._vllm_request_id(request_id),
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                session_id=request_id,
                **kwargs,
            )
            try:
                output = await reference
            except asyncio.CancelledError:
                release = False
                await self._cancel_dispatched(reference)
                release = True
                raise
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            if release:
                self._release_server(server_id, request_id=request_id)


class CancellableFullyAsyncLLMServerClient(FullyAsyncLLMServerClient, CancellableLLMServerClient):
    """Keep native resume/budget handling and dispatch through the owned client."""
