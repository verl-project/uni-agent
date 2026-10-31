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

"""Exercise lifecycle operations through the real verl server class."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm")

from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle
from uni_agent.agent_aware_router.server.http_server import KVMetadataHttpServer
from verl.workers.rollout.replica import RolloutMode

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def server():
    value = object.__new__(KVMetadataHttpServer)
    value.node_rank = 0
    value.config = SimpleNamespace(free_cache_engine=True)
    value.model_config = SimpleNamespace(lora_rank=0, lora={})
    value.rollout_mode = RolloutMode.COLOCATED
    value._submission_paused = value._rejecting = False
    value._admitting = 0
    value._resume_event = asyncio.Event()
    value._resume_event.set()
    value._cache_lifecycle = CacheLifecycle()
    value._ensure_cache_bridge = AsyncMock(return_value=value._cache_lifecycle)
    value.engine = SimpleNamespace(
        reset_prefix_cache=AsyncMock(return_value=True),
        reset_mm_cache=AsyncMock(),
        reset_encoder_cache=AsyncMock(),
        pause_generation=AsyncMock(),
        abort=AsyncMock(),
        output_processor=SimpleNamespace(
            request_states={"first": None, "second": None}, parent_requests={"parent": None}
        ),
        sleep=AsyncMock(),
        wake_up=AsyncMock(),
        is_sleeping=AsyncMock(return_value=False),
        wait_for_requests_to_drain=AsyncMock(),
    )
    return value


@pytest.mark.parametrize(
    "paused,rank,failed", [(False, 0, False), (True, 0, False), (False, 1, False), (False, 0, True)]
)
def test_abort_only_keeps_cache_and_admission_state(paused, rank, failed, server):
    async def run():
        server.node_rank = rank
        server._submission_paused = server._rejecting = paused
        if paused:
            server._resume_event.clear()
        before = server._cache_lifecycle.state.to_dict()
        if rank:
            del server.engine
        elif failed:
            server.engine.abort.side_effect = RuntimeError("cancel failed")
        server._ensure_cache_bridge.side_effect = AssertionError("cancellation must not start a cache operation")
        server._cache_lifecycle.notifier = AsyncMock()
        if failed:
            with pytest.raises(RuntimeError, match="cancel failed"):
                await server.abort_all_requests(abort_only=True)
        else:
            observed = await server.abort_all_requests(reset_prefix_cache=True, reject_request=True, abort_only=True)
            assert observed == {"aborted_count": 0 if rank else 2, "request_ids": [] if rank else ["first", "second"]}
        if not rank:
            server.engine.abort.assert_awaited_once_with(["first", "second", "parent"], internal=True)
            server.engine.pause_generation.assert_not_awaited()
            server.engine.reset_prefix_cache.assert_not_awaited()
        server._ensure_cache_bridge.assert_not_awaited()
        server._cache_lifecycle.notifier.assert_not_awaited()
        assert server._cache_lifecycle.state.to_dict() == before
        assert (server._submission_paused, server._rejecting) == (paused, paused)
        assert server._resume_event.is_set() is not paused

    asyncio.run(run())


def test_clear_ack_precedes_reset_and_reopens_admission(server):
    async def run():
        seen = []

        async def notify(sink, state):
            if state["serving"] is False:
                server.engine.reset_prefix_cache.assert_not_awaited()
            seen.append(state)

        async def reset(**kwargs):
            server.engine.wait_for_requests_to_drain.assert_awaited_once_with()
            return True

        server.engine.reset_prefix_cache.side_effect = reset
        server._cache_lifecycle.notifier = notify
        server._cache_lifecycle.bind({"token": "sink"})
        await server.clear_kv_cache()
        assert seen[0]["index_state"] == "awaiting_clear"
        server.engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=True)
        assert not server._submission_paused and server._cache_lifecycle.state.serving
        assert server._resume_event.is_set()
        assert server._cache_lifecycle.state.index_state == "awaiting_clear"

    asyncio.run(run())


def test_failed_reset_keeps_gate_closed_and_cannot_resume(server):
    async def run():
        server.engine.reset_prefix_cache.return_value = False
        with pytest.raises(RuntimeError, match="confirm"):
            await server.clear_kv_cache()
        assert server._submission_paused and not server._cache_lifecycle.state.serving
        with pytest.raises(RuntimeError, match="repaired"):
            await server.open_submission_gate()

    asyncio.run(run())


def test_sleep_wake_restores_unpaused_server_but_abort_requires_explicit_resume(server):
    async def run():
        await server.sleep()
        assert server._submission_paused and not server._cache_lifecycle.state.serving
        await server.wake_up()
        assert not server._submission_paused and server._cache_lifecycle.state.expected_clears == 2
        await server.abort_all_requests()
        await server.sleep()
        await server.wake_up()
        assert server._submission_paused and not server._cache_lifecycle.state.serving
        await server.open_submission_gate()
        assert server._cache_lifecycle.state.serving

    asyncio.run(run())


def test_partial_wake_and_noop_paths_do_not_open_routing(server):
    async def run():
        server.rollout_mode = RolloutMode.HYBRID
        await server.sleep()
        server.engine.is_sleeping.return_value = True
        await server.wake_up(tags=["weights"])
        assert not server._cache_lifecycle.state.serving
        server.rollout_mode = RolloutMode.STANDALONE
        before = server._cache_lifecycle.state
        await server.sleep()
        await server.wake_up()
        assert server._cache_lifecycle.state == before

    asyncio.run(run())


def test_pause_keep_cache_and_hybrid_protocol_count_each_real_reset_once(server):
    async def run():
        await server.abort_all_requests(reset_prefix_cache=False)
        assert server._cache_lifecycle.state.epoch == 0
        await server.open_submission_gate()
        server.rollout_mode = RolloutMode.HYBRID
        await server.abort_all_requests()
        await server.sleep()
        server.engine.is_sleeping.return_value = True
        await server.wake_up(tags=["weights"])
        await server.clear_kv_cache()
        server.engine.is_sleeping.return_value = False
        await server.resume_kv_cache()
        assert server._cache_lifecycle.state.expected_clears == 5
        assert not server._cache_lifecycle.state.serving
        await server.open_submission_gate()
        assert server._cache_lifecycle.state.serving

    asyncio.run(run())
