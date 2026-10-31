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

"""Cache lifecycle transitions and Clear accounting."""

import asyncio

import msgpack
import pytest

from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def payload(*events):
    return msgpack.packb([0, list(events), 0], use_bin_type=True)


CLEAR = {"type": "AllBlocksCleared"}
STORE = {"type": "BlockStored", "token_ids": [1, 2]}


def test_reset_invalidates_before_ack_and_waits_for_the_last_clear():
    async def run():
        seen = []

        async def notify(sink, state):
            seen.append(dict(state))

        life = CacheLifecycle(notifier=notify)
        life.bind({"token": "sink"})
        await life.begin("clear", 1)
        assert seen[-1]["serving"] is False and seen[-1]["index_state"] == "awaiting_clear"
        await life.finish(True)
        assert await life.accept(payload(STORE)) is None
        await life.begin("clear-again", 1)
        await life.finish(True)
        assert await life.accept(payload(CLEAR, STORE)) is None
        assert life.state.observed_clears == 1
        accepted = await life.accept(payload(STORE, CLEAR, STORE))
        assert msgpack.unpackb(accepted, raw=False)[1] == [CLEAR, STORE]
        assert life.state.index_state == "ready" and life.state.epoch == 2

    asyncio.run(run())


def test_pause_with_cache_preserves_epoch_and_index():
    async def run():
        life = CacheLifecycle()
        await life.begin("pause", 0)
        assert life.state.epoch == 0 and life.state.index_state == "ready"
        assert await life.accept(payload(STORE)) is not None
        await life.finish(True)
        assert life.state.serving

    asyncio.run(run())


def test_notification_failure_cannot_reopen_routing():
    async def run():
        async def fail(sink, state):
            raise TimeoutError("missing router ack")

        life = CacheLifecycle(notifier=fail)
        life.bind({"token": "sink"})
        with pytest.raises(TimeoutError):
            await life.begin("reset", 1)
        assert not life.state.serving and life.state.index_state == "unknown"
        with pytest.raises(RuntimeError, match="repaired"):
            await life.finish(True)

    asyncio.run(run())


def test_gap_blocks_accounting_until_a_fresh_engine_instance():
    async def run():
        life = CacheLifecycle()
        old = life.state.instance_id
        await life.begin("reset", 1)
        await life.stream_gap("lost_clear")
        assert await life.accept(payload(CLEAR, STORE)) is None
        await life.new_instance()
        assert old != life.state.instance_id
        assert await life.accept(payload(STORE)) is not None

    asyncio.run(run())


def test_unmanaged_clear_disables_estimates_instead_of_guessing():
    async def run():
        life = CacheLifecycle()
        assert await life.accept(payload(CLEAR, STORE)) is None
        assert life.state.reason == "unmanaged_clear"

    asyncio.run(run())


def test_clear_batch_with_mapping_encoding_and_all_queued_resets():
    async def run():
        life = CacheLifecycle()
        await life.begin("hybrid", 5)
        batch = {"ts": 1, "events": [CLEAR] * 5 + [STORE], "data_parallel_rank": 0}
        result = msgpack.unpackb(await life.accept(msgpack.packb(batch)), raw=False)
        assert result["events"] == [CLEAR, STORE]
        assert life.state.observed_clears == 5

    asyncio.run(run())


def test_unsubscribe_ack_waits_for_inflight_notification():
    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        calls = []

        async def notify(sink, state):
            calls.append(state)
            started.set()
            await finish.wait()

        life = CacheLifecycle(notifier=notify)
        life.bind({"token": "sink"})
        publishing = asyncio.create_task(life.begin("reset", 1))
        await started.wait()
        removing = asyncio.create_task(life.unbind("sink"))
        await asyncio.sleep(0)
        assert not removing.done()
        finish.set()
        await asyncio.gather(publishing, removing)
        await life.finish(True)
        assert len(calls) == 1

    asyncio.run(run())
