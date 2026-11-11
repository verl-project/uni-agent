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

"""Installed vLLM reset methods and native Clear accounting, without a GPU.

The utility transport, idle executor, renderer and external connector are
substitutes. AsyncLLM, AsyncMPClient, EngineCore, Scheduler and the cache manager
execute their installed reset implementations. This does not exercise a live
Mooncake Store.
"""

import asyncio
from concurrent.futures import Future
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("vllm")

from uni_agent.agent_aware_router.collectors.transport.base import TransportMessage
from uni_agent.agent_aware_router.server.cache_lifecycle import CacheLifecycle
from uni_agent.agent_aware_router.server.http_server import KVMetadataHttpServer
from uni_agent.agent_aware_router.server.kv_event_bridge import KVEventBridge
from uni_agent.vllm_plugins.lifecycle import (
    CHECKED_SLEEP_METHOD,
    RESET_BOUNDARY_METHOD,
    _checked_sleep,
    _reset_cache_boundary,
)
from verl.workers.rollout.replica import RolloutMode

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.fixture
def native_reset(monkeypatch):
    import msgspec
    import torch
    from vllm import SamplingParams
    from vllm.distributed.kv_events import AllBlocksCleared, KVEventBatch
    from vllm.utils.hashing import sha256
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.core import EngineCore, EngineCoreProc
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", None, raising=False)
    monkeypatch.setattr(kv_cache_utils, "_NONE_HASH_SEED", None, raising=False)
    kv_cache_utils.init_none_hash(sha256)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)])
    manager = KVCacheManager(
        config, max_model_len=32, scheduler_block_size=4, hash_block_size=4, enable_kv_cache_events=True
    )
    prompt = Request(
        "native-reset",
        list(range(9)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=kv_cache_utils.get_request_block_hasher(4, sha256),
    )
    blocks = manager.block_pool.get_new_blocks(2)
    manager.block_pool.cache_full_blocks(prompt, blocks, 0, 2, 4, 0)
    assert manager.take_events()
    scheduler = SimpleNamespace(
        kv_cache_manager=manager,
        connector=None,
        log_stats=False,
        running=[],
        prev_step_scheduled_req_ids=set(),
        finish_requests=lambda *args: [],
        set_pause_state=lambda state: None,
    )
    scheduler.reset_prefix_cache = MethodType(Scheduler.reset_prefix_cache, scheduler)
    scheduler.reset_connector_cache = MethodType(Scheduler.reset_connector_cache, scheduler)
    sleeps = []
    core = SimpleNamespace(
        scheduler=scheduler,
        model_executor=SimpleNamespace(collective_rpc=lambda *args: None, sleep=lambda level: sleeps.append(level)),
        reset_mm_cache=lambda: None,
        reset_encoder_cache=lambda: None,
        _pause_complete=lambda: True,
        _send_abort_outputs=lambda requests: None,
        _idle_state_callbacks=[],
    )
    core.reset_prefix_cache = MethodType(EngineCore.reset_prefix_cache, core)
    for name in ("_reset_caches", "_finish_pause", "pause_scheduler", "sleep", "_notify_idle_state_callbacks"):
        if hasattr(EngineCoreProc, name):
            setattr(core, name, MethodType(getattr(EngineCoreProc, name), core))
    setattr(core, CHECKED_SLEEP_METHOD, MethodType(_checked_sleep, core))
    setattr(core, RESET_BOUNDARY_METHOD, MethodType(_reset_cache_boundary, core))
    calls = []

    async def utility(method, *args):
        calls.append((method, args))
        result = getattr(core, method)(*args)
        return await asyncio.wrap_future(result) if isinstance(result, Future) else result

    client = SimpleNamespace(call_utility_async=utility)
    client.reset_prefix_cache_async = MethodType(AsyncMPClient.reset_prefix_cache_async, client)
    client.pause_scheduler_async = MethodType(AsyncMPClient.pause_scheduler_async, client)
    engine = SimpleNamespace(engine_core=client)
    engine.reset_prefix_cache = MethodType(AsyncLLM.reset_prefix_cache, engine)

    async def no_op():
        pass

    engine.wait_for_requests_to_drain = engine.reset_mm_cache = engine.reset_encoder_cache = no_op
    engine.renderer = SimpleNamespace(clear_mm_cache_async=no_op)
    engine.logger_manager = None
    engine.output_processor = SimpleNamespace(request_states={})
    engine.pause_generation = MethodType(AsyncLLM.pause_generation, engine)
    engine.is_sleeping = lambda: asyncio.sleep(0, result=False)
    wakes = []

    async def wake_up(tags=None):
        wakes.append(tags)

    engine.wake_up = wake_up
    server = object.__new__(KVMetadataHttpServer)
    server.node_rank = 0
    server.config = SimpleNamespace(free_cache_engine=True)
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.rollout_mode = RolloutMode.COLOCATED
    server.engine = engine
    server._submission_paused = server._rejecting = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    server._cache_lifecycle = CacheLifecycle(failure_callback=server._close_cache_admission)

    async def ensure_bridge():
        return server._cache_lifecycle

    server._ensure_cache_bridge = ensure_bridge

    def clear_payload():
        events = manager.take_events()
        assert len(events) == 1 and isinstance(events[0], AllBlocksCleared)
        return msgspec.msgpack.encode(KVEventBatch(ts=1.0, events=events, data_parallel_rank=0))

    return SimpleNamespace(
        server=server,
        scheduler=scheduler,
        manager=manager,
        prompt=prompt,
        blocks=blocks,
        calls=calls,
        clear_payload=clear_payload,
        core=core,
        sleeps=sleeps,
        wakes=wakes,
    )


def test_queued_native_clear_invalidates_cache_after_an_earlier_recovery(native_reset):
    async def run():
        server = native_reset.server
        control = server._cache_lifecycle
        control._change(index_state="unknown")
        queried = asyncio.Event()

        async def ensure():
            queried.set()
            return control

        server._ensure_cache_bridge = ensure
        observed = []
        native_reset_method = server.engine.reset_prefix_cache

        async def reset(**kwargs):
            observed.append(control.state.index_state)
            return await native_reset_method(**kwargs)

        server.engine.reset_prefix_cache = reset
        native_reset.manager.block_pool.free_blocks(native_reset.blocks)
        await control.operation_lock.acquire()
        clearing = asyncio.create_task(server.clear_kv_cache())
        await asyncio.wait_for(queried.wait(), 1)
        # The operation holding the lock recovers while this clear waits.
        control._change(index_state="ready")
        control.operation_lock.release()
        await asyncio.wait_for(clearing, 1)
        assert observed == ["awaiting_clear"]
        assert await control.accept(native_reset.clear_payload()) is not None
        assert control.state.index_state == "ready"
        assert control.state.expected_clears == control.state.observed_clears == 1

    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["no-connector", "connector", "busy-blocks", "connector-false", "connector-error"])
def test_native_reset_drives_admission_and_clear_accounting(native_reset, scenario):
    async def run():
        native = native_reset
        server, scheduler = native.server, native.scheduler
        connector_calls = []

        def reset_connector():
            connector_calls.append(True)
            if scenario == "connector-error":
                raise RuntimeError("external reset failed")
            return scenario != "connector-false"

        if scenario != "no-connector":
            scheduler.connector = SimpleNamespace(reset_cache=reset_connector)
        assert native.manager.get_computed_blocks(native.prompt)[1] == 8
        if scenario != "busy-blocks":
            native.manager.block_pool.free_blocks(native.blocks)
        success = scenario in {"no-connector", "connector"}
        if success:
            await server.clear_kv_cache()
        else:
            with pytest.raises(RuntimeError, match="external reset failed|confirm"):
                await server.clear_kv_cache()
        # The installed async engine and client must preserve connector=True.
        assert native.calls == [("reset_prefix_cache", (False, True))]
        assert len(connector_calls) == int(scenario != "no-connector")
        control = server._cache_lifecycle
        assert control.state.expected_clears == 1 and control.state.observed_clears == 0
        assert server._submission_paused is not success
        assert server._resume_event.is_set() is success
        assert control.state.serving is success
        if scenario == "busy-blocks":
            assert native.manager.take_events() == []
            assert native.manager.get_computed_blocks(native.prompt)[1] == 8
        else:
            assert native.manager.get_computed_blocks(native.prompt)[1] == 0
            accepted = await control.accept(native.clear_payload())
            assert (accepted is not None) is success
            assert control.state.observed_clears == 1
        assert control.state.index_state == ("ready" if success else "unknown")
        if success:
            # An already-empty native cache still emits exactly one Clear.
            await server.clear_kv_cache()
            assert await control.accept(native.clear_payload()) is not None
            assert control.state.expected_clears == control.state.observed_clears == 2
        else:
            with pytest.raises(RuntimeError, match="repaired"):
                await server.open_submission_gate()
            assert server._submission_paused and not control.state.serving
            assert not server._resume_event.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("connector_result", [True, False])
def test_native_pause_uses_one_checked_reset_and_propagates_connector_failure(native_reset, connector_result):
    async def run():
        native = native_reset
        native.manager.block_pool.free_blocks(native.blocks)
        connector_calls = []

        def reset_connector():
            connector_calls.append(True)
            return connector_result

        native.scheduler.connector = SimpleNamespace(reset_cache=reset_connector)
        server = native.server
        if connector_result:
            assert await server.abort_all_requests() == {"aborted_count": 0, "request_ids": []}
        else:
            with pytest.raises(RuntimeError, match="confirm"):
                await server.abort_all_requests()
        assert native.calls == [("pause_scheduler", ("abort", False)), ("reset_prefix_cache", (False, True))]
        assert len(connector_calls) == 1
        control = server._cache_lifecycle
        assert control.state.expected_clears == 1
        accepted = await control.accept(native.clear_payload())
        assert (accepted is not None) is connector_result
        assert control.failed is not connector_result
        if connector_result:
            await server.open_submission_gate()
            assert control.state.serving and not server._submission_paused
        else:
            with pytest.raises(RuntimeError, match="repaired"):
                await server.open_submission_gate()
            assert not control.state.serving and server._submission_paused

    asyncio.run(run())


@pytest.mark.parametrize("connector_result", [True, False])
@pytest.mark.parametrize("operation,level", [("sleep", 1), ("release_kv_cache", 2)])
def test_native_sleep_checks_its_single_implicit_reset(native_reset, connector_result, operation, level):
    async def run():
        native = native_reset
        native.manager.block_pool.free_blocks(native.blocks)
        native.scheduler.connector = SimpleNamespace(reset_cache=lambda: connector_result)
        if operation == "release_kv_cache":
            native.server.rollout_mode = RolloutMode.HYBRID
        original = native.core.reset_prefix_cache
        if connector_result:
            await getattr(native.server, operation)()
            assert native.sleeps == [level]
            assert native.wakes == ([["weights"]] if operation == "release_kv_cache" else [])
        else:
            with pytest.raises(RuntimeError, match="confirm"):
                await getattr(native.server, operation)()
            assert native.sleeps == []
            assert native.wakes == []
        assert native.calls == [(CHECKED_SLEEP_METHOD, (level, "abort"))]
        assert native.core.reset_prefix_cache is original
        control = native.server._cache_lifecycle
        assert control.state.expected_clears == 1
        assert (await control.accept(native.clear_payload()) is not None) is connector_result
        assert control.failed is not connector_result
        assert native.server._submission_paused

    asyncio.run(run())


def test_native_recovery_resets_lost_clear_accounting_and_preserves_standard_wire(native_reset):
    async def run():
        import msgspec
        from vllm.distributed.kv_events import AllBlocksCleared, KVEventBatch

        native = native_reset
        native.manager.block_pool.free_blocks(native.blocks)
        control = native.server._cache_lifecycle
        instance = control.state.instance_id
        await control.begin("missing_resets", 3)
        await control.finish(True)
        await control.stream_gap("lost_clear")
        # A queued Clear from an older operation must not satisfy recovery.
        assert native.manager.reset_prefix_cache()
        bridge = KVEventBridge(control, "127.0.0.1")
        bridge._native_sequence = 4
        bridge._consumer = asyncio.create_task(bridge._consume())
        bridge._tasks.append(bridge._consumer)
        published = []

        def publish(batch):
            encoded = msgspec.msgpack.encode(batch)
            # Existing vLLM consumers can decode the unchanged native events
            # while safely ignoring our trailing batch extension.
            decoded = msgspec.msgpack.decode(encoded, type=KVEventBatch)
            assert len(decoded.events) == 1 and isinstance(decoded.events[0], AllBlocksCleared)
            published.append(encoded)
            bridge._enqueue(TransportMessage(encoded, 20), "native")

        native.scheduler.kv_event_publisher = SimpleNamespace(
            get_publisher_config=lambda: SimpleNamespace(publisher="zmq"), publish=publish
        )
        try:
            await native.server.clear_kv_cache()
            assert len(published) == 1
            assert control.state.instance_id == instance
            assert control.state.expected_clears == control.state.observed_clears == 1
            assert control.state.index_state == "ready" and control.state.serving
            assert not native.server._submission_paused
            assert native.calls[0][0] == RESET_BOUNDARY_METHOD
        finally:
            await bridge.close()

    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["success", "false", "error"])
def test_native_deferred_sleep_reset_completes_rpc_and_restores_guard(native_reset, scenario):
    async def run():
        native = native_reset
        native.manager.block_pool.free_blocks(native.blocks)

        def reset_connector():
            if scenario == "error":
                raise RuntimeError("connector reset failed")
            return scenario == "success"

        native.scheduler.connector = SimpleNamespace(reset_cache=reset_connector)
        native.core._pause_complete = lambda: False
        original = native.core.reset_prefix_cache
        task = asyncio.create_task(native.server.sleep())
        await asyncio.sleep(0)
        assert native.core._idle_state_callbacks and not task.done()
        # This is the installed EngineCoreProc callback drain, not an artificial
        # Future failure: reset errors must not escape into the native loop.
        native.core._notify_idle_state_callbacks()
        if scenario == "success":
            await asyncio.wait_for(task, 1)
            assert native.sleeps == [1]
        else:
            with pytest.raises(RuntimeError, match="confirm|connector reset failed"):
                await asyncio.wait_for(task, 1)
            assert native.sleeps == []
            assert native.server._cache_lifecycle.failed
        assert native.core.reset_prefix_cache is original
        assert not native.core._uni_agent_checked_sleep_active
        assert not native.core._idle_state_callbacks
        assert native.server._submission_paused

    asyncio.run(run())


def test_cancelled_client_cannot_recover_or_reopen_before_native_sleep_finishes(native_reset):
    async def run():
        import msgspec
        from vllm.v1.engine.core import EngineCoreProc, UtilityOutput
        from vllm.v1.engine.core_client import AsyncMPClient, _process_utility_output

        native = native_reset
        native.manager.block_pool.free_blocks(native.blocks)
        native.core._pause_complete = lambda: False
        server, core = native.server, native.core
        control = server._cache_lifecycle
        producer_futures = []
        published = []

        class Client:
            client_index = 0
            encoder = SimpleNamespace(encode=lambda value: (value,))

            def __init__(self):
                self.utility_results = {}

            def _ensure_output_queue_task(self):
                pass

            async def _send_input_message(self, message, engine, *transport_args):
                _, call_id, method, args = message[1]

                def invoke():
                    result = getattr(core, method)(*args)
                    if isinstance(result, Future):
                        producer_futures.append(result)
                    return result

                EngineCoreProc._invoke_utility_method(
                    method,
                    invoke,
                    UtilityOutput(call_id),
                    lambda output: _process_utility_output(output, self.utility_results),
                )

            async def call_utility_async(self, method, *args):
                return await AsyncMPClient._call_utility_async(self, method, *args, engine=b"native")

        server.engine.engine_core = Client()
        server.engine.is_sleeping = lambda: asyncio.sleep(0, result=bool(native.sleeps))

        def publish(batch):
            published.append(asyncio.create_task(control.accept(msgspec.msgpack.encode(batch))))

        native.scheduler.kv_event_publisher = SimpleNamespace(
            get_publisher_config=lambda: SimpleNamespace(publisher="zmq"), publish=publish
        )
        sleeping = asyncio.create_task(server.sleep())
        await asyncio.sleep(0)
        assert core._idle_state_callbacks and not producer_futures[0].done()
        sleeping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sleeping
        assert not producer_futures[0].done() and core._uni_agent_checked_sleep_active
        assert control.failed and not native.sleeps
        # The installed utility client wraps producer errors as Exception.
        with pytest.raises(Exception, match="native sleep.*in progress"):
            await server.recover_cache_lifecycle()
        with pytest.raises(RuntimeError, match="repaired"):
            await server.open_submission_gate()
        assert control.failed and not server._resume_event.is_set()
        # The native operation survives cancellation of the actual client
        # utility Future. Complete its real callback before allowing recovery.
        core._notify_idle_state_callbacks()
        assert producer_futures[0].done() and native.sleeps == [1]
        assert not core._uni_agent_checked_sleep_active
        await server.recover_cache_lifecycle()
        assert control.state.index_state == "ready" and not control.failed
        with pytest.raises(RuntimeError, match="allocations are asleep"):
            await server.open_submission_gate()
        assert not control.state.serving and server._submission_paused
        await asyncio.gather(*published)

    asyncio.run(run())
