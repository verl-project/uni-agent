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

"""Paused replicas are excluded before every strategy path."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import msgpack
import pytest
import zmq
import zmq.asyncio

from tests.uni_agent.agent_aware_router.balancer._helpers import _make_balancer, _router_config
from uni_agent.agent_aware_router.collectors.transport.lifecycle import LifecycleControlReceiver, notify_lifecycle
from uni_agent.agent_aware_router.types import MetricKey
from uni_agent.agent_aware_router.types.cache_lifecycle import BRIDGE_SCHEMA_KEY, CacheLifecycleState

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def test_collector_refresh_keeps_control_ack_and_request_callbacks_available(monkeypatch):
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", lambda ref, timeout=None: ref)
    monkeypatch.setattr(KVCAwareBalancer, "_discover_kv_event_endpoints", lambda self, addresses: {})
    router = KVCAwareBalancer({"a": "server-a"}, _router_config())
    receiver = LifecycleControlReceiver(router.update_cache_lifecycle, "127.0.0.1")
    entered, resume = Event(), Event()
    network = next(c for c in router._provider._collectors if c._transport.is_async)
    stop = network.stop

    def slow_stop():
        entered.set()
        assert resume.wait(3)
        stop()

    monkeypatch.setattr(network, "stop", slow_stop)
    added = SimpleNamespace(get_server_address=SimpleNamespace(remote=lambda: ("127.0.0.1", 1)))
    with ThreadPoolExecutor(max_workers=2) as executor:
        updating = None
        try:
            router.acquire_server("before", [1, 2, 3])
            updating = executor.submit(router.add_servers, {"b": added})
            assert entered.wait(3)

            async def notify():
                return await notify_lifecycle(receiver.descriptor("a"), CacheLifecycleState("engine-a").to_dict(), 0.5)

            assert asyncio.run(notify())["applied"]
            assert executor.submit(router.acquire_server, "during", [4, 5]).result(0.5)[0] == "a"
            assert router._store.get_sticky_binding("during") == "a"
            assert router._store.get_metrics("a")[MetricKey.INFLIGHT_COUNT] == 2
            assert router._store.get_metrics("a")[MetricKey.INFLIGHT_TOKENS] == 5
            router.release_server("a", "before")
            router.release_server("a", "during")
            assert router._store.get_metrics("a")[MetricKey.INFLIGHT_COUNT] == 0
            assert router._store.get_metrics("a")[MetricKey.INFLIGHT_TOKENS] == 0
        finally:
            resume.set()
            if updating is not None:
                updating.result(3)
            receiver.close()
            router.close()


def test_failed_endpoint_discovery_keeps_existing_provider_and_added_server_pending(monkeypatch):
    router = _make_balancer({"a": "server-a"})
    provider = router._provider

    def unavailable(ref, timeout):
        assert timeout == router._DISCOVERY_TIMEOUT_SECONDS
        raise TimeoutError("endpoint discovery timed out")

    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", unavailable)
    added = SimpleNamespace(get_server_address=SimpleNamespace(remote=lambda: "address-future"))
    try:
        with pytest.raises(TimeoutError, match="endpoint discovery"):
            router.add_servers({"b": added})
        assert router._provider is provider and not provider.stopped
        assert router._pending_servers == {"b"}
        assert router.acquire_server("existing")[0] == "a"
        router.release_server("a", "existing")
    finally:
        router.close()


@pytest.mark.parametrize(
    "protocol,failure",
    [
        ("group", None),
        ("bridge", None),
        ("legacy", None),
        ("legacy", "construct"),
        ("legacy", "start"),
        ("legacy", "cleanup"),
        ("bridge", "cleanup"),
    ],
)
def test_add_server_resumes_truncated_replay_and_updates_existing_cache(
    monkeypatch, layout, stored, removed, protocol, failure
):
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
    from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", lambda ref, timeout=None: ref)
    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.util.get_node_ip_address", lambda: "127.0.0.1")

    async def run():
        ctx = zmq.asyncio.Context()
        sockets, endpoints = {}, {}
        requests = {node: [] for node in ("a", "b")}
        states = {node: CacheLifecycleState("engine-" + node) for node in requests}
        for node in requests:
            pub, replay = ctx.socket(zmq.PUB), ctx.socket(zmq.ROUTER)
            pub_port = pub.bind_to_random_port("tcp://127.0.0.1")
            replay_port = replay.bind_to_random_port("tcp://127.0.0.1")
            sockets[node] = pub, replay
            endpoints[node] = [f"127.0.0.1:{pub_port}", f"127.0.0.1:{replay_port}", "zmq", "kv"]

        raw = layout((("full_attention", 4),), unit=None)
        if protocol == "legacy":
            raw = {"status": "unsupported"}

        def handle(node):
            server = SimpleNamespace(
                get_kv_cache_metadata=SimpleNamespace(remote=lambda: raw),
                get_server_address=SimpleNamespace(remote=lambda: ("127.0.0.1", 1)),
            )
            if protocol == "bridge":
                source = {
                    "0": {
                        "publisher": "zmq",
                        "topic": "kv",
                        "endpoint": "tcp://" + endpoints[node][0],
                        "replay_endpoint": "tcp://" + endpoints[node][1],
                    }
                }
                server.register_cache_lifecycle_sink = SimpleNamespace(
                    remote=lambda sink: {"state": states[node].to_dict(), "sources": source}
                )
                server.unregister_cache_lifecycle_sink = SimpleNamespace(remote=lambda token, generation=None: None)
            return server

        async def serve(node, expected, batches):
            replay = sockets[node][1]
            identity, delimiter, start = await replay.recv_multipart()
            start = int.from_bytes(start, "big")
            requests[node].append(start)
            assert delimiter == b"" and start == expected
            for sequence, events in batches:
                if sequence < start:
                    continue
                if protocol == "legacy":
                    events = [
                        {key: value for key, value in event.items() if key not in {"group_idx", "kv_cache_spec_kind"}}
                        for event in events
                    ]
                payload = msgpack.packb([0, events], use_bin_type=True)
                if protocol == "bridge":
                    payload = msgpack.packb(
                        {
                            BRIDGE_SCHEMA_KEY: 1,
                            "state": states[node].to_dict(),
                            "sequence": sequence,
                            "payload": payload,
                        },
                        use_bin_type=True,
                    )
                await replay.send_multipart([identity, b"", b"kv", sequence.to_bytes(8, "big"), payload])
            await replay.send_multipart([identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""])

        async def processed(store, sequences):
            async def wait():
                while store._kv._event_sequences != sequences:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait(), 3)

        monkeypatch.setattr(
            KVCAwareBalancer,
            "_discover_kv_event_endpoints",
            lambda self, addresses: {node: endpoints[node] for node in addresses},
        )
        first = asyncio.create_task(
            serve("a", 0, [(0, [stored(tokens=list(range(4)), hashes=["h0"])]), (1, []), (2, [])])
        )
        router = None
        release_cleanup = Event()
        try:
            router = await asyncio.to_thread(KVCAwareBalancer, {"a": handle("a")}, _router_config())
            store = router._store
            await first
            await processed(store, {"a": 2})
            assert store.per_replica_block_counts() == {"a": 1}
            if failure:
                from uni_agent.agent_aware_router.collectors import provider

                factory = provider.get_collector
                originals = list(router._provider._collectors)
                candidates = []

                def failing_factory(name, *args, **kwargs):
                    if failure == "construct" and name == "vllm_zmq":
                        raise ValueError("replacement construction failed")
                    collector = factory(name, *args, **kwargs)
                    candidates.append(collector)
                    if (failure == "start" and name == "vllm_metrics") or (failure == "cleanup" and name == "vllm_zmq"):
                        start = collector.start

                        def fail_start():
                            start()
                            raise ValueError("replacement start failed")

                        collector.start = fail_start
                    if failure == "cleanup" and name == "vllm_zmq":
                        stop = collector.stop

                        def blocked_cleanup(*args, **kwargs):
                            if not release_cleanup.is_set():
                                raise TimeoutError("replacement cleanup timed out")
                            stop(*args, **kwargs)

                        collector.stop = blocked_cleanup
                    return collector

                monkeypatch.setattr(provider, "get_collector", failing_factory)
                recovered = asyncio.create_task(
                    serve(
                        "a",
                        3,
                        [(3, [stored(tokens=list(range(4, 8)), hashes=["h1"])]), (4, [removed(0, ["h0"])])],
                    )
                )
                waiting_b = asyncio.create_task(serve("b", 0, [(0, [])])) if failure == "cleanup" else None
                expected_error = RuntimeError if failure == "cleanup" else ValueError
                with pytest.raises(expected_error):
                    await asyncio.to_thread(router.add_servers, {"b": handle("b")})
                await asyncio.wait_for(recovered, 3)
                if waiting_b is not None:
                    await asyncio.wait_for(waiting_b, 3)
                await processed(store, {"a": 4, "b": 0} if failure == "cleanup" else {"a": 4})
                assert router._provider._collectors == originals
                assert router._pending_servers == {"b"}
                if failure == "cleanup":
                    metrics = originals[router._provider._collection_names.index("vllm_metrics")]
                    worker = metrics._loop_thread
                    assert worker.is_alive() and not metrics._future.done()
                    assert store.per_replica_block_counts().get("a", 0) == 0
                    assert "a" in store._kv._stream_errors
                    with pytest.raises(RuntimeError):
                        await asyncio.to_thread(router.add_servers, {"b": handle("b")})
                    assert metrics._loop_thread is worker and worker.is_alive()
                    assert not metrics._future.done()
                    assert requests == {"a": [0, 3], "b": [0]}
                    release_cleanup.set()
                else:
                    assert store.per_replica_block_counts() == {"a": 1}
                    assert all(c._loop.is_closed() for c in candidates)
                monkeypatch.setattr(provider, "get_collector", factory)
            # The beginning of A's replay buffer has expired. B starts at zero.
            existing = asyncio.create_task(
                serve(
                    "a",
                    0 if failure == "cleanup" else (5 if failure else 3),
                    [(5, [])]
                    if failure in {"construct", "start"}
                    else [
                        (1, []),
                        (2, []),
                        (3, [stored(tokens=list(range(4, 8)), hashes=["h1"])]),
                        (4, [removed(0, ["h0"])]),
                    ],
                )
            )
            added = asyncio.create_task(serve("b", 0, [(0, [stored(tokens=list(range(8, 12)), hashes=["h2"])])]))
            await asyncio.to_thread(router.add_servers, {"b": handle("b")})
            await asyncio.wait_for(asyncio.gather(existing, added), 3)
            await processed(store, {"a": 5 if failure in {"construct", "start"} else 4, "b": 0})
            assert requests == {
                "a": [0, 3, 0] if failure == "cleanup" else ([0, 3, 5] if failure else [0, 3]),
                "b": [0, 0] if failure == "cleanup" else [0],
            }
            assert store.per_replica_block_counts() == {"a": 1, "b": 1}
            assert not store._kv._stream_errors
            for node, ids, hit in [("a", list(range(5)), 0), ("a", list(range(4, 9)), 4), ("b", list(range(8, 13)), 4)]:
                hashes = resolve_prefix_hashes(ids, None, store)
                assert store.estimate_prefixes([node], hashes, len(ids))[node].estimated_hit_tokens == hit
        finally:
            release_cleanup.set()
            if router is not None:
                await asyncio.to_thread(router.close)
            for pub, replay in sockets.values():
                pub.close(0)
                replay.close(0)
            ctx.term()

    asyncio.run(run())


def test_sticky_and_random_fallback_cannot_select_paused_replica(monkeypatch):
    router = _make_balancer()
    router.acquire_server("session")
    bound = router._store.get_sticky_binding("session")
    state = CacheLifecycleState("engine", serving=False, index_state="awaiting_clear")
    router.update_cache_lifecycle(bound, state.to_dict())
    chosen, _ = router.acquire_server("session")
    assert chosen != bound

    def fail(*args, **kwargs):
        raise RuntimeError("scoring failure")

    monkeypatch.setattr(router._strategy, "score", fail)
    assert router.acquire_server("new-session")[0] != bound
    other = "s1" if bound == "s0" else "s0"
    router.update_cache_lifecycle(other, state.to_dict())
    with pytest.raises(RuntimeError, match="no available replicas"):
        router.acquire_server("none")


def test_late_control_update_does_not_reopen_and_does_not_erase_inflight():
    router = _make_balancer({"only": "server"})
    chosen, _ = router.acquire_server("session")
    state = CacheLifecycleState("engine", revision=2, serving=False, index_state="awaiting_clear")
    router.update_cache_lifecycle(chosen, state.to_dict())
    assert router.get_total_inflight() == 1
    router.update_cache_lifecycle(chosen, replace(state, revision=1, serving=True).to_dict())
    with pytest.raises(RuntimeError, match="no available replicas"):
        router.acquire_server("blocked")
    router.release_server(chosen, "session")
    assert router.get_total_inflight() == 0


def test_readded_paused_server_with_equal_revision_stays_unavailable(monkeypatch):
    from tests.uni_agent.agent_aware_router.balancer._helpers import _FakeCollectorManager, _router_config
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    state = CacheLifecycleState("engine", revision=1, serving=False)
    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(
            remote=Mock(return_value={"state": state.to_dict(), "sources": {}})
        ),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=Mock(return_value=None)),
    )
    monkeypatch.setattr("ray.get", lambda value, **kwargs: value)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    router = KVCAwareBalancer({"s0": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        router.remove_servers(["s0"])
        router.add_servers({"s0": handle})
        with pytest.raises(RuntimeError, match="no available replicas"):
            router.acquire_server("paused")
    finally:
        router.close()


@pytest.mark.parametrize("replacement_has_lifecycle", [False, True])
def test_replacement_discards_previous_engine_state_and_rejects_late_updates(
    monkeypatch, layout, stored, parse, replacement_has_lifecycle
):
    from tests.uni_agent.agent_aware_router.balancer._helpers import _FakeCollectorManager
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
    from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate

    monkeypatch.setattr("ray.get", lambda value, **kwargs: value)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(KVCAwareBalancer, "_discover_kv_event_endpoints", lambda self, addresses: {})
    old = CacheLifecycleState("old-engine")
    unregister = Mock(return_value=None)
    original = SimpleNamespace(
        get_kv_cache_metadata=SimpleNamespace(remote=lambda: layout((("full_attention", 4),), unit=None)),
        register_cache_lifecycle_sink=SimpleNamespace(
            remote=lambda sink: {
                "state": old.to_dict(),
                "sources": {
                    "0": {
                        "publisher": "zmq",
                        "topic": "kv",
                        "endpoint": "tcp://127.0.0.1:1234",
                        "replay_endpoint": None,
                    }
                },
            }
        ),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=unregister),
    )
    replacement = SimpleNamespace(get_server_address=SimpleNamespace(remote=lambda: ("127.0.0.1", 1)))
    new = CacheLifecycleState("new-engine")
    if replacement_has_lifecycle:

        def register(sink):
            for state in (old, new):
                result = router._receive_cache_lifecycle("s0", state.to_dict())
                assert not result["applied"] and result["reason"] == "lifecycle_handshake_pending"
            return {"state": new.to_dict(), "sources": {}}

        replacement.register_cache_lifecycle_sink = SimpleNamespace(remote=register)
        replacement.unregister_cache_lifecycle_sink = SimpleNamespace(remote=lambda token, generation=None: None)
    router = KVCAwareBalancer({"s0": original}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        event = KVCacheUpdate("s0", parse([stored()], "s0"), lifecycle=old, sequence=10)
        assert router._store.apply_kv_update(event)
        assert router._store.per_replica_block_counts() == {"s0": 4}
        router.update_cache_lifecycle("s0", replace(old, revision=1, serving=False).to_dict())
        router.add_servers({"s0": replacement})
        assert router.acquire_server("replacement")[1] is replacement
        assert router._store.per_replica_block_counts() == {}
        assert not router._provider.kv_event_endpoints
        assert not router._receive_cache_lifecycle("s0", replace(old, revision=100).to_dict())["applied"]
        assert not router._store.apply_kv_update(event)
        current = router._store.cache_lifecycle_state("s0")
        assert current == (new if replacement_has_lifecycle else None)
        assert "s0" not in router._lifecycle_states or router._lifecycle_states["s0"].instance_id == new.instance_id
        unregister.assert_called_once()
    finally:
        router.close()


def test_returning_engine_restarts_cold_and_cannot_borrow_replacement_state(monkeypatch, layout, stored, parse):
    from tests.uni_agent.agent_aware_router.balancer._helpers import _FakeCollectorManager
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
    from uni_agent.agent_aware_router.types.kv_cache import KVCacheUpdate

    monkeypatch.setattr("ray.get", lambda value, **kwargs: value)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")

    def handle(state):
        return SimpleNamespace(
            get_kv_cache_metadata=SimpleNamespace(remote=lambda: layout((("full_attention", 4),), unit=None)),
            register_cache_lifecycle_sink=SimpleNamespace(
                remote=lambda sink: {"state": state.to_dict(), "sources": {}}
            ),
            unregister_cache_lifecycle_sink=SimpleNamespace(remote=lambda token, generation=None: None),
        )

    old, new = CacheLifecycleState("old-engine"), CacheLifecycleState("new-engine", serving=False)
    original, replacement = handle(old), handle(new)
    router = KVCAwareBalancer({"s0": original}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        router.remove_servers(["s0"])
        router.add_servers({"s0": replacement})
        assert router._store.apply_kv_update(KVCacheUpdate("s0", parse([stored()], "s0"), lifecycle=new))
        router.remove_servers(["s0"])
        assert not router._receive_cache_lifecycle("s0", old.to_dict())["applied"]
        router.add_servers({"s0": original})
        assert "s0" not in router._pending_servers
        assert router._lifecycle_states["s0"] == old
        assert router._store.cache_lifecycle_state("s0") == old
        assert router._store.per_replica_block_counts() == {}
        assert router.acquire_server("returned")[1] is original
        assert not router._receive_cache_lifecycle("s0", replace(new, revision=100).to_dict())["applied"]
        assert router._lifecycle_states["s0"] == old
        assert router._store.apply_kv_update(KVCacheUpdate("s0", parse([stored()], "s0"), lifecycle=old, sequence=0))
        assert router._store.per_replica_block_counts() == {"s0": 4}
    finally:
        router.close()


@pytest.mark.parametrize("detach_fails", [False, True])
def test_close_cleans_lifecycle_after_provider_failure_and_can_retry(monkeypatch, detach_fails):
    from tests.uni_agent.agent_aware_router.balancer._helpers import _FakeCollectorManager
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    monkeypatch.setattr("ray.get", lambda value, **kwargs: value)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    unregister = Mock(side_effect=[RuntimeError("detach failed"), None] if detach_fails else [None])
    state = CacheLifecycleState("engine")
    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(remote=lambda sink: {"state": state.to_dict(), "sources": {}}),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=unregister),
    )
    router = KVCAwareBalancer({"s0": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    receiver = router._lifecycle_receiver
    stop = Mock(side_effect=[TimeoutError("collector cleanup failed"), None])
    monkeypatch.setattr(router._provider, "stop", stop)
    try:
        with pytest.raises(TimeoutError, match="collector cleanup failed"):
            router.close()
        assert not receiver._thread.is_alive()
        assert unregister.call_count == 1
        assert router._lifecycle_receiver is (receiver if detach_fails else None)
        router.close()
        assert router._lifecycle_receiver is None
        assert unregister.call_count == (2 if detach_fails else 1)
        assert stop.call_count == 2
    finally:
        receiver.close()


@pytest.mark.parametrize("fail_first_refresh", [False, True])
def test_same_endpoint_replacement_drops_late_legacy_cache_and_restarts_replay(monkeypatch, fail_first_refresh):
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer
    from uni_agent.agent_aware_router.utils.prefix_cache import resolve_prefix_hashes

    monkeypatch.setattr("uni_agent.agent_aware_router.balancer.ray.get", lambda ref, timeout=None: ref)

    async def run():
        context = zmq.asyncio.Context()
        publisher, replay = context.socket(zmq.PUB), context.socket(zmq.ROUTER)
        pub_port = publisher.bind_to_random_port("tcp://127.0.0.1")
        replay_port = replay.bind_to_random_port("tcp://127.0.0.1")
        endpoint = [f"127.0.0.1:{pub_port}", f"127.0.0.1:{replay_port}", "zmq", "kv"]
        monkeypatch.setattr(
            KVCAwareBalancer,
            "_discover_kv_event_endpoints",
            lambda self, addresses: {node: endpoint for node in addresses},
        )

        def handle():
            return SimpleNamespace(
                get_kv_cache_metadata=SimpleNamespace(remote=lambda: {"status": "unsupported"}),
                get_server_address=SimpleNamespace(remote=lambda: ("127.0.0.1", 1)),
            )

        def payload(tokens, block_hash):
            event = ["BlockStored", [block_hash], None, tokens, 4, None, "GPU"]
            return msgpack.packb([0, [event]], use_bin_type=True)

        async def serve(sequence, tokens, block_hash, expected_start=0):
            identity, delimiter, start = await replay.recv_multipart()
            assert delimiter == b"" and int.from_bytes(start, "big") == expected_start
            await replay.send_multipart(
                [identity, b"", b"kv", sequence.to_bytes(8, "big"), payload(tokens, block_hash)]
            )
            await replay.send_multipart([identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""])

        async def processed(sequence):
            async def wait():
                while router._store._kv._event_sequences.get("a") != sequence:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait(), 3)

        first = asyncio.create_task(serve(50, list(range(4)), "old"))
        entered, resume = Event(), Event()
        router, updating, replacement_replay, recovered_replay = None, None, None, None
        try:
            router = await asyncio.to_thread(KVCAwareBalancer, {"a": handle()}, _router_config())
            await asyncio.wait_for(first, 3)
            await processed(50)
            network = router._provider._collectors[router._provider._collection_names.index("vllm_zmq")]
            stop = network.stop

            def blocked_stop():
                entered.set()
                assert resume.wait(5)
                stop()

            monkeypatch.setattr(network, "stop", blocked_stop)
            replacement = handle()
            if fail_first_refresh:
                from uni_agent.agent_aware_router.collectors import provider

                factory = provider.get_collector

                def failing_factory(name, *args, **kwargs):
                    if name == "vllm_zmq":
                        raise ValueError("replacement construction failed")
                    return factory(name, *args, **kwargs)

                monkeypatch.setattr(provider, "get_collector", failing_factory)
                recovered_replay = asyncio.create_task(serve(52, list(range(12, 16)), "rollback", expected_start=52))
            else:
                replacement_replay = asyncio.create_task(serve(0, list(range(8, 12)), "new"))
            updating = asyncio.create_task(asyncio.to_thread(router.add_servers, {"a": replacement}))
            assert await asyncio.to_thread(entered.wait, 3)

            async def publish_late():
                while router._store._kv._event_sequences.get("a") != 51:
                    await publisher.send_multipart([b"kv", (51).to_bytes(8, "big"), payload(list(range(4, 8)), "late")])
                    await asyncio.sleep(0.01)

            # Deliver through the still-running old collector after the first
            # replacement clear, rather than simulating a direct store write.
            await asyncio.wait_for(publish_late(), 3)
            assert (0, "late") in router._store._kv.group_caches["a"].retained_hashes
            resume.set()
            if fail_first_refresh:
                with pytest.raises(ValueError, match="replacement construction failed"):
                    await asyncio.wait_for(updating, 5)
                await asyncio.wait_for(recovered_replay, 3)
                await processed(52)
                assert router._pending_servers == {"a"}
                assert (0, "rollback") in router._store._kv.group_caches["a"].retained_hashes
                monkeypatch.setattr(provider, "get_collector", factory)
                replacement_replay = asyncio.create_task(serve(0, list(range(8, 12)), "new"))
                # The handle is already installed, but its pending refresh must
                # still discard the old collector's restored cursor and index.
                updating = asyncio.create_task(asyncio.to_thread(router.add_servers, {"a": replacement}))
            await asyncio.wait_for(updating, 5)
            await asyncio.wait_for(replacement_replay, 3)
            await processed(0)
            assert router._store.per_replica_block_counts() == {"a": 1}
            for tokens, expected in [(list(range(5)), 0), (list(range(4, 9)), 0), (list(range(8, 13)), 4)]:
                hashes = resolve_prefix_hashes(tokens, None, router._store)
                estimate = router._store.estimate_prefixes(["a"], hashes, len(tokens))["a"]
                assert estimate.estimated_hit_tokens == expected
        finally:
            resume.set()
            if updating is not None:
                await asyncio.wait_for(asyncio.gather(updating, return_exceptions=True), 5)
            if router is not None:
                await asyncio.to_thread(router.close)
            for task in (first, replacement_replay, recovered_replay):
                if task is not None:
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            publisher.close(0)
            replay.close(0)
            context.term()

    asyncio.run(run())


def test_readded_engine_keeps_notification_newer_than_registration_snapshot(monkeypatch):
    from tests.uni_agent.agent_aware_router.balancer._helpers import _FakeCollectorManager
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    monkeypatch.setattr("ray.get", lambda value, **kwargs: value)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    ready = CacheLifecycleState("engine")
    paused = replace(ready, revision=1, serving=False)
    calls = 0
    previous_sink = None

    def register(sink):
        nonlocal calls, previous_sink
        calls += 1
        if calls == 2:
            # bind() already captured ready0. A concurrent lifecycle operation
            # publishes paused1 before the caller receives that old snapshot.
            unrelated = CacheLifecycleState("other-engine", revision=100)
            result = asyncio.run(notify_lifecycle(previous_sink, unrelated.to_dict()))
            assert not result["applied"] and result["reason"] == "lifecycle_registration_expired"
            result = asyncio.run(notify_lifecycle(sink, paused.to_dict()))
            assert not result["applied"] and result["reason"] == "lifecycle_handshake_pending"
        previous_sink = sink
        return {"state": ready.to_dict(), "sources": {}}

    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(remote=register),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=lambda token, generation=None: None),
    )
    router = KVCAwareBalancer({"s0": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        router.remove_servers(["s0"])
        router.add_servers({"s0": handle})
        assert router._store.cache_lifecycle_state("s0") == paused
        assert router._lifecycle_states["s0"] == paused
        assert not router._lifecycle_handshakes
        with pytest.raises(RuntimeError, match="no available replicas"):
            router.acquire_server("paused")
        resumed = replace(ready, revision=2)
        assert router._receive_cache_lifecycle("s0", resumed.to_dict())["applied"]
        assert router.acquire_server("resumed")[1] is handle
    finally:
        router.close()


@pytest.mark.parametrize("notification_before_reply", [False, True])
def test_repeated_registration_does_not_revive_instance_superseded_by_notification(
    monkeypatch, notification_before_reply
):
    from tests.uni_agent.agent_aware_router.balancer._helpers import _FakeCollectorManager
    from uni_agent.agent_aware_router.balancer import KVCAwareBalancer

    monkeypatch.setattr("ray.get", lambda value, **kwargs: value)
    monkeypatch.setattr("ray.util.get_node_ip_address", lambda: "127.0.0.1")
    original = CacheLifecycleState("old-engine")
    replacement = CacheLifecycleState("new-engine", revision=1, serving=False)
    calls = 0

    def register(sink):
        nonlocal calls
        calls += 1
        if calls == 2 and notification_before_reply:
            # This RPC captured the old instance before an engine change's
            # notification arrived. Its delayed reply must not revive it.
            result = asyncio.run(notify_lifecycle(sink, replacement.to_dict()))
            assert not result["applied"] and result["reason"] == "lifecycle_handshake_pending"
            state = original
        else:
            state = original if calls == 1 else replacement
        return {"state": state.to_dict(), "sources": {}}

    handle = SimpleNamespace(
        register_cache_lifecycle_sink=SimpleNamespace(remote=register),
        unregister_cache_lifecycle_sink=SimpleNamespace(remote=lambda token, generation=None: None),
    )
    router = KVCAwareBalancer({"s0": handle}, _router_config(), provider_factory=_FakeCollectorManager)
    try:
        router.add_servers({"s0": handle})
        assert not router._pending_servers
        assert router._store.cache_lifecycle_state("s0") == replacement
        assert router._lifecycle_states["s0"] == replacement
        assert not router._lifecycle_handshakes
        with pytest.raises(RuntimeError, match="no available replicas"):
            router.acquire_server("paused")
        resumed = replace(replacement, revision=2, serving=True)
        assert router._receive_cache_lifecycle("s0", resumed.to_dict())["applied"]
        assert router.acquire_server("resumed")[1] is handle
    finally:
        router.close()
