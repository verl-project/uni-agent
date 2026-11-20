"""Metadata discovery shares startup work with background recovery."""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from uni_agent.agent_aware_router.collectors.metadata import MetadataTransport

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


class Store:
    def __init__(self):
        self.revisions = {}
        self.requests = {}
        self.diagnostics = []
        self.polls = 0

    def metadata_revision(self, node):
        return self.revisions.get(node, 0)

    def metadata_refresh_requests(self):
        self.polls += 1
        return dict(self.requests)

    def metadata_status(self, node):
        return {"ready": True}

    def record_cache_diagnostic(self, node, stage, reason, details=None, count=1):
        self.diagnostics.append((node, stage, reason, details, count))


class Remote:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def remote(self):
        self.calls += 1
        return next(self.responses)


async def eventually(predicate):
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("condition did not become true")


@asynccontextmanager
async def subscription(transport, handler):
    task = asyncio.create_task(transport.subscribe(handler))
    try:
        yield task
    finally:
        transport.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_startup_timeout_keeps_one_remote_call_and_applies_late_success():
    async def run():
        loop = asyncio.get_running_loop()
        result = loop.create_future()
        remote = Remote([result])
        store, now = Store(), [0.0]
        transport = MetadataTransport({"a": SimpleNamespace(get_kv_cache_metadata=remote)}, store, clock=lambda: now[0])
        transport._QUERY_TIMEOUT = 0.005
        updates = []
        async with subscription(transport, lambda payload, node: updates.append((payload, node))):
            await eventually(lambda: transport.get_stream_status().get("a", {}).get("reason") == "query_timeout")
            now[0] = 60.0
            polls = store.polls
            await eventually(lambda: store.polls > polls)
            await asyncio.sleep(0)
            assert remote.calls == 1
            assert not result.cancelled()
            result.set_result({"status": "ok"})
            await eventually(lambda: bool(updates))
            assert updates[0][1] == "a"
            assert remote.calls == 1

    asyncio.run(run())


def test_concurrent_calls_are_limited_and_queued_sources_continue():
    async def run():
        loop = asyncio.get_running_loop()
        results = [loop.create_future() for _ in range(10)]
        remotes = [Remote([result]) for result in results]
        transport = MetadataTransport(
            {str(index): SimpleNamespace(get_kv_cache_metadata=remote) for index, remote in enumerate(remotes)}, Store()
        )
        async with subscription(transport, lambda *_: None):
            await eventually(lambda: sum(remote.calls for remote in remotes) == 8)
            assert sum(remote.calls for remote in remotes) == 8
            results[0].set_result({"status": "ok"})
            await eventually(lambda: sum(remote.calls for remote in remotes) == 9)
            assert transport.get_stream_status()["0"]["ready"]
            assert sum(status["inflight"] for status in transport.get_stream_status().values()) == 8

    asyncio.run(run())


def test_refresh_revision_fences_old_result_and_queries_again():
    async def run():
        loop = asyncio.get_running_loop()
        old, current = loop.create_future(), loop.create_future()
        current.set_result({"status": "ok", "marker": "current"})
        remote = Remote([old, current])
        store = Store()
        transport = MetadataTransport({"a": SimpleNamespace(get_kv_cache_metadata=remote)}, store)
        updates = []
        async with subscription(transport, lambda payload, node: updates.append(payload)):
            await eventually(lambda: remote.calls == 1)
            store.revisions["a"] = 1
            store.requests["a"] = 1
            old.set_result({"status": "ok", "marker": "stale"})
            await eventually(lambda: len(updates) == 1)
            assert remote.calls == 2
            assert updates[0] == {"metadata": {"status": "ok", "marker": "current"}, "revision": 1}

    asyncio.run(run())


def test_background_retries_obey_backoff_and_stop_when_healthy(caplog, monkeypatch):
    from uni_agent.agent_aware_router.collectors import metadata as metadata_module

    monkeypatch.setattr(metadata_module.logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=metadata_module.logger.name)

    async def run():
        loop = asyncio.get_running_loop()
        responses = [loop.create_future() for _ in range(9)]
        for response, status in zip(
            responses[:-1],
            (
                "not_ready",
                "not_ready",
                "unsupported",
                "unsupported",
                "not_ready",
                "not_ready",
                "not_ready",
                "not_ready",
            ),
            strict=True,
        ):
            response.set_result({"status": status})
        responses[-1].set_result({"status": "ok"})
        remote = Remote(responses)
        now = [0.0]
        store = Store()
        transport = MetadataTransport({"a": SimpleNamespace(get_kv_cache_metadata=remote)}, store, clock=lambda: now[0])
        async with subscription(transport, lambda *_: None):
            for attempt, delay in enumerate((1, 2, 4, 8, 16, 30, 30, 30), start=1):
                await eventually(lambda attempt=attempt: transport.get_stream_status()["a"]["failures"] == attempt)
                now[0] += delay - 0.001
                polls = store.polls
                await eventually(lambda polls=polls: store.polls > polls)
                await asyncio.sleep(0)
                assert remote.calls == attempt
                # Cross the warning throttle while the source is still failing.
                now[0] += 301 if attempt == 7 else 0.002
            await eventually(lambda: transport.get_stream_status()["a"]["last_success"] is not None)
            status = transport.get_stream_status()["a"]
            assert status["attempts"] == remote.calls == 9
            assert status["next_retry"] is None
            now[0] += 1000
            polls = store.polls
            await eventually(lambda: store.polls > polls)
            await asyncio.sleep(0)
            assert remote.calls == 9

    asyncio.run(run())

    records = [record for record in caplog.records if record.name == metadata_module.logger.name]
    warnings = [record.getMessage() for record in records if record.levelno == logging.WARNING]
    assert len(warnings) == 4
    assert all(f"failures={attempt}" in message for attempt, message in zip((1, 3, 5, 8), warnings, strict=False))
    assert any(record.levelno == logging.INFO and "recovered replica=a" in record.getMessage() for record in records)


def test_background_success_updates_real_store_through_collector(layout):
    from uni_agent.agent_aware_router.collectors.collector import Collector
    from uni_agent.agent_aware_router.collectors.metadata import MetadataParser
    from uni_agent.agent_aware_router.store.data_store import DataStore

    async def run():
        loop = asyncio.get_running_loop()
        unavailable, healthy = loop.create_future(), loop.create_future()
        unavailable.set_result({"status": "not_ready"})
        healthy.set_result(layout((("full_attention", 4),), unit=4))
        remote = Remote([unavailable, healthy])
        store = DataStore()
        transport = MetadataTransport({"a": SimpleNamespace(get_kv_cache_metadata=remote)}, store)
        transport._RETRY_DELAYS = (0.005,)
        collector = Collector(transport, MetadataParser())
        task = asyncio.create_task(transport.subscribe(collector._handle_raw_data))
        try:
            await eventually(lambda: store.metadata_status("a")["ready"])
            assert store.dispatch_schema("a").hash_unit == 4
            assert transport.get_stream_status()["a"]["attempts"] == 2
        finally:
            transport.stop()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            collector.stop()

    asyncio.run(run())


@pytest.mark.parametrize("failure,reason", [("timeout", "query_timeout"), ("exception", "metadata_query_failed")])
def test_local_rpc_failure_records_bounded_diagnostic(failure, reason):
    async def run():
        result = asyncio.get_running_loop().create_future()
        if failure == "exception":
            result.set_exception(RuntimeError("worker died"))
        remote = Remote([result])
        store = Store()
        transport = MetadataTransport({"a": SimpleNamespace(get_kv_cache_metadata=remote)}, store)
        transport._QUERY_TIMEOUT = 0.005
        updates = []
        async with subscription(transport, lambda payload, node: updates.append(payload)):
            await eventually(lambda: transport.get_stream_status()["a"]["failures"] == 1)
            assert len(store.diagnostics) == 1
            node, stage, actual_reason, details, count = store.diagnostics[0]
            assert (node, stage, actual_reason, count) == ("a", "metadata", reason, 1)
            if failure == "exception":
                assert details["error_type"] == "RuntimeError"
            assert updates == []

    asyncio.run(run())


def test_response_error_has_readable_reason_and_is_not_counted_twice():
    async def run():
        result = asyncio.get_running_loop().create_future()
        result.set_result({"status": "ok", "metadata_errors": {"native_hash_config": {"reason": "rpc_timeout"}}})
        remote = Remote([result])
        store = Store()
        transport = MetadataTransport({"a": SimpleNamespace(get_kv_cache_metadata=remote)}, store)
        updates = []
        async with subscription(transport, lambda payload, node: updates.append(payload)):
            await eventually(lambda: transport.get_stream_status()["a"]["failures"] == 1)
            assert transport.get_stream_status()["a"]["reason"] == "rpc_timeout"
            assert len(updates) == 1
            assert store.diagnostics == []

    asyncio.run(run())


def test_rpc_failure_flushes_diagnostics_without_metadata_or_metrics(monkeypatch):
    from unittest.mock import Mock

    from uni_agent.agent_aware_router.collectors.collector import get_collector
    from uni_agent.agent_aware_router.config.collector import CollectorConfig
    from uni_agent.agent_aware_router.insight import emitter

    class DiagnosticStore(Store):
        def drain_cache_diagnostics(self):
            recorded, self.diagnostics = self.diagnostics, []
            return [(node, stage, reason, count) for node, stage, reason, _, count in recorded]

        def get_cache_status(self):
            return {}

    async def run():
        result = asyncio.get_running_loop().create_future()
        result.set_exception(RuntimeError("worker died"))
        handle = SimpleNamespace(get_kv_cache_metadata=Remote([result]))
        collector = get_collector("vllm_metadata", CollectorConfig(), metadata_servers={"a": handle})
        collector._data_store = collector._transport._store = DiagnosticStore()
        emitted = Mock()
        monkeypatch.setattr(emitter, "on_cache_diagnostic", emitted)
        task = asyncio.create_task(collector._transport.subscribe(collector._handle_raw_data))
        try:
            await eventually(lambda: collector._transport.get_stream_status()["a"]["failures"] == 1)
            emitted.assert_called_once_with("a", "metadata", "metadata_query_failed", 1)
        finally:
            collector._transport.stop()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            collector.stop()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["empty", "offline", "address_rpc"])
def test_startup_endpoint_failure_retries_without_requerying_healthy_metadata(monkeypatch, failure):
    async def run():
        loop = asyncio.get_running_loop()
        metadata, address = loop.create_future(), loop.create_future()
        metadata.set_result({"status": "ok"})
        address.set_result(("127.0.0.1", 8000))
        addresses = [address]
        if failure == "address_rpc":
            failed_address = loop.create_future()
            failed_address.set_exception(RuntimeError("server starting"))
            addresses.insert(0, failed_address)
        metadata_remote, address_remote = Remote([metadata]), Remote(addresses)
        handle = SimpleNamespace(get_kv_cache_metadata=metadata_remote, get_server_address=address_remote)
        transport = MetadataTransport({"a": handle}, Store())
        transport._RETRY_DELAYS = (0.005,)
        endpoints = ["127.0.0.1:9000", "127.0.0.1:9001", "zmq", "events"]
        queried = []

        async def discover(server_address):
            queried.append(server_address)
            if len(queried) == 1:
                if failure == "offline":
                    raise httpx.ConnectError("not listening yet")
                if failure == "empty":
                    return None
            return endpoints

        monkeypatch.setattr(transport, "_fetch_kv_event_endpoints", discover, raising=False)
        discovered, updates = [], []
        transport._on_endpoints = lambda node, address, endpoint: discovered.append((node, address, endpoint))
        async with subscription(transport, lambda payload, node: updates.append(payload)):
            await eventually(lambda: transport.get_stream_status()["a"]["failures"] == 1)
            assert transport.get_stream_status()["a"]["next_retry"] is not None
            await eventually(lambda: transport.get_stream_status()["a"]["ready"])
            assert metadata_remote.calls == 1
            assert address_remote.calls == (2 if failure == "address_rpc" else 1)
            assert queried == ["127.0.0.1:8000"] * (1 if failure == "address_rpc" else 2)
            assert discovered == [("a", "127.0.0.1:8000", None), ("a", "127.0.0.1:8000", endpoints)]
            assert len(updates) == 1

    asyncio.run(run())


def test_late_initial_endpoint_replays_without_restarting_existing_stream(monkeypatch, layout):
    import msgpack

    from tests.uni_agent.agent_aware_router.collectors.zmq_helpers import Publisher
    from uni_agent.agent_aware_router.collectors.provider import CollectorManager
    from uni_agent.agent_aware_router.config.collector import CollectorConfig
    from uni_agent.agent_aware_router.store.data_store import DataStore
    from uni_agent.agent_aware_router.utils.hash import get_prefix_hashes_incremental

    async def run():
        first, second = Publisher(), Publisher()
        sources = {"127.0.0.1:8000": first, "127.0.0.1:8001": second}

        second_endpoint_available = threading.Event()

        async def discover(address):
            if address.endswith(":8001") and not second_endpoint_available.is_set():
                return None
            return [*sources[address].addresses, "zmq", "topic"]

        monkeypatch.setattr(MetadataTransport, "_fetch_kv_event_endpoints", staticmethod(discover))

        def handle(port):
            async def metadata():
                return layout((("full_attention", 4),), unit=4)

            async def address():
                return "127.0.0.1", port

            return SimpleNamespace(
                get_kv_cache_metadata=SimpleNamespace(remote=metadata),
                get_server_address=SimpleNamespace(remote=address),
            )

        manager = CollectorManager(
            CollectorConfig(), ["vllm_zmq"], metadata_servers={"a": handle(8000), "b": handle(8001)}
        )
        store = DataStore()
        tokens = list(range(4))
        hashes, _ = get_prefix_hashes_incremental(tokens, 4, 0, 0)
        hashes = [str(value) for value in hashes]

        def hit(node):
            return store.estimate_prefixes([node], hashes, 5)[node].estimated_hit_tokens

        async def replay(publisher, remote_hash):
            identity, start = await publisher.requested_start()
            assert start == 0
            payload = msgpack.packb([0, [["BlockStored", [remote_hash], None, tokens, 4, None, "GPU"]]])
            await publisher.batch(identity, 0, payload)
            await publisher.end(identity)

        try:
            manager.start()
            await replay(first, "first")
            await eventually(lambda: hit("a") == 4)
            assert hit("b") == 0
            second_endpoint_available.set()
            await replay(second, "second")
            await eventually(lambda: hit("b") == 4)
            assert hit("a") == 4
        finally:
            manager.stop()
            first.close()
            second.close()

    asyncio.run(run())
