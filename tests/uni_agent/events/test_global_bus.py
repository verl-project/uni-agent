from __future__ import annotations

import threading
import time

import pytest

from uni_agent.events import (
    GATEWAY_TELEMETRY_SUBSCRIPTION,
    SESSION_CLOSED,
    SESSION_OPENED,
    DeliveryMode,
    Event,
    EventContext,
    GlobalBusForwarder,
    GlobalForwarderStatus,
    GlobalSubscriptionEndpoint,
    GlobalSubscriptionSpec,
    LocalEventBus,
    RayGlobalEventBus,
    Scope,
    SubscriptionSpec,
    TelemetryAck,
    TelemetryAckStatus,
    TelemetryBatch,
    TelemetryReporter,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _event(
    sequence: int,
    event_type: str = SESSION_OPENED,
    *,
    source_id: str = "gateway-0",
    run_id: str = "run-1",
) -> Event:
    return Event(
        event_id=f"event-{source_id}-{sequence}",
        event_type=event_type,
        schema_version=1,
        run_id=run_id,
        producer_id=source_id,
        producer_epoch="source-epoch-1",
        producer_seq=sequence,
        context=EventContext(episode_id="episode-1", session_id=f"session-{sequence}"),
        occurred_at_unix_ns=sequence,
        payload={"revision": sequence},
    )


def _batch(*events: Event, batch_id: str = "batch-1") -> TelemetryBatch:
    first = events[0]
    return TelemetryBatch(
        run_id=first.run_id,
        source_id=first.producer_id,
        source_epoch=first.producer_epoch,
        batch_id=batch_id,
        events=events,
    )


def _ok(batch: TelemetryBatch) -> dict:
    return TelemetryAck(batch_id=batch.batch_id, status=TelemetryAckStatus.OK).to_dict()


def _wait_until(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not converge")
        time.sleep(0.005)


def test_global_transport_dtos_round_trip_with_accepted_ack():
    batch = _batch(_event(1), _event(2, SESSION_CLOSED))
    ack = TelemetryAck(batch_id=batch.batch_id, status=TelemetryAckStatus.OK)

    assert TelemetryBatch.from_dict(batch.to_dict()) == batch
    assert TelemetryAck.from_dict(ack.to_dict()) == ack
    assert ack.stage == "accepted"


def test_forwarder_batches_events_and_retries_the_same_batch_id():
    bus = LocalEventBus()
    attempts = []

    def send(dto):
        batch = TelemetryBatch.from_dict(dto)
        attempts.append(batch)
        if len(attempts) == 1:
            return TelemetryAck(
                batch_id=batch.batch_id,
                status=TelemetryAckStatus.BUSY,
                reason="busy",
            ).to_dict()
        return _ok(batch)

    forwarder = GlobalBusForwarder(
        bus,
        SubscriptionSpec(
            "source-global",
            event_types=(SESSION_OPENED, SESSION_CLOSED),
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.INLINE,
            max_queue_events=8,
            max_queue_bytes=64 * 1024,
        ),
        run_id="run-1",
        source_id="gateway-0",
        source_epoch="source-epoch-1",
        send_batch=send,
        flush_interval_s=0.02,
        retry_backoff_s=0,
    )

    bus.publish(_event(1))
    bus.publish(_event(2, SESSION_CLOSED))
    assert forwarder.wait_until_idle(2)

    assert len(attempts) == 2
    assert attempts[0].batch_id == attempts[1].batch_id
    assert attempts[1].events == (_event(1), _event(2, SESSION_CLOSED))
    assert forwarder.health.accepted_batches == 1
    assert forwarder.health.accepted_events == 2
    assert forwarder.health.retries == 1
    assert forwarder.close()
    bus.close()


def test_forwarder_overflow_is_visible_without_blocking_publishers():
    bus = LocalEventBus()
    entered = threading.Event()
    release = threading.Event()

    def send(dto):
        batch = TelemetryBatch.from_dict(dto)
        entered.set()
        release.wait(2)
        return _ok(batch)

    forwarder = GlobalBusForwarder(
        bus,
        SubscriptionSpec(
            "bounded-source",
            event_types=(SESSION_OPENED,),
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.INLINE,
            max_queue_events=2,
            max_queue_bytes=64 * 1024,
        ),
        run_id="run-1",
        source_id="gateway-0",
        source_epoch="source-epoch-1",
        send_batch=send,
        max_batch_events=1,
        flush_interval_s=0,
    )

    bus.publish(_event(1))
    assert entered.wait(1)
    started = time.perf_counter()
    for sequence in range(2, 7):
        bus.publish(_event(sequence))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1
    assert forwarder.health.status is GlobalForwarderStatus.DEGRADED
    assert forwarder.health.dropped_events >= 3
    release.set()
    assert forwarder.close()
    bus.close()


def test_forwarder_non_draining_close_does_not_clear_an_active_batch():
    bus = LocalEventBus()
    entered = threading.Event()
    release = threading.Event()

    def send(dto):
        batch = TelemetryBatch.from_dict(dto)
        entered.set()
        release.wait(2)
        return _ok(batch)

    forwarder = GlobalBusForwarder(
        bus,
        SubscriptionSpec(
            "close-source",
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.INLINE,
            max_queue_events=8,
            max_queue_bytes=64 * 1024,
        ),
        run_id="run-1",
        source_id="gateway-0",
        source_epoch="source-epoch-1",
        send_batch=send,
        max_batch_events=1,
        flush_interval_s=0,
    )
    bus.publish(_event(1))
    bus.publish(_event(2))
    assert entered.wait(1)

    assert not forwarder.close(drain=False, timeout_s=0.01)
    release.set()
    assert forwarder.close(drain=False, timeout_s=1)
    assert forwarder.health.accepted_events == 1
    assert forwarder.health.dropped_events == 1
    bus.close()


def test_broker_deduplicates_ingress_and_isolates_slow_subscribers():
    broker = RayGlobalEventBus(max_queue_events=16, max_queue_bytes=128 * 1024)
    slow_entered = threading.Event()
    slow_release = threading.Event()
    fast_batches = []

    def slow_send(_subscription_id, dto):
        batch = TelemetryBatch.from_dict(dto)
        slow_entered.set()
        slow_release.wait(2)
        return _ok(batch)

    def fast_send(_subscription_id, dto):
        batch = TelemetryBatch.from_dict(dto)
        fast_batches.append(batch)
        return _ok(batch)

    broker.register_subscription(
        GlobalSubscriptionSpec(
            "slow",
            event_types=(SESSION_OPENED,),
            max_queue_events=1,
            max_queue_bytes=64 * 1024,
        ),
        slow_send,
    )
    broker.register_subscription(
        GlobalSubscriptionSpec("fast", event_types=(SESSION_OPENED,)),
        fast_send,
    )

    first = _batch(_event(1), batch_id="batch-1")
    assert broker.publish_batch(first).status is TelemetryAckStatus.OK
    assert broker.publish_batch(first).status is TelemetryAckStatus.OK
    assert slow_entered.wait(1)
    for sequence in range(2, 5):
        ack = broker.publish_batch(_batch(_event(sequence), batch_id=f"batch-{sequence}"))
        assert ack.status is TelemetryAckStatus.OK

    _wait_until(lambda: len(fast_batches) == 4)
    health = broker.health
    slow_health = next(item for item in health.subscribers if item.subscription_id == "slow")
    assert health.duplicate_batches == 1
    assert slow_health.dropped_events >= 2
    assert [batch.batch_id for batch in fast_batches] == ["batch-1", "batch-2", "batch-3", "batch-4"]

    slow_release.set()
    assert broker.close()


def test_broker_deduplication_is_scoped_to_one_run():
    broker = RayGlobalEventBus()
    received = []

    def send(_subscription_id, dto):
        batch = TelemetryBatch.from_dict(dto)
        received.append(batch)
        return _ok(batch)

    broker.register_subscription(GlobalSubscriptionSpec("subscriber"), send)
    assert broker.publish_batch(_batch(_event(1, run_id="run-1"), batch_id="batch-1")).status is TelemetryAckStatus.OK
    assert broker.publish_batch(_batch(_event(1, run_id="run-2"), batch_id="batch-1")).status is TelemetryAckStatus.OK
    _wait_until(lambda: len(received) == 2)

    assert {batch.run_id for batch in received} == {"run-1", "run-2"}
    assert broker.health.duplicate_batches == 0
    assert broker.close()


def test_broker_and_subscription_endpoint_count_busy_rejections():
    broker = RayGlobalEventBus(max_batch_events=1)
    oversized = _batch(_event(1), _event(2))

    assert broker.publish_batch(oversized).status is TelemetryAckStatus.BUSY
    assert broker.health.busy_batches == 1

    bus = LocalEventBus()
    reporter = TelemetryReporter(
        bus,
        SubscriptionSpec(
            "target",
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.QUEUED,
        ),
        lambda _event: None,
    )
    endpoint = GlobalSubscriptionEndpoint(bus, subscription_id="target", max_batch_events=1)
    assert endpoint.receive_batch("other", _batch(_event(1))).status is TelemetryAckStatus.BUSY
    assert endpoint.receive_batch("target", oversized).status is TelemetryAckStatus.BUSY
    assert endpoint.health.busy_batches == 2

    assert broker.close()
    assert endpoint.close()
    assert reporter.close()
    bus.close()


def test_subscription_endpoint_delivers_once_without_republishing():
    bus = LocalEventBus()
    reported = []
    unrelated = []
    reporter = TelemetryReporter(
        bus,
        SubscriptionSpec(
            GATEWAY_TELEMETRY_SUBSCRIPTION,
            event_types=(SESSION_OPENED,),
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.QUEUED,
        ),
        reported.append,
    )
    bus.subscribe(
        SubscriptionSpec(
            "unrelated-global",
            event_types=(SESSION_OPENED,),
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.INLINE,
        ),
        unrelated.append,
    )
    endpoint = GlobalSubscriptionEndpoint(bus, subscription_id=GATEWAY_TELEMETRY_SUBSCRIPTION)
    batch = _batch(_event(1))

    assert endpoint.receive_batch(GATEWAY_TELEMETRY_SUBSCRIPTION, batch).status is TelemetryAckStatus.OK
    assert endpoint.receive_batch(GATEWAY_TELEMETRY_SUBSCRIPTION, batch).status is TelemetryAckStatus.OK
    _wait_until(lambda: len(reported) == 1)

    assert unrelated == []
    assert endpoint.health.duplicate_batches == 1
    assert endpoint.health.delivered_events == 1
    assert endpoint.close()
    assert reporter.close()
    bus.close()


def test_subscription_endpoint_requires_its_named_local_subscription():
    bus = LocalEventBus()

    with pytest.raises(ValueError, match="is not installed"):
        GlobalSubscriptionEndpoint(bus, subscription_id="missing")

    bus.close()


def test_reporter_failure_does_not_block_another_global_subscriber():
    broker = RayGlobalEventBus()
    failing_bus = LocalEventBus()
    healthy_bus = LocalEventBus()
    healthy_events = []

    def fail(_event):
        raise RuntimeError("reporter unavailable")

    failing_reporter = TelemetryReporter(
        failing_bus,
        SubscriptionSpec(
            "failing",
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.QUEUED,
        ),
        fail,
    )
    healthy_reporter = TelemetryReporter(
        healthy_bus,
        SubscriptionSpec(
            "healthy",
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.QUEUED,
        ),
        healthy_events.append,
    )
    failing_endpoint = GlobalSubscriptionEndpoint(failing_bus, subscription_id="failing")
    healthy_endpoint = GlobalSubscriptionEndpoint(healthy_bus, subscription_id="healthy")
    broker.register_subscription(
        GlobalSubscriptionSpec("failing"),
        lambda subscription_id, dto: failing_endpoint.receive_batch(
            subscription_id, TelemetryBatch.from_dict(dto)
        ).to_dict(),
    )
    broker.register_subscription(
        GlobalSubscriptionSpec("healthy"),
        lambda subscription_id, dto: healthy_endpoint.receive_batch(
            subscription_id, TelemetryBatch.from_dict(dto)
        ).to_dict(),
    )

    assert broker.publish_batch(_batch(_event(1))).status is TelemetryAckStatus.OK
    _wait_until(lambda: len(healthy_events) == 1 and failing_reporter.health.failed_events == 1)

    assert healthy_reporter.health.reported_events == 1
    assert failing_reporter.health.failed_events == 1
    assert broker.close()
    assert failing_endpoint.close()
    assert healthy_endpoint.close()
    assert failing_reporter.close()
    assert healthy_reporter.close()
    failing_bus.close()
    healthy_bus.close()
