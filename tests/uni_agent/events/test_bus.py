from __future__ import annotations

import asyncio
import threading

import pytest

from uni_agent.events import (
    DeliveryMode,
    Event,
    EventBatch,
    EventContext,
    LocalEventBus,
    Scope,
    SubscriptionSpec,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _event(sequence: int, *, event_type: str = "GenerationFinished", producer_id: str = "gateway-1") -> Event:
    return Event(
        event_id=f"event-{sequence}",
        event_type=event_type,
        schema_version=1,
        run_id="run-1",
        producer_id=producer_id,
        producer_epoch="epoch-1",
        producer_seq=sequence,
        context=EventContext(session_id="session-1"),
        occurred_at_unix_ns=sequence,
        payload={"sequence": sequence},
    )


def test_subscription_filters_and_unsubscribe_lifecycle():
    bus = LocalEventBus()
    seen = []
    subscription = bus.subscribe(
        SubscriptionSpec(
            subscription_id="gateway-finished",
            event_types=("GenerationFinished",),
            run_ids=("run-1",),
            producer_ids=("gateway-1",),
            scope=Scope.LOCAL,
            delivery=DeliveryMode.INLINE,
        ),
        seen.append,
    )

    assert bus.publish(_event(1)).delivered == 1
    assert bus.publish(_event(2, event_type="GenerationPrepared")).matched_subscriptions == 0
    assert bus.publish(_event(3, producer_id="framework-1")).matched_subscriptions == 0
    subscription.close()
    assert bus.publish(_event(4)).matched_subscriptions == 0
    assert [event.event_id for event in seen] == ["event-1"]
    bus.close()


def test_inline_subscriber_failure_isolated_from_other_subscribers():
    bus = LocalEventBus()
    seen = []

    def fail(_event):
        raise RuntimeError("projector failed")

    failed_subscription = bus.subscribe(
        SubscriptionSpec("failed", scope=Scope.LOCAL, delivery=DeliveryMode.INLINE),
        fail,
    )
    bus.subscribe(
        SubscriptionSpec("healthy", scope=Scope.LOCAL, delivery=DeliveryMode.INLINE),
        seen.append,
    )

    receipt = bus.publish(_event(1))

    assert receipt.matched_subscriptions == 2
    assert receipt.delivered == 1
    assert receipt.failed == 1
    assert not receipt.complete
    assert seen[0].event_id == "event-1"
    assert failed_subscription.stats.failed == 1
    bus.close()


def test_deliver_targets_only_named_subscription_and_never_republishes():
    bus = LocalEventBus()
    direct_seen = []
    unrelated_seen = []
    bus.subscribe(
        SubscriptionSpec("direct-target", scope=Scope.DIRECT, delivery=DeliveryMode.STATE_SYNC),
        direct_seen.append,
    )
    bus.subscribe(
        SubscriptionSpec("unrelated", scope=Scope.GLOBAL, delivery=DeliveryMode.INLINE),
        unrelated_seen.append,
    )
    batch = EventBatch((_event(1), _event(2)))

    ack = bus.deliver("direct-target", batch)

    assert ack.complete
    assert ack.delivered == 2
    assert [event.event_id for event in direct_seen] == ["event-1", "event-2"]
    assert unrelated_seen == []
    missing = bus.deliver("missing", batch)
    assert not missing.subscription_found
    assert missing.dropped == 2
    bus.close()


def test_deliver_rejects_an_event_outside_the_named_subscription_filter():
    bus = LocalEventBus()
    seen = []
    bus.subscribe(
        SubscriptionSpec(
            "direct-target",
            event_types=("SessionClosed",),
            scope=Scope.DIRECT,
            delivery=DeliveryMode.STATE_SYNC,
        ),
        seen.append,
    )

    ack = bus.deliver("direct-target", EventBatch((_event(1, event_type="SessionOpened"),)))

    assert not ack.complete
    assert ack.dropped == 1
    assert seen == []
    bus.close()


@pytest.mark.asyncio
async def test_queued_subscription_is_bounded_without_blocking_inline_subscriber():
    bus = LocalEventBus()
    started = threading.Event()
    release = threading.Event()
    slow_seen = []
    fast_seen = []

    def slow(event):
        started.set()
        release.wait(timeout=2)
        slow_seen.append(event)

    slow_subscription = bus.subscribe(
        SubscriptionSpec(
            "slow",
            scope=Scope.LOCAL,
            delivery=DeliveryMode.QUEUED,
            max_queue_events=1,
            max_queue_bytes=1024 * 1024,
        ),
        slow,
    )
    bus.subscribe(
        SubscriptionSpec("fast", scope=Scope.LOCAL, delivery=DeliveryMode.INLINE),
        fast_seen.append,
    )

    first = bus.publish(_event(1))
    assert first.enqueued == 1
    assert started.wait(timeout=1)
    second = bus.publish(_event(2))
    third = bus.publish(_event(3))

    assert second.enqueued == 1
    assert third.dropped == 1
    assert len(fast_seen) == 3
    release.set()
    report = await bus.flush(scope=Scope.LOCAL, deadline_s=2)
    assert report.pending_events == 0
    assert report.incomplete_subscriptions == ("slow",)
    assert [event.event_id for event in slow_seen] == ["event-1", "event-2"]
    assert slow_subscription.stats.dropped == 1
    bus.close()


@pytest.mark.asyncio
async def test_flush_waits_for_queued_delivery_and_reports_success():
    bus = LocalEventBus()
    seen = []
    bus.subscribe(
        SubscriptionSpec("queued", scope=Scope.GLOBAL, delivery=DeliveryMode.QUEUED),
        seen.append,
    )
    bus.publish(_event(1))
    bus.publish(_event(2))

    report = await bus.flush(scope=Scope.GLOBAL, deadline_s=1)

    assert report.complete
    assert report.pending_events == 0
    assert [event.event_id for event in seen] == ["event-1", "event-2"]
    bus.close()


def test_queue_byte_budget_drops_oversized_event_without_running_handler():
    bus = LocalEventBus()
    seen = []
    subscription = bus.subscribe(
        SubscriptionSpec(
            "tiny",
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.QUEUED,
            max_queue_events=10,
            max_queue_bytes=1,
        ),
        seen.append,
    )

    receipt = bus.publish(_event(1))

    assert receipt.dropped == 1
    assert seen == []
    assert subscription.stats.pending_bytes == 0
    assert not subscription.stats.complete
    bus.close()


def test_active_queued_handler_remains_inside_byte_budget():
    bus = LocalEventBus()
    started = threading.Event()
    release = threading.Event()
    event = _event(1)

    def slow(_event):
        started.set()
        release.wait(timeout=2)

    subscription = bus.subscribe(
        SubscriptionSpec(
            "byte-budget",
            scope=Scope.GLOBAL,
            delivery=DeliveryMode.QUEUED,
            max_queue_events=10,
            max_queue_bytes=event.estimated_size_bytes,
        ),
        slow,
    )

    assert bus.publish(event).enqueued == 1
    assert started.wait(timeout=1)
    assert subscription.stats.pending_bytes == event.estimated_size_bytes
    assert bus.publish(_event(2)).dropped == 1
    release.set()
    subscription.close(drain=True, timeout_s=1)
    bus.close()


def test_unsubscribe_does_not_wait_for_active_queued_handler():
    bus = LocalEventBus()
    started = threading.Event()
    release = threading.Event()

    def slow(_event):
        started.set()
        release.wait(timeout=2)

    subscription = bus.subscribe(
        SubscriptionSpec("non-blocking-close", scope=Scope.GLOBAL, delivery=DeliveryMode.QUEUED),
        slow,
    )
    bus.publish(_event(1))
    assert started.wait(timeout=1)

    assert subscription.close()
    assert not subscription.stats.accepting
    assert subscription.stats.pending_events == 1
    release.set()
    assert subscription.close(drain=True, timeout_s=1)
    bus.close()


def test_async_handler_is_rejected_instead_of_being_silently_unawaited():
    bus = LocalEventBus()

    async def async_handler(_event):
        await asyncio.sleep(0)

    with pytest.raises(TypeError, match="must be synchronous"):
        bus.subscribe(SubscriptionSpec("async-handler"), async_handler)
    bus.close()


def test_duplicate_subscription_id_is_rejected():
    bus = LocalEventBus()
    bus.subscribe(SubscriptionSpec("duplicate"), lambda _event: None)

    with pytest.raises(ValueError, match="duplicate subscription_id"):
        bus.subscribe(SubscriptionSpec("duplicate"), lambda _event: None)
    bus.close()
