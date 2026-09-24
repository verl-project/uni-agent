from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from uni_agent.events import (
    DeliveryMode,
    EventContext,
    EventPublisher,
    LocalEventBus,
    Scope,
    SubscriptionSpec,
    bind_event_context,
    get_current_event_context,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _inline_all(bus: LocalEventBus, seen: list):
    return bus.subscribe(
        SubscriptionSpec(
            subscription_id="capture",
            scope=Scope.LOCAL,
            delivery=DeliveryMode.INLINE,
        ),
        seen.append,
    )


def test_publisher_enriches_event_and_restores_nested_context():
    bus = LocalEventBus()
    seen = []
    _inline_all(bus, seen)
    event_ids = iter(("event-1", "event-2"))
    publisher = EventPublisher(
        bus,
        run_id="run-1",
        producer_id="gateway-1",
        producer_epoch="epoch-1",
        context=EventContext(episode_id="episode-1", session_id="base-session"),
        event_id_factory=lambda: next(event_ids),
        wall_clock_ns=lambda: 123,
    )

    with bind_event_context(EventContext(session_id="bound-session", generation_id="generation-1")):
        receipt = publisher.publish(
            "GenerationPrepared",
            {"context_tokens": 64},
            context=EventContext(attempt_id="attempt-1"),
        )
        assert get_current_event_context().session_id == "bound-session"

    assert get_current_event_context() == EventContext()
    assert receipt.complete
    assert receipt.delivered == 1
    assert seen[0].event_id == "event-1"
    assert seen[0].producer_seq == 1
    assert seen[0].occurred_at_unix_ns == 123
    assert seen[0].context == EventContext(
        episode_id="episode-1",
        session_id="bound-session",
        generation_id="generation-1",
        attempt_id="attempt-1",
    )

    publisher.publish("GenerationFinished")
    assert seen[1].producer_seq == 2
    assert seen[1].context.session_id == "base-session"
    bus.close()


@pytest.mark.asyncio
async def test_contextvar_bindings_are_isolated_between_async_tasks():
    bus = LocalEventBus()
    seen = []
    _inline_all(bus, seen)
    publisher = EventPublisher(bus, run_id="run-1", producer_id="framework-1")

    async def emit(session_id: str) -> None:
        with bind_event_context(EventContext(session_id=session_id)):
            await asyncio.sleep(0)
            publisher.publish("EpisodeStarted", {"owner": session_id})

    await asyncio.gather(emit("session-a"), emit("session-b"))

    assert {event.payload["owner"]: event.context.session_id for event in seen} == {
        "session-a": "session-a",
        "session-b": "session-b",
    }
    bus.close()


def test_publisher_sequence_is_unique_under_concurrent_publishers():
    bus = LocalEventBus()
    seen = []
    _inline_all(bus, seen)
    publisher = EventPublisher(bus, run_id="run-1", producer_id="router-1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: publisher.publish("RouteCommitted", {"value": value}), range(100)))

    assert sorted(event.producer_seq for event in seen) == list(range(1, 101))
    assert len({event.event_id for event in seen}) == 100
    bus.close()


def test_publish_without_subscribers_is_a_successful_noop():
    bus = LocalEventBus()
    publisher = EventPublisher(bus, run_id="run-1", producer_id="gateway-1")

    receipt = publisher.publish("SessionOpened", {"session_id": "session-1"})

    assert receipt.matched_subscriptions == 0
    assert receipt.complete
    bus.close()
