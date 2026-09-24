from __future__ import annotations

import pytest

from uni_agent.events import (
    GENERATION_FINISHED,
    SESSION_CLOSED,
    SESSION_OPENED,
    EventContext,
    EventPublisher,
    LocalEventBus,
)
from uni_agent.metrics import GatewayTaskMetricsProjector, MetricsFragment

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _runtime():
    bus = LocalEventBus()
    publisher = EventPublisher(
        bus,
        run_id="run-1",
        producer_id="gateway-1",
        producer_epoch="epoch-1",
    )
    projector = GatewayTaskMetricsProjector(bus, source_instance="gateway-1")
    return bus, publisher, projector


def _context(session_id: str) -> EventContext:
    return EventContext(episode_id=f"episode-{session_id}", session_id=session_id)


def test_gateway_projector_uses_bounded_sufficient_statistics():
    bus, publisher, projector = _runtime()
    context = _context("session-1")
    publisher.publish(SESSION_OPENED, {"revision": 1}, context=context)
    publisher.publish(
        GENERATION_FINISHED,
        {
            "status": "success",
            "duration_ns": 1_000_000_000,
            "prepare_duration_ns": 100_000_000,
            "backend_duration_ns": 700_000_000,
            "decode_duration_ns": 200_000_000,
        },
        context=context,
    )
    publisher.publish(
        GENERATION_FINISHED,
        {
            "status": "success",
            "coalesced": True,
            "duration_ns": 500_000_000,
            "prepare_duration_ns": None,
            "backend_duration_ns": None,
            "decode_duration_ns": None,
        },
        context=context,
    )
    publisher.publish(SESSION_CLOSED, {"close_reason": "finalized", "revision": 2}, context=context)

    fragment = projector.finalize_session("session-1")

    assert fragment.complete
    assert fragment.metrics["gateway.requests"].count == 2
    assert fragment.metrics["gateway.requests"].value == 2.0
    assert fragment.metrics["gateway.request_s"].to_dict() == {
        "aggregation": "sum",
        "count": 2,
        "sum": 1.5,
        "min": 0.5,
        "max": 1.0,
        "last": 0.5,
    }
    assert fragment.metrics["gateway.backend_generate_s"].count == 1
    assert fragment.metrics["gateway.backend_generate_s"].value == 0.7
    assert "values" not in fragment.to_dict()["metrics"]["gateway.request_s"]
    assert MetricsFragment.from_dict(fragment.to_dict()) == fragment
    projector.close()
    bus.close()


def test_gateway_projector_isolates_sessions_and_marks_malformed_session_incomplete():
    bus, publisher, projector = _runtime()
    good = _context("good")
    bad = _context("bad")
    for context in (good, bad):
        publisher.publish(SESSION_OPENED, {"revision": 1}, context=context)
    publisher.publish(
        GENERATION_FINISHED,
        {"duration_ns": 10, "prepare_duration_ns": None, "backend_duration_ns": None, "decode_duration_ns": None},
        context=good,
    )
    publisher.publish(
        GENERATION_FINISHED,
        {"duration_ns": -1, "prepare_duration_ns": None, "backend_duration_ns": None, "decode_duration_ns": None},
        context=bad,
    )
    for context in (good, bad):
        publisher.publish(SESSION_CLOSED, {"close_reason": "finalized"}, context=context)

    good_fragment = projector.finalize_session("good")
    bad_fragment = projector.finalize_session("bad")

    assert good_fragment.complete
    assert not bad_fragment.complete
    assert bad_fragment.metrics["gateway.requests"].value == 1.0
    assert any("duration_ns" in reason for reason in bad_fragment.incomplete_reasons)
    projector.close()
    bus.close()


def test_finalize_without_opened_session_is_explicitly_incomplete():
    bus, _publisher, projector = _runtime()

    fragment = projector.finalize_session("missing")

    assert not fragment.complete
    assert fragment.metrics == {}
    assert fragment.incomplete_reasons == ("session metrics were not initialized",)
    assert "errors" not in fragment.to_dict()
    projector.close()
    bus.close()
