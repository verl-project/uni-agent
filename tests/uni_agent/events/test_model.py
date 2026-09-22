from __future__ import annotations

from types import MappingProxyType

import pytest

from uni_agent.events import Event, EventBatch, EventContext

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _event(*, payload: dict | None = None) -> Event:
    return Event(
        event_id="event-1",
        event_type="GenerationFinished",
        schema_version=1,
        run_id="run-1",
        producer_id="gateway-1",
        producer_epoch="epoch-1",
        producer_seq=1,
        context=EventContext(session_id="session-1", attempt_id="attempt-1"),
        occurred_at_unix_ns=123,
        payload=payload or {},
    )


def test_event_copies_and_deeply_freezes_payload_and_round_trips_dto():
    source = {
        "status": "ok",
        "tokens": [1, 2],
        "nested": {"reason": "stop"},
        "raw": bytearray(b"ab"),
    }

    event = _event(payload=source)
    source["status"] = "changed"
    source["tokens"].append(3)
    source["nested"]["reason"] = "changed"

    assert isinstance(event.payload, MappingProxyType)
    assert event.payload["status"] == "ok"
    assert event.payload["tokens"] == (1, 2)
    assert event.payload["nested"]["reason"] == "stop"
    assert event.payload["raw"] == b"ab"
    assert event.estimated_size_bytes > 0
    with pytest.raises(TypeError):
        event.payload["new"] = "value"
    with pytest.raises(TypeError):
        event.payload["nested"]["reason"] = "value"

    dto = event.to_dict()
    assert dto["payload"]["tokens"] == [1, 2]
    assert Event.from_dict(dto) == event
    assert EventBatch.from_dict(EventBatch((event,)).to_dict()) == EventBatch((event,))


@pytest.mark.parametrize(
    "payload, error",
    [
        ({1: "not-a-string-key"}, "keys must be strings"),
        ({"unsupported": object()}, "must contain only"),
    ],
)
def test_event_rejects_payloads_that_cannot_cross_process_boundaries(payload, error):
    with pytest.raises(TypeError, match=error):
        _event(payload=payload)


def test_context_overlay_keeps_parent_values_and_accepts_child_values():
    parent = EventContext(episode_id="episode-1", session_id="session-1", global_step=7)
    child = EventContext(session_id="session-2", attempt_id="attempt-1")

    assert parent.overlay(child) == EventContext(
        episode_id="episode-1",
        session_id="session-2",
        attempt_id="attempt-1",
        global_step=7,
    )
    assert EventContext.from_dict({**parent.to_dict(), "future_optional_field": "ignored"}) == parent


def test_event_accepts_non_negative_sequence_and_timestamp_boundaries():
    event = _event()
    dto = event.to_dict()
    dto["producer_seq"] = 0
    dto["occurred_at_unix_ns"] = 0

    restored = Event.from_dict(dto)

    assert restored.producer_seq == 0
    assert restored.occurred_at_unix_ns == 0
