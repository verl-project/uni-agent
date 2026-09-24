from __future__ import annotations

import pytest

from uni_agent.events import (
    GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
    GATEWAY_SESSION_STATE_SCOPE,
    SESSION_CLOSED,
    DirectAckStatus,
    DirectStateBatch,
    Event,
    EventContext,
    SnapshotAndCursor,
    StateSnapshot,
)

from ._helpers import _make_balancer

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _event(*, schema_version: int = 1) -> Event:
    return Event(
        event_id="event-1",
        event_type=SESSION_CLOSED,
        schema_version=schema_version,
        run_id="run-1",
        producer_id="gateway-0",
        producer_epoch="source-1",
        producer_seq=1,
        context=EventContext(episode_id="episode-1", session_id="session-1"),
        occurred_at_unix_ns=1,
        payload={"revision": 2, "close_reason": "finalized"},
    )


def test_router_gateway_session_state_is_lazy_and_shadow_only():
    balancer = _make_balancer()
    before = balancer.get_status()
    assert balancer.get_direct_session_shadow_status()["enabled"] is False
    snapshot = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=StateSnapshot(
            owner="gateway-0",
            source_epoch="source-1",
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=1,
            watermark=0,
            source_health="healthy",
            current_entities={"session-1": {"episode_id": "episode-1", "revision": 1, "status": "open"}},
            terminal_entities={},
        ),
    )
    assert balancer.install_direct_snapshot(snapshot.to_dict())["status"] == DirectAckStatus.OK.value
    batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=1,
        events=(_event(),),
    )
    assert balancer.receive_direct_events(batch.to_dict())["status"] == DirectAckStatus.OK.value

    shadow = balancer.get_direct_session_shadow_status()

    assert shadow["enabled"] is True
    assert shadow["mode"] == "shadow"
    assert shadow["current"] == []
    assert shadow["terminal"] == [
        {
            "owner": "gateway-0",
            "session_id": "session-1",
            "episode_id": "episode-1",
            "revision": 2,
            "status": "closed",
            "close_reason": "finalized",
        }
    ]
    assert shadow["streams"][0]["cursor"] == 1
    assert balancer.get_status() == before


def test_router_rejects_unknown_direct_schema_without_mutating_shadow_state():
    balancer = _make_balancer()

    ack = balancer.receive_direct_events(
        {
            "schema_version": 99,
            "subscription_id": GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            "stream_epoch": "stream-1",
            "events": [],
        }
    )

    assert ack["status"] == DirectAckStatus.RESYNC_REQUIRED.value
    assert balancer.get_direct_session_shadow_status()["enabled"] is False


def test_router_marks_stream_stale_for_unknown_gateway_event_schema():
    balancer = _make_balancer()
    snapshot = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=StateSnapshot(
            owner="gateway-0",
            source_epoch="source-1",
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=0,
            watermark=0,
            source_health="healthy",
            current_entities={},
            terminal_entities={},
        ),
    )
    balancer.install_direct_snapshot(snapshot.to_dict())
    batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=1,
        events=(_event(schema_version=2),),
    )

    ack = balancer.receive_direct_events(batch.to_dict())

    assert ack["status"] == DirectAckStatus.RESYNC_REQUIRED.value
    assert balancer.get_direct_session_shadow_status()["streams"][0]["status"] == "stale"
