from __future__ import annotations

import pytest

from uni_agent.events import (
    GATEWAY_SESSION_STATE_SCOPE,
    REPLICA_CAPACITY_CHANGED,
    ROUTE_COMMITTED,
    Event,
    EventContext,
    StateSnapshot,
)
from uni_agent.gateway.admission import AdmissionSignalsProjector

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _event(event_type: str, sequence: int, payload: dict) -> Event:
    return Event(
        event_id=f"event-{sequence}",
        event_type=event_type,
        schema_version=1,
        run_id="run-1",
        producer_id="router-1",
        producer_epoch="router-epoch-1",
        producer_seq=sequence,
        context=EventContext(),
        occurred_at_unix_ns=sequence,
        payload=payload,
    )


def test_admission_route_history_is_bounded():
    projector = AdmissionSignalsProjector(max_route_observations=2)

    for sequence in range(1, 4):
        projector.apply(
            _event(
                ROUTE_COMMITTED,
                sequence,
                {"request_id": f"request-{sequence}", "replica_id": "s0", "ledger_version": sequence},
            )
        )

    status = projector.status()
    assert status["route_commits"] == 3
    assert status["retained_route_observations"] == 2
    assert status["dropped_route_observations"] == 1
    assert status["latest_route"]["request_id"] == "request-3"


def test_admission_capacity_ignores_duplicate_or_older_versions():
    projector = AdmissionSignalsProjector()
    base = {
        "replica_id": "s0",
        "replica_epoch": "epoch-1",
        "reason": "acquire",
        "health": "healthy",
        "replica_inflight": 1,
        "total_inflight": 1,
    }
    projector.apply(_event(REPLICA_CAPACITY_CHANGED, 1, {**base, "ledger_version": 2}))
    projector.apply(
        _event(
            REPLICA_CAPACITY_CHANGED,
            2,
            {**base, "ledger_version": 1, "reason": "stale", "replica_inflight": 0},
        )
    )

    [capacity] = projector.status()["capacity"]
    assert capacity["ledger_version"] == 2
    assert capacity["reason"] == "acquire"
    assert capacity["replica_inflight"] == 1


def test_admission_capacity_accepts_new_replica_epoch():
    projector = AdmissionSignalsProjector()
    base = {
        "replica_id": "s0",
        "reason": "server_added",
        "health": "healthy",
        "replica_inflight": 0,
        "total_inflight": 0,
    }
    projector.apply(_event(REPLICA_CAPACITY_CHANGED, 1, {**base, "replica_epoch": "epoch-1", "ledger_version": 5}))
    projector.apply(_event(REPLICA_CAPACITY_CHANGED, 2, {**base, "replica_epoch": "epoch-2", "ledger_version": 1}))

    [capacity] = projector.status()["capacity"]
    assert capacity["replica_epoch"] == "epoch-2"
    assert capacity["ledger_version"] == 1


def test_admission_terminal_session_retention_is_bounded():
    projector = AdmissionSignalsProjector(max_terminal_sessions=2)
    projector.install_gateway_snapshot(
        StateSnapshot(
            owner="gateway-0",
            source_epoch="source-1",
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=3,
            watermark=0,
            source_health="healthy",
            current_entities={},
            terminal_entities={
                f"session-{index}": {
                    "episode_id": f"episode-{index}",
                    "revision": 2,
                    "status": "closed",
                }
                for index in range(3)
            },
        )
    )

    status = projector.status()
    assert status["terminal_sessions"] == 2
    assert status["source_revisions"] == {"gateway-0": 3}
    assert status["source_health"] == {"gateway-0": "healthy"}
