from __future__ import annotations

import threading
import time

import pytest

from uni_agent.events import (
    GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
    GATEWAY_SESSION_STATE_SCOPE,
    SESSION_CLOSED,
    SESSION_OPENED,
    DeliveryMode,
    DirectAckStatus,
    DirectEventEndpoint,
    DirectRayEventBridge,
    DirectStateBatch,
    DirectSyncStatus,
    Event,
    EventBatch,
    EventContext,
    LocalEventBus,
    Scope,
    SnapshotAndCursor,
    StateSnapshot,
    SubscriptionSpec,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


def _event(sequence: int, event_type: str = SESSION_OPENED, *, source_epoch: str = "source-epoch-1") -> Event:
    return Event(
        event_id=f"event-{source_epoch}-{sequence}",
        event_type=event_type,
        schema_version=1,
        run_id="run-1",
        producer_id="gateway-0",
        producer_epoch=source_epoch,
        producer_seq=sequence,
        context=EventContext(episode_id="episode-1", session_id="session-1"),
        occurred_at_unix_ns=sequence,
        payload={"revision": 1 if event_type == SESSION_OPENED else 2},
    )


def _snapshot(*, source_epoch: str = "source-epoch-1", watermark: int = 0) -> StateSnapshot:
    return StateSnapshot(
        owner="gateway-0",
        source_epoch=source_epoch,
        scope=GATEWAY_SESSION_STATE_SCOPE,
        revision=0,
        watermark=watermark,
        source_health="healthy",
        current_entities={},
        terminal_entities={},
    )


def _endpoint(handler):
    bus = LocalEventBus()
    snapshots = []
    bus.subscribe(
        SubscriptionSpec(
            GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            event_types=(SESSION_OPENED, SESSION_CLOSED),
            scope=Scope.DIRECT,
            delivery=DeliveryMode.STATE_SYNC,
        ),
        handler,
    )
    return bus, snapshots, DirectEventEndpoint(bus, snapshot_installer=snapshots.append)


def test_direct_transport_dtos_round_trip_without_runtime_objects():
    snapshot_request = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=_snapshot(),
    )
    batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-epoch-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=2,
        events=(_event(1), _event(2, SESSION_CLOSED)),
    )

    assert SnapshotAndCursor.from_dict(snapshot_request.to_dict()) == snapshot_request
    assert DirectStateBatch.from_dict(batch.to_dict()) == batch
    assert EventBatch.from_dict(EventBatch(batch.events).to_dict()).events == batch.events


def test_endpoint_does_not_install_snapshot_for_an_unknown_subscription():
    bus = LocalEventBus()
    snapshots = []
    endpoint = DirectEventEndpoint(bus, snapshot_installer=snapshots.append)
    request = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=_snapshot(),
    )

    ack = endpoint.install_snapshot(request)

    assert ack.status is DirectAckStatus.RESYNC_REQUIRED
    assert snapshots == []
    assert endpoint.status() == ()
    bus.close()


def test_endpoint_applies_contiguous_events_once_and_requires_resync_after_gap():
    seen = []
    bus, snapshots, endpoint = _endpoint(seen.append)
    request = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=_snapshot(),
    )
    assert endpoint.install_snapshot(request).status is DirectAckStatus.OK

    batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-epoch-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=2,
        events=(_event(1), _event(2, SESSION_CLOSED)),
    )
    assert endpoint.receive_events(batch).contiguous_cursor == 2
    assert endpoint.receive_events(batch).contiguous_cursor == 2
    assert len(seen) == 2

    gap = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-epoch-1",
        stream_epoch="stream-1",
        first_seq=4,
        last_seq=4,
        events=(_event(4),),
    )
    assert endpoint.receive_events(gap).status is DirectAckStatus.RESYNC_REQUIRED
    assert (
        endpoint.receive_events(
            DirectStateBatch(
                run_id="run-1",
                subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
                source_epoch="source-epoch-1",
                stream_epoch="stream-1",
                first_seq=3,
                last_seq=3,
                events=(_event(3),),
            )
        ).status
        is DirectAckStatus.RESYNC_REQUIRED
    )
    assert snapshots == [_snapshot()]
    bus.close()


def test_new_source_epoch_snapshot_invalidates_the_old_stream():
    bus, _snapshots, endpoint = _endpoint(lambda _event: None)
    first = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=_snapshot(),
    )
    second = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-2",
        snapshot=_snapshot(source_epoch="source-epoch-2"),
    )

    assert endpoint.install_snapshot(first).status is DirectAckStatus.OK
    assert endpoint.install_snapshot(second).status is DirectAckStatus.OK
    old_batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-epoch-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=1,
        events=(_event(1),),
    )
    assert endpoint.receive_events(old_batch).status is DirectAckStatus.RESYNC_REQUIRED
    bus.close()


def test_unhealthy_or_regressive_snapshot_cannot_restore_a_stream():
    bus, _snapshots, endpoint = _endpoint(lambda _event: None)
    initial = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=StateSnapshot(
            owner="gateway-0",
            source_epoch="source-epoch-1",
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=2,
            watermark=0,
            source_health="healthy",
            current_entities={},
            terminal_entities={},
        ),
    )
    endpoint.install_snapshot(initial)
    unhealthy = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-2",
        snapshot=StateSnapshot(
            owner="gateway-0",
            source_epoch="source-epoch-1",
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=3,
            watermark=0,
            source_health="stale",
            current_entities={},
            terminal_entities={},
        ),
    )
    regressive = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-3",
        snapshot=StateSnapshot(
            owner="gateway-0",
            source_epoch="source-epoch-1",
            scope=GATEWAY_SESSION_STATE_SCOPE,
            revision=1,
            watermark=0,
            source_health="healthy",
            current_entities={},
            terminal_entities={},
        ),
    )

    assert endpoint.install_snapshot(unhealthy).status is DirectAckStatus.RESYNC_REQUIRED
    assert endpoint.install_snapshot(regressive).status is DirectAckStatus.RESYNC_REQUIRED
    assert endpoint.status()[0]["revision"] == 2
    bus.close()


def test_endpoint_rejects_events_outside_the_target_subscription_filter():
    bus, _snapshots, endpoint = _endpoint(lambda _event: None)
    endpoint.install_snapshot(
        SnapshotAndCursor(
            subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            stream_epoch="stream-1",
            snapshot=_snapshot(),
        )
    )
    unexpected = _event(1, event_type="GenerationFinished")
    batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-epoch-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=1,
        events=(unexpected,),
    )

    ack = endpoint.receive_events(batch)

    assert ack.status is DirectAckStatus.RESYNC_REQUIRED
    assert ack.contiguous_cursor == 0
    bus.close()


def test_subscriber_failure_marks_the_stream_stale_without_advancing_cursor():
    def fail(_event):
        raise RuntimeError("cannot commit")

    bus, _snapshots, endpoint = _endpoint(fail)
    request = SnapshotAndCursor(
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        stream_epoch="stream-1",
        snapshot=_snapshot(),
    )
    endpoint.install_snapshot(request)
    batch = DirectStateBatch(
        run_id="run-1",
        subscription_id=GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
        source_epoch="source-epoch-1",
        stream_epoch="stream-1",
        first_seq=1,
        last_seq=1,
        events=(_event(1),),
    )

    ack = endpoint.receive_events(batch)

    assert ack.status is DirectAckStatus.RESYNC_REQUIRED
    assert ack.contiguous_cursor == 0
    assert endpoint.status()[0]["status"] == "stale"
    bus.close()


def test_bridge_retries_the_same_batch_after_ack_loss_and_applies_it_once():
    target_seen = []
    target_bus, _snapshots, endpoint = _endpoint(target_seen.append)
    source_bus = LocalEventBus()
    attempts = []

    def send_batch(data):
        attempts.append(data)
        if len(attempts) == 1:
            raise RuntimeError("ack lost")
        return endpoint.receive_events(DirectStateBatch.from_dict(data)).to_dict()

    bridge = DirectRayEventBridge(
        source_bus,
        SubscriptionSpec(
            GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            event_types=(SESSION_OPENED,),
            scope=Scope.DIRECT,
            delivery=DeliveryMode.STATE_SYNC,
        ),
        run_id="run-1",
        source_epoch="source-epoch-1",
        send_batch=send_batch,
        install_snapshot=lambda data: endpoint.install_snapshot(SnapshotAndCursor.from_dict(data)).to_dict(),
        retry_backoff_s=0,
        stream_epoch_factory=lambda: "stream-1",
    )
    bridge.begin_resync()
    assert bridge.install_authoritative_snapshot(_snapshot()).status is DirectAckStatus.OK

    assert source_bus.publish(_event(1)).delivered == 1
    assert bridge.wait_until_applied(1, timeout_s=1)

    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert [event.event_id for event in target_seen] == [_event(1).event_id]
    assert bridge.health.retries == 1
    bridge.close()
    source_bus.close()
    target_bus.close()


def test_bridge_overflow_marks_stale_and_snapshot_recovery_opens_a_new_stream():
    target_seen = []
    target_bus, _snapshots, endpoint = _endpoint(target_seen.append)
    source_bus = LocalEventBus()
    stream_epochs = iter(("initial", "stream-1", "stream-2"))
    bridge = DirectRayEventBridge(
        source_bus,
        SubscriptionSpec(
            GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            event_types=(SESSION_OPENED,),
            scope=Scope.DIRECT,
            delivery=DeliveryMode.STATE_SYNC,
            max_queue_events=1,
        ),
        run_id="run-1",
        source_epoch="source-epoch-1",
        send_batch=lambda data: endpoint.receive_events(DirectStateBatch.from_dict(data)).to_dict(),
        install_snapshot=lambda data: endpoint.install_snapshot(SnapshotAndCursor.from_dict(data)).to_dict(),
        stream_epoch_factory=lambda: next(stream_epochs),
    )
    bridge.begin_resync()
    source_bus.publish(_event(1))
    source_bus.publish(_event(2))

    assert bridge.health.status is DirectSyncStatus.STALE
    assert bridge.health.overflow_count == 1
    with pytest.raises(RuntimeError, match="begin_resync"):
        bridge.install_authoritative_snapshot(_snapshot())

    bridge.begin_resync()
    assert bridge.install_authoritative_snapshot(_snapshot()).status is DirectAckStatus.OK
    source_bus.publish(_event(3))
    assert bridge.wait_until_applied(1, timeout_s=1)
    assert [event.event_id for event in target_seen] == [_event(3).event_id]
    assert bridge.health.status is DirectSyncStatus.HEALTHY
    bridge.close()
    source_bus.close()
    target_bus.close()


def test_bridge_never_blocks_publish_while_target_is_busy():
    target_entered = threading.Event()
    release_target = threading.Event()
    source_bus = LocalEventBus()

    def send_batch(_data):
        target_entered.set()
        release_target.wait(timeout=1)
        return {
            "subscription_id": GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            "stream_epoch": "stream-1",
            "contiguous_cursor": 1,
            "stage": "applied",
            "status": "OK",
        }

    bridge = DirectRayEventBridge(
        source_bus,
        SubscriptionSpec(
            GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            event_types=(SESSION_OPENED,),
            scope=Scope.DIRECT,
            delivery=DeliveryMode.STATE_SYNC,
        ),
        run_id="run-1",
        source_epoch="source-epoch-1",
        send_batch=send_batch,
        install_snapshot=lambda _data: {
            "subscription_id": GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            "stream_epoch": "stream-1",
            "contiguous_cursor": 0,
            "stage": "applied",
            "status": "OK",
        },
        stream_epoch_factory=lambda: "stream-1",
    )
    bridge.begin_resync()
    bridge.install_authoritative_snapshot(_snapshot())

    receipt = source_bus.publish(_event(1))

    assert receipt.delivered == 1
    assert target_entered.wait(timeout=1)
    release_target.set()
    bridge.close()
    source_bus.close()


def test_bridge_marks_stream_stale_after_retry_budget_is_exhausted():
    source_bus = LocalEventBus()

    def busy(data):
        batch = DirectStateBatch.from_dict(data)
        return {
            "subscription_id": batch.subscription_id,
            "stream_epoch": batch.stream_epoch,
            "contiguous_cursor": 0,
            "stage": "applied",
            "status": "BUSY",
        }

    bridge = DirectRayEventBridge(
        source_bus,
        SubscriptionSpec(
            GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            event_types=(SESSION_OPENED,),
            scope=Scope.DIRECT,
            delivery=DeliveryMode.STATE_SYNC,
        ),
        run_id="run-1",
        source_epoch="source-epoch-1",
        send_batch=busy,
        install_snapshot=lambda _data: {
            "subscription_id": GATEWAY_SESSION_DIRECT_SUBSCRIPTION,
            "stream_epoch": "stream-1",
            "contiguous_cursor": 0,
            "stage": "applied",
            "status": "OK",
        },
        max_retries=2,
        retry_backoff_s=0,
        stream_epoch_factory=lambda: "stream-1",
    )
    bridge.begin_resync()
    bridge.install_authoritative_snapshot(_snapshot())
    source_bus.publish(_event(1))

    deadline = time.monotonic() + 1
    while bridge.health.status is not DirectSyncStatus.STALE and time.monotonic() < deadline:
        time.sleep(0.001)

    assert bridge.health.status is DirectSyncStatus.STALE
    assert bridge.health.retries == 2
    assert bridge.health.stale_reason == "Direct endpoint remained busy"
    bridge.close()
    source_bus.close()
