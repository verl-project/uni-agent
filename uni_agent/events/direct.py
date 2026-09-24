"""Recoverable one-hop delivery for control-state event subscriptions."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .bus import DeliveryMode, LocalEventBus, Scope, Subscription, SubscriptionSpec
from .model import Event, EventBatch, _estimate_value_bytes, _freeze_value, _thaw_value

logger = logging.getLogger(__name__)

_DIRECT_SCHEMA_VERSION = 1


class DirectAckStatus(str, Enum):
    OK = "OK"
    BUSY = "BUSY"
    RESYNC_REQUIRED = "RESYNC_REQUIRED"


class DirectSyncStatus(str, Enum):
    SYNCING = "syncing"
    HEALTHY = "healthy"
    STALE = "stale"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class DirectStateBatch:
    """One contiguous batch in a single Direct subscription stream."""

    run_id: str
    subscription_id: str
    source_epoch: str
    stream_epoch: str
    first_seq: int
    last_seq: int
    events: tuple[Event, ...]
    schema_version: int = _DIRECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "subscription_id", "source_epoch", "stream_epoch"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version != _DIRECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported Direct batch schema_version {self.schema_version}")
        object.__setattr__(self, "events", tuple(self.events))
        if not self.events:
            raise ValueError("events must be non-empty")
        if self.first_seq <= 0:
            raise ValueError(f"first_seq must be positive, got {self.first_seq}")
        if self.last_seq < self.first_seq:
            raise ValueError("last_seq must be greater than or equal to first_seq")
        if self.last_seq - self.first_seq + 1 != len(self.events):
            raise ValueError("Direct batch sequence range must match its event count")
        if any(event.run_id != self.run_id for event in self.events):
            raise ValueError("all Direct batch events must match run_id")
        if any(event.producer_epoch != self.source_epoch for event in self.events):
            raise ValueError("all Direct batch events must match source_epoch")

    @property
    def estimated_size_bytes(self) -> int:
        return 96 + sum(event.estimated_size_bytes for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "subscription_id": self.subscription_id,
            "source_epoch": self.source_epoch,
            "stream_epoch": self.stream_epoch,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DirectStateBatch:
        events = data.get("events")
        if not isinstance(events, list):
            raise TypeError("Direct batch DTO must contain an events list")
        return cls(
            schema_version=int(data.get("schema_version", _DIRECT_SCHEMA_VERSION)),
            run_id=str(data["run_id"]),
            subscription_id=str(data["subscription_id"]),
            source_epoch=str(data["source_epoch"]),
            stream_epoch=str(data["stream_epoch"]),
            first_seq=int(data["first_seq"]),
            last_seq=int(data["last_seq"]),
            events=tuple(Event.from_dict(event) for event in events),
        )


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Authoritative state captured with a Direct stream watermark."""

    owner: str
    source_epoch: str
    scope: str
    revision: int
    watermark: int
    source_health: str
    current_entities: Mapping[str, Any]
    terminal_entities: Mapping[str, Any]
    schema_version: int = _DIRECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("owner", "source_epoch", "scope", "source_health"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version != _DIRECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot schema_version {self.schema_version}")
        if self.revision < 0:
            raise ValueError(f"revision must be non-negative, got {self.revision}")
        if self.watermark < 0:
            raise ValueError(f"watermark must be non-negative, got {self.watermark}")
        object.__setattr__(
            self,
            "current_entities",
            _freeze_value(self.current_entities, path="current_entities"),
        )
        object.__setattr__(
            self,
            "terminal_entities",
            _freeze_value(self.terminal_entities, path="terminal_entities"),
        )

    @property
    def estimated_size_bytes(self) -> int:
        return 128 + _estimate_value_bytes(self.current_entities) + _estimate_value_bytes(self.terminal_entities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner": self.owner,
            "source_epoch": self.source_epoch,
            "scope": self.scope,
            "revision": self.revision,
            "watermark": self.watermark,
            "source_health": self.source_health,
            "current_entities": _thaw_value(self.current_entities),
            "terminal_entities": _thaw_value(self.terminal_entities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateSnapshot:
        current_entities = data.get("current_entities")
        terminal_entities = data.get("terminal_entities")
        if not isinstance(current_entities, Mapping) or not isinstance(terminal_entities, Mapping):
            raise TypeError("snapshot entities must be mappings")
        return cls(
            schema_version=int(data.get("schema_version", _DIRECT_SCHEMA_VERSION)),
            owner=str(data["owner"]),
            source_epoch=str(data["source_epoch"]),
            scope=str(data["scope"]),
            revision=int(data["revision"]),
            watermark=int(data["watermark"]),
            source_health=str(data["source_health"]),
            current_entities=current_entities,
            terminal_entities=terminal_entities,
        )


@dataclass(frozen=True, slots=True)
class SnapshotAndCursor:
    """Snapshot installation request bound to one new stream epoch."""

    subscription_id: str
    stream_epoch: str
    snapshot: StateSnapshot
    schema_version: int = _DIRECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.subscription_id:
            raise ValueError("subscription_id must be non-empty")
        if not self.stream_epoch:
            raise ValueError("stream_epoch must be non-empty")
        if self.schema_version != _DIRECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported snapshot envelope schema_version {self.schema_version}")
        if not isinstance(self.snapshot, StateSnapshot):
            raise TypeError("snapshot must be StateSnapshot")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subscription_id": self.subscription_id,
            "stream_epoch": self.stream_epoch,
            "snapshot": self.snapshot.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SnapshotAndCursor:
        snapshot = data.get("snapshot")
        if not isinstance(snapshot, Mapping):
            raise TypeError("snapshot envelope DTO must contain a snapshot mapping")
        return cls(
            schema_version=int(data.get("schema_version", _DIRECT_SCHEMA_VERSION)),
            subscription_id=str(data["subscription_id"]),
            stream_epoch=str(data["stream_epoch"]),
            snapshot=StateSnapshot.from_dict(snapshot),
        )


@dataclass(frozen=True, slots=True)
class DirectAck:
    subscription_id: str
    stream_epoch: str
    contiguous_cursor: int
    status: DirectAckStatus
    stage: str = "applied"
    reason: str | None = None
    schema_version: int = _DIRECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.subscription_id or not self.stream_epoch:
            raise ValueError("subscription_id and stream_epoch must be non-empty")
        if self.contiguous_cursor < 0:
            raise ValueError("contiguous_cursor must be non-negative")
        object.__setattr__(self, "status", DirectAckStatus(self.status))
        if self.stage != "applied":
            raise ValueError(f"Direct ACK stage must be 'applied', got {self.stage!r}")
        if self.schema_version != _DIRECT_SCHEMA_VERSION:
            raise ValueError(f"unsupported Direct ACK schema_version {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subscription_id": self.subscription_id,
            "stream_epoch": self.stream_epoch,
            "contiguous_cursor": self.contiguous_cursor,
            "stage": self.stage,
            "status": self.status.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DirectAck:
        return cls(
            schema_version=int(data.get("schema_version", _DIRECT_SCHEMA_VERSION)),
            subscription_id=str(data["subscription_id"]),
            stream_epoch=str(data["stream_epoch"]),
            contiguous_cursor=int(data["contiguous_cursor"]),
            stage=str(data.get("stage", "applied")),
            status=DirectAckStatus(str(data["status"])),
            reason=str(data["reason"]) if data.get("reason") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class DirectBridgeHealth:
    status: DirectSyncStatus
    stream_epoch: str
    contiguous_cursor: int
    last_assigned_seq: int
    pending_events: int
    pending_bytes: int
    retries: int
    resyncs: int
    overflow_count: int
    stale_reason: str | None


@dataclass(slots=True)
class _EndpointStream:
    owner: str
    source_epoch: str
    stream_epoch: str
    cursor: int
    status: DirectSyncStatus
    revision: int


class DirectEventEndpoint:
    """Validate Direct transport state and commit batches through local ingress."""

    def __init__(
        self,
        bus: LocalEventBus,
        *,
        snapshot_installer: Callable[[StateSnapshot], None],
        max_batch_events: int = 128,
        max_batch_bytes: int = 256 * 1024,
        max_snapshot_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if min(max_batch_events, max_batch_bytes, max_snapshot_bytes) <= 0:
            raise ValueError("Direct endpoint budgets must be positive")
        self._bus = bus
        self._snapshot_installer = snapshot_installer
        self._max_batch_events = max_batch_events
        self._max_batch_bytes = max_batch_bytes
        self._max_snapshot_bytes = max_snapshot_bytes
        self._lock = threading.RLock()
        self._streams: dict[tuple[str, str], _EndpointStream] = {}
        self._owners: dict[tuple[str, str], str] = {}

    def install_snapshot(self, request: SnapshotAndCursor) -> DirectAck:
        snapshot = request.snapshot
        if not self._bus.deliver(request.subscription_id, EventBatch(())).subscription_found:
            return self._ack(
                request.subscription_id,
                request.stream_epoch,
                snapshot.watermark,
                DirectAckStatus.RESYNC_REQUIRED,
                "target subscription is not installed",
            )
        if snapshot.source_health != "healthy":
            return self._ack(
                request.subscription_id,
                request.stream_epoch,
                snapshot.watermark,
                DirectAckStatus.RESYNC_REQUIRED,
                f"source is not healthy: {snapshot.source_health}",
            )
        if snapshot.estimated_size_bytes > self._max_snapshot_bytes:
            return self._ack(
                request.subscription_id,
                request.stream_epoch,
                snapshot.watermark,
                DirectAckStatus.BUSY,
                "snapshot exceeds endpoint byte budget",
            )
        try:
            with self._lock:
                previous = self._streams.get((request.subscription_id, snapshot.source_epoch))
                if previous is not None and snapshot.revision < previous.revision:
                    return self._ack(
                        request.subscription_id,
                        request.stream_epoch,
                        previous.cursor,
                        DirectAckStatus.RESYNC_REQUIRED,
                        "snapshot revision moved backwards within one source epoch",
                    )
                self._snapshot_installer(snapshot)
                owner_key = (request.subscription_id, snapshot.owner)
                previous_epoch = self._owners.get(owner_key)
                if previous_epoch is not None:
                    self._streams.pop((request.subscription_id, previous_epoch), None)
                self._owners[owner_key] = snapshot.source_epoch
                self._streams[(request.subscription_id, snapshot.source_epoch)] = _EndpointStream(
                    owner=snapshot.owner,
                    source_epoch=snapshot.source_epoch,
                    stream_epoch=request.stream_epoch,
                    cursor=snapshot.watermark,
                    status=DirectSyncStatus.HEALTHY,
                    revision=snapshot.revision,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Direct snapshot installation failed for %s", request.subscription_id)
            return self._ack(
                request.subscription_id,
                request.stream_epoch,
                0,
                DirectAckStatus.BUSY,
                f"snapshot installation failed: {type(exc).__name__}",
            )
        return self._ack(
            request.subscription_id,
            request.stream_epoch,
            snapshot.watermark,
            DirectAckStatus.OK,
        )

    def receive_events(self, batch: DirectStateBatch) -> DirectAck:
        if len(batch.events) > self._max_batch_events or batch.estimated_size_bytes > self._max_batch_bytes:
            return self._ack(
                batch.subscription_id,
                batch.stream_epoch,
                0,
                DirectAckStatus.BUSY,
                "batch exceeds endpoint budget",
            )

        key = (batch.subscription_id, batch.source_epoch)
        with self._lock:
            stream = self._streams.get(key)
            if stream is None or stream.stream_epoch != batch.stream_epoch:
                return self._ack(
                    batch.subscription_id,
                    batch.stream_epoch,
                    0,
                    DirectAckStatus.RESYNC_REQUIRED,
                    "unknown source or stream epoch",
                )
            if stream.status is not DirectSyncStatus.HEALTHY:
                return self._ack(
                    batch.subscription_id,
                    batch.stream_epoch,
                    stream.cursor,
                    DirectAckStatus.RESYNC_REQUIRED,
                    "stream is stale",
                )
            if any(event.producer_id != stream.owner for event in batch.events):
                return self._mark_stale(stream, batch, "batch owner mismatch")
            if batch.last_seq <= stream.cursor:
                return self._ack(
                    batch.subscription_id,
                    batch.stream_epoch,
                    stream.cursor,
                    DirectAckStatus.OK,
                )
            if batch.first_seq > stream.cursor + 1:
                return self._mark_stale(stream, batch, "sequence gap")

            skip = max(stream.cursor - batch.first_seq + 1, 0)
            remaining = batch.events[skip:]
            delivery = self._bus.deliver(batch.subscription_id, EventBatch(remaining))
            if not delivery.complete or delivery.delivered != len(remaining):
                return self._mark_stale(stream, batch, "state-sync subscriber did not apply the full batch")
            stream.cursor = batch.last_seq
            for event in remaining:
                source_revision = event.payload.get("source_revision")
                if isinstance(source_revision, int) and not isinstance(source_revision, bool):
                    stream.revision = max(stream.revision, source_revision)
            return self._ack(
                batch.subscription_id,
                batch.stream_epoch,
                stream.cursor,
                DirectAckStatus.OK,
            )

    def status(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(
                {
                    "subscription_id": subscription_id,
                    "owner": stream.owner,
                    "source_epoch": stream.source_epoch,
                    "stream_epoch": stream.stream_epoch,
                    "cursor": stream.cursor,
                    "status": stream.status.value,
                    "revision": stream.revision,
                }
                for (subscription_id, _source_epoch), stream in self._streams.items()
            )

    @staticmethod
    def _ack(
        subscription_id: str,
        stream_epoch: str,
        cursor: int,
        status: DirectAckStatus,
        reason: str | None = None,
    ) -> DirectAck:
        return DirectAck(
            subscription_id=subscription_id,
            stream_epoch=stream_epoch,
            contiguous_cursor=cursor,
            status=status,
            reason=reason,
        )

    def _mark_stale(self, stream: _EndpointStream, batch: DirectStateBatch, reason: str) -> DirectAck:
        stream.status = DirectSyncStatus.STALE
        return self._ack(
            batch.subscription_id,
            batch.stream_epoch,
            stream.cursor,
            DirectAckStatus.RESYNC_REQUIRED,
            reason,
        )


class DirectRayEventBridge:
    """Source-side bounded outbox with applied-ACK retry and resync state."""

    def __init__(
        self,
        bus: LocalEventBus,
        spec: SubscriptionSpec,
        *,
        run_id: str,
        source_epoch: str,
        send_batch: Callable[[dict[str, Any]], Mapping[str, Any]],
        install_snapshot: Callable[[dict[str, Any]], Mapping[str, Any]],
        max_batch_events: int = 32,
        max_batch_bytes: int = 64 * 1024,
        max_retries: int = 3,
        retry_backoff_s: float = 0.01,
        stream_epoch_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        if spec.scope is not Scope.DIRECT or spec.delivery is not DeliveryMode.STATE_SYNC:
            raise ValueError("Direct bridge requires a direct + state_sync subscription")
        if min(max_batch_events, max_batch_bytes, max_retries) <= 0:
            raise ValueError("Direct bridge batch and retry budgets must be positive")
        if retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be non-negative")
        self._spec = spec
        self._run_id = run_id
        self._source_epoch = source_epoch
        self._send_batch = send_batch
        self._install_snapshot = install_snapshot
        self._max_batch_events = max_batch_events
        self._max_batch_bytes = max_batch_bytes
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._stream_epoch_factory = stream_epoch_factory
        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[tuple[int, Event, int]] = deque()
        self._pending_bytes = 0
        self._stream_epoch = self._stream_epoch_factory()
        self._next_seq = 1
        self._cursor = 0
        self._snapshot_watermark = 0
        self._status = DirectSyncStatus.SYNCING
        self._stale_reason: str | None = None
        self._retries = 0
        self._consecutive_retries = 0
        self._resyncs = 0
        self._overflow_count = 0
        self._stop = False
        self._worker: threading.Thread | None = None
        self._subscription: Subscription = bus.subscribe(spec, self._enqueue)

    def begin_resync(self) -> tuple[str, int]:
        """Open a fresh stream before the owner captures its authoritative snapshot."""
        with self._condition:
            self._stream_epoch = self._stream_epoch_factory()
            self._queue.clear()
            self._pending_bytes = 0
            self._next_seq = 1
            self._cursor = 0
            self._snapshot_watermark = 0
            self._status = DirectSyncStatus.SYNCING
            self._stale_reason = None
            self._consecutive_retries = 0
            self._resyncs += 1
            return self._stream_epoch, self._cursor

    def install_authoritative_snapshot(self, snapshot: StateSnapshot) -> DirectAck:
        with self._condition:
            if self._status is DirectSyncStatus.CLOSED:
                raise RuntimeError("Direct bridge is closed")
            if self._status is not DirectSyncStatus.SYNCING:
                raise RuntimeError("begin_resync() must be called before installing a snapshot")
            stream_epoch = self._stream_epoch
            expected_watermark = self._snapshot_watermark
        if snapshot.source_epoch != self._source_epoch:
            raise ValueError("snapshot source_epoch must match the Direct bridge source")
        if snapshot.watermark != expected_watermark:
            raise ValueError(
                f"snapshot watermark {snapshot.watermark} does not match assigned cursor {expected_watermark}"
            )
        request = SnapshotAndCursor(
            subscription_id=self._spec.subscription_id,
            stream_epoch=stream_epoch,
            snapshot=snapshot,
        )
        ack = DirectAck.from_dict(self._install_snapshot(request.to_dict()))
        with self._condition:
            if ack.stream_epoch != self._stream_epoch:
                return ack
            if (
                ack.subscription_id == self._spec.subscription_id
                and ack.status is DirectAckStatus.OK
                and ack.contiguous_cursor == snapshot.watermark
            ):
                self._cursor = ack.contiguous_cursor
                self._status = DirectSyncStatus.HEALTHY
                self._stale_reason = None
                self._consecutive_retries = 0
                self._ensure_worker_locked()
                self._condition.notify_all()
            else:
                self._mark_stale_locked(ack.reason or "snapshot was not applied")
        return ack

    def _enqueue(self, event: Event) -> None:
        if event.run_id != self._run_id or event.producer_epoch != self._source_epoch:
            raise ValueError("Direct bridge received an event from another source stream")
        event_size = event.estimated_size_bytes
        with self._condition:
            if self._status in {DirectSyncStatus.STALE, DirectSyncStatus.CLOSED}:
                return
            if (
                event_size + 96 > self._max_batch_bytes
                or len(self._queue) >= self._spec.max_queue_events
                or (self._pending_bytes + event_size > self._spec.max_queue_bytes)
            ):
                self._overflow_count += 1
                self._mark_stale_locked("outbox overflow")
                return
            sequence = self._next_seq
            self._next_seq += 1
            self._queue.append((sequence, event, event_size))
            self._pending_bytes += event_size
            self._condition.notify_all()

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._run,
            name=f"direct-event-bridge-{self._spec.subscription_id}",
            daemon=True,
        )
        self._worker.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stop and (self._status is not DirectSyncStatus.HEALTHY or not self._queue):
                    self._condition.wait()
                if self._stop:
                    return
                batch = self._build_batch_locked()
            try:
                ack = DirectAck.from_dict(self._send_batch(batch.to_dict()))
            except Exception:  # noqa: BLE001
                logger.exception("Direct batch delivery failed for %s", self._spec.subscription_id)
                with self._condition:
                    if self._record_retry_locked("Direct delivery retry budget exhausted"):
                        continue
                self._wait_before_retry()
                continue

            with self._condition:
                if batch.stream_epoch != self._stream_epoch:
                    continue
                if ack.status is DirectAckStatus.OK:
                    if (
                        ack.subscription_id != self._spec.subscription_id
                        or ack.stream_epoch != self._stream_epoch
                        or ack.contiguous_cursor < batch.last_seq
                    ):
                        self._mark_stale_locked("invalid applied ACK cursor")
                        continue
                    self._discard_applied_locked(ack.contiguous_cursor)
                    self._cursor = ack.contiguous_cursor
                    self._consecutive_retries = 0
                    self._condition.notify_all()
                    continue
                if ack.status is DirectAckStatus.RESYNC_REQUIRED:
                    self._mark_stale_locked(ack.reason or "endpoint requested resync")
                    continue
                if self._record_retry_locked("Direct endpoint remained busy"):
                    continue
            self._wait_before_retry()

    def _build_batch_locked(self) -> DirectStateBatch:
        selected: list[tuple[int, Event, int]] = []
        selected_bytes = 96
        for item in self._queue:
            if len(selected) >= self._max_batch_events:
                break
            next_bytes = selected_bytes + item[2]
            if selected and next_bytes > self._max_batch_bytes:
                break
            selected.append(item)
            selected_bytes = next_bytes
        first_seq = selected[0][0]
        return DirectStateBatch(
            run_id=self._run_id,
            subscription_id=self._spec.subscription_id,
            source_epoch=self._source_epoch,
            stream_epoch=self._stream_epoch,
            first_seq=first_seq,
            last_seq=selected[-1][0],
            events=tuple(item[1] for item in selected),
        )

    def _discard_applied_locked(self, cursor: int) -> None:
        while self._queue and self._queue[0][0] <= cursor:
            _sequence, _event, size = self._queue.popleft()
            self._pending_bytes -= size

    def _mark_stale_locked(self, reason: str) -> None:
        self._queue.clear()
        self._pending_bytes = 0
        self._status = DirectSyncStatus.STALE
        self._stale_reason = reason
        self._condition.notify_all()

    def _record_retry_locked(self, exhausted_reason: str) -> bool:
        self._retries += 1
        self._consecutive_retries += 1
        if self._consecutive_retries < self._max_retries:
            return False
        self._mark_stale_locked(exhausted_reason)
        return True

    def _wait_before_retry(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._stop, timeout=self._retry_backoff_s)

    def wait_until_applied(self, cursor: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            while self._cursor < cursor and self._status not in {
                DirectSyncStatus.STALE,
                DirectSyncStatus.CLOSED,
            }:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return self._cursor >= cursor

    @property
    def health(self) -> DirectBridgeHealth:
        with self._condition:
            return DirectBridgeHealth(
                status=self._status,
                stream_epoch=self._stream_epoch,
                contiguous_cursor=self._cursor,
                last_assigned_seq=self._next_seq - 1,
                pending_events=len(self._queue),
                pending_bytes=self._pending_bytes,
                retries=self._retries,
                resyncs=self._resyncs,
                overflow_count=self._overflow_count,
                stale_reason=self._stale_reason,
            )

    def close(self, *, timeout_s: float = 5.0) -> None:
        self._subscription.close()
        with self._condition:
            self._stop = True
            self._status = DirectSyncStatus.CLOSED
            self._condition.notify_all()
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(max(timeout_s, 0.0))

    def __enter__(self) -> DirectRayEventBridge:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        self.close()
        return False
