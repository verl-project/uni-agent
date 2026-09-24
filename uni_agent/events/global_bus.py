"""Bounded asynchronous transport for cross-process telemetry events."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .bus import DeliveryMode, LocalEventBus, Scope, Subscription, SubscriptionSpec
from .model import Event, EventBatch

logger = logging.getLogger(__name__)

_GLOBAL_SCHEMA_VERSION = 1


class TelemetryAckStatus(str, Enum):
    OK = "OK"
    BUSY = "BUSY"


class GlobalForwarderStatus(str, Enum):
    RUNNING = "running"
    DEGRADED = "degraded"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class TelemetryBatch:
    """One source-owned telemetry batch admitted independently at each hop."""

    run_id: str
    source_id: str
    source_epoch: str
    batch_id: str
    events: tuple[Event, ...]
    schema_version: int = _GLOBAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "source_id", "source_epoch", "batch_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version != _GLOBAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported telemetry batch schema_version {self.schema_version}")
        object.__setattr__(self, "events", tuple(self.events))
        if not self.events:
            raise ValueError("telemetry batch events must be non-empty")
        if any(event.run_id != self.run_id for event in self.events):
            raise ValueError("all telemetry batch events must match run_id")
        if any(event.producer_id != self.source_id for event in self.events):
            raise ValueError("all telemetry batch events must match source_id")
        if any(event.producer_epoch != self.source_epoch for event in self.events):
            raise ValueError("all telemetry batch events must match source_epoch")

    @property
    def estimated_size_bytes(self) -> int:
        return 96 + sum(event.estimated_size_bytes for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "batch_id": self.batch_id,
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TelemetryBatch:
        events = data.get("events")
        if not isinstance(events, list):
            raise TypeError("telemetry batch DTO must contain an events list")
        return cls(
            schema_version=int(data.get("schema_version", _GLOBAL_SCHEMA_VERSION)),
            run_id=str(data["run_id"]),
            source_id=str(data["source_id"]),
            source_epoch=str(data["source_epoch"]),
            batch_id=str(data["batch_id"]),
            events=tuple(Event.from_dict(event) for event in events),
        )


@dataclass(frozen=True, slots=True)
class TelemetryAck:
    batch_id: str
    status: TelemetryAckStatus
    stage: str = "accepted"
    reason: str | None = None
    schema_version: int = _GLOBAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("batch_id must be non-empty")
        object.__setattr__(self, "status", TelemetryAckStatus(self.status))
        if self.stage != "accepted":
            raise ValueError(f"telemetry ACK stage must be 'accepted', got {self.stage!r}")
        if self.schema_version != _GLOBAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported telemetry ACK schema_version {self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "stage": self.stage,
            "status": self.status.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TelemetryAck:
        return cls(
            schema_version=int(data.get("schema_version", _GLOBAL_SCHEMA_VERSION)),
            batch_id=str(data["batch_id"]),
            stage=str(data.get("stage", "accepted")),
            status=TelemetryAckStatus(str(data["status"])),
            reason=str(data["reason"]) if data.get("reason") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class GlobalSubscriptionSpec:
    subscription_id: str
    event_types: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    max_queue_events: int = 4096
    max_queue_bytes: int = 8 * 1024 * 1024
    max_retries: int = 3
    retry_backoff_s: float = 0.01

    def __post_init__(self) -> None:
        if not self.subscription_id:
            raise ValueError("subscription_id must be non-empty")
        object.__setattr__(self, "event_types", tuple(dict.fromkeys(self.event_types)))
        object.__setattr__(self, "run_ids", tuple(dict.fromkeys(self.run_ids)))
        object.__setattr__(self, "source_ids", tuple(dict.fromkeys(self.source_ids)))
        if min(self.max_queue_events, self.max_queue_bytes, self.max_retries) <= 0:
            raise ValueError("Global subscription queue and retry budgets must be positive")
        if self.retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be non-negative")

    def matches(self, event: Event) -> bool:
        return (
            (not self.event_types or event.event_type in self.event_types)
            and (not self.run_ids or event.run_id in self.run_ids)
            and (not self.source_ids or event.producer_id in self.source_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "event_types": list(self.event_types),
            "run_ids": list(self.run_ids),
            "source_ids": list(self.source_ids),
            "max_queue_events": self.max_queue_events,
            "max_queue_bytes": self.max_queue_bytes,
            "max_retries": self.max_retries,
            "retry_backoff_s": self.retry_backoff_s,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GlobalSubscriptionSpec:
        return cls(
            subscription_id=str(data["subscription_id"]),
            event_types=tuple(str(item) for item in data.get("event_types", ())),
            run_ids=tuple(str(item) for item in data.get("run_ids", ())),
            source_ids=tuple(str(item) for item in data.get("source_ids", ())),
            max_queue_events=int(data.get("max_queue_events", 4096)),
            max_queue_bytes=int(data.get("max_queue_bytes", 8 * 1024 * 1024)),
            max_retries=int(data.get("max_retries", 3)),
            retry_backoff_s=float(data.get("retry_backoff_s", 0.01)),
        )


@dataclass(frozen=True, slots=True)
class GlobalForwarderHealth:
    status: GlobalForwarderStatus
    pending_events: int
    pending_bytes: int
    pending_calls: int
    accepted_batches: int
    accepted_events: int
    retries: int
    dropped_events: int
    incomplete_batches: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class GlobalSubscriberHealth:
    subscription_id: str
    pending_batches: int
    pending_events: int
    pending_bytes: int
    pending_calls: int
    accepted_batches: int
    accepted_events: int
    retries: int
    dropped_events: int
    incomplete_batches: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class GlobalBusHealth:
    accepting: bool
    pending_batches: int
    pending_events: int
    pending_bytes: int
    accepted_batches: int
    accepted_events: int
    duplicate_batches: int
    busy_batches: int
    unrouted_events: int
    subscribers: tuple[GlobalSubscriberHealth, ...]


@dataclass(frozen=True, slots=True)
class GlobalEndpointHealth:
    subscription_id: str
    accepting: bool
    pending_batches: int
    pending_events: int
    pending_bytes: int
    accepted_batches: int
    duplicate_batches: int
    busy_batches: int
    delivered_events: int
    incomplete_batches: int
    dropped_events: int
    failed_events: int


@dataclass(frozen=True, slots=True)
class ReporterHealth:
    subscription_id: str
    reported_events: int
    enqueued_events: int
    dropped_events: int
    failed_events: int
    pending_events: int
    pending_bytes: int
    accepting: bool


class _BatchQueue:
    def __init__(self, max_events: int, max_bytes: int) -> None:
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._items: deque[TelemetryBatch] = deque()
        self.events = 0
        self.bytes = 0

    def append(self, batch: TelemetryBatch) -> bool:
        event_count = len(batch.events)
        size_bytes = batch.estimated_size_bytes
        if self.events + event_count > self._max_events or self.bytes + size_bytes > self._max_bytes:
            return False
        self._items.append(batch)
        self.events += event_count
        self.bytes += size_bytes
        return True

    def popleft(self) -> TelemetryBatch:
        batch = self._items.popleft()
        self.events -= len(batch.events)
        self.bytes -= batch.estimated_size_bytes
        return batch

    def clear(self) -> tuple[int, int]:
        batches = len(self._items)
        events = self.events
        self._items.clear()
        self.events = 0
        self.bytes = 0
        return batches, events

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)


def _accepted_ack(batch: TelemetryBatch) -> TelemetryAck:
    return TelemetryAck(batch_id=batch.batch_id, status=TelemetryAckStatus.OK)


def _busy_ack(batch: TelemetryBatch, reason: str) -> TelemetryAck:
    return TelemetryAck(batch_id=batch.batch_id, status=TelemetryAckStatus.BUSY, reason=reason)


def _valid_accepted_ack(batch: TelemetryBatch, ack: TelemetryAck) -> bool:
    return ack.batch_id == batch.batch_id and ack.stage == "accepted" and ack.status is TelemetryAckStatus.OK


class GlobalBusForwarder:
    """Source-side event batching with a bounded outbox and one pending call."""

    def __init__(
        self,
        bus: LocalEventBus,
        spec: SubscriptionSpec,
        *,
        run_id: str,
        source_id: str,
        source_epoch: str,
        send_batch: Callable[[dict[str, Any]], Mapping[str, Any]],
        max_batch_events: int = 128,
        max_batch_bytes: int = 256 * 1024,
        flush_interval_s: float = 0.02,
        max_retries: int = 3,
        retry_backoff_s: float = 0.01,
        batch_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        if spec.scope is not Scope.GLOBAL or spec.delivery is not DeliveryMode.INLINE:
            raise ValueError("Global forwarder requires a global + inline subscription")
        if min(max_batch_events, max_batch_bytes, max_retries) <= 0:
            raise ValueError("Global forwarder batch and retry budgets must be positive")
        if flush_interval_s < 0 or retry_backoff_s < 0:
            raise ValueError("Global forwarder timing values must be non-negative")
        self._spec = spec
        self._run_id = run_id
        self._source_id = source_id
        self._source_epoch = source_epoch
        self._send_batch = send_batch
        self._max_batch_events = max_batch_events
        self._max_batch_bytes = max_batch_bytes
        self._flush_interval_s = flush_interval_s
        self._max_retries = max_retries
        self._retry_backoff_s = retry_backoff_s
        self._batch_id_factory = batch_id_factory
        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[tuple[Event, int, float]] = deque()
        self._pending_bytes = 0
        self._pending_calls = 0
        self._active_batch_events = 0
        self._accepted_batches = 0
        self._accepted_events = 0
        self._retries = 0
        self._dropped_events = 0
        self._incomplete_batches = 0
        self._last_error: str | None = None
        self._accepting = True
        self._stop = False
        self._drain = True
        self._subscription: Subscription = bus.subscribe(spec, self._enqueue)
        self._worker = threading.Thread(
            target=self._run,
            name=f"global-event-forwarder-{spec.subscription_id}",
            daemon=True,
        )
        self._worker.start()

    def _enqueue(self, event: Event) -> None:
        event_size = event.estimated_size_bytes
        with self._condition:
            if (
                not self._accepting
                or event.run_id != self._run_id
                or event.producer_id != self._source_id
                or event.producer_epoch != self._source_epoch
                or event_size + 96 > self._max_batch_bytes
                or len(self._queue) >= self._spec.max_queue_events
                or self._pending_bytes + event_size > self._spec.max_queue_bytes
            ):
                self._dropped_events += 1
                self._incomplete_batches += 1
                self._last_error = "source outbox rejected an event"
                return
            self._queue.append((event, event_size, time.monotonic()))
            self._pending_bytes += event_size
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop and (not self._drain or not self._queue):
                    return
                if not self._ready_locked():
                    deadline = self._queue[0][2] + self._flush_interval_s
                    self._condition.wait(max(deadline - time.monotonic(), 0.0))
                    continue
                batch, selected_count, selected_bytes = self._build_batch_locked()
                self._active_batch_events = selected_count

            accepted, error = self._send_until_accepted(batch)
            with self._condition:
                for _ in range(selected_count):
                    self._queue.popleft()
                self._pending_bytes -= selected_bytes
                if accepted:
                    self._accepted_batches += 1
                    self._accepted_events += selected_count
                    self._last_error = None
                else:
                    self._dropped_events += selected_count
                    self._incomplete_batches += 1
                    self._last_error = error
                self._active_batch_events = 0
                if self._stop and not self._drain and self._queue:
                    self._dropped_events += len(self._queue)
                    self._incomplete_batches += 1
                    self._queue.clear()
                    self._pending_bytes = 0
                self._condition.notify_all()

    def _ready_locked(self) -> bool:
        return (
            self._stop
            or self._flush_interval_s == 0
            or len(self._queue) >= self._max_batch_events
            or self._pending_bytes >= self._max_batch_bytes
            or time.monotonic() >= self._queue[0][2] + self._flush_interval_s
        )

    def _build_batch_locked(self) -> tuple[TelemetryBatch, int, int]:
        events: list[Event] = []
        event_bytes = 0
        for event, size_bytes, _enqueued_at in self._queue:
            if len(events) >= self._max_batch_events:
                break
            if events and event_bytes + size_bytes + 96 > self._max_batch_bytes:
                break
            events.append(event)
            event_bytes += size_bytes
        return (
            TelemetryBatch(
                run_id=self._run_id,
                source_id=self._source_id,
                source_epoch=self._source_epoch,
                batch_id=self._batch_id_factory(),
                events=tuple(events),
            ),
            len(events),
            event_bytes,
        )

    def _send_until_accepted(self, batch: TelemetryBatch) -> tuple[bool, str | None]:
        last_error: str | None = None
        for attempt in range(self._max_retries):
            with self._condition:
                self._pending_calls = 1
            try:
                ack = TelemetryAck.from_dict(self._send_batch(batch.to_dict()))
                if _valid_accepted_ack(batch, ack):
                    return True, None
                last_error = ack.reason or "Global publish endpoint remained busy"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Global batch delivery failed for %s", self._spec.subscription_id)
                last_error = f"Global delivery failed: {type(exc).__name__}"
            finally:
                with self._condition:
                    self._pending_calls = 0
                    self._condition.notify_all()
            if attempt + 1 < self._max_retries:
                with self._condition:
                    self._retries += 1
                    if self._stop and not self._drain:
                        break
                    self._condition.wait(timeout=self._retry_backoff_s)
        return False, last_error

    def wait_until_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            while self._queue or self._pending_calls:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    @property
    def health(self) -> GlobalForwarderHealth:
        with self._condition:
            if not self._accepting:
                status = GlobalForwarderStatus.CLOSED
            elif self._dropped_events or self._incomplete_batches:
                status = GlobalForwarderStatus.DEGRADED
            else:
                status = GlobalForwarderStatus.RUNNING
            return GlobalForwarderHealth(
                status=status,
                pending_events=len(self._queue),
                pending_bytes=self._pending_bytes,
                pending_calls=self._pending_calls,
                accepted_batches=self._accepted_batches,
                accepted_events=self._accepted_events,
                retries=self._retries,
                dropped_events=self._dropped_events,
                incomplete_batches=self._incomplete_batches,
                last_error=self._last_error,
            )

    def close(self, *, drain: bool = True, timeout_s: float = 5.0) -> bool:
        self._subscription.close()
        with self._condition:
            self._accepting = False
            self._stop = True
            self._drain = drain
            if not drain and self._active_batch_events == 0:
                self._dropped_events += len(self._queue)
                if self._queue:
                    self._incomplete_batches += 1
                self._queue.clear()
                self._pending_bytes = 0
            self._condition.notify_all()
        if self._worker is not threading.current_thread():
            self._worker.join(max(timeout_s, 0.0))
        return not self._worker.is_alive() and not self._queue and self._pending_calls == 0


class _BrokerSubscriber:
    def __init__(
        self,
        spec: GlobalSubscriptionSpec,
        send_batch: Callable[[str, dict[str, Any]], Mapping[str, Any]],
    ) -> None:
        self.spec = spec
        self._send_batch = send_batch
        self._condition = threading.Condition(threading.RLock())
        self._queue = _BatchQueue(spec.max_queue_events, spec.max_queue_bytes)
        self._pending_calls = 0
        self._accepted_batches = 0
        self._accepted_events = 0
        self._retries = 0
        self._dropped_events = 0
        self._incomplete_batches = 0
        self._last_error: str | None = None
        self._stop = False
        self._drain = True
        self._worker = threading.Thread(
            target=self._run,
            name=f"global-bus-subscriber-{spec.subscription_id}",
            daemon=True,
        )
        self._worker.start()

    def enqueue(self, batch: TelemetryBatch) -> bool:
        matched = tuple(event for event in batch.events if self.spec.matches(event))
        if not matched:
            return True
        filtered = TelemetryBatch(
            run_id=batch.run_id,
            source_id=batch.source_id,
            source_epoch=batch.source_epoch,
            batch_id=batch.batch_id,
            events=matched,
        )
        with self._condition:
            if self._stop or not self._queue.append(filtered):
                self._dropped_events += len(matched)
                self._incomplete_batches += 1
                self._last_error = "subscriber queue is full"
                return False
            self._condition.notify_all()
            return True

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop and (not self._drain or not self._queue):
                    return
                batch = self._queue.popleft()
                self._pending_calls = 1

            accepted = False
            last_error: str | None = None
            for attempt in range(self.spec.max_retries):
                try:
                    ack = TelemetryAck.from_dict(self._send_batch(self.spec.subscription_id, batch.to_dict()))
                    if _valid_accepted_ack(batch, ack):
                        accepted = True
                        break
                    last_error = ack.reason or "Global subscription endpoint remained busy"
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Global subscriber delivery failed for %s", self.spec.subscription_id)
                    last_error = f"subscriber delivery failed: {type(exc).__name__}"
                if attempt + 1 < self.spec.max_retries:
                    with self._condition:
                        self._retries += 1
                        if self._stop and not self._drain:
                            break
                        self._condition.wait(timeout=self.spec.retry_backoff_s)

            with self._condition:
                self._pending_calls = 0
                if accepted:
                    self._accepted_batches += 1
                    self._accepted_events += len(batch.events)
                    self._last_error = None
                else:
                    self._dropped_events += len(batch.events)
                    self._incomplete_batches += 1
                    self._last_error = last_error
                self._condition.notify_all()

    @property
    def health(self) -> GlobalSubscriberHealth:
        with self._condition:
            return GlobalSubscriberHealth(
                subscription_id=self.spec.subscription_id,
                pending_batches=len(self._queue),
                pending_events=self._queue.events,
                pending_bytes=self._queue.bytes,
                pending_calls=self._pending_calls,
                accepted_batches=self._accepted_batches,
                accepted_events=self._accepted_events,
                retries=self._retries,
                dropped_events=self._dropped_events,
                incomplete_batches=self._incomplete_batches,
                last_error=self._last_error,
            )

    def close(self, *, drain: bool, timeout_s: float) -> bool:
        with self._condition:
            self._stop = True
            self._drain = drain
            if not drain:
                batches, events = self._queue.clear()
                self._dropped_events += events
                self._incomplete_batches += batches
            self._condition.notify_all()
        if self._worker is not threading.current_thread():
            self._worker.join(max(timeout_s, 0.0))
        return not self._worker.is_alive() and not self._queue and self._pending_calls == 0


class RayGlobalEventBus:
    """Thread-safe broker core hosted by the GlobalBus Ray actor."""

    def __init__(
        self,
        *,
        max_queue_events: int = 8192,
        max_queue_bytes: int = 16 * 1024 * 1024,
        max_batch_events: int = 128,
        max_batch_bytes: int = 256 * 1024,
        dedup_entries: int = 8192,
    ) -> None:
        if min(max_queue_events, max_queue_bytes, max_batch_events, max_batch_bytes, dedup_entries) <= 0:
            raise ValueError("GlobalBus budgets must be positive")
        self._max_batch_events = max_batch_events
        self._max_batch_bytes = max_batch_bytes
        self._dedup_entries = dedup_entries
        self._condition = threading.Condition(threading.RLock())
        self._queue = _BatchQueue(max_queue_events, max_queue_bytes)
        self._accepted: OrderedDict[tuple[str, str, str, str], None] = OrderedDict()
        self._subscribers: dict[str, _BrokerSubscriber] = {}
        self._accepting = True
        self._stop = False
        self._drain = True
        self._accepted_batches = 0
        self._accepted_events = 0
        self._duplicate_batches = 0
        self._busy_batches = 0
        self._unrouted_events = 0
        self._active_dispatch_events = 0
        self._active_dispatch_bytes = 0
        self._worker = threading.Thread(target=self._run, name="global-event-bus", daemon=True)
        self._worker.start()

    def register_subscription(
        self,
        spec: GlobalSubscriptionSpec,
        send_batch: Callable[[str, dict[str, Any]], Mapping[str, Any]],
    ) -> None:
        subscriber = _BrokerSubscriber(spec, send_batch)
        with self._condition:
            if not self._accepting:
                subscriber.close(drain=False, timeout_s=0)
                raise RuntimeError("GlobalBus is closed")
            if spec.subscription_id in self._subscribers:
                subscriber.close(drain=False, timeout_s=0)
                raise ValueError(f"duplicate Global subscription_id {spec.subscription_id!r}")
            self._subscribers[spec.subscription_id] = subscriber

    def unregister_subscription(self, subscription_id: str, *, drain: bool = False, timeout_s: float = 5.0) -> bool:
        with self._condition:
            subscriber = self._subscribers.pop(subscription_id, None)
        return subscriber is None or subscriber.close(drain=drain, timeout_s=timeout_s)

    def publish_batch(self, batch: TelemetryBatch) -> TelemetryAck:
        if len(batch.events) > self._max_batch_events or batch.estimated_size_bytes > self._max_batch_bytes:
            with self._condition:
                self._busy_batches += 1
            return _busy_ack(batch, "batch exceeds Global publish endpoint budget")
        key = (batch.run_id, batch.source_id, batch.source_epoch, batch.batch_id)
        with self._condition:
            if key in self._accepted:
                self._duplicate_batches += 1
                self._accepted.move_to_end(key)
                return _accepted_ack(batch)
            if not self._accepting or not self._queue.append(batch):
                self._busy_batches += 1
                return _busy_ack(batch, "GlobalBus ingress queue is full")
            self._accepted[key] = None
            while len(self._accepted) > self._dedup_entries:
                self._accepted.popitem(last=False)
            self._accepted_batches += 1
            self._accepted_events += len(batch.events)
            self._condition.notify_all()
            return _accepted_ack(batch)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop and (not self._drain or not self._queue):
                    return
                batch = self._queue.popleft()
                self._active_dispatch_events = len(batch.events)
                self._active_dispatch_bytes = batch.estimated_size_bytes
                subscribers = tuple(self._subscribers.values())
            matched = False
            for subscriber in subscribers:
                if any(subscriber.spec.matches(event) for event in batch.events):
                    matched = True
                    subscriber.enqueue(batch)
            if not matched:
                with self._condition:
                    self._unrouted_events += len(batch.events)
            with self._condition:
                self._active_dispatch_events = 0
                self._active_dispatch_bytes = 0
                self._condition.notify_all()

    @property
    def health(self) -> GlobalBusHealth:
        with self._condition:
            subscribers = tuple(item.health for item in self._subscribers.values())
            return GlobalBusHealth(
                accepting=self._accepting,
                pending_batches=len(self._queue) + int(self._active_dispatch_events > 0),
                pending_events=self._queue.events + self._active_dispatch_events,
                pending_bytes=self._queue.bytes + self._active_dispatch_bytes,
                accepted_batches=self._accepted_batches,
                accepted_events=self._accepted_events,
                duplicate_batches=self._duplicate_batches,
                busy_batches=self._busy_batches,
                unrouted_events=self._unrouted_events,
                subscribers=subscribers,
            )

    def wait_until_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while True:
            health = self.health
            if health.pending_batches == 0 and all(
                item.pending_batches == 0 and item.pending_calls == 0 for item in health.subscribers
            ):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.005, max(deadline - time.monotonic(), 0.0)))

    def close(self, *, drain: bool = True, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            self._accepting = False
            self._stop = True
            self._drain = drain
            if not drain:
                self._queue.clear()
            self._condition.notify_all()
        if self._worker is not threading.current_thread():
            self._worker.join(max(deadline - time.monotonic(), 0.0))
        with self._condition:
            subscribers = tuple(self._subscribers.values())
            self._subscribers.clear()
        complete = not self._worker.is_alive() and not self._queue
        for subscriber in subscribers:
            complete = (
                subscriber.close(
                    drain=drain,
                    timeout_s=max(deadline - time.monotonic(), 0.0),
                )
                and complete
            )
        return complete


class GlobalSubscriptionEndpoint:
    """Accept remote batches into a bounded queue before named local delivery."""

    def __init__(
        self,
        bus: LocalEventBus,
        *,
        subscription_id: str,
        max_queue_events: int = 4096,
        max_queue_bytes: int = 8 * 1024 * 1024,
        max_batch_events: int = 128,
        max_batch_bytes: int = 256 * 1024,
        dedup_entries: int = 8192,
    ) -> None:
        if not subscription_id:
            raise ValueError("subscription_id must be non-empty")
        if min(max_queue_events, max_queue_bytes, max_batch_events, max_batch_bytes, dedup_entries) <= 0:
            raise ValueError("Global subscription endpoint budgets must be positive")
        if not bus.deliver(subscription_id, EventBatch(())).subscription_found:
            raise ValueError(f"Global target subscription {subscription_id!r} is not installed")
        self._bus = bus
        self._subscription_id = subscription_id
        self._max_batch_events = max_batch_events
        self._max_batch_bytes = max_batch_bytes
        self._dedup_entries = dedup_entries
        self._condition = threading.Condition(threading.RLock())
        self._queue = _BatchQueue(max_queue_events, max_queue_bytes)
        self._accepted: OrderedDict[tuple[str, str, str, str], None] = OrderedDict()
        self._accepting = True
        self._stop = False
        self._drain = True
        self._accepted_batches = 0
        self._duplicate_batches = 0
        self._busy_batches = 0
        self._delivered_events = 0
        self._incomplete_batches = 0
        self._dropped_events = 0
        self._failed_events = 0
        self._active_events = 0
        self._active_bytes = 0
        self._worker = threading.Thread(
            target=self._run,
            name=f"global-subscription-endpoint-{subscription_id}",
            daemon=True,
        )
        self._worker.start()

    def receive_batch(self, subscription_id: str, batch: TelemetryBatch) -> TelemetryAck:
        if subscription_id != self._subscription_id:
            with self._condition:
                self._busy_batches += 1
            return _busy_ack(batch, "unknown Global subscription")
        if len(batch.events) > self._max_batch_events or batch.estimated_size_bytes > self._max_batch_bytes:
            with self._condition:
                self._busy_batches += 1
            return _busy_ack(batch, "batch exceeds Global subscription endpoint budget")
        key = (batch.run_id, batch.source_id, batch.source_epoch, batch.batch_id)
        with self._condition:
            if key in self._accepted:
                self._duplicate_batches += 1
                self._accepted.move_to_end(key)
                return _accepted_ack(batch)
            if not self._accepting or not self._queue.append(batch):
                self._busy_batches += 1
                return _busy_ack(batch, "Global subscription endpoint queue is full")
            self._accepted[key] = None
            while len(self._accepted) > self._dedup_entries:
                self._accepted.popitem(last=False)
            self._accepted_batches += 1
            self._condition.notify_all()
            return _accepted_ack(batch)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop and (not self._drain or not self._queue):
                    return
                batch = self._queue.popleft()
                self._active_events = len(batch.events)
                self._active_bytes = batch.estimated_size_bytes
            delivery = self._bus.deliver(self._subscription_id, EventBatch(batch.events))
            with self._condition:
                self._delivered_events += delivery.delivered + delivery.enqueued
                self._dropped_events += delivery.dropped
                self._failed_events += delivery.failed
                if not delivery.complete or delivery.accepted != len(batch.events):
                    self._incomplete_batches += 1
                self._active_events = 0
                self._active_bytes = 0
                self._condition.notify_all()

    @property
    def health(self) -> GlobalEndpointHealth:
        with self._condition:
            return GlobalEndpointHealth(
                subscription_id=self._subscription_id,
                accepting=self._accepting,
                pending_batches=len(self._queue) + int(self._active_events > 0),
                pending_events=self._queue.events + self._active_events,
                pending_bytes=self._queue.bytes + self._active_bytes,
                accepted_batches=self._accepted_batches,
                duplicate_batches=self._duplicate_batches,
                busy_batches=self._busy_batches,
                delivered_events=self._delivered_events,
                incomplete_batches=self._incomplete_batches,
                dropped_events=self._dropped_events,
                failed_events=self._failed_events,
            )

    def wait_until_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            while self._queue or self._active_events:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, drain: bool = True, timeout_s: float = 5.0) -> bool:
        with self._condition:
            self._accepting = False
            self._stop = True
            self._drain = drain
            if not drain:
                batches, events = self._queue.clear()
                self._incomplete_batches += batches
                self._dropped_events += events
            self._condition.notify_all()
        if self._worker is not threading.current_thread():
            self._worker.join(max(timeout_s, 0.0))
        return not self._worker.is_alive() and not self._queue and self._active_events == 0


class TelemetryReporter:
    """Queued local Reporter adapter with health derived from its subscription."""

    def __init__(self, bus: LocalEventBus, spec: SubscriptionSpec, report: Callable[[Event], None]) -> None:
        if spec.scope is not Scope.GLOBAL or spec.delivery is not DeliveryMode.QUEUED:
            raise ValueError("Telemetry Reporter requires a global + queued subscription")
        self._subscription = bus.subscribe(spec, report)

    @property
    def health(self) -> ReporterHealth:
        stats = self._subscription.stats
        return ReporterHealth(
            subscription_id=stats.subscription_id,
            reported_events=stats.delivered,
            enqueued_events=stats.enqueued,
            dropped_events=stats.dropped,
            failed_events=stats.failed,
            pending_events=stats.pending_events,
            pending_bytes=stats.pending_bytes,
            accepting=stats.accepting,
        )

    def close(self, *, drain: bool = True, timeout_s: float = 5.0) -> bool:
        return self._subscription.close(drain=drain, timeout_s=timeout_s)


__all__ = [
    "GlobalBusForwarder",
    "GlobalBusHealth",
    "GlobalEndpointHealth",
    "GlobalForwarderHealth",
    "GlobalForwarderStatus",
    "GlobalSubscriberHealth",
    "GlobalSubscriptionEndpoint",
    "GlobalSubscriptionSpec",
    "RayGlobalEventBus",
    "ReporterHealth",
    "TelemetryAck",
    "TelemetryAckStatus",
    "TelemetryBatch",
    "TelemetryReporter",
]
