"""Thread-safe in-process publish/subscribe bus with isolated bounded delivery."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .model import Event, EventBatch

logger = logging.getLogger(__name__)

EventHandler = Callable[[Event], None]


class Scope(str, Enum):
    LOCAL = "local"
    DIRECT = "direct"
    GLOBAL = "global"


class DeliveryMode(str, Enum):
    INLINE = "inline"
    QUEUED = "queued"
    STATE_SYNC = "state_sync"


@dataclass(frozen=True, slots=True)
class SubscriptionSpec:
    """Selection, scope, and queue budget for one local subscriber."""

    subscription_id: str
    event_types: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    producer_ids: tuple[str, ...] = ()
    scope: Scope = Scope.GLOBAL
    delivery: DeliveryMode = DeliveryMode.QUEUED
    max_queue_events: int = 1024
    max_queue_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if not self.subscription_id:
            raise ValueError("subscription_id must be non-empty")
        object.__setattr__(self, "event_types", tuple(dict.fromkeys(self.event_types)))
        object.__setattr__(self, "run_ids", tuple(dict.fromkeys(self.run_ids)))
        object.__setattr__(self, "producer_ids", tuple(dict.fromkeys(self.producer_ids)))
        object.__setattr__(self, "scope", Scope(self.scope))
        object.__setattr__(self, "delivery", DeliveryMode(self.delivery))
        if self.max_queue_events <= 0:
            raise ValueError(f"max_queue_events must be positive, got {self.max_queue_events}")
        if self.max_queue_bytes <= 0:
            raise ValueError(f"max_queue_bytes must be positive, got {self.max_queue_bytes}")

    def matches(self, event: Event) -> bool:
        return (
            (not self.event_types or event.event_type in self.event_types)
            and (not self.run_ids or event.run_id in self.run_ids)
            and (not self.producer_ids or event.producer_id in self.producer_ids)
        )


@dataclass(frozen=True, slots=True)
class PublishReceipt:
    event_id: str
    producer_seq: int
    matched_subscriptions: int = 0
    delivered: int = 0
    enqueued: int = 0
    dropped: int = 0
    failed: int = 0

    @property
    def complete(self) -> bool:
        """Whether all immediate local dispatch work was accepted."""
        return self.dropped == 0 and self.failed == 0


@dataclass(frozen=True, slots=True)
class DeliveryAck:
    subscription_id: str
    subscription_found: bool
    requested_events: int
    delivered: int = 0
    enqueued: int = 0
    dropped: int = 0
    failed: int = 0

    @property
    def accepted(self) -> int:
        return self.delivered + self.enqueued

    @property
    def complete(self) -> bool:
        return self.subscription_found and self.dropped == 0 and self.failed == 0


@dataclass(frozen=True, slots=True)
class SubscriptionStats:
    subscription_id: str
    dispatched: int
    delivered: int
    enqueued: int
    dropped: int
    failed: int
    pending_events: int
    pending_bytes: int
    accepting: bool

    @property
    def complete(self) -> bool:
        return self.dropped == 0 and self.failed == 0


@dataclass(frozen=True, slots=True)
class FlushReport:
    complete: bool
    pending_events: int
    incomplete_subscriptions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DispatchResult:
    delivered: int = 0
    enqueued: int = 0
    dropped: int = 0
    failed: int = 0


def _is_async_handler(handler: EventHandler) -> bool:
    return inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(handler.__call__)


class _SubscriptionState:
    def __init__(self, spec: SubscriptionSpec, handler: EventHandler):
        self.spec = spec
        self.handler = handler
        self._condition = threading.Condition(threading.RLock())
        self._queue: deque[tuple[Event, int]] = deque()
        self._pending_bytes = 0
        self._active_bytes = 0
        self._active_handlers = 0
        self._accepting = True
        self._stop = False
        self._dispatched = 0
        self._delivered = 0
        self._enqueued = 0
        self._dropped = 0
        self._failed = 0
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if self.spec.delivery is not DeliveryMode.QUEUED:
            return
        self._worker = threading.Thread(
            target=self._run,
            name=f"event-subscription-{self.spec.subscription_id}",
            daemon=True,
        )
        self._worker.start()

    def dispatch(self, event: Event) -> _DispatchResult:
        if self.spec.delivery is DeliveryMode.QUEUED:
            return self._enqueue(event)
        return self._deliver_inline(event)

    def _deliver_inline(self, event: Event) -> _DispatchResult:
        with self._condition:
            self._dispatched += 1
            if not self._accepting:
                self._dropped += 1
                return _DispatchResult(dropped=1)
            self._active_handlers += 1
        try:
            self.handler(event)
        except Exception:
            logger.exception(
                "event subscriber %s failed for event %s",
                self.spec.subscription_id,
                event.event_id,
            )
            with self._condition:
                self._failed += 1
            return _DispatchResult(failed=1)
        else:
            with self._condition:
                self._delivered += 1
            return _DispatchResult(delivered=1)
        finally:
            with self._condition:
                self._active_handlers -= 1
                self._condition.notify_all()

    def _enqueue(self, event: Event) -> _DispatchResult:
        size_bytes = event.estimated_size_bytes
        with self._condition:
            self._dispatched += 1
            over_event_limit = len(self._queue) >= self.spec.max_queue_events
            over_byte_limit = self._pending_bytes + self._active_bytes + size_bytes > self.spec.max_queue_bytes
            if not self._accepting or over_event_limit or over_byte_limit:
                self._dropped += 1
                return _DispatchResult(dropped=1)
            self._queue.append((event, size_bytes))
            self._pending_bytes += size_bytes
            self._enqueued += 1
            self._condition.notify()
            return _DispatchResult(enqueued=1)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop and not self._queue:
                    return
                event, size_bytes = self._queue.popleft()
                self._pending_bytes -= size_bytes
                self._active_bytes += size_bytes
                self._active_handlers += 1
            try:
                self.handler(event)
            except Exception:
                logger.exception(
                    "queued event subscriber %s failed for event %s",
                    self.spec.subscription_id,
                    event.event_id,
                )
                with self._condition:
                    self._failed += 1
            else:
                with self._condition:
                    self._delivered += 1
            finally:
                with self._condition:
                    self._active_bytes -= size_bytes
                    self._active_handlers -= 1
                    self._condition.notify_all()

    def wait_empty(self, timeout_s: float | None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            while self._queue or self._active_handlers:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, drain: bool, timeout_s: float | None) -> bool:
        worker = self._worker
        called_from_worker = worker is not None and threading.current_thread() is worker
        deadline = None if timeout_s is None else time.monotonic() + max(timeout_s, 0.0)
        with self._condition:
            self._accepting = False

        drained = False if called_from_worker and drain else (self.wait_empty(timeout_s) if drain else False)

        with self._condition:
            if not drain or not drained:
                self._dropped += len(self._queue)
                self._queue.clear()
                self._pending_bytes = 0
            self._stop = True
            self._condition.notify_all()

        if worker is not None and not called_from_worker and drain:
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
            worker.join(remaining)
        return drained if drain else True

    def stats(self) -> SubscriptionStats:
        with self._condition:
            return SubscriptionStats(
                subscription_id=self.spec.subscription_id,
                dispatched=self._dispatched,
                delivered=self._delivered,
                enqueued=self._enqueued,
                dropped=self._dropped,
                failed=self._failed,
                pending_events=len(self._queue) + self._active_handlers,
                pending_bytes=self._pending_bytes + self._active_bytes,
                accepting=self._accepting,
            )


class Subscription:
    """Lifecycle handle returned by ``LocalEventBus.subscribe``."""

    def __init__(self, bus: LocalEventBus, state: _SubscriptionState):
        self._bus_ref = weakref.ref(bus)
        self._state = state

    @property
    def subscription_id(self) -> str:
        return self._state.spec.subscription_id

    @property
    def stats(self) -> SubscriptionStats:
        return self._state.stats()

    def close(self, *, drain: bool = False, timeout_s: float | None = None) -> bool:
        bus = self._bus_ref()
        if bus is None:
            return self._state.close(drain=drain, timeout_s=timeout_s)
        return bus._unsubscribe_state(self._state, drain=drain, timeout_s=timeout_s)

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        self.close()
        return False


class LocalEventBus:
    """Process-local broker for inline, queued, and state-sync subscriptions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscriptions: dict[str, _SubscriptionState] = {}
        self._closed = False

    def subscribe(self, spec: SubscriptionSpec, handler: EventHandler) -> Subscription:
        if _is_async_handler(handler):
            raise TypeError("LocalEventBus handlers must be synchronous; use a queued adapter for async I/O")
        state = _SubscriptionState(spec, handler)
        with self._lock:
            if self._closed:
                raise RuntimeError("LocalEventBus is closed")
            if spec.subscription_id in self._subscriptions:
                raise ValueError(f"duplicate subscription_id {spec.subscription_id!r}")
            self._subscriptions[spec.subscription_id] = state
            state.start()
        return Subscription(self, state)

    def unsubscribe(self, subscription_id: str) -> None:
        with self._lock:
            state = self._subscriptions.get(subscription_id)
        if state is not None:
            self._unsubscribe_state(state, drain=False, timeout_s=None)

    def _unsubscribe_state(
        self,
        state: _SubscriptionState,
        *,
        drain: bool,
        timeout_s: float | None,
    ) -> bool:
        with self._lock:
            current = self._subscriptions.get(state.spec.subscription_id)
            if current is state:
                del self._subscriptions[state.spec.subscription_id]
        return state.close(drain=drain, timeout_s=timeout_s)

    def publish(self, event: Event) -> PublishReceipt:
        with self._lock:
            if self._closed:
                return PublishReceipt(
                    event_id=event.event_id,
                    producer_seq=event.producer_seq,
                    dropped=1,
                )
            subscriptions = tuple(state for state in self._subscriptions.values() if state.spec.matches(event))

        delivered = enqueued = dropped = failed = 0
        for state in subscriptions:
            result = state.dispatch(event)
            delivered += result.delivered
            enqueued += result.enqueued
            dropped += result.dropped
            failed += result.failed
        return PublishReceipt(
            event_id=event.event_id,
            producer_seq=event.producer_seq,
            matched_subscriptions=len(subscriptions),
            delivered=delivered,
            enqueued=enqueued,
            dropped=dropped,
            failed=failed,
        )

    def deliver(self, subscription_id: str, batch: EventBatch) -> DeliveryAck:
        """Deliver an already-matched batch without publishing it to other subscribers."""
        with self._lock:
            state = self._subscriptions.get(subscription_id)
        if state is None:
            return DeliveryAck(
                subscription_id=subscription_id,
                subscription_found=False,
                requested_events=len(batch.events),
                dropped=len(batch.events),
            )

        delivered = enqueued = dropped = failed = 0
        for event in batch.events:
            if not state.spec.matches(event):
                dropped += 1
                continue
            result = state.dispatch(event)
            delivered += result.delivered
            enqueued += result.enqueued
            dropped += result.dropped
            failed += result.failed
        return DeliveryAck(
            subscription_id=subscription_id,
            subscription_found=True,
            requested_events=len(batch.events),
            delivered=delivered,
            enqueued=enqueued,
            dropped=dropped,
            failed=failed,
        )

    async def flush(self, scope: Scope | str | None = None, deadline_s: float = 5.0) -> FlushReport:
        """Drain matching queued subscriptions without blocking an async Actor loop."""
        return await asyncio.to_thread(self._flush_blocking, scope, deadline_s)

    def _flush_blocking(self, scope: Scope | str | None, deadline_s: float) -> FlushReport:
        selected_scope = Scope(scope) if scope is not None else None
        with self._lock:
            states = tuple(
                state
                for state in self._subscriptions.values()
                if selected_scope is None or state.spec.scope is selected_scope
            )

        deadline = time.monotonic() + max(deadline_s, 0.0)
        incomplete: list[str] = []
        for state in states:
            if not state.wait_empty(max(deadline - time.monotonic(), 0.0)):
                incomplete.append(state.spec.subscription_id)

        stats = tuple(state.stats() for state in states)
        pending_events = sum(item.pending_events for item in stats)
        incomplete.extend(
            item.subscription_id for item in stats if not item.complete and item.subscription_id not in incomplete
        )
        return FlushReport(
            complete=not incomplete and pending_events == 0,
            pending_events=pending_events,
            incomplete_subscriptions=tuple(incomplete),
        )

    def close(self, *, drain: bool = False, deadline_s: float = 5.0) -> FlushReport:
        with self._lock:
            if self._closed:
                return FlushReport(complete=True, pending_events=0, incomplete_subscriptions=())
            self._closed = True
            states = tuple(self._subscriptions.values())
            self._subscriptions.clear()

        deadline = time.monotonic() + max(deadline_s, 0.0)
        incomplete: list[str] = []
        for state in states:
            if not state.close(
                drain=drain,
                timeout_s=max(deadline - time.monotonic(), 0.0),
            ):
                incomplete.append(state.spec.subscription_id)
        stats = tuple(state.stats() for state in states)
        pending_events = sum(item.pending_events for item in stats)
        incomplete.extend(
            item.subscription_id for item in stats if not item.complete and item.subscription_id not in incomplete
        )
        return FlushReport(
            complete=not incomplete and pending_events == 0,
            pending_events=pending_events,
            incomplete_subscriptions=tuple(incomplete),
        )

    def __enter__(self) -> LocalEventBus:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        self.close()
        return False
