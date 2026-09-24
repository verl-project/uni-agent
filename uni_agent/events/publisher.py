"""Uniform event publication entry point."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from .bus import LocalEventBus, PublishReceipt
from .context import get_current_event_context
from .model import Event, EventContext


class EventPublisher:
    """Enrich business facts with identity and publish them to one local bus."""

    def __init__(
        self,
        bus: LocalEventBus,
        *,
        run_id: str,
        producer_id: str,
        producer_epoch: str | None = None,
        context: EventContext | None = None,
        context_provider: Callable[[], EventContext] = get_current_event_context,
        event_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not producer_id:
            raise ValueError("producer_id must be non-empty")
        self._bus = bus
        self.run_id = run_id
        self.producer_id = producer_id
        self.producer_epoch = producer_epoch or str(uuid.uuid4())
        self._base_context = context or EventContext()
        self._context_provider = context_provider
        self._event_id_factory = event_id_factory
        self._wall_clock_ns = wall_clock_ns
        self._sequence = 0
        self._sequence_lock = threading.Lock()

    def publish(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        context: EventContext | None = None,
        schema_version: int = 1,
    ) -> PublishReceipt:
        """Create and synchronously hand one immutable event to the local bus."""
        with self._sequence_lock:
            self._sequence += 1
            producer_seq = self._sequence

        bound_context = self._context_provider()
        effective_context = self._base_context.overlay(bound_context).overlay(context)
        event = Event(
            event_id=self._event_id_factory(),
            event_type=event_type,
            schema_version=schema_version,
            run_id=self.run_id,
            producer_id=self.producer_id,
            producer_epoch=self.producer_epoch,
            producer_seq=producer_seq,
            context=effective_context,
            occurred_at_unix_ns=self._wall_clock_ns(),
            payload=payload or {},
        )
        return self._bus.publish(event)
