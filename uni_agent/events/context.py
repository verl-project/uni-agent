"""ContextVar binding for in-process event correlation."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from .model import EventContext

_EMPTY_CONTEXT = EventContext()
_current_event_context: contextvars.ContextVar[EventContext | None] = contextvars.ContextVar(
    "uni_agent_event_context",
    default=None,
)


def get_current_event_context() -> EventContext:
    """Return the event context bound to the current thread or async task."""
    return _current_event_context.get() or _EMPTY_CONTEXT


@contextmanager
def bind_event_context(context: EventContext | None = None, **overrides: Any) -> Iterator[EventContext]:
    """Bind a nested event context and restore the previous value on exit.

    Non-None values from ``context`` inherit over the current binding. Explicit
    keyword overrides are then applied last and may set a field back to ``None``.
    """
    bound = get_current_event_context().overlay(context)
    if overrides:
        try:
            bound = replace(bound, **overrides)
        except TypeError as error:
            raise TypeError(f"invalid event context override: {error}") from error
    token = _current_event_context.set(bound)
    try:
        yield bound
    finally:
        _current_event_context.reset(token)
