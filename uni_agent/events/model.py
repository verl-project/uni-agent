"""Immutable event envelope and context shared by event publishers and subscribers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

_EVENT_CONTEXT_FIELDS = (
    "episode_id",
    "session_id",
    "generation_id",
    "attempt_id",
    "chain_id",
    "runner_name",
    "global_step",
)
_EMPTY_PAYLOAD: Mapping[str, Any] = MappingProxyType({})


def _freeze_value(value: Any, *, path: str) -> Any:
    """Copy a Ray-serializable value into an immutable in-process representation."""
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings, got {type(key).__name__}")
            frozen[key] = _freeze_value(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item, path=f"{path}[]") for item in value)
    raise TypeError(
        f"{path} must contain only mappings, lists, tuples, bytes, and scalar values; got {type(value).__name__}"
    )


def _thaw_value(value: Any) -> Any:
    """Convert the immutable representation into a Ray-serializable DTO value."""
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _estimate_value_bytes(value: Any) -> int:
    """Estimate payload bytes cheaply enough for local queue budgeting."""
    if value is None:
        return 1
    if isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        return 8
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, Mapping):
        return 32 + sum(len(key.encode("utf-8")) + _estimate_value_bytes(item) + 8 for key, item in value.items())
    if isinstance(value, tuple):
        return 16 + sum(_estimate_value_bytes(item) + 8 for item in value)
    raise TypeError(f"cannot estimate unsupported event value {type(value).__name__}")


def _estimate_context_bytes(context: EventContext) -> int:
    return 32 + sum(
        len(name.encode("utf-8")) + _estimate_value_bytes(getattr(context, name)) + 8 for name in _EVENT_CONTEXT_FIELDS
    )


@dataclass(frozen=True, slots=True)
class EventContext:
    """Correlation identity for one episode, session, generation, or operation."""

    episode_id: str | None = None
    session_id: str | None = None
    generation_id: str | None = None
    attempt_id: str | None = None
    chain_id: int | None = None
    runner_name: str | None = None
    global_step: int | None = None

    def overlay(self, other: EventContext | None) -> EventContext:
        """Overlay non-None values from ``other`` onto this context."""
        if other is None or all(getattr(other, name) is None for name in _EVENT_CONTEXT_FIELDS):
            return self
        return EventContext(
            **{
                name: getattr(other, name) if getattr(other, name) is not None else getattr(self, name)
                for name in _EVENT_CONTEXT_FIELDS
            }
        )

    def to_dict(self) -> dict[str, str | int | None]:
        return {name: getattr(self, name) for name in _EVENT_CONTEXT_FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EventContext:
        return cls(**{name: data[name] for name in _EVENT_CONTEXT_FIELDS if name in data})


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable business fact published inside a process."""

    event_id: str
    event_type: str
    schema_version: int
    run_id: str
    producer_id: str
    producer_epoch: str
    producer_seq: int
    context: EventContext
    occurred_at_unix_ns: int
    payload: Mapping[str, Any]
    estimated_size_bytes: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("event_id", "event_type", "run_id", "producer_id", "producer_epoch"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.schema_version <= 0:
            raise ValueError(f"schema_version must be positive, got {self.schema_version}")
        if self.producer_seq < 0:
            raise ValueError(f"producer_seq must be non-negative, got {self.producer_seq}")
        if self.occurred_at_unix_ns < 0:
            raise ValueError(f"occurred_at_unix_ns must be non-negative, got {self.occurred_at_unix_ns}")
        if not isinstance(self.context, EventContext):
            raise TypeError(f"context must be EventContext, got {type(self.context).__name__}")
        if not isinstance(self.payload, Mapping):
            raise TypeError(f"payload must be a mapping, got {type(self.payload).__name__}")

        frozen_payload = _EMPTY_PAYLOAD if not self.payload else _freeze_value(self.payload, path="payload")
        object.__setattr__(self, "payload", frozen_payload)
        envelope_size = sum(
            len(value.encode("utf-8"))
            for value in (self.event_id, self.event_type, self.run_id, self.producer_id, self.producer_epoch)
        )
        context_size = _estimate_context_bytes(self.context)
        payload_size = 32 if not frozen_payload else _estimate_value_bytes(frozen_payload)
        object.__setattr__(
            self,
            "estimated_size_bytes",
            envelope_size + context_size + payload_size + 32,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable DTO for a future Ray or HTTP boundary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "producer_id": self.producer_id,
            "producer_epoch": self.producer_epoch,
            "producer_seq": self.producer_seq,
            "context": self.context.to_dict(),
            "occurred_at_unix_ns": self.occurred_at_unix_ns,
            "payload": _thaw_value(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Event:
        context_data = data.get("context")
        payload = data.get("payload")
        if not isinstance(context_data, Mapping):
            raise TypeError("event context DTO must be a mapping")
        if not isinstance(payload, Mapping):
            raise TypeError("event payload DTO must be a mapping")
        return cls(
            event_id=str(data["event_id"]),
            event_type=str(data["event_type"]),
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            producer_id=str(data["producer_id"]),
            producer_epoch=str(data["producer_epoch"]),
            producer_seq=int(data["producer_seq"]),
            context=EventContext.from_dict(context_data),
            occurred_at_unix_ns=int(data["occurred_at_unix_ns"]),
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class EventBatch:
    """A serializable batch delivered to one already-matched subscription."""

    events: tuple[Event, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(event, Event) for event in self.events):
            raise TypeError("events must contain only Event instances")

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"events": [event.to_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EventBatch:
        events = data.get("events")
        if not isinstance(events, list):
            raise TypeError("event batch DTO must contain an events list")
        return cls(tuple(Event.from_dict(event) for event in events))
