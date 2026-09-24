"""Shadow projection of Gateway session lifecycle state for the Router."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from uni_agent.events import (
    GATEWAY_SESSION_STATE_SCOPE,
    SESSION_CLOSED,
    SESSION_OPENED,
    Event,
    StateSnapshot,
)


@dataclass(frozen=True, slots=True)
class SessionLifecycleState:
    owner: str
    session_id: str
    episode_id: str | None
    revision: int
    status: str
    close_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "revision": self.revision,
            "status": self.status,
            "close_reason": self.close_reason,
        }


class GatewaySessionStateProjector:
    """Maintain comparison-only session state; never participates in routing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: dict[tuple[str, str], SessionLifecycleState] = {}
        self._terminal: dict[tuple[str, str], SessionLifecycleState] = {}
        self._source_revisions: dict[str, int] = {}

    def install_snapshot(self, snapshot: StateSnapshot) -> None:
        if snapshot.scope != GATEWAY_SESSION_STATE_SCOPE:
            raise ValueError(f"unsupported Gateway session snapshot scope {snapshot.scope!r}")
        current = {
            (snapshot.owner, session_id): self._state_from_snapshot(snapshot.owner, session_id, value, "open")
            for session_id, value in snapshot.current_entities.items()
        }
        terminal = {
            (snapshot.owner, session_id): self._state_from_snapshot(snapshot.owner, session_id, value, "closed")
            for session_id, value in snapshot.terminal_entities.items()
        }
        with self._lock:
            self._current = {key: value for key, value in self._current.items() if key[0] != snapshot.owner}
            self._terminal = {key: value for key, value in self._terminal.items() if key[0] != snapshot.owner}
            self._current.update(current)
            self._terminal.update(terminal)
            self._source_revisions[snapshot.owner] = snapshot.revision

    def apply(self, event: Event) -> None:
        if event.event_type not in {SESSION_OPENED, SESSION_CLOSED}:
            return
        if event.schema_version != 1:
            raise ValueError(f"unsupported Gateway session event schema_version {event.schema_version}")
        session_id = event.context.session_id
        if session_id is None:
            raise ValueError(f"{event.event_type} requires context.session_id")
        revision = event.payload.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError(f"{event.event_type} requires a non-negative integer revision")
        key = (event.producer_id, session_id)
        status = "open" if event.event_type == SESSION_OPENED else "closed"
        state = SessionLifecycleState(
            owner=event.producer_id,
            session_id=session_id,
            episode_id=event.context.episode_id,
            revision=revision,
            status=status,
            close_reason=str(event.payload["close_reason"]) if event.payload.get("close_reason") is not None else None,
        )
        source_revision = event.payload.get("source_revision", revision)
        if not isinstance(source_revision, int) or isinstance(source_revision, bool) or source_revision < 0:
            raise ValueError(f"{event.event_type} requires a non-negative integer source_revision")
        with self._lock:
            previous = self._current.get(key) or self._terminal.get(key)
            if previous is not None and previous.revision >= revision:
                return
            if status == "open":
                self._terminal.pop(key, None)
                self._current[key] = state
            else:
                self._current.pop(key, None)
                self._terminal[key] = state
            self._source_revisions[event.producer_id] = max(
                source_revision,
                self._source_revisions.get(event.producer_id, 0),
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "shadow",
                "current": [state.to_dict() for state in self._ordered(self._current)],
                "terminal": [state.to_dict() for state in self._ordered(self._terminal)],
                "source_revisions": dict(sorted(self._source_revisions.items())),
            }

    @staticmethod
    def _ordered(states: dict[tuple[str, str], SessionLifecycleState]) -> list[SessionLifecycleState]:
        return [states[key] for key in sorted(states)]

    @staticmethod
    def _state_from_snapshot(
        owner: str,
        session_id: str,
        value: Any,
        default_status: str,
    ) -> SessionLifecycleState:
        if not isinstance(value, Mapping):
            raise TypeError("Gateway session snapshot entities must be mappings")
        episode_id = value.get("episode_id")
        revision = value.get("revision")
        status = value.get("status", default_status)
        close_reason = value.get("close_reason")
        if episode_id is not None and not isinstance(episode_id, str):
            raise TypeError("snapshot episode_id must be a string or null")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError("snapshot entity revision must be a non-negative integer")
        if status not in {"open", "closed", "aborted"}:
            raise ValueError(f"unsupported Gateway session status {status!r}")
        if close_reason is not None and not isinstance(close_reason, str):
            raise TypeError("snapshot close_reason must be a string or null")
        return SessionLifecycleState(
            owner=owner,
            session_id=session_id,
            episode_id=episode_id,
            revision=revision,
            status=status,
            close_reason=close_reason,
        )
