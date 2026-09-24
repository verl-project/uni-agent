"""Bounded admission observations without capacity or Grant ownership."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Mapping
from typing import Any

from uni_agent.events import (
    GATEWAY_SESSION_STATE_SCOPE,
    REPLICA_CAPACITY_CHANGED,
    ROUTE_COMMITTED,
    SESSION_CLOSED,
    SESSION_OPENED,
    Event,
    StateSnapshot,
)


class AdmissionSignalsProjector:
    """Correlate committed Router facts and Gateway lifecycle observations."""

    def __init__(
        self,
        *,
        max_route_observations: int = 1024,
        max_terminal_sessions: int = 4096,
    ) -> None:
        if min(max_route_observations, max_terminal_sessions) <= 0:
            raise ValueError("admission observation bounds must be positive")
        self._lock = threading.RLock()
        self._routes: deque[dict[str, Any]] = deque(maxlen=max_route_observations)
        self._max_terminal_sessions = max_terminal_sessions
        self._route_commits = 0
        self._dropped_route_observations = 0
        self._capacity: dict[str, dict[str, Any]] = {}
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self._terminal_sessions: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._source_revisions: dict[str, int] = {}
        self._source_health: dict[str, str] = {}

    def apply(self, event: Event) -> None:
        if event.schema_version != 1:
            raise ValueError(f"unsupported admission event schema_version {event.schema_version}")
        if event.event_type == ROUTE_COMMITTED:
            self._apply_route(event)
        elif event.event_type == REPLICA_CAPACITY_CHANGED:
            self._apply_capacity(event)
        elif event.event_type in {SESSION_OPENED, SESSION_CLOSED}:
            self._apply_session(event)

    def install_gateway_snapshot(self, snapshot: StateSnapshot) -> None:
        if snapshot.scope != GATEWAY_SESSION_STATE_SCOPE:
            raise ValueError(f"unsupported Gateway session snapshot scope {snapshot.scope!r}")
        current = self._snapshot_sessions(snapshot.owner, snapshot.current_entities, "open")
        terminal = self._snapshot_sessions(snapshot.owner, snapshot.terminal_entities, "closed")
        with self._lock:
            self._sessions = {key: value for key, value in self._sessions.items() if key[0] != snapshot.owner}
            self._terminal_sessions = OrderedDict(
                (key, value) for key, value in self._terminal_sessions.items() if key[0] != snapshot.owner
            )
            self._sessions.update(current)
            self._terminal_sessions.update(terminal)
            self._trim_terminal_sessions()
            self._source_revisions[snapshot.owner] = snapshot.revision
            self._source_health[snapshot.owner] = snapshot.source_health

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "observe",
                "authoritative": False,
                "route_commits": self._route_commits,
                "retained_route_observations": len(self._routes),
                "dropped_route_observations": self._dropped_route_observations,
                "latest_route": dict(self._routes[-1]) if self._routes else None,
                "capacity": [dict(self._capacity[key]) for key in sorted(self._capacity)],
                "active_sessions": len(self._sessions),
                "terminal_sessions": len(self._terminal_sessions),
                "source_revisions": dict(sorted(self._source_revisions.items())),
                "source_health": dict(sorted(self._source_health.items())),
            }

    def _apply_route(self, event: Event) -> None:
        request_id = self._required_string(event.payload, "request_id")
        replica_id = self._required_string(event.payload, "replica_id")
        ledger_version = self._required_non_negative_int(event.payload, "ledger_version")
        with self._lock:
            if len(self._routes) == self._routes.maxlen:
                self._dropped_route_observations += 1
            self._routes.append(
                {
                    "request_id": request_id,
                    "replica_id": replica_id,
                    "ledger_version": ledger_version,
                }
            )
            self._route_commits += 1

    def _apply_capacity(self, event: Event) -> None:
        replica_id = self._required_string(event.payload, "replica_id")
        replica_epoch = self._required_string(event.payload, "replica_epoch")
        ledger_version = self._required_non_negative_int(event.payload, "ledger_version")
        reason = self._required_string(event.payload, "reason")
        health = self._required_string(event.payload, "health")
        replica_inflight = self._required_non_negative_int(event.payload, "replica_inflight")
        total_inflight = self._required_non_negative_int(event.payload, "total_inflight")
        with self._lock:
            previous = self._capacity.get(replica_id)
            if previous is not None and previous["replica_epoch"] == replica_epoch:
                if previous["ledger_version"] >= ledger_version:
                    return
            self._capacity[replica_id] = {
                "replica_id": replica_id,
                "replica_epoch": replica_epoch,
                "ledger_version": ledger_version,
                "reason": reason,
                "health": health,
                "replica_inflight": replica_inflight,
                "total_inflight": total_inflight,
            }

    def _apply_session(self, event: Event) -> None:
        session_id = event.context.session_id
        if not session_id:
            raise ValueError(f"{event.event_type} requires context.session_id")
        revision = self._required_non_negative_int(event.payload, "revision")
        source_revision = event.payload.get("source_revision", revision)
        if not isinstance(source_revision, int) or isinstance(source_revision, bool) or source_revision < 0:
            raise ValueError("source_revision must be a non-negative integer")
        key = (event.producer_id, session_id)
        state = {
            "owner": event.producer_id,
            "session_id": session_id,
            "episode_id": event.context.episode_id,
            "revision": revision,
            "status": "open" if event.event_type == SESSION_OPENED else "closed",
        }
        with self._lock:
            previous = self._sessions.get(key) or self._terminal_sessions.get(key)
            if previous is not None and previous["revision"] >= revision:
                return
            if event.event_type == SESSION_OPENED:
                self._terminal_sessions.pop(key, None)
                self._sessions[key] = state
            else:
                self._sessions.pop(key, None)
                self._terminal_sessions[key] = state
                self._terminal_sessions.move_to_end(key)
                self._trim_terminal_sessions()
            self._source_revisions[event.producer_id] = max(
                source_revision,
                self._source_revisions.get(event.producer_id, 0),
            )

    def _trim_terminal_sessions(self) -> None:
        while len(self._terminal_sessions) > self._max_terminal_sessions:
            self._terminal_sessions.popitem(last=False)

    @staticmethod
    def _snapshot_sessions(
        owner: str,
        entities: Mapping[str, Any],
        default_status: str,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for session_id, raw in entities.items():
            if not isinstance(raw, Mapping):
                raise TypeError("Gateway session snapshot entities must be mappings")
            revision = raw.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
                raise ValueError("snapshot session revision must be a non-negative integer")
            result[(owner, session_id)] = {
                "owner": owner,
                "session_id": session_id,
                "episode_id": raw.get("episode_id"),
                "revision": revision,
                "status": raw.get("status", default_status),
            }
        return result

    @staticmethod
    def _required_string(payload: Mapping[str, Any], name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _required_non_negative_int(payload: Mapping[str, Any], name: str) -> int:
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value


__all__ = ["AdmissionSignalsProjector"]
