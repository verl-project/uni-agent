"""Expose uni-agent decode selectors through verl's router protocol."""

from __future__ import annotations

import importlib
from typing import Any

from uni_agent.rollout.decode_selectors import DecodeCandidate


def _load_class(fqn: str) -> type:
    module_name, _, class_name = fqn.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"selector_class must be a fully-qualified class name, got {fqn!r}")
    return getattr(importlib.import_module(module_name), class_name)


class SelectorRouter:
    """Adapt a ``select``/``release`` selector to verl's server-balancer API."""

    def __init__(self, servers: dict[str, Any], config: dict[str, Any]) -> None:
        selector_class = config.get("selector_class")
        if not selector_class:
            raise ValueError("SelectorRouter requires selector_class in its YAML config")
        selector_kwargs = {key: value for key, value in config.items() if key not in {"router_class", "selector_class"}}
        self._selector = _load_class(str(selector_class))(**selector_kwargs)
        self._servers = dict(servers)
        self._order = list(self._servers)
        self._reservations: dict[str, str] = {}

    def require_acquire_fields(self) -> list[str]:
        return ["prompt_ids", "reservation_id"]

    def require_release_fields(self) -> list[str]:
        return ["request_id", "reservation_id"]

    def acquire_server(
        self,
        request_id: str,
        prompt_ids: list[int] | None = None,
        reservation_id: str | None = None,
    ) -> tuple[str, Any]:
        # request_id remains the session-affinity key. PD supplies a separate
        # per-call identity; other callers must use unique active request IDs.
        identity = reservation_id if reservation_id is not None else request_id
        if identity in self._reservations:
            raise ValueError(f"reservation {identity!r} is already active")
        candidates = [DecodeCandidate(key=key, handle=self._servers[key]) for key in self._order]
        selected = self._selector.select(
            {"routing_key": request_id, "prompt_ids": prompt_ids, "reservation_id": identity},
            candidates,
        )
        server_id = str(getattr(selected, "key", selected))
        if server_id not in self._servers:
            raise RuntimeError(f"selector returned unknown server: {server_id!r}")
        self._reservations[identity] = server_id
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str, request_id: str | None = None, reservation_id: str | None = None) -> None:
        identity = reservation_id if reservation_id is not None else request_id
        if identity is None:
            raise ValueError("release requires reservation_id or a unique request_id")
        reserved_server = self._reservations.get(identity)
        if reserved_server is None:
            return  # A duplicate/late release must not decrement someone else's load.
        if reserved_server != server_id:
            raise ValueError(f"reservation {identity!r} belongs to {reserved_server!r}, not {server_id!r}")
        index = self._order.index(server_id)
        release_request = getattr(self._selector, "release_request", None)
        if callable(release_request):
            release_request(index, identity)
        else:
            release = getattr(self._selector, "release", None)
            if callable(release):
                release(index)
        del self._reservations[identity]

    def add_servers(self, servers: dict[str, Any]) -> None:
        replacements = any(key in self._servers and self._servers[key] != handle for key, handle in servers.items())
        if self._reservations and (replacements or any(key not in self._servers for key in servers)):
            raise RuntimeError("drain active reservations before changing the selector server pool")
        if replacements:
            # A restarted engine no longer owns the old handle's estimated KV.
            self.clear_sticky_cache()
        for server_id, handle in servers.items():
            if server_id not in self._servers:
                self._order.append(server_id)
            self._servers[server_id] = handle

    def remove_servers(self, server_ids: list[str]) -> None:
        if self._reservations and any(key in self._servers for key in server_ids):
            raise RuntimeError("drain active reservations before changing the selector server pool")
        for server_id in server_ids:
            self._servers.pop(server_id, None)
            if server_id in self._order:
                self._order.remove(server_id)

    def get_all_servers(self) -> list[str]:
        return list(self._order)

    def get_status(self) -> dict[str, Any]:
        pending_tokens = getattr(self._selector, "_pending_tokens", None)
        return {
            "selector": type(self._selector).__name__,
            "servers": list(self._order),
            "total_inflight": self.get_total_inflight(),
            "pending_prompt_tokens": (
                sum(int(value) for value in pending_tokens) if isinstance(pending_tokens, list) else None
            ),
            "active_servers": len(self._order),
        }

    def clear_sticky_cache(self) -> dict[str, Any]:
        reset = getattr(self._selector, "reset", None)
        if callable(reset):
            reset()
        return {"cleared_entries": 0, "server_loads": {}}

    def get_total_inflight(self) -> int:
        """Count active calls for every policy, including zero-token requests."""
        return len(self._reservations)
