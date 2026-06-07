"""Session-id router that dispatches gateway calls to Ray actor handles."""

from __future__ import annotations

from typing import Any


class GatewayManager:
    """Session-routing component owned by the serving runtime.

    ``GatewayServingRuntime`` creates this after starting gateway actors. The
    manager tracks which actor owns each session and forwards lifecycle calls to
    that actor through Ray remote methods.
    """

    def __init__(self, gateways: list):
        self.gateways = gateways
        self.gateway_count = len(gateways)
        self.active_sessions_per_gateway = [0 for _ in gateways]
        self._session_to_gateway_index: dict[str, int] = {}

    def _select_gateway_index(self) -> int:
        if not self.gateways:
            raise RuntimeError("No gateway actors configured")
        return min(range(len(self.gateways)), key=lambda index: self.active_sessions_per_gateway[index])

    def _get_gateway_index(self, session_id: str) -> int:
        gateway_index = self._session_to_gateway_index.get(session_id)
        if gateway_index is None:
            raise KeyError(session_id)
        return gateway_index

    def _get_gateway(self, session_id: str):
        gateway_index = self._get_gateway_index(session_id)
        return self.gateways[gateway_index], gateway_index

    async def create_session(self, session_id: str, **kwargs):
        """Create a session on the least-loaded actor, record the route, and return its handle."""
        gateway_index = self._select_gateway_index()
        gateway = self.gateways[gateway_index]
        handle = await gateway.create_session.remote(session_id=session_id, **kwargs)
        self._session_to_gateway_index[session_id] = gateway_index
        self.active_sessions_per_gateway[gateway_index] += 1
        return handle

    async def finalize_session(self, session_id: str):
        """Finalize a session on its owning actor, release the route, and return its trajectories."""
        gateway, gateway_index = self._get_gateway(session_id)
        trajectories = await gateway.finalize_session.remote(session_id=session_id)
        self._session_to_gateway_index.pop(session_id, None)
        self.active_sessions_per_gateway[gateway_index] -= 1
        return trajectories

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        """Mark a routed session complete on its owning actor with optional reward metadata."""
        gateway, _ = self._get_gateway(session_id)
        await gateway.complete_session.remote(session_id=session_id, reward_info=reward_info)

    async def abort_session(self, session_id: str) -> None:
        """Abort a routed session on its owning actor and release the route."""
        gateway, gateway_index = self._get_gateway(session_id)
        await gateway.abort_session.remote(session_id=session_id)
        self._session_to_gateway_index.pop(session_id, None)
        self.active_sessions_per_gateway[gateway_index] -= 1

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        """Wait for a routed session on its owning actor."""
        gateway, _ = self._get_gateway(session_id)
        await gateway.wait_for_completion.remote(session_id=session_id, timeout=timeout)
