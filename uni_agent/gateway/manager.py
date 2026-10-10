"""Driver-side gateway manager: owns the gateway actor pool and routes sessions.

The manager spawns ``GatewayActor`` handles, injects the LLM client
backend into each, and tracks which actor owns each session so lifecycle calls
forward to the right actor through Ray remote methods.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray

from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.session.types import SessionFinalizedReleaseError, SessionRouteReleaseError

logger = logging.getLogger(__name__)


def _find_in_cause_chain(exc: BaseException, error_type: type[BaseException]) -> BaseException | None:
    """Return the first ``error_type`` in ``exc`` or its causes.

    Ray may wrap the actor exception. Walk ``cause`` / ``__cause__`` so both the
    in-process actor error and a ``RayTaskError`` resolve to the same instance.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, error_type):
            return current
        cause = getattr(current, "cause", None)
        if not isinstance(cause, BaseException) or cause is current:
            cause = current.__cause__
        current = cause if isinstance(cause, BaseException) and cause is not current else None
    return None


def _trajectories_preserved_after_release_failure(exc: BaseException) -> list[Any] | None:
    """Return finalized trajectories when ``exc`` is a post-finalize release failure."""
    preserved = _find_in_cause_chain(exc, SessionFinalizedReleaseError)
    return None if preserved is None else list(preserved.trajectories)


class GatewayManager:
    """Owns gateway actors and routes sessions to them.

    Spawns ``gateway_count`` actors over the injected LLM client backend
    and tracks which actor owns each session so lifecycle calls reach it.
    """

    def __init__(
        self,
        llm_client,
        *,
        gateway_count: int,
        gateway_actor_config: GatewayActorConfig | None = None,
    ):
        if gateway_count <= 0:
            raise ValueError("gateway_count must be positive")
        if gateway_actor_config is None:
            raise ValueError("gateway_actor_config is required when gateway_count > 0")

        from uni_agent.gateway.gateway import GatewayActor

        # Round-robin across alive CPU nodes so gateway actors do not all pack onto
        # the driver node under Ray's default PACK scheduling. Mirrors
        # AgentLoopWorker placement (verl/experimental/agent_loop/agent_loop.py).
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        if not node_ids:
            raise RuntimeError("No alive CPU nodes available for GatewayActor placement")

        self.gateways = [
            GatewayActor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ids[i % len(node_ids)],
                    soft=True,
                ),
            ).remote(gateway_actor_config, backend=llm_client)
            for i in range(gateway_count)
        ]
        ray.get([gateway.start.remote() for gateway in self.gateways])
        self.gateway_count = len(self.gateways)
        self.active_sessions_per_gateway = [0 for _ in self.gateways]
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
        """Create a session on the least-loaded actor, record the route, and return its handle.

        Cancelling the caller does not cancel the actor call, which may still
        create the session and bind its backend route. The cancelled create waits
        for that call and aborts what it created before dropping the route.
        """
        gateway_index = self._select_gateway_index()
        gateway = self.gateways[gateway_index]
        # Reserve the slot synchronously, before the await. Sessions are created
        # concurrently on one event loop; if the counter were bumped after the
        # await, every coroutine in a burst would read the same stale counts and
        # ``min`` would funnel them all onto the lowest-index gateway. Roll back
        # if the remote create fails so a failed session does not inflate the
        # load estimate.
        self._session_to_gateway_index[session_id] = gateway_index
        self.active_sessions_per_gateway[gateway_index] += 1
        try:
            create = asyncio.ensure_future(gateway.create_session.remote(session_id=session_id, **kwargs))
            return await asyncio.shield(create)
        except asyncio.CancelledError:
            # The manager is the only owner that can abort this session later.
            # Finish the cleanup even if the caller is cancelled again meanwhile.
            cleanup = asyncio.ensure_future(self._abort_cancelled_create(session_id, gateway_index, create))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    # Every wait must stay shielded: a later cancellation would
                    # otherwise cancel cleanup while the actor is still creating.
                    continue
            cleanup.result()
            raise
        except BaseException:
            self._drop_route(session_id, gateway_index)
            raise

    async def _abort_cancelled_create(self, session_id: str, gateway_index: int, create: asyncio.Future) -> None:
        """Abort the session that a cancelled ``create_session`` call still created."""
        try:
            await create
        except (Exception, asyncio.CancelledError):
            # A failed create (the actor discards it on bind_route failure) left
            # nothing of this call to abort; an existing session is not ours.
            self._drop_route(session_id, gateway_index)
            return
        try:
            # Reuse normal abort ownership: a failure before actor removal keeps
            # this route available for an explicit retry instead of orphaning it.
            await self.abort_session(session_id)
        except (Exception, asyncio.CancelledError):
            logger.exception("session %s: abort failed after cancelled create", session_id)

    async def finalize_session(self, session_id: str):
        """Finalize a session on its owning actor, release the route, and return its trajectories.

        When the actor finalized the episode but sticky-route release failed, retry
        release once via ``abort_session`` and still return the trajectories. A
        cleanup failure must not turn a finished episode into a lost sample.
        Other finalize failures keep the routing entry so the caller can abort.
        """
        gateway, gateway_index = self._get_gateway(session_id)
        try:
            trajectories = await gateway.finalize_session.remote(session_id=session_id)
        except BaseException as exc:
            # Also accept CancelledError when a wrapped release failure still
            # carries the finalized trajectories. A bare cancel has none and
            # propagates, leaving the route so the caller can abort.
            trajectories = _trajectories_preserved_after_release_failure(exc)
            if trajectories is None:
                raise
            try:
                # Session is already popped on the actor. abort only retries release.
                await gateway.abort_session.remote(session_id=session_id)
            except Exception:
                logger.exception(
                    "session %s: route release retry failed after finalize; keeping trajectories",
                    session_id,
                )
            self._drop_route(session_id, gateway_index)
            return trajectories
        self._drop_route(session_id, gateway_index)
        return trajectories

    async def abort_session(self, session_id: str) -> None:
        """Abort a routed session on its owning actor and release the route.

        If the actor already dropped the session and only route release failed,
        still free the manager slot, then raise. Other abort failures keep the
        routing entry so the caller can retry.
        """
        gateway, gateway_index = self._get_gateway(session_id)
        try:
            await gateway.abort_session.remote(session_id=session_id)
        except BaseException as exc:
            if _find_in_cause_chain(exc, SessionRouteReleaseError) is not None:
                self._drop_route(session_id, gateway_index)
            raise
        self._drop_route(session_id, gateway_index)

    def _drop_route(self, session_id: str, gateway_index: int) -> None:
        # Idempotent: decrement only when this call removed the entry, so a
        # repeated close for the same session cannot drive the load count negative.
        if self._session_to_gateway_index.pop(session_id, None) is not None:
            self.active_sessions_per_gateway[gateway_index] -= 1

    async def shutdown(self) -> None:
        """Stop owned gateway actors and clear routing state."""
        if self.gateways:
            await asyncio.gather(*(gateway.shutdown.remote() for gateway in self.gateways))
        self.gateways = []
        self.gateway_count = 0
        self.active_sessions_per_gateway = []
        self._session_to_gateway_index = {}
