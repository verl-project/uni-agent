"""Driver-side gateway manager: owns the gateway actor pool and routes sessions.

The manager spawns ``GatewayActor`` handles, injects the LLM client
backend into each, and tracks which actor owns each session so lifecycle calls
forward to the right actor through Ray remote methods.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import uuid4

import ray

from uni_agent.gateway.config import GatewayActorConfig

logger = logging.getLogger(__name__)


@dataclass
class _SessionOwner:
    gateway_index: int
    creation_id: str = field(default_factory=lambda: uuid4().hex)


async def _settle_call(awaitable):
    """Return the settled future and caller cancellation, even when the call failed."""
    task = asyncio.ensure_future(awaitable)
    # An independently cancelled backend call is an outcome, not caller cancellation.
    settled = asyncio.gather(task, return_exceptions=True)
    cancellation = None
    while not settled.done():
        try:
            await asyncio.shield(settled)
        except asyncio.CancelledError as exc:
            cancellation = exc
    return task, cancellation


class GatewayManager:
    """Owns gateway actors and routes sessions to them.

    Spawns ``gateway_count`` actors over the injected backend
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
        # Counts include closing sessions whose backend route still needs release.
        # Framework admission owns runner concurrency; these counts only balance actors.
        self.active_sessions_per_gateway = [0 for _ in self.gateways]
        self._session_owners: dict[str, _SessionOwner] = {}

    def _select_gateway_index(self) -> int:
        if not self.gateways:
            raise RuntimeError("No gateway actors configured")
        return min(range(len(self.gateways)), key=lambda index: self.active_sessions_per_gateway[index])

    async def create_session(self, session_id: str, *, weight_version: int | None = None, **kwargs):
        """Create a session; cancellation settles the remote call and aborts only its result."""
        if session_id in self._session_owners:
            raise RuntimeError(f"Session {session_id} already exists")
        gateway_index = self._select_gateway_index()
        # Reserve the slot synchronously, before the await. Sessions are created
        # concurrently on one event loop; if the counter were bumped after the
        # await, every coroutine in a burst would read the same stale counts and
        # ``min`` would funnel them all onto the lowest-index gateway. Roll back
        # if the remote create fails so a failed session does not inflate the
        # load estimate.
        owner = _SessionOwner(gateway_index)
        self._session_owners[session_id] = owner
        self.active_sessions_per_gateway[gateway_index] += 1
        call = create = None
        try:
            call = self.gateways[gateway_index].create_session.remote(
                session_id=session_id, weight_version=weight_version, creation_id=owner.creation_id, **kwargs
            )
            create = asyncio.ensure_future(call)
            return await asyncio.shield(create)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                if isinstance(call, ray.ObjectRef):
                    ray.cancel(call)
                if create is not None:
                    await _settle_call(create)
            cleanup, _ = await _settle_call(self._close_owner(session_id, owner, abort=True))
            try:
                cleanup.result()
            except (Exception, asyncio.CancelledError):
                logger.exception("session %s: cleanup failed after create failure", session_id)
            raise

    async def finalize_session(self, session_id: str):
        """Finalize a session on its owning actor, release the route, and return its trajectories."""
        owner = self._session_owners[session_id]
        finalize, cancellation = await _settle_call(
            self.gateways[owner.gateway_index].finalize_session.remote(
                session_id=session_id, creation_id=owner.creation_id, close=False
            )
        )
        trajectories = finalize.result()
        # Trajectories are ordinary results before cleanup starts. Release can
        # retry without rematerializing them or transporting data in exceptions.
        for _ in range(2):
            release, release_cancellation = await _settle_call(self._close_owner(session_id, owner))
            cancellation = cancellation or release_cancellation
            try:
                release.result()
                break
            except (Exception, asyncio.CancelledError):
                logger.exception("session %s: route release failed after finalize; keeping trajectories", session_id)
        if cancellation is not None:
            raise cancellation
        return trajectories

    async def abort_session(self, session_id: str) -> None:
        """Abort a routed session on its owning actor and release the route."""
        abort, cancellation = await _settle_call(
            self._close_owner(session_id, self._session_owners[session_id], abort=True)
        )
        abort.result()
        if cancellation is not None:
            raise cancellation

    async def _close_owner(self, session_id: str, owner: _SessionOwner, *, abort: bool = False):
        if self._session_owners.get(session_id) is not owner:
            return None
        gateway = self.gateways[owner.gateway_index]
        close = gateway.abort_session if abort else gateway.close_session
        await close.remote(session_id=session_id, creation_id=owner.creation_id)
        self._drop_route(session_id, owner)

    def _drop_route(self, session_id: str, owner: _SessionOwner) -> None:
        # Concurrent closes must release the load slot at most once.
        if self._session_owners.get(session_id) is owner:
            del self._session_owners[session_id]
            self.active_sessions_per_gateway[owner.gateway_index] -= 1

    async def shutdown(self) -> None:
        """Stop owned gateway actors and clear routing state."""
        if self.gateways:
            await asyncio.gather(*(gateway.shutdown.remote() for gateway in self.gateways))
        self.gateways = []
        self.gateway_count = 0
        self.active_sessions_per_gateway = []
        self._session_owners = {}
