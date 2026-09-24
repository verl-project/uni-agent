"""Driver-side gateway manager: owns the gateway actor pool and routes sessions.

The manager spawns ``GatewayActor`` handles, injects the ``LLMServerClient``
backend into each, and tracks which actor owns each session so lifecycle calls
forward to the right actor through Ray remote methods.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import ray

from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.session import SessionFinalizationResult
from verl.workers.rollout.llm_server import LLMServerClient


class GatewayManager:
    """Owns gateway actors and routes sessions to them.

    Spawns ``gateway_count`` actors over the injected ``LLMServerClient`` backend
    and tracks which actor owns each session so lifecycle calls reach it.
    """

    def __init__(
        self,
        llm_client: LLMServerClient,
        *,
        gateway_count: int,
        gateway_actor_config: GatewayActorConfig | None = None,
        direct_event_target=None,
        global_telemetry_runtime=None,
    ):
        if gateway_count <= 0:
            raise ValueError("gateway_count must be positive")
        if gateway_actor_config is None:
            raise ValueError("gateway_actor_config is required when gateway_count > 0")
        if gateway_actor_config.direct_state_sync_enabled and direct_event_target is None:
            raise ValueError("direct_event_target is required when Direct state sync is enabled")
        if gateway_actor_config.global_telemetry_enabled and global_telemetry_runtime is None:
            raise ValueError("global_telemetry_runtime is required when Global telemetry is enabled")

        from uni_agent.gateway.gateway import GatewayActor

        # Round-robin across alive CPU nodes so gateway actors do not all pack onto
        # the driver node under Ray's default PACK scheduling. Mirrors
        # AgentLoopWorker placement (verl/experimental/agent_loop/agent_loop.py).
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        if not node_ids:
            raise RuntimeError("No alive CPU nodes available for GatewayActor placement")

        cross_actor_events_enabled = (
            gateway_actor_config.direct_state_sync_enabled or gateway_actor_config.global_telemetry_enabled
        )
        event_run_id = f"gateway-run-{uuid4().hex}" if cross_actor_events_enabled else None
        self.global_telemetry_runtime = global_telemetry_runtime
        self.gateways = []
        for i in range(gateway_count):
            actor_class = GatewayActor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ids[i % len(node_ids)],
                    soft=True,
                ),
            )
            if cross_actor_events_enabled:
                transport_targets = {}
                if gateway_actor_config.direct_state_sync_enabled:
                    transport_targets["direct_event_target"] = direct_event_target
                if gateway_actor_config.global_telemetry_enabled:
                    transport_targets["global_event_target"] = global_telemetry_runtime.publish_target
                gateway = actor_class.remote(
                    gateway_actor_config,
                    backend=llm_client,
                    event_run_id=event_run_id,
                    event_source_id=f"gateway-{i}",
                    **transport_targets,
                )
            else:
                gateway = actor_class.remote(gateway_actor_config, backend=llm_client)
            self.gateways.append(gateway)
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
        """Create a session on the least-loaded actor, record the route, and return its handle."""
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
            return await gateway.create_session.remote(session_id=session_id, **kwargs)
        except BaseException:
            self.active_sessions_per_gateway[gateway_index] -= 1
            self._session_to_gateway_index.pop(session_id, None)
            raise

    async def finalize_session(self, session_id: str):
        """Finalize a session on its owning actor, release the route, and return its trajectories."""
        return (await self.finalize_session_result(session_id)).trajectories

    async def finalize_session_result(self, session_id: str) -> SessionFinalizationResult:
        """Finalize a routed session and return trajectories plus metrics."""
        gateway, gateway_index = self._get_gateway(session_id)
        result = await gateway.finalize_session_result.remote(session_id=session_id)
        self._session_to_gateway_index.pop(session_id, None)
        self.active_sessions_per_gateway[gateway_index] -= 1
        return result

    async def abort_session(self, session_id: str) -> None:
        """Abort a routed session on its owning actor and release the route."""
        await self.abort_session_result(session_id)

    async def abort_session_result(self, session_id: str) -> SessionFinalizationResult:
        """Abort a routed session and return metrics observed before failure."""
        gateway, gateway_index = self._get_gateway(session_id)
        result = await gateway.abort_session_result.remote(session_id=session_id)
        self._session_to_gateway_index.pop(session_id, None)
        self.active_sessions_per_gateway[gateway_index] -= 1
        return result

    async def shutdown(self) -> None:
        """Stop owned gateway actors and clear routing state."""
        global_telemetry_runtime = getattr(self, "global_telemetry_runtime", None)
        try:
            if self.gateways:
                await asyncio.gather(*(gateway.shutdown.remote() for gateway in self.gateways))
        finally:
            try:
                if global_telemetry_runtime is not None:
                    await global_telemetry_runtime.shutdown()
            finally:
                self.global_telemetry_runtime = None
                self.gateways = []
                self.gateway_count = 0
                self.active_sessions_per_gateway = []
                self._session_to_gateway_index = {}
