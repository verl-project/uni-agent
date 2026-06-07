"""Serving-runtime adapter that owns gateway actors and session routing."""

from __future__ import annotations

import asyncio
from typing import Any

import ray

from uni_agent.gateway.config import GatewayActorConfig
from verl.workers.rollout.llm_server import LLMServerClient


class GatewayServingRuntime:
    """Standalone serving runtime for gateway-backed agent sessions.

    The runtime receives an ``LLMServerClient`` backend, injects it into each
    ``GatewayActor``, starts the actors, and exposes session lifecycle methods by
    delegating to ``GatewayManager``.
    """

    def __init__(
        self,
        llm_client: LLMServerClient,
        *,
        gateway_count: int,
        gateway_actor_config: GatewayActorConfig | None = None,
    ):
        self._llm_client = llm_client
        self.owned_gateway_actors: list[ray.actor.ActorHandle] = []
        self.gateway_manager = None

        if gateway_count <= 0:
            raise ValueError("gateway_count must be positive")
        if gateway_actor_config is None:
            raise ValueError("gateway_actor_config is required when gateway_count > 0")
        self._initialize_gateway_runtime(
            gateway_count=gateway_count,
            gateway_actor_config=gateway_actor_config,
        )

    def _initialize_gateway_runtime(
        self,
        *,
        gateway_count: int,
        gateway_actor_config: GatewayActorConfig,
    ) -> None:
        from uni_agent.gateway.gateway import GatewayActor
        from uni_agent.gateway.manager import GatewayManager

        # Round-robin across alive CPU nodes so gateway actors do not all pack onto
        # the driver node under Ray's default PACK scheduling. Mirrors
        # AgentLoopWorker placement (verl/experimental/agent_loop/agent_loop.py).
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        if not node_ids:
            raise RuntimeError("No alive CPU nodes available for GatewayActor placement")

        self.owned_gateway_actors = [
            GatewayActor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ids[i % len(node_ids)],
                    soft=True,
                ),
            ).remote(gateway_actor_config, backend=self._llm_client)
            for i in range(gateway_count)
        ]
        ray.get([gateway.start.remote() for gateway in self.owned_gateway_actors])
        self.gateway_manager = GatewayManager(self.owned_gateway_actors)

    def _require_session_runtime(self):
        if self.gateway_manager is None:
            raise RuntimeError("Session runtime is not initialized")
        return self.gateway_manager

    async def create_session(self, session_id: str, **kwargs):
        """Create a gateway session through the session manager and return its handle."""
        gateway_manager = self._require_session_runtime()
        return await gateway_manager.create_session(session_id=session_id, **kwargs)

    async def finalize_session(self, session_id: str):
        """Finalize a gateway session through the session manager and return its trajectories."""
        gateway_manager = self._require_session_runtime()
        return await gateway_manager.finalize_session(session_id=session_id)

    async def complete_session(self, session_id: str, reward_info: dict[str, Any] | None = None) -> None:
        """Mark a gateway session complete with optional reward metadata."""
        gateway_manager = self._require_session_runtime()
        await gateway_manager.complete_session(session_id=session_id, reward_info=reward_info)

    async def abort_session(self, session_id: str) -> None:
        """Abort a gateway session through the session manager."""
        gateway_manager = self._require_session_runtime()
        await gateway_manager.abort_session(session_id=session_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        """Wait for a gateway session to reach a terminal state."""
        gateway_manager = self._require_session_runtime()
        await gateway_manager.wait_for_completion(session_id=session_id, timeout=timeout)

    async def shutdown(self) -> None:
        """Stop owned gateway actors and clear the session manager."""
        if self.owned_gateway_actors:
            await asyncio.gather(*(gateway.shutdown.remote() for gateway in self.owned_gateway_actors))
        self.owned_gateway_actors = []
        self.gateway_manager = None
