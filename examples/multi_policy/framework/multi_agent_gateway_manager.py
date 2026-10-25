"""Gateway manager extension for multi-agent session groups.

With one session and one ``/sessions/{session_id}/v1`` URL per role, request
routing needs no Gateway-side extension at all: the shared gateway actor
resolves the session from the URL and the policy from the session's
creation-time binding. What remains specific to multi-agent scenarios is the
*group* lifecycle: creating every role session for one rollout, finalizing them
together, and aborting them together. That bookkeeping lives on the manager.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.manager import GatewayManager
from uni_agent.gateway.session import Trajectory
from verl.workers.rollout.llm_server import LLMServerClient

from .multi_agent_types import SessionGroupHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RoleSession:
    """Manager-owned binding between a role session and its policy."""

    role_session_id: str
    policy_name: str


class MultiAgentGatewayManager(GatewayManager):
    """Gateway manager for multi-role MAS session groups.

    Each role owns an independent Gateway session, while the manager routes,
    finalizes, and aborts all role sessions as one group. Lifecycle methods
    retain the parent names for framework compatibility; ``session_id``
    identifies the MAS group in this subclass.
    """

    def __init__(
        self,
        llm_client: Mapping[str, LLMServerClient],
        *,
        gateway_count: int,
        gateway_actor_config: Mapping[str, GatewayActorConfig],
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            gateway_count=gateway_count,
            gateway_actor_config=gateway_actor_config,
        )
        # The parent field is intentionally reused for group-level scheduling
        # load. It counts active MAS groups, not individual role sessions.
        # TODO(scheduling-unit): if core starts deriving session-count decisions
        # from active_sessions_per_gateway (autoscaling, max-sessions thresholds,
        # metrics, drain), split this into a session-accurate counter plus a
        # separate group-level scheduling metric instead of reusing the parent
        # field.
        self._group_sessions: dict[str, dict[str, _RoleSession]] = {}
        self._group_gateway_indices: dict[str, int] = {}

    async def create_session(
        self,
        session_id: str,
        *,
        role_policy_mapping: Mapping[str, str],
        sampling_params: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionGroupHandle:
        """Create one Gateway session for each role in a MAS session group.

        Each group occupies one scheduling slot, regardless of its role count.
        The returned handle maps every role to its provider-compatible ``/v1``
        URL.

        Args:
            session_id: MAS session-group ID.
            role_policy_mapping: Mapping from role names to policy names.
            sampling_params: Optional parameters keyed by policy name.
            metadata: Optional metadata copied to each role session.

        Raises:
            RuntimeError: If the manager has already been shut down.
            ValueError: If the group ID is already reserved or no roles are given.

        Returns:
            A handle containing one base URL per role.
        """
        session_group_id = session_id
        if not self.gateways:
            raise RuntimeError("Gateway manager is shut down")
        if session_group_id in self._group_gateway_indices:
            raise ValueError(f"Duplicate session group: {session_group_id}")
        if not role_policy_mapping:
            raise ValueError("role_policy_mapping must not be empty")

        gateway_index = self._select_gateway_index()
        gateway = self.gateways[gateway_index]

        role_sessions: dict[str, _RoleSession] = {}
        for role_name, policy_name in role_policy_mapping.items():
            if not isinstance(role_name, str) or not role_name:
                raise ValueError(f"Invalid role: {role_name!r}")
            if not isinstance(policy_name, str) or not policy_name:
                raise ValueError(
                    f"Invalid policy name for role {role_name}: {policy_name!r}"
                )
            role_sessions[role_name] = _RoleSession(
                role_session_id=f"{session_group_id}:{role_name}",
                policy_name=policy_name,
            )

        # Reserve the group ID and scheduling slot before the first await. The
        # reservation also rejects another create for this ID while sessions
        # are still being created. Register all deterministic role session IDs
        # before the first await so cancellation cannot lose an in-flight role.
        self._group_gateway_indices[session_group_id] = gateway_index
        self._group_sessions[session_group_id] = role_sessions
        self.active_sessions_per_gateway[gateway_index] += 1

        policy_sampling_params = sampling_params or {}
        role_base_urls: dict[str, str] = {}
        try:
            for role_name, role_session in role_sessions.items():
                handle = await gateway.create_session.remote(
                    session_id=role_session.role_session_id,
                    metadata={
                        **dict(metadata or {}),
                        "role": role_name,
                        "policy_name": role_session.policy_name,
                        "session_group_id": session_group_id,
                    },
                    sampling_params=dict(
                        policy_sampling_params.get(role_session.policy_name, {})
                    ),
                    policy_name=role_session.policy_name,
                )
                role_base_urls[role_name] = handle.base_url
        except BaseException:
            try:
                await asyncio.shield(self.abort_session(session_group_id))
            finally:
                self._drop_group_state(session_group_id)
            raise

        return SessionGroupHandle(
            session_group_id=session_group_id,
            role_base_urls=role_base_urls,
        )

    async def finalize_session(self, session_id: str) -> list[Trajectory]:
        """Finalize a session group and aggregate all role trajectories.

        The ``session_id`` argument identifies the MAS group in this subclass.
        Role and policy metadata are attached to each returned trajectory. If
        finalization fails, known role sessions are aborted before the error is
        propagated.

        Args:
            session_id: MAS session-group ID whose role sessions are finalized.
        """
        gateway_index = self._group_gateway_indices[session_id]
        gateway = self.gateways[gateway_index]
        role_sessions = self._group_sessions[session_id]
        trajectories: list[Trajectory] = []
        try:
            for role_name, role_session in role_sessions.items():
                for trajectory in await gateway.finalize_session.remote(
                    session_id=role_session.role_session_id
                ):
                    trajectories.append(
                        replace(
                            trajectory,
                            extra_fields={
                                **trajectory.extra_fields,
                                "session_group_id": session_id,
                                "role": role_name,
                                "policy_name": role_session.policy_name,
                            },
                        )
                    )
        except (Exception, asyncio.CancelledError):
            # A partially finalized group cannot be retried or used for training.
            try:
                await self._abort_group_sessions(gateway, role_sessions)
            finally:
                self._drop_group_state(session_id)
            raise
        self._drop_group_state(session_id)
        return trajectories

    async def abort_session(self, session_id: str) -> None:
        """Abort every role session in a group, on a best-effort basis.

        The ``session_id`` argument identifies the MAS group in this subclass.
        Missing groups are treated as already aborted for idempotent cleanup.

        Args:
            session_id: MAS session-group ID whose role sessions are aborted.
        """
        if session_id not in self._group_sessions:
            return
        gateway_index = self._group_gateway_indices[session_id]
        gateway = self.gateways[gateway_index]
        try:
            await self._abort_group_sessions(gateway, self._group_sessions[session_id])
        finally:
            self._drop_group_state(session_id)

    async def shutdown(self) -> None:
        """Abort active groups, then stop gateway actors and reset group state.

        Group state must be snapshotted before the parent shutdown clears the
        gateway pool. Each active group is aborted on its owning gateway so
        actor-owned role sessions are released before the actors are stopped.
        """
        active_groups: list[tuple[Any, Mapping[str, _RoleSession]]] = []
        for session_group_id, gateway_index in self._group_gateway_indices.items():
            role_sessions = self._group_sessions.get(session_group_id)
            if role_sessions is not None:
                active_groups.append((self.gateways[gateway_index], role_sessions))

        try:
            await asyncio.gather(
                *(
                    self._abort_group_sessions(gateway, role_sessions)
                    for gateway, role_sessions in active_groups
                )
            )
        finally:
            try:
                await super().shutdown()
            finally:
                self._group_sessions = {}
                self._group_gateway_indices = {}

    async def _abort_group_sessions(
        self,
        gateway,
        role_sessions: Mapping[str, _RoleSession],
    ) -> None:
        """Best-effort abort for the role sessions already created in a group."""
        for role_session in role_sessions.values():
            try:
                await gateway.abort_session.remote(
                    session_id=role_session.role_session_id
                )
            except Exception:
                logger.exception(
                    "Failed to abort session %s", role_session.role_session_id
                )

    def _drop_group_state(self, session_group_id: str) -> None:
        """Remove one group reservation and release its scheduling slot."""
        self._group_sessions.pop(session_group_id, None)
        gateway_index = self._group_gateway_indices.pop(session_group_id, None)
        if gateway_index is not None:
            self.active_sessions_per_gateway[gateway_index] -= 1
