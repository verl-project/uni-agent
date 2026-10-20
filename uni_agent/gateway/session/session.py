"""Per-session gateway lifecycle and optional trajectory persistence."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi import HTTPException

from uni_agent.gateway.session.chain_trajectory_manager import ChainTrajectoryManager
from uni_agent.gateway.session.chain_trajectory_manager import TrajectoryBuffer as TrajectoryBuffer
from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.tree_trajectory_manager import TreeTrajectoryManager
from uni_agent.gateway.session.types import InternalGenerationRequest, SessionHandle, Trajectory
from uni_agent.rl_insight.adapter import start_generation_span

logger = logging.getLogger(__name__)


class SessionPhase(str, Enum):
    ACTIVE = "ACTIVE"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


@dataclass
class GenerationOutcome:
    assistant_msg: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class GatewaySession:
    """Coordinate linear chains or an optionally disk-backed trajectory tree."""

    def __init__(
        self,
        handle: SessionHandle,
        codec: MessageCodec,
        prompt_length: int | None = None,
        response_length: int | None = None,
        sampling_params: dict[str, Any] | None = None,
        enable_last_assistant_rollback: bool = True,
        coalesce_reserved_exact_requests: bool = True,
        metadata: dict[str, Any] | None = None,
        use_tree_trajectory_manager: bool = False,
    ):
        if prompt_length is not None and prompt_length <= 0:
            raise ValueError(f"prompt_length must be positive when set, got {prompt_length}")
        if response_length is not None and response_length <= 0:
            raise ValueError(f"response_length must be positive when set, got {response_length}")
        if type(use_tree_trajectory_manager) is not bool:
            raise ValueError("use_tree_trajectory_manager must be a bool")
        self.handle = handle
        self._codec = codec
        self._trajectory_capacity = (
            prompt_length + response_length if prompt_length is not None and response_length is not None else None
        )
        self._sampling_params = dict(sampling_params or {})
        self._metadata = dict(metadata or {})
        self._trace_identity = dict(self._metadata.get("_trace_identity") or {})
        self._manager: ChainTrajectoryManager | TreeTrajectoryManager
        if use_tree_trajectory_manager:
            self._manager = TreeTrajectoryManager(
                coalesce_reserved_exact_requests=coalesce_reserved_exact_requests,
                enable_last_assistant_rollback=enable_last_assistant_rollback,
                session_id=handle.session_id,
                log_dir=self._metadata.get("_log_dir"),
                mm_processor_kwargs=codec.mm_processor_kwargs,
            )
        else:
            self._manager = ChainTrajectoryManager(
                codec=codec,
                trajectory_capacity=self._trajectory_capacity,
                enable_last_assistant_rollback=enable_last_assistant_rollback,
                session_id=handle.session_id,
                coalesce_reserved_exact_requests=coalesce_reserved_exact_requests,
            )
        self.phase = SessionPhase.ACTIVE
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.request_lock = asyncio.Lock()

    @property
    def sampling_params(self) -> dict[str, Any]:
        return dict(self._sampling_params)

    async def run_generation(self, request: InternalGenerationRequest, backend) -> GenerationOutcome:
        """Run one generation while allowing Backend overlap within the session."""
        agent_context = request.get("agent_context")
        agent_context = agent_context.copy() if agent_context is not None else None
        generation_span = start_generation_span(self._trace_identity)
        try:
            async with self.request_lock:
                self._require_active()

            async def generate_fn(context_ids, sampling_params, image_data, video_data, mm_processor_kwargs=None):
                try:
                    if mm_processor_kwargs is None:
                        mm_processor_kwargs = self._codec.mm_processor_kwargs or {}
                    backend_kwargs = {}
                    if agent_context is not None and getattr(backend, "supports_agent_context", False) is True:
                        backend_kwargs["agent_context"] = agent_context.copy()
                    return await backend.generate(
                        request_id=self.handle.session_id,
                        prompt_ids=context_ids,
                        sampling_params=sampling_params,
                        image_data=image_data,
                        video_data=video_data,
                        mm_processor_kwargs=mm_processor_kwargs,
                        **backend_kwargs,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"{exc.__class__.__name__}: {exc}") from exc

            try:
                (
                    assistant_msg,
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    chain_id,
                    turn,
                    capacity_exhausted,
                ) = await self._manager.run_generation(
                    messages=request["messages"],
                    tools=request["tools"],
                    codec=self._codec,
                    trajectory_capacity=self._trajectory_capacity,
                    sampling_params=dict(request["sampling_params"]),
                    generate_fn=generate_fn,
                    agent_context=agent_context,
                )
            except RuntimeError as exc:
                async with self.request_lock:
                    if self.phase != SessionPhase.ACTIVE and str(exc) in (
                        f"Chain trajectory manager is {self.phase.value.lower()}",
                        "generation branch was removed before Backend completion",
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                        ) from exc
                raise
            outcome = GenerationOutcome(
                assistant_msg=assistant_msg,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            self._touch()
            if capacity_exhausted:
                generation_span.capacity_exhausted(prompt_tokens=prompt_tokens, chain_id=chain_id)
            else:
                generation_span.success(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    chain_id=chain_id,
                    turn=turn,
                    assistant_msg=assistant_msg,
                    finish_reason=finish_reason,
                )
            return outcome
        except BaseException as exc:
            generation_span.failure(exc)
            raise
        finally:
            generation_span.report()

    async def finalize(self) -> list[Trajectory]:
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                raise RuntimeError(f"Session {self.handle.session_id} is aborted")
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            trajectories = await self._manager.finalize_trajectories()
            self.phase = SessionPhase.FINALIZED
            manager_state = self._manager.snapshot_state()
            if manager_state["num_coalesced_requests"] or manager_state["rollback_count"]:
                logger.info(
                    "Session finalized with session=%s trajectories=%s num_coalesced_requests=%s "
                    "rollback_count=%s rollback_dropped_trainable_tokens_total=%s",
                    self.handle.session_id,
                    len(trajectories),
                    manager_state["num_coalesced_requests"],
                    manager_state["rollback_count"],
                    manager_state["rollback_dropped_trainable_tokens_total"],
                )
            self._touch()
            return trajectories

    async def abort(self) -> None:
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                return
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            await self._manager.abort()
            self.phase = SessionPhase.ABORTED
            self._touch()

    def snapshot_state(self) -> dict[str, Any]:
        manager_state = self._manager.snapshot_state()
        return {
            "session_id": self.handle.session_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            **manager_state,
        }

    def get_active_chains(self) -> list[dict[str, Any]]:
        return self._manager.get_active_chains()

    def _require_active(self) -> None:
        if self.phase != SessionPhase.ACTIVE:
            raise HTTPException(
                status_code=409,
                detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
            )

    def _touch(self) -> None:
        self.updated_at = time.time()
