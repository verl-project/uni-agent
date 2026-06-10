"""Per-session gateway state, generation envelope, and lifecycle handling."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from fastapi import HTTPException

from uni_agent.gateway.codec import MalformedRequestError, MessageCodec
from uni_agent.gateway.types import SessionHandle, Trajectory


class SessionPhase(str, Enum):
    """Lifecycle state for a gateway session.

    Attributes:
        ACTIVE: The session can accept generation and completion requests.
        COMPLETED: The caller marked the session complete and may finalize it.
        FINALIZED: Final trajectories were returned and the session is closed.
        ABORTED: The session was cancelled and should not produce trajectories.
    """

    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


@dataclass
class TrajectoryBuffer:
    """Mutable token buffer for the active trajectory under construction.

    Attributes:
        prompt_ids: Prompt token IDs for the current trajectory.
        response_ids: Accumulated response-side token IDs.
        response_mask: Labels aligned with ``response_ids``; ``1`` for model
            output and ``0`` for continuation context tokens.
        response_logprobs: Log probabilities aligned with ``response_ids`` when
            present; continuation context tokens use ``0.0``.
    """

    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)


@dataclass
class EncodedData:
    """Session-private data prepared before backend generation.

    The session uses this as the handoff between input preparation, backend
    generation, and the commit step. It is not an actor/runtime API.

    Attributes:
        buffer: Working trajectory buffer that becomes active only after commit.
        context_ids: Token IDs sent to the inference backend.
        sampling_params: Sampling params after request merge and budget clamp.
        messages: Normalized request messages snapshotted for commit.
        tools: Tool schemas used for both encoding and response decoding.
        image_data: Image inputs carried into backend generation and trajectory
            materialization.
        video_data: Video inputs carried into backend generation and trajectory
            materialization.
        materialized_trajectory: Previous active trajectory to append when the
            request changes context.
        length_exhausted_trajectory: Materialized trajectory for a length-budget
            early return, or ``None`` on the normal path.
    """

    buffer: TrajectoryBuffer
    context_ids: list[int]
    sampling_params: dict[str, Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    image_data: list[Any] | None
    video_data: list[Any] | None
    materialized_trajectory: Trajectory | None
    length_exhausted_trajectory: Trajectory | None


@dataclass
class GenerationOutcome:
    """Business result returned by ``GatewaySession.run_generation``.

    The session emits this instead of an HTTP response dict. ``_GatewayActor``
    converts it into the OpenAI chat-completion JSON envelope.

    Attributes:
        assistant_msg: Decoded assistant message, or an empty assistant message
            for length-exhausted early returns.
        finish_reason: Finish reason returned to the actor for serialization.
        prompt_tokens: Number of context tokens sent to the backend.
        completion_tokens: Number of generated response tokens.
    """

    assistant_msg: dict[str, Any]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class GatewaySession:
    """Behavior-bearing state container for one gateway session.

    ``_GatewayActor`` owns instances of this class, calls ``run_generation`` for
    chat requests, and delegates lifecycle operations here. The session owns the
    conversation state and trajectory materialization, while the actor owns HTTP
    routing and OpenAI response serialization.
    """

    def __init__(
        self,
        handle: SessionHandle,
        codec: MessageCodec,
        *,
        prompt_length: int | None = None,
        response_length: int | None = None,
    ):
        """Create an active session bound to a handle and model codec."""
        self.handle = handle
        self._codec = codec
        self._prompt_length = prompt_length
        self._response_length = response_length
        self.active_tool_schemas: list[dict[str, Any]] | None = None
        self.message_history: list[dict[str, Any]] = []
        self.image_data: list[Any] | None = None
        self.video_data: list[Any] | None = None
        self.active_trajectory: TrajectoryBuffer | None = None
        self.trajectories: list[Trajectory] = []
        self.reward_info: dict[str, Any] = {}
        self.completed = asyncio.Event()
        self.phase = SessionPhase.ACTIVE
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.request_lock = asyncio.Lock()
        # Serializes in-flight generation against the single-active state model.
        # This is an implementation detail, not GatewaySession's public contract.
        self.generation_lock = asyncio.Lock()

    async def run_generation(self, payload: dict[str, Any], backend) -> GenerationOutcome:
        """Run one chat-completion request and return its business outcome.

        The backend is passed in for this call only; the session does not own the
        backend lifecycle. Protocol capability checks happen in the actor before
        this method, while malformed payloads and backend errors are converted
        into HTTP exceptions here.
        """
        try:
            request_context = self._codec.normalize_request(payload)
        except MalformedRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async with self.generation_lock:
            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                encoded = await self._prepare_generation_inputs(payload, request_context)
                if encoded.length_exhausted_trajectory is not None:
                    empty_msg = {"role": "assistant", "content": ""}
                    self.trajectories.append(encoded.length_exhausted_trajectory)
                    self.active_trajectory = None
                    self.message_history = list(encoded.messages) + [empty_msg]
                    self.image_data = list(encoded.image_data) if encoded.image_data is not None else None
                    self.video_data = list(encoded.video_data) if encoded.video_data is not None else None
                    self.active_tool_schemas = encoded.tools
                    self._touch()
                    return GenerationOutcome(
                        assistant_msg=empty_msg,
                        finish_reason="length",
                        prompt_tokens=len(encoded.context_ids),
                        completion_tokens=0,
                    )

            try:
                output = await backend.generate(
                    request_id=self.handle.session_id,
                    prompt_ids=encoded.context_ids,
                    sampling_params=encoded.sampling_params,
                    image_data=encoded.image_data,
                    video_data=encoded.video_data,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}") from e

            response_ids = list(output.token_ids)
            encoded.buffer.response_ids.extend(response_ids)
            encoded.buffer.response_mask.extend([1] * len(response_ids))
            if output.log_probs is not None:
                encoded.buffer.response_logprobs.extend(list(output.log_probs))

            assistant_msg, finish_reason = await self._codec.decode_response(
                response_ids,
                tools=encoded.tools,
                stop_reason=output.stop_reason,
            )

            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                if encoded.materialized_trajectory is not None:
                    self.trajectories.append(encoded.materialized_trajectory)
                self.active_trajectory = encoded.buffer
                self.message_history = list(encoded.messages) + [assistant_msg]
                self.image_data = list(encoded.image_data) if encoded.image_data is not None else None
                self.video_data = list(encoded.video_data) if encoded.video_data is not None else None
                self.active_tool_schemas = encoded.tools
                self._touch()
                return GenerationOutcome(
                    assistant_msg=assistant_msg,
                    finish_reason=finish_reason,
                    prompt_tokens=len(encoded.context_ids),
                    completion_tokens=len(response_ids),
                )

    async def _prepare_generation_inputs(self, payload: dict[str, Any], request_context: dict[str, Any]) -> EncodedData:
        messages = request_context["messages"]
        tools = request_context["tools"]
        request_chat_template_kwargs = request_context["chat_template_kwargs"]
        materialized_trajectory = None
        image_data = None
        video_data = None

        if self.active_trajectory is None:
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            prompt_ids = self._codec.encode_full(
                messages,
                tools=tools,
                image_data=image_data,
                video_data=video_data,
                request_chat_template_kwargs=request_chat_template_kwargs,
            )
            buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
        elif self._is_request_context_prefix(messages=messages, tools=tools):
            buffer = self._copy_trajectory_buffer(self.active_trajectory)
            image_data = list(self.image_data) if self.image_data is not None else None
            video_data = list(self.video_data) if self.video_data is not None else None
            incremental_messages = messages[len(self.message_history) :]
            if incremental_messages:
                new_image_data, new_video_data = await self._codec.extract_multi_modal_data(incremental_messages)
                if new_image_data:
                    if image_data is None:
                        image_data = []
                    image_data.extend(new_image_data)
                if new_video_data:
                    if video_data is None:
                        video_data = []
                    video_data.extend(new_video_data)
                incremental_ids = self._codec.encode_incremental(
                    incremental_messages,
                    image_data=new_image_data,
                    video_data=new_video_data,
                    request_chat_template_kwargs=request_chat_template_kwargs,
                )
                if (
                    self._response_length is not None
                    and len(buffer.response_mask) + len(incremental_ids) >= self._response_length
                ):
                    context_ids = buffer.prompt_ids + buffer.response_ids
                    return EncodedData(
                        buffer=buffer,
                        context_ids=context_ids,
                        sampling_params={},
                        messages=list(messages),
                        tools=tools,
                        image_data=image_data,
                        video_data=video_data,
                        materialized_trajectory=None,
                        length_exhausted_trajectory=self._build_materialized_trajectory(
                            active=buffer,
                            extra_fields={"finish_reason": "length"},
                        ),
                    )
                buffer.response_ids.extend(incremental_ids)
                buffer.response_mask.extend([0] * len(incremental_ids))
                if buffer.response_logprobs:
                    buffer.response_logprobs.extend([0.0] * len(incremental_ids))
        else:
            materialized_trajectory = self._build_materialized_trajectory(active=self.active_trajectory)
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            prompt_ids = self._codec.encode_full(
                messages,
                tools=tools,
                image_data=image_data,
                video_data=video_data,
                request_chat_template_kwargs=request_chat_template_kwargs,
            )
            buffer = TrajectoryBuffer(prompt_ids=prompt_ids)

        context_ids = buffer.prompt_ids + buffer.response_ids
        sampling_params = self._codec.build_sampling_params(payload)
        remaining_response_budget = (
            self._response_length - len(buffer.response_mask) if self._response_length is not None else None
        )
        if remaining_response_budget is not None and "max_tokens" in sampling_params:
            sampling_params["max_tokens"] = min(sampling_params["max_tokens"], remaining_response_budget)
        return EncodedData(
            buffer=buffer,
            context_ids=context_ids,
            sampling_params=sampling_params,
            messages=list(messages),
            tools=tools,
            image_data=image_data,
            video_data=video_data,
            materialized_trajectory=materialized_trajectory,
            length_exhausted_trajectory=None,
        )

    async def complete(self, reward_info: dict[str, Any] | None = None) -> None:
        """Mark the session complete and store optional reward metadata."""
        async with self.request_lock:
            if self.phase not in {SessionPhase.COMPLETED, SessionPhase.ACTIVE}:
                raise RuntimeError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")
            if reward_info is not None:
                self.reward_info = dict(reward_info)
            self.phase = SessionPhase.COMPLETED
            self._touch()
            self.completed.set()

    async def finalize(self) -> list[Trajectory]:
        """Close the session and return its materialized trajectories with rewards."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                raise RuntimeError(f"Session {self.handle.session_id} is aborted")
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            self._touch()
            self._materialize_active_trajectory()
            self.phase = SessionPhase.FINALIZED
            self._touch()
            self.completed.set()
            return [replace(trajectory, reward_info=dict(self.reward_info)) for trajectory in self.trajectories]

    async def abort(self) -> None:
        """Abort the session and wake completion waiters."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                return
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            self.phase = SessionPhase.ABORTED
            self._touch()
            self.completed.set()

    async def wait_for_completion(self, timeout: float | None = None) -> None:
        """Wait until the session is completed, finalized, or aborted."""
        await asyncio.wait_for(self.completed.wait(), timeout=timeout)
        if self.phase == SessionPhase.ABORTED:
            raise RuntimeError(f"Session {self.handle.session_id} is aborted")

    def snapshot_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for actor state inspection."""
        return {
            "session_id": self.handle.session_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_trajectories": len(self.trajectories),
            "has_active_trajectory": self.active_trajectory is not None,
        }

    def _is_request_context_prefix(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        if self.active_tool_schemas != tools:
            return False
        history = self.message_history
        if len(history) > len(messages):
            return False
        return [self._codec.canonicalize_message_for_prefix_comparison(m) for m in history] == [
            self._codec.canonicalize_message_for_prefix_comparison(m) for m in messages[: len(history)]
        ]

    def _copy_trajectory_buffer(self, buffer: TrajectoryBuffer) -> TrajectoryBuffer:
        return TrajectoryBuffer(
            prompt_ids=list(buffer.prompt_ids),
            response_ids=list(buffer.response_ids),
            response_mask=list(buffer.response_mask),
            response_logprobs=list(buffer.response_logprobs),
        )

    def _materialize_active_trajectory(self) -> None:
        active = self.active_trajectory
        if active is None:
            return

        self._touch()
        self.trajectories.append(self._build_materialized_trajectory(active=active))
        self.active_trajectory = None

    def _build_materialized_trajectory(
        self,
        *,
        active: TrajectoryBuffer,
        extra_fields: dict[str, Any] | None = None,
    ) -> Trajectory:
        return Trajectory(
            prompt_ids=list(active.prompt_ids),
            response_ids=list(active.response_ids),
            response_mask=list(active.response_mask),
            response_logprobs=list(active.response_logprobs) if active.response_logprobs else None,
            reward_info={},
            num_turns=self._count_chat_turns(self.message_history),
            multi_modal_data=self._build_multi_modal_trajectory_data(self.image_data, self.video_data),
            extra_fields=dict(extra_fields) if extra_fields else {},
        )

    def _count_chat_turns(self, message_history: list[dict[str, Any]]) -> int:
        return sum(1 for m in message_history if m.get("role") in ("user", "assistant")) + 1

    def _build_multi_modal_trajectory_data(
        self,
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> dict[str, Any] | None:
        multi_modal_data: dict[str, Any] = {}
        if image_data:
            multi_modal_data["images"] = list(image_data)
        if video_data:
            multi_modal_data["videos"] = list(video_data)
        return multi_modal_data or None

    def _touch(self) -> None:
        self.updated_at = time.time()
