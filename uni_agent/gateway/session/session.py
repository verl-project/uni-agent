"""Per-session gateway state, generation envelope, and lifecycle handling."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from fastapi import HTTPException

from uni_agent.gateway.session.codec import MalformedRequestError, MessageCodec
from uni_agent.gateway.session.types import SessionHandle, Trajectory


_EMPTY_PREFIX_HASH = hashlib.sha256(b"uni-agent-prefix-v1\0empty").hexdigest()


class SessionPhase(str, Enum):
    """Lifecycle state for a gateway session.

    Attributes:
        ACTIVE: The session can accept generation and reward-info requests.
        FINALIZED: Final trajectories were returned and the session is closed.
        ABORTED: The session was cancelled and should not produce trajectories.
    """

    ACTIVE = "ACTIVE"
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
class ChainState:
    """One active linear trajectory chain in a gateway session."""

    chain_id: int
    message_history: list[dict[str, Any]]
    message_prefix_hashes: list[str]
    active_tool_schemas: list[dict[str, Any]] | None
    effective_chat_template_kwargs: dict[str, Any]
    buffer: TrajectoryBuffer
    image_data: list[Any] | None
    video_data: list[Any] | None
    logprobs_complete: bool
    created_seq: int
    updated_seq: int


@dataclass
class MaterializedChain:
    """A closed chain plus the ordering metadata needed at finalize."""

    chain_id: int
    trajectory: Trajectory
    created_seq: int
    updated_seq: int
    order_seq: int


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
        chain_id: Selected active chain id in multiple-chain mode.
        is_new_chain: Whether commit should append a new chain.
        effective_chat_template_kwargs: Effective chat-template kwargs used for
            chain compatibility and encoding.
        incoming_message_prefix_hashes: Stable prefix hashes for the normalized
            request history.
        logprobs_complete: Whether response logprobs are still complete for the
            working chain.
        length_exhausted_chain_id: Selected chain to close on a multiple-chain
            length-budget early return.
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
    chain_id: int | None = None
    is_new_chain: bool = False
    effective_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    incoming_message_prefix_hashes: list[str] = field(default_factory=list)
    logprobs_complete: bool = True
    length_exhausted_chain_id: int | None = None


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
        enable_multiple_chains: bool = False,
    ):
        """Create an active session bound to a handle and model codec."""
        self.handle = handle
        self._codec = codec
        self._prompt_length = prompt_length
        self._response_length = response_length
        self.enable_multiple_chains = enable_multiple_chains
        self.active_tool_schemas: list[dict[str, Any]] | None = None
        self.message_history: list[dict[str, Any]] = []
        self.image_data: list[Any] | None = None
        self.video_data: list[Any] | None = None
        self.active_trajectory: TrajectoryBuffer | None = None
        self.trajectories: list[Trajectory] = []
        self.active_chains: list[ChainState] = []
        self.materialized_chains: list[MaterializedChain] = []
        self._next_chain_id = 1
        self._order_seq = 0
        self.reward_info: dict[str, Any] = {}
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
                if self.enable_multiple_chains:
                    encoded = await self._prepare_generation_inputs_multiple_chains(payload, request_context)
                else:
                    encoded = await self._prepare_generation_inputs(payload, request_context)
                if encoded.length_exhausted_trajectory is not None:
                    empty_msg = {"role": "assistant", "content": ""}
                    if self.enable_multiple_chains:
                        self._close_length_exhausted_chain(encoded)
                    else:
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
                backend_image_data = encoded.image_data
                backend_video_data = encoded.video_data
                if self.enable_multiple_chains:
                    backend_image_data = self._copy_media_list(encoded.image_data)
                    backend_video_data = self._copy_media_list(encoded.video_data)

                output = await backend.generate(
                    request_id=self.handle.session_id,
                    prompt_ids=encoded.context_ids,
                    sampling_params=encoded.sampling_params,
                    image_data=backend_image_data,
                    video_data=backend_video_data,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"{e.__class__.__name__}: {e}") from e

            response_ids = list(output.token_ids)
            encoded.buffer.response_ids.extend(response_ids)
            encoded.buffer.response_mask.extend([1] * len(response_ids))
            if self.enable_multiple_chains:
                self._append_output_logprobs_multiple_chains(encoded, output.log_probs, response_ids)
            elif output.log_probs is not None:
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
                if self.enable_multiple_chains:
                    self._commit_generation_to_chain(encoded, assistant_msg)
                else:
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
            max(0, self._response_length - len(buffer.response_mask)) if self._response_length is not None else None
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

    async def _prepare_generation_inputs_multiple_chains(
        self,
        payload: dict[str, Any],
        request_context: dict[str, Any],
    ) -> EncodedData:
        messages = request_context["messages"]
        tools = request_context["tools"]
        request_chat_template_kwargs = request_context["chat_template_kwargs"]
        effective_chat_template_kwargs = self._codec.effective_chat_template_kwargs(request_chat_template_kwargs)
        incoming_message_prefix_hashes = self._compute_message_prefix_hashes(messages)
        selected_chain = self._select_chain(
            messages=messages,
            tools=tools,
            request_effective_chat_template_kwargs=effective_chat_template_kwargs,
            incoming_message_prefix_hashes=incoming_message_prefix_hashes,
        )

        if selected_chain is None:
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            prompt_ids = self._codec.encode_full(
                messages,
                tools=tools,
                image_data=image_data,
                video_data=video_data,
                request_chat_template_kwargs=request_chat_template_kwargs,
            )
            buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
            chain_id = None
            is_new_chain = True
            logprobs_complete = True
        else:
            buffer = self._copy_trajectory_buffer(selected_chain.buffer)
            image_data, video_data = self._copy_chain_media(selected_chain)
            chain_id = selected_chain.chain_id
            is_new_chain = False
            logprobs_complete = selected_chain.logprobs_complete
            incremental_messages = messages[len(selected_chain.message_history) :]
            if incremental_messages:
                new_image_data, new_video_data = await self._codec.extract_multi_modal_data(incremental_messages)
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
                        length_exhausted_trajectory=self._build_materialized_chain_trajectory(
                            chain=selected_chain,
                            extra_fields={"finish_reason": "length"},
                        ),
                        chain_id=selected_chain.chain_id,
                        is_new_chain=False,
                        effective_chat_template_kwargs=effective_chat_template_kwargs,
                        incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
                        logprobs_complete=logprobs_complete,
                        length_exhausted_chain_id=selected_chain.chain_id,
                    )
                buffer.response_ids.extend(incremental_ids)
                buffer.response_mask.extend([0] * len(incremental_ids))
                if logprobs_complete:
                    buffer.response_logprobs.extend([0.0] * len(incremental_ids))
                if new_image_data:
                    if image_data is None:
                        image_data = []
                    image_data.extend(new_image_data)
                if new_video_data:
                    if video_data is None:
                        video_data = []
                    video_data.extend(new_video_data)

        context_ids = buffer.prompt_ids + buffer.response_ids
        sampling_params = self._codec.build_sampling_params(payload)
        remaining_response_budget = (
            max(0, self._response_length - len(buffer.response_mask)) if self._response_length is not None else None
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
            materialized_trajectory=None,
            length_exhausted_trajectory=None,
            chain_id=chain_id,
            is_new_chain=is_new_chain,
            effective_chat_template_kwargs=effective_chat_template_kwargs,
            incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
            logprobs_complete=logprobs_complete,
        )

    async def set_reward_info(self, reward_info: dict[str, Any] | None = None) -> None:
        """Store session-level reward metadata without closing the session."""
        async with self.request_lock:
            if self.phase != SessionPhase.ACTIVE:
                raise RuntimeError(f"Session {self.handle.session_id} is {self.phase.value.lower()}")
            if reward_info is not None:
                self.reward_info = dict(reward_info)
            self._touch()

    async def finalize(self) -> list[Trajectory]:
        """Close the session and return its materialized trajectories with rewards."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                raise RuntimeError(f"Session {self.handle.session_id} is aborted")
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            self._touch()
            if self.enable_multiple_chains:
                self._materialize_active_chains()
            else:
                self._materialize_active_trajectory()
            self.phase = SessionPhase.FINALIZED
            self._touch()
            if self.enable_multiple_chains:
                ordered_trajectories = [
                    materialized.trajectory
                    for materialized in sorted(self.materialized_chains, key=lambda chain: chain.order_seq)
                ]
            else:
                ordered_trajectories = self.trajectories
            return [replace(trajectory, reward_info=dict(self.reward_info)) for trajectory in ordered_trajectories]

    async def abort(self) -> None:
        """Abort the session and prevent further generation."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                return
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            self.phase = SessionPhase.ABORTED
            if self.enable_multiple_chains:
                self.active_chains = []
            self._touch()

    def snapshot_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for actor state inspection."""
        if self.enable_multiple_chains:
            return {
                "session_id": self.handle.session_id,
                "phase": self.phase.value,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "num_trajectories": len(self.materialized_chains),
                "has_active_trajectory": bool(self.active_chains),
                "num_active_chains": len(self.active_chains),
                "active_chain_ids": [chain.chain_id for chain in self.active_chains],
                "active_chain_tip_hashes": {
                    chain.chain_id: (
                        chain.message_prefix_hashes[-1] if chain.message_prefix_hashes else _EMPTY_PREFIX_HASH
                    )
                    for chain in self.active_chains
                },
            }
        return {
            "session_id": self.handle.session_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_trajectories": len(self.trajectories),
            "has_active_trajectory": self.active_trajectory is not None,
            "num_active_chains": 0,
            "active_chain_ids": [],
            "active_chain_tip_hashes": {},
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

    def _select_chain(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_effective_chat_template_kwargs: dict[str, Any],
        incoming_message_prefix_hashes: list[str],
    ) -> ChainState | None:
        del messages  # Prefix hashes already encode the normalized message history.
        candidates = [
            chain
            for chain in self.active_chains
            if self._is_chain_request_compatible(
                chain=chain,
                tools=tools,
                request_effective_chat_template_kwargs=request_effective_chat_template_kwargs,
            )
            and self._is_chain_prefix_hash_match(
                chain=chain,
                incoming_message_prefix_hashes=incoming_message_prefix_hashes,
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda chain: (len(chain.message_history), chain.updated_seq, chain.chain_id))

    def _is_chain_request_compatible(
        self,
        *,
        chain: ChainState,
        tools: list[dict[str, Any]] | None,
        request_effective_chat_template_kwargs: dict[str, Any],
    ) -> bool:
        return (
            chain.active_tool_schemas == tools
            and chain.effective_chat_template_kwargs == request_effective_chat_template_kwargs
        )

    def _is_chain_prefix_hash_match(
        self,
        *,
        chain: ChainState,
        incoming_message_prefix_hashes: list[str],
    ) -> bool:
        history_len = len(chain.message_history)
        if history_len > len(incoming_message_prefix_hashes):
            return False
        if history_len == 0:
            return True
        if len(chain.message_prefix_hashes) != history_len:
            return False
        return chain.message_prefix_hashes[-1] == incoming_message_prefix_hashes[history_len - 1]

    def _compute_message_prefix_hashes(self, messages: list[dict[str, Any]]) -> list[str]:
        prefix_hashes: list[str] = []
        previous_prefix_hash = _EMPTY_PREFIX_HASH
        for message in messages:
            message_hash = self._compute_message_hash(message)
            prefix_hash = hashlib.sha256(
                b"uni-agent-prefix-v1\0"
                + previous_prefix_hash.encode("ascii")
                + b"\0"
                + message_hash.encode("ascii")
            ).hexdigest()
            prefix_hashes.append(prefix_hash)
            previous_prefix_hash = prefix_hash
        return prefix_hashes

    def _extend_message_prefix_hashes(
        self,
        existing_prefix_hashes: list[str],
        new_messages: list[dict[str, Any]],
    ) -> list[str]:
        prefix_hashes = list(existing_prefix_hashes)
        previous_prefix_hash = prefix_hashes[-1] if prefix_hashes else _EMPTY_PREFIX_HASH
        for message in new_messages:
            message_hash = self._compute_message_hash(message)
            prefix_hash = hashlib.sha256(
                b"uni-agent-prefix-v1\0"
                + previous_prefix_hash.encode("ascii")
                + b"\0"
                + message_hash.encode("ascii")
            ).hexdigest()
            prefix_hashes.append(prefix_hash)
            previous_prefix_hash = prefix_hash
        return prefix_hashes

    def _compute_message_hash(self, message: dict[str, Any]) -> str:
        canonical = self._codec.canonicalize_message_for_prefix_comparison(message)
        canonical_json = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(b"uni-agent-message-v1\0" + canonical_json).hexdigest()

    def _copy_trajectory_buffer(self, buffer: TrajectoryBuffer) -> TrajectoryBuffer:
        return TrajectoryBuffer(
            prompt_ids=list(buffer.prompt_ids),
            response_ids=list(buffer.response_ids),
            response_mask=list(buffer.response_mask),
            response_logprobs=list(buffer.response_logprobs),
        )

    def _copy_chain_media(self, chain: ChainState) -> tuple[list[Any] | None, list[Any] | None]:
        return (
            self._copy_media_list(chain.image_data),
            self._copy_media_list(chain.video_data),
        )

    def _copy_media_list(self, media: list[Any] | None) -> list[Any] | None:
        return list(media) if media is not None else None

    def _append_output_logprobs_multiple_chains(
        self,
        encoded: EncodedData,
        output_log_probs: list[float] | None,
        response_ids: list[int],
    ) -> None:
        if output_log_probs is None:
            encoded.logprobs_complete = False
            return
        log_probs = list(output_log_probs)
        if len(log_probs) != len(response_ids):
            encoded.logprobs_complete = False
            return
        if encoded.logprobs_complete:
            encoded.buffer.response_logprobs.extend(log_probs)

    def _commit_generation_to_chain(self, encoded: EncodedData, assistant_msg: dict[str, Any]) -> None:
        message_history = list(encoded.messages) + [assistant_msg]
        message_prefix_hashes = self._extend_message_prefix_hashes(
            encoded.incoming_message_prefix_hashes,
            [assistant_msg],
        )
        assert len(message_prefix_hashes) == len(message_history)
        if encoded.is_new_chain:
            order_seq = self._next_order_seq()
            chain_id = self._allocate_chain_id()
            self.active_chains.append(
                ChainState(
                    chain_id=chain_id,
                    message_history=message_history,
                    message_prefix_hashes=message_prefix_hashes,
                    active_tool_schemas=encoded.tools,
                    effective_chat_template_kwargs=dict(encoded.effective_chat_template_kwargs),
                    buffer=encoded.buffer,
                    image_data=self._copy_media_list(encoded.image_data),
                    video_data=self._copy_media_list(encoded.video_data),
                    logprobs_complete=encoded.logprobs_complete,
                    created_seq=order_seq,
                    updated_seq=order_seq,
                )
            )
            return

        if encoded.chain_id is None:
            raise RuntimeError("selected chain id is missing")
        chain_index, previous_chain = self._find_active_chain(encoded.chain_id)
        order_seq = self._next_order_seq()
        self.active_chains[chain_index] = ChainState(
            chain_id=previous_chain.chain_id,
            message_history=message_history,
            message_prefix_hashes=message_prefix_hashes,
            active_tool_schemas=encoded.tools,
            effective_chat_template_kwargs=dict(encoded.effective_chat_template_kwargs),
            buffer=encoded.buffer,
            image_data=self._copy_media_list(encoded.image_data),
            video_data=self._copy_media_list(encoded.video_data),
            logprobs_complete=encoded.logprobs_complete,
            created_seq=previous_chain.created_seq,
            updated_seq=order_seq,
        )

    def _close_length_exhausted_chain(self, encoded: EncodedData) -> None:
        if encoded.length_exhausted_chain_id is None or encoded.length_exhausted_trajectory is None:
            raise RuntimeError("length-exhausted chain metadata is missing")
        chain_index, chain = self._find_active_chain(encoded.length_exhausted_chain_id)
        order_seq = self._next_order_seq()
        self.materialized_chains.append(
            MaterializedChain(
                chain_id=chain.chain_id,
                trajectory=encoded.length_exhausted_trajectory,
                created_seq=chain.created_seq,
                updated_seq=chain.updated_seq,
                order_seq=order_seq,
            )
        )
        del self.active_chains[chain_index]

    def _find_active_chain(self, chain_id: int) -> tuple[int, ChainState]:
        for index, chain in enumerate(self.active_chains):
            if chain.chain_id == chain_id:
                return index, chain
        raise RuntimeError(f"active chain {chain_id} not found")

    def _allocate_chain_id(self) -> int:
        chain_id = self._next_chain_id
        self._next_chain_id += 1
        return chain_id

    def _next_order_seq(self) -> int:
        self._order_seq += 1
        return self._order_seq

    def _materialize_active_trajectory(self) -> None:
        active = self.active_trajectory
        if active is None:
            return

        self._touch()
        self.trajectories.append(self._build_materialized_trajectory(active=active))
        self.active_trajectory = None

    def _materialize_active_chains(self) -> None:
        for chain in self.active_chains:
            self.materialized_chains.append(
                MaterializedChain(
                    chain_id=chain.chain_id,
                    trajectory=self._build_materialized_chain_trajectory(chain=chain),
                    created_seq=chain.created_seq,
                    updated_seq=chain.updated_seq,
                    order_seq=chain.updated_seq,
                )
            )
        self.active_chains = []

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

    def _build_materialized_chain_trajectory(
        self,
        *,
        chain: ChainState,
        extra_fields: dict[str, Any] | None = None,
    ) -> Trajectory:
        response_logprobs = None
        if chain.logprobs_complete and len(chain.buffer.response_logprobs) == len(chain.buffer.response_ids):
            response_logprobs = list(chain.buffer.response_logprobs)
        return Trajectory(
            prompt_ids=list(chain.buffer.prompt_ids),
            response_ids=list(chain.buffer.response_ids),
            response_mask=list(chain.buffer.response_mask),
            response_logprobs=response_logprobs,
            reward_info={},
            num_turns=self._count_chat_turns(chain.message_history),
            multi_modal_data=self._build_multi_modal_trajectory_data(
                self._copy_media_list(chain.image_data),
                self._copy_media_list(chain.video_data),
            ),
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
