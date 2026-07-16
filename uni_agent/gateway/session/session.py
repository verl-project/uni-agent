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

from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.types import InternalGenerationRequest, SessionHandle, Trajectory

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
    message_tip_hash: str
    active_tool_schemas: list[dict[str, Any]] | None
    buffer: TrajectoryBuffer
    image_data: list[Any] | None
    video_data: list[Any] | None
    updated_seq: int


@dataclass
class MaterializedChain:
    """A closed chain plus the ordering metadata needed at finalize."""

    trajectory: Trajectory
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
        length_exhausted_trajectory: Materialized trajectory for a length-budget
            early return, or ``None`` on the normal path.
        chain_id: Selected active chain id, or ``None`` when commit should append
            a new chain.
        incoming_message_prefix_hashes: Stable prefix hashes for the normalized
            request history.
    """

    buffer: TrajectoryBuffer
    context_ids: list[int]
    sampling_params: dict[str, Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    image_data: list[Any] | None
    video_data: list[Any] | None
    length_exhausted_trajectory: Trajectory | None
    chain_id: int | None
    incoming_message_prefix_hashes: list[str] = field(default_factory=list)


@dataclass
class GenerationOutcome:
    """Business result returned by ``GatewaySession.run_generation``.

    The session emits this instead of an HTTP response dict. ``_GatewayActor``
    passes it to the provider adapter for wire response serialization.

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
    conversation state and trajectory materialization, while the actor owns
    HTTP routing and provider response serialization.
    """

    def __init__(
        self,
        handle: SessionHandle,
        codec: MessageCodec,
        *,
        prompt_length: int | None = None,
        response_length: int | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        """Create an active session bound to a handle and model codec."""
        if response_length is not None and response_length <= 0:
            raise ValueError(f"response_length must be positive when set, got {response_length}")

        self.handle = handle
        self._codec = codec
        # Provider adapters merge these trusted defaults before calling the
        # session; the response budget is enforced here during preparation.
        self._prompt_length = prompt_length
        self._response_length = response_length
        self._sampling_params = dict(sampling_params or {})
        self.active_chains: list[ChainState] = []
        self.materialized_chains: list[MaterializedChain] = []
        self.reserved_chain_ids: set[int] = set()
        self._next_chain_id = 1
        self._order_seq = 0
        self.reward_info: dict[str, Any] = {}
        self.phase = SessionPhase.ACTIVE
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.request_lock = asyncio.Lock()

    @property
    def sampling_params(self) -> dict[str, Any]:
        """Return a copy of the trusted per-session sampling defaults."""
        return dict(self._sampling_params)

    async def run_generation(self, request: InternalGenerationRequest, backend) -> GenerationOutcome:
        """Run one provider-normalized generation request and return its business outcome.

        The backend is passed in for this call only; the session does not own the
        backend lifecycle. The actor/provider adapter has already lowered the
        wire payload to the internal canonical request; session never sees raw
        wire payloads. Protocol capability checks happen in the actor before
        this method, while backend errors are converted into HTTP exceptions
        here.
        """
        # Same-session requests overlap backend generation and commit in backend
        # completion order. The framework currently scores session_trajectories[-1]
        # and broadcasts that reward, so concurrent siblings share one reward target.
        reserved_chain_id: int | None = None
        try:
            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                # Prepare can touch codec and multimodal extractor state, so only
                # backend generation runs outside the session lock.
                encoded = await self._prepare_generation_inputs(request)
                if encoded.length_exhausted_trajectory is not None:
                    empty_msg = {"role": "assistant", "content": ""}
                    self._close_length_exhausted_chain(encoded)
                    self._touch()
                    return GenerationOutcome(
                        assistant_msg=empty_msg,
                        finish_reason="length",
                        prompt_tokens=len(encoded.context_ids),
                        completion_tokens=0,
                    )
                if encoded.chain_id is not None:
                    self.reserved_chain_ids.add(encoded.chain_id)
                    reserved_chain_id = encoded.chain_id

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
            if encoded.sampling_params.get("logprobs", False):
                if output.log_probs is None:
                    raise RuntimeError("backend omitted logprobs when requested")
                log_probs = list(output.log_probs)
                if len(log_probs) != len(response_ids):
                    raise RuntimeError(
                        "backend logprobs must align with token_ids: "
                        f"got {len(log_probs)} logprobs for {len(response_ids)} tokens"
                    )
                encoded.buffer.response_logprobs.extend(log_probs)

            async with self.request_lock:
                if self.phase != SessionPhase.ACTIVE:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {self.handle.session_id} is {self.phase.value.lower()}",
                    )
                # Decode runs under request_lock so this session's prepare/commit and
                # decode stay serialized. It does not serialize decode across sessions,
                # which share the actor codec.
                assistant_msg, finish_reason = await self._codec.decode_response(
                    response_ids,
                    tools=encoded.tools,
                    stop_reason=output.stop_reason,
                )
                self._commit_generation_to_chain(encoded, assistant_msg)
                if reserved_chain_id is not None:
                    self.reserved_chain_ids.discard(reserved_chain_id)
                    reserved_chain_id = None
                self._touch()
                return GenerationOutcome(
                    assistant_msg=assistant_msg,
                    finish_reason=finish_reason,
                    prompt_tokens=len(encoded.context_ids),
                    completion_tokens=len(response_ids),
                )
        finally:
            if reserved_chain_id is not None:
                await asyncio.shield(self._release_chain_reservation(reserved_chain_id))

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
            self._materialize_active_chains()
            self.reserved_chain_ids.clear()
            self.phase = SessionPhase.FINALIZED
            self._touch()
            ordered_trajectories = [
                materialized.trajectory
                for materialized in sorted(self.materialized_chains, key=lambda chain: chain.order_seq)
            ]
            return [replace(trajectory, reward_info=dict(self.reward_info)) for trajectory in ordered_trajectories]

    async def abort(self) -> None:
        """Abort the session and prevent further generation."""
        async with self.request_lock:
            if self.phase == SessionPhase.ABORTED:
                return
            if self.phase == SessionPhase.FINALIZED:
                raise RuntimeError(f"Session {self.handle.session_id} is finalized")
            self.phase = SessionPhase.ABORTED
            self.active_chains = []
            self.materialized_chains = []
            self.reserved_chain_ids.clear()
            self._touch()

    def snapshot_state(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for actor state inspection."""
        return {
            "session_id": self.handle.session_id,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "num_trajectories": len(self.materialized_chains),
            "has_active_trajectory": bool(self.active_chains),
            "num_active_chains": len(self.active_chains),
            "active_chain_ids": [chain.chain_id for chain in self.active_chains],
            "active_chain_tip_hashes": {chain.chain_id: chain.message_tip_hash for chain in self.active_chains},
        }

    async def _prepare_generation_inputs(
        self,
        request: InternalGenerationRequest,
    ) -> EncodedData:
        messages = request["messages"]
        tools = request["tools"]
        sampling_params = dict(request["sampling_params"])
        incoming_message_prefix_hashes = self._extend_message_prefix_hashes([], messages)
        selected_chain = self._select_chain(
            tools=tools,
            incoming_message_prefix_hashes=incoming_message_prefix_hashes,
        )

        if selected_chain is None:
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            prompt_ids = self._codec.encode_full(
                messages,
                tools=tools,
                image_data=image_data,
                video_data=video_data,
            )
            buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
            chain_id = None
        else:
            buffer = self._copy_trajectory_buffer(selected_chain.buffer)
            image_data, video_data = self._copy_chain_media(selected_chain)
            chain_id = selected_chain.chain_id
            incremental_messages = messages[len(selected_chain.message_history) :]
            new_image_data = None
            new_video_data = None
            incremental_ids = []
            already_exhausted = self._response_length is not None and len(buffer.response_mask) >= self._response_length
            if incremental_messages and not already_exhausted:
                new_image_data, new_video_data = await self._codec.extract_multi_modal_data(incremental_messages)
                incremental_ids = self._codec.encode_incremental(
                    incremental_messages,
                    image_data=new_image_data,
                    video_data=new_video_data,
                )

            if already_exhausted or (
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
                    length_exhausted_trajectory=self._build_materialized_trajectory(
                        chain=selected_chain,
                        extra_fields={"materialization_reason": "max_response_length"},
                    ),
                    chain_id=selected_chain.chain_id,
                    incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
                )

            buffer.response_ids.extend(incremental_ids)
            buffer.response_mask.extend([0] * len(incremental_ids))
            if sampling_params.get("logprobs", False):
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
        remaining_response_budget = (
            self._response_length - len(buffer.response_mask) if self._response_length is not None else None
        )
        if remaining_response_budget is not None:
            sampling_params["max_tokens"] = min(
                sampling_params.get("max_tokens", remaining_response_budget),
                remaining_response_budget,
            )
        return EncodedData(
            buffer=buffer,
            context_ids=context_ids,
            sampling_params=sampling_params,
            messages=list(messages),
            tools=tools,
            image_data=image_data,
            video_data=video_data,
            length_exhausted_trajectory=None,
            chain_id=chain_id,
            incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
        )

    def _select_chain(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        incoming_message_prefix_hashes: list[str],
    ) -> ChainState | None:
        candidates = [
            chain
            for chain in self.active_chains
            if chain.chain_id not in self.reserved_chain_ids
            and chain.active_tool_schemas == tools
            and self._is_chain_prefix_hash_match(
                chain=chain,
                incoming_message_prefix_hashes=incoming_message_prefix_hashes,
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda chain: (len(chain.message_history), chain.updated_seq, chain.chain_id))

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
        return chain.message_tip_hash == incoming_message_prefix_hashes[history_len - 1]

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
                b"uni-agent-prefix-v1\0" + previous_prefix_hash.encode("ascii") + b"\0" + message_hash.encode("ascii")
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
        # Copy only the container; media payloads may not be deepcopyable.
        return list(media) if media is not None else None

    def _commit_generation_to_chain(self, encoded: EncodedData, assistant_msg: dict[str, Any]) -> None:
        message_history = list(encoded.messages) + [assistant_msg]
        message_prefix_hashes = self._extend_message_prefix_hashes(
            encoded.incoming_message_prefix_hashes,
            [assistant_msg],
        )
        assert len(message_prefix_hashes) == len(message_history)
        if encoded.chain_id is None:
            order_seq = self._next_order_seq()
            chain_id = self._allocate_chain_id()
            self.active_chains.append(
                ChainState(
                    chain_id=chain_id,
                    message_history=message_history,
                    message_tip_hash=message_prefix_hashes[-1],
                    active_tool_schemas=encoded.tools,
                    buffer=encoded.buffer,
                    image_data=self._copy_media_list(encoded.image_data),
                    video_data=self._copy_media_list(encoded.video_data),
                    updated_seq=order_seq,
                )
            )
            return

        chain_index, previous_chain = self._find_active_chain(encoded.chain_id)
        order_seq = self._next_order_seq()
        self.active_chains[chain_index] = ChainState(
            chain_id=previous_chain.chain_id,
            message_history=message_history,
            message_tip_hash=message_prefix_hashes[-1],
            active_tool_schemas=encoded.tools,
            buffer=encoded.buffer,
            image_data=self._copy_media_list(encoded.image_data),
            video_data=self._copy_media_list(encoded.video_data),
            updated_seq=order_seq,
        )

    def _close_length_exhausted_chain(self, encoded: EncodedData) -> None:
        if encoded.chain_id is None or encoded.length_exhausted_trajectory is None:
            raise RuntimeError("length-exhausted chain metadata is missing")
        chain_index, chain = self._find_active_chain(encoded.chain_id)
        order_seq = self._next_order_seq()
        self.materialized_chains.append(
            MaterializedChain(
                trajectory=encoded.length_exhausted_trajectory,
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

    async def _release_chain_reservation(self, chain_id: int) -> None:
        async with self.request_lock:
            self.reserved_chain_ids.discard(chain_id)

    def _next_order_seq(self) -> int:
        self._order_seq += 1
        return self._order_seq

    def _materialize_active_chains(self) -> None:
        for chain in self.active_chains:
            self.materialized_chains.append(
                MaterializedChain(
                    trajectory=self._build_materialized_trajectory(chain=chain),
                    order_seq=chain.updated_seq,
                )
            )
        self.active_chains = []

    def _build_materialized_trajectory(
        self,
        *,
        chain: ChainState,
        extra_fields: dict[str, Any] | None = None,
    ) -> Trajectory:
        response_logprobs = None
        if chain.buffer.response_logprobs and len(chain.buffer.response_logprobs) == len(chain.buffer.response_ids):
            response_logprobs = list(chain.buffer.response_logprobs)
        return Trajectory(
            prompt_ids=list(chain.buffer.prompt_ids),
            response_ids=list(chain.buffer.response_ids),
            response_mask=list(chain.buffer.response_mask),
            response_logprobs=response_logprobs,
            reward_info={},
            num_turns=self._count_chat_turns(chain.message_history),
            multi_modal_data=self._build_multi_modal_trajectory_data(
                chain.image_data,
                chain.video_data,
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
