"""Linear-chain trajectory management extracted from the 0921 GatewaySession.

Chain selection, token preparation, rollback, capacity materialization, and
trajectory construction retain the baseline implementation. GatewaySession
owns backend error mapping and generation tracing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.types import AgentContext, InternalGenerationRequest, Trajectory, UnsupportedMediaError

_EMPTY_PREFIX_HASH = hashlib.sha256(b"uni-agent-prefix-v1\0empty").hexdigest()
logger = logging.getLogger(__name__)
_TRAJECTORY_CAPACITY_HINT = (
    "The trajectory limit is prompt_length + response_length. "
    "For this event, Claude Code's output-token-limit error is misleading: increasing "
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS alone will not help. Check the rollout prompt_length/response_length "
    "budget or compact context earlier; the episode may be unfinished."
)


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
        routed_experts: Latest per-token expert-routing tensor from the backend,
            spanning ``prompt + response``. The backend re-prefills the full
            context each turn, so this is replaced (not accumulated) and the final
            value covers the whole sequence (mirrors verl's tool_agent_loop).
        generation_versions: One ``(min_global_steps, max_global_steps)`` mark
            per generation, in generation order. Rollback drops the last mark
            with the tokens it describes, so a dropped assistant no longer
            widens the trajectory's version span. Marks carry ``None`` when the
            backend reports no version.
    """

    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    routed_experts: Any | None = None
    generation_versions: list[tuple[int | None, int | None]] = field(default_factory=list)


@dataclass
class LastAssistantStart:
    """Stable chain lengths captured immediately before its latest assistant."""

    response_ids_len: int
    message_history_len: int
    image_data_len: int
    video_data_len: int
    tip_hash: str


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
    last_assistant_start: LastAssistantStart
    updated_seq: int
    agent_context: AgentContext | None = None
    num_coalesced_requests: int = 0
    rollback_count: int = 0
    rollback_dropped_trainable_tokens_total: int = 0


@dataclass
class InflightRequest:
    """Owner result and waiters to attribute when its generation commits."""

    future: asyncio.Future[tuple[bool, Any]]
    waiter_count: int = 0


@dataclass
class MaterializedChain:
    """A closed chain plus the ordering metadata needed at finalize."""

    trajectory: Trajectory
    order_seq: int


@dataclass
class EncodedData:
    """Chain-manager-private data prepared before backend generation.

    The manager uses this as the handoff between input preparation, backend
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
        mm_processor_kwargs: Processor kwargs shared by prompt encoding and
            backend multimodal preprocessing.
        capacity_exhausted: Whether preparation determined that no generation
            can fit within the total trajectory capacity.
        chain_id: Selected active chain id, or ``None`` when commit should append
            a new chain.
        incoming_message_prefix_hashes: Stable prefix hashes for the normalized
            request history.
        last_assistant_start: Lengths immediately before this generation's
            assistant output is appended.
        rollback_applied: Whether prepare removed the selected chain's latest
            assistant before re-encoding the incoming suffix.
        rollback_dropped_trainable_tokens: Number of mask=1 tokens removed by
            that rollback.
    """

    buffer: TrajectoryBuffer
    context_ids: list[int]
    sampling_params: dict[str, Any]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    image_data: list[Any] | None
    video_data: list[Any] | None
    mm_processor_kwargs: dict[str, Any]
    capacity_exhausted: bool
    chain_id: int | None
    agent_context: AgentContext | None = None
    incoming_message_prefix_hashes: list[str] = field(default_factory=list)
    last_assistant_start: LastAssistantStart | None = None
    rollback_applied: bool = False
    rollback_dropped_trainable_tokens: int = 0


class ChainTrajectoryManager:
    """Own the baseline's active, reserved, and materialized linear chains."""

    def __init__(
        self,
        codec: MessageCodec,
        trajectory_capacity: int | None = None,
        enable_last_assistant_rollback: bool = True,
        session_id: str = "",
        coalesce_reserved_exact_requests: bool = True,
    ) -> None:
        self._session_id = session_id
        self._codec = codec
        self._trajectory_capacity = trajectory_capacity
        self._enable_last_assistant_rollback = enable_last_assistant_rollback
        self._coalesce_reserved_exact_requests = coalesce_reserved_exact_requests
        self._inflight_exact_requests: dict[str, InflightRequest] = {}
        self.active_chains: list[ChainState] = []
        self.materialized_chains: list[MaterializedChain] = []
        self.reserved_chain_ids: set[int] = set()
        self._next_chain_id = 1
        self._order_seq = 0
        self._rollback_count = 0
        self._rollback_dropped_trainable_tokens_total = 0
        self._coalesced_request_count = 0
        self.request_lock = asyncio.Lock()
        self._phase = "ACTIVE"

    def _require_active(self) -> None:
        if self._phase != "ACTIVE":
            raise RuntimeError(f"Chain trajectory manager is {self._phase.lower()}")

    async def run_generation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        codec: MessageCodec,
        trajectory_capacity: int | None,
        sampling_params: dict[str, Any],
        generate_fn: Callable[
            [list[int], dict[str, Any], list[Any] | None, list[Any] | None, dict[str, Any]], Awaitable[Any]
        ],
        agent_context: AgentContext | None = None,
    ) -> tuple[dict[str, Any], str, int, int, int | None, int, bool]:
        """Adapt the baseline prepare/generate/commit transaction to Session's callback."""
        if codec is not self._codec or trajectory_capacity != self._trajectory_capacity:
            raise ValueError("Chain manager codec and capacity must match its session")
        request: InternalGenerationRequest = {
            "messages": messages,
            "tools": tools,
            "sampling_params": dict(sampling_params),
        }
        if agent_context is not None:
            request["agent_context"] = agent_context.copy()
        request_fingerprint = (
            hashlib.sha256(
                b"uni-agent-request-v1\0"
                + json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if self._coalesce_reserved_exact_requests
            else None
        )
        owner_request = None
        joined_request = None
        reserved_chain_id: int | None = None
        try:
            async with self.request_lock:
                self._require_active()
                if request_fingerprint is not None:
                    joined_request = self._inflight_exact_requests.get(request_fingerprint)
                    if joined_request is not None:
                        joined_request.waiter_count += 1
                if joined_request is None:
                    encoded = await self._prepare_generation_inputs(request)
                    if encoded.capacity_exhausted:
                        logger.warning(
                            "Trajectory capacity prevents generation (max_trajectory_length): "
                            "session=%s chain_id=%s trajectory_capacity=%s requested_max_tokens=%s "
                            "effective_max_tokens=0 completion_tokens=0 backend_called=false. %s",
                            self._session_id,
                            encoded.chain_id,
                            self._trajectory_capacity,
                            request["sampling_params"].get("max_tokens"),
                            _TRAJECTORY_CAPACITY_HINT,
                        )
                        if encoded.chain_id is not None:
                            self._close_length_exhausted_chain(encoded)
                        return (
                            {"role": "assistant", "content": ""},
                            "length",
                            len(encoded.context_ids),
                            0,
                            encoded.chain_id,
                            self._order_seq,
                            True,
                        )
                    if encoded.chain_id is not None:
                        self.reserved_chain_ids.add(encoded.chain_id)
                        reserved_chain_id = encoded.chain_id
                    # Match 0921: preparation and reservation must succeed before
                    # publishing ownership, including for first turns/new chains.
                    if request_fingerprint is not None:
                        owner_request = InflightRequest(asyncio.get_running_loop().create_future())
                        self._inflight_exact_requests[request_fingerprint] = owner_request

            if joined_request is not None:
                self._coalesced_request_count += 1
                if self._coalesced_request_count == 1:
                    logger.warning(
                        "Exact in-flight request coalesced: session=%s fingerprint=%s role=waiter. "
                        "The waiter shares the owner's backend generation and trajectory result. "
                        "Set coalesce_reserved_exact_requests=false to preserve independent "
                        "same-session sampling.",
                        self._session_id,
                        request_fingerprint[:12],
                    )
                succeeded, value = await asyncio.shield(joined_request.future)
                if not succeeded:
                    raise value
                return value

            output = await generate_fn(
                encoded.context_ids,
                encoded.sampling_params,
                encoded.image_data,
                encoded.video_data,
                encoded.mm_processor_kwargs,
            )
            response_ids = list(output.token_ids)
            assistant_logprobs = None
            if encoded.sampling_params.get("logprobs", False):
                if output.log_probs is None:
                    raise RuntimeError("backend omitted logprobs when requested")
                assistant_logprobs = list(output.log_probs)
                if len(assistant_logprobs) != len(response_ids):
                    raise RuntimeError(
                        "backend logprobs must align with token_ids: "
                        f"got {len(assistant_logprobs)} logprobs for {len(response_ids)} tokens"
                    )

            runtime_token_ids = encoded.buffer.prompt_ids + encoded.buffer.response_ids
            merged_token_ids, response_mask, response_logprobs = self._codec.merge_assistant_tokens(
                runtime_token_ids,
                response_ids,
                encoded.buffer.response_mask,
                encoded.buffer.response_logprobs if encoded.sampling_params.get("logprobs", False) else None,
                assistant_logprobs=assistant_logprobs,
            )
            prompt_length = len(encoded.buffer.prompt_ids)
            if merged_token_ids[:prompt_length] != encoded.buffer.prompt_ids:
                raise RuntimeError("Continuous Token merge modified the immutable trajectory prompt")
            encoded.buffer.response_ids = list(merged_token_ids[prompt_length:])
            encoded.buffer.response_mask = list(response_mask)
            encoded.buffer.response_logprobs = list(response_logprobs or [])
            encoded.buffer.generation_versions.append(
                (
                    output.extra_fields.get("min_global_steps"),
                    output.extra_fields.get("max_global_steps"),
                )
            )
            self._assert_response_logprob_alignment(encoded.buffer)

            # R3 router replay: the backend returns routing for the full context
            # it just prefilled (prompt + response so far + new tokens), so keep
            # the latest value; it supersedes prior turns. The framework aligns it
            # to input_ids when writing to TransferQueue.
            routed_experts = getattr(output, "routed_experts", None)
            if routed_experts is not None:
                encoded.buffer.routed_experts = routed_experts

            async with self.request_lock:
                self._require_active()
                assistant_msg, finish_reason = await self._codec.decode_response(
                    response_ids,
                    tools=encoded.tools,
                    stop_reason=output.stop_reason,
                )
                chain_id = self._commit_generation_to_chain(encoded, assistant_msg)
                if owner_request is not None and owner_request.waiter_count:
                    # Attribute at commit, even if waiters cancel or resume after finalize.
                    _, chain = self._find_active_chain(chain_id)
                    chain.num_coalesced_requests += owner_request.waiter_count
                if (
                    finish_reason == "length"
                    and self._trajectory_capacity is not None
                    and len(encoded.context_ids) + len(response_ids) >= self._trajectory_capacity
                ):
                    logger.warning(
                        "Generation reached trajectory capacity (max_trajectory_length): "
                        "session=%s chain_id=%s trajectory_capacity=%s context_tokens=%s "
                        "requested_max_tokens=%s effective_max_tokens=%s completion_tokens=%s backend_called=true. %s",
                        self._session_id,
                        chain_id,
                        self._trajectory_capacity,
                        len(encoded.context_ids),
                        request["sampling_params"].get("max_tokens"),
                        encoded.sampling_params["max_tokens"],
                        len(response_ids),
                        _TRAJECTORY_CAPACITY_HINT,
                    )
                if reserved_chain_id is not None:
                    self.reserved_chain_ids.discard(reserved_chain_id)
                    reserved_chain_id = None
                result = (
                    assistant_msg,
                    finish_reason,
                    len(encoded.context_ids),
                    len(response_ids),
                    chain_id,
                    self._order_seq,
                    False,
                )
                if owner_request is not None and request_fingerprint is not None:
                    if self._inflight_exact_requests.get(request_fingerprint) is owner_request:
                        self._inflight_exact_requests.pop(request_fingerprint, None)
                        if not owner_request.future.done():
                            owner_request.future.set_result((True, result))
                return result
        except BaseException as exc:
            if owner_request is not None and request_fingerprint is not None:
                # Release the chain before waking waiters so an immediate retry
                # can reuse it rather than fork a sibling.
                if reserved_chain_id is not None:
                    await asyncio.shield(self._release_chain_reservation(reserved_chain_id))
                    reserved_chain_id = None
                if self._inflight_exact_requests.get(request_fingerprint) is owner_request:
                    self._inflight_exact_requests.pop(request_fingerprint, None)
                    if not owner_request.future.done():
                        owner_request.future.set_result((False, exc))
            raise
        finally:
            if reserved_chain_id is not None:
                await asyncio.shield(self._release_chain_reservation(reserved_chain_id))

    async def finalize_trajectories(self) -> list[Trajectory]:
        async with self.request_lock:
            self._require_active()
            self._materialize_active_chains()
            self.reserved_chain_ids.clear()
            self._phase = "FINALIZED"
            return [
                materialized.trajectory
                for materialized in sorted(self.materialized_chains, key=lambda chain: chain.order_seq)
            ]

    async def abort(self) -> None:
        """Abort generation and discard retained chain trajectories."""
        async with self.request_lock:
            if self._phase == "ABORTED":
                return
            self._require_active()
            self._phase = "ABORTED"
            self.active_chains = []
            self.materialized_chains = []
            self.reserved_chain_ids.clear()

    def snapshot_state(self) -> dict[str, Any]:
        """Read in-memory chain metadata synchronously, as in the 0921 session."""
        return {
            "num_trajectories": len(self.materialized_chains),
            "has_active_trajectory": bool(self.active_chains),
            "num_active_chains": len(self.active_chains),
            "active_chain_ids": [chain.chain_id for chain in self.active_chains],
            "active_chain_tip_hashes": {chain.chain_id: chain.message_tip_hash for chain in self.active_chains},
            "rollback_count": self._rollback_count,
            "rollback_dropped_trainable_tokens_total": self._rollback_dropped_trainable_tokens_total,
            "num_coalesced_requests": self._coalesced_request_count,
        }

    def get_active_chains(self) -> list[dict[str, Any]]:
        return [
            {
                "chain_id": chain.chain_id,
                "message_history": [dict(message) for message in chain.message_history],
                "active_tool_schemas": (
                    [dict(tool) for tool in chain.active_tool_schemas]
                    if chain.active_tool_schemas is not None
                    else None
                ),
            }
            for chain in self.active_chains
        ]

    async def _prepare_generation_inputs(
        self,
        request: InternalGenerationRequest,
    ) -> EncodedData:
        messages = request["messages"]
        tools = request["tools"]
        sampling_params = dict(request["sampling_params"])
        mm_processor_kwargs = self._codec.mm_processor_kwargs or {}
        incoming_message_prefix_hashes = self._extend_message_prefix_hashes([], messages)
        selection = self._select_chain(
            incoming_assistant_count=sum(message["role"] == "assistant" for message in messages),
            tools=tools,
            incoming_message_prefix_hashes=incoming_message_prefix_hashes,
            agent_context=request.get("agent_context"),
        )
        rollback_applied = False
        rollback_dropped_trainable_tokens = 0

        if selection is None:
            image_data, video_data = await self._codec.extract_multi_modal_data(messages)
            prompt_ids = self._codec.build_initial_tokens(
                messages,
                tools=tools,
                image_data=image_data,
                video_data=video_data,
            )
            buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
            chain_id = None
            capacity_exhausted = self._trajectory_capacity is not None and len(prompt_ids) >= self._trajectory_capacity
        else:
            selected_chain, rollback_to_last_assistant = selection
            buffer = self._copy_trajectory_buffer(selected_chain.buffer)
            self._assert_response_logprob_alignment(buffer)
            image_data, video_data = self._copy_chain_media(selected_chain)
            chain_id = selected_chain.chain_id
            if rollback_to_last_assistant:
                last_assistant_start = selected_chain.last_assistant_start
                assert last_assistant_start.response_ids_len <= len(buffer.response_ids)
                rollback_dropped_trainable_tokens = sum(buffer.response_mask[last_assistant_start.response_ids_len :])
                # One generation appends exactly one mark, so the rewritten
                # assistant is always the last one.
                del buffer.generation_versions[-1:]

                if image_data is not None:
                    assert last_assistant_start.image_data_len <= len(image_data)
                    image_data = image_data[: last_assistant_start.image_data_len] or None
                if video_data is not None:
                    assert last_assistant_start.video_data_len <= len(video_data)
                    video_data = video_data[: last_assistant_start.video_data_len] or None

                rollback_response_len = last_assistant_start.response_ids_len
                del buffer.response_ids[rollback_response_len:]
                del buffer.response_mask[rollback_response_len:]
                del buffer.response_logprobs[rollback_response_len:]

                assistant_prefix = self._codec.turn_separator + self._codec.generation_prompt
                stored_ids = buffer.prompt_ids if rollback_response_len == 0 else buffer.response_ids
                assistant_prefix_start = len(stored_ids) - len(assistant_prefix)
                if stored_ids[assistant_prefix_start:] != assistant_prefix:
                    raise ValueError("Stored trajectory does not end with the assistant prefix")
                del stored_ids[assistant_prefix_start:]
                if rollback_response_len:
                    del buffer.response_mask[assistant_prefix_start:]
                    del buffer.response_logprobs[assistant_prefix_start:]
                self._assert_response_logprob_alignment(buffer)
                incremental_messages = messages[last_assistant_start.message_history_len :]
                previous_messages = messages[: last_assistant_start.message_history_len]
                rollback_applied = True
            else:
                incremental_messages = messages[len(selected_chain.message_history) :]
                previous_messages = selected_chain.message_history

            appended_images = None
            updated_messages = previous_messages + incremental_messages
            merged_token_ids = None
            merged_response_mask = None
            merged_response_logprobs = None
            current_trajectory_length = len(buffer.prompt_ids) + len(buffer.response_ids)
            capacity_exhausted = (
                self._trajectory_capacity is not None and current_trajectory_length >= self._trajectory_capacity
            )
            if incremental_messages and not capacity_exhausted:
                self._codec.validate_incremental_media(incremental_messages)
                appended_images, appended_videos = await self._codec.extract_multi_modal_data(incremental_messages)
                if appended_videos:
                    raise UnsupportedMediaError(
                        "Continuous Token context merging does not currently support incremental video data"
                    )
                runtime_token_ids = buffer.prompt_ids + buffer.response_ids
                # The merge view covers the whole updated conversation, so the
                # images handed to Continuous Token combine resolved history
                # with newly extracted images.
                updated_images = None
                if image_data is not None or appended_images is not None:
                    updated_images = [*(image_data or []), *(appended_images or [])]
                merged_token_ids, merged_response_mask, merged_response_logprobs = self._codec.merge_context_tokens(
                    previous_messages,
                    updated_messages,
                    runtime_token_ids,
                    buffer.response_mask,
                    buffer.response_logprobs if sampling_params.get("logprobs", False) else None,
                    tools=tools,
                    image_data=updated_images,
                )
                capacity_exhausted = (
                    self._trajectory_capacity is not None and len(merged_token_ids) >= self._trajectory_capacity
                )
            if not capacity_exhausted and merged_token_ids is not None:
                assert merged_response_mask is not None
                prompt_length = len(buffer.prompt_ids)
                if merged_token_ids[:prompt_length] != buffer.prompt_ids:
                    raise RuntimeError("Continuous Token merge modified the immutable trajectory prompt")
                buffer.response_ids = list(merged_token_ids[prompt_length:])
                buffer.response_mask = list(merged_response_mask)
                buffer.response_logprobs = list(merged_response_logprobs or [])
                self._assert_response_logprob_alignment(buffer)
                if appended_images:
                    if image_data is None:
                        image_data = []
                    image_data.extend(appended_images)

        context_ids = buffer.prompt_ids + buffer.response_ids
        if capacity_exhausted:
            return EncodedData(
                buffer=buffer,
                context_ids=context_ids,
                sampling_params={},
                messages=list(messages),
                tools=tools,
                image_data=image_data,
                video_data=video_data,
                mm_processor_kwargs=mm_processor_kwargs,
                capacity_exhausted=True,
                chain_id=chain_id,
                agent_context=request.get("agent_context"),
                incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
                rollback_applied=rollback_applied,
                rollback_dropped_trainable_tokens=rollback_dropped_trainable_tokens,
            )

        remaining_trajectory_capacity = (
            self._trajectory_capacity - len(context_ids) if self._trajectory_capacity is not None else None
        )
        if remaining_trajectory_capacity is not None:
            sampling_params["max_tokens"] = min(
                sampling_params.get("max_tokens", remaining_trajectory_capacity),
                remaining_trajectory_capacity,
            )
        last_assistant_start = self._snapshot_last_assistant_start(
            buffer=buffer,
            message_history_len=len(messages),
            image_data=image_data,
            video_data=video_data,
            tip_hash=incoming_message_prefix_hashes[-1],
        )
        return EncodedData(
            buffer=buffer,
            context_ids=context_ids,
            sampling_params=sampling_params,
            messages=list(messages),
            tools=tools,
            image_data=image_data,
            video_data=video_data,
            mm_processor_kwargs=mm_processor_kwargs,
            capacity_exhausted=False,
            chain_id=chain_id,
            agent_context=request.get("agent_context"),
            incoming_message_prefix_hashes=list(incoming_message_prefix_hashes),
            last_assistant_start=last_assistant_start,
            rollback_applied=rollback_applied,
            rollback_dropped_trainable_tokens=rollback_dropped_trainable_tokens,
        )

    def _select_chain(
        self,
        *,
        incoming_assistant_count: int,
        tools: list[dict[str, Any]] | None,
        incoming_message_prefix_hashes: list[str],
        agent_context: AgentContext | None = None,
    ) -> tuple[ChainState, bool] | None:
        ranked_candidates = []
        deepest_rollback_candidates = []
        deepest_rollback_service_value = -1
        for chain in self.active_chains:
            if chain.agent_context != agent_context:
                continue
            if chain.chain_id in self.reserved_chain_ids or chain.active_tool_schemas != tools:
                continue
            assistant_start = chain.last_assistant_start
            assistant_start_len = assistant_start.message_history_len
            # A request ending exactly at the boundary is a fresh sample from
            # the same prompt, not a rewrite of the abandoned assistant.
            if assistant_start_len >= len(incoming_message_prefix_hashes):
                continue
            if incoming_message_prefix_hashes[assistant_start_len - 1] != assistant_start.tip_hash:
                continue
            if incoming_assistant_count > sum(message["role"] == "assistant" for message in chain.message_history):
                continue
            if self._is_chain_prefix_hash_match(
                chain=chain,
                incoming_message_prefix_hashes=incoming_message_prefix_hashes,
            ):
                ranked_candidates.append((chain, len(chain.message_history), True))
                continue
            if self._enable_last_assistant_rollback:
                if assistant_start_len > deepest_rollback_service_value:
                    deepest_rollback_candidates = [chain]
                    deepest_rollback_service_value = assistant_start_len
                elif assistant_start_len == deepest_rollback_service_value:
                    deepest_rollback_candidates.append(chain)

        if len(deepest_rollback_candidates) == 1:
            rollback_chain = deepest_rollback_candidates[0]
            ranked_candidates.append((rollback_chain, deepest_rollback_service_value, False))
        elif deepest_rollback_candidates and deepest_rollback_service_value > max(
            (candidate[1] for candidate in ranked_candidates),
            default=-1,
        ):
            return None
        if not ranked_candidates:
            return None
        selected_chain, _, exact_prefix_match = max(
            ranked_candidates,
            key=lambda candidate: (
                candidate[1],
                candidate[2],
                candidate[0].updated_seq,
                candidate[0].chain_id,
            ),
        )
        return selected_chain, not exact_prefix_match

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
            routed_experts=buffer.routed_experts,
            generation_versions=list(buffer.generation_versions),
        )

    def _copy_chain_media(self, chain: ChainState) -> tuple[list[Any] | None, list[Any] | None]:
        return (
            self._copy_media_list(chain.image_data),
            self._copy_media_list(chain.video_data),
        )

    def _copy_media_list(self, media: list[Any] | None) -> list[Any] | None:
        # Copy only the container; media payloads may not be deepcopyable.
        return list(media) if media is not None else None

    def _assert_response_logprob_alignment(self, buffer: TrajectoryBuffer) -> None:
        assert len(buffer.response_mask) == len(buffer.response_ids), "response_mask must be aligned with response_ids"
        assert len(buffer.response_logprobs) in {
            0,
            len(buffer.response_ids),
        }, "response_logprobs must be empty or aligned with response_ids"

    def _snapshot_last_assistant_start(
        self,
        *,
        buffer: TrajectoryBuffer,
        message_history_len: int,
        image_data: list[Any] | None,
        video_data: list[Any] | None,
        tip_hash: str,
    ) -> LastAssistantStart:
        return LastAssistantStart(
            response_ids_len=len(buffer.response_ids),
            message_history_len=message_history_len,
            image_data_len=len(image_data or []),
            video_data_len=len(video_data or []),
            tip_hash=tip_hash,
        )

    def _record_rollback_stats(self, encoded: EncodedData, chain: ChainState) -> None:
        if not encoded.rollback_applied:
            return
        chain.rollback_count += 1
        chain.rollback_dropped_trainable_tokens_total += encoded.rollback_dropped_trainable_tokens
        self._rollback_count += 1
        self._rollback_dropped_trainable_tokens_total += encoded.rollback_dropped_trainable_tokens

    def _commit_generation_to_chain(self, encoded: EncodedData, assistant_msg: dict[str, Any]) -> int:
        message_history = list(encoded.messages) + [assistant_msg]
        message_prefix_hashes = self._extend_message_prefix_hashes(
            encoded.incoming_message_prefix_hashes,
            [assistant_msg],
        )
        assert len(message_prefix_hashes) == len(message_history)
        if encoded.last_assistant_start is None:
            raise RuntimeError("last assistant start is missing")
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
                    last_assistant_start=encoded.last_assistant_start,
                    updated_seq=order_seq,
                    agent_context=encoded.agent_context,
                )
            )
            return chain_id

        chain_index, previous_chain = self._find_active_chain(encoded.chain_id)
        self._record_rollback_stats(encoded, previous_chain)
        order_seq = self._next_order_seq()
        self.active_chains[chain_index] = replace(
            previous_chain,
            message_history=message_history,
            message_tip_hash=message_prefix_hashes[-1],
            active_tool_schemas=encoded.tools,
            buffer=encoded.buffer,
            image_data=self._copy_media_list(encoded.image_data),
            video_data=self._copy_media_list(encoded.video_data),
            last_assistant_start=encoded.last_assistant_start,
            updated_seq=order_seq,
        )
        return previous_chain.chain_id

    def _close_length_exhausted_chain(self, encoded: EncodedData) -> None:
        if encoded.chain_id is None:
            raise RuntimeError("length-exhausted chain id is missing")
        chain_index, chain = self._find_active_chain(encoded.chain_id)
        self._record_rollback_stats(encoded, chain)
        if encoded.rollback_applied and chain.last_assistant_start.response_ids_len == 0:
            # No trainable response prefix survives this rollback.
            del self.active_chains[chain_index]
            return
        chain_to_materialize = chain
        if encoded.rollback_applied:
            message_history_len = chain.last_assistant_start.message_history_len
            chain_to_materialize = replace(
                chain,
                message_history=list(encoded.messages[:message_history_len]),
                message_tip_hash=encoded.incoming_message_prefix_hashes[message_history_len - 1],
                buffer=encoded.buffer,
                image_data=self._copy_media_list(encoded.image_data),
                video_data=self._copy_media_list(encoded.video_data),
            )
        order_seq = self._next_order_seq()
        self.materialized_chains.append(
            MaterializedChain(
                trajectory=self._build_materialized_trajectory(
                    chain=chain_to_materialize,
                    extra_fields={"materialization_reason": "max_trajectory_length"},
                ),
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
        self._assert_response_logprob_alignment(chain.buffer)
        response_logprobs = None
        if chain.buffer.response_logprobs:
            response_logprobs = list(chain.buffer.response_logprobs)
        # Fold the surviving marks into the trajectory-level version span the trainer
        # reads for staleness metrics. Omit the keys when no backend reported a
        # version so the framework falls back instead of tagging None.
        marks = [mark for mark in chain.buffer.generation_versions if mark[0] is not None]
        trajectory_extra_fields: dict[str, Any] = {}
        if chain.agent_context is not None:
            trajectory_extra_fields["agent_context"] = chain.agent_context.copy()
        if marks:
            trajectory_extra_fields["min_global_steps"] = min(mark[0] for mark in marks)
            trajectory_extra_fields["max_global_steps"] = max(mark[1] for mark in marks)
        if extra_fields:
            trajectory_extra_fields.update(extra_fields)
        if chain.num_coalesced_requests:
            trajectory_extra_fields["num_coalesced_requests"] = chain.num_coalesced_requests
        if chain.rollback_count:
            trajectory_extra_fields["rollback_count"] = chain.rollback_count
            trajectory_extra_fields["rollback_dropped_trainable_tokens_total"] = (
                chain.rollback_dropped_trainable_tokens_total
            )
        mm_processor_kwargs = self._codec.mm_processor_kwargs
        if mm_processor_kwargs:
            trajectory_extra_fields["mm_processor_kwargs"] = mm_processor_kwargs
        return Trajectory(
            prompt_ids=list(chain.buffer.prompt_ids),
            response_ids=list(chain.buffer.response_ids),
            response_mask=list(chain.buffer.response_mask),
            response_logprobs=response_logprobs,
            num_turns=self._count_chat_turns(chain.message_history),
            chain_id=chain.chain_id,
            routed_experts=chain.buffer.routed_experts,
            multi_modal_data=self._build_multi_modal_trajectory_data(
                chain.image_data,
                chain.video_data,
            ),
            extra_fields=trajectory_extra_fields,
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
