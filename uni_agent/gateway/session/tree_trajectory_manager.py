"""Disk-persistable, per-session trajectory tree management."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field, replace
from itertools import repeat
from pathlib import Path
from typing import Any, Awaitable, Callable

from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.types import AgentContext, Trajectory, UnsupportedMediaError

logger = logging.getLogger(__name__)
_TRAJECTORY_CAPACITY_HINT = (
    "The trajectory limit is prompt_length + response_length. "
    "For this event, Claude Code's output-token-limit error is misleading: increasing "
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS alone will not help. Check the rollout prompt_length/response_length "
    "budget or compact context earlier; the episode may be unfinished."
)


@dataclass(slots=True)
class MessageNode:
    """One Backend generation with any unmatched history needed to reach it.

    A node starts with the request's unmatched messages and prompt token delta,
    which may include reconstructed history. It becomes completed in place after
    the model output and decoded assistant message arrive. ``chain_id == 0``
    means a new branch is not committed yet; successful commits allocate the
    next 1-based id, matching the latest Gateway chain semantics.
    """

    node_id: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    parent: MessageNode | None = field(default=None, repr=False)
    children: list[MessageNode] = field(default_factory=list, repr=False)
    status: str = "PENDING"
    prompt_ids: list[int] = field(default_factory=list)
    # Only populated when assistant merging edits the pre-generation token delta.
    pre_assistant_prompt_ids: list[int] | None = None
    response_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = None
    # Context merge trimming; prompt_ids already includes inserted boundary tokens.
    removed_prefix_token_count: int = 0
    image_data: list[Any] | None = None
    video_data: list[Any] | None = None
    agent_context: AgentContext | None = None
    tools_fingerprint: str | None = None
    min_global_steps: int | None = None
    max_global_steps: int | None = None
    chain_id: int = 0
    token_length: int = 0
    message_turn_count: int = 0
    num_coalesced_requests: int = 0
    rollback_count: int = 0
    rollback_dropped_trainable_tokens_total: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent is None and self.node_id == 0


@dataclass
class InflightRequest:
    """Owner result and waiters to attribute when its generation commits."""

    future: asyncio.Future[tuple[bool, Any]]
    waiter_count: int = 0
    owner: InflightRequest | None = None


@dataclass(frozen=True, slots=True)
class _MountedGeneration:
    """Request-local tree state kept while its Backend generation is running."""

    context_ids: list[int]
    image_data: list[Any] | None
    video_data: list[Any] | None
    target_node: MessageNode
    merge_backup: MessageNode | None
    rewrite_action: str | None


class TreeTrajectoryManager:
    """Own, synchronize, and optionally persist one session trajectory tree."""

    def __init__(
        self,
        enable_last_assistant_rollback: bool = True,
        session_id: str | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        log_dir: str | Path | None = None,
        coalesce_reserved_exact_requests: bool = True,
    ) -> None:
        if type(coalesce_reserved_exact_requests) is not bool:
            raise ValueError("coalesce_reserved_exact_requests must be a bool")
        if type(enable_last_assistant_rollback) is not bool:
            raise ValueError("enable_last_assistant_rollback must be a bool")
        self.root = MessageNode(node_id=0, status="COMPLETED")
        self.tool_schemas_by_fingerprint: dict[str, list[dict[str, Any]] | None] = {}
        # A node is active; a Trajectory is closed. Both retain commit order.
        self.chains: dict[int, MessageNode | Trajectory] = {}
        self._routed_experts_by_chain: dict[int, Any] = {}
        self._num_trajectories = 0
        self.enable_last_assistant_rollback = enable_last_assistant_rollback
        self._coalesce_reserved_exact_requests = coalesce_reserved_exact_requests
        self._inflight_exact_requests: dict[str, InflightRequest] = {}
        self._coalesced_request_count = 0
        self._session_id = session_id
        self._next_node_id = 1
        self._next_chain_id = 1
        self.rewrite_merge_count = 0
        self.rewrite_fork_count = 0
        self.rewrite_dropped_trainable_tokens = 0
        self._pending_requests: dict[int, tuple[dict[str, Any], int | None, InflightRequest]] = {}
        self._tree_lock = asyncio.Lock()
        self._mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self._storage_path = Path(log_dir) / "trajectory_tree.pkl" if log_dir else None

    def _storage_state(self) -> dict[str, Any]:
        # Keep topology flat: even a children-only tree can exceed pickle's recursion limit.
        nodes: dict[int, MessageNode] = {}
        stack = [self.root] if self.root is not None else []
        stack.extend(leaf for leaf in self.chains.values() if isinstance(leaf, MessageNode))
        while stack:
            node = stack.pop()
            if id(node) in nodes:
                continue
            nodes[id(node)] = node
            stack.extend(node.children)
        copies = {key: replace(node, parent=None, children=[]) for key, node in nodes.items()}
        return {
            "root": copies[id(self.root)] if self.root is not None else None,
            "node_children": [
                (copies[key], [copies[id(child)] for child in node.children]) for key, node in nodes.items()
            ],
            "tool_schemas_by_fingerprint": self.tool_schemas_by_fingerprint,
            "chains": {
                chain_id: copies[id(leaf)] if isinstance(leaf, MessageNode) else leaf
                for chain_id, leaf in self.chains.items()
            },
            "routed_experts_by_chain": self._routed_experts_by_chain,
            "rewrite_merge_count": self.rewrite_merge_count,
            "rewrite_fork_count": self.rewrite_fork_count,
            "rewrite_dropped_trainable_tokens": self.rewrite_dropped_trainable_tokens,
            "coalesced_request_count": self._coalesced_request_count,
        }

    @staticmethod
    def _write_storage_state(path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as output_file:
            pickle.dump(state, output_file, protocol=pickle.HIGHEST_PROTOCOL)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)

    @property
    def storage_path(self) -> Path | None:
        return self._storage_path

    async def run_generation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        codec: MessageCodec,
        trajectory_capacity: int | None,
        sampling_params: dict[str, Any],
        generate_fn: Callable[[list[int], dict[str, Any], list[Any] | None, list[Any] | None], Awaitable[Any]],
        agent_context: AgentContext | None = None,
    ) -> tuple[dict[str, Any], str, int, int, int | None, int, bool]:
        """Share the outcome of identical requests while generation is pending."""
        agent_context = agent_context.copy() if agent_context is not None else None
        request_fingerprint = (
            hashlib.sha256(
                b"uni-agent-request-v1\0"
                + json.dumps(
                    {
                        "agent_context": agent_context,
                        "messages": messages,
                        "tools": tools,
                        "sampling_params": sampling_params,
                        "trajectory_capacity": trajectory_capacity,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            if self._coalesce_reserved_exact_requests
            else None
        )
        # No await between lookup and registration: retain Session's early
        # coalescing while keeping node mounting under the existing tree lock.
        if request_fingerprint is not None:
            joined_request = self._inflight_exact_requests.get(request_fingerprint)
            if joined_request is not None:
                owner = joined_request
                while owner.owner is not None:
                    owner = owner.owner
                if not owner.future.done():
                    self._record_coalesced_request(joined_request, request_fingerprint)
                    succeeded, value = await asyncio.shield(joined_request.future)
                    if not succeeded:
                        raise value
                    return value
        owner_request = InflightRequest(asyncio.get_running_loop().create_future())
        if request_fingerprint is not None:
            self._inflight_exact_requests[request_fingerprint] = owner_request
        try:
            result = await self._run_generation(
                messages=messages,
                tools=tools,
                codec=codec,
                trajectory_capacity=trajectory_capacity,
                sampling_params=sampling_params,
                generate_fn=generate_fn,
                owner_request=owner_request,
                request_fingerprint=request_fingerprint,
                agent_context=agent_context,
            )
        except BaseException as error:
            owner_request.future.set_result((False, error))
            raise
        else:
            owner_request.future.set_result((True, result))
            return result
        finally:
            if (
                request_fingerprint is not None
                and self._inflight_exact_requests.get(request_fingerprint) is owner_request
            ):
                self._inflight_exact_requests.pop(request_fingerprint)

    def _record_coalesced_request(self, joined_request: InflightRequest, request_fingerprint: str) -> None:
        while joined_request.owner is not None:
            joined_request = joined_request.owner
        joined_request.waiter_count += 1
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

    async def _run_generation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        codec: MessageCodec,
        trajectory_capacity: int | None,
        sampling_params: dict[str, Any],
        generate_fn: Callable[[list[int], dict[str, Any], list[Any] | None, list[Any] | None], Awaitable[Any]],
        owner_request: InflightRequest,
        request_fingerprint: str | None,
        agent_context: AgentContext | None = None,
    ) -> tuple[dict[str, Any], str, int, int, int | None, int, bool]:
        """Run one concurrent tree transaction around an unlocked Backend call."""
        async with self._tree_lock:
            if self.root is None:
                raise RuntimeError("generation branch was removed before Backend completion")
            mounted = await self._mount_generation(
                messages=messages,
                tools=tools,
                codec=codec,
                trajectory_capacity=trajectory_capacity,
                sampling_params=sampling_params,
                owner_request=owner_request,
                agent_context=agent_context,
            )
            if isinstance(mounted, InflightRequest):
                owner_request.owner = mounted
                mounted.waiter_count += owner_request.waiter_count
                owner_request.waiter_count = 0
                assert request_fingerprint is not None
                self._record_coalesced_request(mounted, request_fingerprint)
            else:
                node = mounted.target_node
                prompt_tokens = len(mounted.context_ids)
                remaining_capacity = trajectory_capacity - prompt_tokens if trajectory_capacity is not None else None
                if remaining_capacity is not None and remaining_capacity <= 0:
                    try:
                        prompt_tokens = self._close_length_exhausted_chain(mounted, codec)
                    finally:
                        self._rollback_mounted_branch(target_node=node, merge_backup=mounted.merge_backup)
                    logger.warning(
                        "Trajectory capacity prevents generation: session=%s. %s",
                        self._session_id,
                        _TRAJECTORY_CAPACITY_HINT,
                    )
                    return (
                        {"role": "assistant", "content": ""},
                        "length",
                        prompt_tokens,
                        0,
                        node.chain_id or None,
                        0,
                        True,
                    )

                effective_sampling_params = dict(sampling_params)
                if remaining_capacity is not None:
                    effective_sampling_params["max_tokens"] = min(
                        effective_sampling_params.get("max_tokens", remaining_capacity), remaining_capacity
                    )

        if isinstance(mounted, InflightRequest):
            succeeded, value = await asyncio.shield(mounted.future)
            if not succeeded:
                raise value
            return value

        try:
            output = await generate_fn(
                mounted.context_ids, effective_sampling_params, mounted.image_data, mounted.video_data
            )
            response_ids = list(output.token_ids)
            response_logprobs: list[float] | None = None
            if effective_sampling_params.get("logprobs", False):
                if output.log_probs is None:
                    raise RuntimeError("backend omitted logprobs when requested")
                response_logprobs = list(output.log_probs)
                if len(response_logprobs) != len(response_ids):
                    raise RuntimeError(
                        "backend logprobs must align with token_ids: "
                        f"got {len(response_logprobs)} logprobs for {len(response_ids)} tokens"
                    )
            async with self._tree_lock:
                if self.root is None:
                    raise RuntimeError("generation branch was removed before Backend completion")
                assistant_message, finish_reason = await codec.decode_response(
                    response_ids, tools=tools, stop_reason=output.stop_reason
                )
                extra_fields = output.extra_fields or {}
                merge_result = codec.merge_assistant_tokens(mounted.context_ids, response_ids)
                if response_logprobs is not None and len(response_logprobs) != merge_result.appended_token_count:
                    raise ValueError(
                        "assistant_logprobs length must match appended assistant token count, "
                        f"got {len(response_logprobs)} and {merge_result.appended_token_count}"
                    )
                if merge_result.removed_prefix_token_count > len(node.prompt_ids):
                    raise RuntimeError("Assistant merge removes tokens from shared ancestors")
                if merge_result.removed_prefix_token_count or merge_result.inserted_token_ids:
                    node.pre_assistant_prompt_ids = list(node.prompt_ids)
                if merge_result.removed_prefix_token_count:
                    del node.prompt_ids[-merge_result.removed_prefix_token_count :]
                node.prompt_ids.extend(merge_result.inserted_token_ids)
                node.messages.append(dict(assistant_message))
                node.response_ids = response_ids
                node.response_logprobs = response_logprobs
                parent = node.parent
                node.token_length = len(merge_result.token_ids)
                node.message_turn_count = (parent.message_turn_count if parent is not None else 0) + sum(
                    message.get("role") in ("user", "assistant") for message in node.messages
                )
                node.min_global_steps = extra_fields.get("min_global_steps")
                node.max_global_steps = extra_fields.get("max_global_steps")
                routed_experts = getattr(output, "routed_experts", None)
                node.status = "COMPLETED"
                self._pending_requests.pop(id(node), None)
                previous_leaf = self.chains.get(node.chain_id)
                if previous_leaf is not None:
                    assert isinstance(previous_leaf, MessageNode), "Cannot commit to a closed chain"
                    node.num_coalesced_requests = previous_leaf.num_coalesced_requests
                    node.rollback_count = previous_leaf.rollback_count
                    node.rollback_dropped_trainable_tokens_total = previous_leaf.rollback_dropped_trainable_tokens_total
                node.num_coalesced_requests += owner_request.waiter_count
                if mounted.merge_backup is not None:
                    self.rewrite_merge_count += 1
                    self.rewrite_dropped_trainable_tokens += len(mounted.merge_backup.response_ids)
                    node.rollback_count += 1
                    node.rollback_dropped_trainable_tokens_total += len(mounted.merge_backup.response_ids)
                    mounted.merge_backup.image_data = mounted.merge_backup.video_data = None
                elif mounted.rewrite_action == "fork":
                    self.rewrite_fork_count += 1
                if node.chain_id == 0:
                    node.chain_id = self._next_chain_id
                    self._next_chain_id += 1
                chain_id = node.chain_id
                # Registry order tracks the latest commit or capacity materialization.
                self.chains.pop(chain_id, None)
                self.chains[chain_id] = node
                if routed_experts is not None:
                    self._routed_experts_by_chain[chain_id] = routed_experts
        except BaseException as error:
            cancellation: asyncio.CancelledError | None = None
            while True:
                try:
                    async with self._tree_lock:
                        # Rollback must stay synchronous: only waiting for the lock may be cancelled.
                        self._rollback_mounted_branch(target_node=node, merge_backup=mounted.merge_backup)
                    break
                except asyncio.CancelledError as exc:
                    cancellation = exc
            if cancellation is not None:
                raise cancellation from error
            raise
        if finish_reason == "length" and remaining_capacity is not None and len(response_ids) >= remaining_capacity:
            logger.warning(
                "Generation reached trajectory capacity: session=%s context_tokens=%s completion_tokens=%s. %s",
                self._session_id,
                prompt_tokens,
                len(response_ids),
                _TRAJECTORY_CAPACITY_HINT,
            )
        return (
            assistant_message,
            finish_reason,
            prompt_tokens,
            len(response_ids),
            node.chain_id,
            node.message_turn_count + 1,
            False,
        )

    async def _mount_generation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        codec: MessageCodec,
        owner_request: InflightRequest,
        trajectory_capacity: int | None = None,
        sampling_params: dict[str, Any] | None = None,
        agent_context: AgentContext | None = None,
    ) -> _MountedGeneration | InflightRequest:
        """Match only stable nodes and mount one request-local PENDING branch."""
        merge_candidate: MessageNode | None = None
        rewrite_action: str | None = None
        target_node: MessageNode | None = None
        try:
            serialized_tools = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            tools_fingerprint = hashlib.sha256(serialized_tools.encode("utf-8")).hexdigest()
            if tools_fingerprint not in self.tool_schemas_by_fingerprint:
                self.tool_schemas_by_fingerprint[tools_fingerprint] = self._copy_tools(tools)

            parent = self.root
            parent_path: list[MessageNode] = []
            offset = 0
            assistant_candidate: MessageNode | None = None
            active_path_ranks: dict[int, tuple[int, bool]] = {}
            for leaf in self.chains.values():
                if not isinstance(leaf, MessageNode):
                    continue
                if leaf.agent_context != agent_context or leaf.tools_fingerprint != tools_fingerprint:
                    continue
                path = self._path_from_root(leaf)
                history = self._path_messages(path)
                prefix_length = len(history) - 1
                if len(messages) <= prefix_length or not self._messages_equal(
                    history[:-1], messages[:prefix_length], codec
                ):
                    continue
                exact = self._messages_equal(history, messages[: len(history)], codec)
                rank = (len(history) if exact else prefix_length, exact)
                for node in path:
                    active_path_ranks[id(node)] = max(active_path_ranks.get(id(node), (0, False)), rank)
            while offset < len(messages):
                incoming = messages[offset:]
                matching_full: list[MessageNode] = []
                candidate: MessageNode | None = None
                for child in parent.children:
                    if child.agent_context != agent_context:
                        continue
                    if child.tools_fingerprint != tools_fingerprint:
                        continue
                    if child.status == "PENDING":
                        pending = self._pending_requests.get(id(child))
                        if (
                            self._coalesce_reserved_exact_requests
                            and pending is not None
                            and pending[0] == sampling_params
                            and pending[1] == trajectory_capacity
                            and self._messages_equal(child.messages, incoming, codec)
                        ):
                            return pending[2]
                        continue
                    if child.status != "COMPLETED" or not child.messages:
                        continue
                    prefix_length = len(child.messages) - 1
                    if len(incoming) < prefix_length or not self._messages_equal(
                        child.messages[:-1], incoming[:prefix_length], codec
                    ):
                        continue
                    if len(incoming) == prefix_length and child.messages[-1].get("role") == "assistant":
                        continue  # Same prompt after completion starts an independent sample.
                    if len(incoming) > prefix_length and self._messages_equal(
                        [child.messages[-1]], [incoming[prefix_length]], codec
                    ):
                        matching_full.append(child)
                        continue
                    if child.messages[-1].get("role") == "assistant" and (
                        candidate is None or self.chains.get(child.chain_id) is child
                    ):
                        candidate = child
                if matching_full:
                    # Prefer the best matching active history, including its shared ancestors.
                    child = max(matching_full, key=lambda node: active_path_ranks.get(id(node), (0, False)))
                    parent = child
                    parent_path.append(child)
                    offset += len(child.messages)
                    continue
                assistant_candidate = candidate
                break

            pending_messages = messages[offset:]
            if assistant_candidate is not None:
                parent = assistant_candidate.parent
                sibling_assistant_nodes = [
                    child
                    for child in parent.children
                    if child.status == "COMPLETED"
                    and self.chains.get(child.chain_id) is child
                    and child.agent_context == agent_context
                    and child.tools_fingerprint == tools_fingerprint
                    and child.messages
                    and child.messages[-1].get("role") == "assistant"
                    and len(child.messages) == len(assistant_candidate.messages)
                    and self._messages_equal(child.messages[:-1], assistant_candidate.messages[:-1], codec)
                ]
                merge = (
                    self.enable_last_assistant_rollback
                    and self.chains.get(assistant_candidate.chain_id) is assistant_candidate
                    and len(sibling_assistant_nodes) == 1
                    and not assistant_candidate.children
                    and not any(
                        child.status == "PENDING" and child.agent_context == agent_context for child in parent.children
                    )
                )
                rewrite_action = "merge" if merge else "fork"
                if merge:
                    merge_candidate = assistant_candidate

            context_ids: list[int] = []
            for node in parent_path:
                if node.removed_prefix_token_count:
                    del context_ids[-node.removed_prefix_token_count :]
                context_ids.extend(node.prompt_ids)
                context_ids.extend(node.response_ids)
            merge_result = None
            new_image_data = new_video_data = None
            image_data = [image for node in parent_path for image in node.image_data or []] or None
            video_data = [video for node in parent_path for video in node.video_data or []] or None
            if not pending_messages and not parent.is_root:
                prompt_ids = []
            else:
                if not parent.is_root:
                    codec.validate_incremental_media(pending_messages)
                new_image_data, new_video_data = await codec.extract_multi_modal_data(pending_messages)
                if not parent.is_root and new_video_data:
                    raise UnsupportedMediaError(
                        "Continuous Token context merging does not currently support incremental video data"
                    )
                if new_image_data:
                    image_data = [*(image_data or []), *new_image_data]
                if new_video_data:
                    video_data = [*(video_data or []), *new_video_data]
                if parent.is_root:
                    prompt_ids = codec.build_initial_tokens(
                        pending_messages, tools=tools, image_data=image_data, video_data=video_data
                    )
                    context_ids = prompt_ids
                else:
                    previous_messages = self._path_messages(parent_path)
                    merge_result = codec.merge_context_tokens(
                        previous_messages,
                        previous_messages + pending_messages,
                        context_ids,
                        tools=tools,
                        image_data=image_data,
                    )
                    prompt_ids = merge_result.token_ids[len(context_ids) - merge_result.removed_prefix_token_count :]
                    context_ids = merge_result.token_ids
            chain_id = 0
            if rewrite_action == "merge":
                chain_id = merge_candidate.chain_id
            elif (
                not parent.is_root
                and parent.chain_id in self.chains
                and self.chains[parent.chain_id] is parent
                and not parent.children
            ):
                chain_id = parent.chain_id
            if (
                sampling_params.get("logprobs", False)
                and (trajectory_capacity is None or len(context_ids) < trajectory_capacity)
                and chain_id != 0
                and parent_path
                and (merge_candidate or parent).response_logprobs is None
            ):
                retained_response_tokens = 0
                for index, node in enumerate(parent_path):
                    if node.chain_id != chain_id:
                        continue
                    retained_response_tokens = max(0, retained_response_tokens - node.removed_prefix_token_count)
                    if index and parent_path[index - 1].chain_id == chain_id:
                        retained_response_tokens += len(node.prompt_ids)
                    retained_response_tokens += len(node.response_ids)
                trimmed_tokens = merge_result.removed_prefix_token_count if merge_result else 0
                assert retained_response_tokens <= trimmed_tokens, (
                    "response_logprobs must be empty or aligned with response_ids"
                )
            node_id = merge_candidate.node_id if merge_candidate is not None else self._next_node_id
            if merge_candidate is None:
                self._next_node_id += 1
            target_node = MessageNode(
                node_id=node_id,
                messages=pending_messages,
                parent=parent,
                prompt_ids=prompt_ids,
                removed_prefix_token_count=merge_result.removed_prefix_token_count if merge_result else 0,
                image_data=list(new_image_data) if new_image_data else None,
                video_data=list(new_video_data) if new_video_data else None,
                tools_fingerprint=tools_fingerprint,
                agent_context=agent_context.copy() if agent_context is not None else None,
                chain_id=chain_id,
            )
            parent.children.append(target_node)
            self._pending_requests[id(target_node)] = (dict(sampling_params or {}), trajectory_capacity, owner_request)
            if merge_candidate is not None:
                parent.children.remove(merge_candidate)
        except BaseException:
            self._rollback_mounted_branch(target_node=target_node, merge_backup=merge_candidate)
            raise
        return _MountedGeneration(
            context_ids=context_ids,
            image_data=image_data,
            video_data=video_data,
            target_node=target_node,
            merge_backup=merge_candidate,
            rewrite_action=rewrite_action,
        )

    def _rollback_mounted_branch(
        self,
        target_node: MessageNode | None,
        merge_backup: MessageNode | None,
    ) -> None:
        if target_node is not None:
            self._pending_requests.pop(id(target_node), None)
            target_node.image_data = target_node.video_data = None
        if self.root is None:
            return
        if target_node is not None and target_node.parent is not None:
            parent = target_node.parent
            parent.children = [child for child in parent.children if child is not target_node]
        if merge_backup is not None:
            parent = merge_backup.parent
            if not any(child is merge_backup for child in parent.children):
                parent.children.append(merge_backup)

    def _close_length_exhausted_chain(self, mounted: _MountedGeneration, codec: MessageCodec) -> int:
        """Freeze the committed chain, applying an accepted rollback before closing."""
        chain_id = mounted.target_node.chain_id
        leaf = self.chains.get(chain_id)
        if not isinstance(leaf, MessageNode):
            return len(mounted.context_ids)
        path = self._path_from_root(leaf)
        backup = mounted.merge_backup
        if backup is not None:
            # Keep this turn's original input/media, but omit its generated assistant.
            path[-1] = replace(
                backup,
                prompt_ids=(
                    backup.pre_assistant_prompt_ids
                    if backup.pre_assistant_prompt_ids is not None
                    else backup.prompt_ids
                ),
                messages=backup.messages[:-1],
                response_ids=[],
                response_logprobs=[] if backup.response_logprobs is not None else None,
                min_global_steps=None,
                max_global_steps=None,
                message_turn_count=backup.message_turn_count - 1,
                rollback_count=backup.rollback_count + 1,
                rollback_dropped_trainable_tokens_total=(
                    backup.rollback_dropped_trainable_tokens_total + len(backup.response_ids)
                ),
            )
        trajectory = self._linearize_path(path)
        drop_chain = False
        if backup is not None:
            # Match Chain's pre-assistant response-length check, not sum(mask).
            drop_chain = not trajectory.response_ids
            stored_ids = trajectory.prompt_ids if drop_chain else trajectory.response_ids
            assistant_prefix = codec.turn_separator + codec.generation_prompt
            if assistant_prefix:
                if stored_ids[-len(assistant_prefix) :] != assistant_prefix:
                    raise ValueError("Stored trajectory does not end with the assistant prefix")
                del stored_ids[-len(assistant_prefix) :]
                if not drop_chain:
                    del trajectory.response_mask[-len(assistant_prefix) :]
                    if trajectory.response_logprobs is not None:
                        del trajectory.response_logprobs[-len(assistant_prefix) :]
            trajectory.response_logprobs = trajectory.response_logprobs or None
        trajectory.extra_fields["materialization_reason"] = "max_trajectory_length"
        # All conversion/validation precedes the state change; this runs under the tree lock.
        self.chains.pop(chain_id)
        if not drop_chain:
            self.chains[chain_id] = trajectory
        self._routed_experts_by_chain.pop(chain_id, None)
        if backup is not None:
            self.rewrite_merge_count += 1
            self.rewrite_dropped_trainable_tokens += len(backup.response_ids)
        return len(trajectory.prompt_ids) + len(trajectory.response_ids)

    def snapshot_state(self) -> dict[str, Any]:
        """Read resident metadata without waiting for locks or loading the tree."""
        chain_ids = sorted(chain_id for chain_id, leaf in self.chains.items() if isinstance(leaf, MessageNode))
        return {
            "num_trajectories": self._num_trajectories
            + sum(isinstance(leaf, Trajectory) for leaf in self.chains.values()),
            "has_active_trajectory": bool(chain_ids),
            "num_active_chains": len(chain_ids),
            "active_chain_ids": chain_ids,
            "rollback_count": self.rewrite_merge_count,
            "rollback_dropped_trainable_tokens_total": self.rewrite_dropped_trainable_tokens,
            "num_coalesced_requests": self._coalesced_request_count,
            "rewrite_fork_count": self.rewrite_fork_count,
            "num_nodes": self._next_node_id - 1,
        }

    async def finalize_trajectories(self) -> list[Trajectory]:
        async with self._tree_lock:
            if self.root is None:
                raise RuntimeError("trajectory tree is finalized or aborted")
            trajectories = [
                leaf if isinstance(leaf, Trajectory) else self._linearize_path(self._path_from_root(leaf))
                for leaf in self.chains.values()
            ]
            if self._storage_path is not None:
                await self._write_storage_state_locked(self._storage_path)
            self._num_trajectories = len(trajectories)
            self._unload_tree()
            return trajectories

    async def abort(self) -> None:
        async with self._tree_lock:
            storage_path = self._storage_path
            try:
                if storage_path is not None and self.root is not None:
                    storage_path = storage_path.with_name("trajectory_tree_abort.pkl")
                    await self._write_storage_state_locked(storage_path)
                    self._storage_path = storage_path
            except Exception:
                logger.warning(
                    "Failed to persist aborted trajectory tree %s",
                    storage_path,
                    exc_info=True,
                )
            self._num_trajectories = 0
            self._unload_tree()

    def get_active_chains(self) -> list[dict[str, Any]]:
        return [
            {
                "chain_id": chain_id,
                "message_history": self._path_messages(self._path_from_root(leaf)),
                "active_tool_schemas": self._copy_tools(self.tool_schemas_by_fingerprint.get(leaf.tools_fingerprint)),
            }
            for chain_id, leaf in sorted(self.chains.items())
            if isinstance(leaf, MessageNode)
        ]

    def _unload_tree(self) -> None:
        """Drop the loaded tree after breaking MessageNode reference cycles."""
        stack = [leaf for leaf in self.chains.values() if isinstance(leaf, MessageNode)]
        if self.root is not None:
            stack.append(self.root)
        self.root = None
        self.chains.clear()
        self.tool_schemas_by_fingerprint.clear()
        self._routed_experts_by_chain.clear()
        self._pending_requests.clear()
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            node.parent = None
            node.children.clear()
            node.image_data = node.video_data = None

    async def _write_storage_state_locked(self, storage_path: Path) -> None:
        """Keep the tree locked until its writer exits, even after cancellation."""
        task = asyncio.create_task(asyncio.to_thread(self._write_storage_state, storage_path, self._storage_state()))
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = exc
            except Exception:
                if cancellation is None:
                    raise
        if cancellation is not None:
            if not task.cancelled():
                task.exception()  # Retrieve any write failure before propagating cancellation.
            raise cancellation
        task.result()

    def _messages_equal(
        self,
        left: list[dict[str, Any]],
        right: list[dict[str, Any]],
        codec: MessageCodec,
    ) -> bool:
        return len(left) == len(right) and all(
            codec.canonicalize_message_for_prefix_comparison(left_message)
            == codec.canonicalize_message_for_prefix_comparison(right_message)
            for left_message, right_message in zip(left, right, strict=True)
        )

    def _linearize_path(self, path: list[MessageNode]) -> Trajectory:
        leaf = path[-1]
        chain_start = next(index for index, node in enumerate(path) if node.chain_id == leaf.chain_id)
        prompt_ids = list(path[0].prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if leaf.response_logprobs is not None else None
        for index, node in enumerate(path):
            if node.removed_prefix_token_count:
                if node.removed_prefix_token_count > len(response_ids):
                    del prompt_ids[-(node.removed_prefix_token_count - len(response_ids)) :]
                del response_ids[-node.removed_prefix_token_count :]
                del response_mask[-node.removed_prefix_token_count :]
                if response_logprobs is not None:
                    del response_logprobs[-node.removed_prefix_token_count :]
            if index:
                response_ids.extend(node.prompt_ids)
                response_mask.extend(repeat(0, len(node.prompt_ids)))
                if response_logprobs is not None:
                    response_logprobs.extend(repeat(0.0, len(node.prompt_ids)))
            if index == chain_start:
                # History owned by another chain is this chain's initial prompt.
                prompt_ids.extend(response_ids)
                response_ids.clear()
                response_mask.clear()
                if response_logprobs is not None:
                    response_logprobs.clear()
            response_ids.extend(node.response_ids)
            response_mask.extend(repeat(1, len(node.response_ids)))
            if response_logprobs is not None:
                response_logprobs.extend(
                    node.response_logprobs
                    if node.response_logprobs is not None
                    else repeat(0.0, len(node.response_ids))
                )
        if len(response_ids) != len(response_mask):
            raise RuntimeError("linearized response_ids and response_mask are not aligned")
        if response_logprobs is not None and len(response_logprobs) != len(response_ids):
            raise RuntimeError("linearized response_ids and response_logprobs are not aligned")
        marks = [
            (node.min_global_steps, node.max_global_steps)
            for node in path[chain_start:]
            if node.min_global_steps is not None
        ]
        extra_fields: dict[str, Any] = {}
        if marks:
            extra_fields["min_global_steps"] = min(mark[0] for mark in marks)
            extra_fields["max_global_steps"] = max(mark[1] for mark in marks)
        if self._mm_processor_kwargs:
            extra_fields["mm_processor_kwargs"] = dict(self._mm_processor_kwargs)
        image_data = [image for node in path for image in node.image_data or []]
        video_data = [video for node in path for video in node.video_data or []]
        multi_modal_data: dict[str, Any] = {}
        if image_data:
            multi_modal_data["images"] = image_data
        if video_data:
            multi_modal_data["videos"] = video_data
        if leaf.agent_context is not None:
            extra_fields["agent_context"] = leaf.agent_context.copy()
        if leaf.num_coalesced_requests:
            extra_fields["num_coalesced_requests"] = leaf.num_coalesced_requests
        if leaf.rollback_count:
            extra_fields["rollback_count"] = leaf.rollback_count
            extra_fields["rollback_dropped_trainable_tokens_total"] = leaf.rollback_dropped_trainable_tokens_total
        return Trajectory(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs or None,
            num_turns=leaf.message_turn_count + 1,
            chain_id=leaf.chain_id,
            routed_experts=self._routed_experts_by_chain.get(leaf.chain_id),
            multi_modal_data=multi_modal_data or None,
            extra_fields=extra_fields,
        )

    def _path_from_root(self, node: MessageNode) -> list[MessageNode]:
        path: list[MessageNode] = []
        current: MessageNode | None = node
        while current is not None and not current.is_root:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def _path_messages(self, path: list[MessageNode]) -> list[dict[str, Any]]:
        return [dict(message) for node in path for message in node.messages]

    def _copy_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        return [dict(tool) for tool in tools] if tools is not None else None
