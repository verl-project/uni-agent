"""Tree trajectory behavior and session mode compatibility; CPU-only."""

from __future__ import annotations

import asyncio
import pickle
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from tests.uni_agent.support import FakeTokenizer, SequencedBackend, fake_vision_info_extractor
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.session import MessageCodec
from uni_agent.gateway.session.session import GatewaySession
from uni_agent.gateway.session.tree_session import MessageNode, TreeGatewaySession
from uni_agent.gateway.session.types import SessionHandle, Trajectory, UnsupportedMediaError
from verl.utils.tokenizer.continuous_token import MergeResult


class FakeContinuousCodec:
    mm_processor_kwargs = {"image_patch_size": 14}
    turn_separator = []
    generation_prompt = []

    def __init__(self, trim_boundary: bool = False):
        self.trim_boundary = trim_boundary
        self.initial_calls = 0
        self.context_calls = 0

    async def extract_multi_modal_data(self, messages):
        return None, None

    def validate_incremental_media(self, messages):
        pass

    def build_initial_tokens(self, messages, tools=None, image_data=None, video_data=None):
        self.initial_calls += 1
        return [101, len(messages)]

    def merge_context_tokens(
        self,
        previous_messages,
        updated_messages,
        runtime_token_ids,
        response_mask=None,
        *,
        tools=None,
        image_data=None,
    ):
        assert response_mask is None
        self.context_calls += 1
        cut = int(self.trim_boundary)
        appended = [200 + len(updated_messages) - len(previous_messages)]
        tokens = runtime_token_ids[: len(runtime_token_ids) - cut] + appended
        return MergeResult(
            token_ids=tokens,
            removed_prefix_token_count=cut,
            inserted_token_ids=[],
            appended_token_count=1,
        )

    def merge_assistant_tokens(
        self,
        runtime_token_ids,
        assistant_token_ids,
        response_mask=None,
    ):
        assert response_mask is None
        return MergeResult(
            token_ids=runtime_token_ids + assistant_token_ids,
            kind="assistant",
            removed_prefix_token_count=0,
            inserted_token_ids=[],
            appended_token_count=len(assistant_token_ids),
        )

    async def decode_response(self, response_ids, *, tools=None, stop_reason=None):
        return {"role": "assistant", "content": "".join(map(chr, response_ids))}, "stop"

    def canonicalize_message_for_prefix_comparison(self, message):
        return message


class Backend:
    def __init__(self, *responses, routed_experts=None):
        self.responses = list(responses)
        self.calls = []
        self.routed_experts = routed_experts

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, asyncio.Event):
            await response.wait()
            response = "A"
        return SimpleNamespace(
            token_ids=list(map(ord, response)),
            log_probs=[-0.1] * len(response),
            extra_fields={"min_global_steps": 2, "max_global_steps": 2},
            routed_experts=self.routed_experts,
            stop_reason="stop",
        )


async def generate(session, backend, messages, *, logprobs=False, trajectory_capacity=None):
    session._trajectory_capacity = trajectory_capacity
    return await session.run_generation(
        {"messages": messages, "tools": None, "sampling_params": {"max_tokens": 32, "logprobs": logprobs}}, backend
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_continuous_context_merge_preserves_ancestors_and_metadata():
    codec = FakeContinuousCodec(trim_boundary=True)
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    backend = Backend("AB", "C", "D", routed_experts=[[1]])
    first = [{"role": "user", "content": "task"}]
    await generate(session, backend, first, logprobs=True)
    original = session.leaves[1]
    backend.routed_experts = [[2]]
    continuation = [
        *first,
        {"role": "assistant", "content": "AB"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "historical reply"},
        {"role": "tool", "content": "more context"},
    ]
    await generate(session, backend, continuation, logprobs=True)
    [child] = original.children
    assert child.messages == continuation[2:] + [{"role": "assistant", "content": "C"}]
    history = [*continuation, {"role": "assistant", "content": "C"}, {"role": "tool", "content": "next"}]
    await generate(session, backend, history, logprobs=True)

    assert original.response_ids == [65, 66]
    assert backend.calls[-1]["prompt_ids"] == [101, 1, 65, 203, 201]
    [trajectory] = await session.finalize()
    assert trajectory.response_ids == [65, 203, 201, 68]
    assert trajectory.response_mask == [1, 0, 0, 1]
    assert trajectory.response_logprobs == [-0.1, 0.0, 0.0, -0.1]
    assert trajectory.routed_experts is backend.routed_experts
    assert session._routed_experts_by_chain == {}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_video_from_extractor_is_rejected_before_context_merge():
    class Codec(FakeContinuousCodec):
        async def extract_multi_modal_data(self, messages):
            return (None, [object()]) if messages[0]["role"] == "tool" else (None, None)

    codec = Codec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)
    with pytest.raises(UnsupportedMediaError, match="incremental video"):
        await generate(
            session,
            Backend("B"),
            [*first, {"role": "assistant", "content": "A"}, {"role": "tool", "content": "video"}],
        )
    assert codec.context_calls == 0 and not session._inflight_exact_requests
    assert session.leaves[1].children == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_missing_backend_logprobs_still_fail_for_empty_output():
    async def generate_fn(**kwargs):
        return SimpleNamespace(token_ids=[], log_probs=None)

    session = TreeGatewaySession(SessionHandle("test-session"), FakeContinuousCodec())
    with pytest.raises(RuntimeError, match="omitted logprobs"):
        await session.run_generation(
            {"messages": [{"role": "user", "content": "task"}], "tools": None, "sampling_params": {"logprobs": True}},
            SimpleNamespace(generate=generate_fn),
        )
    assert session.root.children == [] and not session._inflight_exact_requests


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_action,write_fails", [("finalize", False), ("abort", True)])
async def test_cancelled_terminal_write_finishes_before_session_retry(tmp_path, terminal_action, write_fails):
    session = TreeGatewaySession(
        SessionHandle("cancelled-write"),
        FakeContinuousCodec(),
        metadata={"_log_dir": str(tmp_path)},
    )
    request = {"messages": [{"role": "user", "content": "task"}], "tools": None, "sampling_params": {}}
    await session.run_generation(request, Backend("A"))
    write = session._write_storage_state
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    writes = 0

    def delayed_write(path, state):
        nonlocal writes
        writes += 1
        first_write = writes == 1
        try:
            if first_write:
                loop.call_soon_threadsafe(started.set)
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release the writer")
                if write_fails:
                    raise OSError("disk failure after cancellation")
            write(path, state)
        finally:
            if first_write:
                finished.set()

    session._write_storage_state = delayed_write
    terminal = asyncio.create_task(getattr(session, terminal_action)())
    retry = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        for _ in range(2):
            terminal.cancel()
            await asyncio.sleep(0)
            assert not terminal.done()
        backend = Backend("B")
        retry = asyncio.create_task(session.run_generation(request, backend))
        await asyncio.sleep(0)
        assert not retry.done()
        assert backend.calls == []
        assert session.request_lock.locked()
    finally:
        release.set()
        await asyncio.gather(terminal, *([retry] if retry is not None else []), return_exceptions=True)
        await asyncio.to_thread(finished.wait, 5)

    assert terminal.cancelled()
    assert retry is not None
    assert retry.result().assistant_msg["content"] == "B"
    trajectories = await session.finalize()
    assert [trajectory.response_ids for trajectory in trajectories] == [[65], [66]]
    state = pickle.loads(session.storage_path.read_bytes())
    root_children = next(children for node, children in state["node_children"] if node is state["root"])
    assert len(root_children) == len(state["chains"]) == 2
    assert all(leaf.parent is None and leaf in root_children for leaf in state["chains"].values())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("initial_logprobs", [False, True])
async def test_logprob_mode_changes_keep_one_chain(initial_logprobs):
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    messages = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), messages, logprobs=initial_logprobs)
    original = session.leaves[1]

    messages += [{"role": "assistant", "content": "A"}, {"role": "user", "content": "continue"}]
    backend = Backend("B", "C")
    await generate(session, backend, messages, logprobs=not initial_logprobs)

    assert list(session.leaves) == [1]
    assert session.leaves[1] is not original
    assert session.leaves[1].parent is original

    messages += [{"role": "assistant", "content": "B"}, {"role": "user", "content": "continue again"}]
    await generate(session, backend, messages, logprobs=not initial_logprobs)

    assert list(session.leaves) == [1]
    [trajectory] = await session.finalize()
    assert trajectory.response_ids == [ord("A"), 201, ord("B"), 201, ord("C")]
    assert trajectory.response_logprobs is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("sibling_response", ["A", "B"])
async def test_resample_and_rewrite_keep_one_pending_node_per_request(sibling_response):
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec, enable_last_assistant_rollback=True)
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)
    await generate(session, Backend(sibling_response), first)
    assert len(session.root.children) == 2
    await generate(session, Backend("C"), [*first, {"role": "user", "content": "rewrite"}])
    assert list(session.leaves) == [1, 2, 3]  # Two assistant siblings make MERGE ambiguous.
    assert all(node.status == "COMPLETED" for node in session.root.children)

    expected_chain_id = 2 if sibling_response == "A" else 1
    previous_leaf = session.leaves[expected_chain_id]
    await generate(
        session,
        Backend("D"),
        [*first, {"role": "assistant", "content": "A"}, {"role": "user", "content": "continue"}],
    )
    assert next(reversed(session.leaves)) == expected_chain_id
    assert session.leaves[expected_chain_id].parent is previous_leaf
    assert len(session.leaves) == 3


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("enable_last_assistant_rollback", "expected_chain_id", "expected_chain_ids", "expected_rollback_count"),
    [
        (True, 1, [1], 1),
        (False, 2, [1, 2], 0),
    ],
)
@pytest.mark.asyncio
async def test_rollback_switch_controls_rewrite_merge_and_fork(
    enable_last_assistant_rollback,
    expected_chain_id,
    expected_chain_ids,
    expected_rollback_count,
):
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(
        SessionHandle("test-session"), codec, enable_last_assistant_rollback=enable_last_assistant_rollback
    )
    images, videos = ["image-a", "image-b"], [("video", {"fps": 24})]
    codec.extract_multi_modal_data = AsyncMock(return_value=(images, videos))
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)

    original = session.leaves[1]
    await generate(
        session,
        Backend("B"),
        [*first, {"role": "user", "content": "rewrite"}],
    )

    assert next(reversed(session.leaves)) == expected_chain_id
    assert list(session.leaves) == expected_chain_ids
    assert session._rollback_count == expected_rollback_count
    assert not session._inflight_exact_requests
    assert (original.image_data, original.video_data) == (
        (None, None) if enable_last_assistant_rollback else (images, videos)
    )
    assert all(node.image_data == images and node.video_data == videos for node in session.root.children)
    assert session.leaves[expected_chain_id].image_data is not images
    trajectories = await session.finalize()
    assert all(trajectory.multi_modal_data == {"images": images, "videos": videos} for trajectory in trajectories)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_failed_rewrite_merge_restores_node_and_clears_inflight_request():
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec, enable_last_assistant_rollback=True)
    codec.extract_multi_modal_data = AsyncMock(
        side_effect=[(["old-image"], [("old-video", {"fps": 24})]), (["new-image"], ["new-video"])]
    )
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)
    original = session.root.children[0]
    session._rollback_mounted_branch = Mock(wraps=session._rollback_mounted_branch)

    with pytest.raises(Exception, match="backend failure"):
        await generate(
            session,
            Backend(RuntimeError("backend failure")),
            [*first, {"role": "user", "content": "rewrite"}],
        )

    assert not session._inflight_exact_requests
    assert session.root.children == [original]
    assert session.leaves == {1: original}
    removed = session._rollback_mounted_branch.call_args.kwargs["target_node"]
    assert removed.image_data is removed.video_data is None
    [trajectory] = await session.finalize()
    assert trajectory.multi_modal_data == {"images": ["old-image"], "videos": [("old-video", {"fps": 24})]}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [False, True])
async def test_cancel_while_waiting_for_commit_lock_rolls_back(rewrite):
    waiting = asyncio.Event()

    class CommitLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                waiting.set()
            return await super().acquire()

    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec, enable_last_assistant_rollback=True)
    session.request_lock = CommitLock()
    messages = [{"role": "user", "content": "task"}]
    original = None
    if rewrite:
        await generate(session, Backend("A", routed_experts=[[1]]), messages)
        original = session.leaves[1]
        messages = [*messages, {"role": "user", "content": "rewrite"}]

    gate = asyncio.Event()
    task = asyncio.create_task(generate(session, Backend(gate, routed_experts=[[2]]), messages))
    while not session._inflight_exact_requests:
        await asyncio.sleep(0)
    waiter = asyncio.create_task(generate(session, Backend("UNEXPECTED"), messages))
    await asyncio.sleep(0)
    assert session.snapshot_state()["num_coalesced_requests"] == 1
    async with session.request_lock:
        gate.set()
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()  # Cleanup must acquire the request lock first.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=5)

    assert not session._inflight_exact_requests
    assert session.root.children == ([original] if rewrite else [])
    assert session.leaves == ({1: original} if rewrite else {})
    assert session._routed_experts_by_chain == ({1: [[1]]} if rewrite else {})
    await generate(session, Backend("B"), messages)
    assert list(session.leaves) == [1]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("trim_boundary", [0, 2])
async def test_capacity_is_checked_after_context_merge(trim_boundary):
    codec = FakeContinuousCodec(trim_boundary=trim_boundary)
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)
    backend = Backend("B")
    result = await generate(
        session,
        backend,
        [*first, {"role": "assistant", "content": "A"}, {"role": "tool", "content": "result"}],
        trajectory_capacity=3,
    )
    assert codec.context_calls == 1
    assert not session._inflight_exact_requests
    assert session.snapshot_state()["active_chain_ids"] == ([1] if trim_boundary else [])
    assert session.snapshot_state()["num_trajectories"] == int(not trim_boundary)
    [trajectory] = await session.finalize()
    if trim_boundary:
        assert backend.calls[-1]["prompt_ids"] == [101, 201]
        assert backend.calls[-1]["sampling_params"]["max_tokens"] == 1
        assert trajectory.prompt_ids == [101]
        assert trajectory.response_ids == [201, ord("B")]
    else:
        assert (result.finish_reason, result.prompt_tokens) == ("length", 3)
        assert backend.calls == []
        assert trajectory.response_ids == [ord("A")]
        assert trajectory.extra_fields["materialization_reason"] == "max_trajectory_length"


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("coalesce", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_pending_siblings_are_not_matched_and_failed_request_is_removed(coalesce, fails):
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec, coalesce_reserved_exact_requests=coalesce)
    codec.canonicalize_message_for_prefix_comparison = MessageCodec(
        FakeTokenizer()
    ).canonicalize_message_for_prefix_comparison
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)
    ancestor = session.leaves[1]
    first = [*first, {"role": "assistant", "content": "A"}, {"role": "tool", "content": "result", "tool_call_id": "a"}]
    second = [*first[:-1], {**first[-1], "tool_call_id": "b"}] if coalesce else first
    assert session._messages_equal(first, second)
    gate = asyncio.Event()
    pending = asyncio.create_task(generate(session, Backend(gate), first))
    while not ancestor.children:
        await asyncio.sleep(0)
    pending_node = ancestor.children[0]
    assert pending_node.status == "PENDING"
    backend = Backend(RuntimeError("backend failure") if fails else "B")
    if fails:
        with pytest.raises(Exception, match="backend failure"):
            await generate(session, backend, second)
    else:
        await generate(session, backend, second)
    assert len(backend.calls) == 1
    assert len(ancestor.children) == (1 if fails else 2)
    assert len(session._inflight_exact_requests) == int(coalesce)
    assert session.snapshot_state()["num_coalesced_requests"] == 0
    gate.set()
    await pending
    assert pending_node.status == "COMPLETED"
    assert not session._inflight_exact_requests
    trajectories = await session.finalize()
    assert [t.response_ids for t in trajectories] == (
        [[ord("A"), 201, ord("A")]] if fails else [[ord("B")], [ord("A"), 201, ord("A")]]
    )
    assert [t.response_mask for t in trajectories] == ([[1, 0, 1]] if fails else [[1], [1, 0, 1]])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,rewrite",
    [
        ("success", False),
        ("success", True),
        ("capacity", False),
        ("capacity", True),
    ],
)
async def test_chain_registry_uses_contiguous_ids_and_tracks_committed_leaves(outcome, rewrite):
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    codec.extract_multi_modal_data = AsyncMock(side_effect=[(["image"], ["video"]), *[(None, None)] * 3])
    first = [{"role": "user", "content": "task"}]

    await generate(session, Backend("A", routed_experts=[[1]]), first)
    assert list(session.leaves) == [1]
    history = [*first, {"role": "assistant", "content": "A"}]
    await generate(
        session,
        Backend("B", routed_experts=[[2]]),
        [*history, {"role": "user", "content": "branch one"}],
    )
    fork_routing = None if rewrite else [[3]]
    assert list(session.leaves) == [1]
    await generate(
        session,
        Backend("C", routed_experts=fork_routing),
        [*history, {"role": "user", "content": "branch two"}],
    )

    assert list(session.leaves) == [1, 2]
    [first_node] = session.root.children
    assert [child.chain_id for child in first_node.children] == [1, 2]
    expected_routing = {1: [[2]]}
    if fork_routing is not None:
        expected_routing[2] = fork_routing

    messages = [
        *history,
        {"role": "user", "content": "branch one"},
        {"role": "assistant", "content": "edited" if rewrite else "B"},
    ]
    messages.append({"role": "user", "content": "continue"})
    new_routing = None if rewrite else [[4]]
    backend = Backend("D", routed_experts=new_routing)
    result = await generate(
        session,
        backend,
        messages,
        trajectory_capacity=1 if outcome == "capacity" else None,
    )
    assert (result.finish_reason == "length") == (outcome == "capacity")
    assert len(backend.calls) == int(outcome == "success")

    assert list(session.leaves) == [2, 1]
    assert session.snapshot_state()["active_chain_ids"] == ([2] if outcome == "capacity" else [1, 2])
    assert session.snapshot_state()["num_trajectories"] == int(outcome == "capacity")
    assert [chain.chain_id for chain in session.active_chains] == ([2] if outcome == "capacity" else [1, 2])
    assert not session._inflight_exact_requests
    if outcome == "success" and new_routing is not None:
        expected_routing[1] = new_routing
    assert session._routed_experts_by_chain == {
        chain_id: routing for chain_id, routing in expected_routing.items() if outcome != "capacity" or chain_id != 1
    }
    trajectories = await session.finalize()
    assert [trajectory.chain_id for trajectory in trajectories] == [2, 1]
    assert all(trajectory.multi_modal_data == {"images": ["image"], "videos": ["video"]} for trajectory in trajectories)
    assert {trajectory.chain_id: trajectory.routed_experts for trajectory in trajectories} == {
        1: expected_routing[1],
        2: fork_routing,
    }
    if outcome == "success":
        assert trajectories[-1].response_ids[-1] == ord("D")
    elif outcome == "capacity":
        assert trajectories[-1].response_ids == ([ord("A"), 201] if rewrite else [ord("A"), 201, ord("B")])
        assert trajectories[-1].num_turns == (4 if rewrite else 5)
        if rewrite:
            assert trajectories[-1].extra_fields["rollback_count"] == 1
            assert trajectories[-1].extra_fields["rollback_dropped_trainable_tokens_total"] == 1
        assert trajectories[-1].extra_fields["materialization_reason"] == "max_trajectory_length"
    assert session.root is None
    assert first_node.parent is None
    assert first_node.children == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("trim_boundary", [1, 3])
@pytest.mark.parametrize("initial_logprobs", [False, True])
async def test_chain_ownership_preserves_prompt_boundary_and_commit_order(trim_boundary, initial_logprobs):
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("AB"), first, logprobs=initial_logprobs)
    session.leaves[1].min_global_steps = session.leaves[1].max_global_steps = 1
    history = [*first, {"role": "assistant", "content": "AB"}]
    codec.trim_boundary = trim_boundary
    await generate(
        session,
        Backend("BBBB"),
        [*history, {"role": "user", "content": "short"}],
        logprobs=True,
    )
    codec.trim_boundary = 0
    fork_backend = Backend("C")
    await generate(
        session,
        fork_backend,
        [*history, {"role": "user", "content": "long"}],
        logprobs=True,
    )

    await generate(
        session,
        Backend("D"),
        [
            *history,
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "BBBB"},
            {"role": "user", "content": "continue"},
        ],
        logprobs=True,
    )
    long, short = await session.finalize()
    assert [long.chain_id, short.chain_id] == [2, 1]
    retained = max(0, 2 - trim_boundary)
    assert short.response_mask == [1] * retained + [0, 1, 1, 1, 1, 0, 1]
    assert long.prompt_ids == fork_backend.calls[0]["prompt_ids"]
    assert long.response_ids == [ord("C")]
    assert long.response_mask == [1]
    assert long.response_logprobs == [-0.1]
    assert long.extra_fields["min_global_steps"] == long.extra_fields["max_global_steps"] == 2
    assert short.extra_fields["min_global_steps"] == 1
    if retained and not initial_logprobs:
        assert short.response_logprobs is None
    else:
        assert short.response_logprobs == [-0.1 if mask else 0.0 for mask in short.response_mask]
    assert short.response_ids[-1] == ord("D")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_new_chain_logprobs_do_not_require_another_chains_history():
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)
    history = [*first, {"role": "assistant", "content": "A"}]
    await generate(session, Backend("B"), [*history, {"role": "user", "content": "original"}])
    fork = [*history, {"role": "user", "content": "fork"}]
    backend = Backend("C", "D")
    await generate(session, backend, fork, logprobs=True)
    await generate(
        session,
        backend,
        [*fork, {"role": "assistant", "content": "C"}, {"role": "user", "content": "continue"}],
        logprobs=True,
    )
    original, forked = await session.finalize()
    assert [original.chain_id, forked.chain_id] == [1, 2]
    assert original.response_logprobs is None
    assert forked.prompt_ids == backend.calls[0]["prompt_ids"]
    assert forked.response_ids == [ord("C"), 201, ord("D")]
    assert forked.response_mask == [1, 0, 1]
    assert forked.response_logprobs == [-0.1, 0.0, -0.1]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_concurrent_new_chains_allocate_ids_in_commit_order(cancel_waiter):
    codec = FakeContinuousCodec()
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    gate = asyncio.Event()
    slow = asyncio.create_task(generate(session, Backend(gate), [{"role": "user", "content": "slow"}]))
    while not session.root.children:
        await asyncio.sleep(0)
    assert len(session._inflight_exact_requests) == 1
    duplicate = asyncio.create_task(generate(session, Backend("UNEXPECTED"), [{"role": "user", "content": "slow"}]))
    await asyncio.sleep(0)
    assert session.snapshot_state()["num_coalesced_requests"] == 1
    if cancel_waiter:
        duplicate.cancel()
        with pytest.raises(asyncio.CancelledError):
            await duplicate
        assert not slow.done()

    await generate(
        session,
        Backend("B"),
        [{"role": "user", "content": "fast"}],
    )
    assert len(session._inflight_exact_requests) == 1
    assert list(session.leaves) == [1]
    gate.set()
    slow_result = await slow
    if not cancel_waiter:
        assert await duplicate == slow_result

    assert list(session.leaves) == [1, 2]
    assert not session._inflight_exact_requests

    gate.clear()
    continuation = [
        {"role": "user", "content": "fast"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "continue"},
    ]
    slow = asyncio.create_task(generate(session, Backend(gate), continuation))
    while not session._inflight_exact_requests:
        await asyncio.sleep(0)
    await generate(session, Backend("C"), [{"role": "user", "content": "new chain"}])
    gate.set()
    await slow
    trajectories = await session.finalize()
    assert [trajectory.chain_id for trajectory in trajectories] == [2, 3, 1]
    assert trajectories[-1].response_ids[-1] == ord("A")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_action", ["finalize", "abort"])
@pytest.mark.parametrize("rewrite", [False, True])
async def test_terminal_action_rejects_pending_owner_and_waiter(terminal_action, rewrite):
    session = TreeGatewaySession(SessionHandle("test-session"), FakeContinuousCodec())
    first = [{"role": "user", "content": "task"}]
    await generate(session, Backend("A"), first)
    original = session.leaves[1]
    messages = [
        *first,
        *([] if rewrite else [{"role": "assistant", "content": "A"}]),
        {"role": "user", "content": "continue"},
    ]
    gate = asyncio.Event()
    backend = Backend(gate)
    owner = asyncio.create_task(generate(session, backend, messages))
    waiter = None
    try:
        while not backend.calls:
            await asyncio.sleep(0)
        waiter = asyncio.create_task(generate(session, backend, messages))
        await asyncio.sleep(0)
        assert not session.request_lock.locked()
        assert session.active_chains[0].message_history == [*first, {"role": "assistant", "content": "A"}]
        trajectories = await asyncio.wait_for(getattr(session, terminal_action)(), timeout=5)
        if terminal_action == "finalize":
            assert [trajectory.response_ids for trajectory in trajectories] == [[ord("A")]]
    finally:
        gate.set()
        results = await asyncio.wait_for(
            asyncio.gather(owner, *([waiter] if waiter is not None else []), return_exceptions=True), timeout=5
        )
    assert len(results) == 2 and all(isinstance(error, HTTPException) and error.status_code == 409 for error in results)
    assert len(backend.calls) == 1 and not session._inflight_exact_requests
    assert session.root is None and session.leaves == {} and session.active_chains == []
    assert original.parent is None and original.children == []
    with pytest.raises(HTTPException) as error:
        await generate(session, backend, messages)
    assert error.value.status_code == 409 and len(backend.calls) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_actor_selects_session_and_captures_linear_token_truth(tmp_path):
    from uni_agent.gateway.gateway import _GatewayActor

    media = (["image-a", "image-b"], [("video", {"fps": 24})])
    trajectories = []
    for use_tree in (False, True):
        options = {"use_tree_trajectory_manager": True} if use_tree else {}
        actor = _GatewayActor(GatewayActorConfig(tokenizer=FakeTokenizer(), **options), SequencedBackend(["A", "B"]))
        actor._server_base_url = "http://test"
        await actor.create_session("mode-selection", metadata={"_log_dir": str(tmp_path)})
        session = actor._sessions["mode-selection"]
        session._codec.extract_multi_modal_data = AsyncMock(side_effect=[media, (None, None)])
        if use_tree:
            assert isinstance(session, TreeGatewaySession)
            assert session.storage_path == tmp_path / "trajectory_tree.pkl"
        else:
            assert isinstance(session, GatewaySession)
        backend = actor._backend
        messages = [{"role": "user", "content": "task"}]
        for _ in range(2):
            result = await session.run_generation(
                {"messages": messages, "tools": None, "sampling_params": {"max_tokens": 32, "logprobs": True}},
                backend,
            )
            messages = [*messages, result.assistant_msg, {"role": "user", "content": "continue"}]

        [active_chain] = session.active_chains
        assert active_chain.message_history == messages[:-1]
        [trajectory] = await session.finalize()
        assert session.active_chains == []
        trajectories.append(trajectory)
        assert trajectory.prompt_ids + trajectory.response_ids == backend.calls[1]["prompt_ids"] + [ord("B")]
        assert sum(trajectory.response_mask) == 2
        assert [p for p, mask in zip(trajectory.response_logprobs, trajectory.response_mask, strict=True) if mask] == [
            -0.1,
            -0.1,
        ]
        assert [(call["image_data"], call["video_data"]) for call in backend.calls] == [media, media]
    assert trajectories[0] == trajectories[1]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_action,has_log_dir,write_fails",
    [
        ("finalize", True, True),
        ("abort", True, False),
        ("abort", True, True),
        ("finalize", False, False),
    ],
)
async def test_tree_persistence_follows_session_log_directory(tmp_path, terminal_action, has_log_dir, write_fails):
    log_dir = tmp_path / "step_12" / "tree-persistence"
    session = TreeGatewaySession(
        SessionHandle("tree-persistence"),
        MessageCodec(FakeTokenizer()),
        metadata={"_log_dir": str(log_dir) if has_log_dir else None},
    )
    await session.run_generation(
        {"messages": [{"role": "user", "content": "task"}], "tools": None, "sampling_params": {"max_tokens": 32}},
        SequencedBackend(["A"]),
    )
    await generate(
        session,
        Backend("UNUSED"),
        [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "continue"},
        ],
        trajectory_capacity=1,
    )
    assert isinstance(session.leaves[1], Trajectory)
    assert session.root is not None
    assert session.storage_path == (log_dir / "trajectory_tree.pkl" if has_log_dir else None)
    assert not list(tmp_path.iterdir())
    writer = Mock(wraps=session._write_storage_state, side_effect=OSError("disk failure") if write_fails else None)
    session._write_storage_state = writer
    result = await getattr(session, terminal_action)()
    if terminal_action == "finalize":
        [trajectory] = result
        assert trajectory.response_ids == [ord("A")]
        assert trajectory.response_mask == [1]
    assert session.root is None
    assert session.leaves == session.tool_schemas_by_fingerprint == session._routed_experts_by_chain == {}
    assert session.active_chains == []
    assert session.snapshot_state()["active_chain_ids"] == []
    if not has_log_dir:
        writer.assert_not_called()
        assert session.storage_path is None
        assert not list(tmp_path.iterdir())
        return
    expected_path = log_dir / "trajectory_tree.pkl"
    if terminal_action == "abort":
        expected_path = log_dir / "trajectory_tree_abort.pkl"
    assert writer.call_count == 1
    assert writer.call_args.args[0] == expected_path
    if write_fails:
        assert not session.storage_path.exists()
        assert session.phase.value == ("ABORTED" if terminal_action == "abort" else "FINALIZED")
        return
    assert session.storage_path == expected_path
    if terminal_action == "abort":
        assert not (log_dir / "trajectory_tree.pkl").exists()
    state = pickle.loads(session.storage_path.read_bytes())
    assert isinstance(state["root"], MessageNode)
    assert list(state["chains"]) == [1]
    assert isinstance(state["chains"][1], Trajectory)
    assert all(node.parent is None and node.children == [] for node, _ in state["node_children"])


@pytest.mark.cpu
@pytest.mark.level0
def test_gateway_actor_config_rejects_non_bool_tree_option():
    with pytest.raises(ValueError, match="use_tree_trajectory_manager must be a bool"):
        GatewayActorConfig(tokenizer=FakeTokenizer(), use_tree_trajectory_manager="true")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("merge_kind", ["assistant", "context"])
def test_codec_merge_returns_raw_edits_only_when_mask_is_none(merge_kind):
    codec = MessageCodec.__new__(MessageCodec)
    builder = Mock()
    codec._continuous_token_builder = builder
    edits = MergeResult(token_ids=[10, 11], appended_token_count=1, kind=merge_kind)
    builder.merge_assistant_tokens.return_value = edits
    builder.merge_context_tokens.return_value = edits
    builder.align_response_metadata.return_value = ([1], [-0.1])
    previous = [{"role": "user", "content": "task"}]
    args = (
        ([10], [11])
        if merge_kind == "assistant"
        else (previous, [*previous, {"role": "tool", "content": "result"}], [10])
    )
    merge = getattr(codec, f"merge_{merge_kind}_tokens")
    assert merge(*args) is edits
    builder.align_response_metadata.assert_not_called()
    assert merge(*args, response_mask=[]) == ([10, 11], [1], [-0.1])
    kwargs = {"assistant_logprobs": None} if merge_kind == "assistant" else {}
    builder.align_response_metadata.assert_called_once_with(edits, [], None, **kwargs)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_tree_codec_merges_complete_images_and_stores_only_node_deltas():
    tokenizer = FakeTokenizer()

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)

        def apply_chat_template(self, *args, **kwargs):
            return tokenizer.apply_chat_template(*args, **kwargs)

        def __call__(self, *, text, **kwargs):
            return {"input_ids": [tokenizer.encode(text[0])]}

    def image_message(url):
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "describe"},
            ],
        }

    extractor = AsyncMock(wraps=fake_vision_info_extractor)
    codec = MessageCodec(tokenizer, processor=Processor(), vision_info_extractor=extractor)
    codec.merge_context_tokens = Mock(wraps=codec.merge_context_tokens)
    codec._continuous_token_builder.align_response_metadata = Mock(
        side_effect=AssertionError("Tree must retain raw token edits")
    )
    session = TreeGatewaySession(SessionHandle("test-session"), codec)
    backend = Backend("A", "B", "C")
    first = [image_message("image://source.png")]
    await generate(session, backend, first, logprobs=True)
    first_node = session.leaves[1]
    added_image = image_message("image://crop.png")
    added_image.update(role="tool", tool_call_id="crop")
    second = [*first, {"role": "assistant", "content": "A"}, added_image]
    await generate(session, backend, second, logprobs=True)
    second_node = session.leaves[1]
    await generate(
        session,
        backend,
        [*second, {"role": "assistant", "content": "B"}, {"role": "user", "content": "continue"}],
        logprobs=True,
    )

    original_images = ["image://source.png"]
    all_images = [*original_images, "image://crop.png"]
    assert [call.kwargs["image_data"] for call in codec.merge_context_tokens.call_args_list] == [all_images, all_images]
    assert [call["image_data"] for call in backend.calls] == [original_images or None, all_images, all_images]
    assert first_node.image_data == (original_images or None)
    assert second_node.image_data == ["image://crop.png"]
    assert session.leaves[1].image_data is None
    assert extractor.await_count == 2
    codec._continuous_token_builder.align_response_metadata.assert_not_called()
    [trajectory] = await session.finalize()
    assert trajectory.multi_modal_data == {"images": all_images}
    assert trajectory.prompt_ids + trajectory.response_ids == backend.calls[-1]["prompt_ids"] + [ord("C")]
    assert sum(trajectory.response_mask) == 3
    assert [p for p, mask in zip(trajectory.response_logprobs, trajectory.response_mask, strict=True) if mask] == [
        -0.1,
        -0.1,
        -0.1,
    ]
