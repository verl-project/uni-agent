"""Tree trajectory behavior and session mode compatibility; CPU-only."""

from __future__ import annotations

import asyncio
import pickle
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.uni_agent.support import FakeTokenizer, SequencedBackend, fake_vision_info_extractor
from uni_agent.gateway.config import GatewayActorConfig
from uni_agent.gateway.session import MessageCodec
from uni_agent.gateway.session.chain_trajectory_manager import ChainTrajectoryManager
from uni_agent.gateway.session.session import GatewaySession
from uni_agent.gateway.session.tree_trajectory_manager import MessageNode, TreeTrajectoryManager
from uni_agent.gateway.session.types import SessionHandle, Trajectory, UnsupportedMediaError
from verl.utils.tokenizer.continuous_token import MergeResult

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


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


async def generate(manager, codec, backend, messages, *, logprobs=False, trajectory_capacity=None):
    async def generate_fn(context_ids, sampling_params, image_data, video_data):
        return await backend.generate(
            request_id="test-session",
            prompt_ids=context_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            mm_processor_kwargs=codec.mm_processor_kwargs,
        )

    return await manager.run_generation(
        messages, None, codec, trajectory_capacity, {"max_tokens": 32, "logprobs": logprobs}, generate_fn
    )


@pytest.mark.asyncio
async def test_continuous_context_merge_preserves_ancestors_and_metadata():
    manager = TreeTrajectoryManager()
    codec = FakeContinuousCodec(trim_boundary=True)
    backend = Backend("AB", "C", "D", routed_experts=[[1]])
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, backend, first, logprobs=True)
    original = manager.chains[1]
    backend.routed_experts = [[2]]
    continuation = [
        *first,
        {"role": "assistant", "content": "AB"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "historical reply"},
        {"role": "tool", "content": "more context"},
    ]
    await generate(manager, codec, backend, continuation, logprobs=True)
    [child] = original.children
    assert child.messages == continuation[2:] + [{"role": "assistant", "content": "C"}]
    history = [*continuation, {"role": "assistant", "content": "C"}, {"role": "tool", "content": "next"}]
    await generate(manager, codec, backend, history, logprobs=True)

    assert original.response_ids == [65, 66]
    assert backend.calls[-1]["prompt_ids"] == [101, 1, 65, 203, 201]
    [trajectory] = await manager.finalize_trajectories()
    assert trajectory.response_ids == [65, 203, 201, 68]
    assert trajectory.response_mask == [1, 0, 0, 1]
    assert trajectory.response_logprobs == [-0.1, 0.0, 0.0, -0.1]
    assert trajectory.routed_experts is backend.routed_experts
    assert manager._routed_experts_by_chain == {}


@pytest.mark.asyncio
async def test_incremental_video_from_extractor_is_rejected_before_context_merge():
    class Codec(FakeContinuousCodec):
        async def extract_multi_modal_data(self, messages):
            return (None, [object()]) if messages[0]["role"] == "tool" else (None, None)

    manager = TreeTrajectoryManager()
    codec = Codec()
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("A"), first)
    with pytest.raises(UnsupportedMediaError, match="incremental video"):
        await generate(
            manager,
            codec,
            Backend("B"),
            [*first, {"role": "assistant", "content": "A"}, {"role": "tool", "content": "video"}],
        )
    assert codec.context_calls == 0 and not manager._pending_requests
    assert manager.chains[1].children == []


@pytest.mark.asyncio
async def test_missing_backend_logprobs_still_fail_for_empty_output():
    async def generate_fn(*args):
        return SimpleNamespace(token_ids=[], log_probs=None)

    manager = TreeTrajectoryManager()
    with pytest.raises(RuntimeError, match="omitted logprobs"):
        await manager.run_generation(
            [{"role": "user", "content": "task"}],
            None,
            FakeContinuousCodec(),
            None,
            {"logprobs": True},
            generate_fn,
        )
    assert manager.root.children == [] and not manager._pending_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_action,write_fails", [("finalize", False), ("abort", True)])
async def test_cancelled_terminal_write_finishes_before_session_retry(tmp_path, terminal_action, write_fails):
    session = GatewaySession(
        SessionHandle("cancelled-write"),
        FakeContinuousCodec(),
        use_tree_trajectory_manager=True,
        metadata={"_log_dir": str(tmp_path)},
    )
    request = {"messages": [{"role": "user", "content": "task"}], "tools": None, "sampling_params": {}}
    await session.run_generation(request, Backend("A"))
    manager = session._manager
    write = manager._write_storage_state
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

    manager._write_storage_state = delayed_write
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
        assert manager._tree_lock.locked()
    finally:
        release.set()
        await asyncio.gather(terminal, *([retry] if retry is not None else []), return_exceptions=True)
        await asyncio.to_thread(finished.wait, 5)

    assert terminal.cancelled()
    assert retry is not None
    assert retry.result().assistant_msg["content"] == "B"
    trajectories = await session.finalize()
    assert [trajectory.response_ids for trajectory in trajectories] == [[65], [66]]
    state = pickle.loads(manager.storage_path.read_bytes())
    root_children = next(children for node, children in state["node_children"] if node is state["root"])
    assert len(root_children) == len(state["chains"]) == 2
    assert all(leaf.parent is None and leaf in root_children for leaf in state["chains"].values())


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_logprobs", [False, True])
async def test_logprob_mode_changes_require_aligned_history(initial_logprobs):
    manager = TreeTrajectoryManager()
    codec = FakeContinuousCodec()
    messages = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("A"), messages, logprobs=initial_logprobs)
    messages += [{"role": "assistant", "content": "A"}, {"role": "user", "content": "continue"}]
    backend = Backend("B")
    if initial_logprobs:
        await generate(manager, codec, backend, messages, logprobs=False)
    else:
        with pytest.raises(AssertionError, match="response_logprobs must be empty or aligned"):
            await generate(manager, codec, backend, messages, logprobs=True)
        assert backend.calls == []
    [trajectory] = await manager.finalize_trajectories()
    assert trajectory.response_ids == ([ord("A"), 201, ord("B")] if initial_logprobs else [ord("A")])
    assert trajectory.response_logprobs is None


@pytest.mark.asyncio
async def test_resample_and_rewrite_keep_one_pending_node_per_request():
    manager = TreeTrajectoryManager(enable_last_assistant_rollback=True)
    codec = FakeContinuousCodec()
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("A"), first)
    await generate(manager, codec, Backend("B"), first)
    assert len(manager.root.children) == 2
    await generate(manager, codec, Backend("C"), [*first, {"role": "user", "content": "rewrite"}])
    assert manager.rewrite_fork_count == 1  # Two assistant siblings make MERGE ambiguous.
    assert all(node.status == "COMPLETED" for node in manager.root.children)


@pytest.mark.parametrize(
    ("enable_last_assistant_rollback", "expected_chain_id", "expected_chain_ids", "expected_counts"),
    [
        (True, 1, [1], (1, 0)),
        (False, 2, [1, 2], (0, 1)),
    ],
)
@pytest.mark.asyncio
async def test_rollback_switch_controls_rewrite_merge_and_fork(
    enable_last_assistant_rollback,
    expected_chain_id,
    expected_chain_ids,
    expected_counts,
):
    manager = TreeTrajectoryManager(enable_last_assistant_rollback=enable_last_assistant_rollback)
    codec = FakeContinuousCodec()
    images, videos = ["image-a", "image-b"], [("video", {"fps": 24})]
    codec.extract_multi_modal_data = AsyncMock(return_value=(images, videos))
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("A"), first)

    original = manager.chains[1]
    rewritten = await generate(
        manager,
        codec,
        Backend("B"),
        [*first, {"role": "user", "content": "rewrite"}],
    )

    assert rewritten[4] == expected_chain_id
    assert list(manager.chains) == expected_chain_ids
    assert (manager.rewrite_merge_count, manager.rewrite_fork_count) == expected_counts
    assert not manager._pending_requests
    assert (original.image_data, original.video_data) == (
        (None, None) if enable_last_assistant_rollback else (images, videos)
    )
    assert all(node.image_data == images and node.video_data == videos for node in manager.root.children)
    assert manager.chains[expected_chain_id].image_data is not images
    trajectories = await manager.finalize_trajectories()
    assert all(trajectory.multi_modal_data == {"images": images, "videos": videos} for trajectory in trajectories)


@pytest.mark.asyncio
async def test_failed_rewrite_merge_restores_node_and_pending_requests():
    manager = TreeTrajectoryManager(enable_last_assistant_rollback=True)
    codec = FakeContinuousCodec()
    codec.extract_multi_modal_data = AsyncMock(
        side_effect=[(["old-image"], [("old-video", {"fps": 24})]), (["new-image"], ["new-video"])]
    )
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("A"), first)
    original = manager.root.children[0]
    manager._rollback_mounted_branch = Mock(wraps=manager._rollback_mounted_branch)

    with pytest.raises(Exception, match="backend failure"):
        await generate(
            manager,
            codec,
            Backend(RuntimeError("backend failure")),
            [*first, {"role": "user", "content": "rewrite"}],
        )

    assert not manager._pending_requests
    assert manager.root.children == [original]
    assert manager.chains == {1: original}
    removed = manager._rollback_mounted_branch.call_args.kwargs["target_node"]
    assert removed.image_data is removed.video_data is None
    [trajectory] = await manager.finalize_trajectories()
    assert trajectory.multi_modal_data == {"images": ["old-image"], "videos": [("old-video", {"fps": 24})]}


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [False, True])
async def test_cancel_while_waiting_for_commit_lock_rolls_back(rewrite):
    waiting = asyncio.Event()

    class CommitLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                waiting.set()
            return await super().acquire()

    manager = TreeTrajectoryManager(enable_last_assistant_rollback=True)
    manager._tree_lock = CommitLock()
    codec = FakeContinuousCodec()
    messages = [{"role": "user", "content": "task"}]
    original = None
    if rewrite:
        await generate(manager, codec, Backend("A", routed_experts=[[1]]), messages)
        original = manager.chains[1]
        messages = [*messages, {"role": "user", "content": "rewrite"}]

    gate = asyncio.Event()
    task = asyncio.create_task(generate(manager, codec, Backend(gate, routed_experts=[[2]]), messages))
    while not manager._pending_requests:
        await asyncio.sleep(0)
    async with manager._tree_lock:
        gate.set()
        await asyncio.wait_for(waiting.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()  # Cleanup must acquire the tree lock first.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    assert not manager._pending_requests
    assert manager.root.children == ([original] if rewrite else [])
    assert manager.chains == ({1: original} if rewrite else {})
    assert manager._routed_experts_by_chain == ({1: [[1]]} if rewrite else {})
    result = await generate(manager, codec, Backend("B"), messages)
    assert result[4] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("trim_boundary", [0, 2])
async def test_capacity_is_checked_after_context_merge(trim_boundary):
    manager = TreeTrajectoryManager()
    codec = FakeContinuousCodec(trim_boundary=trim_boundary)
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("A"), first)
    backend = Backend("B")
    result = await generate(
        manager,
        codec,
        backend,
        [*first, {"role": "assistant", "content": "A"}, {"role": "tool", "content": "result"}],
        trajectory_capacity=3,
    )
    assert codec.context_calls == 1
    assert not manager._pending_requests
    assert manager.snapshot_state()["active_chain_ids"] == ([1] if trim_boundary else [])
    assert manager.snapshot_state()["num_trajectories"] == int(not trim_boundary)
    [trajectory] = await manager.finalize_trajectories()
    if trim_boundary:
        assert backend.calls[-1]["prompt_ids"] == [101, 201]
        assert backend.calls[-1]["sampling_params"]["max_tokens"] == 1
        assert trajectory.prompt_ids == [101]
        assert trajectory.response_ids == [201, ord("B")]
    else:
        assert result[1:3] == ("length", 3)
        assert backend.calls == []
        assert trajectory.response_ids == [ord("A")]
        assert trajectory.extra_fields["materialization_reason"] == "max_trajectory_length"


@pytest.mark.asyncio
async def test_pending_siblings_are_not_matched_and_failed_request_is_removed():
    manager = TreeTrajectoryManager(coalesce_reserved_exact_requests=False)
    codec = FakeContinuousCodec()
    first = [{"role": "user", "content": "task"}]
    gate = asyncio.Event()
    pending = asyncio.create_task(generate(manager, codec, Backend(gate), first))
    while not manager.root.children:
        await asyncio.sleep(0)
    pending_node = manager.root.children[0]
    assert pending_node.status == "PENDING"
    assert len(manager._pending_requests) == 1
    with pytest.raises(Exception, match="backend failure"):
        await generate(manager, codec, Backend(RuntimeError("backend failure")), first)
    assert len(manager.root.children) == 1
    assert len(manager._pending_requests) == 1
    gate.set()
    await pending
    assert pending_node.status == "COMPLETED"
    assert not manager._pending_requests


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
    manager = TreeTrajectoryManager()
    codec = FakeContinuousCodec()
    codec.extract_multi_modal_data = AsyncMock(side_effect=[(["image"], ["video"]), *[(None, None)] * 3])
    first = [{"role": "user", "content": "task"}]

    first_result = await generate(manager, codec, Backend("A", routed_experts=[[1]]), first)
    history = [*first, {"role": "assistant", "content": "A"}]
    continued_result = await generate(
        manager,
        codec,
        Backend("B", routed_experts=[[2]]),
        [*history, {"role": "user", "content": "branch one"}],
    )
    fork_routing = None if rewrite else [[3]]
    forked_result = await generate(
        manager,
        codec,
        Backend("C", routed_experts=fork_routing),
        [*history, {"role": "user", "content": "branch two"}],
    )

    assert [first_result[4], continued_result[4], forked_result[4]] == [1, 1, 2]
    [first_node] = manager.root.children
    assert [child.chain_id for child in first_node.children] == [1, 2]
    expected_routing = {1: [[2]]}
    if fork_routing is not None:
        expected_routing[2] = fork_routing

    messages = [
        *history,
        {"role": "user", "content": "branch one"},
        {"role": "assistant", "content": "edited" if rewrite else "B"},
    ]
    if not rewrite:
        messages.append({"role": "user", "content": "continue"})
    new_routing = None if rewrite else [[4]]
    backend = Backend("D", routed_experts=new_routing)
    result = await generate(
        manager,
        codec,
        backend,
        messages,
        trajectory_capacity=1 if outcome == "capacity" else None,
    )
    assert result[6] == (outcome == "capacity")
    assert len(backend.calls) == int(outcome == "success")

    assert list(manager.chains) == [2, 1]
    assert manager.snapshot_state()["active_chain_ids"] == ([2] if outcome == "capacity" else [1, 2])
    assert manager.snapshot_state()["num_trajectories"] == int(outcome == "capacity")
    assert [chain["chain_id"] for chain in manager.get_active_chains()] == ([2] if outcome == "capacity" else [1, 2])
    assert not manager._pending_requests
    if outcome == "success" and new_routing is not None:
        expected_routing[1] = new_routing
    assert manager._routed_experts_by_chain == {
        chain_id: routing for chain_id, routing in expected_routing.items() if outcome != "capacity" or chain_id != 1
    }
    trajectories = await manager.finalize_trajectories()
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
    assert manager.root is None
    assert first_node.parent is None
    assert first_node.children == []


@pytest.mark.asyncio
@pytest.mark.parametrize("trim_boundary", [1, 3])
async def test_chain_ownership_preserves_prompt_boundary_and_commit_order(trim_boundary):
    manager = TreeTrajectoryManager()
    codec = FakeContinuousCodec()
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("AB"), first, logprobs=True)
    manager.chains[1].min_global_steps = manager.chains[1].max_global_steps = 1
    history = [*first, {"role": "assistant", "content": "AB"}]
    codec.trim_boundary = trim_boundary
    await generate(
        manager,
        codec,
        Backend("BBBB"),
        [*history, {"role": "user", "content": "short"}],
        logprobs=True,
    )
    codec.trim_boundary = 0
    fork_backend = Backend("C")
    await generate(
        manager,
        codec,
        fork_backend,
        [*history, {"role": "user", "content": "long"}],
        logprobs=True,
    )

    await generate(
        manager,
        codec,
        Backend("D"),
        [
            *history,
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "BBBB"},
            {"role": "user", "content": "continue"},
        ],
        logprobs=True,
    )
    long, short = await manager.finalize_trajectories()
    assert [long.chain_id, short.chain_id] == [2, 1]
    retained = max(0, 2 - trim_boundary)
    assert short.response_mask == [1] * retained + [0, 1, 1, 1, 1, 0, 1]
    assert long.prompt_ids == fork_backend.calls[0]["prompt_ids"]
    assert long.response_ids == [ord("C")]
    assert long.response_mask == [1]
    assert long.response_logprobs == [-0.1]
    assert long.extra_fields["min_global_steps"] == long.extra_fields["max_global_steps"] == 2
    assert short.extra_fields["min_global_steps"] == 1
    assert short.response_logprobs == [-0.1 if mask else 0.0 for mask in short.response_mask]
    assert short.response_ids[-1] == ord("D")


@pytest.mark.asyncio
async def test_new_chain_logprobs_do_not_require_another_chains_history():
    manager = TreeTrajectoryManager()
    codec = FakeContinuousCodec()
    first = [{"role": "user", "content": "task"}]
    await generate(manager, codec, Backend("A"), first)
    history = [*first, {"role": "assistant", "content": "A"}]
    await generate(manager, codec, Backend("B"), [*history, {"role": "user", "content": "original"}])
    fork = [*history, {"role": "user", "content": "fork"}]
    backend = Backend("C", "D")
    await generate(manager, codec, backend, fork, logprobs=True)
    await generate(
        manager,
        codec,
        backend,
        [*fork, {"role": "assistant", "content": "C"}, {"role": "user", "content": "continue"}],
        logprobs=True,
    )
    original, forked = await manager.finalize_trajectories()
    assert [original.chain_id, forked.chain_id] == [1, 2]
    assert original.response_logprobs is None
    assert forked.prompt_ids == backend.calls[0]["prompt_ids"]
    assert forked.response_ids == [ord("C"), 201, ord("D")]
    assert forked.response_mask == [1, 0, 1]
    assert forked.response_logprobs == [-0.1, 0.0, -0.1]


@pytest.mark.asyncio
async def test_concurrent_new_chains_allocate_ids_in_commit_order():
    manager = TreeTrajectoryManager()
    codec = FakeContinuousCodec()
    gate = asyncio.Event()
    slow = asyncio.create_task(generate(manager, codec, Backend(gate), [{"role": "user", "content": "slow"}]))
    while not manager.root.children:
        await asyncio.sleep(0)
    assert len(manager._pending_requests) == 1

    fast_result = await generate(
        manager,
        codec,
        Backend("B"),
        [{"role": "user", "content": "fast"}],
    )
    assert len(manager._pending_requests) == 1
    gate.set()
    slow_result = await slow

    assert [fast_result[4], slow_result[4]] == [1, 2]
    assert list(manager.chains) == [1, 2]
    assert not manager._pending_requests

    gate.clear()
    continuation = [
        {"role": "user", "content": "fast"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "continue"},
    ]
    slow = asyncio.create_task(generate(manager, codec, Backend(gate), continuation))
    while not manager._pending_requests:
        await asyncio.sleep(0)
    await generate(manager, codec, Backend("C"), [{"role": "user", "content": "new chain"}])
    gate.set()
    await slow
    trajectories = await manager.finalize_trajectories()
    assert [trajectory.chain_id for trajectory in trajectories] == [2, 3, 1]
    assert trajectories[-1].response_ids[-1] == ord("A")


@pytest.mark.asyncio
async def test_session_selects_manager_and_captures_linear_token_truth(tmp_path):
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
        assert isinstance(session._manager, TreeTrajectoryManager if use_tree else ChainTrajectoryManager)
        if use_tree:
            assert session._manager.storage_path == tmp_path / "trajectory_tree.pkl"
        backend = actor._backend
        messages = [{"role": "user", "content": "task"}]
        for _ in range(2):
            result = await session.run_generation(
                {"messages": messages, "tools": None, "sampling_params": {"max_tokens": 32, "logprobs": True}},
                backend,
            )
            messages = [*messages, result.assistant_msg, {"role": "user", "content": "continue"}]

        manager = session._manager
        async with manager._tree_lock if use_tree else manager.request_lock:
            [active_chain] = session.get_active_chains()
            assert active_chain["message_history"] == messages[:-1]
        [trajectory] = await session.finalize()
        assert session.get_active_chains() == []
        trajectories.append(trajectory)
        assert trajectory.prompt_ids + trajectory.response_ids == backend.calls[1]["prompt_ids"] + [ord("B")]
        assert sum(trajectory.response_mask) == 2
        assert [p for p, mask in zip(trajectory.response_logprobs, trajectory.response_mask, strict=True) if mask] == [
            -0.1,
            -0.1,
        ]
        assert [(call["image_data"], call["video_data"]) for call in backend.calls] == [media, media]
    assert trajectories[0] == trajectories[1]


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
    session = GatewaySession(
        SessionHandle("tree-persistence"),
        MessageCodec(FakeTokenizer()),
        use_tree_trajectory_manager=True,
        metadata={"_log_dir": str(log_dir) if has_log_dir else None},
    )
    await session.run_generation(
        {"messages": [{"role": "user", "content": "task"}], "tools": None, "sampling_params": {"max_tokens": 32}},
        SequencedBackend(["A"]),
    )
    manager = session._manager
    await generate(
        manager,
        session._codec,
        Backend("UNUSED"),
        [{"role": "user", "content": "task"}, {"role": "assistant", "content": "A"}],
        trajectory_capacity=1,
    )
    assert isinstance(manager.chains[1], Trajectory)
    assert manager.root is not None
    assert manager.storage_path == (log_dir / "trajectory_tree.pkl" if has_log_dir else None)
    assert not list(tmp_path.iterdir())
    writer = Mock(wraps=manager._write_storage_state, side_effect=OSError("disk failure") if write_fails else None)
    manager._write_storage_state = writer
    if terminal_action == "finalize" and write_fails:
        with pytest.raises(OSError, match="disk failure"):
            await session.finalize()
        assert manager.root is not None
        writer.side_effect = None
        writer.reset_mock()
        write_fails = False
    result = await getattr(session, terminal_action)()
    if terminal_action == "finalize":
        [trajectory] = result
        assert trajectory.response_ids == [ord("A")]
        assert trajectory.response_mask == [1]
    assert manager.root is None
    assert manager.chains == manager.tool_schemas_by_fingerprint == manager._routed_experts_by_chain == {}
    assert session.get_active_chains() == []
    assert session.snapshot_state()["active_chain_ids"] == []
    if not has_log_dir:
        writer.assert_not_called()
        assert manager.storage_path is None
        assert not list(tmp_path.iterdir())
        return
    expected_path = log_dir / "trajectory_tree.pkl"
    if terminal_action == "abort":
        expected_path = log_dir / "trajectory_tree_abort.pkl"
    assert writer.call_count == 1
    assert writer.call_args.args[0] == expected_path
    if write_fails:
        assert not manager.storage_path.exists()
        assert session.phase.value == "ABORTED"
        return
    assert manager.storage_path == expected_path
    if terminal_action == "abort":
        assert not (log_dir / "trajectory_tree.pkl").exists()
    state = pickle.loads(manager.storage_path.read_bytes())
    assert isinstance(state["root"], MessageNode)
    assert list(state["chains"]) == [1]
    assert isinstance(state["chains"][1], Trajectory)
    assert all(node.parent is None and node.children == [] for node, _ in state["node_children"])


def test_gateway_actor_config_rejects_non_bool_tree_option():
    with pytest.raises(ValueError, match="use_tree_trajectory_manager must be a bool"):
        GatewayActorConfig(tokenizer=FakeTokenizer(), use_tree_trajectory_manager="true")


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
    manager = TreeTrajectoryManager()
    backend = Backend("A", "B", "C")
    first = [image_message("image://source.png")]
    await generate(manager, codec, backend, first, logprobs=True)
    first_node = manager.chains[1]
    added_image = image_message("image://crop.png")
    added_image.update(role="tool", tool_call_id="crop")
    second = [*first, {"role": "assistant", "content": "A"}, added_image]
    await generate(manager, codec, backend, second, logprobs=True)
    second_node = manager.chains[1]
    await generate(
        manager,
        codec,
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
    assert manager.chains[1].image_data is None
    assert extractor.await_count == 2
    codec._continuous_token_builder.align_response_metadata.assert_not_called()
    [trajectory] = await manager.finalize_trajectories()
    assert trajectory.multi_modal_data == {"images": all_images}
    assert trajectory.prompt_ids + trajectory.response_ids == backend.calls[-1]["prompt_ids"] + [ord("C")]
    assert sum(trajectory.response_mask) == 3
    assert [p for p, mask in zip(trajectory.response_logprobs, trajectory.response_mask, strict=True) if mask] == [
        -0.1,
        -0.1,
        -0.1,
    ]
