"""CPU integration tests for the trie-backed gateway session (issue #51, M1).

Drives a real ``GatewaySession`` (real ``MessageCodec`` + ``FakeTokenizer`` +
``SequencedBackend``) end-to-end through ``run_generation``/``complete``/
``finalize`` to check:

- **flag parity**: a linear multi-turn conversation produces identical backend
  prompts and identical finalized trajectories whether the trie is on or off
  (the M1 compatibility gate);
- **multi-trajectory flag parity**: repeated best-of-N style requests produce
  identical per-trajectory token buffers whether the trie is on or off;
- **sub-agent flag parity**: switching to a different system prompt produces a
  sibling root branch with identical exported token buffers to legacy mode;
- **best-of-N**: repeated requests with the same messages fan out into N sibling
  trajectories under the trie;
- **multi-turn reuse**: a tool/assistant continuation reattaches to its branch.

These need the codec's runtime deps (verl tokenizer/template utils), so they
live alongside the other ``*_on_cpu`` gateway tests rather than in the
dependency-light pure-trie unit tests.
"""

from __future__ import annotations

import asyncio

from tests.uni_agent.support import (
    FakeProcessor,
    FakeTokenizer,
    SequencedBackend,
    fake_vision_info_extractor,
)
from uni_agent.gateway.session.codec import MessageCodec
from uni_agent.gateway.session.session import GatewaySession
from uni_agent.gateway.session.types import SessionHandle


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _session(trie_enabled, response_length=None):
    return GatewaySession(
        SessionHandle(session_id="s"),
        MessageCodec(FakeTokenizer()),
        response_length=response_length,
        trie_enabled=trie_enabled,
    )


SYS = {"role": "system", "content": "sys"}
USER = {"role": "user", "content": "fix bug"}
HELPFUL_SYS = {"role": "system", "content": "You are helpful. Reply in 1 short sentence."}
SUBAGENT_SYS = {"role": "system", "content": "You are a sub-agent. Reply in 1 short sentence."}


async def _linear_conversation(trie_enabled):
    """Two-turn linear conversation; returns (backend prompts, trajectories)."""
    backend = SequencedBackend(["A1", "A2"])
    session = _session(trie_enabled)

    out1 = await session.run_generation({"messages": [SYS, USER]}, backend)
    a1 = {"role": "assistant", "content": out1.assistant_msg["content"]}
    messages2 = [SYS, USER, a1, {"role": "user", "content": "more"}]
    await session.run_generation({"messages": messages2}, backend)

    await session.set_reward_info({"score": 1.0})
    trajectories = await session.finalize()
    prompts = [call["prompt_ids"] for call in backend.calls]
    return prompts, trajectories


async def _repeated_prompt_samples(trie_enabled):
    """Best-of-N style repeated prompt; returns (backend prompts, trajectories)."""
    backend = SequencedBackend(["ans-A", "ans-B", "ans-C"])
    session = _session(trie_enabled)

    for _ in range(3):
        await session.run_generation({"messages": [SYS, USER]}, backend)

    await session.set_reward_info({"score": 0.5})
    trajectories = await session.finalize()
    prompts = [call["prompt_ids"] for call in backend.calls]
    return prompts, trajectories


async def _subagent_system_split(trie_enabled):
    """Main-agent continuation followed by an independent sub-agent branch."""
    backend = SequencedBackend(["Mango.", "Apple.", "Blue."])
    session = _session(trie_enabled)

    main_messages = [HELPFUL_SYS, {"role": "user", "content": "Pick a fruit name."}]
    first = await session.run_generation({"messages": main_messages}, backend)
    main_messages = main_messages + [
        {"role": "assistant", "content": first.assistant_msg["content"]},
        {"role": "user", "content": "Now pick another fruit."},
    ]
    await session.run_generation({"messages": main_messages}, backend)

    subagent_messages = [SUBAGENT_SYS, {"role": "user", "content": "Pick a color."}]
    await session.run_generation({"messages": subagent_messages}, backend)

    await session.set_reward_info({"score": 0.75})
    trajectories = await session.finalize()
    prompts = [call["prompt_ids"] for call in backend.calls]
    return prompts, trajectories


def test_trie_flag_parity_linear_conversation():
    off_prompts, off_trajs = _run(_linear_conversation(trie_enabled=False))
    on_prompts, on_trajs = _run(_linear_conversation(trie_enabled=True))

    # The backend must see identical prompts on every turn.
    assert on_prompts == off_prompts

    # And the finalized linear trajectory must match token-for-token.
    assert len(on_trajs) == len(off_trajs) == 1
    a, b = on_trajs[0], off_trajs[0]
    assert a.prompt_ids == b.prompt_ids
    assert a.response_ids == b.response_ids
    assert a.response_mask == b.response_mask
    assert a.response_logprobs == b.response_logprobs
    assert a.reward_info == b.reward_info == {"score": 1.0}


def test_trie_flag_parity_repeated_prompt_multi_trajectory():
    off_prompts, off_trajs = _run(_repeated_prompt_samples(trie_enabled=False))
    on_prompts, on_trajs = _run(_repeated_prompt_samples(trie_enabled=True))

    # The backend must see identical prompts for each repeated sample.
    assert on_prompts == off_prompts

    # Trie mode represents these as assistant siblings; legacy mode materializes
    # the previous active trajectory on each prefix mismatch. The exported token
    # buffers should still match one-for-one.
    assert len(on_trajs) == len(off_trajs) == 3
    for on_traj, off_traj in zip(on_trajs, off_trajs, strict=True):
        assert on_traj.prompt_ids == off_traj.prompt_ids
        assert on_traj.response_ids == off_traj.response_ids
        assert on_traj.response_mask == off_traj.response_mask
        assert on_traj.response_logprobs == off_traj.response_logprobs
        assert on_traj.reward_info == off_traj.reward_info == {"score": 0.5}


def test_trie_flag_parity_subagent_system_split_multi_trajectory():
    off_prompts, off_trajs = _run(_subagent_system_split(trie_enabled=False))
    on_prompts, on_trajs = _run(_subagent_system_split(trie_enabled=True))

    # Main-agent turns should reuse their branch prefix, while the sub-agent
    # request starts from a different system prompt. Both modes should send the
    # same backend prompts for the three generations.
    assert on_prompts == off_prompts

    assert len(on_trajs) == len(off_trajs) == 2
    for on_traj, off_traj in zip(on_trajs, off_trajs, strict=True):
        assert on_traj.prompt_ids == off_traj.prompt_ids
        assert on_traj.response_ids == off_traj.response_ids
        assert on_traj.response_mask == off_traj.response_mask
        assert on_traj.response_logprobs == off_traj.response_logprobs
        assert on_traj.reward_info == off_traj.reward_info == {"score": 0.75}

    main_text = "".join(chr(token_id) for token_id in on_trajs[0].response_ids)
    subagent_text = "".join(chr(token_id) for token_id in on_trajs[1].response_ids)
    assert "Mango." in main_text
    assert "Apple." in main_text
    assert 0 in on_trajs[0].response_mask
    assert subagent_text == "Blue."
    assert on_trajs[1].response_mask == [1] * len(on_trajs[1].response_ids)


def test_trie_best_of_n_fans_out_to_sibling_trajectories():
    async def scenario():
        _, trajectories = await _repeated_prompt_samples(trie_enabled=True)
        return trajectories

    trajectories = _run(scenario())
    assert len(trajectories) == 3
    contents = sorted("".join(chr(t) for t in traj.response_ids) for traj in trajectories)
    assert contents == ["ans-A", "ans-B", "ans-C"]
    assert all(traj.reward_info == {"score": 0.5} for traj in trajectories)


def test_trie_multi_turn_reattaches_and_finalizes_single_branch():
    async def scenario():
        backend = SequencedBackend(["step1", "step2", "step3"])
        session = _session(trie_enabled=True)
        msgs = [SYS, USER]
        for i in range(3):
            out = await session.run_generation({"messages": msgs}, backend)
            msgs = msgs + [
                {"role": "assistant", "content": out.assistant_msg["content"]},
                {"role": "user", "content": f"turn {i}"},
            ]
        return await session.finalize()

    trajectories = _run(scenario())
    # One continuous branch -> one terminal trajectory whose response covers all
    # three generated turns plus the interstitial context tokens.
    assert len(trajectories) == 1
    traj = trajectories[0]
    assert len(traj.response_ids) == len(traj.response_mask) == len(traj.response_logprobs)
    # mask has both generated (1) and continuation (0) tokens.
    assert set(traj.response_mask) == {0, 1}


def test_trie_abandons_pending_node_when_encode_fails():
    """An encode failure inside input prep must not leak the trie pending node
    (fails if _prepare_generation_inputs_trie doesn't abandon on error)."""

    async def scenario():
        session = _session(trie_enabled=True)

        def boom(*args, **kwargs):
            raise RuntimeError("encode failed")

        session._codec.encode_full = boom  # force prep to raise after prepare()
        raised = False
        try:
            await session.run_generation({"messages": [SYS, USER]}, SequencedBackend(["x"]))
        except RuntimeError:
            raised = True
        return raised, session._trie.num_inflight()

    raised, inflight = _run(scenario())
    assert raised
    assert inflight == 0, "pending node must be abandoned on encode failure"


def test_trie_no_duplicate_multimodal_on_full_encode_midbranch():
    """A tools change mid-branch forces a full re-encode; the committed node must
    store only this turn's delta media, not the whole history, so finalize does
    not double-count ancestor media (fails if the node stores the full lists)."""

    async def scenario():
        codec = MessageCodec(
            FakeTokenizer(), processor=FakeProcessor(), vision_info_extractor=fake_vision_info_extractor
        )
        session = GatewaySession(SessionHandle(session_id="s"), codec, trie_enabled=True)
        backend = SequencedBackend(["R0", "R1"])
        img_a = {"type": "image_url", "image_url": {"url": "http://x/a.png"}}
        img_b = {"type": "image_url", "image_url": {"url": "http://x/b.png"}}

        # turn 1: image A, no tools
        msgs1 = [SYS, {"role": "user", "content": [img_a, {"type": "text", "text": "a"}]}]
        out1 = await session.run_generation({"messages": msgs1}, backend)
        # turn 2: append assistant + a new user with image B AND change tools ->
        # use_incremental=False -> full re-encode mid-branch
        msgs2 = msgs1 + [
            {"role": "assistant", "content": out1.assistant_msg["content"]},
            {"role": "user", "content": [img_b, {"type": "text", "text": "b"}]},
        ]
        await session.run_generation(
            {"messages": msgs2, "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
            backend,
        )
        return await session.finalize()

    trajectories = _run(scenario())
    assert len(trajectories) == 1
    images = trajectories[0].multi_modal_data["images"]
    assert images == ["http://x/a.png", "http://x/b.png"], f"no duplicate media expected, got {images}"


def test_trie_abandons_pending_node_on_cancellation():
    """A cancellation during backend.generate (CancelledError is BaseException,
    not Exception) must still abandon the pending node (fails if only
    ValueError/Exception are caught)."""

    class CancellingBackend:
        async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
            raise asyncio.CancelledError()

    async def scenario():
        session = _session(trie_enabled=True)
        cancelled = False
        try:
            await session.run_generation({"messages": [SYS, USER]}, CancellingBackend())
        except asyncio.CancelledError:
            cancelled = True
        return cancelled, session._trie.num_inflight()

    cancelled, inflight = _run(scenario())
    assert cancelled
    assert inflight == 0, "pending node must be abandoned on cancellation"
