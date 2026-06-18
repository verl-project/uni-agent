"""CPU integration tests for the trie-backed gateway session (issue #51, M1).

Drives a real ``GatewaySession`` (real ``MessageCodec`` + ``FakeTokenizer`` +
``SequencedBackend``) end-to-end through ``run_generation``/``complete``/
``finalize`` to check:

- **flag parity**: a linear multi-turn conversation produces identical backend
  prompts and identical finalized trajectories whether the trie is on or off
  (the M1 compatibility gate);
- **best-of-N**: repeated requests with the same messages fan out into N sibling
  trajectories under the trie;
- **multi-turn reuse**: a tool/assistant continuation reattaches to its branch.

These need the codec's runtime deps (verl tokenizer/template utils), so they
live alongside the other ``*_on_cpu`` gateway tests rather than in the
dependency-light pure-trie unit tests.
"""

from __future__ import annotations

import asyncio

from tests.uni_agent.support import FakeTokenizer, SequencedBackend
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


def test_trie_best_of_n_fans_out_to_sibling_trajectories():
    async def scenario():
        backend = SequencedBackend(["ans-A", "ans-B", "ans-C"])
        session = _session(trie_enabled=True)
        for _ in range(3):
            await session.run_generation({"messages": [SYS, USER]}, backend)
        await session.set_reward_info({"score": 0.5})
        return await session.finalize()

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
