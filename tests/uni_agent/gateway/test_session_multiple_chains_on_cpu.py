import asyncio

import pytest
from fastapi import HTTPException

from tests.uni_agent.support import FakeProcessor, FakeTokenizer, SequencedBackend, fake_vision_info_extractor
from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle
from verl.workers.rollout.replica import TokenOutput


HELPFUL_SYS = {"role": "system", "content": "You are helpful."}
SUBAGENT_SYS = {"role": "system", "content": "You are a focused subagent."}


def _ids(text: str) -> list[int]:
    return [ord(char) for char in text]


def _decode_response_ids(response_ids: list[int]) -> str:
    return FakeTokenizer().decode(response_ids)


def _session(
    session_id: str,
    *,
    apply_chat_template_kwargs: dict | None = None,
    response_length: int | None = None,
    processor=None,
    vision_info_extractor=None,
    tool_parser_name: str | None = None,
) -> GatewaySession:
    return GatewaySession(
        SessionHandle(session_id=session_id),
        MessageCodec(
            FakeTokenizer(),
            processor=processor,
            vision_info_extractor=vision_info_extractor,
            tool_parser_name=tool_parser_name,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        ),
        response_length=response_length,
    )


async def _run(session: GatewaySession, backend: SequencedBackend, messages: list[dict], **payload_extra):
    return await session.run_generation({"model": "dummy-model", "messages": messages, **payload_extra}, backend)


def test_encode_incremental_uses_request_chat_template_kwargs_for_prefix_slice(monkeypatch):
    import uni_agent.gateway.session.codec as codec_mod

    class PrefixChangingTokenizer:
        @staticmethod
        def _prefix(prefix_style: str = "short") -> str:
            return "<s>" if prefix_style == "short" else "<long-system-prefix>"

        def system_prompt(self, **kwargs) -> list[int]:
            return _ids(self._prefix(**kwargs))

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
            text = self._prefix(**kwargs)
            for message in messages:
                text += f"{message['role']}:{message.get('content', '')}\n"
            if add_generation_prompt:
                text += "assistant:"
            if tokenize:
                return _ids(text)
            return text

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(tokenizer, messages, **kwargs):
        return tokenizer.apply_chat_template(messages, **kwargs)

    def initialize_system_prompt(tokenizer, **kwargs):
        return tokenizer.system_prompt(**kwargs)

    monkeypatch.setattr(codec_mod, "_apply_chat_template", apply_chat_template)
    monkeypatch.setattr(codec_mod, "initialize_system_prompt", initialize_system_prompt)

    tokenizer = PrefixChangingTokenizer()
    codec = MessageCodec(tokenizer, apply_chat_template_kwargs={"prefix_style": "short"})
    incremental_ids = codec.encode_incremental(
        [{"role": "user", "content": "delta"}],
        request_chat_template_kwargs={"prefix_style": "long"},
    )

    assert tokenizer.decode(incremental_ids) == "user:delta\nassistant:"


def test_encode_incremental_processor_uses_request_chat_template_kwargs_for_prefix_slice(monkeypatch):
    import torch

    import uni_agent.gateway.session.codec as codec_mod

    class _PrefixChangingEncoder:
        @staticmethod
        def _prefix(prefix_style: str = "short") -> str:
            return "<s>" if prefix_style == "short" else "<long-system-prefix>"

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
            text = self._prefix(**kwargs)
            for message in messages:
                text += f"{message['role']}:{message.get('content', '')}\n"
            if add_generation_prompt:
                text += "assistant:"
            return _ids(text) if tokenize else text

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(token_id) for token_id in token_ids)

    # The processor prepends a BOS token the bare tokenizer never emits, so its system-prompt
    # prefix is one token longer. Slicing the continuation with a tokenizer-derived prefix
    # would be off by one; this double makes the test fail unless encode_incremental measures
    # the strip prefix with the processor that produced the ids.
    processor_bos = 9999

    class _PrefixChangingProcessor(_PrefixChangingEncoder):
        def system_prompt(self, **kwargs) -> list[int]:
            return [processor_bos] + _ids(self._prefix(**kwargs))

        def __call__(
            self, *, text, images=None, videos=None, video_metadata=None, return_tensors=None, do_sample_frames=False
        ):
            assert len(text) == 1
            return {"input_ids": torch.tensor([[processor_bos] + _ids(text[0])], dtype=torch.long)}

    class _PlainTokenizer(_PrefixChangingEncoder):
        def system_prompt(self, **kwargs) -> list[int]:
            return _ids(self._prefix(**kwargs))

    def apply_chat_template(encoder, messages, **kwargs):
        return encoder.apply_chat_template(messages, **kwargs)

    def initialize_system_prompt(encoder, **kwargs):
        return encoder.system_prompt(**kwargs)

    monkeypatch.setattr(codec_mod, "_apply_chat_template", apply_chat_template)
    monkeypatch.setattr(codec_mod, "initialize_system_prompt", initialize_system_prompt)

    codec = MessageCodec(
        _PlainTokenizer(),
        processor=_PrefixChangingProcessor(),
        apply_chat_template_kwargs={"prefix_style": "short"},
    )
    incremental_ids = codec.encode_incremental(
        [{"role": "user", "content": "delta"}],
        request_chat_template_kwargs={"prefix_style": "long"},
    )

    assert _PlainTokenizer().decode(incremental_ids) == "user:delta\nassistant:"


def test_encode_incremental_processor_default_kwargs_uses_processor_prefix_for_slice(monkeypatch):
    import torch

    import uni_agent.gateway.session.codec as codec_mod

    class _Encoder:
        @staticmethod
        def _prefix(prefix_style: str = "short") -> str:
            return "<s>" if prefix_style == "short" else "<long-system-prefix>"

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
            text = self._prefix(**kwargs)
            for message in messages:
                text += f"{message['role']}:{message.get('content', '')}\n"
            if add_generation_prompt:
                text += "assistant:"
            return _ids(text) if tokenize else text

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(token_id) for token_id in token_ids)

    # The processor prepends a BOS token the bare tokenizer never emits, so its system-prompt
    # prefix is one token longer even under default kwargs. On the default-kwargs path
    # (request kwargs omitted or effective-equal to codec defaults) encode_incremental() must
    # still strip with the processor-derived prefix, not the cached tokenizer-derived
    # _system_prompt, or the continuation delta is off by the BOS token.
    processor_bos = 9999

    class _Processor(_Encoder):
        def system_prompt(self, **kwargs) -> list[int]:
            return [processor_bos] + _ids(self._prefix(**kwargs))

        def __call__(
            self, *, text, images=None, videos=None, video_metadata=None, return_tensors=None, do_sample_frames=False
        ):
            assert len(text) == 1
            return {"input_ids": torch.tensor([[processor_bos] + _ids(text[0])], dtype=torch.long)}

    class _PlainTokenizer(_Encoder):
        def system_prompt(self, **kwargs) -> list[int]:
            return _ids(self._prefix(**kwargs))

    def apply_chat_template(encoder, messages, **kwargs):
        return encoder.apply_chat_template(messages, **kwargs)

    def initialize_system_prompt(encoder, **kwargs):
        return encoder.system_prompt(**kwargs)

    monkeypatch.setattr(codec_mod, "_apply_chat_template", apply_chat_template)
    monkeypatch.setattr(codec_mod, "initialize_system_prompt", initialize_system_prompt)

    codec = MessageCodec(
        _PlainTokenizer(),
        processor=_Processor(),
        apply_chat_template_kwargs={"prefix_style": "short"},
    )

    # Request kwargs omitted entirely -> default path.
    omitted = codec.encode_incremental([{"role": "user", "content": "delta"}])
    assert _PlainTokenizer().decode(omitted) == "user:delta\nassistant:"

    # Request kwargs explicitly equal to codec defaults -> still the default path.
    effective_equal = codec.encode_incremental(
        [{"role": "user", "content": "delta"}],
        request_chat_template_kwargs={"prefix_style": "short"},
    )
    assert _PlainTokenizer().decode(effective_equal) == "user:delta\nassistant:"


class _LogprobBackend:
    def __init__(self, steps):
        self.steps = list(steps)

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        text, log_probs = self.steps.pop(0)
        token_ids = _ids(text)
        if log_probs == "full":
            log_probs = [-0.1] * len(token_ids)
        elif log_probs == "short":
            log_probs = [-0.1]
        return TokenOutput(token_ids=token_ids, log_probs=log_probs, stop_reason="completed")


class _DelayedBackend:
    def __init__(self, text: str):
        self.text = text
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "image_data": image_data,
                "video_data": video_data,
            }
        )
        self.entered.set()
        await self.release.wait()
        token_ids = _ids(self.text)
        return TokenOutput(token_ids=token_ids, log_probs=[-0.1] * len(token_ids), stop_reason="completed")


def _image_message(url: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text},
        ],
    }


def _video_message(url: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": url}},
            {"type": "text", "text": text},
        ],
    }


async def _codec_compatible_video_extractor(messages, image_patch_size, config=None):
    images, videos = await fake_vision_info_extractor(messages, image_patch_size=image_patch_size, config=config)
    if videos is not None:
        videos = [(video, {"url": video}) for video in videos]
    return images, videos


def _assert_active_chain_hashes_match_history(session: GatewaySession) -> None:
    state = session.snapshot_state()
    for chain in session.active_chains:
        assert len(chain.message_prefix_hashes) == len(chain.message_history)
        assert chain.message_prefix_hashes == session._compute_message_prefix_hashes(chain.message_history)
        assert state["active_chain_tip_hashes"][chain.chain_id] == chain.message_prefix_hashes[-1]


@pytest.mark.asyncio
async def test_multiple_chains_linear_conversation_stays_single_chain():
    session = _session("linear")
    backend = SequencedBackend(["FIRST", "SECOND"])
    first_messages = [{"role": "user", "content": "first turn"}]
    second_messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "follow up"},
    ]

    await _run(session, backend, first_messages, temperature=0.2)
    await _run(session, backend, second_messages, temperature=0.3)
    await session.set_reward_info({"label": "linear"})
    chain_trajectories = await session.finalize()

    assert len(chain_trajectories) == 1
    assert 0 in chain_trajectories[0].response_mask
    assert chain_trajectories[0].response_mask[-len("SECOND") :] == [1] * len("SECOND")
    assert chain_trajectories[0].reward_info == {"label": "linear"}


@pytest.mark.asyncio
async def test_multiple_chains_subagent_system_split_returns_to_main_chain():
    session = _session("subagent-return")
    backend = SequencedBackend(["Mango", "Blue", "Apple"])
    main_first = [HELPFUL_SYS, {"role": "user", "content": "name a fruit"}]
    subagent = [SUBAGENT_SYS, {"role": "user", "content": "name a color"}]
    main_continuation = [
        HELPFUL_SYS,
        {"role": "user", "content": "name a fruit"},
        {"role": "assistant", "content": "Mango"},
        {"role": "user", "content": "name another fruit"},
    ]

    await _run(session, backend, main_first)
    await _run(session, backend, subagent)
    await _run(session, backend, main_continuation)

    # active_chains insertion order is main (id=1, created first) then subagent (id=2),
    # but main is the *later-updated* chain (it committed last via the continuation).
    assert [chain.chain_id for chain in session.active_chains] == [1, 2]
    active_by_id = {chain.chain_id: chain for chain in session.active_chains}
    main_updated_seq = active_by_id[1].updated_seq
    subagent_updated_seq = active_by_id[2].updated_seq
    assert main_updated_seq > subagent_updated_seq
    order_seq_before = session._order_seq

    trajectories = await session.finalize()

    # finalize must not advance _order_seq nor rewrite any chain's updated_seq; active
    # chains keep order_seq == updated_seq, so the return order is decided by order_seq,
    # not by active_chains insertion order.
    assert session._order_seq == order_seq_before
    materialized_by_id = {chain.chain_id: chain for chain in session.materialized_chains}
    assert materialized_by_id[1].updated_seq == main_updated_seq
    assert materialized_by_id[2].updated_seq == subagent_updated_seq
    assert materialized_by_id[1].order_seq == main_updated_seq
    assert materialized_by_id[2].order_seq == subagent_updated_seq
    # materialized_chains keeps active-insertion order [1, 2]; finalize derives the returned
    # order by sorting on order_seq, yielding [2, 1] (subagent then main) — the reverse of
    # insertion order. The decoded assertions below confirm the returned trajectories follow it.
    assert [chain.chain_id for chain in session.materialized_chains] == [1, 2]
    assert [
        chain.chain_id for chain in sorted(session.materialized_chains, key=lambda chain: chain.order_seq)
    ] == [2, 1]

    assert len(trajectories) == 2
    decoded = [_decode_response_ids(t.response_ids) for t in trajectories]
    assert "Blue" in decoded[0]
    assert "Mango" in decoded[1]
    assert "Apple" in decoded[1]
    assert "Blue" not in decoded[1]
    assert 0 in trajectories[1].response_mask
    assert trajectories[1].response_mask[-len("Apple") :] == [1] * len("Apple")


@pytest.mark.asyncio
async def test_multiple_chains_context_compaction_starts_new_chain():
    session = _session("compaction")
    backend = SequencedBackend(["DETAILED", "AFTER_SUMMARY"])

    await _run(session, backend, [HELPFUL_SYS, {"role": "user", "content": "produce a detailed answer"}])
    await _run(
        session,
        backend,
        [
            {"role": "system", "content": "Summary so far: the detailed answer was compacted."},
            {"role": "user", "content": "continue from the summary"},
        ],
    )
    trajectories = await session.finalize()

    decoded = [_decode_response_ids(t.response_ids) for t in trajectories]
    assert len(trajectories) == 2
    assert decoded == ["DETAILED", "AFTER_SUMMARY"]
    assert all(t.response_mask == [1] * len(t.response_ids) for t in trajectories)


@pytest.mark.asyncio
async def test_multiple_chains_repeated_same_prompt_creates_siblings_and_continues_latest():
    session = _session("siblings")
    backend = SequencedBackend(["SAME", "SAME", "SAME", "NEXT"])
    prompt = [{"role": "user", "content": "try the same prompt"}]

    await _run(session, backend, prompt)
    await _run(session, backend, prompt)
    await _run(session, backend, prompt)
    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "try the same prompt"},
            {"role": "assistant", "content": "SAME"},
            {"role": "user", "content": "continue the latest sibling"},
        ],
    )
    trajectories = await session.finalize()

    decoded = [_decode_response_ids(t.response_ids) for t in trajectories]
    assert len(trajectories) == 3
    assert decoded.count("SAME") == 2
    assert decoded[-1].startswith("SAME")
    assert decoded[-1].endswith("NEXT")
    assert trajectories[-1].response_mask[-len("NEXT") :] == [1] * len("NEXT")
    assert 0 in trajectories[-1].response_mask


@pytest.mark.asyncio
async def test_multiple_chains_tools_and_effective_chat_template_kwargs_gate_chain_reuse():
    search_tool = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    lookup_tool = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

    tools_session = _session("tools-gate")
    tools_backend = SequencedBackend(["SEARCH", "LOOKUP"])
    await _run(tools_session, tools_backend, [{"role": "user", "content": "use a tool"}], tools=search_tool)
    await _run(
        tools_session,
        tools_backend,
        [
            {"role": "user", "content": "use a tool"},
            {"role": "assistant", "content": "SEARCH"},
            {"role": "user", "content": "continue with a renamed tool"},
        ],
        tools=lookup_tool,
    )
    tool_trajectories = await tools_session.finalize()
    assert [_decode_response_ids(t.response_ids) for t in tool_trajectories] == ["SEARCH", "LOOKUP"]

    kwargs_session = _session("kwargs-gate", apply_chat_template_kwargs={"enable_thinking": False})
    kwargs_backend = SequencedBackend(["BASE", "CONT", "SPLIT"])
    await _run(kwargs_session, kwargs_backend, [{"role": "user", "content": "template default"}])
    await _run(
        kwargs_session,
        kwargs_backend,
        [
            {"role": "user", "content": "template default"},
            {"role": "assistant", "content": "BASE"},
            {"role": "user", "content": "explicit same default"},
        ],
        chat_template_kwargs={"enable_thinking": False},
    )
    await _run(
        kwargs_session,
        kwargs_backend,
        [
            {"role": "user", "content": "template default"},
            {"role": "assistant", "content": "BASE"},
            {"role": "user", "content": "explicit same default"},
            {"role": "assistant", "content": "CONT"},
            {"role": "user", "content": "change effective kwargs"},
        ],
        chat_template_kwargs={"enable_thinking": True},
    )
    kwargs_trajectories = await kwargs_session.finalize()
    decoded_kwargs = [_decode_response_ids(t.response_ids) for t in kwargs_trajectories]
    assert len(kwargs_trajectories) == 2
    assert decoded_kwargs[0].startswith("BASE")
    assert decoded_kwargs[0].endswith("CONT")
    assert decoded_kwargs[1] == "SPLIT"
    assert 0 in kwargs_trajectories[0].response_mask


@pytest.mark.asyncio
async def test_multiple_chains_finalize_clears_active_chains():
    session = _session("finalize-clears")
    backend = SequencedBackend(["ONE", "TWO"])

    await _run(session, backend, [{"role": "user", "content": "first branch"}])
    await _run(session, backend, [{"role": "user", "content": "second branch"}])
    trajectories = await session.finalize()
    state = session.snapshot_state()

    assert len(trajectories) == 2
    assert state["phase"] == "FINALIZED"
    assert state["num_active_chains"] == 0
    assert state["active_chain_ids"] == []
    assert state["has_active_trajectory"] is False
    assert state["num_trajectories"] == 2


@pytest.mark.asyncio
async def test_multiple_chains_committed_assistant_tip_hash_round_trips_through_echoed_request():
    session = _session("hash-round-trip")
    backend = SequencedBackend(["FIRST", "SECOND"])

    await _run(session, backend, [{"role": "user", "content": "first turn"}])
    state_after_first = session.snapshot_state()
    active_chain_ids = state_after_first["active_chain_ids"]
    tip_hashes = state_after_first["active_chain_tip_hashes"]

    assert len(active_chain_ids) == 1
    assert len(tip_hashes) == 1
    assert all(isinstance(tip_hash, str) and len(tip_hash) == 64 for tip_hash in tip_hashes.values())

    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "second turn"},
        ],
    )
    state_after_second = session.snapshot_state()
    trajectories = await session.finalize()

    assert state_after_second["active_chain_ids"] == active_chain_ids
    assert state_after_second["active_chain_tip_hashes"] != tip_hashes
    assert len(trajectories) == 1
    assert trajectories[0].response_ids[: len("FIRST")] == _ids("FIRST")
    assert trajectories[0].response_ids[-len("SECOND") :] == _ids("SECOND")
    assert 0 in trajectories[0].response_mask


@pytest.mark.asyncio
async def test_multiple_chains_backend_failure_does_not_mutate_selected_chain():
    session = _session("backend-failure")
    backend = SequencedBackend(["FIRST", RuntimeError("boom")])
    first_messages = [{"role": "user", "content": "first turn"}]
    second_messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "follow up"},
    ]

    await _run(session, backend, first_messages)
    with pytest.raises(Exception, match="boom"):
        await _run(session, backend, second_messages)
    trajectories = await session.finalize()

    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids) == "FIRST"
    assert trajectories[0].response_mask == [1] * len("FIRST")


@pytest.mark.asyncio
async def test_multiple_chains_new_chain_backend_failure_does_not_leave_partial_chain():
    session = _session("new-chain-backend-failure")
    backend = SequencedBackend(["MAIN", RuntimeError("boom")])
    main_messages = [HELPFUL_SYS, {"role": "user", "content": "main request"}]
    split_messages = [SUBAGENT_SYS, {"role": "user", "content": "independent subtask"}]

    await _run(session, backend, main_messages)
    before_failure = session.snapshot_state()

    with pytest.raises(HTTPException, match="RuntimeError: boom"):
        await _run(session, backend, split_messages)
    after_failure = session.snapshot_state()
    trajectories = await session.finalize()

    assert before_failure["active_chain_ids"] == [1]
    assert after_failure["active_chain_ids"] == before_failure["active_chain_ids"]
    assert after_failure["active_chain_tip_hashes"] == before_failure["active_chain_tip_hashes"]
    assert after_failure["num_active_chains"] == before_failure["num_active_chains"]
    assert after_failure["num_trajectories"] == before_failure["num_trajectories"] == 0
    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids) == "MAIN"
    assert trajectories[0].response_mask == [1] * len("MAIN")


@pytest.mark.asyncio
async def test_multiple_chains_length_exhaustion_closes_selected_chain_and_orders_it_last():
    session = _session("length-close", response_length=len("MAIN1") + 1)
    backend = SequencedBackend(["MAIN1", "SUB"])
    main_first = [HELPFUL_SYS, {"role": "user", "content": "main"}]
    subagent = [SUBAGENT_SYS, {"role": "user", "content": "sub"}]
    main_too_long = [
        HELPFUL_SYS,
        {"role": "user", "content": "main"},
        {"role": "assistant", "content": "MAIN1"},
        {"role": "user", "content": "too long"},
    ]

    await _run(session, backend, main_first)
    await _run(session, backend, subagent)
    outcome = await _run(session, backend, main_too_long)
    trajectories = await session.finalize()

    assert outcome.finish_reason == "length"
    assert backend.steps == []
    assert len(trajectories) == 2
    assert _decode_response_ids(trajectories[0].response_ids) == "SUB"
    assert _decode_response_ids(trajectories[1].response_ids) == "MAIN1"
    assert trajectories[1].extra_fields["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_multiple_chains_length_exhaustion_surviving_chain_still_continues():
    # Generous budget so only the deliberately oversized main continuation length-closes; the
    # subagent chain stays well under budget and must remain active and continuable afterwards.
    session = _session("length-close-survivor", response_length=200)
    backend = SequencedBackend(["MAIN1", "SUB", "SUB2"])
    main_first = [HELPFUL_SYS, {"role": "user", "content": "main"}]
    subagent = [SUBAGENT_SYS, {"role": "user", "content": "sub"}]
    main_too_long = [
        HELPFUL_SYS,
        {"role": "user", "content": "main"},
        {"role": "assistant", "content": "MAIN1"},
        {"role": "user", "content": "L" * 400},
    ]
    subagent_continue = [
        SUBAGENT_SYS,
        {"role": "user", "content": "sub"},
        {"role": "assistant", "content": "SUB"},
        {"role": "user", "content": "go"},
    ]

    await _run(session, backend, main_first)
    await _run(session, backend, subagent)
    main_outcome = await _run(session, backend, main_too_long)

    # The main chain length-closes, but the subagent chain survives as the only active chain.
    assert main_outcome.finish_reason == "length"
    snapshot = session.snapshot_state()
    assert snapshot["num_active_chains"] == 1
    assert snapshot["active_chain_ids"] == [2]

    # The surviving subagent chain still accepts a further continuation and commits it.
    sub_continue_outcome = await _run(session, backend, subagent_continue)
    assert sub_continue_outcome.finish_reason == "stop"
    assert backend.steps == []
    assert session.snapshot_state()["active_chain_ids"] == [2]

    trajectories = await session.finalize()
    assert len(trajectories) == 2
    # The subagent continuation is the last visible interaction, so order_seq puts it last.
    sub_decoded = _decode_response_ids(trajectories[-1].response_ids)
    assert sub_decoded.startswith("SUB")
    assert sub_decoded.endswith("SUB2")
    # The length-closed main chain is retained, ordered before the later subagent interaction.
    assert trajectories[0].extra_fields["finish_reason"] == "length"
    assert _decode_response_ids(trajectories[0].response_ids) == "MAIN1"


@pytest.mark.asyncio
async def test_multiple_chains_new_chain_over_budget_clamps_not_early_returns():
    session = _session("new-chain-clamp", response_length=0)
    backend = SequencedBackend(["NORMAL"])

    outcome = await _run(
        session,
        backend,
        [{"role": "user", "content": "new chain should still call backend"}],
        max_tokens=8,
    )
    # The over-budget new chain still went through the backend (clamped), so it is an active
    # chain, not an early-closed one. Assert before finalize, which clears active_chains.
    assert session.snapshot_state()["num_active_chains"] == 1

    trajectories = await session.finalize()

    assert len(backend.calls) == 1
    assert backend.calls[-1]["sampling_params"]["max_tokens"] == 0
    assert outcome.finish_reason == "stop"
    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids) == "NORMAL"
    assert trajectories[0].response_mask == [1] * len("NORMAL")


@pytest.mark.asyncio
async def test_multiple_chains_existing_chain_over_budget_clamps_to_zero():
    session = _session("existing-chain-clamp", response_length=0)
    backend = SequencedBackend(["NORMAL", "SECOND"])
    messages = [{"role": "user", "content": "request that already exceeded budget"}]

    await _run(session, backend, messages, max_tokens=8)
    await _run(session, backend, list(session.active_chains[0].message_history), max_tokens=8)

    assert len(backend.calls) == 2
    assert backend.calls[-1]["sampling_params"]["max_tokens"] == 0


@pytest.mark.asyncio
async def test_multiple_chains_multimodal_media_stays_chain_local():
    session = _session(
        "mm-chain-local",
        processor=FakeProcessor(),
        vision_info_extractor=fake_vision_info_extractor,
    )
    backend = SequencedBackend(["MAIN1", "SUB", "MAIN2"])
    main_first = [HELPFUL_SYS, _image_message("image://main-a.png", "describe main")]
    subagent = [SUBAGENT_SYS, _image_message("image://sub-b.png", "describe sub")]
    main_continuation = [
        HELPFUL_SYS,
        _image_message("image://main-a.png", "describe main"),
        {"role": "assistant", "content": "MAIN1"},
        {"role": "user", "content": "continue main"},
    ]

    await _run(session, backend, main_first)
    await _run(session, backend, subagent)
    await _run(session, backend, main_continuation)
    trajectories = await session.finalize()

    assert [call["image_data"] for call in backend.calls] == [
        ["image://main-a.png"],
        ["image://sub-b.png"],
        ["image://main-a.png"],
    ]
    assert len(trajectories) == 2
    decoded = [_decode_response_ids(t.response_ids) for t in trajectories]
    assert decoded[0] == "SUB"
    assert decoded[1].startswith("MAIN1")
    assert decoded[1].endswith("MAIN2")
    assert trajectories[0].multi_modal_data == {"images": ["image://sub-b.png"]}
    assert trajectories[1].multi_modal_data == {"images": ["image://main-a.png"]}


@pytest.mark.asyncio
async def test_multiple_chains_video_media_stays_chain_local():
    session = _session(
        "video-chain-local",
        processor=FakeProcessor(),
        vision_info_extractor=_codec_compatible_video_extractor,
    )
    backend = SequencedBackend(["MAIN1", "SUB", "MAIN2"])
    main_video = ("video://main-a.mp4", {"url": "video://main-a.mp4"})
    subagent_video = ("video://sub-b.mp4", {"url": "video://sub-b.mp4"})
    main_first = [HELPFUL_SYS, _video_message("video://main-a.mp4", "describe main video")]
    subagent = [SUBAGENT_SYS, _video_message("video://sub-b.mp4", "describe sub video")]
    main_continuation = [
        HELPFUL_SYS,
        _video_message("video://main-a.mp4", "describe main video"),
        {"role": "assistant", "content": "MAIN1"},
        {"role": "user", "content": "continue main"},
    ]

    await _run(session, backend, main_first)
    await _run(session, backend, subagent)
    await _run(session, backend, main_continuation)
    trajectories = await session.finalize()

    assert [call["video_data"] for call in backend.calls] == [
        [main_video],
        [subagent_video],
        [main_video],
    ]
    assert len(trajectories) == 2
    decoded = [_decode_response_ids(t.response_ids) for t in trajectories]
    assert decoded[0] == "SUB"
    assert decoded[1].startswith("MAIN1")
    assert decoded[1].endswith("MAIN2")
    assert trajectories[0].multi_modal_data == {"videos": [subagent_video]}
    assert trajectories[1].multi_modal_data == {"videos": [main_video]}


@pytest.mark.asyncio
async def test_multiple_chains_length_exhaustion_with_incremental_media_does_not_record_unsent_media():
    session = _session(
        "length-incremental-media",
        response_length=len("FIRST") + 1,
        processor=FakeProcessor(),
        vision_info_extractor=fake_vision_info_extractor,
    )
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    first_messages = [_image_message("image://sent-a.png", "describe first")]
    exhausted_messages = [
        _image_message("image://sent-a.png", "describe first"),
        {"role": "assistant", "content": "FIRST"},
        _image_message("image://unsent-b.png", "new media that exhausts length"),
    ]

    await _run(session, backend, first_messages)
    outcome = await _run(session, backend, exhausted_messages)
    trajectories = await session.finalize()

    assert outcome.finish_reason == "length"
    assert len(backend.calls) == 1
    assert backend.steps == ["SHOULD_NOT_RUN"]
    assert backend.calls[0]["image_data"] == ["image://sent-a.png"]
    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids) == "FIRST"
    assert trajectories[0].multi_modal_data == {"images": ["image://sent-a.png"]}
    assert trajectories[0].extra_fields["finish_reason"] == "length"
    assert trajectories[0].num_turns == 3


@pytest.mark.asyncio
async def test_multiple_chains_length_exhaustion_with_incremental_video_does_not_record_unsent_video():
    session = _session(
        "length-incremental-video",
        response_length=len("FIRST") + 1,
        processor=FakeProcessor(),
        vision_info_extractor=_codec_compatible_video_extractor,
    )
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    sent_video = ("video://sent-a.mp4", {"url": "video://sent-a.mp4"})
    unsent_video = ("video://unsent-b.mp4", {"url": "video://unsent-b.mp4"})
    first_messages = [_video_message("video://sent-a.mp4", "describe first video")]
    exhausted_messages = [
        _video_message("video://sent-a.mp4", "describe first video"),
        {"role": "assistant", "content": "FIRST"},
        _video_message("video://unsent-b.mp4", "new video that exhausts length"),
    ]

    await _run(session, backend, first_messages)
    outcome = await _run(session, backend, exhausted_messages)
    trajectories = await session.finalize()

    assert outcome.finish_reason == "length"
    assert len(backend.calls) == 1
    assert backend.steps == ["SHOULD_NOT_RUN"]
    assert backend.calls[0]["video_data"] == [sent_video]
    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids) == "FIRST"
    assert trajectories[0].multi_modal_data == {"videos": [sent_video]}
    assert unsent_video not in trajectories[0].multi_modal_data["videos"]
    assert trajectories[0].extra_fields["finish_reason"] == "length"
    assert trajectories[0].num_turns == 3


@pytest.mark.asyncio
async def test_multiple_chains_abort_clears_length_materialized_chains_from_snapshot():
    session = _session("abort-clears-materialized", response_length=len("FIRST") + 1)
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    first_messages = [{"role": "user", "content": "first turn"}]
    exhausted_messages = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "this continuation exhausts the length budget"},
    ]

    await _run(session, backend, first_messages)
    outcome = await _run(session, backend, exhausted_messages)
    before_abort = session.snapshot_state()

    assert outcome.finish_reason == "length"
    assert before_abort["num_trajectories"] == 1
    assert before_abort["active_chain_ids"] == []

    await session.abort()
    after_abort = session.snapshot_state()

    assert after_abort["phase"] == "ABORTED"
    assert after_abort["num_trajectories"] == 0
    assert after_abort["num_active_chains"] == 0
    assert after_abort["active_chain_ids"] == []
    assert after_abort["active_chain_tip_hashes"] == {}
    with pytest.raises(RuntimeError, match="aborted"):
        await session.finalize()


@pytest.mark.asyncio
async def test_multiple_chains_committed_media_is_not_mutated_by_external_lists():
    session = _session(
        "mutable-media",
        processor=FakeProcessor(),
        vision_info_extractor=fake_vision_info_extractor,
    )
    backend = SequencedBackend(["FIRST"])
    initial_message = _image_message("image://stable-a.png", "describe stable")

    await _run(session, backend, [initial_message])
    backend.calls[0]["image_data"].append("image://backend-mutated.png")
    initial_message["content"][0]["image_url"]["url"] = "image://message-mutated.png"
    trajectories = await session.finalize()

    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data == {"images": ["image://stable-a.png"]}


@pytest.mark.asyncio
async def test_multiple_chains_media_objects_are_list_copied_without_deepcopy():
    class NonDeepCopyableImage:
        def __deepcopy__(self, memo):
            raise RuntimeError("media object should not be deep-copied")

    image = NonDeepCopyableImage()

    async def vision_info_extractor(messages, image_patch_size, config=None):
        assert image_patch_size == 16
        return [image], None

    session = _session(
        "non-deepcopyable-media",
        processor=FakeProcessor(),
        vision_info_extractor=vision_info_extractor,
    )
    backend = SequencedBackend(["FIRST"])

    outcome = await _run(session, backend, [_image_message("image://raw.png", "describe raw")])
    trajectories = await session.finalize()

    assert outcome.finish_reason == "stop"
    assert len(backend.calls) == 1
    assert backend.calls[0]["image_data"][0] is image
    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data["images"][0] is image
    assert backend.calls[0]["image_data"] is not trajectories[0].multi_modal_data["images"]


@pytest.mark.asyncio
async def test_multiple_chains_late_commit_after_finalize_is_rejected_without_mutating_session():
    session = _session("late-finalize")
    await _run(session, SequencedBackend(["FIRST"]), [{"role": "user", "content": "first turn"}])
    before_late = session.snapshot_state()
    delayed_backend = _DelayedBackend("SECOND")
    late_task = asyncio.create_task(
        _run(
            session,
            delayed_backend,
            [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "FIRST"},
                {"role": "user", "content": "follow up"},
            ],
        )
    )

    await asyncio.wait_for(delayed_backend.entered.wait(), timeout=5)
    trajectories = await session.finalize()
    delayed_backend.release.set()
    with pytest.raises(HTTPException) as exc_info:
        await late_task

    assert exc_info.value.status_code == 409
    assert [_decode_response_ids(t.response_ids) for t in trajectories] == ["FIRST"]
    state = session.snapshot_state()
    assert state["phase"] == "FINALIZED"
    assert state["num_active_chains"] == 0
    assert delayed_backend.calls[0]["request_id"] == "late-finalize"
    assert before_late["active_chain_ids"] == [1]


@pytest.mark.asyncio
async def test_multiple_chains_late_commit_after_abort_is_rejected_without_advancing_chain():
    session = _session("late-abort")
    await _run(session, SequencedBackend(["FIRST"]), [{"role": "user", "content": "first turn"}])
    before_late = session.snapshot_state()
    delayed_backend = _DelayedBackend("SECOND")
    late_task = asyncio.create_task(
        _run(
            session,
            delayed_backend,
            [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "FIRST"},
                {"role": "user", "content": "follow up"},
            ],
        )
    )

    await asyncio.wait_for(delayed_backend.entered.wait(), timeout=5)
    await session.abort()
    delayed_backend.release.set()
    with pytest.raises(HTTPException) as exc_info:
        await late_task

    assert exc_info.value.status_code == 409
    after_late = session.snapshot_state()
    assert after_late["phase"] == "ABORTED"
    assert before_late["active_chain_ids"] == [1]
    assert after_late["active_chain_ids"] == []
    assert after_late["active_chain_tip_hashes"] == {}
    assert after_late["has_active_trajectory"] is False
    with pytest.raises(RuntimeError, match="aborted"):
        await session.finalize()


@pytest.mark.asyncio
async def test_multiple_chains_prefix_content_change_does_not_reuse_chain_and_hashes_match_history():
    session = _session("hash-prefix-content")
    backend = SequencedBackend(["FIRST", "SECOND"])

    await _run(session, backend, [{"role": "user", "content": "same length a"}])
    _assert_active_chain_hashes_match_history(session)
    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "same length b"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "follow up should split"},
        ],
    )
    _assert_active_chain_hashes_match_history(session)
    trajectories = await session.finalize()

    assert len(trajectories) == 2
    assert [_decode_response_ids(t.response_ids) for t in trajectories] == ["FIRST", "SECOND"]
    assert all(t.response_mask == [1] * len(t.response_ids) for t in trajectories)


@pytest.mark.asyncio
async def test_multiple_chains_hash_prefixes_extend_on_commit():
    session = _session("hash-extend")
    backend = SequencedBackend(["FIRST", "SECOND"])

    await _run(session, backend, [{"role": "user", "content": "turn one"}])
    chain = session.active_chains[0]
    # turn 1 commit -> history is [user, assistant FIRST]
    assert len(chain.message_prefix_hashes) == len(chain.message_history) == 2
    hashes_after_first = list(chain.message_prefix_hashes)

    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "turn two"},
        ],
    )
    chain = session.active_chains[0]
    # continuation commit -> history is [user, assistant FIRST, user, assistant SECOND]
    assert len(chain.message_prefix_hashes) == len(chain.message_history) == 4
    # Continuation must preserve the earlier prefix-hash entries byte-for-byte and only
    # append new hashes for the incremental context + assistant message.
    assert chain.message_prefix_hashes[: len(hashes_after_first)] == hashes_after_first


def test_compute_message_prefix_hashes_canonicalizes_json_tool_call_arguments():
    session = _session("hash-tool-arguments")

    def assistant_tool_call(arguments) -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search", "arguments": arguments},
                }
            ],
        }

    canonical_a = session._compute_message_prefix_hashes([assistant_tool_call('{"query":"weather","limit":2}')])
    canonical_b = session._compute_message_prefix_hashes([assistant_tool_call('{"limit":2,"query":"weather"}')])
    canonical_c = session._compute_message_prefix_hashes([assistant_tool_call({"limit": 2, "query": "weather"})])
    raw_a = session._compute_message_prefix_hashes([assistant_tool_call('{"query":"weather","limit":2')])
    raw_b = session._compute_message_prefix_hashes([assistant_tool_call('{"limit":2,"query":"weather"')])

    assert canonical_a == canonical_b
    assert canonical_a == canonical_c
    assert raw_a != raw_b


@pytest.mark.asyncio
async def test_multiple_chains_tool_call_assistant_echo_hits_same_chain():
    session = _session("tool-call-echo", tool_parser_name="hermes")
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    tool_call_text = '<tool_call>\n{"name": "search", "arguments": {"query": "weather"}}\n</tool_call>'
    backend = SequencedBackend([tool_call_text, "FINAL"])

    first = await _run(
        session,
        backend,
        [{"role": "user", "content": "what is the weather?"}],
        tools=tools,
    )
    first_chain_ids = session.snapshot_state()["active_chain_ids"]
    assert first.finish_reason == "tool_calls"
    assert first.assistant_msg["tool_calls"][0]["function"]["name"] == "search"

    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "what is the weather?"},
            {"role": "assistant", "content": None, "tool_calls": first.assistant_msg["tool_calls"]},
            {
                "role": "tool",
                "tool_call_id": first.assistant_msg["tool_calls"][0]["id"],
                "content": "sunny and warm",
            },
        ],
        tools=tools,
    )
    _assert_active_chain_hashes_match_history(session)
    trajectories = await session.finalize()

    assert session.snapshot_state()["active_chain_ids"] == []
    assert len(trajectories) == 1
    assert first_chain_ids == [1]
    decoded = _decode_response_ids(trajectories[0].response_ids)
    assert decoded.startswith(tool_call_text)
    assert decoded.endswith("FINAL")
    assert 0 in trajectories[0].response_mask


@pytest.mark.asyncio
async def test_multiple_chains_tool_call_id_rewrite_starts_new_chain():
    session = _session("tool-call-id-rewrite", tool_parser_name="hermes")
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    tool_call_text = '<tool_call>\n{"name": "search", "arguments": {"query": "weather"}}\n</tool_call>'
    backend = SequencedBackend([tool_call_text, "FINAL"])

    first = await _run(
        session,
        backend,
        [{"role": "user", "content": "what is the weather?"}],
        tools=tools,
    )
    assert session.snapshot_state()["active_chain_ids"] == [1]
    committed_id = first.assistant_msg["tool_calls"][0]["id"]

    # Echo the committed assistant tool call but regenerate its id (and the matching tool
    # message tool_call_id). The committed prefix carries the original id, so the canonical
    # prefix hash no longer matches chain 1; selection must split into a new sibling chain
    # instead of reusing the old token buffer. Only the committed-prefix id is rewritten here,
    # which is the case Hash 定义第 8 条 requires to split.
    assert committed_id != "call_rewritten"
    rewritten_tool_calls = [{**first.assistant_msg["tool_calls"][0], "id": "call_rewritten"}]
    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "what is the weather?"},
            {"role": "assistant", "content": None, "tool_calls": rewritten_tool_calls},
            {"role": "tool", "tool_call_id": "call_rewritten", "content": "sunny and warm"},
        ],
        tools=tools,
    )

    # Split, not continuation: two sibling chains, two trajectories.
    assert session.snapshot_state()["active_chain_ids"] == [1, 2]
    trajectories = await session.finalize()
    assert len(trajectories) == 2
    decoded = [_decode_response_ids(trajectory.response_ids) for trajectory in trajectories]
    assert decoded == [tool_call_text, "FINAL"]
    assert trajectories[1].response_mask == [1] * len("FINAL")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("steps", "expected_logprobs"),
    [
        ([("FIRST", "full"), ("SECOND", "full")], "aligned"),
        ([("FIRST", "full"), ("SECOND", None)], None),
        ([("FIRST", None), ("SECOND", "full")], None),
        ([("FIRST", "short"), ("SECOND", "full")], None),
    ],
)
async def test_multiple_chains_response_logprobs_stay_aligned_or_none(steps, expected_logprobs):
    session = _session(f"logprobs-{expected_logprobs}")
    backend = _LogprobBackend(steps)

    await _run(session, backend, [{"role": "user", "content": "first turn"}])
    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "follow up"},
        ],
    )
    [trajectory] = await session.finalize()

    if expected_logprobs == "aligned":
        assert trajectory.response_logprobs is not None
        assert len(trajectory.response_logprobs) == len(trajectory.response_ids)
        assert 0.0 in trajectory.response_logprobs
    else:
        assert trajectory.response_logprobs is None
