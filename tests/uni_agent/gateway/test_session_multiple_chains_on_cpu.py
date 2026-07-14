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
        ),
        response_length=response_length,
    )


@pytest.mark.parametrize("response_length", [0, -1])
def test_gateway_session_rejects_non_positive_response_length(response_length):
    with pytest.raises(ValueError, match="response_length must be positive"):
        _session("invalid-response-length", response_length=response_length)


async def _run(session: GatewaySession, backend: SequencedBackend, messages: list[dict], **payload_extra):
    return await session.run_generation({"model": "dummy-model", "messages": messages, **payload_extra}, backend)


def test_encode_incremental_processor_uses_processor_prefix_for_slice(monkeypatch):
    import torch

    import uni_agent.gateway.session.codec as codec_mod

    class _Encoder:
        @staticmethod
        def _prefix(prefix_style="short"):
            return "<s>" if prefix_style == "short" else "<long-system-prefix>"

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **kwargs):
            text = self._prefix(**kwargs)
            text += "".join(f"{message['role']}:{message.get('content', '')}\n" for message in messages)
            if add_generation_prompt:
                text += "assistant:"
            return _ids(text) if tokenize else text

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(token_id) for token_id in token_ids)

    processor_bos = 9999

    class _Processor(_Encoder):
        def system_prompt(self, **kwargs):
            return [processor_bos] + _ids(self._prefix(**kwargs))

        def __call__(
            self, *, text, images=None, videos=None, video_metadata=None, return_tensors=None, do_sample_frames=False
        ):
            return {"input_ids": torch.tensor([[processor_bos] + _ids(text[0])], dtype=torch.long)}

    class _Tokenizer(_Encoder):
        def system_prompt(self, **kwargs):
            return _ids(self._prefix(**kwargs))

    monkeypatch.setattr(
        codec_mod,
        "_apply_chat_template",
        lambda encoder, messages, **kwargs: encoder.apply_chat_template(messages, **kwargs),
    )
    monkeypatch.setattr(
        codec_mod,
        "initialize_system_prompt",
        lambda encoder, **kwargs: encoder.system_prompt(**kwargs),
    )

    codec = MessageCodec(
        _Tokenizer(),
        processor=_Processor(),
        apply_chat_template_kwargs={"prefix_style": "long"},
    )

    incremental_ids = codec.encode_incremental([{"role": "user", "content": "delta"}])

    assert _Tokenizer().decode(incremental_ids) == "user:delta\nassistant:"


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


class _ControlledParallelBackend:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []
        self._call_added = asyncio.Event()

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        step = self.steps.pop(0)
        call = {
            "request_id": request_id,
            "prompt_ids": list(prompt_ids),
            "sampling_params": dict(sampling_params),
            "image_data": image_data,
            "video_data": video_data,
            "release": asyncio.Event(),
            "step": step,
        }
        self.calls.append(call)
        self._call_added.set()
        self._call_added = asyncio.Event()
        await call["release"].wait()
        if isinstance(step, Exception):
            raise step
        token_ids = _ids(step)
        return TokenOutput(token_ids=token_ids, log_probs=[-0.1] * len(token_ids), stop_reason="completed")

    async def wait_for_calls(self, count: int) -> None:
        while len(self.calls) < count:
            await asyncio.wait_for(self._call_added.wait(), timeout=5)

    def release_call(self, index: int) -> None:
        self.calls[index]["release"].set()


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
    """Continue a linear conversation on one active chain and trajectory."""
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
    """Split subagent history into a sibling and later resume the main chain."""
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

    state = session.snapshot_state()
    assert state["active_chain_ids"] == [1, 2]
    assert state["active_chain_updated_seq"][1] > state["active_chain_updated_seq"][2]

    trajectories = await session.finalize()

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
    """Start a new chain when compacted context no longer matches a stored prefix."""
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
    """Create siblings for repeated prompts and continue the most recently updated one."""
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
async def test_multiple_chains_distinct_sibling_continuation_matches_older_assistant_prefix():
    """Select an older sibling when its assistant prefix uniquely matches the request."""
    session = _session("distinct-sibling")
    backend = SequencedBackend(["OLDER", "NEWER", "CONT"])
    prompt = [{"role": "user", "content": "same prompt"}]

    await _run(session, backend, prompt)
    await _run(session, backend, prompt)
    before_continuation = session.snapshot_state()

    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "same prompt"},
            {"role": "assistant", "content": "OLDER"},
            {"role": "user", "content": "continue older sibling"},
        ],
    )

    assert before_continuation["active_chain_ids"] == [1, 2]
    chains_by_id = {chain.chain_id: chain for chain in session.active_chains}
    assert chains_by_id[1].message_history[1]["content"] == "OLDER"
    assert chains_by_id[1].message_history[-1]["content"] == "CONT"
    assert chains_by_id[2].message_history == [
        {"role": "user", "content": "same prompt"},
        {"role": "assistant", "content": "NEWER"},
    ]
    assert chains_by_id[1].updated_seq > chains_by_id[2].updated_seq

    trajectories = await session.finalize()
    decoded = [_decode_response_ids(trajectory.response_ids) for trajectory in trajectories]
    assert decoded == ["NEWER", "OLDERuser:continue older sibling\nassistant:CONT"]


@pytest.mark.asyncio
async def test_multiple_chains_reserved_siblings_fall_back_before_starting_new_chain():
    """Reserve matching siblings newest-first, then full-encode when all are busy."""
    session = _session("reserved-siblings")
    prompt = [{"role": "user", "content": "same prompt"}]
    for _ in range(3):
        await _run(session, SequencedBackend(["SAME"]), prompt)

    continuation = [
        *prompt,
        {"role": "assistant", "content": "SAME"},
        {"role": "user", "content": "continue"},
    ]
    backend = _ControlledParallelBackend(["CHAIN3", "CHAIN2", "CHAIN1", "NEW"])
    tasks = []
    for call_count in range(1, 5):
        tasks.append(asyncio.create_task(_run(session, backend, continuation)))
        await backend.wait_for_calls(call_count)

    assert session.snapshot_state()["active_chain_ids"] == [1, 2, 3]
    assert [call["request_id"] for call in backend.calls] == ["reserved-siblings"] * 4

    for index in (3, 2, 1, 0):
        backend.release_call(index)
    await asyncio.gather(*tasks)

    chains_by_id = {chain.chain_id: chain for chain in session.active_chains}
    assert set(chains_by_id) == {1, 2, 3, 4}
    assert _decode_response_ids(chains_by_id[3].buffer.response_ids).endswith("CHAIN3")
    assert _decode_response_ids(chains_by_id[2].buffer.response_ids).endswith("CHAIN2")
    assert _decode_response_ids(chains_by_id[1].buffer.response_ids).endswith("CHAIN1")
    assert all(0 in chains_by_id[chain_id].buffer.response_mask for chain_id in (1, 2, 3))

    new_chain = chains_by_id[4]
    assert new_chain.buffer.prompt_ids == session._codec.encode_full(continuation)
    assert new_chain.buffer.response_ids == _ids("NEW")
    assert new_chain.buffer.response_mask == [1] * len("NEW")


@pytest.mark.asyncio
async def test_multiple_chains_parallel_different_chains_commit_in_place():
    """Commit parallel generations in place when they target distinct live chains."""
    session = _session("parallel-different-chains")
    backend = SequencedBackend(["MAIN1", "SUB1"])
    main_first = [HELPFUL_SYS, {"role": "user", "content": "main"}]
    subagent = [SUBAGENT_SYS, {"role": "user", "content": "sub"}]
    await _run(session, backend, main_first)
    await _run(session, backend, subagent)

    main_continuation = [
        *main_first,
        {"role": "assistant", "content": "MAIN1"},
        {"role": "user", "content": "main next"},
    ]
    subagent_continuation = [
        *subagent,
        {"role": "assistant", "content": "SUB1"},
        {"role": "user", "content": "sub next"},
    ]
    parallel_backend = _ControlledParallelBackend(["MAIN2", "SUB2"])

    main_task = asyncio.create_task(_run(session, parallel_backend, main_continuation))
    sub_task = asyncio.create_task(_run(session, parallel_backend, subagent_continuation))
    await parallel_backend.wait_for_calls(2)
    parallel_backend.release_call(1)
    await sub_task
    parallel_backend.release_call(0)
    await main_task

    assert session.snapshot_state()["active_chain_ids"] == [1, 2]
    trajectories = await session.finalize()
    decoded = [_decode_response_ids(trajectory.response_ids) for trajectory in trajectories]
    assert any(text.startswith("MAIN1") and text.endswith("MAIN2") for text in decoded)
    assert any(text.startswith("SUB1") and text.endswith("SUB2") for text in decoded)


@pytest.mark.asyncio
async def test_multiple_chains_parallel_new_siblings_reuse_session_request_id():
    """Retain concurrent first-turn siblings while reusing the sticky session id."""
    session = _session("parallel-new-siblings")
    backend = _ControlledParallelBackend(["A", "B", "C"])
    prompt = [{"role": "user", "content": "same first turn"}]

    tasks = [asyncio.create_task(_run(session, backend, prompt)) for _ in range(3)]
    await backend.wait_for_calls(3)
    for index in (2, 0, 1):
        backend.release_call(index)
    await asyncio.gather(*tasks)

    request_ids = [call["request_id"] for call in backend.calls]
    assert request_ids == ["parallel-new-siblings"] * 3
    assert session.snapshot_state()["active_chain_ids"] == [1, 2, 3]
    trajectories = await session.finalize()
    assert sorted(_decode_response_ids(trajectory.response_ids) for trajectory in trajectories) == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_multiple_chains_tools_gate_chain_reuse():
    """Start a sibling chain when a continuation changes the available tools."""
    search_tool = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    lookup_tool = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

    session = _session("tools-gate")
    backend = SequencedBackend(["SEARCH", "LOOKUP"])
    await _run(session, backend, [{"role": "user", "content": "use a tool"}], tools=search_tool)
    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "use a tool"},
            {"role": "assistant", "content": "SEARCH"},
            {"role": "user", "content": "continue with a renamed tool"},
        ],
        tools=lookup_tool,
    )
    trajectories = await session.finalize()
    assert [_decode_response_ids(t.response_ids) for t in trajectories] == ["SEARCH", "LOOKUP"]


@pytest.mark.asyncio
async def test_multiple_chains_committed_assistant_tip_hash_round_trips_through_echoed_request():
    """Match a continuation that echoes the canonical committed assistant message."""
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
async def test_multiple_chains_backend_failure_releases_reserved_chain_for_retry():
    """Release an existing-chain reservation when backend generation fails."""
    session = _session("backend-failure")
    first_messages = [{"role": "user", "content": "first turn"}]
    continuation = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "follow up"},
    ]

    await _run(session, SequencedBackend(["FIRST"]), first_messages)
    with pytest.raises(HTTPException, match="RuntimeError: boom"):
        await _run(session, SequencedBackend([RuntimeError("boom")]), continuation)

    await _run(session, SequencedBackend(["SECOND"]), continuation)
    assert session.snapshot_state()["active_chain_ids"] == [1]
    trajectories = await session.finalize()

    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids).endswith("SECOND")
    assert 0 in trajectories[0].response_mask


@pytest.mark.asyncio
async def test_multiple_chains_new_chain_backend_failure_does_not_leave_partial_chain():
    """Avoid creating partial state when a new-chain backend request fails."""
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
    """Close and order the selected chain when its response budget is exhausted."""
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
async def test_multiple_chains_exactly_exhausted_chain_closes_without_backend_call():
    """Close an exhausted chain even when the repeated request has no incremental tail."""
    session = _session("chain-budget-exhausted", response_length=len("NORMAL"))
    backend = SequencedBackend(["NORMAL", "SHOULD_NOT_RUN"])
    first_messages = [{"role": "user", "content": "fill the response budget"}]
    repeated_history = [
        *first_messages,
        {"role": "assistant", "content": "NORMAL"},
    ]

    await _run(session, backend, first_messages)
    outcome = await _run(session, backend, repeated_history)
    trajectories = await session.finalize()

    assert outcome.finish_reason == "length"
    assert len(backend.calls) == 1
    assert backend.steps == ["SHOULD_NOT_RUN"]
    assert len(trajectories) == 1
    assert trajectories[0].extra_fields["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_multiple_chains_sets_remaining_budget_when_request_omits_max_tokens():
    session = _session("default-max-tokens-budget", response_length=10)
    backend = SequencedBackend(["A"])

    await _run(session, backend, [{"role": "user", "content": "use session budget"}])

    assert backend.calls[0]["sampling_params"]["max_tokens"] == 10


@pytest.mark.asyncio
async def test_multiple_chains_multimodal_media_stays_chain_local():
    """Keep image media isolated between independently selected chains."""
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
    """Keep video media and metadata isolated between sibling chains."""
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


@pytest.mark.parametrize(
    ("media_kind", "message_factory", "extractor", "sent_url", "unsent_url", "backend_field", "trajectory_key"),
    [
        (
            "image",
            _image_message,
            fake_vision_info_extractor,
            "image://sent-a.png",
            "image://unsent-b.png",
            "image_data",
            "images",
        ),
        (
            "video",
            _video_message,
            _codec_compatible_video_extractor,
            "video://sent-a.mp4",
            "video://unsent-b.mp4",
            "video_data",
            "videos",
        ),
    ],
)
@pytest.mark.asyncio
async def test_multiple_chains_length_exhaustion_does_not_materialize_unsent_media(
    media_kind, message_factory, extractor, sent_url, unsent_url, backend_field, trajectory_key
):
    session = _session(
        f"length-unsent-{media_kind}",
        response_length=len("FIRST") + 1,
        processor=FakeProcessor(),
        vision_info_extractor=extractor,
    )
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    first_messages = [message_factory(sent_url, "describe first")]
    exhausted_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        message_factory(unsent_url, "new media that exhausts length"),
    ]
    expected_sent = [sent_url] if media_kind == "image" else [(sent_url, {"url": sent_url})]

    await _run(session, backend, first_messages)
    outcome = await _run(session, backend, exhausted_messages)
    trajectories = await session.finalize()

    assert outcome.finish_reason == "length"
    assert len(backend.calls) == 1
    assert backend.steps == ["SHOULD_NOT_RUN"]
    assert backend.calls[0][backend_field] == expected_sent
    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data == {trajectory_key: expected_sent}
    assert trajectories[0].extra_fields["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_multiple_chains_abort_clears_length_materialized_trajectories():
    session = _session("abort-clears-materialized", response_length=len("FIRST") + 1)
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    first_messages = [{"role": "user", "content": "first turn"}]
    exhausted_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "this continuation exhausts the length budget"},
    ]

    await _run(session, backend, first_messages)
    outcome = await _run(session, backend, exhausted_messages)

    assert outcome.finish_reason == "length"
    assert session.snapshot_state()["num_trajectories"] == 1

    await session.abort()

    state = session.snapshot_state()
    assert state["phase"] == "ABORTED"
    assert state["num_trajectories"] == 0
    assert state["num_active_chains"] == 0
    with pytest.raises(RuntimeError, match="aborted"):
        await session.finalize()


@pytest.mark.asyncio
async def test_multiple_chains_backend_media_list_mutation_does_not_change_committed_chain():
    class _MutatingMediaBackend:
        async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
            image_data.append("image://backend-mutated.png")
            return TokenOutput(token_ids=_ids("FIRST"), log_probs=[-0.1] * len("FIRST"), stop_reason="completed")

    session = _session(
        "backend-mutates-media",
        processor=FakeProcessor(),
        vision_info_extractor=fake_vision_info_extractor,
    )
    backend = _MutatingMediaBackend()

    await _run(session, backend, [_image_message("image://stable-a.png", "describe stable")])
    trajectories = await session.finalize()

    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data == {"images": ["image://stable-a.png"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_action", ["finalize", "abort"])
async def test_multiple_chains_terminal_state_rejects_late_commit(terminal_action):
    """Reject a backend result that arrives after finalization or abort."""
    session = _session(f"parallel-late-{terminal_action}")
    await _run(session, SequencedBackend(["FIRST"]), [{"role": "user", "content": "first turn"}])
    pending_backend = _ControlledParallelBackend(["SECOND"])
    pending_task = asyncio.create_task(
        _run(
            session,
            pending_backend,
            [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "FIRST"},
                {"role": "user", "content": "follow up"},
            ],
        )
    )
    await pending_backend.wait_for_calls(1)

    if terminal_action == "finalize":
        terminal_result = await session.finalize()
        assert [_decode_response_ids(trajectory.response_ids) for trajectory in terminal_result] == ["FIRST"]
    else:
        await session.abort()

    pending_backend.release_call(0)
    with pytest.raises(HTTPException) as exc_info:
        await pending_task
    assert exc_info.value.status_code == 409
    assert session.snapshot_state()["active_chain_ids"] == []


@pytest.mark.asyncio
async def test_multiple_chains_reserved_chain_is_not_closed_by_concurrent_length_request():
    """Split rather than length-close a matching chain reserved by another request."""
    session = _session("reserved-length", response_length=len("BASE") + 1)
    await _run(session, SequencedBackend(["BASE"]), [{"role": "user", "content": "base"}])
    pending_backend = _ControlledParallelBackend(["CONT"])
    pending_messages = list(session.active_chains[0].message_history)
    pending_task = asyncio.create_task(_run(session, pending_backend, pending_messages))
    await pending_backend.wait_for_calls(1)

    split_outcome = await _run(
        session,
        SequencedBackend(["FRESH"]),
        [
            {"role": "user", "content": "base"},
            {"role": "assistant", "content": "BASE"},
            {"role": "user", "content": "this continuation is long enough to close the chain"},
        ],
    )
    assert split_outcome.finish_reason == "stop"
    assert session.snapshot_state()["active_chain_ids"] == [1, 2]
    assert session.snapshot_state()["num_trajectories"] == 0
    split_chain = next(chain for chain in session.active_chains if chain.chain_id == 2)
    assert split_chain.buffer.response_ids == _ids("FRESH")
    assert split_chain.buffer.response_mask == [1] * len("FRESH")

    pending_backend.release_call(0)
    await pending_task
    assert session.snapshot_state()["active_chain_ids"] == [1, 2]

    trajectories = await session.finalize()
    decoded = [_decode_response_ids(trajectory.response_ids) for trajectory in trajectories]
    assert len(trajectories) == 2
    assert all("finish_reason" not in trajectory.extra_fields for trajectory in trajectories)
    assert any(text.endswith("CONT") for text in decoded)
    assert "FRESH" in decoded


@pytest.mark.asyncio
async def test_multiple_chains_decode_failure_releases_reserved_chain_for_retry(monkeypatch):
    """Release an existing-chain reservation when response decoding fails."""
    session = _session("decode-failure")
    first_messages = [{"role": "user", "content": "first turn"}]
    continuation = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "follow up"},
    ]
    await _run(session, SequencedBackend(["FIRST"]), first_messages)

    async def decode_response_raises(*args, **kwargs):
        raise RuntimeError("decode boom")

    with monkeypatch.context() as patch:
        patch.setattr(session._codec, "decode_response", decode_response_raises)
        with pytest.raises(HTTPException) as exc_info:
            await _run(session, SequencedBackend(["IGNORED"]), continuation)
    assert exc_info.value.status_code == 500
    assert "decode boom" in str(exc_info.value.detail)

    await _run(session, SequencedBackend(["SECOND"]), continuation)
    assert session.snapshot_state()["active_chain_ids"] == [1]
    trajectories = await session.finalize()
    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids).endswith("SECOND")


@pytest.mark.asyncio
async def test_multiple_chains_cancelled_generation_releases_reserved_chain_for_retry():
    """Release an existing-chain reservation when generation is cancelled."""
    session = _session("cancelled")
    first_messages = [{"role": "user", "content": "first turn"}]
    continuation = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "follow up"},
    ]
    await _run(session, SequencedBackend(["FIRST"]), first_messages)
    pending_backend = _ControlledParallelBackend(["SECOND"])
    pending_task = asyncio.create_task(_run(session, pending_backend, continuation))
    await pending_backend.wait_for_calls(1)

    pending_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_task

    await _run(session, SequencedBackend(["RETRY"]), continuation)
    assert session.snapshot_state()["active_chain_ids"] == [1]
    trajectories = await session.finalize()
    assert len(trajectories) == 1
    assert _decode_response_ids(trajectories[0].response_ids).endswith("RETRY")


@pytest.mark.asyncio
async def test_multiple_chains_prefix_content_change_does_not_reuse_chain_and_hashes_match_history():
    """Split on changed prefix content and keep stored hashes aligned with history."""
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


def test_compute_message_prefix_hashes_canonicalizes_json_tool_call_arguments():
    """Canonicalize JSON-equivalent tool arguments before computing prefix hashes."""
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
@pytest.mark.parametrize("rewrite_fresh_tool_result_id", [False, True], ids=["matching-id", "rewritten-fresh-id"])
async def test_multiple_chains_tool_call_echo_reuses_chain_despite_fresh_tool_result_id(
    rewrite_fresh_tool_result_id,
):
    """Match on the committed prefix; a fresh tool-result ID is outside that boundary."""
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
    tool_result_id = (
        "call_rewritten_fresh_tail" if rewrite_fresh_tool_result_id else first.assistant_msg["tool_calls"][0]["id"]
    )

    await _run(
        session,
        backend,
        [
            {"role": "user", "content": "what is the weather?"},
            {"role": "assistant", "content": None, "tool_calls": first.assistant_msg["tool_calls"]},
            {
                "role": "tool",
                "tool_call_id": tool_result_id,
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
    """Start a sibling when the caller rewrites a committed tool-call ID."""
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
    """Return aligned response logprobs only when every generated segment is covered."""
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
