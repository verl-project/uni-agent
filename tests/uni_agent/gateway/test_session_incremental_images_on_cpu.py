import copy
from types import SimpleNamespace

import pytest

from tests.uni_agent.support import (
    FakeProcessor,
    FakeTokenizer,
    QwenVLTokenizer,
    SequencedBackend,
    fake_vision_info_extractor,
)
from uni_agent.gateway.adapters.openai import openai_to_internal
from uni_agent.gateway.session import GatewaySession, MessageCodec, SessionHandle
from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

CROP_TOOLS = [{"type": "function", "function": {"name": "crop_image", "parameters": {"type": "object"}}}]
CROP_TOOL_CALL_TEXT = '<tool_call>\n{"name": "crop_image", "arguments": {"path": "image://source.png"}}\n</tool_call>'


class _QwenVLProcessor(FakeProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer


async def _crop_tool_call_dispatch(self, response_ids, tools, parser_name):
    text = self._tokenizer.decode(response_ids, skip_special_tokens=False)
    if "<tool_call>" not in text:
        return text, []
    return "", [SimpleNamespace(name="crop_image", arguments='{"path": "image://source.png"}')]


def _image_message(url: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text},
        ],
    }


def _tool_image_message(url: str, text: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": "call_crop",
        "name": "crop_image",
        "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": text},
        ],
    }


def _qwen_vl_codec(vision_info_extractor=None, tool_parser_name=None) -> MessageCodec:
    tokenizer = QwenVLTokenizer()
    return MessageCodec(
        tokenizer,
        processor=_QwenVLProcessor(tokenizer),
        hf_model_type="qwen2_5_vl",
        vision_info_extractor=vision_info_extractor,
        tool_parser_name=tool_parser_name,
    )


async def _run(session: GatewaySession, backend, messages, tools=None):
    request = openai_to_internal(
        {"model": "dummy-vl-model", "messages": messages, "logprobs": True, "tools": tools},
        base_sampling_params=session.sampling_params,
        allowed_sampling_keys=frozenset({"logprobs", "max_tokens", "temperature", "top_p", "top_k", "stop"}),
    )
    return await session.run_generation(request, backend)


def _vision_token_count(prompt_ids: list[int]) -> int:
    return prompt_ids.count(FakeProcessor.image_token_id)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_images_qwen_vl_accumulates_tool_images_across_turns(monkeypatch):
    """Merge tool-returned images from legal tool calls into context and trajectory."""
    monkeypatch.setattr(MessageCodec, "_extract_tool_calls", _crop_tool_call_dispatch)
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor, tool_parser_name="hermes")
    assert isinstance(codec._continuous_token_builder, QwenVLContinuousTokenBuilder)
    session = GatewaySession(SessionHandle(session_id="incremental-images-qwen-vl"), codec)
    backend = SequencedBackend([CROP_TOOL_CALL_TEXT, CROP_TOOL_CALL_TEXT, "FINAL"])

    source = _image_message("image://source.png", "find a crop")
    first_messages = [source]
    first = await _run(session, backend, first_messages, tools=CROP_TOOLS)
    assert first.finish_reason == "tool_calls"

    assistant_one = first.assistant_msg
    tool_one = {
        "role": "tool",
        "tool_call_id": assistant_one["tool_calls"][0]["id"],
        "name": "crop_image",
        "content": [
            {"type": "image_url", "image_url": {"url": "image://crop-one.png"}},
            {"type": "text", "text": "first crop"},
        ],
    }
    second_messages = [source, assistant_one, tool_one]
    second = await _run(session, backend, second_messages, tools=CROP_TOOLS)
    assert second.finish_reason == "tool_calls"

    assistant_two = second.assistant_msg
    tool_two = {
        "role": "tool",
        "tool_call_id": assistant_two["tool_calls"][0]["id"],
        "name": "crop_image",
        "content": [
            {"type": "image_url", "image_url": {"url": "image://crop-two.png"}},
            {"type": "text", "text": "second crop"},
            {"type": "image_url", "image_url": {"url": "image://crop-three.png"}},
            {"type": "text", "text": "done"},
        ],
    }
    third_messages = [*second_messages, assistant_two, tool_two]
    original_messages = copy.deepcopy([first_messages, second_messages, third_messages])

    third = await _run(session, backend, third_messages, tools=CROP_TOOLS)
    assert third.finish_reason == "stop"
    [trajectory] = await session.finalize()

    assert [call["image_data"] for call in backend.calls] == [
        ["image://source.png"],
        ["image://source.png", "image://crop-one.png"],
        ["image://source.png", "image://crop-one.png", "image://crop-two.png", "image://crop-three.png"],
    ]
    # The images are actually encoded into the context, not just listed.
    assert [_vision_token_count(call["prompt_ids"]) for call in backend.calls] == [1, 2, 4]
    assert [call["video_data"] for call in backend.calls] == [None, None, None]
    assert [first_messages, second_messages, third_messages] == original_messages
    assert trajectory.multi_modal_data == {
        "images": ["image://source.png", "image://crop-one.png", "image://crop-two.png", "image://crop-three.png"]
    }
    assert trajectory.response_ids
    assert len(trajectory.response_ids) == len(trajectory.response_mask) == len(trajectory.response_logprobs)
    assert trajectory.response_mask[-len("FINAL") :] == [1] * len("FINAL")
    assert trajectory.response_logprobs[-len("FINAL") :] == [-0.1] * len("FINAL")
    assert all(
        logprob == 0.0
        for mask, logprob in zip(trajectory.response_mask, trajectory.response_logprobs, strict=True)
        if mask == 0
    )


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_images_initial_request_with_tool_image_history():
    """Encode a first request that already contains tool-returned images."""
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    session = GatewaySession(SessionHandle(session_id="incremental-images-initial-tool-history"), codec)
    backend = SequencedBackend(["FIRST"])
    messages = [
        _image_message("image://source.png", "find a crop"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_crop",
                    "type": "function",
                    "function": {"name": "crop_image", "arguments": "{}"},
                }
            ],
        },
        _tool_image_message("image://crop.png", "crop"),
        {"role": "user", "content": "what do you see"},
    ]

    await _run(session, backend, messages)
    [trajectory] = await session.finalize()

    assert backend.calls[0]["image_data"] == ["image://source.png", "image://crop.png"]
    assert _vision_token_count(backend.calls[0]["prompt_ids"]) == 2
    assert trajectory.multi_modal_data == {"images": ["image://source.png", "image://crop.png"]}
    assert len(trajectory.response_ids) == len(trajectory.response_mask)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_images_accept_image_after_text_only_start():
    """Append an image to a chain whose accepted history is pure text."""
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    session = GatewaySession(SessionHandle(session_id="incremental-images-text-start"), codec)
    backend = SequencedBackend(["FIRST", "SECOND"])

    first_messages = [{"role": "user", "content": "text only start"}]
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        _image_message("image://appended.png", "now with an image"),
    ]

    await _run(session, backend, first_messages)
    await _run(session, backend, second_messages)
    [trajectory] = await session.finalize()

    assert backend.calls[0]["image_data"] is None
    assert backend.calls[1]["image_data"] == ["image://appended.png"]
    assert trajectory.multi_modal_data == {"images": ["image://appended.png"]}
    assert len(trajectory.response_ids) == len(trajectory.response_mask)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_text_only_continuation_does_not_call_image_extractor():
    """Pure-text Qwen turns stay on the existing CT path without media work."""

    async def forbidden_extractor(*args, **kwargs):
        raise AssertionError("text-only messages must not invoke the image extractor")

    codec = _qwen_vl_codec(vision_info_extractor=forbidden_extractor)
    session = GatewaySession(SessionHandle(session_id="incremental-images-text-only"), codec)
    backend = SequencedBackend(["FIRST", "SECOND"])
    first_messages = [{"role": "user", "content": "text only start"}]
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "text only continuation"},
    ]

    await _run(session, backend, first_messages)
    await _run(session, backend, continuation_messages)

    assert len(backend.calls) == 2
    assert all(call["image_data"] is None for call in backend.calls)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_images_stay_chain_local():
    """Keep incremental images isolated between chains branching off one root."""
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    session = GatewaySession(SessionHandle(session_id="incremental-images-chain-local"), codec)
    backend = SequencedBackend(["ROOT", "BRANCH_A", "BRANCH_B"])

    root = [_image_message("image://root.png", "describe the root")]
    branch_a = [
        *root,
        {"role": "assistant", "content": "ROOT"},
        _image_message("image://branch-a.png", "branch a"),
    ]
    branch_b = [
        *root,
        {"role": "assistant", "content": "ROOT"},
        _image_message("image://branch-b.png", "branch b"),
    ]

    await _run(session, backend, root)
    await _run(session, backend, branch_a)
    await _run(session, backend, branch_b)
    trajectories = await session.finalize()

    assert [call["image_data"] for call in backend.calls] == [
        ["image://root.png"],
        ["image://root.png", "image://branch-a.png"],
        ["image://root.png", "image://branch-b.png"],
    ]
    assert len(trajectories) == 2
    assert trajectories[0].multi_modal_data == {"images": ["image://root.png", "image://branch-a.png"]}
    assert trajectories[1].multi_modal_data == {"images": ["image://root.png", "image://branch-b.png"]}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("codec_factory", "expected_trajectory_media"),
    [
        (
            lambda: MessageCodec(
                FakeTokenizer(),
                processor=FakeProcessor(),
                vision_info_extractor=fake_vision_info_extractor,
            ),
            {"images": ["image://sent-a.png"]},
        ),
        (
            lambda: MessageCodec(FakeTokenizer()),
            None,
        ),
    ],
    ids=["generic-vl-builder", "text-builder"],
)
@pytest.mark.asyncio
async def test_incremental_images_rejected_for_out_of_scope_builders(codec_factory, expected_trajectory_media):
    """Builders outside the Qwen VL path reject appended images instead of dropping them."""
    codec = codec_factory()
    session = GatewaySession(SessionHandle(session_id="incremental-images-out-of-scope"), codec)
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    first_messages = [_image_message("image://sent-a.png", "describe first")]
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        _image_message("image://appended.png", "new incremental image"),
    ]

    await _run(session, backend, first_messages)
    with pytest.raises(ValueError, match="does not currently support incremental images for this model"):
        await _run(session, backend, continuation_messages)
    trajectories = await session.finalize()

    assert len(backend.calls) == 1
    assert backend.steps == ["SHOULD_NOT_RUN"]
    assert len(trajectories) == 1
    assert trajectories[0].multi_modal_data == expected_trajectory_media


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_non_qwen_incremental_image_rejected_before_loading():
    """Out-of-scope builders reject appended images before resolving their source."""

    async def forbidden_extractor(*args, **kwargs):
        raise AssertionError("unsupported incremental images must fail before loading")

    codec = MessageCodec(
        FakeTokenizer(),
        processor=FakeProcessor(),
        vision_info_extractor=forbidden_extractor,
    )
    session = GatewaySession(SessionHandle(session_id="incremental-images-non-qwen"), codec)
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    first_messages = [{"role": "user", "content": "text only start"}]
    continuation_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        _image_message("image://appended.png", "unsupported image"),
    ]

    await _run(session, backend, first_messages)
    with pytest.raises(ValueError, match="Qwen VL Continuous Token builder is required"):
        await _run(session, backend, continuation_messages)

    assert len(backend.calls) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_images_mismatch_does_not_pollute_accepted_chain(monkeypatch):
    """Reject unresolved appended image blocks and keep the chain usable."""
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    session = GatewaySession(SessionHandle(session_id="incremental-images-mismatch"), codec)
    backend = SequencedBackend(["FIRST", "SECOND"])
    first_messages = [_image_message("image://source.png", "describe")]
    bad_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        _tool_image_message("image://crop.png", "crop"),
    ]
    good_messages = [
        *first_messages,
        {"role": "assistant", "content": "FIRST"},
        {"role": "user", "content": "text only follow-up"},
    ]

    async def extract_without_images(messages):
        return None, None

    await _run(session, backend, first_messages)
    monkeypatch.setattr(codec, "extract_multi_modal_data", extract_without_images)
    with pytest.raises(ValueError, match="must align with image blocks"):
        await _run(session, backend, bad_messages)
    monkeypatch.undo()
    outcome = await _run(session, backend, good_messages)
    [trajectory] = await session.finalize()

    assert outcome.finish_reason == "stop"
    assert [call["image_data"] for call in backend.calls] == [
        ["image://source.png"],
        ["image://source.png"],
    ]
    assert trajectory.multi_modal_data == {"images": ["image://source.png"]}
    assert FakeTokenizer().decode(trajectory.response_ids).endswith("SECOND")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_images_survive_last_assistant_rollback():
    """Rewrite the last assistant and append a new image on the rolled-back chain."""
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    session = GatewaySession(SessionHandle(session_id="incremental-images-rollback"), codec)
    backend = SequencedBackend(["FIRST", "REWRITTEN"])

    first_messages = [_image_message("image://source.png", "describe")]
    rewrite_messages = [
        *first_messages,
        {"role": "assistant", "content": "DIFFERENT"},
        _tool_image_message("image://crop.png", "crop"),
    ]

    await _run(session, backend, first_messages)
    await _run(session, backend, rewrite_messages)
    [trajectory] = await session.finalize()

    assert [call["image_data"] for call in backend.calls] == [
        ["image://source.png"],
        ["image://source.png", "image://crop.png"],
    ]
    assert trajectory.multi_modal_data == {"images": ["image://source.png", "image://crop.png"]}
    # The abandoned assistant's tokens are gone; only the rewritten response trains.
    assert sum(trajectory.response_mask) == len("REWRITTEN")
    assert FakeTokenizer().decode(trajectory.response_ids).endswith("REWRITTEN")


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_images_reject_initial_image_count_mismatch(monkeypatch):
    """Reject an initial request whose extractor resolves fewer images than blocks."""
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    session = GatewaySession(SessionHandle(session_id="incremental-images-initial-mismatch"), codec)
    backend = SequencedBackend(["SHOULD_NOT_RUN"])

    async def extract_without_images(messages):
        return [], None

    monkeypatch.setattr(codec, "extract_multi_modal_data", extract_without_images)
    with pytest.raises(ValueError, match="found 1 image blocks but received 0 resolved images"):
        await _run(session, backend, [_image_message("image://source.png", "describe")])

    assert backend.steps == ["SHOULD_NOT_RUN"]
    assert session.active_chains == []
    assert await session.finalize() == []


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["extract", "merge"])
async def test_incremental_image_failure_preserves_chain(monkeypatch, failure_stage):
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    session = GatewaySession(SessionHandle(session_id="image-failure"), codec)
    backend = SequencedBackend(["FIRST", "SECOND"])
    first = [_image_message("image://source.png", "start")]
    await _run(session, backend, first)
    history = [*first, {"role": "assistant", "content": "FIRST"}]

    async def fail_extract(messages):
        raise RuntimeError("image failure")

    def fail_merge(*args, **kwargs):
        raise RuntimeError("image failure")

    with monkeypatch.context() as patch:
        if failure_stage == "extract":
            patch.setattr(codec, "extract_multi_modal_data", fail_extract)
        else:
            patch.setattr(codec._continuous_token_builder, "merge_context_tokens", fail_merge)
        with pytest.raises(RuntimeError, match="image failure"):
            await _run(session, backend, [*history, _image_message("image://bad.png", "bad")])
    await _run(session, backend, [*history, {"role": "user", "content": "continue"}])
    [trajectory] = await session.finalize()
    assert trajectory.multi_modal_data == {"images": ["image://source.png"]}
    assert len(backend.calls) == 2
    assert backend.calls[-1]["image_data"] == ["image://source.png"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_incremental_image_capacity_rejection_keeps_accepted_media():
    codec = _qwen_vl_codec(vision_info_extractor=fake_vision_info_extractor)
    first = [_image_message("image://source.png", "start")]
    images, _ = await codec.extract_multi_modal_data(first)
    prompt_length = len(codec.build_initial_tokens(first, image_data=images))
    session = GatewaySession(
        SessionHandle(session_id="image-capacity"),
        codec,
        prompt_length=prompt_length,
        response_length=12,
    )
    backend = SequencedBackend(["FIRST", "SHOULD_NOT_RUN"])
    await _run(session, backend, first)
    outcome = await _run(
        session,
        backend,
        [
            *first,
            {"role": "assistant", "content": "FIRST"},
            _image_message("image://unused.png", "a long continuation that exceeds the remaining capacity"),
        ],
    )
    [trajectory] = await session.finalize()
    assert outcome.finish_reason == "length"
    assert len(backend.calls) == 1
    assert trajectory.multi_modal_data == {"images": ["image://source.png"]}
    assert sum(trajectory.response_mask) == len("FIRST")
