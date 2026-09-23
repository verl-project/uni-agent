import pytest

from uni_agent.gateway.session.codec import MessageCodec


class _DeepSeekV4Tokenizer:
    chat_template = None
    eos_token_id = 1

    _SPECIAL_TOKENS = {
        "<｜begin▁of▁sentence｜>": 0,
        "<｜end▁of▁sentence｜>": 1,
        "<｜User｜>": 2,
        "<｜Assistant｜>": 3,
        "<think>": 4,
        "</think>": 5,
    }
    _SPECIAL_IDS = {token_id: token for token, token_id in _SPECIAL_TOKENS.items()}

    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        raise ValueError("Cannot use chat template functions because tokenizer.chat_template is not set")

    def convert_tokens_to_ids(self, token):
        return self._SPECIAL_TOKENS.get(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        token_ids = []
        cursor = 0
        ordered_special_tokens = sorted(self._SPECIAL_TOKENS, key=len, reverse=True)
        while cursor < len(text):
            special_token = next(
                (token for token in ordered_special_tokens if text.startswith(token, cursor)),
                None,
            )
            if special_token is not None:
                token_ids.append(self._SPECIAL_TOKENS[special_token])
                cursor += len(special_token)
            else:
                token_ids.append(1000 + ord(text[cursor]))
                cursor += 1
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        decoded = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self._SPECIAL_IDS:
                if not skip_special_tokens:
                    decoded.append(self._SPECIAL_IDS[token_id])
            else:
                decoded.append(chr(token_id - 1000))
        return "".join(decoded)


@pytest.mark.cpu
@pytest.mark.level0
def test_deepseek_v4_codec_uses_continuous_token_without_chat_template():
    tokenizer = _DeepSeekV4Tokenizer()
    codec = MessageCodec(tokenizer, hf_model_type="deepseek_v4")
    previous_messages = [{"role": "user", "content": "hello"}]

    initial_ids = codec.build_initial_tokens(previous_messages)
    assert tokenizer.decode(initial_ids) == ("<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>")

    assistant_ids = tokenizer.encode("FIRST")
    assistant_token_ids, response_mask, response_logprobs = codec.merge_assistant_tokens(
        initial_ids,
        assistant_ids,
        [],
        [],
        assistant_logprobs=[-0.1] * len(assistant_ids),
    )
    previous_messages = previous_messages + [{"role": "assistant", "content": "FIRST"}]
    updated_messages = previous_messages + [{"role": "tool", "content": "result"}]
    continuation_token_ids, response_mask, response_logprobs = codec.merge_context_tokens(
        previous_messages,
        updated_messages,
        assistant_token_ids,
        response_mask,
        response_logprobs,
    )

    assert continuation_token_ids[len(initial_ids) + len(assistant_ids)] == tokenizer.eos_token_id
    assert tokenizer.decode(continuation_token_ids).endswith(
        "FIRST<｜end▁of▁sentence｜><｜User｜><tool_result>result</tool_result><｜Assistant｜></think>"
    )
    assert response_mask[: len(assistant_ids)] == [1] * len(assistant_ids)
    assert response_mask[len(assistant_ids) :] == [0] * (len(response_mask) - len(assistant_ids))
    assert response_logprobs is not None
    assert response_logprobs[: len(assistant_ids)] == [-0.1] * len(assistant_ids)
    assert response_logprobs[len(assistant_ids) :] == [0.0] * (len(response_logprobs) - len(assistant_ids))


class _NoPatchSizeProcessor:
    """Multimodal processor without the Qwen ``patch_size`` attribute."""

    chat_template = None

    class image_processor:
        pass


def _image_url_message() -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "image://a.png"}},
            {"type": "text", "text": "describe"},
        ],
    }


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_extract_multi_modal_data_prefers_configured_patch_size():
    """An explicitly configured patch size wins and the processor is not read."""
    from tests.uni_agent.support import FakeProcessor, FakeTokenizer

    extractor_calls = []

    async def recording_extractor(messages, image_patch_size, config=None):
        extractor_calls.append(image_patch_size)
        return ["image://a.png"], None

    codec = MessageCodec(
        FakeTokenizer(),
        processor=FakeProcessor(),
        vision_info_extractor=recording_extractor,
        vision_info_extractor_kwargs={"image_patch_size": 14},
    )

    images, videos = await codec.extract_multi_modal_data([_image_url_message()])

    assert images == ["image://a.png"]
    assert videos is None
    assert extractor_calls == [14]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_extract_multi_modal_data_reports_missing_patch_size():
    """The Qwen patch size read fails clearly when the processor lacks it and nothing is configured."""
    from tests.uni_agent.support import FakeTokenizer

    async def extractor(messages, image_patch_size):
        raise AssertionError("extractor must not run when the patch size is unavailable")

    codec = MessageCodec(
        FakeTokenizer(),
        processor=_NoPatchSizeProcessor(),
        vision_info_extractor=extractor,
    )

    with pytest.raises(ValueError, match="processor.image_processor.patch_size"):
        await codec.extract_multi_modal_data([_image_url_message()])


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_extract_multi_modal_data_configured_patch_size_avoids_processor_read():
    """A configured patch size makes the missing processor attribute harmless."""
    from tests.uni_agent.support import FakeTokenizer

    async def extractor(messages, image_patch_size, config=None):
        return ["image://a.png"], None

    codec = MessageCodec(
        FakeTokenizer(),
        processor=_NoPatchSizeProcessor(),
        vision_info_extractor=extractor,
        vision_info_extractor_kwargs={"image_patch_size": 14},
    )

    images, videos = await codec.extract_multi_modal_data([_image_url_message()])

    assert images == ["image://a.png"]
    assert videos is None


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
@pytest.mark.parametrize("signature_kind", ["optional", "kwargs"])
async def test_extract_multi_modal_data_optional_patch_size_does_not_require_processor_attribute(signature_kind):
    """Optional patch-size contracts work with non-Qwen processor shapes."""
    from tests.uni_agent.support import FakeTokenizer

    calls = []
    if signature_kind == "optional":

        async def extractor(messages, image_patch_size=None):
            calls.append(image_patch_size)
            return ["image://a.png"], None

    else:

        async def extractor(messages, **kwargs):
            calls.append(kwargs)
            return ["image://a.png"], None

    codec = MessageCodec(
        FakeTokenizer(),
        processor=_NoPatchSizeProcessor(),
        vision_info_extractor=extractor,
    )

    images, videos = await codec.extract_multi_modal_data([_image_url_message()])

    assert images == ["image://a.png"]
    assert videos is None
    assert calls == ([None] if signature_kind == "optional" else [{}])


@pytest.mark.cpu
@pytest.mark.level0
def test_merge_context_tokens_rejects_misaligned_image_data():
    """Image data shorter or longer than the image blocks never reaches the builder."""
    from tests.uni_agent.support import FakeProcessor, QwenVLTokenizer

    tokenizer = QwenVLTokenizer()
    codec = MessageCodec(
        tokenizer,
        processor=FakeProcessor(),
        hf_model_type="qwen2_5_vl",
    )
    previous_messages = [{"role": "user", "content": "start"}]
    appended_tool_image = {
        "role": "tool",
        "tool_call_id": "call_crop",
        "name": "crop_image",
        "content": [
            {"type": "image_url", "image_url": {"url": "image://crop.png"}},
            {"type": "text", "text": "crop"},
        ],
    }
    updated_messages = [*previous_messages, {"role": "assistant", "content": "FIRST"}, appended_tool_image]

    with pytest.raises(ValueError, match="found 1 image blocks but received 0 resolved images"):
        codec.merge_context_tokens(previous_messages, updated_messages, [1, 2, 3], [1, 1, 1], image_data=[])
    with pytest.raises(ValueError, match="found 1 image blocks but received 2 resolved images"):
        codec.merge_context_tokens(
            previous_messages,
            updated_messages,
            [1, 2, 3],
            [1, 1, 1],
            image_data=["image://a.png", "image://b.png"],
        )

    # A history image block without its resolved object is also rejected, even
    # when the appended messages are pure text.
    history_messages = [_image_url_message()]
    text_appended = [*history_messages, {"role": "assistant", "content": "FIRST"}, {"role": "user", "content": "go on"}]
    with pytest.raises(ValueError, match="found 1 image blocks but received 0 resolved images"):
        codec.merge_context_tokens(history_messages, text_appended, [1, 2, 3], [1, 1, 1], image_data=[])
