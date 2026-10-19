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
async def test_extract_multi_modal_data_configured_patch_size_avoids_processor_read():
    """A configured patch size makes the missing processor attribute harmless."""
    from tests.uni_agent.support import FakeTokenizer

    async def extractor(messages, image_patch_size, config=None):
        assert image_patch_size == 14
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
@pytest.mark.parametrize("has_history_image", [False, True], ids=["no-url-fallback", "requires-history-images"])
def test_merge_context_tokens_requires_complete_resolved_images(has_history_image):
    from tests.uni_agent.support import FakeProcessor, QwenVLTokenizer

    codec = MessageCodec(QwenVLTokenizer(), processor=FakeProcessor(), hf_model_type="qwen2_5_vl")
    previous_messages = [_image_url_message()] if has_history_image else [{"role": "user", "content": "start"}]
    updated_messages = [*previous_messages, {"role": "assistant", "content": "FIRST"}, _image_url_message()]
    # None must not fall back to URLs; supplying only new images must not omit history.
    image_data = ["image://a.png"] if has_history_image else None
    with pytest.raises(ValueError, match="image data must align with message blocks"):
        codec.merge_context_tokens(previous_messages, updated_messages, [1, 2, 3], [1, 1, 1], image_data=image_data)
