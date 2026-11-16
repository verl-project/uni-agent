import pytest


class _UnstableGenerationPromptTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize=True, **kwargs):
        del messages, kwargs
        return [10, 20] if not add_generation_prompt else [99, 30, 40]


@pytest.mark.cpu
@pytest.mark.level0
def test_initialize_generation_prompt_returns_generation_suffix():
    from tests.uni_agent.support import FakeTokenizer
    from uni_agent.gateway.session import codec as codec_module

    assert codec_module.initialize_generation_prompt(FakeTokenizer()) == [ord(char) for char in "assistant:"]


@pytest.mark.cpu
@pytest.mark.level0
def test_initialize_generation_prompt_rejects_non_prefix_template():
    from uni_agent.gateway.session import codec as codec_module

    with pytest.raises(ValueError, match="stable token suffix"):
        codec_module.initialize_generation_prompt(_UnstableGenerationPromptTokenizer())


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("image_shape", ["nested_url", "flat_url", "image"])
@pytest.mark.asyncio
async def test_default_vision_extractor_lowers_chat_image_urls_without_mutation(monkeypatch, image_shape):
    from tests.uni_agent.support import FakeProcessor, FakeTokenizer
    from uni_agent.gateway.session.codec import MessageCodec

    qwen_vl_utils = pytest.importorskip("qwen_vl_utils")
    image = object()
    block = {
        "nested_url": {"type": "image_url", "image_url": {"url": "image://crop.png", "detail": "high"}},
        "flat_url": {"type": "image_url", "image_url": "image://crop.png"},
        "image": {"type": "image", "image": image},
    }[image_shape]
    saved_block = dict(block)
    if isinstance(block.get("image_url"), dict):
        saved_block["image_url"] = dict(block["image_url"])
    messages = [{"role": "tool", "tool_call_id": "crop", "content": [block]}]

    def process_vision_info(vision_messages, **kwargs):
        assert vision_messages[0]["role"] == "tool"
        assert vision_messages[0]["tool_call_id"] == "crop"
        source = vision_messages[0]["content"][0]
        if image_shape == "image":
            assert source["image"] is image
        else:
            assert source["image_url"] == "image://crop.png"
        assert kwargs == {"image_patch_size": 16, "return_video_metadata": True}
        return [image], None

    monkeypatch.setattr(qwen_vl_utils, "process_vision_info", process_vision_info)
    codec = MessageCodec(FakeTokenizer(), processor=FakeProcessor())
    images, videos = await codec.extract_multi_modal_data(messages)
    assert images == [image]
    assert videos is None
    assert messages == [{"role": "tool", "tool_call_id": "crop", "content": [saved_block]}]
