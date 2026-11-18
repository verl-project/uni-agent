import copy
import sys
from types import SimpleNamespace

import pytest

from tests.uni_agent.support import FakeProcessor, FakeTokenizer
from uni_agent.gateway.session.codec import MessageCodec

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


@pytest.mark.asyncio
async def test_default_extractor_normalizes_urls_without_mutating_messages(monkeypatch):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "video_url", "video_url": {"url": "file:///tmp/demo.mp4"}},
                {"type": "text", "text": "Describe them."},
            ],
        }
    ]

    original = copy.deepcopy(messages)
    calls = []
    images, videos = [object()], [(object(), {"fps": 1})]

    def process_vision_info(normalized, **kwargs):
        calls.append((normalized, kwargs))
        return images, videos

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=process_vision_info))
    codec = MessageCodec(FakeTokenizer(), processor=FakeProcessor())
    assert await codec.extract_multi_modal_data(messages) == (images, videos)
    [(normalized, kwargs)] = calls
    assert kwargs == {"image_patch_size": 16, "return_video_metadata": True}
    assert messages == original

    assert normalized[0]["content"][0]["image_url"] == "data:image/png;base64,AAAA"
    assert normalized[0]["content"][1]["video"] == "file:///tmp/demo.mp4"
    assert "video_url" not in normalized[0]["content"][1]
    assert messages[0]["content"][0]["image_url"] == {"url": "data:image/png;base64,AAAA"}
    assert messages[0]["content"][1]["video_url"] == {"url": "file:///tmp/demo.mp4"}


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {}},
        {"type": "image_url", "image_url": {"url": 1}},
        {"type": "video_url", "video_url": {"url": ""}},
    ],
)
@pytest.mark.asyncio
async def test_default_extractor_rejects_invalid_openai_url_blocks(part, monkeypatch):
    def process_vision_info(*args, **kwargs):
        pytest.fail("invalid URLs must fail before media loading")

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=process_vision_info))
    codec = MessageCodec(FakeTokenizer(), processor=FakeProcessor())
    with pytest.raises(ValueError, match="url"):
        await codec.extract_multi_modal_data([{"role": "user", "content": [part]}])
