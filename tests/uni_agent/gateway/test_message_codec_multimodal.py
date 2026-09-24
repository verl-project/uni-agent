import pytest

from uni_agent.gateway.session.codec import _normalize_messages_for_qwen_vision_info


def test_qwen_vision_info_unwraps_openai_multimodal_urls_without_mutating_messages():
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

    normalized = _normalize_messages_for_qwen_vision_info(messages)

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
def test_qwen_vision_info_rejects_invalid_openai_url_blocks(part):
    with pytest.raises(ValueError, match="url"):
        _normalize_messages_for_qwen_vision_info([{"role": "user", "content": [part]}])
