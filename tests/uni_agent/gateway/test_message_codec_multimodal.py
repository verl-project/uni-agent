import copy

import pytest

from uni_agent.gateway.session.multimodal import (
    _normalize_messages_for_qwen_vision_info,
    bind_message_images,
    validate_image_count,
)

pytestmark = [pytest.mark.cpu, pytest.mark.level0]


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


def test_bind_message_images_preserves_source_and_binds_in_block_order():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "image://first.png"}},
                {"type": "text", "text": "between"},
                {"type": "image", "image": "image://second.png"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    resolved_images = [object(), object()]

    bound = bind_message_images(messages, resolved_images)

    assert messages == original
    assert bound is not messages
    assert bound[0]["content"][0]["image"] is resolved_images[0]
    assert bound[0]["content"][2]["image"] is resolved_images[1]
    assert bound[0]["content"][1] is messages[0]["content"][1]


@pytest.mark.parametrize("resolved_images", [[], [object(), object()]])
def test_image_count_rejects_missing_or_surplus_images(resolved_images):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "image://one.png"}}],
        }
    ]

    with pytest.raises(ValueError, match="found 1 image blocks"):
        validate_image_count(messages, resolved_images)
