"""Multimodal input preparation for Gateway messages.

Provides media extraction, input normalization, and temporary image binding
for model encoding. Model-specific preparation is kept in dedicated helpers.
The codec retains processor/extractor configuration, but no session state or
media cache. Preparation never mutates original messages.
"""

from __future__ import annotations

import inspect
from typing import Any


def count_media_blocks(messages: list[dict[str, Any]], block_types: set[str]) -> int:
    """Count content blocks of the given types across messages."""
    count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in block_types:
                count += 1
    return count


def has_media_blocks(
    messages: list[dict[str, Any]],
    block_types: set[str] | None = None,
) -> bool:
    """Return whether any message contains a block of the requested types."""
    if block_types is None:
        block_types = {"image", "image_url", "video", "video_url"}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(isinstance(block, dict) and block.get("type") in block_types for block in content):
            return True
    return False


def bind_message_images(
    messages: list[dict[str, Any]],
    image_data: list[Any] | None,
) -> list[dict[str, Any]]:
    """Return message copies with each image block bound to its resolved image.

    Image blocks consume ``image_data`` in message order, matching how media
    extraction and the Continuous Token builders order images. Binding is
    strict: every image block must receive a resolved image and no image may
    be left over, so mixed raw references and resolved objects can never reach
    a builder. Only messages and blocks that receive an image are copied; the
    input messages are never mutated.
    """
    images = image_data if image_data is not None else []
    bound_messages = messages
    image_index = 0
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        bound_content = content
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") not in {"image", "image_url"}:
                continue
            if image_index < len(images):
                if bound_content is content:
                    bound_content = list(content)
                bound_content[block_index] = {**block, "image": images[image_index]}
            image_index += 1
        if bound_content is not content:
            if bound_messages is messages:
                bound_messages = list(messages)
            bound_messages[message_index] = {**message, "content": bound_content}
    if image_index != len(images):
        raise ValueError(
            "Continuous Token image data must align with image blocks: "
            f"found {image_index} image blocks but received {len(images)} resolved images"
        )
    return bound_messages


def _normalize_url_value(value: Any, *, field: str) -> str:
    """Normalize a URL field to a non-empty string."""
    if isinstance(value, dict):
        if "url" not in value:
            raise ValueError(f"{field} must contain a url field")
        value = value["url"]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}.url must be a non-empty string")
    return value


def _normalize_messages_for_qwen_vision_info(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize multimodal message content for ``qwen_vl_utils``.

    ``qwen_vl_utils`` expects ``image_url`` values to be URL strings and
    video URLs under the ``video`` key. Normalize only the copied messages
    passed to the vision extractor.
    """
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        normalized_message = dict(message)
        content = message.get("content")
        if not isinstance(content, list):
            normalized_messages.append(normalized_message)
            continue

        normalized_parts: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                normalized_parts.append(part)
                continue
            normalized_part = dict(part)
            if "image_url" in normalized_part:
                normalized_part["image_url"] = _normalize_url_value(
                    normalized_part["image_url"],
                    field="image_url",
                )
            if "video_url" in normalized_part:
                normalized_part["video"] = _normalize_url_value(normalized_part.pop("video_url"), field="video_url")
            normalized_parts.append(normalized_part)
        normalized_message["content"] = normalized_parts
        normalized_messages.append(normalized_message)
    return normalized_messages


def validate_image_count(messages: list[dict[str, Any]], image_data: list[Any] | None) -> None:
    """Require one image per block; count alone cannot detect reordered images."""
    block_count = count_media_blocks(messages, {"image", "image_url"})
    image_count = len(image_data) if image_data is not None else 0
    if block_count != image_count:
        raise ValueError(
            "Continuous Token image data must align with image blocks: "
            f"found {block_count} image blocks but received {image_count} resolved images"
        )


def extract_qwen_vision_inputs(
    messages: list[dict[str, Any]],
    *,
    image_patch_size: int,
) -> tuple[list[Any] | None, list[Any] | None]:
    """Resolve Qwen media from a temporary OpenAI URL-normalized view."""
    from qwen_vl_utils import process_vision_info

    return process_vision_info(
        _normalize_messages_for_qwen_vision_info(messages),
        image_patch_size=image_patch_size,
        return_video_metadata=True,
    )


def _image_patch_size_contract(extractor) -> tuple[bool, bool]:
    """Return whether a custom extractor accepts and requires patch size."""
    try:
        parameters = inspect.signature(extractor).parameters
    except (TypeError, ValueError):
        # Opaque callables retain the legacy patch-size contract.
        return True, True

    parameter = parameters.get("image_patch_size")
    accepts = parameter is not None or any(
        candidate.kind == inspect.Parameter.VAR_KEYWORD for candidate in parameters.values()
    )
    requires = parameter is not None and parameter.default is inspect.Parameter.empty
    return accepts, requires


class MultimodalCodec:
    """Resolve Gateway media and adapt it to verl's message-based VL CT input."""

    def __init__(
        self,
        processor: Any,
        *,
        vision_info_extractor=None,
        vision_info_extractor_kwargs: dict[str, Any] | None = None,
        supports_incremental_images: bool = False,
    ) -> None:
        self._processor = processor
        self._vision_info_extractor = vision_info_extractor or self._default_vision_info_extractor
        self._vision_info_extractor_kwargs = dict(vision_info_extractor_kwargs or {})
        self._supports_incremental_images = supports_incremental_images
        self._custom_extractor_accepts_patch_size = False
        self._custom_extractor_requires_patch_size = False
        if vision_info_extractor is not None:
            (
                self._custom_extractor_accepts_patch_size,
                self._custom_extractor_requires_patch_size,
            ) = _image_patch_size_contract(vision_info_extractor)

    async def _default_vision_info_extractor(
        self,
        messages: list[dict[str, Any]],
        *,
        image_patch_size: int | None = None,
        **_extra: Any,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        if image_patch_size is None:
            image_patch_size = self._image_processor_patch_size()
        return extract_qwen_vision_inputs(messages, image_patch_size=image_patch_size)

    def _image_processor_patch_size(self) -> int:
        patch_size = getattr(getattr(self._processor, "image_processor", None), "patch_size", None)
        if patch_size is None:
            raise ValueError(
                "The configured vision_info_extractor requires processor.image_processor.patch_size, "
                "but this processor does not provide one; configure a vision_info_extractor "
                "that does not depend on the Qwen patch size"
            )
        return patch_size

    async def extract(self, messages: list[dict[str, Any]]) -> tuple[list[Any] | None, list[Any] | None]:
        """Resolve media only for processor-backed requests containing media blocks.

        A custom extractor must return images in message/content-block order,
        with exactly one processor-compatible object per image block. Count
        validation detects missing or surplus objects, but not reordering.
        """
        if self._processor is None or not has_media_blocks(messages):
            return None, None

        extractor_kwargs = dict(self._vision_info_extractor_kwargs)
        if self._custom_extractor_accepts_patch_size and "image_patch_size" not in extractor_kwargs:
            patch_size = getattr(getattr(self._processor, "image_processor", None), "patch_size", None)
            if patch_size is not None:
                extractor_kwargs["image_patch_size"] = patch_size
            elif self._custom_extractor_requires_patch_size:
                extractor_kwargs["image_patch_size"] = self._image_processor_patch_size()
        return await self._vision_info_extractor(messages, **extractor_kwargs)

    def validate_incremental_messages(self, messages: list[dict[str, Any]]) -> None:
        """Reject unsupported media before extraction or CT rendering."""
        if has_media_blocks(messages, {"video", "video_url"}):
            raise ValueError("Continuous Token context merging does not currently support incremental video data")
        if not self._supports_incremental_images and has_media_blocks(messages, {"image", "image_url"}):
            raise ValueError(
                "Continuous Token context merging does not currently support incremental images for this model; "
                "a Qwen VL Continuous Token builder is required"
            )

    def validate_images(self, messages: list[dict[str, Any]], image_data: list[Any] | None) -> None:
        """Enforce exact image-block alignment on the supported Qwen VL path."""
        if self._supports_incremental_images:
            validate_image_count(messages, image_data)

    def validate_incremental_images(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None,
        videos: list[Any] | None,
    ) -> None:
        if videos:
            raise ValueError("Continuous Token context merging does not currently support incremental video data")
        self.validate_images(messages, images)

    def prepare_context_merge(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        *,
        image_data: list[Any] | None,
        video_data: list[Any] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return the message view required by verl's CT builder.

        ``image_data`` covers the complete ``updated_messages``: one resolved
        image per image block, in message and content-block order. Bind those
        images to temporary blocks so verl can use its existing merge logic
        without changing the stored OpenAI messages or CT algorithm.
        """
        if video_data:
            raise ValueError("Continuous Token context merging does not currently support incremental video data")
        if updated_messages[: len(previous_messages)] != previous_messages:
            raise ValueError("Continuous Token messages must be append-only; prefix messages changed")
        self.validate_incremental_messages(updated_messages[len(previous_messages) :])
        if self._supports_incremental_images:
            updated_messages = bind_message_images(updated_messages, image_data)
        return updated_messages[: len(previous_messages)], updated_messages
