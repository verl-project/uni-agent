"""Minimal dataset for the DeepEyes gateway recipe.

Produces ``raw_prompt`` and reward-related fields only.
It does not perform tokenization or vision processing.
"""

from __future__ import annotations

import copy
import io
import logging
import re

import torch
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class DeepEyesGatewayDataset(RLHFDataset):
    """Thin dataset that leaves prompt encoding and vision extraction to the gateway."""

    def _build_messages(self, example: dict, key: str) -> tuple[list[dict], object | None]:
        messages = copy.deepcopy(example[key])
        images = example.get(self.image_key, None) or []
        videos = example.get(self.video_key, None) or []
        first_image = None
        image_offset = 0
        video_offset = 0

        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                normalized = []
                for part in content:
                    normalized_part = _normalize_content_part(part)
                    if (
                        first_image is None
                        and isinstance(normalized_part, dict)
                        and normalized_part.get("type") in {"image", "image_url"}
                    ):
                        first_image = _decode_image_payload(normalized_part.get("image", normalized_part))
                        normalized_part = dict(normalized_part)
                        normalized_part["image"] = first_image
                    normalized.append(normalized_part)
                message["content"] = normalized
                continue
            if not isinstance(content, str) or ("<image>" not in content and "<video>" not in content):
                continue

            content_list = []
            for segment in (segment for segment in re.split("(<image>|<video>)", content) if segment):
                if segment == "<image>":
                    assert image_offset < len(images), f"image placeholder count exceeds images at index {image_offset}"
                    image = _decode_image_payload(images[image_offset])
                    if first_image is None:
                        first_image = image
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video placeholder count exceeds videos at index {video_offset}"
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), (
            f"image placeholder count {image_offset} does not match images count {len(images)}"
        )
        assert video_offset == len(videos), (
            f"video placeholder count {video_offset} does not match videos count {len(videos)}"
        )
        return messages, first_image

    def maybe_filter_out_long_prompts(self, dataframe=None):
        return self.dataframe if dataframe is None else dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        raw_messages, first_image = self._build_messages(row_dict, key=self.prompt_key)

        assert isinstance(raw_messages, list) and len(raw_messages) >= 2, raw_messages
        assert raw_messages[0].get("role") == "system" and raw_messages[1].get("role") == "user", raw_messages

        row_dict["raw_prompt"] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You can call functions to assist with the user query. "
                    "Important: You must call only one function at a time."
                ),
            },
            {"role": "user", "content": raw_messages[1]["content"]},
        ]

        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        extra_info = row_dict.get("extra_info") or {}
        row_dict["extra_info"] = extra_info
        index = extra_info.get("index", 0)
        tools_kwargs = extra_info.get("tools_kwargs", {})
        if not tools_kwargs and first_image is not None:
            tools_kwargs = {"image_zoom_in_tool": {"create_kwargs": {"image": first_image}}}
        if extra_info.get("need_tools_kwargs", self.need_tools_kwargs) and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict.get("data_source"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


def _normalize_content_part(part):
    if not isinstance(part, dict):
        return part
    if part.get("type") in {"image", "image_url"} and "bytes" in part and "image" not in part:
        normalized = dict(part)
        normalized["type"] = "image"
        normalized["image"] = {"bytes": normalized.pop("bytes")}
        return normalized
    return part


def _decode_image_payload(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    assert isinstance(image, dict) and "bytes" in image, f"unexpected image payload: {type(image).__name__}"
    return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
