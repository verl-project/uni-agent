"""Runtime dataset adapter for DeepEyes multimodal training."""

from __future__ import annotations

import base64
import binascii
import copy
import io
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from uni_agent.agents.deepeyes import messages_for_gateway
from verl.utils.dataset.rl_dataset import RLHFDataset


class DeepEyesDataset(RLHFDataset):
    """Build serializable OpenAI messages and one TaskConfig per sample."""

    def maybe_filter_out_long_prompts(self, dataframe=None):
        # The effective length includes tool schemas and incremental multimodal
        # turns, so it can only be checked at the Gateway boundary.
        return self.dataframe if dataframe is None else dataframe

    def __getitem__(self, item):
        row = dict(self.dataframe[item])
        messages, source_image = self._build_messages(row, key=self.prompt_key)
        user_messages = [message for message in messages if message.get("role") == "user"]
        if not user_messages:
            raise ValueError("DeepEyes samples require at least one user message")
        if source_image is None:
            raise ValueError("DeepEyes samples require an image")

        raw_prompt = messages_for_gateway([{"role": "user", "content": user_messages[0]["content"]}])

        extra_info = row.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            raise TypeError("extra_info must be a mapping")
        extra_info = dict(extra_info)
        question = extra_info.get("question") or _content_text(user_messages[0].get("content"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError("DeepEyes samples require extra_info.question or user text")
        extra_info["question"] = question.strip()

        reward_model = row.get("reward_model") or {}
        if not isinstance(reward_model, dict):
            raise TypeError("reward_model must be a mapping")
        reward_model = dict(reward_model)
        if reward_model.get("ground_truth") is None:
            fallback_answer = row.get("answer", extra_info.get("answer"))
            if fallback_answer is None:
                raise ValueError("DeepEyes samples require reward_model.ground_truth")
            reward_model["ground_truth"] = fallback_answer

        data_source = row.get("data_source", "deepeyes")
        sample_index = extra_info.get("index", item)
        task_config = {
            "name": "deepeyes",
            "question": extra_info["question"],
            "ground_truth": str(reward_model["ground_truth"]),
            "metadata": {"index": sample_index},
        }

        row.pop(self.image_key, None)
        row.pop(self.video_key, None)
        row.pop(getattr(self, "audio_key", "audios"), None)
        row[self.prompt_key] = raw_prompt
        row["raw_prompt"] = raw_prompt
        row["extra_info"] = extra_info
        row["reward_model"] = reward_model
        row["data_source"] = data_source
        row["index"] = sample_index
        row["tools_kwargs"] = {"task": task_config}
        # TensorDict batches still require at least one tensor-valued field.
        row["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        return row

    def _build_messages(self, example: dict[str, Any], *, key: str) -> tuple[list[dict[str, Any]], Image.Image | None]:
        messages = copy.deepcopy(example[key])
        if not isinstance(messages, list):
            raise TypeError(f"{key} must be a list of chat messages")

        raw_images = example.get(self.image_key)
        images = (
            list(raw_images) if isinstance(raw_images, list | tuple) else ([] if raw_images is None else [raw_images])
        )
        image_offset = 0
        first_image: Image.Image | None = None

        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("chat messages must be mappings")
            content = message.get("content")
            if isinstance(content, list):
                normalized_parts = []
                for part in content:
                    column_image = None
                    if _is_image_part(part) and _image_payload(part) is None:
                        if image_offset >= len(images):
                            raise ValueError("image content-part count exceeds the images column")
                        column_image = images[image_offset]
                        image_offset += 1
                    normalized_part, part_image = _normalize_content_part(part, fallback_image=column_image)
                    if first_image is None and part_image is not None:
                        first_image = part_image
                    normalized_parts.append(normalized_part)
                message["content"] = normalized_parts
                continue
            if not isinstance(content, str) or "<image>" not in content:
                continue

            content_parts = []
            for segment in (segment for segment in re.split("(<image>)", content) if segment):
                if segment == "<image>":
                    if image_offset >= len(images):
                        raise ValueError("image placeholder count exceeds the images column")
                    image = _decode_image(images[image_offset])
                    first_image = first_image or image
                    content_parts.append({"type": "image", "image": image})
                    image_offset += 1
                else:
                    content_parts.append({"type": "text", "text": segment})
            message["content"] = content_parts

        if image_offset != len(images):
            raise ValueError(f"image placeholder count {image_offset} does not match images count {len(images)}")
        return messages, first_image


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and (
        part.get("type") in {"image", "image_url"} or "image" in part or "image_url" in part or "bytes" in part
    )


def _image_payload(part: dict[str, Any]) -> Any:
    if "image" in part:
        return part["image"]
    if "bytes" in part:
        return {"bytes": part["bytes"]}
    image_url = part.get("image_url")
    return image_url.get("url") if isinstance(image_url, dict) else image_url


def _normalize_content_part(part: Any, *, fallback_image: Any = None) -> tuple[Any, Image.Image | None]:
    if not isinstance(part, dict):
        return part, None
    if not _is_image_part(part):
        return dict(part), None
    payload = _image_payload(part)
    if payload is None:
        payload = fallback_image
    if payload is None:
        raise ValueError("image content part has no payload and the images column is empty")
    image = _decode_image(payload)
    return {"type": "image", "image": image}, image


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    elif isinstance(value, bytes | bytearray):
        image = Image.open(io.BytesIO(bytes(value))).convert("RGB")
    elif isinstance(value, dict) and "bytes" in value:
        image = _decode_image(value["bytes"])
    elif isinstance(value, dict) and "image" in value:
        image = _decode_image(value["image"])
    elif isinstance(value, dict) and value.get("path"):
        image = _decode_image(value["path"])
    elif isinstance(value, str) and value.startswith("data:image"):
        try:
            _, encoded = value.split(",", 1)
            image = _decode_image(base64.b64decode(encoded))
        except (ValueError, binascii.Error) as error:
            raise ValueError("invalid image data URL") from error
    elif isinstance(value, str) and Path(value).is_file():
        image = Image.open(value).convert("RGB")
    else:
        raise TypeError(f"unsupported DeepEyes image payload: {type(value).__name__}")
    return _upscale_small_image(image, min_dimension=28)


def _upscale_small_image(image: Image.Image, *, min_dimension: int) -> Image.Image:
    width, height = image.size
    if width >= min_dimension and height >= min_dimension:
        return image
    scale = max(min_dimension / width, min_dimension / height)
    target = (max(min_dimension, round(width * scale)), max(min_dimension, round(height * scale)))
    return image.resize(target, Image.Resampling.BICUBIC)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return " ".join(
        part.get("text", "").strip()
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part.get("text", "").strip()
    )
