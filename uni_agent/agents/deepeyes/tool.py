"""Multimodal crop tool used by the DeepEyes agent."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import io
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from uni_agent.tools import Tool, ToolResult


class ImageZoomInArgs(BaseModel):
    """Arguments exposed to the policy model."""

    bbox_2d: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Normalized 0-1000 coordinates [x1, y1, x2, y2].",
    )
    label: str | None = Field(default=None, description="Optional name of the inspected object or region.")

    model_config = ConfigDict(extra="forbid")


class ImageZoomInConfig(BaseModel):
    """Construction parameters that do not vary between calls."""

    min_dimension: int = Field(default=28, gt=0)
    max_aspect_ratio: float = Field(default=100.0, gt=1.0)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0.0)

    model_config = ConfigDict(extra="forbid")


@dataclasses.dataclass
class ImageZoomInResult(ToolResult):
    """Tool result with image payloads retained for the recipe Agent adapter."""

    images: list[Image.Image] = dataclasses.field(default_factory=list)
    info: dict[str, Any] = dataclasses.field(default_factory=dict)


class ImageZoomInTool(Tool):
    """Crop a region from the source image while keeping Uni-Agent's Tool contract."""

    name = "image_zoom_in_tool"
    description = "Zoom in on a region of the source image. bbox_2d uses Qwen's normalized 0-1000 coordinates."
    args_model = ImageZoomInArgs
    config_model = ImageZoomInConfig

    def __init__(self, sandbox, *, image: Any, **kwargs: Any) -> None:
        super().__init__(sandbox, **kwargs)
        cfg: ImageZoomInConfig = self.config  # type: ignore[assignment]
        self._image = coerce_image(image, timeout=cfg.fetch_timeout_seconds)

    async def run(self, args: dict[str, Any], *, timeout: float | None = None) -> ToolResult:
        cfg: ImageZoomInConfig = self.config  # type: ignore[assignment]
        try:
            bbox = processor_safe_bbox(
                args.get("bbox_2d"),
                image_size=self._image.size,
                min_dimension=cfg.min_dimension,
                max_aspect_ratio=cfg.max_aspect_ratio,
            )
        except ValueError as error:
            return ImageZoomInResult(
                text=f"Error: Could not zoom in: {error}",
                status="error",
                info={"success": False, "error": "invalid_bbox"},
            )

        crop = self._image.crop(bbox)
        label = args.get("label")
        label_text = f" with label {label}" if label else ""
        return ImageZoomInResult(
            text=f"Zoomed in on the image to the region {list(bbox)}{label_text}.",
            images=[crop],
            info={"success": True, "bbox": bbox, "crop_size": crop.size},
        )


def coerce_image(value: Any, *, timeout: float) -> Image.Image:
    """Decode a supported image payload into RGB pixels."""

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes | bytearray):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value:
            return coerce_image(value["bytes"], timeout=timeout)
        if "image" in value:
            return coerce_image(value["image"], timeout=timeout)
        if value.get("path"):
            return coerce_image(value["path"], timeout=timeout)
        image_url = value.get("image_url")
        if isinstance(image_url, dict) and "url" in image_url:
            return coerce_image(image_url["url"], timeout=timeout)
    if isinstance(value, str):
        if value.startswith("data:image"):
            try:
                _, encoded = value.split(",", 1)
                return coerce_image(base64.b64decode(encoded), timeout=timeout)
            except (ValueError, binascii.Error) as error:
                raise ValueError("invalid image data URL") from error
        if value.startswith(("http://", "https://")):
            import requests

            response = requests.get(value, timeout=timeout)
            response.raise_for_status()
            return coerce_image(response.content, timeout=timeout)
        parsed = urlparse(value)
        path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(value)
        if path.is_file():
            return Image.open(path).convert("RGB")
        raise ValueError(f"image path does not exist: {value}")
    raise TypeError(f"unsupported image payload: {type(value).__name__}")


def processor_safe_bbox(
    value: Any,
    *,
    image_size: tuple[int, int],
    min_dimension: int,
    max_aspect_ratio: float,
) -> tuple[int, int, int, int]:
    """Convert normalized coordinates into a processor-safe pixel crop."""

    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("bbox_2d must contain four normalized coordinates")
    try:
        left, top, right, bottom = (float(coord) for coord in value)
    except (TypeError, ValueError) as error:
        raise ValueError("bbox_2d coordinates must be numeric") from error
    if not all(math.isfinite(coord) for coord in (left, top, right, bottom)):
        raise ValueError("bbox_2d coordinates must be finite")
    if left >= right or top >= bottom:
        raise ValueError("bbox_2d must satisfy x1 < x2 and y1 < y2")

    image_width, image_height = image_size
    left = left / 1000.0 * image_width
    right = right / 1000.0 * image_width
    top = top / 1000.0 * image_height
    bottom = bottom / 1000.0 * image_height
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(image_width), right)
    bottom = min(float(image_height), bottom)
    if left >= right or top >= bottom:
        raise ValueError(f"bbox_2d does not overlap image bounds {image_size}")

    width = right - left
    height = bottom - top
    if max(width, height) / min(width, height) > max_aspect_ratio:
        raise ValueError(f"bbox_2d aspect ratio exceeds {max_aspect_ratio:g}")

    target_width = min(float(image_width), max(width, float(min_dimension)))
    target_height = min(float(image_height), max(height, float(min_dimension)))
    if target_width < min_dimension or target_height < min_dimension:
        raise ValueError(f"image {image_size} cannot provide the minimum crop size {min_dimension}x{min_dimension}")

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    left = min(max(0.0, center_x - target_width / 2.0), image_width - target_width)
    top = min(max(0.0, center_y - target_height / 2.0), image_height - target_height)
    right = left + target_width
    bottom = top + target_height

    bbox = (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(image_width, math.ceil(right)),
        min(image_height, math.ceil(bottom)),
    )
    if bbox[2] - bbox[0] < min_dimension or bbox[3] - bbox[1] < min_dimension:
        raise ValueError(f"processed crop is smaller than {min_dimension}x{min_dimension}")
    return bbox
