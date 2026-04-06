from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RecognitionPreprocessConfig:
    target_height: int
    target_width: int
    preserve_aspect_ratio: bool = True
    rotate_vertical_text: bool = True
    vertical_aspect_threshold: float = 1.2
    padding_value: int = 255

    def validate(self) -> None:
        if self.target_height <= 0 or self.target_width <= 0:
            raise ValueError("preprocess target size must be > 0")
        if self.vertical_aspect_threshold <= 0:
            raise ValueError("preprocess.vertical_aspect_threshold must be > 0")
        if not 0 <= self.padding_value <= 255:
            raise ValueError("preprocess.padding_value must be in [0, 255]")


@dataclass(frozen=True)
class RecognitionPreprocessMeta:
    original_height: int
    original_width: int
    rotated_clockwise: bool
    content_height: int
    content_width: int
    offset_y: int
    offset_x: int


def prepare_recognition_image(
    image: np.ndarray,
    *,
    config: RecognitionPreprocessConfig,
) -> tuple[np.ndarray, RecognitionPreprocessMeta]:
    config.validate()
    gray = _ensure_grayscale(image)
    original_height, original_width = gray.shape[:2]
    rotated_clockwise = False
    working = gray

    if _should_rotate_vertical(gray, config=config):
        working = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        rotated_clockwise = True

    if config.preserve_aspect_ratio:
        processed, content_height, content_width, offset_y, offset_x = _resize_with_padding(
            working,
            config=config,
        )
    else:
        resized = cv2.resize(working, (config.target_width, config.target_height))
        processed = resized.astype(np.float32) / 255.0
        content_height = config.target_height
        content_width = config.target_width
        offset_y = 0
        offset_x = 0

    return processed, RecognitionPreprocessMeta(
        original_height=original_height,
        original_width=original_width,
        rotated_clockwise=rotated_clockwise,
        content_height=content_height,
        content_width=content_width,
        offset_y=offset_y,
        offset_x=offset_x,
    )


def _should_rotate_vertical(
    image: np.ndarray,
    *,
    config: RecognitionPreprocessConfig,
) -> bool:
    if not config.rotate_vertical_text:
        return False
    height, width = image.shape[:2]
    return (height / max(width, 1)) >= config.vertical_aspect_threshold


def _resize_with_padding(
    image: np.ndarray,
    *,
    config: RecognitionPreprocessConfig,
) -> tuple[np.ndarray, int, int, int, int]:
    height, width = image.shape[:2]
    height_scale = config.target_height / max(height, 1)
    width_scale = config.target_width / max(width, 1)
    scale = min(height_scale, width_scale)
    content_width = max(1, min(config.target_width, int(round(width * scale))))
    content_height = max(1, min(config.target_height, int(round(height * scale))))
    resized = cv2.resize(image, (content_width, content_height))

    canvas = np.full(
        (config.target_height, config.target_width),
        config.padding_value,
        dtype=np.uint8,
    )
    offset_y = max((config.target_height - content_height) // 2, 0)
    offset_x = 0
    canvas[offset_y : offset_y + content_height, offset_x : offset_x + content_width] = resized
    return canvas.astype(np.float32) / 255.0, content_height, content_width, offset_y, offset_x


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported recognition image rank: {image.ndim}")
