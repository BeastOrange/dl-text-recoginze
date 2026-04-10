from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LEGEND_WIDTH = 360
LABEL_PADDING_X = 6
LABEL_PADDING_Y = 5
LABEL_FONT_SIZE = 16
LEGEND_TITLE_FONT_SIZE = 18
LEGEND_BODY_FONT_SIZE = 13
LEGEND_ROW_HEIGHT = 42

CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
)

FALLBACK_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "service_info": (20, 120, 255),
    "advertisement": (0, 170, 255),
    "traffic_or_warning": (0, 90, 230),
    "public_notice": (70, 130, 180),
    "shop_sign": (20, 170, 120),
    "other": (90, 90, 90),
}


@dataclass(frozen=True)
class AnnotationPlacement:
    label: str
    anchor: tuple[int, int]
    box: tuple[int, int, int, int]
    color: tuple[int, int, int]


@dataclass(frozen=True)
class TextOverlay:
    text: str
    point: tuple[int, int]
    color: tuple[int, int, int]
    size: int


def build_compact_preview_label(index: int, text: str, max_chars: int = 10) -> str:
    normalized = " ".join(text.strip().split())
    compact = normalized[:max_chars]
    if len(normalized) > max_chars:
        compact = f"{compact}..."
    if not compact:
        compact = "-"
    return f"#{index} {compact}"


def layout_annotation_placements(
    image_shape: tuple[int, ...],
    line_results: list[Any],
) -> list[AnnotationPlacement]:
    image_height, image_width = image_shape[:2]
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[AnnotationPlacement] = []
    for index, item in enumerate(line_results, start=1):
        label = build_compact_preview_label(index, _renderable_text(_item_text(item)))
        color = _item_color(item)
        polygon = np.asarray(_item_polygon(item), dtype=np.int32).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(polygon)
        text_width, text_height = _measure_text(label, LABEL_FONT_SIZE)
        box_width = text_width + LABEL_PADDING_X * 2
        box_height = text_height + LABEL_PADDING_Y * 2
        candidates = [
            (x, y - box_height - 6),
            (x, y + h + 6),
            (x + w - box_width, y - box_height - 6),
            (x + w + 6, y),
            (x - box_width - 6, y),
        ]
        chosen = None
        for candidate_x, candidate_y in candidates:
            clamped_x = min(max(candidate_x, 0), max(image_width - box_width, 0))
            clamped_y = min(max(candidate_y, 0), max(image_height - box_height, 0))
            candidate_box = (clamped_x, clamped_y, box_width, box_height)
            if any(_boxes_intersect(candidate_box, existing) for existing in occupied):
                continue
            chosen = candidate_box
            break
        if chosen is None:
            fallback_y = min(
                max(y + h + 6 + len(occupied) * 4, 0),
                max(image_height - box_height, 0),
            )
            chosen = (
                min(max(x, 0), max(image_width - box_width, 0)),
                fallback_y,
                box_width,
                box_height,
            )
        occupied.append(chosen)
        placements.append(
            AnnotationPlacement(
                label=label,
                anchor=(chosen[0] + LABEL_PADDING_X, chosen[1] + LABEL_PADDING_Y),
                box=chosen,
                color=color,
            )
        )
    return placements


def render_end_to_end_preview(
    image: np.ndarray,
    line_results: list[Any],
) -> np.ndarray:
    base = _ensure_bgr(image)
    height, width = base.shape[:2]
    canvas = np.full((height, width + LEGEND_WIDTH, 3), 247, dtype=np.uint8)
    canvas[:, :width] = base
    cv2.rectangle(canvas, (width, 0), (width + LEGEND_WIDTH, height), (242, 244, 247), thickness=-1)
    cv2.line(canvas, (width, 0), (width, height), (220, 224, 228), thickness=2)

    text_overlays: list[TextOverlay] = []
    placements = layout_annotation_placements(base.shape, line_results)
    for item, placement in zip(line_results, placements, strict=True):
        polygon = np.asarray(_item_polygon(item), dtype=np.int32).reshape(-1, 2)
        cv2.polylines(
            canvas[:, :width],
            [polygon],
            isClosed=True,
            color=placement.color,
            thickness=2,
        )
        text_overlays.append(_draw_label(canvas[:, :width], placement))

    text_overlays.extend(_draw_legend(canvas, width, line_results))
    _draw_text_overlays(canvas, text_overlays)
    return canvas


def _draw_label(image: np.ndarray, placement: AnnotationPlacement) -> TextOverlay:
    x, y, w, h = placement.box
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), placement.color, thickness=-1)
    cv2.addWeighted(overlay, 0.88, image, 0.12, 0.0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), thickness=1)
    return TextOverlay(
        text=placement.label,
        point=placement.anchor,
        color=(255, 255, 255),
        size=LABEL_FONT_SIZE,
    )


def _draw_legend(canvas: np.ndarray, offset_x: int, line_results: list[Any]) -> list[TextOverlay]:
    x = offset_x + 18
    overlays = [
        TextOverlay("OCR Preview", (x, 14), (40, 40, 40), LEGEND_TITLE_FONT_SIZE),
        TextOverlay(
            f"Lines: {len(line_results)}",
            (x, 40),
            (90, 90, 90),
            LEGEND_BODY_FONT_SIZE,
        ),
    ]
    row_top = 78
    for index, item in enumerate(line_results, start=1):
        top = row_top + (index - 1) * LEGEND_ROW_HEIGHT
        if top + LEGEND_ROW_HEIGHT > canvas.shape[0] - 12:
            remaining = len(line_results) - index + 1
            overlays.append(
                TextOverlay(
                    f"... {remaining} more",
                    (x, canvas.shape[0] - 24),
                    (90, 90, 90),
                    LEGEND_BODY_FONT_SIZE,
                )
            )
            break
        color = _item_color(item)
        cv2.rectangle(canvas, (x, top + 2), (x + 10, top + 12), color, thickness=-1)
        title = f"#{index} {_item_label(item)} {_item_confidence(item):.2f}"
        detail = _truncate_text(_renderable_text(_item_text(item)), max_chars=28)
        if getattr(item, "second_pass_applied", False):
            detail = f"{detail} [2nd]"
        overlays.append(TextOverlay(title, (x + 18, top), (32, 32, 32), LEGEND_BODY_FONT_SIZE))
        overlays.append(
            TextOverlay(
                detail,
                (x + 18, top + 18),
                (80, 80, 80),
                LEGEND_BODY_FONT_SIZE,
            )
        )
    return overlays


def _draw_text_overlays(image: np.ndarray, overlays: list[TextOverlay]) -> None:
    if not overlays:
        return
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    for overlay in overlays:
        draw.text(
            overlay.point,
            overlay.text,
            font=_load_font(overlay.size),
            fill=_bgr_to_rgb(overlay.color),
        )
    image[:, :] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_chars:
        return normalized or "-"
    return f"{normalized[:max_chars]}..."


def _item_polygon(item: Any) -> list[int]:
    return list(item.polygon)


def _item_text(item: Any) -> str:
    return str(getattr(item, "text", "")).strip()


def _item_label(item: Any) -> str:
    return str(getattr(item, "analysis_label", "other")).strip() or "other"


def _item_confidence(item: Any) -> float:
    return float(getattr(item, "recognition_confidence", 0.0))


def _item_color(item: Any) -> tuple[int, int, int]:
    return LABEL_COLORS.get(_item_label(item), LABEL_COLORS["other"])


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _boxes_intersect(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return not (
        lx + lw <= rx
        or rx + rw <= lx
        or ly + lh <= ry
        or ry + rh <= ly
    )


def _renderable_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return "-"
    if _contains_non_ascii(normalized) and not _has_cjk_font():
        return "text"
    return normalized


def _contains_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def _measure_text(text: str, font_size: int) -> tuple[int, int]:
    left, top, right, bottom = _load_font(font_size).getbbox(text)
    width = max(1, right - left)
    height = max(1, bottom - top)
    return width, height


@lru_cache(maxsize=1)
def _has_cjk_font() -> bool:
    return _resolve_cjk_font_path() is not None


@lru_cache(maxsize=1)
def _resolve_cjk_font_path() -> Path | None:
    return _find_first_existing_font(CJK_FONT_CANDIDATES)


@lru_cache(maxsize=1)
def _resolve_font_path() -> Path | None:
    cjk_path = _resolve_cjk_font_path()
    if cjk_path is not None:
        return cjk_path
    return _find_first_existing_font(FALLBACK_FONT_CANDIDATES)


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.ImageFont:
    font_path = _resolve_font_path()
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _find_first_existing_font(candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    return None


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = color
    return red, green, blue
