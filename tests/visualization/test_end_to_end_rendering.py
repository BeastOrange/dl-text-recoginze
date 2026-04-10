from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dltr.visualization.end_to_end_rendering import (
    build_compact_preview_label,
    layout_annotation_placements,
    render_end_to_end_preview,
)


@dataclass(frozen=True)
class _Item:
    line_id: str
    polygon: list[int]
    text: str
    analysis_label: str
    recognition_confidence: float
    second_pass_applied: bool = False


def test_build_compact_preview_label_truncates_long_text() -> None:
    label = build_compact_preview_label(3, "这是一段非常长的文本内容", max_chars=6)

    assert label.startswith("#3 ")
    assert label.endswith("...")


def test_layout_annotation_placements_avoids_overlap_for_dense_boxes() -> None:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    items = [
        _Item("line-0", [10, 20, 90, 20, 90, 50, 10, 50], "营业时间", "service_info", 0.9),
        _Item("line-1", [20, 28, 100, 28, 100, 58, 20, 58], "电话", "service_info", 0.8),
    ]

    placements = layout_annotation_placements(image.shape, items)

    assert len(placements) == 2
    first = placements[0].box
    second = placements[1].box
    assert not _boxes_intersect(first, second)


def test_render_end_to_end_preview_adds_legend_panel_and_annotations() -> None:
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    items = [
        _Item(
            "line-0",
            [10, 20, 90, 20, 90, 50, 10, 50],
            "营业时间 09:00-21:00",
            "service_info",
            0.9,
        ),
        _Item(
            "line-1",
            [20, 60, 120, 60, 120, 95, 20, 95],
            "新品促销 买一送一",
            "advertisement",
            0.8,
            True,
        ),
    ]

    preview = render_end_to_end_preview(image, items)

    assert preview.shape[0] == image.shape[0]
    assert preview.shape[1] > image.shape[1]
    assert np.any(preview[:, image.shape[1] :, :] != 255)
    assert np.any(preview[:, : image.shape[1], :] != 255)


def _boxes_intersect(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return not (
        lx + lw <= rx
        or rx + rw <= lx
        or ly + lh <= ry
        or ry + rh <= ly
    )
