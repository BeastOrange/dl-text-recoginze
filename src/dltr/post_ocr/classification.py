from __future__ import annotations

from dataclasses import dataclass

from dltr.post_ocr.slots import extract_post_ocr_slots

ANALYSIS_LABELS: tuple[str, ...] = (
    "shop_sign",
    "advertisement",
    "public_notice",
    "traffic_or_warning",
    "service_info",
    "other",
)


def validate_analysis_label(label: str) -> str:
    normalized = label.strip()
    if normalized not in ANALYSIS_LABELS:
        raise ValueError(f"Unsupported analysis label: {label}")
    return normalized


@dataclass(frozen=True)
class PostOCRClassification:
    label: str
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        validate_analysis_label(self.label)


def analyze_scene_text(text: str) -> PostOCRClassification:
    normalized = text.strip()
    slots = extract_post_ocr_slots(normalized)

    if slots.warning_terms:
        return PostOCRClassification("traffic_or_warning", 0.92, "warning_terms")
    if slots.phone or slots.price or slots.time:
        return PostOCRClassification("service_info", 0.86, "service_slots")
    if any(keyword in normalized for keyword in ("促销", "特价", "优惠", "折", "大促", "开业")):
        return PostOCRClassification("advertisement", 0.82, "promotion_keywords")
    if any(keyword in normalized for keyword in ("公告", "通知", "须知", "通告", "提示")):
        return PostOCRClassification("public_notice", 0.8, "notice_keywords")
    if any(
        keyword in normalized
        for keyword in ("店", "馆", "酒店", "餐厅", "服饰", "超市", "商场")
    ):
        return PostOCRClassification("shop_sign", 0.76, "shop_keywords")
    return PostOCRClassification("other", 0.55, "fallback")
