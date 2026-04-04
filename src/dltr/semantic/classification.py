from __future__ import annotations

from dataclasses import dataclass

from dltr.semantic.slots import extract_semantic_slots


@dataclass(frozen=True)
class SemanticClassification:
    semantic_class: str
    confidence: float
    rationale: str


def classify_scene_text(text: str) -> SemanticClassification:
    normalized = text.strip()
    slots = extract_semantic_slots(normalized)

    if slots.warning_terms:
        return SemanticClassification("traffic_or_warning", 0.92, "warning_terms")
    if slots.phone or slots.price or slots.time:
        return SemanticClassification("service_info", 0.86, "service_slots")
    if any(keyword in normalized for keyword in ("促销", "特价", "优惠", "折", "大促", "开业")):
        return SemanticClassification("advertisement", 0.82, "promotion_keywords")
    if any(keyword in normalized for keyword in ("公告", "通知", "须知", "通告", "提示")):
        return SemanticClassification("public_notice", 0.8, "notice_keywords")
    if any(
        keyword in normalized
        for keyword in ("店", "馆", "酒店", "餐厅", "服饰", "超市", "商场")
    ):
        return SemanticClassification("shop_sign", 0.76, "shop_keywords")
    return SemanticClassification("other", 0.55, "fallback")
