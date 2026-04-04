from __future__ import annotations

SEMANTIC_CLASSES: tuple[str, ...] = (
    "shop_sign",
    "advertisement",
    "public_notice",
    "traffic_or_warning",
    "service_info",
    "other",
)


def validate_semantic_class(label: str) -> str:
    normalized = label.strip()
    if normalized not in SEMANTIC_CLASSES:
        raise ValueError(f"Unsupported semantic class: {label}")
    return normalized
