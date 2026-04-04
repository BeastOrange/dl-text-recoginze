from __future__ import annotations

from dltr.data.types import DatasetInventory, HardCaseMetadata

_KEYWORDS = {
    "small_text_like": ("small", "tiny", "micro"),
    "dense_text_like": ("dense", "crowded", "cluster"),
    "rotated_text_like": ("rotate", "tilt", "skew", "angle"),
    "vertical_text_like": ("vertical", "vert", "vtext"),
    "occluded_text_like": ("occl", "遮挡"),
    "low_quality_like": ("blur", "noise", "dark", "glare", "reflection", "lowres"),
    "artistic_text_like": ("art", "style", "handwriting"),
}


def analyze_hardcase_metadata(inventory: DatasetInventory) -> HardCaseMetadata:
    total = inventory.total_images
    hit_counts = {name: 0 for name in _KEYWORDS}
    deep_paths = 0
    long_names = 0

    for relpath in inventory.image_relpaths:
        lowered = relpath.lower()
        for name, keywords in _KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                hit_counts[name] += 1
        if relpath.count("/") >= 3:
            deep_paths += 1
        filename = relpath.rsplit("/", maxsplit=1)[-1]
        if len(filename) >= 28:
            long_names += 1

    ratio = {key: _safe_ratio(value, total) for key, value in hit_counts.items()}
    deep_path_ratio = _safe_ratio(deep_paths, total)
    long_name_ratio = _safe_ratio(long_names, total)

    recommendations: list[str] = []
    if total == 0:
        recommendations.append("No images found. Verify dataset paths before EDA.")
    if ratio["small_text_like"] < 0.1 and total > 0:
        recommendations.append(
            "Small-text-like samples appear low. Consider targeted augmentation."
        )
    if ratio["rotated_text_like"] < 0.1 and total > 0:
        recommendations.append(
            "Rotated-text-like samples appear low. Add affine/rotation augmentation."
        )
    if ratio["low_quality_like"] < 0.05 and total > 0:
        recommendations.append("Low-quality-like samples appear low. Add blur/noise simulation.")
    if not recommendations:
        recommendations.append("Current metadata indicates balanced hard-case keyword coverage.")

    return HardCaseMetadata(
        dataset_name=inventory.dataset_name,
        total_images=total,
        keyword_hit_counts=hit_counts,
        keyword_hit_ratio=ratio,
        deep_path_ratio=deep_path_ratio,
        long_name_ratio=long_name_ratio,
        recommendations=recommendations,
    )


def _safe_ratio(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return value / total
