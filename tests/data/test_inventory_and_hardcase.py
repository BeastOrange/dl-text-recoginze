from pathlib import Path

from dltr.data.hardcase import analyze_hardcase_metadata
from dltr.data.inventory import scan_dataset_inventory


def test_scan_dataset_inventory_collects_counts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rctw17"
    image_a = dataset_root / "scene_small_rotate" / "sample_1.jpg"
    image_b = dataset_root / "scene_normal" / "sample_2.png"
    label_a = image_a.with_suffix(".txt")
    image_a.parent.mkdir(parents=True, exist_ok=True)
    image_b.parent.mkdir(parents=True, exist_ok=True)

    image_a.write_bytes(b"img")
    image_b.write_bytes(b"img")
    label_a.write_text("hello", encoding="utf-8")

    inventory = scan_dataset_inventory(
        dataset_name="rctw17",
        dataset_root=dataset_root,
        image_extensions={".jpg", ".png"},
        label_extensions={".txt"},
    )

    assert inventory.total_images == 2
    assert inventory.total_labels == 1
    assert inventory.matched_label_images == 1
    assert inventory.missing_label_images == 1


def test_analyze_hardcase_metadata_uses_path_keywords(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rects"
    image_path = dataset_root / "dense_vertical_blur" / "very_long_filename_sample_for_testing.jpg"
    label_path = image_path.with_suffix(".txt")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    label_path.write_text("text", encoding="utf-8")

    inventory = scan_dataset_inventory(
        dataset_name="rects",
        dataset_root=dataset_root,
        image_extensions={".jpg"},
        label_extensions={".txt"},
    )
    metadata = analyze_hardcase_metadata(inventory)

    assert metadata.total_images == 1
    assert metadata.keyword_hit_counts["dense_text_like"] == 1
    assert metadata.keyword_hit_counts["vertical_text_like"] == 1
    assert metadata.keyword_hit_counts["low_quality_like"] == 1


def test_scan_dataset_inventory_matches_rects_img_and_gt_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rects"
    image_path = dataset_root / "img" / "train_ReCTS_000001.jpg"
    label_path = dataset_root / "gt" / "train_ReCTS_000001.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    label_path.write_text('{"text": "中文招牌"}', encoding="utf-8")

    inventory = scan_dataset_inventory(
        dataset_name="rects",
        dataset_root=dataset_root,
        image_extensions={".jpg"},
        label_extensions={".json"},
    )

    assert inventory.total_images == 1
    assert inventory.total_labels == 1
    assert inventory.matched_label_images == 1
