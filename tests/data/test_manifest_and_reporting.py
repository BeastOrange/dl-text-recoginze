import json
from pathlib import Path

from dltr.data.config import build_default_data_config
from dltr.data.hardcase import analyze_hardcase_metadata
from dltr.data.inventory import collect_inventories
from dltr.data.manifest import build_recognition_manifest
from dltr.data.reporting import write_eda_markdown_report
from dltr.data.validation import validate_dataset_paths
from dltr.project import ProjectPaths


def test_build_recognition_manifest_emits_jsonl(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "sample.jpg"
    label_path = dataset_root / "sample.txt"
    dataset_root.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    label_path.write_text("x1,y1,x2,y2,中文文本", encoding="utf-8")

    output_path = tmp_path / "manifest.jsonl"
    result = build_recognition_manifest(
        dataset_name="rctw17",
        dataset_root=dataset_root,
        output_path=output_path,
        image_extensions={".jpg"},
        label_extensions={".txt"},
    )

    assert result.emitted_rows == 1
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["dataset"] == "rctw17"
    assert payload["text"] == "中文文本"


def test_write_eda_markdown_report_generates_english_sections(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    (paths.data_raw / "rctw17").mkdir(parents=True, exist_ok=True)
    image_path = paths.data_raw / "rctw17" / "small_rotate_blur.jpg"
    label_path = image_path.with_suffix(".txt")
    image_path.write_bytes(b"img")
    label_path.write_text("text", encoding="utf-8")

    config = build_default_data_config(paths)
    validation = validate_dataset_paths(paths, config)
    inventories = collect_inventories(paths, config)
    hardcases = {
        name: analyze_hardcase_metadata(inventory) for name, inventory in inventories.items()
    }

    report_path = write_eda_markdown_report(
        project_paths=paths,
        config=config,
        validation=validation,
        inventories=inventories,
        hardcases=hardcases,
        filename="summary.md",
    )

    content = report_path.read_text(encoding="utf-8")
    assert "Chinese Scene-Text Dataset EDA Summary" in content
    assert "Validation Status" in content
    assert "Hard-Case Heuristic Snapshot" in content


def test_build_recognition_manifest_supports_rects_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rects"
    image_path = dataset_root / "img" / "train_ReCTS_000002.jpg"
    label_path = dataset_root / "gt" / "train_ReCTS_000002.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    label_path.write_text('{"text": "营业时间"}', encoding="utf-8")

    output_path = tmp_path / "rects_manifest.jsonl"
    result = build_recognition_manifest(
        dataset_name="rects",
        dataset_root=dataset_root,
        output_path=output_path,
        image_extensions={".jpg"},
        label_extensions={".json"},
    )

    assert result.emitted_rows == 1
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["text"] == "营业时间"
