from pathlib import Path

from dltr.demo.streamlit_app import discover_report_files, load_end_to_end_preview


def test_discover_report_files_collects_expected_sections(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    (reports_dir / "train").mkdir(parents=True, exist_ok=True)
    (reports_dir / "eval").mkdir(parents=True, exist_ok=True)
    (reports_dir / "eda").mkdir(parents=True, exist_ok=True)
    (reports_dir / "train" / "project_training_summary.md").write_text(
        "# project\n",
        encoding="utf-8",
    )
    (reports_dir / "train" / "recognition_summary.png").write_bytes(b"png")
    (reports_dir / "eval" / "end_to_end_result.json").write_text("{}", encoding="utf-8")
    (reports_dir / "eda" / "rects_shopsign_eda.md").write_text("# eda\n", encoding="utf-8")

    files = discover_report_files(reports_dir)

    assert "train_markdown" in files
    assert "train_png" in files
    assert "extension_markdown" in files
    assert "extension_png" in files
    assert "eval_json" in files
    assert "eda_markdown" in files


def test_discover_report_files_hides_legacy_semantic_train_reports(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    train_dir = reports_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "detection_summary.md").write_text("# det\n", encoding="utf-8")
    (train_dir / "recognition_summary.md").write_text("# rec\n", encoding="utf-8")
    (train_dir / "semantic_summary.md").write_text("# semantic\n", encoding="utf-8")
    (train_dir / "recognition_summary.png").write_bytes(b"png")
    (train_dir / "semantic_summary.png").write_bytes(b"png")

    files = discover_report_files(reports_dir)

    assert [path.name for path in files["train_markdown"]] == [
        "detection_summary.md",
        "recognition_summary.md",
    ]
    assert [path.name for path in files["train_png"]] == ["recognition_summary.png"]


def test_load_end_to_end_preview_returns_empty_dict_when_missing(tmp_path: Path) -> None:
    preview = load_end_to_end_preview(tmp_path / "missing.json")

    assert preview == {}
