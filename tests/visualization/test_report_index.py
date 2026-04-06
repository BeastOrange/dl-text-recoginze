from pathlib import Path

from dltr.visualization.report_index import build_ablation_template, build_training_report_index


def test_build_training_report_index_writes_markdown(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    for name in (
        "detection_summary.md",
        "recognition_summary.md",
        "project_training_summary.md",
    ):
        (train_dir / name).write_text(f"# {name}\n", encoding="utf-8")

    output = build_training_report_index(train_reports_dir=train_dir, output_dir=train_dir)

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Training Report Index" in content
    assert "project_training_summary.md" in content


def test_build_ablation_template_writes_sections(tmp_path: Path) -> None:
    output = build_ablation_template(
        output_dir=tmp_path,
        task_name="recognition",
        experiments=["crnn_baseline", "transformer_baseline", "transformer_hardcase"],
    )

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "Ablation Report Template" in content
    assert "crnn_baseline" in content
