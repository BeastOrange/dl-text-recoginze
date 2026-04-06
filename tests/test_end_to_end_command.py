from pathlib import Path

from dltr.cli import main


def test_evaluate_end2end_image_mode_uses_pipeline(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"fake")
    detector_checkpoint = tmp_path / "det.pt"
    recognizer_checkpoint = tmp_path / "rec.pt"
    detector_checkpoint.write_bytes(b"pt")
    recognizer_checkpoint.write_bytes(b"pt")

    called = {}

    class _Artifacts:
        def __init__(self) -> None:
            self.output_dir = tmp_path / "output"
            self.json_path = self.output_dir / "result.json"
            self.markdown_path = self.output_dir / "result.md"
            self.preview_image_path = self.output_dir / "preview.png"
            self.line_results = []
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.json_path.write_text("{}", encoding="utf-8")
            self.markdown_path.write_text("# report\n", encoding="utf-8")
            self.preview_image_path.write_bytes(b"png")

    def fake_run_end_to_end_pipeline(**kwargs):
        called.update(kwargs)
        return _Artifacts()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dltr.commands.run_end_to_end_pipeline",
        fake_run_end_to_end_pipeline,
    )

    exit_code = main(
        [
            "evaluate",
            "end2end",
            "--image",
            str(image_path),
            "--detector-checkpoint",
            str(detector_checkpoint),
            "--recognizer-checkpoint",
            str(recognizer_checkpoint),
            "--output-dir",
            "output",
        ]
    )

    assert exit_code == 0
    assert called["image_path"] == image_path


def test_evaluate_end2end_image_mode_accepts_run_dirs(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"fake")
    detector_run = tmp_path / "artifacts" / "detection" / "det_a"
    recognizer_run = tmp_path / "artifacts" / "recognition" / "rec_a"
    detector_run.mkdir(parents=True, exist_ok=True)
    recognizer_run.mkdir(parents=True, exist_ok=True)
    det_best = detector_run / "checkpoints" / "best.pt"
    rec_best = recognizer_run / "best.pt"
    det_best.parent.mkdir(parents=True, exist_ok=True)
    rec_best.parent.mkdir(parents=True, exist_ok=True)
    det_best.write_bytes(b"pt")
    rec_best.write_bytes(b"pt")
    (detector_run / "training_summary.json").write_text(
        '{"best_checkpoint_path": "' + str(det_best) + '"}',
        encoding="utf-8",
    )
    (recognizer_run / "training_summary.json").write_text(
        '{"best_checkpoint_path": "' + str(rec_best) + '"}',
        encoding="utf-8",
    )

    called = {}

    class _Artifacts:
        def __init__(self) -> None:
            self.output_dir = tmp_path / "output"
            self.json_path = self.output_dir / "result.json"
            self.markdown_path = self.output_dir / "result.md"
            self.preview_image_path = self.output_dir / "preview.png"
            self.line_results = []
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.json_path.write_text("{}", encoding="utf-8")
            self.markdown_path.write_text("# report\n", encoding="utf-8")
            self.preview_image_path.write_bytes(b"png")

    def fake_run_end_to_end_pipeline(**kwargs):
        called.update(kwargs)
        return _Artifacts()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("dltr.commands.run_end_to_end_pipeline", fake_run_end_to_end_pipeline)

    exit_code = main(
        [
            "evaluate",
            "end2end",
            "--image",
            str(image_path),
            "--detector-run-dir",
            str(detector_run),
            "--recognizer-run-dir",
            str(recognizer_run),
            "--output-dir",
            "output",
        ]
    )

    assert exit_code == 0
    assert called["detector_checkpoint"] == det_best
    assert called["recognizer_checkpoint"] == rec_best


def test_evaluate_end2end_image_mode_auto_discovers_latest_runs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"fake")

    detector_run = tmp_path / "artifacts" / "detection" / "det_a" / "20250102-000000"
    recognizer_run = (
        tmp_path / "artifacts" / "checkpoints" / "recognition" / "rec_a" / "20250103-000000"
    )
    detector_run.mkdir(parents=True, exist_ok=True)
    recognizer_run.mkdir(parents=True, exist_ok=True)
    det_best = detector_run / "checkpoints" / "best.pt"
    rec_best = recognizer_run / "best.pt"
    det_best.parent.mkdir(parents=True, exist_ok=True)
    rec_best.parent.mkdir(parents=True, exist_ok=True)
    det_best.write_bytes(b"pt")
    rec_best.write_bytes(b"pt")
    (detector_run / "training_summary.json").write_text(
        '{"best_checkpoint_path": "' + str(det_best) + '"}',
        encoding="utf-8",
    )
    (recognizer_run / "training_summary.json").write_text(
        '{"best_checkpoint_path": "' + str(rec_best) + '"}',
        encoding="utf-8",
    )

    called = {}

    class _Artifacts:
        def __init__(self) -> None:
            self.output_dir = tmp_path / "output"
            self.json_path = self.output_dir / "result.json"
            self.markdown_path = self.output_dir / "result.md"
            self.preview_image_path = self.output_dir / "preview.png"
            self.line_results = []
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.json_path.write_text("{}", encoding="utf-8")
            self.markdown_path.write_text("# report\n", encoding="utf-8")
            self.preview_image_path.write_bytes(b"png")

    def fake_run_end_to_end_pipeline(**kwargs):
        called.update(kwargs)
        return _Artifacts()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("dltr.commands.run_end_to_end_pipeline", fake_run_end_to_end_pipeline)

    exit_code = main(
        [
            "evaluate",
            "end2end",
            "--image",
            str(image_path),
            "--output-dir",
            "output",
        ]
    )

    assert exit_code == 0
    assert called["detector_checkpoint"] == det_best
    assert called["recognizer_checkpoint"] == rec_best


def test_evaluate_end2end_manifest_mode_writes_summary(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    manifest_path = tmp_path / "data" / "processed" / "detection_splits" / "val.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    detector_checkpoint = tmp_path / "det.pt"
    recognizer_checkpoint = tmp_path / "rec.pt"
    detector_checkpoint.write_bytes(b"pt")
    recognizer_checkpoint.write_bytes(b"pt")

    called = {}

    def fake_evaluate_end_to_end_manifest(**kwargs):
        called.update(kwargs)
        output_dir = tmp_path / "reports" / "eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "end2end_baseline_summary.json"
        markdown_path = output_dir / "end2end_baseline_summary.md"
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("# summary\n", encoding="utf-8")
        return {"json": json_path, "markdown": markdown_path}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dltr.commands.evaluate_end_to_end_manifest",
        fake_evaluate_end_to_end_manifest,
    )

    exit_code = main(
        [
            "evaluate",
            "end2end",
            "--manifest",
            str(manifest_path),
            "--detector-checkpoint",
            str(detector_checkpoint),
            "--recognizer-checkpoint",
            str(recognizer_checkpoint),
            "--max-images",
            "5",
            "--output-dir",
            "reports/eval",
        ]
    )

    assert exit_code == 0
    assert called["manifest_path"] == manifest_path
    assert called["max_images"] == 5


def test_evaluate_end2end_manifest_sweep_mode_writes_summary(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    manifest_path = tmp_path / "data" / "processed" / "detection_splits" / "val.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    detector_checkpoint = tmp_path / "det.pt"
    recognizer_checkpoint = tmp_path / "rec.pt"
    detector_checkpoint.write_bytes(b"pt")
    recognizer_checkpoint.write_bytes(b"pt")

    called = {}

    def fake_sweep_end_to_end_manifest(**kwargs):
        called.update(kwargs)
        output_dir = tmp_path / "reports" / "eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "end2end_sweep_summary.json"
        markdown_path = output_dir / "end2end_sweep_summary.md"
        json_path.write_text("{}", encoding="utf-8")
        markdown_path.write_text("# sweep\n", encoding="utf-8")
        return {"json": json_path, "markdown": markdown_path}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dltr.commands.sweep_end_to_end_manifest",
        fake_sweep_end_to_end_manifest,
    )

    exit_code = main(
        [
            "evaluate",
            "end2end",
            "--manifest",
            str(manifest_path),
            "--detector-checkpoint",
            str(detector_checkpoint),
            "--recognizer-checkpoint",
            str(recognizer_checkpoint),
            "--sweep",
            "--output-dir",
            "reports/eval",
        ]
    )

    assert exit_code == 0
    assert called["manifest_path"] == manifest_path
