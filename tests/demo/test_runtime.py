from pathlib import Path

from dltr.demo.runtime import resolve_demo_checkpoints, run_uploaded_inference


def test_resolve_demo_checkpoints_prefers_latest_runs(tmp_path: Path) -> None:
    detector_run = tmp_path / "artifacts" / "detection" / "det_exp" / "20260101-000000"
    recognizer_run = (
        tmp_path / "artifacts" / "checkpoints" / "recognition" / "rec_exp" / "20260102-000000"
    )
    detector_run.mkdir(parents=True)
    recognizer_run.mkdir(parents=True)
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

    checkpoints = resolve_demo_checkpoints(project_root=tmp_path)

    assert checkpoints["detector"] == det_best
    assert checkpoints["recognizer"] == rec_best


def test_run_uploaded_inference_saves_temp_image(monkeypatch, tmp_path: Path) -> None:
    image_bytes = b"fake-bytes"
    called = {}

    class _Artifacts:
        def __init__(self) -> None:
            self.output_dir = tmp_path / "reports" / "eval"
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

    monkeypatch.setattr("dltr.demo.runtime.run_end_to_end_pipeline", fake_run_end_to_end_pipeline)

    artifacts = run_uploaded_inference(
        image_bytes=image_bytes,
        project_root=tmp_path,
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
    )

    assert called["image_path"].exists()
    assert artifacts.json_path.exists()
