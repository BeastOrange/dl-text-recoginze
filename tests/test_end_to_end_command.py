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
