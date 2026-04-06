import json
from pathlib import Path

from dltr.cli import main


def test_train_end2end_command_writes_summary(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset_dir = tmp_path / "data" / "raw" / "rects"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = tmp_path / "data" / "processed" / "det_train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "det_val.jsonl"
    rec_train = tmp_path / "data" / "processed" / "recognition_splits" / "train.jsonl"
    rec_val = tmp_path / "data" / "processed" / "recognition_splits" / "val.jsonl"
    charset = tmp_path / "data" / "processed" / "charset.txt"
    for path in (train_manifest, val_manifest, rec_train, rec_val):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    charset.write_text("营\n业\n", encoding="utf-8")

    det_config = tmp_path / "configs" / "detection" / "end2end.yaml"
    det_config.parent.mkdir(parents=True, exist_ok=True)
    det_config.write_text(
        "\n".join(
            [
                "experiment_name: det_end2end_smoke",
                "model_name: dbnet",
                f"dataset_dir: {dataset_dir.relative_to(tmp_path)}",
                f"train_manifest: {train_manifest.relative_to(tmp_path)}",
                f"validation_manifest: {val_manifest.relative_to(tmp_path)}",
                "output_root: artifacts/detection/det_end2end_smoke",
                "epochs: 1",
                "batch_size: 1",
                "learning_rate: 0.001",
                "image_height: 32",
                "image_width: 32",
                "device: cpu",
            ]
        ),
        encoding="utf-8",
    )
    rec_config = tmp_path / "configs" / "recognition" / "end2end.yaml"
    rec_config.parent.mkdir(parents=True, exist_ok=True)
    rec_config.write_text(
        "\n".join(
            [
                "experiment_name: rec_end2end_smoke",
                "model_name: transformer",
                f"dataset_manifest: {rec_train.relative_to(tmp_path)}",
                f"validation_manifest: {rec_val.relative_to(tmp_path)}",
                f"charset_file: {charset.relative_to(tmp_path)}",
                "output_dir: artifacts/checkpoints/recognition/rec_end2end_smoke",
                "epochs: 1",
                "batch_size: 1",
                "image_height: 32",
                "image_width: 128",
                "learning_rate: 0.001",
                "device: cpu",
            ]
        ),
        encoding="utf-8",
    )

    class _DetResult:
        def __init__(self) -> None:
            self.context = type(
                "Context",
                (),
                {
                    "run_dir": tmp_path / "artifacts" / "detection" / "det_end2end_smoke" / "run-1",
                },
            )()
            self.checkpoint_path = self.context.run_dir / "checkpoints" / "last.pt"
            self.best_checkpoint_path = self.context.run_dir / "checkpoints" / "best.pt"
            self.summary_path = self.context.run_dir / "training_summary.json"
            self.context.run_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_bytes(b"pt")
            self.best_checkpoint_path.write_bytes(b"pt")
            self.summary_path.write_text(
                json.dumps(
                    {
                        "metrics": {"hmean": 0.61},
                        "best_checkpoint_path": str(self.best_checkpoint_path),
                    }
                ),
                encoding="utf-8",
            )

    class _RecResult:
        def __init__(self) -> None:
            self.run_dir = (
                tmp_path
                / "artifacts"
                / "checkpoints"
                / "recognition"
                / "rec_end2end_smoke"
                / "run-1"
            )
            self.checkpoint_path = self.run_dir / "last.pt"
            self.best_checkpoint_path = self.run_dir / "best.pt"
            self.summary_path = self.run_dir / "training_summary.json"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_bytes(b"pt")
            self.best_checkpoint_path.write_bytes(b"pt")
            self.summary_path.write_text(
                json.dumps(
                    {
                        "metrics": {"word_accuracy": 0.78, "cer": 0.22, "ned": 0.18},
                        "best_checkpoint_path": str(self.best_checkpoint_path),
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("dltr.commands.train_dbnet_detector", lambda *args, **kwargs: _DetResult())
    monkeypatch.setattr(
        "dltr.commands.train_transformer_recognizer",
        lambda *args, **kwargs: _RecResult(),
    )

    exit_code = main(
        [
            "train",
            "end2end",
            "--detector-config",
            str(det_config),
            "--recognizer-config",
            str(rec_config),
            "--run-id",
            "pipeline-smoke",
        ]
    )

    summary_path = (
        tmp_path / "artifacts" / "end2end" / "pipeline-smoke" / "training_summary.json"
    )
    assert exit_code == 0
    assert summary_path.exists()
