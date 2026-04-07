import json
from dataclasses import dataclass
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

    @dataclass(frozen=True)
    class _JointResult:
        run_dir: Path
        checkpoint_path: Path
        summary_path: Path
        detector_proxy_summary_path: Path
        recognizer_proxy_summary_path: Path

    def fake_train_end2end_system(*args, **kwargs):  # noqa: ANN002, ANN003
        run_dir = tmp_path / "artifacts" / "end2end" / "pipeline-smoke"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "best.pt"
        checkpoint_path.write_bytes(b"pt")
        summary_path = run_dir / "training_summary.json"
        detector_proxy = (
            tmp_path
            / "artifacts"
            / "detection"
            / "det_end2end_smoke_multitask"
            / "pipeline-smoke"
            / "training_summary.json"
        )
        recognizer_proxy = (
            tmp_path
            / "artifacts"
            / "checkpoints"
            / "recognition"
            / "rec_end2end_smoke_multitask"
            / "pipeline-smoke"
            / "training_summary.json"
        )
        detector_proxy.parent.mkdir(parents=True, exist_ok=True)
        recognizer_proxy.parent.mkdir(parents=True, exist_ok=True)
        detector_proxy.write_text(
            json.dumps(
                {
                    "run_id": "pipeline-smoke",
                    "metrics": {"hmean": 0.61, "precision": 0.63, "recall": 0.6},
                    "best_checkpoint_path": str(checkpoint_path),
                }
            ),
            encoding="utf-8",
        )
        recognizer_proxy.write_text(
            json.dumps(
                {
                    "run_id": "pipeline-smoke",
                    "metrics": {"word_accuracy": 0.78, "cer": 0.22, "ned": 0.18},
                    "best_checkpoint_path": str(checkpoint_path),
                }
            ),
            encoding="utf-8",
        )
        return _JointResult(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            detector_proxy_summary_path=detector_proxy,
            recognizer_proxy_summary_path=recognizer_proxy,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dltr.commands.train_end2end_multitask_system",
        fake_train_end2end_system,
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
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["coordination_strategy"] == "shared_backbone_multitask_training"
    assert payload["unified_checkpoint_path"] == str(
        tmp_path / "artifacts" / "end2end" / "pipeline-smoke" / "best.pt"
    )
    assert payload["detector_variant"] == "Det-B0"
    assert payload["recognizer_variant"] == "Rec-B2"
    assert payload["system_variant"] == "Sys-B2"
