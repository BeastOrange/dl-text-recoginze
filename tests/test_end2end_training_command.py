import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from dltr.cli import main
from dltr.models.detection.scaffold import load_detection_run_config
from dltr.models.end2end_system import train_end2end_multitask_system
from dltr.models.recognition.config import load_recognition_config
from dltr.project import ProjectPaths


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


def test_train_end2end_command_passes_resume_from(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset_dir = tmp_path / "data" / "raw" / "rects"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    det_train = tmp_path / "data" / "processed" / "det_train.jsonl"
    det_val = tmp_path / "data" / "processed" / "det_val.jsonl"
    rec_train = tmp_path / "data" / "processed" / "recognition_splits" / "train.jsonl"
    rec_val = tmp_path / "data" / "processed" / "recognition_splits" / "val.jsonl"
    charset = tmp_path / "data" / "processed" / "charset.txt"
    for path in (det_train, det_val, rec_train, rec_val):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    charset.write_text("营\n业\n", encoding="utf-8")

    det_config = tmp_path / "configs" / "detection" / "cmd_resume.yaml"
    det_config.parent.mkdir(parents=True, exist_ok=True)
    det_config.write_text(
        "\n".join(
            [
                "experiment_name: det_end2end_resume",
                "model_name: dbnet",
                f"dataset_dir: {dataset_dir.relative_to(tmp_path)}",
                f"train_manifest: {det_train.relative_to(tmp_path)}",
                f"validation_manifest: {det_val.relative_to(tmp_path)}",
                "output_root: artifacts/detection/det_end2end_resume",
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
    rec_config = tmp_path / "configs" / "recognition" / "cmd_resume.yaml"
    rec_config.parent.mkdir(parents=True, exist_ok=True)
    rec_config.write_text(
        "\n".join(
            [
                "experiment_name: rec_end2end_resume",
                "model_name: transformer",
                f"dataset_manifest: {rec_train.relative_to(tmp_path)}",
                f"validation_manifest: {rec_val.relative_to(tmp_path)}",
                f"charset_file: {charset.relative_to(tmp_path)}",
                "output_dir: artifacts/checkpoints/recognition/rec_end2end_resume",
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
    resume_dir = tmp_path / "artifacts" / "end2end" / "old-run"
    resume_dir.mkdir(parents=True, exist_ok=True)
    calls = {}

    @dataclass(frozen=True)
    class _JointResult:
        run_dir: Path
        checkpoint_path: Path
        summary_path: Path
        detector_proxy_summary_path: Path
        recognizer_proxy_summary_path: Path

    def fake_train_end2end_system(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["resume_from"] = kwargs.get("resume_from")
        run_dir = tmp_path / "artifacts" / "end2end" / "resume-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "best.pt"
        checkpoint_path.write_bytes(b"pt")
        summary_path = run_dir / "training_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        detector_proxy = run_dir / "detector_proxy.json"
        recognizer_proxy = run_dir / "recognizer_proxy.json"
        detector_proxy.write_text("{}", encoding="utf-8")
        recognizer_proxy.write_text("{}", encoding="utf-8")
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
            "resume-run",
            "--resume-from",
            str(resume_dir),
        ]
    )

    assert exit_code == 0
    assert calls["resume_from"] == resume_dir


def test_train_end2end_multitask_system_resumes_from_checkpoint(tmp_path: Path) -> None:
    det_config, rec_config, paths = _prepare_end2end_training_context(tmp_path, epochs=1)

    initial = train_end2end_multitask_system(
        det_config,
        rec_config,
        paths=paths,
        run_id="resume-run",
    )

    det_config_epoch2, rec_config_epoch2, _ = _prepare_end2end_training_context(tmp_path, epochs=2)
    resumed = train_end2end_multitask_system(
        det_config_epoch2,
        rec_config_epoch2,
        paths=paths,
        run_id="resume-run",
        resume_from=initial.checkpoint_path,
    )

    history_lines = resumed.history_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(resumed.summary_path.read_text(encoding="utf-8"))
    assert resumed.run_dir == initial.run_dir
    assert len(history_lines) == 2
    assert json.loads(history_lines[-1])["epoch"] == 2
    assert payload["run_id"] == "resume-run"


def test_train_end2end_multitask_system_resumes_from_run_dir(tmp_path: Path) -> None:
    det_config, rec_config, paths = _prepare_end2end_training_context(tmp_path, epochs=1)

    initial = train_end2end_multitask_system(
        det_config,
        rec_config,
        paths=paths,
        run_id="resume-dir-run",
    )

    det_config_epoch2, rec_config_epoch2, _ = _prepare_end2end_training_context(tmp_path, epochs=2)
    resumed = train_end2end_multitask_system(
        det_config_epoch2,
        rec_config_epoch2,
        paths=paths,
        run_id="resume-dir-run",
        resume_from=initial.run_dir,
    )

    history_lines = resumed.history_path.read_text(encoding="utf-8").splitlines()
    assert resumed.run_dir == initial.run_dir
    assert len(history_lines) == 2
    assert json.loads(history_lines[-1])["epoch"] == 2


def _prepare_end2end_training_context(
    tmp_path: Path,
    *,
    epochs: int,
) -> tuple[object, object, ProjectPaths]:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset_dir = tmp_path / "data" / "raw" / "rects"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    det_image = tmp_path / "det_scene.png"
    rec_image = tmp_path / "rec_crop.png"
    _write_box_image(det_image)
    _write_box_image(rec_image)

    det_train = tmp_path / "data" / "processed" / "detection_splits" / "train.jsonl"
    det_val = tmp_path / "data" / "processed" / "detection_splits" / "val.jsonl"
    rec_train = tmp_path / "data" / "processed" / "recognition_crop_splits" / "train.jsonl"
    rec_val = tmp_path / "data" / "processed" / "recognition_crop_splits" / "val.jsonl"
    charset = tmp_path / "data" / "processed" / "charset.txt"
    for path in (det_train, det_val, rec_train, rec_val):
        path.parent.mkdir(parents=True, exist_ok=True)
    det_payload = {
        "dataset": "rects",
        "image_path": str(det_image),
        "label_path": str(det_image.with_suffix(".json")),
        "instances": [
            {"points": [4, 4, 28, 4, 28, 28, 4, 28], "text": "营业时间", "ignore": 0}
        ],
    }
    rec_payload = {
        "dataset": "rects",
        "split": "train",
        "source_image_path": str(det_image),
        "image_path": str(rec_image),
        "text": "营业时间",
        "instance_index": 0,
    }
    det_train.write_text(json.dumps(det_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    det_val.write_text(json.dumps(det_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    rec_train.write_text(json.dumps(rec_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    rec_val.write_text(json.dumps(rec_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    charset.write_text("营\n业\n时\n间\n", encoding="utf-8")

    det_config_path = tmp_path / "configs" / "detection" / "resume.yaml"
    det_config_path.parent.mkdir(parents=True, exist_ok=True)
    det_config_path.write_text(
        "\n".join(
            [
                "experiment_name: det_resume_smoke",
                "model_name: dbnet",
                "dataset_dir: data/raw/rects",
                "train_manifest: data/processed/detection_splits/train.jsonl",
                "validation_manifest: data/processed/detection_splits/val.jsonl",
                "output_root: artifacts/detection/det_resume_smoke",
                f"epochs: {epochs}",
                "batch_size: 1",
                "learning_rate: 0.001",
                "image_height: 32",
                "image_width: 32",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )
    rec_config_path = tmp_path / "configs" / "recognition" / "resume.yaml"
    rec_config_path.parent.mkdir(parents=True, exist_ok=True)
    rec_config_path.write_text(
        "\n".join(
            [
                "experiment_name: rec_resume_smoke",
                "model_name: transformer",
                "dataset_manifest: data/processed/recognition_crop_splits/train.jsonl",
                "validation_manifest: data/processed/recognition_crop_splits/val.jsonl",
                "charset_file: data/processed/charset.txt",
                "output_dir: artifacts/checkpoints/recognition/rec_resume_smoke",
                f"epochs: {epochs}",
                "batch_size: 1",
                "image_height: 32",
                "image_width: 128",
                "learning_rate: 0.001",
                "device: cpu",
                "num_workers: 0",
                "preprocess:",
                "  preserve_aspect_ratio: true",
                "  rotate_vertical_text: true",
                "  vertical_aspect_threshold: 1.2",
                "  padding_value: 255",
                "second_pass:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    return (
        load_detection_run_config(det_config_path),
        load_recognition_config(rec_config_path),
        ProjectPaths.from_root(tmp_path),
    )


def _write_box_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 28, 28), outline="black", width=2)
    image.save(path)
