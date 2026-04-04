import json
from pathlib import Path

from dltr.cli import main


def test_data_validate_returns_nonzero_when_required_datasets_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    config_dir = tmp_path / "configs" / "data"
    config_dir.mkdir(parents=True)
    (config_dir / "datasets.yaml").write_text(
        "datasets:\n"
        "  - name: rctw17\n"
        "    relative_path: data/raw/rctw17\n"
        "    required: true\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["data", "validate", "--config", "configs/data/datasets.yaml"])

    assert exit_code == 1


def test_build_manifest_command_emits_jsonl_rows(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset_root = tmp_path / "data" / "raw" / "rctw17"
    dataset_root.mkdir(parents=True)
    (dataset_root / "sample.jpg").write_bytes(b"image")
    (dataset_root / "sample.txt").write_text("0,0,10,10,测试文本", encoding="utf-8")
    config_dir = tmp_path / "configs" / "data"
    config_dir.mkdir(parents=True)
    (config_dir / "datasets.yaml").write_text(
        "datasets:\n"
        "  - name: rctw17\n"
        "    relative_path: data/raw/rctw17\n"
        "    required: true\n"
        "    image_extensions: ['.jpg']\n"
        "    label_extensions: ['.txt']\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "data",
            "build-rec-lmdb",
            "--config",
            "configs/data/datasets.yaml",
            "--dataset",
            "rctw17",
        ]
    )

    manifest_path = tmp_path / "data" / "processed" / "rctw17_manifest.jsonl"
    payload = json.loads(manifest_path.read_text(encoding="utf-8").strip())
    assert exit_code == 0
    assert payload["text"] == "测试文本"


def test_train_recognizer_creates_train_plan(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    manifest = tmp_path / "data" / "processed" / "train.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"text":"测试"}\n', encoding="utf-8")
    charset = tmp_path / "data" / "processed" / "charset.txt"
    charset.write_text("测\n试\n", encoding="utf-8")
    config_dir = tmp_path / "configs" / "recognition"
    config_dir.mkdir(parents=True)
    (config_dir / "recognition.yaml").write_text(
        "\n".join(
            [
                "experiment_name: rec_smoke",
                "model_name: transocr",
                "dataset_manifest: data/processed/train.jsonl",
                "charset_file: data/processed/charset.txt",
                "output_dir: artifacts/checkpoints/recognition/smoke",
                "epochs: 1",
                "batch_size: 2",
                "image_height: 48",
                "image_width: 320",
                "learning_rate: 0.001",
                "second_pass:",
                "  enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "train",
            "recognizer",
            "--config",
            "configs/recognition/recognition.yaml",
            "--run-id",
            "smoke-run",
        ]
    )

    plan_path = (
        tmp_path
        / "artifacts"
        / "checkpoints"
        / "recognition"
        / "smoke"
        / "smoke-run"
        / "train_plan.md"
    )
    assert exit_code == 0
    assert plan_path.exists()


def test_evaluate_semantic_writes_report(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    predictions = tmp_path / "semantic_predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "source_id": "demo-1",
                "text": "营业时间09:00-21:00 电话13800138000",
                "semantic_class": "service_info",
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "evaluate",
            "semantic",
            "--run-name",
            "semantic-smoke",
            "--predictions-jsonl",
            "semantic_predictions.jsonl",
            "--output-dir",
            "reports/eval",
        ]
    )

    report_path = tmp_path / "reports" / "eval" / "semantic-smoke_semantic_eval.md"
    assert exit_code == 0
    assert report_path.exists()


def test_demo_and_end2end_generate_outputs(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    demo_code = main(["demo", "--text", "当心高压 电话13800138000"])
    end2end_code = main(
        [
            "evaluate",
            "end2end",
            "--text",
            "营业时间09:00-21:00",
            "--confidence",
            "0.52",
            "--blur-score",
            "0.6",
        ]
    )

    assert demo_code == 0
    assert end2end_code == 0
    assert (tmp_path / "reports" / "demo_assets" / "demo_preview_semantic_eval.md").exists()
    assert (tmp_path / "reports" / "eval" / "end2end_preview.json").exists()
