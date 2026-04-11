import json

from dltr.cli import main
from dltr.models.recognition.evaluation import RecognitionMetrics
from dltr.models.recognition.pretrained_benchmark import (
    ManifestEvaluationResult,
)


def test_evaluate_recognizer_benchmark_command_writes_outputs(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    manifest = tmp_path / "iiit5k_test.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "iiit5k_test",
                "image_path": str(tmp_path / "sample.png"),
                "label_path": "",
                "text": "OPEN",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "sample.png").write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dltr.commands.evaluate_recognition_manifest_with_backend",
        lambda **_: ManifestEvaluationResult(
            metrics=RecognitionMetrics(
                samples=1,
                word_accuracy=1.0,
                cer=0.0,
                ned=0.0,
                mean_edit_distance=0.0,
                latency_ms=1.2,
            ),
            predictions=[
                {"image_path": str(tmp_path / "sample.png"), "target": "OPEN", "prediction": "OPEN"}
            ],
        ),
    )

    exit_code = main(
        [
            "evaluate",
            "recognizer-benchmark",
            "--run-name",
            "easyocr_iiit5k",
            "--model-name",
            "easyocr_en",
            "--backend",
            "easyocr",
            "--manifest",
            str(manifest),
            "--normalize",
            "upper",
            "--benchmark-name",
            "iiit5k",
            "--benchmark-category",
            "main",
            "--output-dir",
            "reports/eval",
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (tmp_path / "reports" / "eval" / "easyocr_iiit5k_recognition_eval.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["benchmark_name"] == "iiit5k"
    assert payload["metrics"]["word_accuracy"] == 1.0
