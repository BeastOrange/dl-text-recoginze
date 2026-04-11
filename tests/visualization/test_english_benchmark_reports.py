import json

from dltr.visualization.english_benchmark_reports import (
    BenchmarkRecord,
    build_english_benchmark_summary,
)


def test_build_english_benchmark_summary_writes_main_and_hard_kpis(tmp_path) -> None:
    outputs = build_english_benchmark_summary(
        records=[
            BenchmarkRecord(
                benchmark="IIIT5K",
                category="main",
                word_accuracy=0.91,
                samples=3000,
            ),
            BenchmarkRecord(
                benchmark="SVT",
                category="main",
                word_accuracy=0.89,
                samples=647,
            ),
            BenchmarkRecord(
                benchmark="ICDAR15",
                category="hard",
                word_accuracy=0.73,
                samples=1811,
            ),
        ],
        output_dir=tmp_path,
        report_name="english_benchmark_summary",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    content = outputs["markdown"].read_text(encoding="utf-8")

    assert payload["summary"]["main_average_word_accuracy"] == 0.9
    assert payload["summary"]["hard_average_word_accuracy"] == 0.73
    assert outputs["png"].exists()
    assert outputs["png"].stat().st_size > 0
    assert "Main-English-Accuracy" in content
    assert "Hard-English-Accuracy" in content


def test_build_english_benchmark_summary_handles_missing_hard_records(tmp_path) -> None:
    outputs = build_english_benchmark_summary(
        records=[
            BenchmarkRecord(
                benchmark="IIIT5K",
                category="main",
                word_accuracy=0.92,
                samples=3000,
            )
        ],
        output_dir=tmp_path,
        report_name="english_benchmark_summary",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))

    assert payload["summary"]["main_average_word_accuracy"] == 0.92
    assert payload["summary"]["hard_average_word_accuracy"] is None
    assert outputs["png"].exists()
