import json
from pathlib import Path

from dltr.visualization.ablation_reports import build_ablation_overview


def test_build_ablation_overview_writes_outputs(tmp_path: Path) -> None:
    detection_json = tmp_path / "detection_summary.json"
    recognition_json = tmp_path / "recognition_summary.json"
    semantic_json = tmp_path / "semantic_summary.json"
    detection_json.write_text(
        json.dumps([{"run_name": "det_a", "primary_metric": 0.61}], ensure_ascii=False),
        encoding="utf-8",
    )
    recognition_json.write_text(
        json.dumps([{"run_name": "rec_a", "primary_metric": 0.81}], ensure_ascii=False),
        encoding="utf-8",
    )
    semantic_json.write_text(
        json.dumps([{"run_name": "sem_a", "primary_metric": 0.72}], ensure_ascii=False),
        encoding="utf-8",
    )

    outputs = build_ablation_overview(
        detection_summary_json=detection_json,
        recognition_summary_json=recognition_json,
        semantic_summary_json=semantic_json,
        output_dir=tmp_path / "reports" / "ablation",
    )

    assert outputs["markdown"].exists()
    assert outputs["png"].exists()
