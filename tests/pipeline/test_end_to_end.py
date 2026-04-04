import json
from pathlib import Path

from dltr.pipeline.end_to_end import (
    EndToEndLineResult,
    EndToEndPipelineArtifacts,
    run_end_to_end_pipeline,
)
from dltr.semantic.slots import extract_semantic_slots


def test_run_end_to_end_pipeline_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"fake")

    def fake_pipeline(**_: object) -> EndToEndPipelineArtifacts:
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "result.json"
        markdown_path = output_dir / "result.md"
        preview_path = output_dir / "preview.png"
        payload = {
            "image_path": str(image_path),
            "lines": [
                {
                    "text": "营业时间 09:00-21:00",
                    "semantic_class": "service_info",
                }
            ],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text("# End-to-End OCR Result\n", encoding="utf-8")
        preview_path.write_bytes(b"png")
        return EndToEndPipelineArtifacts(
            output_dir=output_dir,
            json_path=json_path,
            markdown_path=markdown_path,
            preview_image_path=preview_path,
            line_results=[
                EndToEndLineResult(
                    line_id="line-0",
                    polygon=[10, 10, 100, 10, 100, 40, 10, 40],
                    text="营业时间 09:00-21:00",
                    recognition_confidence=0.75,
                    semantic_class="service_info",
                    semantic_confidence=0.8,
                    slots=extract_semantic_slots("营业时间 09:00-21:00"),
                )
            ],
        )

    monkeypatch.setattr(
        "dltr.pipeline.end_to_end._run_pipeline_internal",
        fake_pipeline,
    )

    artifacts = run_end_to_end_pipeline(
        image_path=image_path,
        output_dir=tmp_path / "output",
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
    )

    assert artifacts.json_path.exists()
    assert artifacts.markdown_path.exists()
    assert artifacts.preview_image_path.exists()
    assert artifacts.line_results[0].semantic_class == "service_info"
