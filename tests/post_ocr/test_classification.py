from dltr.post_ocr.classification import analyze_scene_text


def test_analyze_scene_text_detects_warning_label() -> None:
    result = analyze_scene_text("当心高压 危险")

    assert result.label == "traffic_or_warning"
    assert result.confidence > 0.0


def test_analyze_scene_text_detects_service_info() -> None:
    result = analyze_scene_text("营业时间 09:00-21:00 电话 13800138000")

    assert result.label == "service_info"


def test_analyze_scene_text_detects_advertisement() -> None:
    result = analyze_scene_text("开业大促 全场五折 特价")

    assert result.label == "advertisement"
