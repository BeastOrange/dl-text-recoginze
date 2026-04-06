from dltr.post_ocr.classification import ANALYSIS_LABELS, validate_analysis_label


def test_analysis_labels_match_project_scope() -> None:
    assert "shop_sign" in ANALYSIS_LABELS
    assert "advertisement" in ANALYSIS_LABELS
    assert "public_notice" in ANALYSIS_LABELS
    assert "traffic_or_warning" in ANALYSIS_LABELS
    assert "service_info" in ANALYSIS_LABELS
    assert "other" in ANALYSIS_LABELS


def test_validate_analysis_label_accepts_known_label() -> None:
    assert validate_analysis_label("shop_sign") == "shop_sign"
