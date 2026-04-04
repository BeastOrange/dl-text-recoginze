from dltr.cli import main


def test_demo_static_mode_writes_assets(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["demo", "--text", "营业时间 09:00-21:00 电话 13800138000"])

    assert exit_code == 0
    assert (
        tmp_path / "reports" / "demo_assets" / "generated" / "demo_preview_semantic_eval.md"
    ).exists()
