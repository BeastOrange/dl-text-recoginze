import subprocess

from dltr.cli import main


def test_demo_static_mode_writes_assets(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["demo", "--text", "营业时间 09:00-21:00 电话 13800138000"])

    assert exit_code == 0
    assert (
        tmp_path / "reports" / "demo_assets" / "generated" / "demo_preview_semantic_eval.md"
    ).exists()


def test_demo_serve_mode_invokes_streamlit(monkeypatch, tmp_path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = {}

    def fake_run(cmd, cwd, check):  # noqa: ANN001
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("dltr.commands.subprocess.run", fake_run)

    exit_code = main(["demo", "--serve"])

    assert exit_code == 0
    assert "streamlit" in " ".join(calls["cmd"])
