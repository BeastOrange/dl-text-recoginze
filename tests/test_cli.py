from pathlib import Path

import pytest

from dltr.cli import build_parser, main


def test_parser_supports_required_top_level_groups() -> None:
    parser = build_parser()

    parsed = parser.parse_args(["data", "validate"])

    assert parsed.group == "data"
    assert parsed.command == "validate"
    assert callable(parsed.handler)


def test_main_returns_zero_for_demo_command(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["demo", "--text", "测试电话13800138000"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "demo" in captured.out.lower()


def test_parser_requires_subcommand() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_parser_rejects_removed_semantic_commands() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "semantic"])

    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "semantic"])


def test_parser_uses_transformer_recognizer_default_config() -> None:
    parser = build_parser()

    parsed = parser.parse_args(["train", "recognizer"])

    assert parsed.config == "configs/recognition/transformer_baseline.yaml"


def test_parser_supports_end2end_training_command() -> None:
    parser = build_parser()

    parsed = parser.parse_args(
        [
            "train",
            "end2end",
            "--detector-config",
            "configs/detection/dbnet_baseline.yaml",
            "--recognizer-config",
            "configs/recognition/transformer_baseline.yaml",
        ]
    )

    assert parsed.group == "train"
    assert parsed.command == "end2end"


def test_parser_supports_resume_for_training_commands() -> None:
    parser = build_parser()

    detector = parser.parse_args(
        ["train", "detector", "--resume-from", "artifacts/detection/run-1"]
    )
    recognizer = parser.parse_args(
        ["train", "recognizer", "--resume-from", "artifacts/checkpoints/recognition/run-1"]
    )
    end2end = parser.parse_args(
        [
            "train",
            "end2end",
            "--detector-config",
            "configs/detection/dbnet_baseline.yaml",
            "--recognizer-config",
            "configs/recognition/transformer_baseline.yaml",
            "--resume-from",
            "artifacts/end2end/run-1",
        ]
    )

    assert detector.resume_from == "artifacts/detection/run-1"
    assert recognizer.resume_from == "artifacts/checkpoints/recognition/run-1"
    assert end2end.resume_from == "artifacts/end2end/run-1"
