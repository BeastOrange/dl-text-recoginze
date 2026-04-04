import pytest

from dltr.cli import build_parser, main


def test_parser_supports_required_top_level_groups() -> None:
    parser = build_parser()

    parsed = parser.parse_args(["data", "validate"])

    assert parsed.group == "data"
    assert parsed.command == "validate"
    assert callable(parsed.handler)


def test_main_returns_zero_for_demo_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["demo", "--text", "测试电话13800138000"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "demo" in captured.out.lower()


def test_parser_requires_subcommand() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
