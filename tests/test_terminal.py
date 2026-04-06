import sys

from dltr.terminal import _build_tqdm_kwargs


def test_build_tqdm_kwargs_uses_prettier_defaults() -> None:
    kwargs = _build_tqdm_kwargs(total=120, description="识别训练", stream=sys.stdout)

    assert kwargs["total"] == 120
    assert kwargs["desc"] == "识别训练"
    assert kwargs["colour"] == "cyan"
    assert kwargs["unit"] == "step"
    assert kwargs["mininterval"] == 0.2
    assert kwargs["smoothing"] == 0.1
    assert "{percentage:3.0f}%" in kwargs["bar_format"]
    assert "{rate_fmt}" in kwargs["bar_format"]
