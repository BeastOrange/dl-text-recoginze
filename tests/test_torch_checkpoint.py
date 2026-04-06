from pathlib import Path

from dltr.torch_checkpoint import load_torch_checkpoint


def test_load_torch_checkpoint_prefers_weights_only_false(tmp_path: Path) -> None:
    called = {}

    class _Torch:
        @staticmethod
        def load(path, **kwargs):  # noqa: ANN001
            called["path"] = path
            called["kwargs"] = kwargs
            return {"ok": True}

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"pt")

    result = load_torch_checkpoint(_Torch(), checkpoint_path)

    assert result == {"ok": True}
    assert called["path"] == checkpoint_path
    assert called["kwargs"]["weights_only"] is False
    assert called["kwargs"]["map_location"] == "cpu"


def test_load_torch_checkpoint_falls_back_when_weights_only_is_unsupported(
    tmp_path: Path,
) -> None:
    calls = []

    class _Torch:
        @staticmethod
        def load(path, **kwargs):  # noqa: ANN001
            calls.append(kwargs)
            if "weights_only" in kwargs:
                raise TypeError("unexpected keyword argument 'weights_only'")
            return {"fallback": True}

    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"pt")

    result = load_torch_checkpoint(_Torch(), checkpoint_path)

    assert result == {"fallback": True}
    assert len(calls) == 2
    assert calls[0]["weights_only"] is False
    assert "weights_only" not in calls[1]
