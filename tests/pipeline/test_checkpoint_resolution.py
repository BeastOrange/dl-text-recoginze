import json
from pathlib import Path

from dltr.pipeline.checkpoints import discover_latest_run_dir, resolve_best_checkpoint


def test_resolve_best_checkpoint_prefers_summary_pointer(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    best_path = run_dir / "checkpoints" / "best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_bytes(b"pt")
    (run_dir / "training_summary.json").write_text(
        json.dumps({"best_checkpoint_path": str(best_path)}, ensure_ascii=False),
        encoding="utf-8",
    )

    resolved = resolve_best_checkpoint(run_dir)

    assert resolved == best_path


def test_discover_latest_run_dir_uses_summary_presence(tmp_path: Path) -> None:
    older = tmp_path / "20250101-000000"
    newer = tmp_path / "20250102-000000"
    older.mkdir()
    newer.mkdir()
    (older / "training_summary.json").write_text("{}", encoding="utf-8")
    (newer / "training_summary.json").write_text("{}", encoding="utf-8")

    discovered = discover_latest_run_dir(tmp_path)

    assert discovered == newer


def test_discover_latest_run_dir_supports_nested_run_layout(tmp_path: Path) -> None:
    older = tmp_path / "artifacts" / "detection" / "det_a" / "20250101-000000"
    newer = tmp_path / "artifacts" / "detection" / "det_b" / "20250103-000000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "training_summary.json").write_text("{}", encoding="utf-8")
    (newer / "training_summary.json").write_text("{}", encoding="utf-8")

    discovered = discover_latest_run_dir(tmp_path / "artifacts")

    assert discovered == newer
