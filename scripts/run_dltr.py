from __future__ import annotations

import sys
from pathlib import Path


def _inject_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    src_str = str(src)
    if src.exists() and src.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> int:
    _inject_src_path()
    from dltr.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
