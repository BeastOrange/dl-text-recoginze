from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path


class ProgressBar:
    def __init__(self, total: int, description: str) -> None:
        self.total = max(total, 1)
        self.description = description
        self.started_at = time.perf_counter()
        self.is_tty = sys.stdout.isatty()
        self.last_render_width = 0
        self.last_logged_step = -1

    def update(self, current: int, *, metrics: dict[str, float | int] | None = None) -> None:
        clamped = min(max(current, 0), self.total)
        line = self._build_line(clamped, metrics or {})
        if self.is_tty:
            padded = line.ljust(self.last_render_width)
            sys.stdout.write(f"\r{padded}")
            sys.stdout.flush()
            self.last_render_width = max(self.last_render_width, len(line))
            return

        step = int((clamped / self.total) * 20)
        if step == self.last_logged_step and clamped != self.total:
            return
        self.last_logged_step = step
        print(line, flush=True)

    def finish(self, *, metrics: dict[str, float | int] | None = None) -> None:
        self.update(self.total, metrics=metrics)
        if self.is_tty:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _build_line(self, current: int, metrics: dict[str, float | int]) -> str:
        percent = current / self.total
        width = 24
        try:
            terminal_width = shutil.get_terminal_size((100, 20)).columns
            width = 32 if terminal_width >= 120 else 24
        except OSError:
            width = 24
        filled = int(round(width * percent))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.perf_counter() - self.started_at
        metric_text = _format_metrics(metrics)
        return (
            f"{self.description} [{bar}] {current}/{self.total} "
            f"{percent * 100:5.1f}% elapsed={elapsed:6.1f}s{metric_text}"
        )


def print_stage_header(title: str, details: list[tuple[str, object]] | None = None) -> None:
    print(f"=== {title} ===")
    for label, value in details or []:
        print(f"{label}：{value}")
    print(flush=True)


def print_artifact_summary(title: str, artifacts: list[tuple[str, str | Path]]) -> None:
    print(title)
    for label, path in artifacts:
        print(f"- {label}：{path}")
    print(flush=True)


def _format_metrics(metrics: dict[str, float | int]) -> str:
    if not metrics:
        return ""
    parts: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " " + " ".join(parts)
