from __future__ import annotations

import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - fallback only when runtime dependency is missing
    tqdm = None


class ProgressBar:
    def __init__(self, total: int, description: str) -> None:
        self.total = max(total, 1)
        self.description = description
        self.is_tty = sys.stdout.isatty()
        self.last_logged_step = -1
        self.current = 0
        self.bar = None
        if tqdm is not None and self.is_tty:
            self.bar = tqdm(
                total=self.total,
                desc=self.description,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            )

    def update(self, current: int, *, metrics: dict[str, float | int] | None = None) -> None:
        clamped = min(max(current, 0), self.total)
        if self.bar is not None:
            delta = clamped - self.current
            if delta > 0:
                self.bar.update(delta)
            self.current = clamped
            if metrics:
                self.bar.set_postfix(_stringify_metrics(metrics), refresh=False)
            return

        line = self._build_line(clamped, metrics or {})
        step = int((clamped / self.total) * 20)
        if step == self.last_logged_step and clamped != self.total:
            return
        self.last_logged_step = step
        print(line, flush=True)

    def finish(self, *, metrics: dict[str, float | int] | None = None) -> None:
        self.update(self.total, metrics=metrics)
        if self.bar is not None:
            self.bar.close()

    def _build_line(self, current: int, metrics: dict[str, float | int]) -> str:
        percent = current / self.total
        width = 24
        filled = int(round(width * percent))
        bar = "#" * filled + "-" * (width - filled)
        metric_text = _format_metrics(metrics)
        return (
            f"{self.description} [{bar}] {current}/{self.total} "
            f"{percent * 100:5.1f}%{metric_text}"
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


def _stringify_metrics(metrics: dict[str, float | int]) -> dict[str, str]:
    rendered: dict[str, str] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rendered[key] = f"{value:.4f}"
        else:
            rendered[key] = str(value)
    return rendered
