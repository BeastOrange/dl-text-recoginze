from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dltr.models.detection.dataset import load_detection_samples, rasterize_text_mask
from dltr.models.detection.metrics import compute_detection_scores
from dltr.models.detection.scaffold import (
    DetectionRunConfig,
    DetectionRunContext,
    prepare_detection_run,
    write_evaluation_summary,
    write_experiment_metadata,
)
from dltr.project import ProjectPaths
from dltr.visualization.training_reports import render_detection_history_plot


@dataclass(frozen=True)
class DetectionTrainingResult:
    context: DetectionRunContext
    checkpoint_path: Path
    best_checkpoint_path: Path
    history_path: Path
    history_markdown_path: Path
    history_plot_path: Path
    summary_path: Path
    report_paths: dict[str, Path]


def train_dbnet_detector(
    config: DetectionRunConfig,
    *,
    paths: ProjectPaths | None = None,
    run_id: str | None = None,
) -> DetectionTrainingResult:
    torch = _import_torch()
    nn = torch.nn
    optim = torch.optim
    data_utils = torch.utils.data

    project_paths = paths or ProjectPaths.from_root()
    context = prepare_detection_run(config, paths=project_paths, run_id=run_id)
    train_manifest = _resolve_required_path(
        project_paths.root,
        config.train_manifest,
        "train_manifest",
    )
    val_manifest = _resolve_required_path(
        project_paths.root,
        config.validation_manifest,
        "validation_manifest",
    )
    train_samples = load_detection_samples(train_manifest)
    val_samples = load_detection_samples(val_manifest)
    if not train_samples:
        raise ValueError(f"No detection training samples found in {train_manifest}")
    if not val_samples:
        raise ValueError(f"No detection validation samples found in {val_manifest}")

    train_dataset = _TorchDetectionDataset(
        train_samples,
        image_height=config.image_height,
        image_width=config.image_width,
    )
    val_dataset = _TorchDetectionDataset(
        val_samples,
        image_height=config.image_height,
        image_width=config.image_width,
    )
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = data_utils.DataLoader(
        val_dataset,
        batch_size=max(1, min(config.batch_size, 8)),
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = _select_device(torch, config.device)
    model = _build_dbnet_tiny(nn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_checkpoint_path = context.checkpoints_dir / "best.pt"
    history_path = context.run_dir / "training_history.jsonl"
    history_markdown_path = context.run_dir / "training_history.md"
    best_hmean = float("-inf")
    metrics = {"precision": 0.0, "recall": 0.0, "hmean": 0.0}
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1

        metrics = _evaluate_detector(model, val_loader, device, torch)
        train_loss = train_loss_total / max(train_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_hmean": metrics["hmean"],
            }
        )
        if metrics["hmean"] >= best_hmean:
            best_hmean = metrics["hmean"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": metrics,
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )
    report_paths = write_evaluation_summary(
        context,
        split="val",
        metrics=metrics,
    )
    write_experiment_metadata(
        context,
        notes="DBNet-style tiny segmentation baseline training run.",
    )
    checkpoint_path = context.checkpoints_dir / "last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    summary_path = context.run_dir / "training_summary.json"
    history_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in history) + "\n",
        encoding="utf-8",
    )
    history_markdown_path.write_text(
        _build_history_markdown(config.experiment_name, history),
        encoding="utf-8",
    )
    history_plot_paths = render_detection_history_plot(
        run_name=config.experiment_name,
        history_path=history_path,
        output_dir=context.run_dir / "reports",
    )
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "history_path": str(history_path),
                "history_plot_path": str(history_plot_paths["png"]),
                "report_paths": {key: str(value) for key, value in report_paths.items()},
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return DetectionTrainingResult(
        context=context,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        history_path=history_path,
        history_markdown_path=history_markdown_path,
        history_plot_path=history_plot_paths["png"],
        summary_path=summary_path,
        report_paths=report_paths,
    )


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the current environment. "
            "Run `uv sync --extra train-cu` before training the detector."
        ) from exc
    return torch


def _select_device(torch: Any, configured_device: str) -> str:
    if configured_device != "auto":
        return configured_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _TorchDetectionDataset:
    def __init__(self, samples, *, image_height: int, image_width: int) -> None:
        self.samples = samples
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch = _import_torch()
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((self.image_width, self.image_height))
        image_array = np.asarray(image, dtype=np.float32) / 255.0

        polygons = [
            _scale_polygon(
                instance.points,
                original_width=original_width,
                original_height=original_height,
                target_width=self.image_width,
                target_height=self.image_height,
            )
            for instance in sample.instances
            if instance.ignore == 0
        ]
        mask_array = rasterize_text_mask(
            image_height=self.image_height,
            image_width=self.image_width,
            polygons=polygons,
        )
        image_tensor = torch.tensor(np.transpose(image_array, (2, 0, 1)), dtype=torch.float32)
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32).unsqueeze(0)
        return image_tensor, mask_tensor


def _build_dbnet_tiny(nn: Any) -> Any:
    class _Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )

        def forward(self, images):
            return self.model(images)

    return _Detector()


def _evaluate_detector(model: Any, loader: Any, device: str, torch: Any) -> dict[str, float]:
    model.eval()
    aggregated = {"precision": 0.0, "recall": 0.0, "hmean": 0.0}
    batches = 0
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            probs = torch.sigmoid(logits)
            metrics = compute_detection_scores(probs.cpu(), masks.cpu())
            for key, value in metrics.items():
                aggregated[key] += value
            batches += 1
    if batches == 0:
        return aggregated
    return {key: value / batches for key, value in aggregated.items()}


def _resolve_required_path(root: Path, raw_path: Path | None, field_name: str) -> Path:
    if raw_path is None:
        raise ValueError(f"{field_name} must be configured for detector training")
    return (root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()


def _scale_polygon(
    polygon: list[int],
    *,
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> list[int]:
    scaled: list[int] = []
    width_scale = target_width / max(original_width, 1)
    height_scale = target_height / max(original_height, 1)
    for index, value in enumerate(polygon):
        if index % 2 == 0:
            scaled.append(int(round(value * width_scale)))
        else:
            scaled.append(int(round(value * height_scale)))
    return scaled


def _build_history_markdown(
    experiment_name: str,
    history: list[dict[str, float | int]],
) -> str:
    lines = [
        f"# Training History: {experiment_name}",
        "",
        "| Epoch | Train Loss | Val Precision | Val Recall | Val Hmean |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in history:
        lines.append(
            f"| {record['epoch']} | {record['train_loss']:.6f} | "
            f"{record['val_precision']:.6f} | {record['val_recall']:.6f} | "
            f"{record['val_hmean']:.6f} |"
        )
    return "\n".join(lines) + "\n"
