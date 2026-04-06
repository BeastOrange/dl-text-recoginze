from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
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
from dltr.terminal import ProgressBar
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
        multi_scale_augmentation=config.multi_scale_augmentation,
    )
    val_dataset = _TorchDetectionDataset(
        val_samples,
        image_height=config.image_height,
        image_width=config.image_width,
    )
    train_sampler = _build_train_sampler(train_samples) if config.hard_case_sampling else None
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    print(
        f"检测训练开始：设备={device} 训练样本={len(train_dataset)} "
        f"验证样本={len(val_dataset)}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        train_progress = ProgressBar(
            total=len(train_loader),
            description=f"检测训练 第 {epoch}/{config.epochs} 轮",
        )
        for batch_index, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1
            train_progress.update(batch_index, metrics={"loss": float(loss.item())})
        train_progress.finish(metrics={"avg_loss": train_loss_total / max(train_batches, 1)})

        metrics = _evaluate_detector(
            model,
            val_loader,
            device,
            torch,
            epoch=epoch,
            total_epochs=config.epochs,
        )
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
        print(
            "检测训练轮次完成："
            f"epoch={epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_precision={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f} "
            f"val_hmean={metrics['hmean']:.4f}",
            flush=True,
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
    def __init__(
        self,
        samples,
        *,
        image_height: int,
        image_width: int,
        multi_scale_augmentation: bool = False,
    ) -> None:
        self.samples = samples
        self.image_height = image_height
        self.image_width = image_width
        self.multi_scale_augmentation = multi_scale_augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch = _import_torch()
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        polygons = [
            instance.points
            for instance in sample.instances
            if instance.ignore == 0
        ]
        image_array = np.asarray(image, dtype=np.uint8)
        original_height, original_width = image_array.shape[:2]
        if self.multi_scale_augmentation and polygons:
            scale_factor = float(np.random.choice([0.75, 1.0, 1.25, 1.5]))
            if scale_factor != 1.0:
                max_offset_x = max(int(round(original_width * scale_factor)) - original_width, 0)
                max_offset_y = max(int(round(original_height * scale_factor)) - original_height, 0)
                if scale_factor >= 1.0:
                    offset_x = int(np.random.randint(0, max_offset_x + 1)) if max_offset_x else 0
                    offset_y = int(np.random.randint(0, max_offset_y + 1)) if max_offset_y else 0
                else:
                    scaled_width = int(round(original_width * scale_factor))
                    scaled_height = int(round(original_height * scale_factor))
                    offset_x = int(
                        np.random.randint(0, original_width - scaled_width + 1)
                    )
                    offset_y = int(
                        np.random.randint(0, original_height - scaled_height + 1)
                    )
                image_array, polygons = _apply_multi_scale_augmentation(
                    image_array,
                    polygons,
                    scale_factor=scale_factor,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )

        image_array = (
            cv2.resize(image_array, (self.image_width, self.image_height)).astype(np.float32)
            / 255.0
        )
        polygons = [
            _scale_polygon(
                polygon,
                original_width=original_width,
                original_height=original_height,
                target_width=self.image_width,
                target_height=self.image_height,
            )
            for polygon in polygons
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


def _build_train_sampler(samples) -> Any:
    torch = _import_torch()
    weights = torch.tensor(
        [_estimate_hard_case_weight(sample) for sample in samples],
        dtype=torch.double,
    )
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def _estimate_hard_case_weight(sample) -> float:
    active_instances = [instance for instance in sample.instances if instance.ignore == 0]
    if not active_instances:
        return 1.0
    small_text = any(_polygon_bbox_area(instance.points) <= 400 for instance in active_instances)
    dense_text = len(active_instances) >= 4
    rotated_text = any(_is_rotated_polygon(instance.points) for instance in active_instances)
    weight = 1.0
    if small_text:
        weight += 0.75
    if dense_text:
        weight += 0.5
    if rotated_text:
        weight += 0.5
    return weight


def _polygon_bbox_area(polygon: list[int]) -> float:
    points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    width = float(points[:, 0].max() - points[:, 0].min())
    height = float(points[:, 1].max() - points[:, 1].min())
    return max(width, 0.0) * max(height, 0.0)


def _is_rotated_polygon(polygon: list[int]) -> bool:
    points = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    dx = float(points[1, 0] - points[0, 0])
    dy = float(points[1, 1] - points[0, 1])
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    distance_to_axis = min(abs(angle), abs(angle - 90.0), abs(angle - 180.0))
    return distance_to_axis > 5.0


def _apply_multi_scale_augmentation(
    image_array: np.ndarray,
    polygons: list[list[int]],
    *,
    scale_factor: float,
    offset_x: int,
    offset_y: int,
) -> tuple[np.ndarray, list[list[int]]]:
    original_height, original_width = image_array.shape[:2]
    scaled_width = max(int(round(original_width * scale_factor)), 1)
    scaled_height = max(int(round(original_height * scale_factor)), 1)
    resized = cv2.resize(image_array, (scaled_width, scaled_height))
    channels = 1 if image_array.ndim == 2 else image_array.shape[2]
    if scale_factor >= 1.0:
        crop_x = min(offset_x, max(scaled_width - original_width, 0))
        crop_y = min(offset_y, max(scaled_height - original_height, 0))
        cropped = resized[crop_y : crop_y + original_height, crop_x : crop_x + original_width]
        transformed_polygons = [
            _transform_polygon(polygon, scale_factor=scale_factor, shift_x=-crop_x, shift_y=-crop_y)
            for polygon in polygons
        ]
        return cropped, transformed_polygons

    canvas_shape = (
        (original_height, original_width)
        if channels == 1
        else (original_height, original_width, channels)
    )
    canvas = np.full(canvas_shape, 255, dtype=image_array.dtype)
    paste_x = min(offset_x, max(original_width - scaled_width, 0))
    paste_y = min(offset_y, max(original_height - scaled_height, 0))
    canvas[paste_y : paste_y + scaled_height, paste_x : paste_x + scaled_width] = resized
    transformed_polygons = [
        _transform_polygon(polygon, scale_factor=scale_factor, shift_x=paste_x, shift_y=paste_y)
        for polygon in polygons
    ]
    return canvas, transformed_polygons


def _transform_polygon(
    polygon: list[int],
    *,
    scale_factor: float,
    shift_x: int,
    shift_y: int,
) -> list[int]:
    transformed: list[int] = []
    for index, value in enumerate(polygon):
        if index % 2 == 0:
            transformed.append(int(round(value * scale_factor + shift_x)))
        else:
            transformed.append(int(round(value * scale_factor + shift_y)))
    return transformed


def _evaluate_detector(
    model: Any,
    loader: Any,
    device: str,
    torch: Any,
    *,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.eval()
    aggregated = {"precision": 0.0, "recall": 0.0, "hmean": 0.0}
    batches = 0
    progress = ProgressBar(
        total=len(loader),
        description=f"检测验证 第 {epoch}/{total_epochs} 轮",
    )
    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(loader, start=1):
            logits = model(images.to(device))
            probs = torch.sigmoid(logits)
            metrics = compute_detection_scores(probs.cpu(), masks.cpu())
            for key, value in metrics.items():
                aggregated[key] += value
            batches += 1
            progress.update(
                batch_index,
                metrics={
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "hmean": metrics["hmean"],
                },
            )
    progress.finish()
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
