from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dltr.models.recognition.charset import CharacterVocabulary
from dltr.models.recognition.config import RecognitionExperimentConfig
from dltr.models.recognition.dataset import (
    RecognitionSample,
    load_recognition_samples,
)
from dltr.models.recognition.evaluation import (
    RecognitionMetrics,
    generate_recognition_evaluation_report,
)
from dltr.models.recognition.metrics import compute_recognition_scores
from dltr.project import ProjectPaths
from dltr.terminal import ProgressBar
from dltr.visualization.training_reports import render_recognition_history_plot


@dataclass(frozen=True)
class RecognitionTrainingResult:
    run_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    history_path: Path
    history_markdown_path: Path
    history_plot_path: Path
    summary_path: Path
    report_path: Path
    metrics: RecognitionMetrics


@dataclass(frozen=True)
class RuntimeOptimizations:
    pin_memory: bool
    non_blocking: bool
    use_amp: bool
    loader_kwargs: dict[str, bool | int]


def train_crnn_recognizer(
    config: RecognitionExperimentConfig,
    *,
    paths: ProjectPaths | None = None,
    run_id: str | None = None,
) -> RecognitionTrainingResult:
    return _train_ctc_recognizer(
        config,
        paths=paths,
        run_id=run_id,
        model_builder=_build_crnn_model,
        training_note="CRNN baseline training run.",
    )


def train_transformer_recognizer(
    config: RecognitionExperimentConfig,
    *,
    paths: ProjectPaths | None = None,
    run_id: str | None = None,
) -> RecognitionTrainingResult:
    return _train_ctc_recognizer(
        config,
        paths=paths,
        run_id=run_id,
        model_builder=_build_transformer_model,
        training_note="Transformer-CTC baseline training run.",
    )


def _train_ctc_recognizer(
    config: RecognitionExperimentConfig,
    *,
    paths: ProjectPaths | None = None,
    run_id: str | None = None,
    model_builder: Any,
    training_note: str,
) -> RecognitionTrainingResult:
    torch = _import_torch()
    nn = torch.nn
    optim = torch.optim
    data_utils = torch.utils.data

    project_paths = paths or ProjectPaths.from_root()
    train_manifest = (project_paths.root / config.dataset_manifest).resolve()
    val_manifest = (project_paths.root / config.validation_manifest).resolve()
    charset_path = (project_paths.root / config.charset_file).resolve()
    if not train_manifest.exists():
        raise FileNotFoundError(f"Recognition manifest not found: {train_manifest}")
    if not val_manifest.exists():
        raise FileNotFoundError(f"Validation manifest not found: {val_manifest}")
    if not charset_path.exists():
        raise FileNotFoundError(f"Charset file not found: {charset_path}")

    vocabulary = CharacterVocabulary.from_file(charset_path)
    train_samples = load_recognition_samples(train_manifest)
    val_samples = load_recognition_samples(val_manifest)
    if not train_samples:
        raise ValueError(f"No training samples found in {train_manifest}")
    if not val_samples:
        raise ValueError(f"No validation samples found in {val_manifest}")

    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = (project_paths.root / config.output_dir / resolved_run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    best_checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "training_history.jsonl"
    history_markdown_path = run_dir / "training_history.md"

    train_dataset = _TorchRecognitionDataset(
        train_samples,
        vocabulary,
        image_height=config.image_height,
        image_width=config.image_width,
    )
    val_dataset = _TorchRecognitionDataset(
        val_samples,
        vocabulary,
        image_height=config.image_height,
        image_width=config.image_width,
    )
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate_ctc_batch,
        **_build_runtime_optimizations(
            device=_select_device(torch, config.device),
            num_workers=config.num_workers,
        ).loader_kwargs,
    )
    val_loader = data_utils.DataLoader(
        val_dataset,
        batch_size=max(1, min(config.batch_size, 32)),
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate_ctc_batch,
        **_build_runtime_optimizations(
            device=_select_device(torch, config.device),
            num_workers=config.num_workers,
        ).loader_kwargs,
    )

    device = _select_device(torch, config.device)
    runtime = _build_runtime_optimizations(device=device, num_workers=config.num_workers)
    _configure_cuda_backend(torch, device=device)
    model = model_builder(nn=nn, vocabulary_size=vocabulary.size).to(device)
    criterion = nn.CTCLoss(blank=vocabulary.blank_index, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = _build_grad_scaler(torch, use_amp=runtime.use_amp)
    history: list[dict[str, float | int]] = []
    best_word_accuracy = float("-inf")
    metrics = RecognitionMetrics(
        samples=1,
        word_accuracy=0.0,
        cer=1.0,
        ned=1.0,
        mean_edit_distance=1.0,
    )

    print(
        f"识别训练开始：设备={device} 训练样本={len(train_dataset)} "
        f"验证样本={len(val_dataset)} 字符集大小={vocabulary.size}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        train_progress = ProgressBar(
            total=len(train_loader),
            description=f"识别训练 第 {epoch}/{config.epochs} 轮",
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=runtime.non_blocking)
            targets = batch["targets"].to(device, non_blocking=runtime.non_blocking)
            target_lengths = batch["target_lengths"].to(device, non_blocking=runtime.non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(torch, use_amp=runtime.use_amp):
                log_probs = model(images)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs.float(), targets, input_lengths, target_lengths)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1
            train_progress.update(batch_index, metrics={"loss": float(loss.item())})
        train_progress.finish(metrics={"avg_loss": train_loss_total / max(train_batches, 1)})

        started_at = time.perf_counter()
        predictions, targets = _evaluate_ctc_model(
            model,
            val_loader,
            vocabulary,
            device,
            torch,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        latency_ms = ((time.perf_counter() - started_at) / max(1, len(targets))) * 1000
        score_summary = compute_recognition_scores(predictions, targets)
        metrics = RecognitionMetrics(
            samples=score_summary.samples,
            word_accuracy=score_summary.word_accuracy,
            cer=score_summary.cer,
            ned=score_summary.ned,
            mean_edit_distance=score_summary.mean_edit_distance,
            latency_ms=latency_ms,
        )
        train_loss = train_loss_total / max(train_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_word_accuracy": metrics.word_accuracy,
                "val_cer": metrics.cer,
                "val_ned": metrics.ned,
                "val_mean_edit_distance": metrics.mean_edit_distance,
                "val_latency_ms": metrics.latency_ms or 0.0,
            }
        )
        if metrics.word_accuracy >= best_word_accuracy:
            best_word_accuracy = metrics.word_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": asdict(metrics),
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )
        print(
            "识别训练轮次完成："
            f"epoch={epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_word_accuracy={metrics.word_accuracy:.4f} "
            f"val_cer={metrics.cer:.4f} "
            f"val_ned={metrics.ned:.4f}",
            flush=True,
        )
    report_path = generate_recognition_evaluation_report(
        run_name=config.experiment_name,
        model_name=config.model_name,
        metrics=metrics,
        output_dir=run_dir,
        notes=training_note,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": asdict(metrics),
        },
        checkpoint_path,
    )
    summary_path = run_dir / "training_summary.json"
    history_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in history) + "\n",
        encoding="utf-8",
    )
    history_markdown_path.write_text(
        _build_history_markdown(config.experiment_name, history),
        encoding="utf-8",
    )
    history_plot_paths = render_recognition_history_plot(
        run_name=config.experiment_name,
        history_path=history_path,
        output_dir=run_dir,
    )
    summary_path.write_text(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "history_path": str(history_path),
                "history_plot_path": str(history_plot_paths["png"]),
                "report_path": str(report_path),
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return RecognitionTrainingResult(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        history_path=history_path,
        history_markdown_path=history_markdown_path,
        history_plot_path=history_plot_paths["png"],
        summary_path=summary_path,
        report_path=report_path,
        metrics=metrics,
    )


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the current environment. "
            "Run `uv sync --extra train-cu` before training the recognizer."
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


def _build_runtime_optimizations(device: str, num_workers: int) -> RuntimeOptimizations:
    is_cuda = device == "cuda"
    loader_kwargs: dict[str, bool | int] = {
        "pin_memory": is_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4 if is_cuda else 2
    return RuntimeOptimizations(
        pin_memory=is_cuda,
        non_blocking=is_cuda,
        use_amp=is_cuda,
        loader_kwargs=loader_kwargs,
    )


def _configure_cuda_backend(torch: Any, *, device: str) -> None:
    if device != "cuda":
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _build_grad_scaler(torch: Any, *, use_amp: bool) -> Any | None:
    if not use_amp:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_context(torch: Any, *, use_amp: bool) -> Any:
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


class _TorchRecognitionDataset:
    def __init__(
        self,
        samples: list[RecognitionSample],
        vocabulary: CharacterVocabulary,
        *,
        image_height: int,
        image_width: int,
    ) -> None:
        self.samples = samples
        self.vocabulary = vocabulary
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("L")
        image = image.resize((self.image_width, self.image_height))
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        encoded = self.vocabulary.encode(sample.text)
        return {
            "image": image_array,
            "target": encoded,
            "text": sample.text,
        }


def _collate_ctc_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    torch = _import_torch()
    images = torch.tensor(
        np.stack([item["image"] for item in batch], axis=0),
        dtype=torch.float32,
    ).unsqueeze(1)
    targets_list = [torch.tensor(item["target"], dtype=torch.long) for item in batch]
    targets = torch.cat(targets_list) if targets_list else torch.tensor([], dtype=torch.long)
    target_lengths = torch.tensor([len(item["target"]) for item in batch], dtype=torch.long)
    texts = [str(item["text"]) for item in batch]
    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
    }


def _build_crnn_model(nn: Any, vocabulary_size: int) -> Any:
    class _CRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, None)),
            )
            self.sequence_model = nn.LSTM(
                input_size=256,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
            )
            self.classifier = nn.Linear(512, vocabulary_size)

        def forward(self, images: Any) -> Any:
            torch = _import_torch()
            features = self.features(images)
            sequence = features.squeeze(2).permute(2, 0, 1)
            recurrent, _ = self.sequence_model(sequence)
            logits = self.classifier(recurrent)
            return torch.log_softmax(logits, dim=2)

    return _CRNN()


def _build_transformer_model(nn: Any, vocabulary_size: int) -> Any:
    torch = _import_torch()

    class _TransformerCTC(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, None)),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
            )
            self.position_embedding = nn.Parameter(torch.zeros(1, 512, 256))
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(256, vocabulary_size)

        def forward(self, images: Any) -> Any:
            features = self.features(images)
            sequence = features.squeeze(2).permute(0, 2, 1)
            positional = self.position_embedding[:, : sequence.size(1)]
            encoded = self.encoder(sequence + positional)
            logits = self.classifier(encoded).permute(1, 0, 2)
            return torch.log_softmax(logits, dim=2)

    return _TransformerCTC()


def _build_recognizer_model(nn: Any, model_name: str, vocabulary_size: int) -> Any:
    if model_name == "crnn":
        return _build_crnn_model(nn=nn, vocabulary_size=vocabulary_size)
    if model_name == "transformer":
        return _build_transformer_model(nn=nn, vocabulary_size=vocabulary_size)
    raise ValueError(f"Unsupported recognition model: {model_name}")


def _evaluate_ctc_model(
    model: Any,
    loader: Any,
    vocabulary: CharacterVocabulary,
    device: str,
    torch: Any,
    *,
    epoch: int,
    total_epochs: int,
) -> tuple[list[str], list[str]]:
    model.eval()
    predictions: list[str] = []
    targets: list[str] = []
    progress = ProgressBar(
        total=len(loader),
        description=f"识别验证 第 {epoch}/{total_epochs} 轮",
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["images"].to(device)
            log_probs = model(images)
            greedy_indices = log_probs.argmax(dim=2).permute(1, 0)
            for indices, target in zip(greedy_indices, batch["texts"], strict=True):
                predictions.append(vocabulary.decode_greedy(indices.tolist()))
                targets.append(target)
            progress.update(batch_index, metrics={"samples": len(targets)})
    progress.finish(metrics={"samples": len(targets)})
    return predictions, targets


def _build_history_markdown(
    experiment_name: str,
    history: list[dict[str, float | int]],
) -> str:
    lines = [
        f"# Training History: {experiment_name}",
        "",
        "| Epoch | Train Loss | Val Word Accuracy | Val CER | Val NED |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in history:
        lines.append(
            f"| {record['epoch']} | {record['train_loss']:.6f} | "
            f"{record['val_word_accuracy']:.6f} | {record['val_cer']:.6f} | "
            f"{record['val_ned']:.6f} |"
        )
    return "\n".join(lines) + "\n"
