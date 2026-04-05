from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dltr.project import ProjectPaths
from dltr.semantic.config import SemanticExperimentConfig
from dltr.semantic.dataset import SemanticSample, load_semantic_samples
from dltr.semantic.metrics import SemanticScoreSummary, compute_semantic_scores
from dltr.terminal import ProgressBar
from dltr.visualization.training_reports import render_semantic_history_plot


@dataclass(frozen=True)
class SemanticTrainingResult:
    run_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    history_path: Path
    history_markdown_path: Path
    history_plot_path: Path
    summary_path: Path
    report_path: Path
    metrics: SemanticScoreSummary


def train_semantic_classifier(
    config: SemanticExperimentConfig,
    *,
    paths: ProjectPaths | None = None,
    run_id: str | None = None,
) -> SemanticTrainingResult:
    torch = _import_torch()
    nn = torch.nn
    optim = torch.optim
    data_utils = torch.utils.data

    project_paths = paths or ProjectPaths.from_root()
    train_manifest = (project_paths.root / config.dataset_manifest).resolve()
    val_manifest = (project_paths.root / config.validation_manifest).resolve()
    train_samples = load_semantic_samples(train_manifest)
    val_samples = load_semantic_samples(val_manifest)
    if not train_samples:
        raise ValueError(f"No semantic training samples found in {train_manifest}")
    if not val_samples:
        raise ValueError(f"No semantic validation samples found in {val_manifest}")

    vocabulary = _build_character_vocab(train_samples)
    label_to_index = {label: index for index, label in enumerate(config.label_set)}

    resolved_run_id = run_id or "semantic-run"
    run_dir = (project_paths.root / config.output_dir / resolved_run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    best_checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "training_history.jsonl"
    history_markdown_path = run_dir / "training_history.md"

    train_dataset = _SemanticTorchDataset(train_samples, vocabulary, label_to_index)
    val_dataset = _SemanticTorchDataset(val_samples, vocabulary, label_to_index)
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = data_utils.DataLoader(
        val_dataset,
        batch_size=max(1, min(config.batch_size, 32)),
        shuffle=False,
        num_workers=config.num_workers,
    )

    device = _select_device(torch, config.device)
    model = nn.Linear(len(vocabulary), len(label_to_index)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float | int]] = []
    best_accuracy = float("-inf")
    metrics = SemanticScoreSummary(samples=1, accuracy=0.0, macro_f1=0.0)

    print(
        f"语义训练开始：设备={device} 训练样本={len(train_dataset)} "
        f"验证样本={len(val_dataset)} 类别数={len(label_to_index)}",
        flush=True,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        train_progress = ProgressBar(
            total=len(train_loader),
            description=f"语义训练 第 {epoch}/{config.epochs} 轮",
        )
        for batch_index, (features, labels) in enumerate(train_loader, start=1):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1
            train_progress.update(batch_index, metrics={"loss": float(loss.item())})
        train_progress.finish(metrics={"avg_loss": train_loss_total / max(train_batches, 1)})

        metrics = _evaluate_semantic_model(
            model,
            val_loader,
            device,
            config.label_set,
            torch,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        train_loss = train_loss_total / max(train_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": metrics.accuracy,
                "val_macro_f1": metrics.macro_f1,
            }
        )
        if metrics.accuracy >= best_accuracy:
            best_accuracy = metrics.accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "metrics": asdict(metrics),
                    "label_to_index": label_to_index,
                    "vocabulary": vocabulary,
                },
                best_checkpoint_path,
            )
        print(
            "语义训练轮次完成："
            f"epoch={epoch}/{config.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_accuracy={metrics.accuracy:.4f} "
            f"val_macro_f1={metrics.macro_f1:.4f}",
            flush=True,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": asdict(metrics),
            "label_to_index": label_to_index,
            "vocabulary": vocabulary,
        },
        checkpoint_path,
    )
    history_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in history) + "\n",
        encoding="utf-8",
    )
    history_markdown_path.write_text(
        _build_history_markdown(config.experiment_name, history),
        encoding="utf-8",
    )
    history_plot = render_semantic_history_plot(
        run_name=config.experiment_name,
        history_path=history_path,
        output_dir=run_dir,
    )
    report_path = run_dir / f"{config.experiment_name}_semantic_training_eval.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Semantic Training Evaluation: {config.experiment_name}",
                "",
                f"- Accuracy: `{metrics.accuracy:.6f}`",
                f"- Macro-F1: `{metrics.macro_f1:.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "history_path": str(history_path),
                "history_plot_path": str(history_plot["png"]),
                "report_path": str(report_path),
                "metrics": asdict(metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return SemanticTrainingResult(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        history_path=history_path,
        history_markdown_path=history_markdown_path,
        history_plot_path=history_plot["png"],
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
            "Run `uv sync --extra train-cu` before training the semantic model."
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


def _build_character_vocab(samples: list[SemanticSample]) -> list[str]:
    counter = Counter("".join(sample.text for sample in samples))
    return sorted(
        character
        for character, count in counter.items()
        if character.strip() and count > 0
    )


class _SemanticTorchDataset:
    def __init__(
        self,
        samples: list[SemanticSample],
        vocabulary: list[str],
        label_to_index: dict[str, int],
    ) -> None:
        self.samples = samples
        self.vocab_index = {character: index for index, character in enumerate(vocabulary)}
        self.label_to_index = label_to_index
        self.vocab_size = len(vocabulary)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        torch = _import_torch()
        sample = self.samples[index]
        vector = np.zeros((self.vocab_size,), dtype=np.float32)
        for character in sample.text:
            if character in self.vocab_index:
                vector[self.vocab_index[character]] += 1.0
        label = self.label_to_index[sample.semantic_class]
        return torch.tensor(vector, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def _evaluate_semantic_model(
    model: Any,
    loader: Any,
    device: str,
    labels: list[str],
    torch: Any,
    *,
    epoch: int,
    total_epochs: int,
) -> SemanticScoreSummary:
    predictions: list[str] = []
    targets: list[str] = []
    model.eval()
    progress = ProgressBar(
        total=len(loader),
        description=f"语义验证 第 {epoch}/{total_epochs} 轮",
    )
    with torch.no_grad():
        for batch_index, (features, batch_labels) in enumerate(loader, start=1):
            logits = model(features.to(device))
            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(labels[index] for index in batch_predictions)
            targets.extend(labels[index] for index in batch_labels.tolist())
            progress.update(batch_index, metrics={"samples": len(targets)})
    progress.finish(metrics={"samples": len(targets)})
    return compute_semantic_scores(predictions=predictions, targets=targets)


def _build_history_markdown(experiment_name: str, history: list[dict[str, float | int]]) -> str:
    lines = [
        f"# Training History: {experiment_name}",
        "",
        "| Epoch | Train Loss | Val Accuracy | Val Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for record in history:
        lines.append(
            f"| {record['epoch']} | {record['train_loss']:.6f} | "
            f"{record['val_accuracy']:.6f} | {record['val_macro_f1']:.6f} |"
        )
    return "\n".join(lines) + "\n"
