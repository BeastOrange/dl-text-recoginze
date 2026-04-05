from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def render_recognition_history_plot(
    *,
    run_name: str,
    history_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    history = _load_history(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{run_name}_training_curve.png"
    markdown_path = output_dir / f"{run_name}_training_curve.md"

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    word_accuracy = [float(item["val_word_accuracy"]) for item in history]
    cer = [float(item["val_cer"]) for item in history]
    ned = [float(item["val_ned"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, cer, marker="s", label="Val CER")
    axes[0].set_title("Recognition Loss / CER")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(epochs, word_accuracy, marker="o", label="Val Word Accuracy")
    axes[1].plot(epochs, ned, marker="s", label="Val NED")
    axes[1].set_title("Recognition Accuracy / NED")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    markdown_path.write_text(
        "\n".join(
            [
                f"# Recognition Training Curves: {run_name}",
                "",
                f"- Image: `{png_path.name}`",
                f"- Epochs: `{len(history)}`",
                f"- Best Word Accuracy: `{max(word_accuracy):.6f}`",
                f"- Lowest CER: `{min(cer):.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"png": png_path, "markdown": markdown_path}


def render_detection_history_plot(
    *,
    run_name: str,
    history_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    history = _load_history(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{run_name}_training_curve.png"
    markdown_path = output_dir / f"{run_name}_training_curve.md"

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    precision = [float(item["val_precision"]) for item in history]
    recall = [float(item["val_recall"]) for item in history]
    hmean = [float(item["val_hmean"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, hmean, marker="s", label="Val Hmean")
    axes[0].set_title("Detection Loss / Hmean")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(epochs, precision, marker="o", label="Val Precision")
    axes[1].plot(epochs, recall, marker="s", label="Val Recall")
    axes[1].set_title("Detection Precision / Recall")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    markdown_path.write_text(
        "\n".join(
            [
                f"# Detection Training Curves: {run_name}",
                "",
                f"- Image: `{png_path.name}`",
                f"- Epochs: `{len(history)}`",
                f"- Best Hmean: `{max(hmean):.6f}`",
                f"- Best Precision: `{max(precision):.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"png": png_path, "markdown": markdown_path}


def render_semantic_history_plot(
    *,
    run_name: str,
    history_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    history = _load_history(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{run_name}_training_curve.png"
    markdown_path = output_dir / f"{run_name}_training_curve.md"

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    accuracy = [float(item["val_accuracy"]) for item in history]
    macro_f1 = [float(item["val_macro_f1"]) for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, accuracy, marker="s", label="Val Accuracy")
    axes[0].set_title("Semantic Loss / Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].legend()

    axes[1].plot(epochs, accuracy, marker="o", label="Val Accuracy")
    axes[1].plot(epochs, macro_f1, marker="s", label="Val Macro-F1")
    axes[1].set_title("Semantic Accuracy / Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    markdown_path.write_text(
        "\n".join(
            [
                f"# Semantic Training Curves: {run_name}",
                "",
                f"- Image: `{png_path.name}`",
                f"- Epochs: `{len(history)}`",
                f"- Best Accuracy: `{max(accuracy):.6f}`",
                f"- Best Macro-F1: `{max(macro_f1):.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"png": png_path, "markdown": markdown_path}


def aggregate_training_runs(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    task_name: str,
    primary_metric: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for run_dir in run_dirs:
        summary_path = run_dir / "training_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        records.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "primary_metric": float(metrics.get(primary_metric, 0.0)),
                "metrics": metrics,
                "best_checkpoint_path": payload.get("best_checkpoint_path"),
            }
        )

    records.sort(key=lambda item: item["primary_metric"], reverse=True)
    json_path = output_dir / f"{task_name}_summary.json"
    markdown_path = output_dir / f"{task_name}_summary.md"
    png_path = output_dir / f"{task_name}_summary.png"

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                f"# {task_name.title()} Training Summary",
                "",
                f"- Primary Metric: `{primary_metric}`",
                "",
                "| Run | Primary Metric | Best Checkpoint |",
                "|---|---:|---|",
            ]
            + [
                f"| {item['run_name']} | {item['primary_metric']:.6f} | "
                f"{item['best_checkpoint_path'] or '-'} |"
                for item in records
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([item["run_name"] for item in records], [item["primary_metric"] for item in records])
    ax.set_title(f"{task_name.title()} Primary Metric Comparison")
    ax.set_ylabel(primary_metric)
    ax.set_xlabel("Run")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return {"json": json_path, "markdown": markdown_path, "png": png_path}


def _load_history(history_path: Path) -> list[dict[str, float | int]]:
    return [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
