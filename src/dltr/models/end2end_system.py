from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dltr.models.detection.dataset import load_detection_samples
from dltr.models.detection.inference import DetectionPrediction, _decode_detection_map
from dltr.models.detection.metrics import compute_detection_scores
from dltr.models.detection.scaffold import DetectionRunConfig
from dltr.models.detection.trainer import (
    _build_detection_objective,
    _import_torch,
    _prepare_detection_image,
    _select_device,
    _TorchDetectionDataset,
)
from dltr.models.recognition.charset import CharacterVocabulary
from dltr.models.recognition.config import RecognitionExperimentConfig, SecondPassConfig
from dltr.models.recognition.dataset import load_recognition_samples
from dltr.models.recognition.evaluation import RecognitionMetrics
from dltr.models.recognition.metrics import compute_recognition_scores
from dltr.models.recognition.preprocessing import (
    RecognitionPreprocessConfig,
    prepare_recognition_image,
)
from dltr.models.recognition.refinement import (
    QualitySignals,
    second_pass_reasons,
    should_apply_second_pass,
)
from dltr.models.recognition.trainer import (
    _autocast_context,
    _build_grad_scaler,
    _build_runtime_optimizations,
    _collate_ctc_batch,
    _configure_cuda_backend,
    _TorchRecognitionDataset,
)
from dltr.post_ocr.classification import analyze_scene_text
from dltr.post_ocr.slots import extract_post_ocr_slots
from dltr.project import ProjectPaths
from dltr.terminal import ProgressBar
from dltr.torch_checkpoint import load_torch_checkpoint


@dataclass(frozen=True)
class UnifiedLineResult:
    line_id: str
    polygon: list[int]
    text: str
    recognition_confidence: float
    analysis_label: str
    analysis_confidence: float
    slots: Any
    second_pass_applied: bool = False
    second_pass_reasons: list[str] | None = None
    first_pass_text: str = ""
    first_pass_confidence: float = 0.0
    second_pass_text: str | None = None
    second_pass_confidence: float | None = None


@dataclass(frozen=True)
class EndToEndMultitaskTrainingResult:
    run_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    history_path: Path
    summary_path: Path
    detector_proxy_summary_path: Path
    recognizer_proxy_summary_path: Path
    detector_metrics: dict[str, float]
    recognizer_metrics: dict[str, float]


class UnifiedEndToEndPredictorSession:
    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        vocabulary: CharacterVocabulary,
        device: str,
        detector_height: int,
        detector_width: int,
        recognition_height: int,
        recognition_width: int,
        recognition_model_name: str,
        second_pass: SecondPassConfig,
    ) -> None:
        self._torch = torch
        self._model = model
        self._vocabulary = vocabulary
        self._device = device
        self._detector_height = detector_height
        self._detector_width = detector_width
        self._recognition_height = recognition_height
        self._recognition_width = recognition_width
        self._recognition_model_name = recognition_model_name
        self._second_pass = second_pass

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> UnifiedEndToEndPredictorSession:
        torch = _import_torch()
        checkpoint = load_torch_checkpoint(torch, checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        detector_config = config.get("detector", {})
        recognizer_config = config.get("recognizer", {})
        detector_height = int(detector_config.get("image_height", 256))
        detector_width = int(detector_config.get("image_width", 256))
        recognition_height = int(recognizer_config.get("image_height", 48))
        recognition_width = int(recognizer_config.get("image_width", 320))
        recognition_model_name = (
            str(recognizer_config.get("model_name", "transformer")).strip().lower() or "transformer"
        )
        charset_file = _resolve_charset_path(
            checkpoint_path=checkpoint_path,
            charset_raw=str(recognizer_config.get("charset_file", "")),
        )
        vocabulary = CharacterVocabulary.from_file(charset_file)
        raw_second_pass = recognizer_config.get("second_pass", {})
        second_pass = SecondPassConfig(
            enabled=bool(raw_second_pass.get("enabled", True)),
            confidence_threshold=float(raw_second_pass.get("confidence_threshold", 0.78)),
            max_blur_score=float(raw_second_pass.get("max_blur_score", 0.45)),
            min_contrast_score=float(raw_second_pass.get("min_contrast_score", 0.35)),
            min_text_length=int(raw_second_pass.get("min_text_length", 2)),
            min_aspect_ratio=float(raw_second_pass.get("min_aspect_ratio", 0.2)),
            max_aspect_ratio=float(raw_second_pass.get("max_aspect_ratio", 20.0)),
        )
        device = _select_device(torch, str(config.get("device", "auto")))
        model = _build_multitask_model(
            nn=torch.nn,
            vocabulary_size=vocabulary.size,
            recognition_model_name=recognition_model_name,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(
            torch=torch,
            model=model,
            vocabulary=vocabulary,
            device=device,
            detector_height=detector_height,
            detector_width=detector_width,
            recognition_height=recognition_height,
            recognition_width=recognition_width,
            recognition_model_name=recognition_model_name,
            second_pass=second_pass,
        )

    def infer_image(
        self,
        image: np.ndarray,
        *,
        threshold: float,
        min_area: float,
    ) -> dict[str, object]:
        original = image.copy()
        total_started_at = time.perf_counter()
        detector_started_at = time.perf_counter()
        detections = self._predict_detections(original, threshold=threshold, min_area=min_area)
        detector_latency_ms = (time.perf_counter() - detector_started_at) * 1000.0

        preview = original.copy()
        line_results: list[UnifiedLineResult] = []
        cropped_items: list[tuple[int, DetectionPrediction, np.ndarray]] = []
        for index, detection in enumerate(detections):
            crop = _crop_polygon(original, detection.polygon)
            if crop is None or crop.size == 0:
                continue
            cropped_items.append((index, detection, crop))

        recognizer_started_at = time.perf_counter()
        first_pass_predictions = self.recognize_images([crop for _, _, crop in cropped_items])
        recognizer_latency_ms = (time.perf_counter() - recognizer_started_at) * 1000.0

        second_pass_indexes: list[int] = []
        second_pass_crops: list[np.ndarray] = []
        prepared: list[dict[str, object]] = []
        for item_index, ((index, detection, crop), first_pass) in enumerate(
            zip(cropped_items, first_pass_predictions, strict=True)
        ):
            quality = _estimate_quality_signals(crop)
            reasons = second_pass_reasons(
                first_pass.confidence,
                first_pass.text,
                quality,
                self._second_pass,
            )
            prepared.append(
                {
                    "index": index,
                    "detection": detection,
                    "first_pass": first_pass,
                    "reasons": reasons,
                    "crop": crop,
                }
            )
            if should_apply_second_pass(
                first_pass.confidence,
                first_pass.text,
                quality,
                self._second_pass,
            ):
                second_pass_indexes.append(item_index)
                second_pass_crops.append(_apply_second_pass_enhancement(crop))

        second_pass_started_at = time.perf_counter()
        second_pass_predictions = (
            self.recognize_images(second_pass_crops) if second_pass_crops else []
        )
        second_pass_latency_ms = (time.perf_counter() - second_pass_started_at) * 1000.0
        second_pass_map = dict(zip(second_pass_indexes, second_pass_predictions, strict=True))

        post_ocr_started_at = time.perf_counter()
        for prepared_index, item in enumerate(prepared):
            detection = item["detection"]
            first_pass = item["first_pass"]
            reasons = item["reasons"]
            recognition = first_pass
            second_pass_applied = False
            second_pass_text: str | None = None
            second_pass_confidence: float | None = None
            if prepared_index in second_pass_map:
                second_pass_prediction = second_pass_map[prepared_index]
                second_pass_applied = True
                second_pass_text = second_pass_prediction.text
                second_pass_confidence = second_pass_prediction.confidence
                if (
                    second_pass_prediction.confidence >= first_pass.confidence
                    or not first_pass.text.strip()
                ):
                    recognition = second_pass_prediction
            analysis = analyze_scene_text(recognition.text)
            slots = extract_post_ocr_slots(recognition.text)
            result = UnifiedLineResult(
                line_id=f"line-{item['index']}",
                polygon=detection.polygon,
                text=recognition.text,
                recognition_confidence=recognition.confidence,
                analysis_label=analysis.label,
                analysis_confidence=analysis.confidence,
                slots=slots,
                second_pass_applied=second_pass_applied,
                second_pass_reasons=list(reasons),
                first_pass_text=first_pass.text,
                first_pass_confidence=first_pass.confidence,
                second_pass_text=second_pass_text,
                second_pass_confidence=second_pass_confidence,
            )
            line_results.append(result)
            _draw_polygon(preview, detection.polygon, f"{recognition.text[:12]} | {analysis.label}")

        post_ocr_latency_ms = (time.perf_counter() - post_ocr_started_at) * 1000.0
        total_latency_ms = (time.perf_counter() - total_started_at) * 1000.0
        fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0
        return {
            "preview": preview,
            "line_results": line_results,
            "runtime_metrics": {
                "total_latency_ms": total_latency_ms,
                "detector_latency_ms": detector_latency_ms,
                "recognizer_latency_ms": recognizer_latency_ms,
                "second_pass_latency_ms": second_pass_latency_ms,
                "post_ocr_latency_ms": post_ocr_latency_ms,
                "fps": fps,
            },
        }

    def recognize_images(self, images: list[np.ndarray]) -> list[_RecognitionPrediction]:
        if not images:
            return []
        batch = np.stack(
            [
                _prepare_multitask_recognition_image(
                    image,
                    target_height=self._recognition_height,
                    target_width=self._recognition_width,
                )
                for image in images
            ],
            axis=0,
        )
        tensor = self._torch.tensor(batch, dtype=self._torch.float32).to(self._device)
        with self._torch.no_grad():
            log_probs = self._model.forward_recognition(tensor)
            probs = self._torch.exp(log_probs)
            greedy_batch = probs.argmax(dim=2).permute(1, 0)
            confidence_batch = probs.max(dim=2).values.permute(1, 0)
        predictions: list[_RecognitionPrediction] = []
        for greedy, confidence in zip(greedy_batch, confidence_batch, strict=True):
            predictions.append(
                _RecognitionPrediction(
                    text=self._vocabulary.decode_greedy(greedy.tolist()),
                    confidence=float(confidence.mean().item()) if confidence.numel() else 0.0,
                )
            )
        return predictions

    def _predict_detections(
        self,
        image: np.ndarray,
        *,
        threshold: float,
        min_area: float,
    ) -> list[DetectionPrediction]:
        original_height, original_width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        resized = _prepare_detection_image(
            rgb_image,
            target_height=self._detector_height,
            target_width=self._detector_width,
        )
        tensor = (
            self._torch.tensor(
                np.transpose(resized, (2, 0, 1)),
                dtype=self._torch.float32,
            )
            .unsqueeze(0)
            .to(self._device)
        )
        with self._torch.no_grad():
            probs = (
                self._torch.sigmoid(self._model.forward_detection(tensor))
                .squeeze()
                .cpu()
                .numpy()
            )
        return _decode_detection_map(
            probs=probs,
            threshold=threshold,
            min_area=min_area,
            original_width=original_width,
            original_height=original_height,
            model_width=self._detector_width,
            model_height=self._detector_height,
        )


@dataclass(frozen=True)
class _RecognitionPrediction:
    text: str
    confidence: float


def train_end2end_multitask_system(
    detector_config: DetectionRunConfig,
    recognizer_config: RecognitionExperimentConfig,
    *,
    paths: ProjectPaths | None = None,
    run_id: str | None = None,
    output_dir: Path | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> EndToEndMultitaskTrainingResult:
    torch = _import_torch()
    nn = torch.nn
    optim = torch.optim
    data_utils = torch.utils.data

    project_paths = paths or ProjectPaths.from_root()
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir or (project_paths.artifacts / "end2end" / resolved_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    det_train_path = _resolve_required_path(project_paths.root, detector_config.train_manifest)
    det_val_path = _resolve_required_path(project_paths.root, detector_config.validation_manifest)
    rec_train_path = (project_paths.root / recognizer_config.dataset_manifest).resolve()
    rec_val_path = (project_paths.root / recognizer_config.validation_manifest).resolve()
    charset_path = (project_paths.root / recognizer_config.charset_file).resolve()
    if not rec_train_path.exists():
        raise FileNotFoundError(f"Recognition manifest not found: {rec_train_path}")
    if not rec_val_path.exists():
        raise FileNotFoundError(f"Recognition validation manifest not found: {rec_val_path}")
    if not charset_path.exists():
        raise FileNotFoundError(f"Charset file not found: {charset_path}")

    vocabulary = CharacterVocabulary.from_file(charset_path)
    det_train_samples = load_detection_samples(det_train_path)
    det_val_samples = load_detection_samples(det_val_path)
    rec_train_samples = load_recognition_samples(rec_train_path)
    rec_val_samples = load_recognition_samples(rec_val_path)
    if not det_train_samples or not det_val_samples:
        raise ValueError("Detection manifests must contain valid training and validation samples")
    if not rec_train_samples or not rec_val_samples:
        raise ValueError("Recognition manifests must contain valid training and validation samples")

    det_train_dataset = _TorchDetectionDataset(
        det_train_samples,
        image_height=detector_config.image_height,
        image_width=detector_config.image_width,
        multi_scale_augmentation=detector_config.multi_scale_augmentation,
    )
    det_val_dataset = _TorchDetectionDataset(
        det_val_samples,
        image_height=detector_config.image_height,
        image_width=detector_config.image_width,
    )
    rec_train_dataset = _TorchRecognitionDataset(
        rec_train_samples,
        vocabulary,
        preprocess_config=recognizer_config.preprocess,
    )
    rec_val_dataset = _TorchRecognitionDataset(
        rec_val_samples,
        vocabulary,
        preprocess_config=recognizer_config.preprocess,
    )

    device = _select_device(torch, recognizer_config.device)
    runtime = _build_runtime_optimizations(device=device, num_workers=recognizer_config.num_workers)
    _configure_cuda_backend(torch, device=device)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    det_train_loader = data_utils.DataLoader(
        det_train_dataset,
        batch_size=detector_config.batch_size,
        shuffle=True,
        num_workers=detector_config.num_workers,
    )
    det_val_loader = data_utils.DataLoader(
        det_val_dataset,
        batch_size=max(1, min(detector_config.batch_size, 8)),
        shuffle=False,
        num_workers=detector_config.num_workers,
    )
    rec_train_loader = data_utils.DataLoader(
        rec_train_dataset,
        batch_size=recognizer_config.batch_size,
        shuffle=True,
        num_workers=recognizer_config.num_workers,
        collate_fn=_collate_ctc_batch,
        **runtime.loader_kwargs,
    )
    rec_val_loader = data_utils.DataLoader(
        rec_val_dataset,
        batch_size=max(1, min(recognizer_config.batch_size, 32)),
        shuffle=False,
        num_workers=recognizer_config.num_workers,
        collate_fn=_collate_ctc_batch,
        **runtime.loader_kwargs,
    )

    model = _build_multitask_model(
        nn=nn,
        vocabulary_size=vocabulary.size,
        recognition_model_name=recognizer_config.model_name,
    ).to(device)
    detection_criterion = _build_detection_objective(nn)
    recognition_criterion = nn.CTCLoss(blank=vocabulary.blank_index, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=min(detector_config.learning_rate, recognizer_config.learning_rate),
        weight_decay=1e-4,
    )
    scaler = _build_grad_scaler(torch, use_amp=runtime.use_amp)
    best_score = float("-inf")
    best_checkpoint_path = run_dir / "best.pt"
    checkpoint_path = run_dir / "last.pt"
    history_path = run_dir / "training_history.jsonl"
    history: list[dict[str, float | int]] = []
    detector_metrics = {"precision": 0.0, "recall": 0.0, "hmean": 0.0}
    recognizer_metrics = RecognitionMetrics(
        samples=0,
        word_accuracy=0.0,
        cer=1.0,
        ned=1.0,
        mean_edit_distance=1.0,
    )

    total_epochs = max(detector_config.epochs, recognizer_config.epochs)
    rec_train_iter = _endless_loader(rec_train_loader)
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_steps = 0
        det_batches = len(det_train_loader)
        planned_steps = (
            det_batches
            if max_train_batches is None
            else min(det_batches, max_train_batches)
        )
        progress = ProgressBar(
            total=planned_steps,
            description=f"端到端多任务训练 第 {epoch}/{total_epochs} 轮",
        )
        for batch_index, (det_images, det_masks) in enumerate(det_train_loader, start=1):
            if max_train_batches is not None and batch_index > max_train_batches:
                break
            rec_batch = next(rec_train_iter)
            det_images = det_images.to(device)
            det_masks = det_masks.to(device)
            rec_images = rec_batch["images"].to(device, non_blocking=runtime.non_blocking)
            rec_targets = rec_batch["targets"].to(device, non_blocking=runtime.non_blocking)
            rec_target_lengths = rec_batch["target_lengths"].to(
                device,
                non_blocking=runtime.non_blocking,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(torch, use_amp=runtime.use_amp):
                det_logits = model.forward_detection(det_images)
                rec_log_probs = model.forward_recognition(rec_images)
                det_loss = detection_criterion(det_logits, det_masks)
                rec_input_lengths = torch.full(
                    size=(rec_images.size(0),),
                    fill_value=rec_log_probs.size(0),
                    dtype=torch.long,
                    device=device,
                )
                rec_loss = recognition_criterion(
                    rec_log_probs.float(),
                    rec_targets,
                    rec_input_lengths,
                    rec_target_lengths,
                )
                total_loss = det_loss + rec_loss
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            train_loss_total += float(total_loss.item())
            train_steps += 1
            progress.update(batch_index, metrics={"loss": float(total_loss.item())})
        progress.finish(metrics={"avg_loss": train_loss_total / max(train_steps, 1)})

        detector_metrics = _evaluate_detection_branch(
            model=model,
            loader=det_val_loader,
            device=device,
            torch=torch,
            max_batches=max_val_batches,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        recognizer_metrics = _evaluate_recognition_branch(
            model=model,
            loader=rec_val_loader,
            vocabulary=vocabulary,
            device=device,
            torch=torch,
            max_batches=max_val_batches,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        system_score = (detector_metrics["hmean"] + recognizer_metrics.word_accuracy) / 2.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_total / max(train_steps, 1),
                "val_precision": detector_metrics["precision"],
                "val_recall": detector_metrics["recall"],
                "val_hmean": detector_metrics["hmean"],
                "val_word_accuracy": recognizer_metrics.word_accuracy,
                "val_cer": recognizer_metrics.cer,
                "val_ned": recognizer_metrics.ned,
                "system_score": system_score,
            }
        )
        if system_score >= best_score:
            best_score = system_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "device": recognizer_config.device,
                        "detector": asdict(detector_config),
                        "recognizer": asdict(recognizer_config),
                    },
                    "metrics": {
                        "system_score": system_score,
                        "detector_hmean": detector_metrics["hmean"],
                        "recognizer_word_accuracy": recognizer_metrics.word_accuracy,
                    },
                },
                best_checkpoint_path,
            )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "device": recognizer_config.device,
                "detector": asdict(detector_config),
                "recognizer": asdict(recognizer_config),
            },
            "metrics": {
                "detector_hmean": detector_metrics["hmean"],
                "recognizer_word_accuracy": recognizer_metrics.word_accuracy,
            },
        },
        checkpoint_path,
    )
    history_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in history) + "\n",
        encoding="utf-8",
    )
    detector_proxy_summary_path = _write_proxy_summary(
        run_dir=project_paths.artifacts
        / "detection"
        / f"{detector_config.experiment_name}_multitask"
        / resolved_run_id,
        run_id=resolved_run_id,
        metrics=detector_metrics,
        best_checkpoint_path=best_checkpoint_path,
    )
    recognizer_proxy_summary_path = _write_proxy_summary(
        run_dir=project_paths.artifacts
        / "checkpoints"
        / "recognition"
        / f"{recognizer_config.experiment_name}_multitask"
        / resolved_run_id,
        run_id=resolved_run_id,
        metrics={
            "word_accuracy": recognizer_metrics.word_accuracy,
            "cer": recognizer_metrics.cer,
            "ned": recognizer_metrics.ned,
            "mean_edit_distance": recognizer_metrics.mean_edit_distance,
            "latency_ms": recognizer_metrics.latency_ms or 0.0,
        },
        best_checkpoint_path=best_checkpoint_path,
    )
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "coordination_strategy": "shared_backbone_multitask_training",
                "unified_checkpoint_path": str(best_checkpoint_path),
                "best_checkpoint_path": str(best_checkpoint_path),
                "history_path": str(history_path),
                "detector_metrics": detector_metrics,
                "recognizer_metrics": asdict(recognizer_metrics),
                "detector_proxy_summary_path": str(detector_proxy_summary_path),
                "recognizer_proxy_summary_path": str(recognizer_proxy_summary_path),
                "metrics": {
                    "system_score": best_score,
                    "detector_hmean": detector_metrics["hmean"],
                    "recognizer_word_accuracy": recognizer_metrics.word_accuracy,
                },
                "created_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return EndToEndMultitaskTrainingResult(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        history_path=history_path,
        summary_path=summary_path,
        detector_proxy_summary_path=detector_proxy_summary_path,
        recognizer_proxy_summary_path=recognizer_proxy_summary_path,
        detector_metrics=detector_metrics,
        recognizer_metrics={
            "word_accuracy": recognizer_metrics.word_accuracy,
            "cer": recognizer_metrics.cer,
            "ned": recognizer_metrics.ned,
            "mean_edit_distance": recognizer_metrics.mean_edit_distance,
            "latency_ms": recognizer_metrics.latency_ms or 0.0,
        },
    )


def _evaluate_detection_branch(
    *,
    model: Any,
    loader: Any,
    device: str,
    torch: Any,
    max_batches: int | None,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.eval()
    aggregated = {"precision": 0.0, "recall": 0.0, "hmean": 0.0}
    batches = 0
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = ProgressBar(total=total, description=f"端到端检测验证 第 {epoch}/{total_epochs} 轮")
    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            logits = model.forward_detection(images.to(device))
            probs = torch.sigmoid(logits)
            metrics = compute_detection_scores(probs.cpu(), masks.cpu())
            for key, value in metrics.items():
                aggregated[key] += value
            batches += 1
            progress.update(batch_index, metrics={"hmean": metrics["hmean"]})
    progress.finish()
    if batches == 0:
        return aggregated
    return {key: value / batches for key, value in aggregated.items()}


def _evaluate_recognition_branch(
    *,
    model: Any,
    loader: Any,
    vocabulary: CharacterVocabulary,
    device: str,
    torch: Any,
    max_batches: int | None,
    epoch: int,
    total_epochs: int,
) -> RecognitionMetrics:
    model.eval()
    predictions: list[str] = []
    targets: list[str] = []
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = ProgressBar(total=total, description=f"端到端识别验证 第 {epoch}/{total_epochs} 轮")
    started_at = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            images = batch["images"].to(device)
            log_probs = model.forward_recognition(images)
            greedy_indices = log_probs.argmax(dim=2).permute(1, 0)
            for indices, target in zip(greedy_indices, batch["texts"], strict=True):
                predictions.append(vocabulary.decode_greedy(indices.tolist()))
                targets.append(target)
            progress.update(batch_index, metrics={"samples": len(targets)})
    progress.finish(metrics={"samples": len(targets)})
    scores = compute_recognition_scores(predictions, targets) if predictions else None
    latency_ms = ((time.perf_counter() - started_at) / max(1, len(targets))) * 1000.0
    if scores is None:
        return RecognitionMetrics(
            samples=0,
            word_accuracy=0.0,
            cer=1.0,
            ned=1.0,
            mean_edit_distance=1.0,
            latency_ms=latency_ms,
        )
    return RecognitionMetrics(
        samples=scores.samples,
        word_accuracy=scores.word_accuracy,
        cer=scores.cer,
        ned=scores.ned,
        mean_edit_distance=scores.mean_edit_distance,
        latency_ms=latency_ms,
    )


def _write_proxy_summary(
    *,
    run_dir: Path,
    run_id: str,
    metrics: dict[str, float],
    best_checkpoint_path: Path,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "metrics": metrics,
                "best_checkpoint_path": str(best_checkpoint_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


def _endless_loader(loader: Any):
    while True:
        yield from loader


def _resolve_required_path(root: Path, path: Path | None) -> Path:
    if path is None:
        raise ValueError("Required manifest path is not configured")
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _prepare_multitask_recognition_image(
    image: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    preprocess_config = RecognitionPreprocessConfig(
        target_height=target_height,
        target_width=target_width,
        preserve_aspect_ratio=True,
        rotate_vertical_text=True,
        vertical_aspect_threshold=1.2,
        padding_value=255,
    )
    processed, _ = prepare_recognition_image(
        image,
        config=preprocess_config,
    )
    if processed.ndim == 2:
        processed = np.repeat(processed[None, ...], 3, axis=0)
    else:
        processed = np.transpose(processed, (2, 0, 1))
    return processed.astype(np.float32)


def _build_multitask_model(
    *,
    nn: Any,
    vocabulary_size: int,
    recognition_model_name: str,
) -> Any:
    torch = _import_torch()
    functional = torch.nn.functional

    class _SharedBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        def forward(self, images):
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            return self.blocks(images)

    class _RecognitionHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model_name = recognition_model_name
            if recognition_model_name == "crnn":
                self.sequence_model = nn.LSTM(
                    input_size=256,
                    hidden_size=256,
                    num_layers=2,
                    bidirectional=True,
                )
                self.classifier = nn.Linear(512, vocabulary_size)
            else:
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

        def forward(self, features):
            pooled = functional.adaptive_avg_pool2d(features, (1, None)).squeeze(2)
            if self.model_name == "crnn":
                sequence = pooled.permute(2, 0, 1)
                recurrent, _ = self.sequence_model(sequence)
                logits = self.classifier(recurrent)
                return torch.log_softmax(logits, dim=2)
            sequence = pooled.permute(0, 2, 1)
            positional = self.position_embedding[:, : sequence.size(1)]
            encoded = self.encoder(sequence + positional)
            logits = self.classifier(encoded).permute(1, 0, 2)
            return torch.log_softmax(logits, dim=2)

    class _MultitaskModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _SharedBackbone()
            self.detection_head = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1),
            )
            self.recognition_head = _RecognitionHead()

        def forward_detection(self, images):
            features = self.backbone(images)
            logits = self.detection_head(features)
            return functional.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        def forward_recognition(self, images):
            features = self.backbone(images)
            return self.recognition_head(features)

    return _MultitaskModel()


def _resolve_charset_path(*, checkpoint_path: Path, charset_raw: str) -> Path:
    charset_file = Path(charset_raw)
    if charset_file.is_absolute():
        return charset_file
    repo_candidate = Path.cwd() / charset_file
    if repo_candidate.exists():
        return repo_candidate
    return checkpoint_path.resolve().parents[2] / charset_file


def _polygon_points(points: list[int]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _polygon_to_quad(points: list[int]) -> np.ndarray:
    pts = _polygon_points(points)
    if len(points) == 8:
        return pts
    rect = cv2.minAreaRect(pts.astype(np.float32))
    return cv2.boxPoints(rect).astype(np.float32)


def _crop_polygon(image: np.ndarray, polygon: list[int]) -> np.ndarray | None:
    quad = _polygon_to_quad(polygon)
    width_a = np.linalg.norm(quad[2] - quad[3])
    width_b = np.linalg.norm(quad[1] - quad[0])
    height_a = np.linalg.norm(quad[1] - quad[2])
    height_b = np.linalg.norm(quad[0] - quad[3])
    target_width = max(int(round(max(width_a, width_b))), 1)
    target_height = max(int(round(max(height_a, height_b))), 1)
    destination = np.asarray(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(image, transform, (target_width, target_height))


def _estimate_quality_signals(crop: np.ndarray) -> QualitySignals:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur_score = 1.0 / (1.0 + laplacian_var / 100.0)
    contrast_score = min(float(gray.std()) / 64.0, 1.0)
    aspect_ratio = float(gray.shape[1]) / max(float(gray.shape[0]), 1.0)
    return QualitySignals(
        blur_score=blur_score,
        contrast_score=contrast_score,
        aspect_ratio=aspect_ratio,
    )


def _apply_second_pass_enhancement(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(equalized, 1.6, blurred, -0.6, 0.0)


def _draw_polygon(image: np.ndarray, polygon: list[int], label: str) -> None:
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=(0, 180, 0), thickness=2)
    x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 6
    cv2.putText(
        image,
        label,
        (max(x, 0), max(y, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 220),
        1,
        cv2.LINE_AA,
    )
