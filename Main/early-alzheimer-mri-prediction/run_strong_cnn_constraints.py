from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.transforms import InterpolationMode

from run_plan_d_solution import EXPECTED_SPLIT_COUNTS, list_image_rows


CONSTRAINTS = {
    "no_type1_limit": None,
    "type1_le_0_05": 0.05,
    "type1_le_0_025": 0.025,
}
OUTPUT_DIR = Path("solutions") / "strong_cnn_constraints"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Winner:
    candidate: str
    epoch: int
    phase: str
    threshold: float
    metrics: dict[str, float | int]
    score_key: tuple[float, ...]
    state_dict: dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constraint-aware pretrained CNN for dementia screening."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--losses", nargs="+", choices=["bce", "focal"], default=["bce", "focal"])
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BinaryMRIDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform: transforms.Compose):
        self.paths = rows["path"].tolist()
        self.labels = (rows["label"].to_numpy(dtype=np.int64) != 0).astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, torch.tensor(self.labels[index], dtype=torch.float32)


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.025, 0.025),
                scale=(0.97, 1.03),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.08, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class SpatialMobileNetV3(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        feature_layers = list(base.features.children())
        self.frozen_features = nn.Sequential(*feature_layers[:-3])
        self.tune_features = nn.Sequential(*feature_layers[-3:])
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        final_channels = int(base.classifier[0].in_features)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(final_channels * 2 * 2, 192),
            nn.Hardswish(),
            nn.Dropout(0.15),
            nn.Linear(192, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.frozen_features(images)
        features = self.tune_features(features)
        return self.head(self.pool(features))


def build_model(pretrained: bool) -> nn.Module:
    return SpatialMobileNetV3(pretrained)


def configure_phase(model: nn.Module, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    if phase == "finetune":
        for parameter in model.tune_features.parameters():
            parameter.requires_grad = True


def set_training_modes(model: nn.Module, phase: str) -> None:
    model.train()
    model.frozen_features.eval()
    if phase == "head":
        model.tune_features.eval()


def make_optimizer(model: nn.Module, phase: str) -> torch.optim.Optimizer:
    if phase == "head":
        return torch.optim.AdamW(model.head.parameters(), lr=1.5e-3, weight_decay=2e-4)
    return torch.optim.AdamW(
        [
            {"params": model.tune_features.parameters(), "lr": 1.5e-4},
            {"params": model.head.parameters(), "lr": 4e-4},
        ],
        weight_decay=2e-4,
    )


def binary_loss(logits: torch.Tensor, targets: torch.Tensor, kind: str) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if kind == "focal":
        probabilities = torch.sigmoid(logits)
        correct_probability = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
        losses = losses * (1.0 - correct_probability).pow(1.5)
    return losses.mean()


def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    loss_sum = 0.0
    seen = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(1)
            scores = torch.sigmoid(logits)
            if tta:
                flipped_scores = torch.sigmoid(model(torch.flip(images, dims=(3,))).squeeze(1))
                scores = 0.5 * (scores + flipped_scores)
            loss_sum += float(F.binary_cross_entropy(scores, labels, reduction="sum").item())
            seen += int(labels.numel())
            score_parts.append(scores.cpu().numpy())
            label_parts.append(labels.cpu().numpy().astype(np.int64))
    scores = np.concatenate(score_parts)
    labels = np.concatenate(label_parts)
    if not np.isfinite(scores).all():
        raise ValueError("CNN produced non-finite validation scores.")
    return scores, labels, loss_sum / max(seen, 1)


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    eps_up = np.nextafter(scores, np.inf)
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], scores, eps_up)))


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    cap: float | None,
) -> tuple[float, dict[str, float | int]]:
    thresholds = threshold_candidates(scores)
    predictions = scores[None, :] >= thresholds[:, None]
    positives = labels == 1
    negatives = ~positives
    tp = predictions[:, positives].sum(axis=1)
    fn = positives.sum() - tp
    fp = predictions[:, negatives].sum(axis=1)
    tn = negatives.sum() - fp
    type1 = fp / max(int(negatives.sum()), 1)
    eligible = np.ones(thresholds.shape[0], dtype=bool) if cap is None else type1 <= cap + 1e-12
    eligible_indices = np.flatnonzero(eligible)
    order = np.lexsort(
        (
            -thresholds[eligible_indices],
            fp[eligible_indices],
            fn[eligible_indices],
        )
    )
    index = int(eligible_indices[order[0]])
    selected_prediction = predictions[index].astype(np.int64)
    metrics = binary_metrics(labels, selected_prediction)
    return float(thresholds[index]), metrics


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    negative_count = tn + fp
    positive_count = tp + fn
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "type1_error": float(fp / max(negative_count, 1)),
        "type2_error": float(fn / max(positive_count, 1)),
        "specificity": float(tn / max(negative_count, 1)),
        "sensitivity": float(tp / max(positive_count, 1)),
        "accuracy": float((tn + tp) / max(negative_count + positive_count, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def selection_key(metrics: dict[str, float | int], partial_auc: float, auc: float, loss: float) -> tuple[float, ...]:
    return (
        -float(metrics["type2_error"]),
        -float(metrics["type1_error"]),
        partial_auc,
        auc,
        -loss,
    )


def train_candidate(
    candidate: str,
    loss_kind: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    head_epochs: int,
    finetune_epochs: int,
    tta: bool,
    winners: dict[str, Winner],
) -> list[dict[str, object]]:
    model = build_model(pretrained=True).to(device)
    history: list[dict[str, object]] = []
    epoch_number = 0
    for phase, phase_epochs in (("head", head_epochs), ("finetune", finetune_epochs)):
        if phase_epochs == 0:
            continue
        configure_phase(model, phase)
        optimizer = make_optimizer(model, phase)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)
        for _ in range(phase_epochs):
            epoch_number += 1
            set_training_modes(model, phase)
            train_loss_sum = 0.0
            seen = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images).squeeze(1)
                loss = binary_loss(logits, labels, loss_kind)
                loss.backward()
                nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
                optimizer.step()
                train_loss_sum += float(loss.item()) * int(labels.numel())
                seen += int(labels.numel())
            scheduler.step()

            val_scores, val_labels, val_loss = predict_scores(model, val_loader, device, tta)
            auc = float(roc_auc_score(val_labels, val_scores))
            partial_auc = float(roc_auc_score(val_labels, val_scores, max_fpr=0.05))
            average_precision = float(average_precision_score(val_labels, val_scores))
            row_base: dict[str, object] = {
                "candidate": candidate,
                "loss": loss_kind,
                "epoch": epoch_number,
                "phase": phase,
                "train_loss": train_loss_sum / max(seen, 1),
                "val_loss": val_loss,
                "val_roc_auc": auc,
                "val_partial_auc_fpr_0_05": partial_auc,
                "val_average_precision": average_precision,
            }
            for constraint, cap in CONSTRAINTS.items():
                threshold, metrics = select_threshold(val_labels, val_scores, cap)
                row = dict(row_base)
                row.update({"constraint": constraint, "threshold": threshold})
                row.update({f"val_{name}": value for name, value in metrics.items()})
                history.append(row)
                key = selection_key(metrics, partial_auc, auc, val_loss)
                if constraint not in winners or key > winners[constraint].score_key:
                    winners[constraint] = Winner(
                        candidate=candidate,
                        epoch=epoch_number,
                        phase=phase,
                        threshold=threshold,
                        metrics=metrics,
                        score_key=key,
                        state_dict=state_to_cpu(model),
                    )
            cap5 = next(row for row in history[-3:] if row["constraint"] == "type1_le_0_05")
            cap25 = next(row for row in history[-3:] if row["constraint"] == "type1_le_0_025")
            print(
                f"{candidate} epoch={epoch_number:02d} phase={phase} "
                f"auc={auc:.4f} pauc05={partial_auc:.4f} "
                f"sens@5%={float(cap5['val_sensitivity']):.4f} "
                f"sens@2.5%={float(cap25['val_sensitivity']):.4f}",
                flush=True,
            )
    del model
    return history


def save_binary_confusion(matrix: np.ndarray, csv_path: Path, png_path: Path, title: str) -> None:
    frame = pd.DataFrame(
        matrix,
        index=["actual_non_demented", "actual_dementia"],
        columns=["pred_non_demented", "pred_dementia"],
    )
    frame.to_csv(csv_path)
    plt.figure(figsize=(5.2, 4.2))
    sns.heatmap(frame, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def dataframe_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    set_seeds(args.seed)
    project_root = args.project_root.resolve()
    output_dir = project_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    tta = not args.no_tta

    rows = {split: list_image_rows(project_root, split) for split in ("train", "val", "test")}
    for split, frame in rows.items():
        if len(frame) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"Unexpected {split} count: {len(frame)}")

    train_transform, eval_transform = make_transforms(args.image_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        BinaryMRIDataset(rows["train"], train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        BinaryMRIDataset(rows["val"], eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    winners: dict[str, Winner] = {}
    history_rows: list[dict[str, object]] = []
    for loss_kind in args.losses:
        candidate = f"mobilenet_v3_large_spatial_{loss_kind}"
        history_rows.extend(
            train_candidate(
                candidate,
                loss_kind,
                train_loader,
                val_loader,
                device,
                args.head_epochs,
                args.finetune_epochs,
                tta,
                winners,
            )
        )
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "validation_candidate_history.csv", index=False)

    test_loader = DataLoader(
        BinaryMRIDataset(rows["test"], eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_rows: list[dict[str, object]] = []
    cached_predictions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for constraint, winner in winners.items():
        winner_id = (winner.candidate, winner.epoch)
        if winner_id not in cached_predictions:
            model = build_model(pretrained=False)
            model.load_state_dict(winner.state_dict)
            model.to(device)
            test_scores, test_labels, _ = predict_scores(model, test_loader, device, tta)
            cached_predictions[winner_id] = (test_scores, test_labels)
            del model
        test_scores, test_labels = cached_predictions[winner_id]
        test_predictions = (test_scores >= winner.threshold).astype(np.int64)
        test_metrics = binary_metrics(test_labels, test_predictions)
        row: dict[str, object] = {
            "model": winner.candidate,
            "constraint": constraint,
            "validation_type1_cap": CONSTRAINTS[constraint],
            "selected_epoch": winner.epoch,
            "selected_phase": winner.phase,
            "threshold": winner.threshold,
        }
        row.update({f"val_{name}": value for name, value in winner.metrics.items()})
        row.update({f"test_{name}": value for name, value in test_metrics.items()})
        test_rows.append(row)

        stem = f"{constraint}_{winner.candidate}_epoch_{winner.epoch}"
        checkpoint = {
            "model": winner.candidate,
            "epoch": winner.epoch,
            "threshold": winner.threshold,
            "constraint": constraint,
            "state_dict": winner.state_dict,
        }
        torch.save(checkpoint, output_dir / f"checkpoint_{stem}.pt")
        matrix = confusion_matrix(test_labels, test_predictions, labels=[0, 1])
        save_binary_confusion(
            matrix,
            output_dir / f"test_confusion_matrix_{stem}.csv",
            output_dir / f"test_confusion_matrix_{stem}.png",
            f"Test: {constraint}",
        )
        predictions = rows["test"][["relative_path", "filename", "actual", "label"]].copy()
        predictions["actual_binary"] = test_labels
        predictions["dementia_score"] = test_scores
        predictions["threshold"] = winner.threshold
        predictions["predicted_binary"] = test_predictions
        predictions.to_csv(output_dir / f"test_predictions_{stem}.csv", index=False)

    selected = pd.DataFrame(test_rows).sort_values("constraint")
    selected.to_csv(output_dir / "selected_test_results.csv", index=False)

    previous_path = project_root / "solutions" / "type2_constraints" / "constraint_comparison.csv"
    comparison = selected.copy()
    comparison["source"] = "strong_pretrained_cnn"
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        previous = previous.rename(columns={"type1_cap": "validation_type1_cap"})
        keep = [
            "model",
            "constraint",
            "validation_type1_cap",
            "threshold",
            "test_type1_error",
            "test_type2_error",
            "test_specificity",
            "test_dementia_sensitivity",
            "test_balanced_accuracy",
        ]
        previous = previous[keep].rename(columns={"test_dementia_sensitivity": "test_sensitivity"})
        previous["source"] = "previous_experiment"
        current_keep = [column for column in keep if column in comparison.columns]
        current = comparison[current_keep + ["source"]]
        comparison = pd.concat([previous, current], ignore_index=True, sort=False)
    comparison.to_csv(output_dir / "comparison_with_previous.csv", index=False)

    compact_columns = [
        "model",
        "constraint",
        "threshold",
        "val_type1_error",
        "val_type2_error",
        "test_type1_error",
        "test_type2_error",
        "test_sensitivity",
        "test_specificity",
        "test_balanced_accuracy",
    ]
    report_table = selected[compact_columns]
    report = [
        "# Strong CNN Constraint Experiment",
        "",
        "Positive class: Very Mild Demented or Mild Demented. Thresholds and checkpoints were selected using validation only.",
        "",
        dataframe_markdown(report_table),
        "",
        "The test split was evaluated only after the winning validation checkpoint and threshold were fixed for each constraint.",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf8")
    manifest = {
        "seed": args.seed,
        "device": str(device),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "losses": args.losses,
        "tta_horizontal_flip": tta,
        "backbone": "ImageNet-pretrained MobileNetV3-Large",
        "head": "2x2 spatial pooling, 3840->192->1",
        "selection": "minimum validation Type II error under each Type I cap; ties prefer lower Type I, pAUC@5%, ROC-AUC, and validation loss",
        "elapsed_seconds": time.perf_counter() - start,
        "torch": torch.__version__,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(f"Output directory: {output_dir}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
