from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from scipy.stats import beta
from torch.utils.data import DataLoader
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    mobilenet_v3_large,
    resnet18,
)

from run_plan_d_solution import EXPECTED_SPLIT_COUNTS, list_image_rows
from run_strong_cnn_constraints import (
    BinaryMRIDataset,
    CONSTRAINTS,
    binary_metrics,
    dataframe_markdown,
    make_transforms,
    save_binary_confusion,
    select_threshold,
)


OUTPUT_DIR = Path("solutions") / "frozen_cnn_constraint_search"


@dataclass
class Winner:
    candidate: str
    backbone: str
    representation: str
    threshold: float
    metrics: dict[str, float | int]
    score_key: tuple[float, ...]
    scaler: StandardScaler
    pca: PCA | None
    head: object


class SpatialExtractor(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        if backbone == "resnet18":
            base = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.output_channels = 512
        elif backbone == "mobilenet_v3_large":
            base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            self.features = base.features
            self.output_channels = int(base.classifier[0].in_features)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.features(images)), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search deterministic heads on frozen CNN MRI embeddings.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=["resnet18", "mobilenet_v3_large"],
        default=["resnet18", "mobilenet_v3_large"],
    )
    parser.add_argument(
        "--pca-components",
        nargs="+",
        type=int,
        default=[96, 128, 160, 224, 320],
    )
    parser.add_argument(
        "--cap-confidence",
        type=float,
        default=0.95,
        help="One-sided confidence used to guard validation Type I caps; use 0 for empirical caps.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_embeddings(
    backbone: str,
    rows: pd.DataFrame,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    _, eval_transform = make_transforms(image_size)
    loader = DataLoader(
        BinaryMRIDataset(rows, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = SpatialExtractor(backbone).to(device).eval()
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            feature_parts.append(model(images.to(device)).cpu().numpy().astype(np.float32))
            label_parts.append(labels.numpy().astype(np.int64))
            if batch_index % 10 == 0:
                print(f"{backbone}: extracted {min(batch_index * batch_size, len(rows))}/{len(rows)}", flush=True)
    features = np.vstack(feature_parts)
    labels = np.concatenate(label_parts)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite {backbone} embeddings")
    del model
    return features, labels


def load_or_extract(
    cache_path: Path,
    backbone: str,
    rows: pd.DataFrame,
    image_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["features"], cached["labels"]
    features, labels = extract_embeddings(backbone, rows, image_size, batch_size, device)
    np.savez_compressed(cache_path, features=features, labels=labels)
    return features, labels


def build_direct_heads(seed: int) -> list[tuple[str, object]]:
    heads: list[tuple[str, object]] = []
    for c_value in (0.01, 0.1, 1.0):
        heads.append(
            (
                f"direct_logistic_c_{c_value:g}",
                LogisticRegression(C=c_value, max_iter=1500, random_state=seed),
            )
        )
    for c_value in (0.001, 0.01, 0.1):
        heads.append(
            (
                f"direct_linear_svm_c_{c_value:g}",
                LinearSVC(C=c_value, class_weight=None, dual="auto", max_iter=5000, random_state=seed),
            )
        )
    return heads


def build_pca_heads(seed: int, component_count: int) -> list[tuple[str, object]]:
    heads: list[tuple[str, object]] = []
    for c_value in (0.1, 1.0, 10.0):
        heads.append(
            (
                f"pca_logistic_c_{c_value:g}",
                LogisticRegression(C=c_value, max_iter=1500, random_state=seed),
            )
        )
    for c_value in (3.0, 10.0, 30.0):
        for gamma_multiplier in (0.5, 1.0, 2.0):
            gamma = gamma_multiplier / component_count
            heads.append(
                (
                    f"pca_rbf_svm_c_{c_value:g}_gamma_{gamma_multiplier:g}x",
                    SVC(C=c_value, kernel="rbf", gamma=gamma, cache_size=1500),
                )
            )
    heads.append(
        (
            "pca_mlp_96_32",
            MLPClassifier(
                hidden_layer_sizes=(96, 32),
                alpha=1e-3,
                learning_rate_init=7e-4,
                max_iter=350,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=seed,
            ),
        )
    )
    return heads


def model_scores(model: object, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    else:
        scores = model.decision_function(features)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Head produced non-finite scores")
    return scores


def key_for(metrics: dict[str, float | int], partial_auc: float, auc: float, average_precision: float) -> tuple[float, ...]:
    return (
        -float(metrics["type2_error"]),
        -float(metrics["type1_error"]),
        partial_auc,
        auc,
        average_precision,
    )


def binomial_upper_bound(false_positives: int, negatives: int, confidence: float) -> float:
    if confidence <= 0.0:
        return false_positives / max(negatives, 1)
    if false_positives >= negatives:
        return 1.0
    return float(beta.ppf(confidence, false_positives + 1, negatives - false_positives))


def effective_empirical_cap(target_cap: float, negatives: int, confidence: float) -> float:
    if confidence <= 0.0:
        return target_cap
    allowed = [
        false_positives
        for false_positives in range(negatives + 1)
        if binomial_upper_bound(false_positives, negatives, confidence) <= target_cap
    ]
    return max(allowed, default=0) / max(negatives, 1)


def search_backbone(
    backbone: str,
    train_features: np.ndarray,
    y_train: np.ndarray,
    val_features: np.ndarray,
    y_val: np.ndarray,
    pca_components: list[int],
    seed: int,
    cap_confidence: float,
    winners: dict[str, Winner],
) -> list[dict[str, object]]:
    scaler = StandardScaler()
    train_direct = scaler.fit_transform(train_features).astype(np.float32)
    val_direct = scaler.transform(val_features).astype(np.float32)
    rows: list[dict[str, object]] = []

    def evaluate_head(
        head_name: str,
        representation: str,
        head: object,
        train_matrix: np.ndarray,
        val_matrix: np.ndarray,
        fitted_pca: PCA | None,
    ) -> None:
        head.fit(train_matrix, y_train)
        scores = model_scores(head, val_matrix)
        auc = float(roc_auc_score(y_val, scores))
        partial_auc = float(roc_auc_score(y_val, scores, max_fpr=0.05))
        average_precision = float(average_precision_score(y_val, scores))
        candidate = f"{backbone}_{head_name}"
        for constraint, cap in CONSTRAINTS.items():
            negative_count = int((y_val == 0).sum())
            effective_cap = (
                None
                if cap is None
                else effective_empirical_cap(cap, negative_count, cap_confidence)
            )
            threshold, metrics = select_threshold(y_val, scores, effective_cap)
            metrics["type1_upper_confidence_bound"] = binomial_upper_bound(
                int(metrics["fp"]), negative_count, cap_confidence
            )
            row: dict[str, object] = {
                "candidate": candidate,
                "backbone": backbone,
                "head": head_name,
                "representation": representation,
                "constraint": constraint,
                "target_type1_cap": cap,
                "effective_validation_type1_cap": effective_cap,
                "threshold": threshold,
                "val_roc_auc": auc,
                "val_partial_auc_fpr_0_05": partial_auc,
                "val_average_precision": average_precision,
            }
            row.update({f"val_{name}": value for name, value in metrics.items()})
            rows.append(row)
            score_key = key_for(metrics, partial_auc, auc, average_precision)
            if constraint not in winners or score_key > winners[constraint].score_key:
                winners[constraint] = Winner(
                    candidate=candidate,
                    backbone=backbone,
                    representation=representation,
                    threshold=threshold,
                    metrics=metrics,
                    score_key=score_key,
                    scaler=scaler,
                    pca=fitted_pca,
                    head=head,
                )
        cap5 = rows[-2]
        cap25 = rows[-1]
        print(
            f"{candidate}: auc={auc:.4f} pauc05={partial_auc:.4f} "
            f"sens@5%={float(cap5['val_sensitivity']):.4f} "
            f"sens@2.5%={float(cap25['val_sensitivity']):.4f}",
            flush=True,
        )

    for head_name, head in build_direct_heads(seed):
        evaluate_head(head_name, "direct", head, train_direct, val_direct, None)

    for requested_components in pca_components:
        component_count = min(requested_components, train_direct.shape[0] - 1, train_direct.shape[1])
        pca = PCA(n_components=component_count, whiten=True, svd_solver="randomized", random_state=seed)
        train_pca = pca.fit_transform(train_direct).astype(np.float32)
        val_pca = pca.transform(val_direct).astype(np.float32)
        for head_name, head in build_pca_heads(seed, component_count):
            evaluate_head(
                f"pca_{component_count}_{head_name}",
                f"pca_{component_count}",
                head,
                train_pca,
                val_pca,
                pca,
            )
    return rows


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    set_seeds(args.seed)
    project_root = args.project_root.resolve()
    cache_dir = project_root / OUTPUT_DIR
    confidence_label = "empirical" if args.cap_confidence <= 0.0 else f"guarded_{args.cap_confidence:.2f}"
    output_dir = cache_dir / confidence_label
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_rows = {split: list_image_rows(project_root, split) for split in ("train", "val", "test")}
    for split, rows in split_rows.items():
        if len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"Unexpected {split} count: {len(rows)}")

    winners: dict[str, Winner] = {}
    validation_rows: list[dict[str, object]] = []
    for backbone in args.backbones:
        train_cache = cache_dir / f"features_{backbone}_train_{args.image_size}.npz"
        val_cache = cache_dir / f"features_{backbone}_val_{args.image_size}.npz"
        train_features, y_train = load_or_extract(
            train_cache, backbone, split_rows["train"], args.image_size, args.batch_size, device
        )
        val_features, y_val = load_or_extract(
            val_cache, backbone, split_rows["val"], args.image_size, args.batch_size, device
        )
        validation_rows.extend(
            search_backbone(
                backbone,
                train_features,
                y_train,
                val_features,
                y_val,
                args.pca_components,
                args.seed,
                args.cap_confidence,
                winners,
            )
        )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output_dir / "validation_candidate_comparison.csv", index=False)

    test_results: list[dict[str, object]] = []
    test_feature_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for constraint, winner in winners.items():
        if winner.backbone not in test_feature_cache:
            cache_path = cache_dir / f"features_{winner.backbone}_test_{args.image_size}.npz"
            test_feature_cache[winner.backbone] = load_or_extract(
                cache_path,
                winner.backbone,
                split_rows["test"],
                args.image_size,
                args.batch_size,
                device,
            )
        test_features, y_test = test_feature_cache[winner.backbone]
        test_matrix = winner.scaler.transform(test_features).astype(np.float32)
        if winner.pca is not None:
            test_matrix = winner.pca.transform(test_matrix).astype(np.float32)
        scores = model_scores(winner.head, test_matrix)
        predictions = (scores >= winner.threshold).astype(np.int64)
        metrics = binary_metrics(y_test, predictions)
        row: dict[str, object] = {
            "model": winner.candidate,
            "constraint": constraint,
            "validation_type1_cap": CONSTRAINTS[constraint],
            "threshold": winner.threshold,
        }
        row.update({f"val_{name}": value for name, value in winner.metrics.items()})
        row.update({f"test_{name}": value for name, value in metrics.items()})
        test_results.append(row)

        stem = f"{constraint}_{winner.candidate}"
        joblib.dump(
            {
                "backbone": winner.backbone,
                "image_size": args.image_size,
                "scaler": winner.scaler,
                "pca": winner.pca,
                "head": winner.head,
                "threshold": winner.threshold,
            },
            output_dir / f"selected_pipeline_{stem}.joblib",
        )
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        save_binary_confusion(
            matrix,
            output_dir / f"test_confusion_matrix_{stem}.csv",
            output_dir / f"test_confusion_matrix_{stem}.png",
            f"Frozen CNN Test: {constraint}",
        )
        prediction_rows = split_rows["test"][["relative_path", "filename", "actual", "label"]].copy()
        prediction_rows["actual_binary"] = y_test
        prediction_rows["dementia_score"] = scores
        prediction_rows["threshold"] = winner.threshold
        prediction_rows["predicted_binary"] = predictions
        prediction_rows.to_csv(output_dir / f"test_predictions_{stem}.csv", index=False)

    selected = pd.DataFrame(test_results).sort_values("constraint")
    selected.to_csv(output_dir / "selected_test_results.csv", index=False)

    previous_path = project_root / "solutions" / "type2_constraints" / "constraint_comparison.csv"
    current = selected[
        [
            "model",
            "constraint",
            "validation_type1_cap",
            "threshold",
            "test_type1_error",
            "test_type2_error",
            "test_sensitivity",
            "test_specificity",
            "test_balanced_accuracy",
        ]
    ].copy()
    current["source"] = "pretrained_frozen_cnn_search"
    comparison = current
    if previous_path.exists():
        previous = pd.read_csv(previous_path).rename(
            columns={
                "type1_cap": "validation_type1_cap",
                "test_dementia_sensitivity": "test_sensitivity",
            }
        )
        previous = previous[list(current.columns[:-1])]
        previous["source"] = "previous_experiment"
        comparison = pd.concat([previous, current], ignore_index=True)
    comparison.to_csv(output_dir / "comparison_with_previous.csv", index=False)

    report_columns = [
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
    report = [
        "# Frozen Pretrained CNN Constraint Search",
        "",
        "Backbones and heads were selected using train and validation only. Test was evaluated after each constraint winner was fixed.",
        "",
        dataframe_markdown(selected[report_columns]),
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf8")
    manifest = {
        "seed": args.seed,
        "device": str(device),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "backbones": args.backbones,
        "spatial_pooling": "2x2",
        "pca_components": args.pca_components,
        "cap_confidence": args.cap_confidence,
        "selection": "minimum validation Type II under each Type I cap; ties prefer lower Type I, pAUC@5%, ROC-AUC, AP",
        "elapsed_seconds": time.perf_counter() - start,
        "torch": torch.__version__,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(f"Output directory: {output_dir}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
