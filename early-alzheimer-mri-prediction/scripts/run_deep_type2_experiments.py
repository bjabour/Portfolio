from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from run_plan_d_solution import (
    CLASS_NAMES,
    build_feature_matrix,
    dataframe_to_markdown,
    load_split,
    metric_row,
    package_versions,
    save_confusion_matrix,
)


OUTPUT_DIR = Path("results") / "experiments" / "deep_type2"
POSITIVE_LABELS = {1, 2}


@dataclass
class FeatureBundle:
    name: str
    z_train: np.ndarray
    z_val: np.ndarray
    z_test: np.ndarray
    scaler: StandardScaler
    pca: PCA
    raw_feature_count: int


@dataclass
class CandidateResult:
    name: str
    selected_threshold: float
    val_pred: np.ndarray
    test_pred: np.ndarray
    val_scores: np.ndarray
    test_scores: np.ndarray
    val_prob_frame: pd.DataFrame
    test_prob_frame: pd.DataFrame
    metrics: dict[str, float | int | str | bool]
    model_payload: dict[str, object]
    threshold_summary: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Type II focused neural experiments for ALZ BrainMRI."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the ALZ-BrainMRI project root.",
    )
    parser.add_argument(
        "--min-specificity",
        type=float,
        default=0.50,
        help="Validation specificity guardrail while minimizing Type II error.",
    )
    parser.add_argument("--max-iter", type=int, default=120)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def as_binary(y: np.ndarray) -> np.ndarray:
    return (y != 0).astype(int)


def dementia_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float | int]:
    actual_positive = y_true != 0
    predicted_positive = y_pred != 0
    tn = int((~actual_positive & ~predicted_positive).sum())
    fp = int((~actual_positive & predicted_positive).sum())
    fn = int((actual_positive & ~predicted_positive).sum())
    tp = int((actual_positive & predicted_positive).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    type2_error = fn / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    type1_error = fp / (tn + fp) if (tn + fp) else 0.0
    return {
        f"{prefix}_tn_non_demented": tn,
        f"{prefix}_fp_dementia": fp,
        f"{prefix}_fn_dementia": fn,
        f"{prefix}_tp_dementia": tp,
        f"{prefix}_specificity": specificity,
        f"{prefix}_type1_error": type1_error,
        f"{prefix}_dementia_sensitivity": sensitivity,
        f"{prefix}_type2_error": type2_error,
    }


def weighted_samples(y: np.ndarray, dementia_multiplier: float = 1.0) -> np.ndarray:
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
    weights[y != 0] *= dementia_multiplier
    return weights


def fit_transform(
    name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
    seed: int,
) -> FeatureBundle:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    components = min(n_components, x_train_scaled.shape[0] - 1, x_train_scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    z_train = pca.fit_transform(x_train_scaled).astype(np.float32)
    z_val = pca.transform(x_val_scaled).astype(np.float32)
    z_test = pca.transform(x_test_scaled).astype(np.float32)
    for split_name, values in (("train", z_train), ("val", z_val), ("test", z_test)):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} produced non-finite {split_name} features.")
    return FeatureBundle(name, z_train, z_val, z_test, scaler, pca, x_train.shape[1])


def pooled_pixels(images: np.ndarray, size: int = 64) -> np.ndarray:
    if size == 64:
        pooled = images.reshape(images.shape[0], 64, 2, 64, 2).mean(axis=(2, 4))
    elif size == 32:
        pooled = images.reshape(images.shape[0], 32, 4, 32, 4).mean(axis=(2, 4))
    else:
        raise ValueError("Only 32 or 64 pooled pixel size is supported.")
    return pooled.reshape(images.shape[0], -1).astype(np.float32)


def build_feature_matrix_batched(
    images: np.ndarray, batch_size: int = 64
) -> tuple[np.ndarray, list[str]]:
    parts = []
    feature_names: list[str] | None = None
    for start in range(0, images.shape[0], batch_size):
        batch_features, batch_names = build_feature_matrix(images[start : start + batch_size])
        if feature_names is None:
            feature_names = batch_names
        elif feature_names != batch_names:
            raise ValueError("Feature names changed between batches.")
        parts.append(batch_features)
    return np.vstack(parts).astype(np.float32), feature_names or []


def load_rows_and_features(project_root: Path, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows, images = load_split(project_root, split)
    domain_features, _ = build_feature_matrix_batched(images)
    pixel_features = pooled_pixels(images, size=64)
    del images
    return rows, domain_features, pixel_features


def make_mlp(
    seed: int,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=64,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=seed,
        tol=1e-4,
        n_iter_no_change=15,
        verbose=False,
    )


def probability_frame(prefix: str, classes: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(proba.shape[0]))
    for idx, label in enumerate(classes):
        if int(label) in range(len(CLASS_NAMES)):
            out[f"{prefix}_prob_{CLASS_NAMES[int(label)]}"] = proba[:, idx]
        else:
            out[f"{prefix}_prob_class_{int(label)}"] = proba[:, idx]
    return out


def positive_scores_from_multiclass(model: MLPClassifier, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    proba = model.predict_proba(z)
    if not np.isfinite(proba).all():
        raise ValueError("Non-finite MLP probabilities detected.")
    if ((proba < -1e-8) | (proba > 1 + 1e-8)).any():
        raise ValueError("MLP probabilities outside [0, 1].")
    class_to_col = {int(label): idx for idx, label in enumerate(model.classes_)}
    scores = proba[:, class_to_col[1]] + proba[:, class_to_col[2]]
    severity = np.where(proba[:, class_to_col[2]] >= proba[:, class_to_col[1]], 2, 1)
    return scores, severity.astype(int), probability_frame("mlp", model.classes_, proba)


def positive_scores_from_two_stage(
    binary_model: MLPClassifier,
    severity_model: MLPClassifier,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    binary_proba = binary_model.predict_proba(z)
    severity_proba = severity_model.predict_proba(z)
    for proba in (binary_proba, severity_proba):
        if not np.isfinite(proba).all():
            raise ValueError("Non-finite two-stage MLP probabilities detected.")
        if ((proba < -1e-8) | (proba > 1 + 1e-8)).any():
            raise ValueError("Two-stage probabilities outside [0, 1].")

    binary_cols = {int(label): idx for idx, label in enumerate(binary_model.classes_)}
    scores = binary_proba[:, binary_cols[1]]
    severity_classes = np.asarray(severity_model.classes_, dtype=int)
    severity = severity_classes[np.argmax(severity_proba, axis=1)]

    frame = pd.concat(
        [
            probability_frame("binary", binary_model.classes_, binary_proba),
            probability_frame("severity", severity_model.classes_, severity_proba),
        ],
        axis=1,
    )
    return scores, severity.astype(int), frame


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 0.0, 1.0)
    eps = np.finfo(np.float64).eps * 128
    candidates = np.concatenate([[0.0, 0.5, 1.0], clipped, clipped + eps, clipped - eps])
    return np.unique(np.clip(candidates, 0.0, 1.0))


def apply_threshold(scores: np.ndarray, severity: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, severity, 0).astype(int)


def select_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    severity_val: np.ndarray,
    min_specificity: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in threshold_grid(scores_val):
        pred = apply_threshold(scores_val, severity_val, float(threshold))
        row: dict[str, float | int] = {"threshold": float(threshold)}
        row.update(metric_row(y_val, pred, "val"))
        row.update(dementia_metrics(y_val, pred, "val"))
        rows.append(row)

    frame = pd.DataFrame(rows)
    eligible = frame[frame["val_specificity"] >= min_specificity].copy()
    if eligible.empty:
        eligible = frame.copy()
    selected = eligible.sort_values(
        ["val_type2_error", "val_macro_f1", "val_balanced_accuracy", "val_specificity"],
        ascending=[True, False, False, False],
    ).iloc[0]

    frame["selected"] = False
    frame.loc[selected.name, "selected"] = True
    ordered = ["threshold", "selected"] + [col for col in frame.columns if col not in {"threshold", "selected"}]
    return float(selected["threshold"]), frame[ordered].sort_values("threshold")


def prediction_frame(
    rows: pd.DataFrame,
    y_pred: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    prob_frame: pd.DataFrame,
) -> pd.DataFrame:
    out = rows[["split", "relative_path", "filename", "actual", "label"]].copy()
    out["predicted"] = [CLASS_NAMES[int(label)] for label in y_pred]
    out["predicted_label"] = y_pred
    out["dementia_score"] = scores
    out["threshold"] = threshold
    return pd.concat([out.reset_index(drop=True), prob_frame.reset_index(drop=True)], axis=1)


def evaluate_candidate(
    name: str,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scores_val: np.ndarray,
    severity_val: np.ndarray,
    scores_test: np.ndarray,
    severity_test: np.ndarray,
    val_prob_frame: pd.DataFrame,
    test_prob_frame: pd.DataFrame,
    min_specificity: float,
    model_payload: dict[str, object],
) -> CandidateResult:
    threshold, threshold_summary = select_threshold(y_val, scores_val, severity_val, min_specificity)
    val_pred = apply_threshold(scores_val, severity_val, threshold)
    test_pred = apply_threshold(scores_test, severity_test, threshold)
    metrics: dict[str, float | int | str | bool] = {
        "model": name,
        "selected": False,
        "threshold": threshold,
        "min_specificity_guardrail": min_specificity,
    }
    metrics.update(metric_row(y_val, val_pred, "val"))
    metrics.update(dementia_metrics(y_val, val_pred, "val"))
    metrics.update(metric_row(y_test, test_pred, "test"))
    metrics.update(dementia_metrics(y_test, test_pred, "test"))
    return CandidateResult(
        name,
        threshold,
        val_pred,
        test_pred,
        scores_val,
        scores_test,
        val_prob_frame,
        test_prob_frame,
        metrics,
        model_payload,
        threshold_summary,
    )


def fit_candidates(
    domain: FeatureBundle,
    pixels: FeatureBundle,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    max_iter: int,
    min_specificity: float,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []

    model = make_mlp(seed, (128, 64), alpha=1e-3, learning_rate_init=8e-4, max_iter=max_iter)
    model.fit(domain.z_train, y_train, sample_weight=weighted_samples(y_train, dementia_multiplier=1.5))
    val_scores, val_severity, val_probs = positive_scores_from_multiclass(model, domain.z_val)
    test_scores, test_severity, test_probs = positive_scores_from_multiclass(model, domain.z_test)
    candidates.append(
        evaluate_candidate(
            "mlp_domain_pca_multiclass",
            y_val,
            y_test,
            val_scores,
            val_severity,
            test_scores,
            test_severity,
            val_probs,
            test_probs,
            min_specificity,
            {"kind": "multiclass", "feature_bundle": domain, "model": model},
        )
    )

    model = make_mlp(seed + 1, (256, 128), alpha=1e-3, learning_rate_init=6e-4, max_iter=max_iter)
    model.fit(pixels.z_train, y_train, sample_weight=weighted_samples(y_train, dementia_multiplier=2.0))
    val_scores, val_severity, val_probs = positive_scores_from_multiclass(model, pixels.z_val)
    test_scores, test_severity, test_probs = positive_scores_from_multiclass(model, pixels.z_test)
    candidates.append(
        evaluate_candidate(
            "mlp_pixel_pca_multiclass",
            y_val,
            y_test,
            val_scores,
            val_severity,
            test_scores,
            test_severity,
            val_probs,
            test_probs,
            min_specificity,
            {"kind": "multiclass", "feature_bundle": pixels, "model": model},
        )
    )

    binary_model = make_mlp(seed + 2, (128, 64), alpha=7e-4, learning_rate_init=8e-4, max_iter=max_iter)
    y_train_binary = as_binary(y_train)
    binary_weights = compute_sample_weight(class_weight="balanced", y=y_train_binary).astype(np.float64)
    binary_weights[y_train_binary == 1] *= 3.0
    binary_model.fit(domain.z_train, y_train_binary, sample_weight=binary_weights)

    severity_model = make_mlp(seed + 3, (96, 48), alpha=1e-3, learning_rate_init=8e-4, max_iter=max_iter)
    dementia_train = np.isin(y_train, list(POSITIVE_LABELS))
    severity_model.fit(
        domain.z_train[dementia_train],
        y_train[dementia_train],
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_train[dementia_train]),
    )
    val_scores, val_severity, val_probs = positive_scores_from_two_stage(
        binary_model, severity_model, domain.z_val
    )
    test_scores, test_severity, test_probs = positive_scores_from_two_stage(
        binary_model, severity_model, domain.z_test
    )
    candidates.append(
        evaluate_candidate(
            "mlp_two_stage_dementia_first",
            y_val,
            y_test,
            val_scores,
            val_severity,
            test_scores,
            test_severity,
            val_probs,
            test_probs,
            min_specificity,
            {
                "kind": "two_stage",
                "feature_bundle": domain,
                "binary_model": binary_model,
                "severity_model": severity_model,
            },
        )
    )

    return candidates


def select_model(results: list[CandidateResult]) -> CandidateResult:
    selected = sorted(
        results,
        key=lambda result: (
            float(result.metrics["val_type2_error"]),
            -float(result.metrics["val_macro_f1"]),
            -float(result.metrics["val_balanced_accuracy"]),
            -float(result.metrics["val_specificity"]),
        ),
    )[0]
    selected.metrics["selected"] = True
    return selected


def write_outputs(
    output_dir: Path,
    results: list[CandidateResult],
    selected: CandidateResult,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([result.metrics for result in results])
    summary.to_csv(output_dir / "model_comparison.csv", index=False)

    for result in results:
        result.threshold_summary.to_csv(output_dir / f"threshold_sweep_{result.name}.csv", index=False)
        save_confusion_matrix(
            y_val,
            result.val_pred,
            output_dir / f"val_confusion_matrix_{result.name}.csv",
            output_dir / f"val_confusion_matrix_{result.name}.png",
            f"Validation Confusion Matrix: {result.name}",
        )
        save_confusion_matrix(
            y_test,
            result.test_pred,
            output_dir / f"test_confusion_matrix_{result.name}.csv",
            output_dir / f"test_confusion_matrix_{result.name}.png",
            f"Test Confusion Matrix: {result.name}",
        )
        prediction_frame(
            val_rows,
            result.val_pred,
            result.val_scores,
            result.selected_threshold,
            result.val_prob_frame,
        ).to_csv(output_dir / f"predictions_val_{result.name}.csv", index=False)
        prediction_frame(
            test_rows,
            result.test_pred,
            result.test_scores,
            result.selected_threshold,
            result.test_prob_frame,
        ).to_csv(output_dir / f"predictions_test_{result.name}.csv", index=False)

    joblib.dump(selected.model_payload, output_dir / "selected_model.joblib")

    manifest = {
        "seed": seed,
        "elapsed_seconds": elapsed_seconds,
        "selection_metric": "lowest validation Type II error, then validation macro-F1/balanced accuracy/specificity",
        "positive_definition": "dementia = Very_Mild_Demented or Mild_Demented",
        "type2_definition": "actual dementia predicted as Non_Demented",
        "split_counts": {
            "train": int(len(train_rows)),
            "val": int(len(val_rows)),
            "test": int(len(test_rows)),
        },
        "packages": package_versions(),
        "note": "PyTorch/TensorFlow were not installed locally; these experiments use scikit-learn MLP neural networks.",
    }
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

    report_lines = [
        "# Deep Type II MRI Experiment Report",
        "",
        "Goal: minimize Type II error, where a Type II error means an actual dementia image is predicted as `Non_Demented`.",
        "",
        f"Selected model: `{selected.name}`",
        f"Validation Type II error: `{float(selected.metrics['val_type2_error']):.6f}`",
        f"Test Type II error: `{float(selected.metrics['test_type2_error']):.6f}`",
        f"Test dementia sensitivity: `{float(selected.metrics['test_dementia_sensitivity']):.6f}`",
        f"Test specificity tradeoff: `{float(selected.metrics['test_specificity']):.6f}`",
        "",
        "## Model Comparison",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Selected Validation Report",
        "",
        "```text",
        classification_report(y_val, selected.val_pred, target_names=CLASS_NAMES, zero_division=0),
        "```",
        "",
        "## Selected Test Report",
        "",
        "```text",
        classification_report(y_test, selected.test_pred, target_names=CLASS_NAMES, zero_division=0),
        "```",
        "",
        "## Validation Rules",
        "",
        "- Preprocessing was fit on train only.",
        "- Thresholds were selected on validation only.",
        "- Test was used once after each model's validation threshold was fixed.",
        "- The specificity guardrail prevents the trivial all-dementia classifier.",
        "",
        "## Caveat",
        "",
        "This is prediction research, not medical diagnosis. Patient-level leakage cannot be checked because patient IDs are unavailable.",
        "",
    ]
    (output_dir / "classification_report.md").write_text("\n".join(report_lines), encoding="utf8")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    set_seeds(args.seed)
    project_root = args.project_root.resolve()
    output_dir = project_root / OUTPUT_DIR

    train_rows, x_train_domain, x_train_pixels = load_rows_and_features(project_root, "train")
    val_rows, x_val_domain, x_val_pixels = load_rows_and_features(project_root, "val")
    test_rows, x_test_domain, x_test_pixels = load_rows_and_features(project_root, "test")
    y_train = train_rows["label"].to_numpy(dtype=int)
    y_val = val_rows["label"].to_numpy(dtype=int)
    y_test = test_rows["label"].to_numpy(dtype=int)

    domain = fit_transform("domain_pca", x_train_domain, x_val_domain, x_test_domain, 120, args.seed)
    pixels = fit_transform("pixel_pca", x_train_pixels, x_val_pixels, x_test_pixels, 160, args.seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        results = fit_candidates(
            domain,
            pixels,
            y_train,
            y_val,
            y_test,
            args.seed,
            args.max_iter,
            args.min_specificity,
        )

    selected = select_model(results)
    elapsed = time.perf_counter() - start
    write_outputs(
        output_dir,
        results,
        selected,
        train_rows,
        val_rows,
        test_rows,
        y_val,
        y_test,
        args.seed,
        elapsed,
    )

    print(f"Output directory: {output_dir}")
    print(pd.DataFrame([result.metrics for result in results]).to_string(index=False))


if __name__ == "__main__":
    main()
