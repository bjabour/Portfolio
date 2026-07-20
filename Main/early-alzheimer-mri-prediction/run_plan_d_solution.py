from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import random
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


CLASS_NAMES = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented"]
SPLIT_DIRS = {
    "train": Path("train") / "train",
    "val": Path("val") / "val",
    "test": Path("test") / "test",
}
EXPECTED_SPLIT_COUNTS = {"train": 4435, "val": 950, "test": 951}
PCA_COMPONENTS = 80
SELECTION_TOLERANCE = 0.02
TYPE1_TARGET_SPECIFICITY = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan D: deterministic MRI features plus scikit-learn classifiers."
    )
    parser.add_argument(
        "--stage",
        choices=["deterministic", "cnn", "both"],
        default="deterministic",
        help="Run deterministic features, dependency-gated CNN status, or both.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to Projects/ALZ-BrainMRI.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def list_image_rows(project_root: Path, split: str) -> pd.DataFrame:
    rows = []
    split_root = project_root / SPLIT_DIRS[split]
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_root / class_name
        image_paths = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(class_dir.glob(pattern))
        for path in sorted(image_paths):
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "actual": class_name,
                    "filename": path.name,
                    "path": str(path),
                    "relative_path": str(path.relative_to(project_root)),
                }
            )
    return pd.DataFrame(rows)


def load_images(rows: pd.DataFrame) -> np.ndarray:
    images = np.empty((len(rows), 128, 128), dtype=np.float32)
    resample = getattr(Image, "Resampling", Image).BILINEAR
    for i, path in enumerate(rows["path"]):
        with Image.open(path) as img:
            img = img.convert("L").resize((128, 128), resample)
            images[i] = np.asarray(img, dtype=np.float32) / 255.0
    return images


def grid_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = images.shape[0]
    grid = images.reshape(n, 4, 32, 4, 32).transpose(0, 1, 3, 2, 4)
    means = grid.mean(axis=(3, 4))
    mean_squares = np.einsum("nrcxy,nrcxy->nrc", grid, grid) / (32 * 32)
    stds = np.sqrt(np.maximum(mean_squares - means**2, 0.0))
    dark_fracs = (grid < 0.25).mean(axis=(3, 4))
    values = np.concatenate(
        [
            means.reshape(n, 16),
            stds.reshape(n, 16),
            dark_fracs.reshape(n, 16),
        ],
        axis=1,
    )
    names = []
    for stat in ("mean", "std", "dark_frac"):
        names.extend([f"grid_{r}_{c}_{stat}" for r in range(4) for c in range(4)])
    return values, names


def region_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    regions = {
        "central_ventricle_proxy": (44, 84, 44, 84),
        "left_temporal_proxy": (72, 120, 4, 48),
        "right_temporal_proxy": (72, 120, 80, 124),
        "upper_cortex_proxy": (8, 48, 24, 104),
        "lower_cortex_proxy": (80, 120, 24, 104),
    }
    parts = []
    names = []
    for region_name, (r0, r1, c0, c1) in regions.items():
        region = images[:, r0:r1, c0:c1].reshape(images.shape[0], -1)
        parts.append(
            np.column_stack(
                [
                    region.mean(axis=1),
                    region.std(axis=1),
                    (region < 0.25).mean(axis=1),
                    (region > 0.65).mean(axis=1),
                ]
            )
        )
        names.extend(
            [
                f"{region_name}_mean",
                f"{region_name}_std",
                f"{region_name}_dark_frac",
                f"{region_name}_bright_frac",
            ]
        )
    return np.concatenate(parts, axis=1), names


def handcrafted_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = images.shape[0]
    flat = images.reshape(n, -1)
    mask = images > 0.08
    mask_count = mask.sum(axis=(1, 2)).clip(min=1)
    foreground_sum = (images * mask).sum(axis=(1, 2))
    foreground_mean = foreground_sum / mask_count
    foreground_second = ((images**2) * mask).sum(axis=(1, 2)) / mask_count
    foreground_std = np.sqrt(np.maximum(foreground_second - foreground_mean**2, 0.0))

    gx = np.abs(np.diff(images, axis=2))
    gy = np.abs(np.diff(images, axis=1))
    left = images[:, :, :64]
    right = images[:, :, 64:][:, :, ::-1]
    asymmetry = np.abs(left - right)
    quantile_source = images[:, ::4, ::4].reshape(n, -1)
    percentiles = np.percentile(
        quantile_source, [5, 10, 25, 50, 75, 90, 95], axis=1
    ).T

    scalar_values = np.column_stack(
        [
            images.mean(axis=(1, 2)),
            images.std(axis=(1, 2)),
            mask.mean(axis=(1, 2)),
            foreground_mean,
            foreground_std,
            gx.mean(axis=(1, 2)),
            gy.mean(axis=(1, 2)),
            (gx > 0.12).mean(axis=(1, 2)),
            (gy > 0.12).mean(axis=(1, 2)),
            asymmetry.mean(axis=(1, 2)),
            asymmetry.std(axis=(1, 2)),
        ]
    )
    scalar_names = [
        "global_mean",
        "global_std",
        "brain_area_proxy",
        "foreground_mean",
        "foreground_std",
        "x_edge_mean",
        "y_edge_mean",
        "x_edge_density",
        "y_edge_density",
        "left_right_asymmetry_mean",
        "left_right_asymmetry_std",
    ]
    percentile_names = [f"intensity_p{p}" for p in (5, 10, 25, 50, 75, 90, 95)]

    regions, region_names = region_features(images)
    grids, grid_names = grid_features(images)
    values = np.concatenate([scalar_values, percentiles, regions, grids], axis=1)
    names = scalar_names + percentile_names + region_names + grid_names
    return values, names


def build_feature_matrix(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = images.shape[0]
    pooled = images.reshape(n, 32, 4, 32, 4).mean(axis=(2, 4))
    pooled_flat = pooled.reshape(n, -1)
    pooled_names = [f"pooled_pixel_{i}" for i in range(pooled_flat.shape[1])]
    handcrafted, handcrafted_names = handcrafted_features(images)
    features = np.concatenate([pooled_flat, handcrafted], axis=1).astype(np.float32)
    feature_names = pooled_names + handcrafted_names
    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature values detected.")
    return features, feature_names


def load_split(project_root: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows = list_image_rows(project_root, split)
    expected = EXPECTED_SPLIT_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(f"{split} expected {expected} images, found {len(rows)}.")
    if rows["actual"].nunique() != len(CLASS_NAMES):
        raise ValueError(f"{split} does not contain all expected classes.")
    images = load_images(rows)
    return rows, images


def fit_feature_transform(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    n_components = min(PCA_COMPONENTS, x_train_scaled.shape[0] - 1, x_train_scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    z_train = pca.fit_transform(x_train_scaled)
    z_val = pca.transform(x_val_scaled)
    z_test = pca.transform(x_test_scaled)
    for name, matrix in (("train", z_train), ("val", z_val), ("test", z_test)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite PCA features detected for {name}.")
    return z_train, z_val, z_test, scaler, pca


def candidate_models(seed: int) -> list[tuple[str, object]]:
    return [
        ("null_most_frequent", DummyClassifier(strategy="most_frequent")),
        (
            "logistic_balanced",
            LogisticRegression(class_weight="balanced", C=1.0, max_iter=1500),
        ),
        (
            "linear_svm_balanced",
            LinearSVC(class_weight="balanced", C=1.0, dual=False, max_iter=10000, random_state=seed),
        ),
        (
            "rbf_svm_balanced",
            SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", cache_size=512),
        ),
    ]


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{prefix}_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def select_model(rows: list[dict[str, object]]) -> str:
    best = max(float(row["val_macro_f1"]) for row in rows)
    threshold = best - SELECTION_TOLERANCE
    for row in rows:
        if float(row["val_macro_f1"]) >= threshold:
            return str(row["model"])
    return str(rows[0]["model"])


def score_columns(model: object, z: np.ndarray) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(z)
        if not np.isfinite(proba).all():
            raise ValueError("Non-finite probabilities detected.")
        if ((proba < -1e-8) | (proba > 1 + 1e-8)).any():
            raise ValueError("Probability values outside [0, 1].")
        return pd.DataFrame(proba, columns=[f"prob_{name}" for name in CLASS_NAMES])
    if hasattr(model, "decision_function"):
        scores = model.decision_function(z)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite decision scores detected.")
        return pd.DataFrame(scores, columns=[f"score_class_{i}" for i in range(scores.shape[1])])
    return pd.DataFrame(index=np.arange(z.shape[0]))


def decision_scores(model: object, z: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "decision_function"):
        return None
    scores = model.decision_function(z)
    if scores.ndim != 2 or scores.shape[1] != len(CLASS_NAMES):
        return None
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite decision scores detected.")
    return scores


def prediction_frame(model: object, rows: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    pred = model.predict(z)
    pred_names = [CLASS_NAMES[int(i)] for i in pred]
    out = rows[["split", "relative_path", "filename", "actual", "label"]].copy()
    out["predicted"] = pred_names
    out["predicted_label"] = pred
    return pd.concat([out.reset_index(drop=True), score_columns(model, z).reset_index(drop=True)], axis=1)


def dementia_threshold_predictions(
    scores: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    dementia_scores = scores[:, 1:]
    best_dementia_class = np.argmax(dementia_scores, axis=1) + 1
    dementia_margin = dementia_scores.max(axis=1) - scores[:, 0]
    pred = np.where(dementia_margin >= threshold, best_dementia_class, 0)
    return pred.astype(int), dementia_margin


def binary_type1_row(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str
) -> dict[str, float | int]:
    actual_positive = y_true != 0
    predicted_positive = y_pred != 0
    fp = int((~actual_positive & predicted_positive).sum())
    tn = int((~actual_positive & ~predicted_positive).sum())
    fn = int((actual_positive & ~predicted_positive).sum())
    tp = int((actual_positive & predicted_positive).sum())
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (fp + tn) if (fp + tn) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        f"{prefix}_tn_non_demented": tn,
        f"{prefix}_fp_dementia": fp,
        f"{prefix}_fn_dementia": fn,
        f"{prefix}_tp_dementia": tp,
        f"{prefix}_type1_fpr": false_positive_rate,
        f"{prefix}_specificity": specificity,
        f"{prefix}_dementia_sensitivity": sensitivity,
    }


def threshold_candidates(margins: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(margins, np.linspace(0.0, 1.0, 401))
    eps = np.finfo(np.float64).eps * 16
    candidates = np.unique(np.concatenate([[0.0], quantiles, quantiles + eps]))
    return candidates


def threshold_summary_rows(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    _, val_margins = dementia_threshold_predictions(val_scores, 0.0)
    rows = []
    for threshold in threshold_candidates(val_margins):
        val_pred, _ = dementia_threshold_predictions(val_scores, float(threshold))
        test_pred, _ = dementia_threshold_predictions(test_scores, float(threshold))
        row: dict[str, float | int | bool] = {"threshold": float(threshold)}
        row.update(metric_row(y_val, val_pred, "val"))
        row.update(binary_type1_row(y_val, val_pred, "val"))
        row.update(metric_row(y_test, test_pred, "test"))
        row.update(binary_type1_row(y_test, test_pred, "test"))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["val_specificity", "val_macro_f1", "threshold"],
        ascending=[False, False, True],
    )
    eligible = summary[summary["val_specificity"] >= TYPE1_TARGET_SPECIFICITY].copy()
    if eligible.empty:
        selected = summary.iloc[0]
    else:
        selected = eligible.sort_values(
            ["val_macro_f1", "val_specificity", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    selected_index = selected.name
    selected_threshold = float(selected["threshold"])
    summary["type1_control_selected"] = False
    summary.loc[selected_index, "type1_control_selected"] = True
    ordered_cols = ["threshold", "type1_control_selected"] + [
        col for col in summary.columns if col not in {"threshold", "type1_control_selected"}
    ]
    return summary[ordered_cols], selected_threshold


def thresholded_prediction_frame(
    rows: pd.DataFrame, scores: np.ndarray, threshold: float
) -> pd.DataFrame:
    pred, margins = dementia_threshold_predictions(scores, threshold)
    out = rows[["split", "relative_path", "filename", "actual", "label"]].copy()
    out["predicted"] = [CLASS_NAMES[int(i)] for i in pred]
    out["predicted_label"] = pred
    out["dementia_margin"] = margins
    out["threshold"] = threshold
    score_frame = pd.DataFrame(scores, columns=[f"score_{name}" for name in CLASS_NAMES])
    return pd.concat([out.reset_index(drop=True), score_frame.reset_index(drop=True)], axis=1)


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, csv_path: Path, png_path: Path, title: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    if cm.shape != (3, 3):
        raise ValueError(f"Expected 3x3 confusion matrix, got {cm.shape}.")
    frame = pd.DataFrame(
        cm,
        index=[f"actual_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    )
    frame.to_csv(csv_path)

    plt.figure(figsize=(7, 5))
    sns.heatmap(frame, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def package_versions() -> dict[str, str]:
    package_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pillow": "Pillow",
        "scikit-learn": "scikit-learn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "joblib": "joblib",
    }
    versions = {}
    for key, package_name in package_map.items():
        try:
            versions[key] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    display = display.astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator] + rows)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    selected_name: str,
    val_report: str,
    test_report: str,
    type1_summary: pd.DataFrame | None,
    type1_val_report: str | None,
    type1_test_report: str | None,
    split_counts: dict[str, int],
    feature_count: int,
    pca_components: int,
    seed: int,
) -> None:
    selected = summary.loc[summary["model"] == selected_name].iloc[0]
    if type1_summary is not None:
        type1_selected = type1_summary.loc[type1_summary["type1_control_selected"]].iloc[0]
        type1_section = f"""
## Type I Error Control Experiment

Positive class for this section: dementia = `{CLASS_NAMES[1]}` or `{CLASS_NAMES[2]}`.

- Type I error: actual `{CLASS_NAMES[0]}` predicted as dementia.
- Threshold rule: predict dementia only if the best dementia SVM score exceeds the non-dementia score by at least $\\tau$.
- Threshold selected on validation only: $\\tau = {type1_selected['threshold']:.6f}$.
- Validation specificity target: `{TYPE1_TARGET_SPECIFICITY:.2f}`.
- Validation Type I FPR after thresholding: `{type1_selected['val_type1_fpr']:.6f}`.
- Test Type I FPR after thresholding: `{type1_selected['test_type1_fpr']:.6f}`.
- Test macro-F1 after thresholding: `{type1_selected['test_macro_f1']:.6f}`.

Selected threshold row:

{dataframe_to_markdown(type1_summary.loc[type1_summary["type1_control_selected"]])}

### Thresholded Validation Classification Report

```text
{type1_val_report}
```

### Thresholded Final Test Classification Report

```text
{type1_test_report}
```
"""
    else:
        type1_section = "\n## Type I Error Control Experiment\n\nNot available for this model because usable multiclass decision scores were not produced.\n"
    markdown = f"""# Plan D MRI Classification Report

## Recommendation

Selected `{selected_name}` using validation macro-F1 with the simpler-within-0.02 rule.

## Problem And Data

- Task: 3-class Alzheimer MRI severity classification.
- Classes: `{CLASS_NAMES[0]}`, `{CLASS_NAMES[1]}`, `{CLASS_NAMES[2]}`.
- Split counts: train `{split_counts['train']}`, validation `{split_counts['val']}`, test `{split_counts['test']}`.
- Raw feature count before scaling/PCA: `{feature_count}`.
- PCA components fitted on train only: `{pca_components}`.

## Validation Design

- Primary metric: validation `macro_f1`, to avoid hiding minority-class performance.
- Preprocessing boundary: scaler and PCA fit on train only, then applied to validation/test.
- Model choice: validation only; test evaluated once for the selected model.
- Class imbalance: handled with `class_weight='balanced'` for logistic/SVM candidates.

## Results

{dataframe_to_markdown(summary)}

## Validation Classification Report

```text
{val_report}
```

## Final Test Classification Report

```text
{test_report}
```

{type1_section}

## Limitations

- Image-level split only; patient IDs are not available, so patient-level generalization is unverified.
- Features are structural/texture proxies, not clinical segmentations.
- This is prediction research, not medical diagnosis.

## Reproducibility

- Script: `run_plan_d_solution.py`
- Seed: `{seed}`
- Outputs: `model_summary.csv`, confusion matrices, prediction CSVs, `feature_manifest.json`, `selected_model.joblib`
"""
    (output_dir / "classification_report.md").write_text(markdown, encoding="utf-8")


def write_manifest(
    output_dir: Path,
    feature_names: list[str],
    scaler: StandardScaler,
    pca: PCA,
    split_counts: dict[str, int],
    seed: int,
    selected_name: str,
    type1_threshold: float | None,
) -> None:
    manifest = {
        "stage": "deterministic",
        "class_order": CLASS_NAMES,
        "split_counts": split_counts,
        "seed": seed,
        "selection_metric": "val_macro_f1",
        "selection_rule": f"simplest model within {SELECTION_TOLERANCE} of best validation macro_f1",
        "raw_feature_count": len(feature_names),
        "raw_feature_names": feature_names,
        "scaler": scaler.__class__.__name__,
        "pca": {
            "class": pca.__class__.__name__,
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        },
        "selected_model": selected_name,
        "type1_control": {
            "positive_definition": f"{CLASS_NAMES[1]} or {CLASS_NAMES[2]}",
            "type1_definition": f"actual {CLASS_NAMES[0]} predicted as dementia",
            "validation_specificity_target": TYPE1_TARGET_SPECIFICITY,
            "selected_threshold": type1_threshold,
            "threshold_selection_split": "validation",
        },
        "package_versions": package_versions(),
        "domain_knowledge_file": "domain knowledge/mri_indicators.md",
        "validation_boundary": "test set was not used for model selection",
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run_model_screen(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    summaries = []
    fitted = {}
    for model_name, model in candidate_models(seed):
        model.fit(z_train, y_train)
        fitted[model_name] = model
        val_pred = model.predict(z_val)
        row = {"model": model_name, "selected": False, "test_evaluated": False}
        row.update(metric_row(y_val, val_pred, "val"))
        summaries.append(row)

    selected_name = select_model(summaries)
    selected_model = fitted[selected_name]
    test_pred = selected_model.predict(z_test)
    test_metrics = metric_row(y_test, test_pred, "test")
    for row in summaries:
        if row["model"] == selected_name:
            row["selected"] = True
            row["test_evaluated"] = True
            row.update(test_metrics)
        else:
            for key in test_metrics:
                row[key] = np.nan
    return summaries, fitted, selected_name


def run_deterministic_stage(project_root: Path, seed: int) -> Path:
    output_dir = project_root / "solutions" / "plan_d"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_images = load_split(project_root, "train")
    val_rows, val_images = load_split(project_root, "val")
    test_rows, test_images = load_split(project_root, "test")
    split_counts = {
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
    }

    x_train, feature_names = build_feature_matrix(train_images)
    x_val, val_feature_names = build_feature_matrix(val_images)
    x_test, test_feature_names = build_feature_matrix(test_images)
    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise ValueError("Feature names differ across splits.")

    z_train, z_val, z_test, scaler, pca = fit_feature_transform(
        x_train, x_val, x_test, seed
    )
    y_train = train_rows["label"].to_numpy()
    y_val = val_rows["label"].to_numpy()
    y_test = test_rows["label"].to_numpy()

    summaries, fitted, selected_name = run_model_screen(
        z_train, y_train, z_val, y_val, z_test, y_test, seed
    )
    summary = pd.DataFrame(summaries)
    metric_cols = [
        col
        for col in summary.columns
        if col.startswith(("val_", "test_")) and col != "test_evaluated"
    ]
    summary[metric_cols] = summary[metric_cols].astype(float).round(6)
    summary.to_csv(output_dir / "model_summary.csv", index=False)

    selected_model = fitted[selected_name]
    val_predictions = prediction_frame(selected_model, val_rows, z_val)
    test_predictions = prediction_frame(selected_model, test_rows, z_test)
    val_predictions.to_csv(output_dir / "predictions_val.csv", index=False)
    test_predictions.to_csv(output_dir / "predictions_test.csv", index=False)

    save_confusion_matrix(
        y_val,
        val_predictions["predicted_label"].to_numpy(),
        output_dir / "val_confusion_matrix.csv",
        output_dir / "val_confusion_matrix.png",
        f"Validation Confusion Matrix: {selected_name}",
    )
    save_confusion_matrix(
        y_test,
        test_predictions["predicted_label"].to_numpy(),
        output_dir / "test_confusion_matrix.csv",
        output_dir / "test_confusion_matrix.png",
        f"Test Confusion Matrix: {selected_name}",
    )

    val_scores = decision_scores(selected_model, z_val)
    test_scores = decision_scores(selected_model, z_test)
    type1_summary = None
    type1_val_report = None
    type1_test_report = None
    type1_threshold = None
    if val_scores is not None and test_scores is not None:
        type1_summary, type1_threshold = threshold_summary_rows(
            y_val, val_scores, y_test, test_scores
        )
        metric_cols = [
            col
            for col in type1_summary.columns
            if col.startswith(("val_", "test_")) and type1_summary[col].dtype != bool
        ]
        type1_summary[metric_cols] = type1_summary[metric_cols].astype(float).round(6)
        type1_summary.to_csv(output_dir / "type1_threshold_summary.csv", index=False)

        thresholded_val_predictions = thresholded_prediction_frame(
            val_rows, val_scores, type1_threshold
        )
        thresholded_test_predictions = thresholded_prediction_frame(
            test_rows, test_scores, type1_threshold
        )
        thresholded_val_predictions.to_csv(
            output_dir / "predictions_val_type1_thresholded.csv", index=False
        )
        thresholded_test_predictions.to_csv(
            output_dir / "predictions_test_type1_thresholded.csv", index=False
        )
        save_confusion_matrix(
            y_val,
            thresholded_val_predictions["predicted_label"].to_numpy(),
            output_dir / "val_confusion_matrix_type1_thresholded.csv",
            output_dir / "val_confusion_matrix_type1_thresholded.png",
            f"Validation Confusion Matrix: {selected_name} type-I threshold",
        )
        save_confusion_matrix(
            y_test,
            thresholded_test_predictions["predicted_label"].to_numpy(),
            output_dir / "test_confusion_matrix_type1_thresholded.csv",
            output_dir / "test_confusion_matrix_type1_thresholded.png",
            f"Test Confusion Matrix: {selected_name} type-I threshold",
        )
        type1_val_report = classification_report(
            y_val,
            thresholded_val_predictions["predicted_label"].to_numpy(),
            target_names=CLASS_NAMES,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )
        type1_test_report = classification_report(
            y_test,
            thresholded_test_predictions["predicted_label"].to_numpy(),
            target_names=CLASS_NAMES,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )

    val_report = classification_report(
        y_val,
        val_predictions["predicted_label"].to_numpy(),
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    test_report = classification_report(
        y_test,
        test_predictions["predicted_label"].to_numpy(),
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    write_report(
        output_dir,
        summary,
        selected_name,
        val_report,
        test_report,
        type1_summary,
        type1_val_report,
        type1_test_report,
        split_counts,
        len(feature_names),
        int(pca.n_components_),
        seed,
    )
    write_manifest(
        output_dir,
        feature_names,
        scaler,
        pca,
        split_counts,
        seed,
        selected_name,
        type1_threshold,
    )
    joblib.dump(
        {
            "stage": "deterministic",
            "class_order": CLASS_NAMES,
            "feature_names": feature_names,
            "scaler": scaler,
            "pca": pca,
            "classifier": selected_model,
            "selected_model": selected_name,
            "seed": seed,
        },
        output_dir / "selected_model.joblib",
    )
    return output_dir


def write_cnn_status(project_root: Path) -> None:
    output_dir = project_root / "solutions" / "plan_d"
    output_dir.mkdir(parents=True, exist_ok=True)
    available = {
        "torch": importlib.util.find_spec("torch") is not None,
        "torchvision": importlib.util.find_spec("torchvision") is not None,
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
    }
    if available["torch"] and available["torchvision"]:
        status = (
            "CNN stage dependencies are available, but this script keeps v1 deterministic. "
            "Use the deterministic artifacts as the validated baseline before adding frozen "
            "pretrained embeddings."
        )
    elif available["tensorflow"]:
        status = (
            "TensorFlow is available, but this script keeps v1 deterministic. "
            "Use the deterministic artifacts as the validated baseline before adding frozen "
            "pretrained embeddings."
        )
    else:
        status = (
            "CNN stage skipped: PyTorch/torchvision or TensorFlow is not installed. "
            "Deterministic Plan D outputs are still produced."
        )
    (output_dir / "cnn_stage_status.txt").write_text(
        json.dumps({"available_dependencies": available, "status": status}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    set_seeds(args.seed)

    run_deterministic = args.stage in {"deterministic", "both", "cnn"}
    if run_deterministic:
        output_dir = run_deterministic_stage(project_root, args.seed)
    else:
        output_dir = project_root / "solutions" / "plan_d"
    if args.stage in {"cnn", "both"}:
        write_cnn_status(project_root)

    summary = pd.read_csv(output_dir / "model_summary.csv")
    selected = summary.loc[summary["selected"] == True].iloc[0]
    print(f"Selected model: {selected['model']}")
    print(f"Validation macro-F1: {selected['val_macro_f1']:.6f}")
    print(f"Test macro-F1: {selected['test_macro_f1']:.6f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
