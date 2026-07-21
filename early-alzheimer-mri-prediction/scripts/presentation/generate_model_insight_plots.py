from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "presentation" / "assets"
OUTPUT = ROOT / "results" / "experiments" / "presentation_analysis"
sys.path.insert(0, str(SCRIPT_ROOT))

from run_plan_d_solution import build_feature_matrix, list_image_rows, load_images

BG = "#081923"
AX_BG = "#0b2530"
TEXT = "#ecfbf7"
MUTED = "#a8c7c3"
GREEN = "#39d7b5"
BLUE = "#7ab8ff"
GOLD = "#f0d47a"
GRID = "#31515e"


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.7)


def load_two_stage_mlp() -> tuple[dict, object]:
    payloads = joblib.load(ROOT / "results" / "experiments" / "type2_constraints" / "models.joblib")
    payload = payloads["two_stage_mlp"]
    return payload, payload["feature_bundle"]


def make_lda_plot(bundle: object) -> dict[str, float]:
    train_rows = list_image_rows(ROOT, "train")
    val_rows = list_image_rows(ROOT, "val")
    y_train = train_rows["label"].to_numpy(dtype=int)
    y_val = val_rows["label"].to_numpy(dtype=int)

    lda = LinearDiscriminantAnalysis(n_components=2, solver="svd")
    lda.fit(bundle.z_train, y_train)
    projected = lda.transform(bundle.z_val)
    predicted = lda.predict(bundle.z_val)
    balanced_accuracy = balanced_accuracy_score(y_val, predicted)

    colors = [BLUE, GREEN, GOLD]
    display_names = ["Non-demented", "Very mild", "Mild"]
    fig, ax = plt.subplots(figsize=(7.7, 5.0), facecolor=BG)
    style_axis(ax)
    for label, (name, color) in enumerate(zip(display_names, colors, strict=True)):
        points = projected[y_val == label]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=19,
            alpha=0.52,
            color=color,
            edgecolors="none",
            label=f"{name} (n={len(points)})",
        )
        center = points.mean(axis=0)
        ax.scatter(center[0], center[1], s=105, marker="X", color=color, edgecolor=TEXT, linewidth=1.0)
        ax.annotate(
            name,
            center,
            xytext=(7, 7),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("LDA direction 1")
    ax.set_ylabel("LDA direction 2")
    ax.set_title("Validation MRI features in train-fitted LDA space", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=TEXT, loc="best", fontsize=9)
    ax.text(
        0.01,
        0.01,
        f"Visualization only | validation balanced accuracy: {balanced_accuracy:.1%}",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "lda_validation_projection.png", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    output = pd.DataFrame(projected, columns=["lda_1", "lda_2"])
    output.insert(0, "actual", val_rows["actual"].to_numpy())
    output.insert(0, "relative_path", val_rows["relative_path"].to_numpy())
    output.to_csv(OUTPUT / "lda_validation_projection.csv", index=False)
    return {"lda_validation_balanced_accuracy": float(balanced_accuracy)}


def transform_raw(bundle: object, features: np.ndarray) -> np.ndarray:
    scaled = bundle.scaler.transform(features).astype(np.float32)
    return bundle.pca.transform(scaled).astype(np.float32)


def dementia_scores(model: object, matrix: np.ndarray) -> np.ndarray:
    class_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(matrix)[:, class_index]


def make_roc_plot(payload: dict, bundle: object) -> dict[str, float]:
    model = payload["binary_model"]
    val_rows = list_image_rows(ROOT, "val")
    test_rows = list_image_rows(ROOT, "test")
    y_val = (val_rows["label"].to_numpy(dtype=int) > 0).astype(int)
    y_test = (test_rows["label"].to_numpy(dtype=int) > 0).astype(int)
    val_scores = dementia_scores(model, bundle.z_val)
    test_scores = dementia_scores(model, bundle.z_test)
    val_auc = roc_auc_score(y_val, val_scores)
    test_auc = roc_auc_score(y_test, test_scores)

    fig, ax = plt.subplots(figsize=(7.7, 5.0), facecolor=BG)
    style_axis(ax)
    for labels, scores, name, color, linestyle in (
        (y_val, val_scores, f"Validation AUC {val_auc:.3f}", GREEN, "-"),
        (y_test, test_scores, f"Test audit AUC {test_auc:.3f}", BLUE, "--"),
    ):
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr * 100, tpr * 100, color=color, linewidth=2.4, linestyle=linestyle, label=name)
        predicted = scores >= 0.50
        fp_rate = ((labels == 0) & predicted).sum() / (labels == 0).sum()
        sensitivity = ((labels == 1) & predicted).sum() / (labels == 1).sum()
        ax.scatter(fp_rate * 100, sensitivity * 100, s=72, color=color, edgecolor=TEXT, linewidth=1.0, zorder=5)

    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Alpha: false-positive rate (%)")
    ax.set_ylabel("Sensitivity: dementia detected (%)")
    ax.set_title("Two-stage MLP ROC: dementia vs non-demented", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    ax.text(0.02, 0.03, "Dots mark the recommended threshold = 0.50", transform=ax.transAxes, color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "mlp_roc_auc.png", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return {"mlp_test_roc_auc": float(test_auc)}


def make_operating_point_plots() -> None:
    policies = pd.DataFrame(
        {
            "policy": ["Maximum sensitivity", "Recommended", "Lower alpha"],
            "alpha": [68.125, 7.7083333333, 5.8333333333],
            "beta": [0.0, 3.1847133758, 4.2462845011],
            "false_alarms": [327, 37, 28],
            "misses": [0, 15, 20],
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=BG)
    style_axis(ax)
    colors = [GOLD, GREEN, BLUE]
    ax.plot(policies["alpha"], policies["beta"], color=MUTED, linewidth=1.2, alpha=0.65)
    for row, color in zip(policies.itertuples(index=False), colors, strict=True):
        ax.scatter(row.alpha, row.beta, s=105, color=color, edgecolor=TEXT, linewidth=1.0, zorder=4)
        ax.annotate(
            f"{row.policy}\nalpha {row.alpha:.2f}% | beta {row.beta:.2f}%",
            (row.alpha, row.beta),
            xytext=(8, 8 if row.policy == "Maximum sensitivity" else -31),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlim(0, 75)
    ax.set_ylim(-0.4, 5.4)
    ax.set_xlabel("Alpha: false-positive rate on test (%)")
    ax.set_ylabel("Beta: missed-dementia rate on test (%)")
    ax.set_title("MLP operating points: minimize beta without accepting extreme alpha", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ASSETS / "mlp_operating_points.png", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    comparison = policies.iloc[1:].reset_index(drop=True)
    positions = np.arange(2)
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=BG)
    style_axis(ax)
    recommended = ax.bar(
        positions - width / 2,
        comparison.loc[0, ["false_alarms", "misses"]].to_numpy(dtype=float),
        width,
        color=GREEN,
        label="Recommended (threshold 0.50)",
    )
    lower_alpha = ax.bar(
        positions + width / 2,
        comparison.loc[1, ["false_alarms", "misses"]].to_numpy(dtype=float),
        width,
        color=BLUE,
        label="Lower alpha (threshold 0.748)",
    )
    ax.bar_label(recommended, padding=4, color=TEXT, fontsize=10, fontweight="bold")
    ax.bar_label(lower_alpha, padding=4, color=TEXT, fontsize=10, fontweight="bold")
    ax.set_xticks(positions, ["False alarms\n(non-demented flagged)", "Missed dementia\n(false negatives)"])
    ax.set_ylabel("Test images")
    ax.set_ylim(0, 43)
    ax.set_title("Lower alpha removes 9 false alarms but adds 5 misses", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "mlp_test_error_counts.png", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def bootstrap_auc_drop(
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    occluded_scores: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 400,
) -> tuple[float, float]:
    sample_indices = rng.integers(0, len(y_true), size=(iterations, len(y_true)))
    drops = []
    for indices in sample_indices:
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        drops.append(
            roc_auc_score(sampled_y, baseline_scores[indices])
            - roc_auc_score(sampled_y, occluded_scores[indices])
        )
    return tuple(np.quantile(drops, [0.025, 0.975]))


def make_regional_occlusion_plot(payload: dict, bundle: object) -> dict[str, float]:
    train_rows = list_image_rows(ROOT, "train")
    val_rows = list_image_rows(ROOT, "val")
    train_images = load_images(train_rows)
    train_mean_image = train_images.mean(axis=0)
    del train_images
    val_images = load_images(val_rows)
    y_binary = (val_rows["label"].to_numpy(dtype=int) > 0).astype(int)

    raw_val, feature_names = build_feature_matrix(val_images)
    if raw_val.shape[1] != bundle.raw_feature_count:
        raise ValueError("The saved MLP and reconstructed deterministic features do not match.")
    z_val = transform_raw(bundle, raw_val)
    if not np.allclose(z_val, bundle.z_val, rtol=2e-4, atol=2e-4):
        raise ValueError("Reconstructed validation features differ from the saved MLP inputs.")

    binary_model = payload["binary_model"]
    baseline_scores = dementia_scores(binary_model, z_val)
    baseline_auc = roc_auc_score(y_binary, baseline_scores)
    rng = np.random.default_rng(42)
    regions = [
        ("Central / ventricles", (44, 84, 44, 84)),
        ("Upper cortical proxy", (8, 48, 24, 104)),
        ("Lower cortical proxy", (80, 120, 24, 104)),
        ("Left lower-lateral proxy", (72, 120, 4, 48)),
        ("Right lower-lateral proxy", (72, 120, 80, 124)),
    ]

    rows = []
    for region_name, (r0, r1, c0, c1) in regions:
        occluded = val_images.copy()
        occluded[:, r0:r1, c0:c1] = train_mean_image[r0:r1, c0:c1]
        raw_occluded, occluded_names = build_feature_matrix(occluded)
        if occluded_names != feature_names:
            raise ValueError("Feature ordering changed during regional occlusion.")
        scores = dementia_scores(binary_model, transform_raw(bundle, raw_occluded))
        occluded_auc = roc_auc_score(y_binary, scores)
        low, high = bootstrap_auc_drop(y_binary, baseline_scores, scores, rng)
        rows.append(
            {
                "region": region_name,
                "baseline_auc": baseline_auc,
                "occluded_auc": occluded_auc,
                "auc_drop": baseline_auc - occluded_auc,
                "auc_drop_ci_low": low,
                "auc_drop_ci_high": high,
            }
        )

    results = pd.DataFrame(rows).sort_values("auc_drop", ascending=True).reset_index(drop=True)
    results.to_csv(OUTPUT / "mlp_regional_occlusion.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.7), facecolor=BG)
    style_axis(ax)
    values = results["auc_drop"].to_numpy() * 100
    lower = (results["auc_drop"] - results["auc_drop_ci_low"]).clip(lower=0).to_numpy() * 100
    upper = (results["auc_drop_ci_high"] - results["auc_drop"]).clip(lower=0).to_numpy() * 100
    colors = np.where(values > 0, GREEN, BLUE)
    bars = ax.barh(
        results["region"],
        values,
        color=colors,
        xerr=np.vstack([lower, upper]),
        error_kw={"ecolor": TEXT, "elinewidth": 1.0, "capsize": 3},
    )
    ax.axvline(0, color=TEXT, linewidth=0.9, alpha=0.7)
    for bar, value in zip(bars, values, strict=True):
        offset = 0.08 if value >= 0 else -0.08
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f} pt",
            va="center",
            ha="left" if value >= 0 else "right",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel("Change in validation ROC-AUC after area is hidden (percentage points)")
    ax.set_title("Two-stage MLP regional occlusion test", fontsize=13, fontweight="bold")
    ax.text(
        0.01,
        0.01,
        f"Baseline validation ROC-AUC: {baseline_auc:.3f} | bars show paired bootstrap 95% intervals",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "mlp_regional_occlusion.png", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return {"mlp_validation_roc_auc": float(baseline_auc)}


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload, bundle = load_two_stage_mlp()
    summary = {}
    summary.update(make_lda_plot(bundle))
    summary.update(make_roc_plot(payload, bundle))
    make_operating_point_plots()
    summary.update(make_regional_occlusion_plot(payload, bundle))
    pd.Series(summary, name="value").to_csv(OUTPUT / "summary.csv")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
