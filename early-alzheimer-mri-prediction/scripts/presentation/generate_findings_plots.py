from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[2]
PRESENTATION = ROOT / "presentation"
ASSETS = PRESENTATION / "assets"
SEARCH = ROOT / "results" / "experiments" / "frozen_cnn_constraint_search"
GUARDED = SEARCH / "guarded_0.99"

BG = "#081923"
AX_BG = "#0b2530"
TEXT = "#ecfbf7"
MUTED = "#a8c7c3"
GREEN = "#39d7b5"
BLUE = "#7ab8ff"
GOLD = "#f0d47a"


def scores_from_pipeline(pipeline_path: Path, features: np.ndarray) -> np.ndarray:
    pipeline = joblib.load(pipeline_path)
    matrix = pipeline["scaler"].transform(features).astype(np.float32)
    if pipeline["pca"] is not None:
        matrix = pipeline["pca"].transform(matrix).astype(np.float32)
    head = pipeline["head"]
    if hasattr(head, "predict_proba"):
        return head.predict_proba(matrix)[:, 1]
    return head.decision_function(matrix)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#31515e")
    ax.grid(color="#31515e", alpha=0.38, linewidth=0.7)


def make_threshold_frontier() -> None:
    cache = np.load(SEARCH / "features_resnet18_val_128.npz")
    features = cache["features"]
    labels = cache["labels"]
    results = pd.read_csv(GUARDED / "selected_test_results.csv").set_index("constraint")

    policies = [
        ("type1_le_0_05", "5% policy", GREEN),
        ("type1_le_0_025", "2.5% policy", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=BG)
    style_axis(ax)
    for constraint, label, color in policies:
        pipeline_path = next(GUARDED.glob(f"selected_pipeline_{constraint}_*.joblib"))
        scores = scores_from_pipeline(pipeline_path, features)
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1.0 - tpr
        view = fpr <= 0.11
        ax.plot(fpr[view] * 100, fnr[view] * 100, color=color, linewidth=2.5, label=label)

        row = results.loc[constraint]
        x = float(row["val_type1_error"]) * 100
        y = float(row["val_type2_error"]) * 100
        ax.scatter([x], [y], s=78, color=color, edgecolor=TEXT, linewidth=1.1, zorder=5)
        ax.annotate(
            f"{label}: {x:.2f}% / {y:.2f}%",
            (x, y),
            xytext=(7, 8 if constraint.endswith("05") else -18),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )

    ax.axvline(5.0, color=GREEN, linestyle="--", linewidth=1.1, alpha=0.7)
    ax.axvline(2.5, color=BLUE, linestyle="--", linewidth=1.1, alpha=0.7)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 32)
    ax.set_xlabel("Type I error: false-positive rate (%)")
    ax.set_ylabel("Type II error: missed-dementia rate (%)")
    ax.set_title("Validation threshold frontier (lower-left is better)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "validation_threshold_frontier.png", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def make_error_count_plot() -> None:
    results = pd.read_csv(GUARDED / "selected_test_results.csv").set_index("constraint")
    five = results.loc["type1_le_0_05"]
    two = results.loc["type1_le_0_025"]
    values_five = [int(five["test_fp"]), int(five["test_fn"])]
    values_two = [int(two["test_fp"]), int(two["test_fn"])]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=BG)
    style_axis(ax)
    positions = np.arange(2)
    width = 0.32
    bars_five = ax.bar(positions - width / 2, values_five, width, color=GREEN, label="5% policy")
    bars_two = ax.bar(positions + width / 2, values_two, width, color=BLUE, label="2.5% policy")
    ax.bar_label(bars_five, padding=4, color=TEXT, fontsize=10, fontweight="bold")
    ax.bar_label(bars_two, padding=4, color=TEXT, fontsize=10, fontweight="bold")
    ax.set_xticks(positions, ["False alarms\n(non-demented flagged)", "Missed dementia\n(false negatives)"])
    ax.set_ylabel("Test images")
    ax.set_ylim(0, 98)
    ax.set_title("Stricter cap removes 14 false alarms but adds 40 misses", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "test_policy_error_counts.png", dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_threshold_frontier()
    make_error_count_plot()


if __name__ == "__main__":
    main()
