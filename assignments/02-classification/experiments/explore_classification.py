import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TRAIN_PATH = os.path.join("..", "classification_train.csv")
OUT_DIR = "."


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


train = pd.read_csv(TRAIN_PATH)
features = ["x1", "x2", "x3"]

sns.set_theme(style="whitegrid", context="notebook")

# Pairwise scatter/density plot colored by class.
pair = sns.pairplot(
    train,
    vars=features,
    hue="y",
    diag_kind="hist",
    corner=False,
    plot_kws={"alpha": 0.78, "s": 34, "edgecolor": "white", "linewidth": 0.25},
    diag_kws={"alpha": 0.60, "bins": 22},
    palette={0: "#2f6f9f", 1: "#c94f37"},
)
pair.fig.suptitle("Pairwise relationships by class", y=1.02)
pair.savefig(os.path.join(OUT_DIR, "classification_pairwise_by_class.png"), dpi=180, bbox_inches="tight")
plt.close(pair.fig)

# Correlation heatmap including the binary response.
corr = train[features + ["y"]].corr()
plt.figure(figsize=(6.2, 5.2))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    square=True,
    linewidths=0.6,
    cbar_kws={"shrink": 0.78},
)
plt.title("Correlation heatmap: predictors and y")
savefig("classification_correlation_heatmap.png")

# Predictor distributions by class.
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, feature in zip(axes, features):
    sns.histplot(
        data=train,
        x=feature,
        hue="y",
        bins=24,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.32,
        palette={0: "#2f6f9f", 1: "#c94f37"},
        ax=ax,
    )
    ax.set_title(f"{feature} distribution")
    ax.set_xlabel(feature)
fig.suptitle("Marginal predictor distributions by class", y=1.04)
savefig("classification_histograms_by_class.png")

# Box plots by class.
long_df = train.melt(id_vars="y", value_vars=features, var_name="predictor", value_name="value")
plt.figure(figsize=(8.2, 4.8))
sns.boxplot(
    data=long_df,
    x="predictor",
    y="value",
    hue="y",
    palette={0: "#2f6f9f", 1: "#c94f37"},
)
plt.title("Predictor distributions by class")
savefig("classification_boxplots_by_class.png")

# Class proportions across binned pairwise planes: helpful for nonlinear boundaries.
pairs = [("x1", "x2"), ("x1", "x3"), ("x2", "x3")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
for ax, (a, b) in zip(axes, pairs):
    sns.scatterplot(
        data=train,
        x=a,
        y=b,
        hue="y",
        palette={0: "#2f6f9f", 1: "#c94f37"},
        alpha=0.78,
        s=36,
        edgecolor="white",
        linewidth=0.25,
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color="#808080", linewidth=0.7, alpha=0.45)
    ax.axvline(0, color="#808080", linewidth=0.7, alpha=0.45)
    ax.set_title(f"{a} vs {b}")
fig.suptitle("Class pattern in pairwise predictor planes", y=1.04)
savefig("classification_pairwise_scatter_panels.png")

# Binned empirical probability of class 1 in each two-predictor plane.
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (a, b) in zip(axes, pairs):
    x_edges = np.linspace(train[a].min(), train[a].max(), 9)
    y_edges = np.linspace(train[b].min(), train[b].max(), 9)
    x_bin = pd.cut(train[a], bins=x_edges, include_lowest=True)
    y_bin = pd.cut(train[b], bins=y_edges, include_lowest=True)
    grouped = train.groupby([y_bin, x_bin], observed=False)["y"].agg(["mean", "count"])
    rate = grouped["mean"].unstack()
    count = grouped["count"].unstack()
    labels = rate.copy().astype(object)
    for row in rate.index:
        for col in rate.columns:
            if pd.isna(rate.loc[row, col]):
                labels.loc[row, col] = ""
            else:
                labels.loc[row, col] = f"{rate.loc[row, col]:.2f}\nn={int(count.loc[row, col])}"
    sns.heatmap(
        rate,
        ax=ax,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        center=0.5,
        annot=labels,
        fmt="",
        cbar=ax is axes[-1],
        annot_kws={"fontsize": 7},
    )
    ax.set_title(f"P(y=1) by bins: {a} and {b}")
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.tick_params(axis="x", labelrotation=75, labelsize=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=6)
fig.suptitle("Empirical class-1 rates in pairwise bins", y=1.05)
savefig("classification_binned_class_rate_heatmaps.png")

# Text summary for supervision and later reporting.
class_counts = train["y"].value_counts().sort_index()
class_rates = train["y"].value_counts(normalize=True).sort_index()
by_class = train.groupby("y")[features].agg(["mean", "std", "median"])
feature_corr = corr["y"].drop("y").sort_values(key=lambda s: s.abs(), ascending=False)

summary_lines = [
    "Assignment 2 exploratory data analysis",
    "======================================",
    "",
    f"Rows: {len(train)}",
    f"Columns: {', '.join(train.columns)}",
    "",
    "Class balance:",
]
for cls in class_counts.index:
    summary_lines.append(f"  y={cls}: {class_counts[cls]} ({class_rates[cls]:.3f})")

summary_lines.extend([
    "",
    "Correlations with y:",
])
for feature, value in feature_corr.items():
    summary_lines.append(f"  {feature}: {value:.3f}")

summary_lines.extend([
    "",
    "Means / standard deviations / medians by class:",
    by_class.round(3).to_string(),
    "",
    "Initial interpretation:",
    (
        "The marginal correlations with y are modest, so no single predictor appears "
        "to define the response by itself. The pairwise plots should be inspected for "
        "curved, quadrant, or interaction-like separation. If the classes overlap in "
        "one-dimensional histograms but form regions in two-dimensional planes, a "
        "plain additive logistic regression is likely too simple and polynomial or "
        "interaction terms should be tested."
    ),
])

with open(os.path.join(OUT_DIR, "classification_exploration_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Wrote exploratory outputs:")
for name in [
    "classification_pairwise_by_class.png",
    "classification_correlation_heatmap.png",
    "classification_histograms_by_class.png",
    "classification_boxplots_by_class.png",
    "classification_pairwise_scatter_panels.png",
    "classification_binned_class_rate_heatmaps.png",
    "classification_exploration_summary.txt",
]:
    print(f"  {name}")
