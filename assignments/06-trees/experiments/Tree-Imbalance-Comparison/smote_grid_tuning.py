from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from tree_imbalance_comparison import (
    FEATURES,
    RANDOM_STATE,
    ScaledSmoteTreeClassifier,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = ASSIGNMENT_DIR / "trees_train_year_diffs_2011.csv"
TARGET = "distressed"
NEIGHBOR_COUNTS = [2, 3, 4]
SMOTE_RATIOS = [0.2, 0.3, 0.4]
OUTER_SPLITS = 5
OUTER_REPEATS = 5


def calculate_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "mean_predicted_probability": probabilities.mean(),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    x = data[FEATURES]
    y = data[TARGET].astype(int)
    y_array = y.to_numpy()

    candidates = [
        {
            "candidate": f"smote_ratio_{ratio:.1f}_k{neighbors}",
            "smote_ratio": ratio,
            "k_neighbors": neighbors,
        }
        for ratio in SMOTE_RATIOS
        for neighbors in NEIGHBOR_COUNTS
    ]

    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )
    probability_sums = {
        settings["candidate"]: np.zeros(len(y), dtype=float)
        for settings in candidates
    }
    prediction_counts = np.zeros(len(y), dtype=int)
    fold_records = []

    for fold_number, (train_index, valid_index) in enumerate(splitter.split(x, y), 1):
        prediction_counts[valid_index] += 1
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]
        x_valid = x.iloc[valid_index]
        y_valid = y.iloc[valid_index].to_numpy()

        for settings in candidates:
            model = ScaledSmoteTreeClassifier(
                strategy="smote_moderate",
                smote_ratio=settings["smote_ratio"],
                k_neighbors=settings["k_neighbors"],
                max_depth=4,
                min_samples_leaf=5,
                random_state=RANDOM_STATE + fold_number,
            )
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_valid)[:, 1]
            probability_sums[settings["candidate"]][valid_index] += probabilities
            fold_records.append(
                {
                    "outer_fold": fold_number,
                    **settings,
                    **calculate_metrics(y_valid, probabilities),
                }
            )

    summary_records = []
    probability_columns = {}
    for settings in candidates:
        candidate = settings["candidate"]
        probabilities = probability_sums[candidate] / prediction_counts
        probability_columns[candidate] = probabilities
        summary_records.append(
            {
                **settings,
                **calculate_metrics(y_array, probabilities),
                "observed_prevalence": y.mean(),
            }
        )

    summary = pd.DataFrame(summary_records).sort_values("log_loss")
    fold_metrics = pd.DataFrame(fold_records)
    variability = (
        fold_metrics.groupby(
            ["candidate", "smote_ratio", "k_neighbors"]
        )
        .agg(
            mean_fold_log_loss=("log_loss", "mean"),
            sd_fold_log_loss=("log_loss", "std"),
            mean_fold_auc=("roc_auc", "mean"),
            sd_fold_auc=("roc_auc", "std"),
        )
        .reset_index()
    )

    summary.to_csv(OUTPUT_DIR / "smote_grid_summary.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "smote_grid_fold_metrics.csv", index=False)
    variability.to_csv(OUTPUT_DIR / "smote_grid_variability.csv", index=False)
    pd.DataFrame(
        {
            "source_row_index": data.index,
            "actual_distressed": y,
            **probability_columns,
        }
    ).to_csv(OUTPUT_DIR / "smote_grid_oof_probabilities.csv", index=False)

    heatmap = summary.pivot(
        index="smote_ratio",
        columns="k_neighbors",
        values="log_loss",
    )
    sns.set_theme(style="white")
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu_r",
        cbar_kws={"label": "Out-of-fold log loss"},
    )
    plt.xlabel("SMOTE neighbor count")
    plt.ylabel("Minority-to-majority ratio")
    plt.title("Manual SMOTE Hyperparameter Tuning")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "smote_grid_log_loss_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=summary,
        x="smote_ratio",
        y="log_loss",
        hue="k_neighbors",
        marker="o",
        palette="tab10",
    )
    plt.xlabel("Target minority-to-majority ratio")
    plt.ylabel("Out-of-fold log loss")
    plt.title("SMOTE Ratio and Neighbor Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "smote_grid_log_loss_lines.png", dpi=200)
    plt.close()

    best_ensemble = summary.iloc[0]
    best_single_fit = variability.sort_values("mean_fold_log_loss").iloc[0]
    notes = f"""# SMOTE Grid Tuning

- Outer evaluation: {OUTER_SPLITS}-fold stratified CV repeated {OUTER_REPEATS} times.
- Neighbor counts: {NEIGHBOR_COUNTS}.
- Target minority-to-majority ratios: {SMOTE_RATIOS}.
- Fixed tree: max depth 4, minimum leaf size 5.
- Single-fit selection metric: mean log loss across the 25 untouched validation folds.
- Original minority-to-majority ratio: {int(y.sum())}/{int((1-y).sum())} = {y.sum() / (1-y).sum():.4f}.
- Best single-fit setting: ratio `{best_single_fit["smote_ratio"]:.1f}`, `k_neighbors={int(best_single_fit["k_neighbors"])}`.
- Best mean fold log loss: {best_single_fit["mean_fold_log_loss"]:.6f}.
- Best repeated-prediction ensemble setting: ratio `{best_ensemble["smote_ratio"]:.1f}`, `k_neighbors={int(best_ensemble["k_neighbors"])}`.
- Its log loss after averaging five out-of-fold predictions per row: {best_ensemble["log_loss"]:.6f}.

Scaling and SMOTE were fitted only inside each outer training fold.
The ensemble score is smoother because each observation's five predictions
are averaged before scoring; it is not the expected score of one fitted tree.
"""
    (OUTPUT_DIR / "smote_grid_summary.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
