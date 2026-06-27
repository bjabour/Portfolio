from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
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


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "real_estate_risk_train_year_diffs_2011.csv"
TARGET = "distressed"
CALIBRATION_FOLDS = [2, 3, 4, 5, 6, 8, 10]
OUTER_SPLITS = 5
OUTER_REPEATS = 5


def score_predictions(y: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "mean_predicted_probability": probabilities.mean(),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    x = data[FEATURES]
    y = data[TARGET].astype(int)
    y_array = y.to_numpy()

    candidates = {
        "no_balancing": {
            "kind": "base",
            "strategy": "none",
            "calibration_folds": None,
        },
        "moderate_smote_ratio_0.5": {
            "kind": "base",
            "strategy": "smote_moderate",
            "smote_ratio": 0.5,
            "calibration_folds": None,
        },
    }
    for folds in CALIBRATION_FOLDS:
        candidates[f"sigmoid_calibration_k{folds}"] = {
            "kind": "calibrated",
            "strategy": "none",
            "calibration_folds": folds,
        }

    splitter = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )
    probability_sums = {
        candidate: np.zeros(len(y), dtype=float) for candidate in candidates
    }
    prediction_counts = np.zeros(len(y), dtype=int)
    fold_records = []

    for fold_number, (train_index, valid_index) in enumerate(splitter.split(x, y), 1):
        prediction_counts[valid_index] += 1
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]
        x_valid = x.iloc[valid_index]
        y_valid = y.iloc[valid_index].to_numpy()

        for candidate_name, settings in candidates.items():
            estimator = ScaledSmoteTreeClassifier(
                strategy=settings["strategy"],
                smote_ratio=settings.get("smote_ratio", 0.5),
                max_depth=4,
                min_samples_leaf=5,
                random_state=RANDOM_STATE + fold_number,
            )

            if settings["kind"] == "calibrated":
                model = CalibratedClassifierCV(
                    estimator=estimator,
                    method="sigmoid",
                    cv=settings["calibration_folds"],
                )
            else:
                model = estimator

            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_valid)[:, 1]
            probability_sums[candidate_name][valid_index] += probabilities
            fold_records.append(
                {
                    "outer_fold": fold_number,
                    "candidate": candidate_name,
                    "method": (
                        "sigmoid_calibration"
                        if settings["kind"] == "calibrated"
                        else settings["strategy"]
                    ),
                    "calibration_folds": settings["calibration_folds"],
                    **score_predictions(y_valid, probabilities),
                }
            )

    oof_probabilities = {
        candidate: probability_sums[candidate] / prediction_counts
        for candidate in candidates
    }
    summary_records = []
    for candidate_name, probabilities in oof_probabilities.items():
        settings = candidates[candidate_name]
        summary_records.append(
            {
                "candidate": candidate_name,
                "method": (
                    "sigmoid_calibration"
                    if settings["kind"] == "calibrated"
                    else settings["strategy"]
                ),
                "calibration_folds": settings["calibration_folds"],
                **score_predictions(y_array, probabilities),
                "observed_prevalence": y.mean(),
            }
        )

    summary = pd.DataFrame(summary_records).sort_values("log_loss")
    fold_metrics = pd.DataFrame(fold_records)
    fold_variability = (
        fold_metrics.groupby(
            ["candidate", "method", "calibration_folds"],
            dropna=False,
        )
        .agg(
            mean_fold_log_loss=("log_loss", "mean"),
            sd_fold_log_loss=("log_loss", "std"),
            mean_fold_auc=("roc_auc", "mean"),
            sd_fold_auc=("roc_auc", "std"),
        )
        .reset_index()
    )

    probabilities_output = pd.DataFrame(
        {
            "source_row_index": data.index,
            "actual_distressed": y,
            **oof_probabilities,
        }
    )
    summary.to_csv(OUTPUT_DIR / "calibration_fold_tuning_summary.csv", index=False)
    fold_metrics.to_csv(
        OUTPUT_DIR / "calibration_fold_tuning_fold_metrics.csv",
        index=False,
    )
    fold_variability.to_csv(
        OUTPUT_DIR / "calibration_fold_tuning_variability.csv",
        index=False,
    )
    probabilities_output.to_csv(
        OUTPUT_DIR / "calibration_fold_tuning_oof_probabilities.csv",
        index=False,
    )

    sns.set_theme(style="whitegrid")
    sigmoid_summary = summary[summary["method"] == "sigmoid_calibration"].copy()
    sigmoid_summary["calibration_folds"] = sigmoid_summary[
        "calibration_folds"
    ].astype(int)

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=sigmoid_summary.sort_values("calibration_folds"),
        x="calibration_folds",
        y="log_loss",
        marker="o",
        linewidth=2.5,
        color="#4C78A8",
    )
    plt.axhline(
        summary.loc[summary["candidate"] == "no_balancing", "log_loss"].iloc[0],
        color="#F58518",
        linestyle="--",
        label="No balancing",
    )
    plt.axhline(
        summary.loc[
            summary["candidate"] == "moderate_smote_ratio_0.5",
            "log_loss",
        ].iloc[0],
        color="#E45756",
        linestyle="--",
        label="Moderate SMOTE",
    )
    plt.xlabel("Inner sigmoid calibration folds (k)")
    plt.ylabel("Repeated out-of-fold log loss")
    plt.title("Tuning Sigmoid Calibration Fold Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_fold_tuning.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5.5))
    plot_summary = summary.copy()
    sns.barplot(
        data=plot_summary,
        x="log_loss",
        y="candidate",
        color="#4C78A8",
    )
    plt.xlabel("Repeated out-of-fold log loss (lower is better)")
    plt.ylabel("")
    plt.title("Imbalance Method and Calibration-Fold Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_fold_log_loss_comparison.png", dpi=200)
    plt.close()

    best = summary.iloc[0]
    notes = f"""# Calibration Fold Tuning

- Outer evaluation: {OUTER_SPLITS}-fold stratified CV repeated {OUTER_REPEATS} times.
- Candidate inner calibration folds: {", ".join(map(str, CALIBRATION_FOLDS))}.
- Fixed tree: max depth 4, minimum leaf size 5.
- Moderate SMOTE ratio: 0.5.
- Selection metric: repeated out-of-fold log loss.
- Best candidate: `{best["candidate"]}`.
- Best log loss: {best["log_loss"]:.6f}.
- Best ROC AUC: {best["roc_auc"]:.6f} for the selected candidate.

The number of folds is a genuine fitted-model setting only for sigmoid
calibration. Plain and SMOTE trees are included as fixed reference methods
and are evaluated on exactly the same outer validation folds.
"""
    (OUTPUT_DIR / "calibration_fold_tuning_summary.md").write_text(
        notes,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
