from pathlib import Path

import numpy as np
import pandas as pd
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
SMOTE_RATIO = 0.7
K_NEIGHBORS = 10


def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
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

    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )
    probability_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)
    fold_records = []

    for fold_number, (train_index, valid_index) in enumerate(splitter.split(x, y), 1):
        model = ScaledSmoteTreeClassifier(
            strategy="smote_moderate",
            smote_ratio=SMOTE_RATIO,
            k_neighbors=K_NEIGHBORS,
            max_depth=4,
            min_samples_leaf=5,
            random_state=RANDOM_STATE + fold_number,
        )
        model.fit(x.iloc[train_index], y.iloc[train_index])
        probabilities = model.predict_proba(x.iloc[valid_index])[:, 1]
        probability_sum[valid_index] += probabilities
        prediction_count[valid_index] += 1
        fold_records.append(
            {
                "outer_fold": fold_number,
                "smote_ratio": SMOTE_RATIO,
                "k_neighbors": K_NEIGHBORS,
                **metrics(y.iloc[valid_index].to_numpy(), probabilities),
            }
        )

    averaged_probabilities = probability_sum / prediction_count
    fold_metrics = pd.DataFrame(fold_records)
    ensemble_metrics = metrics(y.to_numpy(), averaged_probabilities)

    result = pd.DataFrame(
        [
            {
                "smote_ratio": SMOTE_RATIO,
                "k_neighbors": K_NEIGHBORS,
                "mean_fold_log_loss": fold_metrics["log_loss"].mean(),
                "sd_fold_log_loss": fold_metrics["log_loss"].std(),
                "mean_fold_auc": fold_metrics["roc_auc"].mean(),
                "sd_fold_auc": fold_metrics["roc_auc"].std(),
                "ensemble_oof_log_loss": ensemble_metrics["log_loss"],
                "ensemble_oof_brier_score": ensemble_metrics["brier_score"],
                "ensemble_oof_auc": ensemble_metrics["roc_auc"],
                "ensemble_oof_precision": ensemble_metrics["precision"],
                "ensemble_oof_recall": ensemble_metrics["recall"],
                "ensemble_mean_probability": ensemble_metrics[
                    "mean_predicted_probability"
                ],
                "observed_prevalence": y.mean(),
            }
        ]
    )
    result.to_csv(OUTPUT_DIR / "smote_k10_ratio07_result.csv", index=False)
    fold_metrics.to_csv(
        OUTPUT_DIR / "smote_k10_ratio07_fold_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "source_row_index": data.index,
            "actual_distressed": y,
            "smote_k10_ratio07_probability": averaged_probabilities,
        }
    ).to_csv(
        OUTPUT_DIR / "smote_k10_ratio07_oof_probabilities.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
