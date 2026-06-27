from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "real_estate_risk_train_year_diffs_2011.csv"
TARGET = "distressed"
RANDOM_STATE = 42

LOG_COLUMNS = [
    "LotArea",
    "TotalBsmtSF",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
]
STANDARD_COLUMNS = [
    "GrLivArea",
    "OverallQual",
    "OverallCond",
    "years_since_built_2011",
    "years_since_remodel_2011",
    "BedroomAbvGr",
    "Fireplaces",
]
FEATURES = LOG_COLUMNS + STANDARD_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "log_standard",
                Pipeline(
                    [
                        (
                            "log1p",
                            FunctionTransformer(
                                np.log1p,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("standard_scale", StandardScaler()),
                    ]
                ),
                LOG_COLUMNS,
            ),
            ("standard_scale", StandardScaler(), STANDARD_COLUMNS),
        ],
        remainder="drop",
    )


def manual_smote(
    x: np.ndarray,
    y: np.ndarray,
    target_ratio: float,
    k_neighbors: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_count = int(counts.max())
    minority_x = x[y == minority_class]
    target_minority_count = int(np.ceil(majority_count * target_ratio))
    synthetic_count = target_minority_count - len(minority_x)

    if synthetic_count <= 0:
        return x.copy(), y.copy()

    k = min(k_neighbors, len(minority_x) - 1)
    distances = np.linalg.norm(
        minority_x[:, np.newaxis, :] - minority_x[np.newaxis, :, :],
        axis=2,
    )
    neighbor_indices = np.argsort(distances, axis=1)[:, 1 : k + 1]

    synthetic_rows = np.empty((synthetic_count, x.shape[1]), dtype=float)
    for synthetic_index in range(synthetic_count):
        source_index = rng.integers(len(minority_x))
        neighbor_index = rng.choice(neighbor_indices[source_index])
        interpolation = rng.random()
        source = minority_x[source_index]
        neighbor = minority_x[neighbor_index]
        synthetic_rows[synthetic_index] = source + interpolation * (
            neighbor - source
        )

    synthetic_y = np.full(synthetic_count, minority_class, dtype=y.dtype)
    return (
        np.vstack([x, synthetic_rows]),
        np.concatenate([y, synthetic_y]),
    )


class ScaledSmoteTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        strategy: str = "none",
        smote_ratio: float = 0.5,
        k_neighbors: int = 5,
        max_depth: int = 4,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.smote_ratio = smote_ratio
        self.k_neighbors = k_neighbors
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.preprocessor_ = build_preprocessor()
        x_scaled = self.preprocessor_.fit_transform(x)
        y_array = np.asarray(y)

        class_weight = None
        if self.strategy == "class_weight_balanced":
            class_weight = "balanced"
        elif self.strategy in {"smote_moderate", "smote_full"}:
            ratio = self.smote_ratio if self.strategy == "smote_moderate" else 1.0
            x_scaled, y_array = manual_smote(
                x_scaled,
                y_array,
                target_ratio=ratio,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )
        elif self.strategy != "none":
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.model_ = DecisionTreeClassifier(
            criterion="log_loss",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=class_weight,
            random_state=self.random_state,
        )
        self.model_.fit(x_scaled, y_array)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.preprocessor_.transform(x))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(x), axis=1)]


def calculate_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "mean_predicted_probability": probabilities.mean(),
        "observed_prevalence": y.mean(),
    }


def repeated_cv_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    strategies: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )
    probability_sums = {
        name: np.zeros(len(y), dtype=float) for name in strategies
    }
    prediction_counts = np.zeros(len(y), dtype=int)
    fold_records = []

    for fold_number, (train_index, valid_index) in enumerate(splitter.split(x, y), 1):
        prediction_counts[valid_index] += 1
        for strategy_name, parameters in strategies.items():
            estimator = ScaledSmoteTreeClassifier(
                strategy=parameters["strategy"],
                smote_ratio=parameters.get("smote_ratio", 0.5),
                random_state=RANDOM_STATE + fold_number,
            )
            estimator.fit(x.iloc[train_index], y.iloc[train_index])
            probabilities = estimator.predict_proba(x.iloc[valid_index])[:, 1]
            probability_sums[strategy_name][valid_index] += probabilities

            fold_metrics = calculate_metrics(y.iloc[valid_index].to_numpy(), probabilities)
            fold_records.append(
                {
                    "fold": fold_number,
                    "strategy": strategy_name,
                    **fold_metrics,
                }
            )

    averaged = pd.DataFrame(
        {
            name: probability_sums[name] / prediction_counts
            for name in strategies
        }
    )
    return averaged, pd.DataFrame(fold_records)


def calibrated_cv_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    strategy_parameters: dict,
) -> np.ndarray:
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )
    probability_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)

    for fold_number, (train_index, valid_index) in enumerate(splitter.split(x, y), 1):
        base_estimator = ScaledSmoteTreeClassifier(
            strategy=strategy_parameters["strategy"],
            smote_ratio=strategy_parameters.get("smote_ratio", 0.5),
            random_state=RANDOM_STATE + fold_number,
        )
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=3,
        )
        calibrated.fit(x.iloc[train_index], y.iloc[train_index])
        probability_sum[valid_index] += calibrated.predict_proba(
            x.iloc[valid_index]
        )[:, 1]
        prediction_count[valid_index] += 1

    return probability_sum / prediction_count


def save_plots(
    y: np.ndarray,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    selected_strategy: str,
    calibrated_name: str,
) -> None:
    sns.set_theme(style="whitegrid")

    plot_data = summary.sort_values("log_loss")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=plot_data, x="log_loss", y="strategy", color="#4C78A8")
    plt.axvline(
        -(
            y.mean() * np.log(y.mean())
            + (1 - y.mean()) * np.log(1 - y.mean())
        ),
        color="#E45756",
        linestyle="--",
        label="Constant-prevalence baseline",
    )
    plt.title("Decision-Tree Imbalance Strategies: Out-of-Fold Log Loss")
    plt.xlabel("Log loss (lower is better)")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_log_loss_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for strategy in predictions.columns:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y, predictions[strategy]
        )
        auc = roc_auc_score(y, predictions[strategy])
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{strategy} (AUC={auc:.3f})",
        )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Out-of-Fold ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for strategy in predictions.columns:
        observed, predicted = calibration_curve(
            y,
            predictions[strategy],
            n_bins=6,
            strategy="quantile",
        )
        plt.plot(predicted, observed, marker="o", label=strategy)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed distressed rate")
    plt.title("Out-of-Fold Calibration Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "strategy_calibration_curves.png", dpi=200)
    plt.close()

    selected_probabilities = predictions[calibrated_name]
    selected_predictions = (selected_probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y, selected_predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title(f"Confusion Matrix: Calibrated {selected_strategy}")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "selected_calibrated_confusion_matrix.png", dpi=200)
    plt.close()


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    x = data[FEATURES]
    y = data[TARGET].astype(int)

    strategies = {
        "none": {"strategy": "none"},
        "class_weight_balanced": {"strategy": "class_weight_balanced"},
        "smote_ratio_0.5": {
            "strategy": "smote_moderate",
            "smote_ratio": 0.5,
        },
        "smote_ratio_1.0": {
            "strategy": "smote_full",
            "smote_ratio": 1.0,
        },
    }

    predictions, fold_metrics = repeated_cv_predictions(x, y, strategies)
    summary_records = [
        {"strategy": strategy, **calculate_metrics(y.to_numpy(), predictions[strategy])}
        for strategy in predictions.columns
    ]
    initial_summary = pd.DataFrame(summary_records).sort_values("log_loss")
    selected_strategy = initial_summary.iloc[0]["strategy"]

    calibrated_name = f"{selected_strategy}_sigmoid_calibrated"
    predictions[calibrated_name] = calibrated_cv_predictions(
        x,
        y,
        strategies[selected_strategy],
    )
    summary_records.append(
        {
            "strategy": calibrated_name,
            **calculate_metrics(y.to_numpy(), predictions[calibrated_name]),
        }
    )
    summary = pd.DataFrame(summary_records).sort_values("log_loss")

    output_predictions = pd.DataFrame(
        {
            "source_row_index": data.index,
            "actual_distressed": y,
            **{column: predictions[column] for column in predictions.columns},
        }
    )
    output_predictions.to_csv(
        OUTPUT_DIR / "out_of_fold_probabilities.csv",
        index=False,
    )
    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    fold_summary = (
        fold_metrics.groupby("strategy")
        .agg(
            mean_fold_log_loss=("log_loss", "mean"),
            sd_fold_log_loss=("log_loss", "std"),
            mean_fold_auc=("roc_auc", "mean"),
            sd_fold_auc=("roc_auc", "std"),
        )
        .reset_index()
    )
    fold_summary.to_csv(OUTPUT_DIR / "fold_variability_summary.csv", index=False)

    save_plots(
        y.to_numpy(),
        predictions,
        summary,
        selected_strategy,
        calibrated_name,
    )

    best = summary.iloc[0]
    notes = f"""# Tree Imbalance Strategy Comparison

- Dataset: `real_estate_risk_train_year_diffs_2011.csv`
- Evaluation: 5-fold stratified CV repeated 5 times
- Tree: `criterion="log_loss"`, `max_depth=4`, `min_samples_leaf=5`
- Primary selection metric: out-of-fold log loss
- Training prevalence: {y.mean():.3%}
- Initial best strategy: `{selected_strategy}`
- Sigmoid calibration was then tested for that strategy using inner 3-fold CV.
- Overall best result: `{best["strategy"]}` with log loss {best["log_loss"]:.4f}

SMOTE and all preprocessing were fitted only inside each training fold.
Validation observations remained untouched. The area features used `log1p`
followed by standard scaling; the remaining predictors used standard scaling.
"""
    (OUTPUT_DIR / "experiment_summary.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
