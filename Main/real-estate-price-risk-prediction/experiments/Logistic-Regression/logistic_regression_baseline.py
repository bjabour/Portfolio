from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "real_estate_risk_train_year_diffs_2011.csv"
TARGET = "distressed"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    features = [column for column in data.columns if column not in {TARGET, "sale_price_keur"}]
    x = data[features]
    y = data[TARGET]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), features)],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_valid, predictions, labels=[0, 1])

    metrics = pd.DataFrame(
        [
            {
                "train_rows": len(x_train),
                "validation_rows": len(x_valid),
                "validation_distressed": int(y_valid.sum()),
                "threshold": 0.5,
                "accuracy": accuracy_score(y_valid, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_valid, predictions),
                "precision": precision_score(y_valid, predictions, zero_division=0),
                "recall": recall_score(y_valid, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_valid, probabilities),
                "log_loss": log_loss(y_valid, probabilities),
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            }
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "logistic_regression_metrics.csv", index=False)

    validation_results = x_valid.copy()
    validation_results.insert(0, "source_row_index", validation_results.index)
    validation_results["actual_distressed"] = y_valid
    validation_results["predicted_probability"] = probabilities
    validation_results["predicted_distressed"] = predictions
    validation_results.sort_values("source_row_index").to_csv(
        OUTPUT_DIR / "validation_predictions.csv", index=False
    )

    coefficients = pd.DataFrame(
        {
            "feature": features,
            "standardized_coefficient": model.named_steps[
                "logistic_regression"
            ].coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients[
        "standardized_coefficient"
    ].abs()
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(
        OUTPUT_DIR / "logistic_regression_coefficients.csv", index=False
    )

    sns.set_theme(style="white")
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Baseline Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    false_positive_rate, true_positive_rate, _ = roc_curve(y_valid, probabilities)
    auc = roc_auc_score(y_valid, probabilities)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="#4C78A8",
        linewidth=2.5,
        label=f"Logistic regression (AUC = {auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Baseline Logistic Regression ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
