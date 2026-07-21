import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=ConvergenceWarning)

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
TRAIN_PATH = PROJECT_DIR / "real_estate_risk_train.csv"
TEST_PATH = PROJECT_DIR / "real_estate_risk_test.csv"
RF_DIR = EXPERIMENTS_DIR / "Random-Forest-Classification"
XGB_DIR = EXPERIMENTS_DIR / "XGBoost-Classification"
RF_OOF_PATH = RF_DIR / "rf_clf_selected_oof_probabilities.csv"
XGB_OOF_PATH = XGB_DIR / "xgb_clf_selected_oof_probabilities.csv"
RF_TEST_PATH = RF_DIR / "rf_clf_final_test_probabilities.csv"
XGB_TEST_PATH = XGB_DIR / "xgb_clf_final_test_probabilities.csv"

SEEDS = (0, 42, 2026, 9618, 19618, 31415)
FEATURES = [
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GarageArea",
    "BedroomAbvGr",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF",
]
TARGET = "distressed"
RAW_C_GRID = np.logspace(-4, 3, 15)
NONLINEAR_C_GRID = np.logspace(-3, 2, 11)
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 101)
EPSILON = 1e-15


def clip_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1 - EPSILON)


def domain_log_transform(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=float).copy()
    for column in (0, 1, 6, 7, 10, 11):
        transformed[:, column] = np.log1p(
            np.maximum(transformed[:, column], 0.0)
        )
    return transformed


def candidate_specifications() -> dict[str, tuple[Pipeline, dict]]:
    categorical = [
        "OverallQual",
        "OverallCond",
        "BedroomAbvGr",
        "Fireplaces",
    ]
    numeric = [feature for feature in FEATURES if feature not in categorical]
    one_hot_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical,
            ),
        ]
    )
    return {
        "raw_l2": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=10000,
                        ),
                    ),
                ]
            ),
            {"logistic__C": RAW_C_GRID},
        ),
        "domain_log_l2": (
            Pipeline(
                [
                    (
                        "domain_log",
                        FunctionTransformer(
                            domain_log_transform,
                            validate=False,
                        ),
                    ),
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=10000,
                        ),
                    ),
                ]
            ),
            {"logistic__C": RAW_C_GRID},
        ),
        "raw_l1": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            solver="saga",
                            l1_ratio=1.0,
                            max_iter=20000,
                        ),
                    ),
                ]
            ),
            {"logistic__C": RAW_C_GRID},
        ),
        "ordinal_onehot_l2": (
            Pipeline(
                [
                    ("preprocessor", one_hot_preprocessor),
                    (
                        "logistic",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=10000,
                        ),
                    ),
                ]
            ),
            {"logistic__C": RAW_C_GRID},
        ),
        "quadratic_l2": (
            Pipeline(
                [
                    (
                        "basis",
                        PolynomialFeatures(degree=2, include_bias=False),
                    ),
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=20000,
                        ),
                    ),
                ]
            ),
            {"logistic__C": NONLINEAR_C_GRID},
        ),
        "spline_degree2_l2": (
            Pipeline(
                [
                    (
                        "basis",
                        SplineTransformer(
                            n_knots=3,
                            degree=2,
                            include_bias=False,
                        ),
                    ),
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=20000,
                        ),
                    ),
                ]
            ),
            {"logistic__C": NONLINEAR_C_GRID},
        ),
        "spline_degree3_l2": (
            Pipeline(
                [
                    (
                        "basis",
                        SplineTransformer(
                            n_knots=3,
                            degree=3,
                            include_bias=False,
                        ),
                    ),
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=20000,
                        ),
                    ),
                ]
            ),
            {"logistic__C": NONLINEAR_C_GRID},
        ),
    }


def nested_logistic_evaluation(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = []
    for seed in SEEDS:
        outer = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
        for fold, (training, validation) in enumerate(
            outer.split(x, y),
            start=1,
        ):
            inner = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed * 100 + fold + 17,
            )
            for model_name, (
                estimator,
                parameter_grid,
            ) in candidate_specifications().items():
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=parameter_grid,
                    cv=inner,
                    scoring="neg_log_loss",
                    n_jobs=1,
                    error_score="raise",
                )
                search.fit(x.iloc[training], y.iloc[training])
                probabilities = clip_probabilities(
                    search.best_estimator_.predict_proba(
                        x.iloc[validation]
                    )[:, 1]
                )
                actual = y.iloc[validation].to_numpy(dtype=int)
                metric_rows.append(
                    {
                        "cv_seed": seed,
                        "fold": fold,
                        "model": model_name,
                        "log_loss": float(log_loss(actual, probabilities)),
                        "brier_score": float(
                            brier_score_loss(actual, probabilities)
                        ),
                        "roc_auc": float(
                            roc_auc_score(actual, probabilities)
                        ),
                        "mean_probability": float(
                            probabilities.mean()
                        ),
                        "observed_prevalence": float(actual.mean()),
                        "inner_selected_log_loss": float(
                            -search.best_score_
                        ),
                        "selected_parameters": json.dumps(
                            {
                                key: float(value)
                                if isinstance(value, np.floating)
                                else value
                                for key, value in search.best_params_.items()
                            },
                            sort_keys=True,
                        ),
                    }
                )
                for row_index, outcome, probability in zip(
                    validation,
                    actual,
                    probabilities,
                ):
                    prediction_rows.append(
                        {
                            "cv_seed": seed,
                            "fold": fold,
                            "row_index": int(row_index),
                            "actual": int(outcome),
                            "model": model_name,
                            "probability": float(probability),
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def validate_tree_predictions(
    frame: pd.DataFrame,
    model_name: str,
    y: pd.Series,
) -> pd.DataFrame:
    expected_columns = {
        "cv_seed",
        "fold",
        "row_index",
        "actual",
        "probability",
    }
    if set(frame.columns) != expected_columns:
        raise AssertionError(
            f"Unexpected {model_name} OOF columns: {frame.columns.tolist()}"
        )
    if len(frame) != len(SEEDS) * len(y):
        raise AssertionError(f"Unexpected {model_name} OOF row count")
    if set(frame["cv_seed"]) != set(SEEDS):
        raise AssertionError(f"Unexpected {model_name} CV seeds")
    for seed, group in frame.groupby("cv_seed"):
        ordered = group.sort_values("row_index")
        if not np.array_equal(
            ordered["row_index"].to_numpy(),
            np.arange(len(y)),
        ):
            raise AssertionError(
                f"{model_name} seed {seed} does not cover all rows"
            )
        if not np.array_equal(
            ordered["actual"].to_numpy(dtype=int),
            y.to_numpy(dtype=int),
        ):
            raise AssertionError(
                f"{model_name} targets differ from source data"
            )
    validated = frame.copy()
    validated["model"] = model_name
    return validated


def metric_summary(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    for (model_name, seed), group in predictions.groupby(
        ["model", "cv_seed"]
    ):
        probabilities = clip_probabilities(group["probability"].to_numpy())
        actual = group["actual"].to_numpy(dtype=int)
        repeat_rows.append(
            {
                "model": model_name,
                "cv_seed": int(seed),
                "log_loss": float(log_loss(actual, probabilities)),
                "brier_score": float(
                    brier_score_loss(actual, probabilities)
                ),
                "roc_auc": float(roc_auc_score(actual, probabilities)),
                "mean_probability": float(probabilities.mean()),
                "observed_prevalence": float(actual.mean()),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)
    summary = (
        repeat_metrics.groupby("model")
        .agg(
            mean_log_loss=("log_loss", "mean"),
            sd_repeat_log_loss=("log_loss", "std"),
            mean_brier_score=("brier_score", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_probability=("mean_probability", "mean"),
            observed_prevalence=("observed_prevalence", "mean"),
            n_repeats=("cv_seed", "count"),
        )
        .reset_index()
    )
    summary["se_repeat_log_loss"] = (
        summary["sd_repeat_log_loss"] / np.sqrt(summary["n_repeats"])
    )
    summary["absolute_prevalence_bias"] = (
        summary["mean_probability"] - summary["observed_prevalence"]
    ).abs()
    return summary.sort_values("mean_log_loss"), repeat_metrics


def paired_comparisons(
    repeat_metrics: pd.DataFrame,
    reference_model: str,
    comparison_models: list[str],
) -> pd.DataFrame:
    reference = repeat_metrics.loc[
        repeat_metrics["model"] == reference_model,
        ["cv_seed", "log_loss"],
    ].rename(columns={"log_loss": "reference_log_loss"})
    rows = []
    for comparison_model in comparison_models:
        compared = repeat_metrics.loc[
            repeat_metrics["model"] == comparison_model,
            ["cv_seed", "log_loss"],
        ].rename(columns={"log_loss": "comparison_log_loss"})
        paired = reference.merge(compared, on="cv_seed", validate="one_to_one")
        difference = (
            paired["comparison_log_loss"]
            - paired["reference_log_loss"]
        ).to_numpy()
        rng = np.random.default_rng(9618 + len(comparison_model))
        bootstrap = np.asarray(
            [
                rng.choice(difference, size=len(difference)).mean()
                for _ in range(20000)
            ]
        )
        rows.append(
            {
                "reference_model": reference_model,
                "comparison_model": comparison_model,
                "mean_comparison_minus_reference_log_loss": float(
                    difference.mean()
                ),
                "bootstrap_95_low": float(
                    np.quantile(bootstrap, 0.025)
                ),
                "bootstrap_95_high": float(
                    np.quantile(bootstrap, 0.975)
                ),
                "reference_wins_out_of_repeats": int(
                    (difference > 0).sum()
                ),
                "n_repeats": len(difference),
            }
        )
    return pd.DataFrame(rows)


def tune_blend_weights(
    predictions: pd.DataFrame,
    first_model: str,
    second_model: str,
    blend_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    selected = predictions.loc[
        predictions["model"].isin([first_model, second_model])
    ]
    wide = selected.pivot(
        index=["cv_seed", "fold", "row_index", "actual"],
        columns="model",
        values="probability",
    ).reset_index()
    if wide[[first_model, second_model]].isna().any().any():
        raise AssertionError("Incomplete blend inputs")

    weight_rows = []
    prediction_rows = []
    for held_seed in SEEDS:
        fit = wide["cv_seed"] != held_seed
        validation = ~fit
        losses = []
        for weight in BLEND_WEIGHTS:
            probabilities = (
                weight * wide.loc[fit, first_model]
                + (1 - weight) * wide.loc[fit, second_model]
            )
            losses.append(
                log_loss(wide.loc[fit, "actual"], probabilities)
            )
        selected_weight = float(BLEND_WEIGHTS[int(np.argmin(losses))])
        held = wide.loc[validation].copy()
        held["probability"] = (
            selected_weight * held[first_model]
            + (1 - selected_weight) * held[second_model]
        )
        weight_rows.append(
            {
                "held_out_seed": held_seed,
                "first_model": first_model,
                "second_model": second_model,
                "weight_on_first_model": selected_weight,
                "training_log_loss": float(np.min(losses)),
                "held_out_log_loss": float(
                    log_loss(held["actual"], held["probability"])
                ),
            }
        )
        for row in held.itertuples(index=False):
            prediction_rows.append(
                {
                    "cv_seed": int(row.cv_seed),
                    "fold": int(row.fold),
                    "row_index": int(row.row_index),
                    "actual": int(row.actual),
                    "model": blend_name,
                    "probability": float(row.probability),
                }
            )

    all_losses = []
    for weight in BLEND_WEIGHTS:
        probabilities = (
            weight * wide[first_model]
            + (1 - weight) * wide[second_model]
        )
        all_losses.append(log_loss(wide["actual"], probabilities))
    full_weight = float(BLEND_WEIGHTS[int(np.argmin(all_losses))])
    return (
        pd.DataFrame(weight_rows),
        pd.DataFrame(prediction_rows),
        full_weight,
    )


def repeated_c_tuning(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, float]:
    rows = []
    for c_value in RAW_C_GRID:
        fold_losses = []
        for seed in SEEDS:
            splitter = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed,
            )
            for training, validation in splitter.split(x, y):
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "logistic",
                            LogisticRegression(
                                C=float(c_value),
                                solver="lbfgs",
                                max_iter=10000,
                            ),
                        ),
                    ]
                )
                model.fit(x.iloc[training], y.iloc[training])
                probabilities = model.predict_proba(
                    x.iloc[validation]
                )[:, 1]
                fold_losses.append(
                    log_loss(y.iloc[validation], probabilities)
                )
        rows.append(
            {
                "C": float(c_value),
                "mean_log_loss": float(np.mean(fold_losses)),
                "sd_fold_log_loss": float(
                    np.std(fold_losses, ddof=1)
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("C")
    selected_c = float(
        table.sort_values(["mean_log_loss", "C"]).iloc[0]["C"]
    )
    table["selected"] = np.isclose(table["C"], selected_c)
    return table, selected_c


def save_plots(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    plot_order = [
        "blend_raw_l2_rf",
        "spline_degree2_l2",
        "raw_l2",
        "random_forest",
        "xgboost",
        "constant_prevalence",
        "balanced_logistic",
    ]
    available = [
        model for model in plot_order if model in set(summary["model"])
    ]
    plot_data = summary.set_index("model").loc[available].reset_index()
    colors = [
        "#4C956C"
        if model == "blend_raw_l2_rf"
        else "#2A6F97"
        if "l2" in model
        else "#C14953"
        if model in {"random_forest", "xgboost"}
        else "#888888"
        for model in plot_data["model"]
    ]
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    axis.bar(
        plot_data["model"],
        plot_data["mean_log_loss"],
        yerr=plot_data["se_repeat_log_loss"],
        capsize=4,
        color=colors,
    )
    axis.set_ylabel("Repeated out-of-fold log loss")
    axis.set_title("trees Classification: Logistic Regression vs Tree Models")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        EXPERIMENT_DIR / "classifier_log_loss_comparison.png",
        dpi=180,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    curve_models = [
        "blend_raw_l2_rf",
        "raw_l2",
        "random_forest",
        "xgboost",
    ]
    for model_name in curve_models:
        group = predictions.loc[predictions["model"] == model_name]
        fraction_positive, mean_predicted = calibration_curve(
            group["actual"],
            group["probability"],
            n_bins=8,
            strategy="quantile",
        )
        axis.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=model_name,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="black")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed distressed fraction")
    axis.set_title("Repeated OOF Calibration")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        EXPERIMENT_DIR / "classifier_calibration_comparison.png",
        dpi=180,
    )
    plt.close(figure)


def write_readme(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    blend_weights: pd.DataFrame,
    selected_c: float,
    full_blend_weight: float,
) -> None:
    metrics = summary.set_index("model")
    logistic = metrics.loc["raw_l2"]
    spline = metrics.loc["spline_degree2_l2"]
    forest = metrics.loc["random_forest"]
    xgboost = metrics.loc["xgboost"]
    blend = metrics.loc["blend_raw_l2_rf"]
    logistic_vs_rf = paired.loc[
        paired["comparison_model"] == "random_forest"
    ].iloc[0]
    logistic_vs_xgb = paired.loc[
        paired["comparison_model"] == "xgboost"
    ].iloc[0]
    text = f"""# Logistic Regression vs RF and XGBoost

## Conclusion

Regularized logistic regression is competitive with the selected random
forest and better than the selected XGBoost classifier on repeated
out-of-fold log loss.

| Model | Log loss | Brier | ROC AUC |
|---|---:|---:|---:|
| Raw-feature L2 logistic | {logistic["mean_log_loss"]:.4f} | {logistic["mean_brier_score"]:.4f} | {logistic["mean_roc_auc"]:.4f} |
| Degree-2 spline logistic | {spline["mean_log_loss"]:.4f} | {spline["mean_brier_score"]:.4f} | {spline["mean_roc_auc"]:.4f} |
| Random forest | {forest["mean_log_loss"]:.4f} | {forest["mean_brier_score"]:.4f} | {forest["mean_roc_auc"]:.4f} |
| XGBoost | {xgboost["mean_log_loss"]:.4f} | {xgboost["mean_brier_score"]:.4f} | {xgboost["mean_roc_auc"]:.4f} |
| Logistic + RF blend | {blend["mean_log_loss"]:.4f} | {blend["mean_brier_score"]:.4f} | {blend["mean_roc_auc"]:.4f} |

Plain L2 logistic is recommended over spline logistic because its log-loss
difference is only {logistic["mean_log_loss"] - spline["mean_log_loss"]:.4f},
which is much smaller than repeat-to-repeat uncertainty.

Against RF, logistic won
{int(logistic_vs_rf["reference_wins_out_of_repeats"])} of
{int(logistic_vs_rf["n_repeats"])} repeats. The mean RF-minus-logistic
difference was
{logistic_vs_rf["mean_comparison_minus_reference_log_loss"]:.4f}, with a
repeat-bootstrap interval of
[{logistic_vs_rf["bootstrap_95_low"]:.4f},
{logistic_vs_rf["bootstrap_95_high"]:.4f}]. They should be treated as tied.

Against XGBoost, logistic won
{int(logistic_vs_xgb["reference_wins_out_of_repeats"])} of
{int(logistic_vs_xgb["n_repeats"])} repeats. The mean
XGBoost-minus-logistic difference was
{logistic_vs_xgb["mean_comparison_minus_reference_log_loss"]:.4f}.

## Blend

A logistic/RF blend was tuned by leaving out one complete CV repeat at a
time. Its mean held-repeat log loss was {blend["mean_log_loss"]:.4f}. The
selected logistic weights ranged from
{blend_weights["weight_on_first_model"].min():.2f} to
{blend_weights["weight_on_first_model"].max():.2f}.

For the exported test benchmark, the full OOF optimum places
{full_blend_weight:.2f} weight on logistic and
{1 - full_blend_weight:.2f} on RF.

## Final Logistic Model

The final raw-feature model uses standardized predictors, natural class
prevalence, L2 regularization, and `C={selected_c:g}`. Balanced class weights
were not used because they inflate probabilities and worsen log loss.

## Project Note

This experiment does not replace tasks 4 and 5. trees still requires RF and
XGBoost outputs and their MDI importances. Logistic regression and the blend
are diagnostic benchmarks outside the required JSON contract.
"""
    (EXPERIMENT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if list(train.columns[:12]) != FEATURES or list(test.columns) != FEATURES:
        raise AssertionError("Unexpected trees feature columns")
    x = train[FEATURES]
    y = train[TARGET].astype(int)

    fold_results, logistic_oof = nested_logistic_evaluation(x, y)
    fold_results.to_csv(
        EXPERIMENT_DIR / "logistic_nested_cv_fold_results.csv",
        index=False,
    )
    logistic_oof.to_csv(
        EXPERIMENT_DIR / "logistic_nested_oof_probabilities.csv",
        index=False,
    )

    rf_oof = validate_tree_predictions(
        pd.read_csv(RF_OOF_PATH),
        "random_forest",
        y,
    )
    xgb_oof = validate_tree_predictions(
        pd.read_csv(XGB_OOF_PATH),
        "xgboost",
        y,
    )

    baseline_rows = []
    balanced_rows = []
    for seed in SEEDS:
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
        for fold, (training, validation) in enumerate(
            splitter.split(x, y),
            start=1,
        ):
            prevalence = float(y.iloc[training].mean())
            balanced = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            solver="lbfgs",
                            max_iter=10000,
                        ),
                    ),
                ]
            ).fit(x.iloc[training], y.iloc[training])
            balanced_probabilities = balanced.predict_proba(
                x.iloc[validation]
            )[:, 1]
            for row_index, actual, balanced_probability in zip(
                validation,
                y.iloc[validation],
                balanced_probabilities,
            ):
                common = {
                    "cv_seed": seed,
                    "fold": fold,
                    "row_index": int(row_index),
                    "actual": int(actual),
                }
                baseline_rows.append(
                    {
                        **common,
                        "model": "constant_prevalence",
                        "probability": prevalence,
                    }
                )
                balanced_rows.append(
                    {
                        **common,
                        "model": "balanced_logistic",
                        "probability": float(balanced_probability),
                    }
                )

    all_predictions = pd.concat(
        [
            logistic_oof,
            rf_oof,
            xgb_oof,
            pd.DataFrame(baseline_rows),
            pd.DataFrame(balanced_rows),
        ],
        ignore_index=True,
    )

    blend_weights, blend_predictions, full_blend_weight = tune_blend_weights(
        all_predictions,
        "raw_l2",
        "random_forest",
        "blend_raw_l2_rf",
    )
    blend_weights.to_csv(
        EXPERIMENT_DIR / "logistic_rf_blend_weights.csv",
        index=False,
    )
    all_predictions = pd.concat(
        [all_predictions, blend_predictions],
        ignore_index=True,
    )
    all_predictions.to_csv(
        EXPERIMENT_DIR / "all_classifier_oof_probabilities.csv",
        index=False,
    )

    summary, repeat_metrics = metric_summary(all_predictions)
    summary.to_csv(
        EXPERIMENT_DIR / "classifier_comparison_summary.csv",
        index=False,
    )
    repeat_metrics.to_csv(
        EXPERIMENT_DIR / "classifier_repeat_metrics.csv",
        index=False,
    )

    paired = paired_comparisons(
        repeat_metrics,
        reference_model="raw_l2",
        comparison_models=[
            "spline_degree2_l2",
            "random_forest",
            "xgboost",
        ],
    )
    paired.to_csv(
        EXPERIMENT_DIR / "paired_logistic_comparisons.csv",
        index=False,
    )

    c_tuning, selected_c = repeated_c_tuning(x, y)
    c_tuning.to_csv(
        EXPERIMENT_DIR / "final_logistic_c_tuning.csv",
        index=False,
    )
    final_logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=selected_c,
                    solver="lbfgs",
                    max_iter=10000,
                ),
            ),
        ]
    ).fit(x, y)
    logistic_test = clip_probabilities(
        final_logistic.predict_proba(test[FEATURES])[:, 1]
    )

    rf_test = pd.read_csv(RF_TEST_PATH).sort_values("row_index")
    xgb_test = pd.read_csv(XGB_TEST_PATH).sort_values("row_index")
    if len(rf_test) != len(test) or len(xgb_test) != len(test):
        raise AssertionError("Unexpected tree test prediction count")
    blended_test = (
        full_blend_weight * logistic_test
        + (1 - full_blend_weight)
        * rf_test["rf_clf_pred_prob"].to_numpy(dtype=float)
    )
    pd.DataFrame(
        {
            "row_index": np.arange(len(test)),
            "logistic_pred_prob": logistic_test,
            "rf_pred_prob": rf_test["rf_clf_pred_prob"].to_numpy(dtype=float),
            "xgb_pred_prob": xgb_test["gbm_clf_pred_prob"].to_numpy(dtype=float),
            "logistic_rf_blend_pred_prob": blended_test,
        }
    ).to_csv(
        EXPERIMENT_DIR / "classifier_test_probability_benchmarks.csv",
        index=False,
    )

    scaler = final_logistic.named_steps["scale"]
    logistic = final_logistic.named_steps["logistic"]
    coefficient_table = pd.DataFrame(
        {
            "feature": FEATURES,
            "standardized_coefficient": logistic.coef_[0],
            "raw_scale_coefficient": logistic.coef_[0] / scaler.scale_,
        }
    )
    coefficient_table["absolute_standardized_coefficient"] = (
        coefficient_table["standardized_coefficient"].abs()
    )
    coefficient_table.sort_values(
        "absolute_standardized_coefficient",
        ascending=False,
    ).to_csv(
        EXPERIMENT_DIR / "final_logistic_coefficients.csv",
        index=False,
    )

    save_plots(summary, all_predictions)
    write_readme(
        summary,
        paired,
        blend_weights,
        selected_c,
        full_blend_weight,
    )

    results = {
        "recommended_logistic_model": "raw_l2",
        "recommended_reason": (
            "Simpler than spline logistic with a negligible log-loss "
            "difference relative to repeat variability"
        ),
        "selected_final_C": selected_c,
        "full_oof_logistic_weight_for_rf_blend": full_blend_weight,
        "summary": summary.to_dict(orient="records"),
        "paired_comparisons": paired.to_dict(orient="records"),
        "blend_repeat_weights": blend_weights.to_dict(orient="records"),
        "experiment_settings": {
            "outer_cv_seeds": list(SEEDS),
            "outer_folds": 5,
            "inner_folds": 5,
            "selection_metric": "log_loss",
            "class_weight": None,
        },
    }
    (EXPERIMENT_DIR / "logistic_vs_tree_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    display_models = [
        "blend_raw_l2_rf",
        "spline_degree2_l2",
        "raw_l2",
        "random_forest",
        "xgboost",
        "constant_prevalence",
        "balanced_logistic",
    ]
    print(
        summary.set_index("model")
        .loc[display_models]
        .reset_index()[
            [
                "model",
                "mean_log_loss",
                "mean_brier_score",
                "mean_roc_auc",
            ]
        ]
        .to_string(index=False)
    )
    print(f"\nSelected final C: {selected_c:g}")
    print(f"Final logistic blend weight: {full_blend_weight:.2f}")


if __name__ == "__main__":
    main()
