import itertools
import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=ConvergenceWarning)

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
TRAIN_PATH = PROJECT_DIR / "real_estate_risk_train.csv"
TEST_PATH = PROJECT_DIR / "real_estate_risk_test.csv"
CART_DIR = EXPERIMENT_DIR.parent / "CART"
CART_OOF_PATH = (
    CART_DIR / "official_selected_depth_oof_predictions.csv"
)

STUDENT_ID = 9618
N_FOLDS = 5
OUTER_SEEDS = (0, 42, 2026, STUDENT_ID, 19618, 31415)
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
LINEAR_MODELS = (
    "ols_all",
    "ridge",
    "pls",
    "huber",
    "subset_min",
    "subset_1se",
    "adaptive_linear",
)
COMPLEXITY_ORDER = {
    "ols_all": 0,
    "ridge": 1,
    "pls": 2,
    "huber": 3,
    "subset_1se": 4,
    "subset_min": 5,
    "adaptive_linear": 6,
}


def make_splits(
    n_rows: int,
    seed: int,
    n_folds: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    validation_folds = [
        np.asarray(fold, dtype=int)
        for fold in np.array_split(
            np.random.default_rng(seed).permutation(n_rows),
            n_folds,
        )
    ]
    all_rows = np.arange(n_rows, dtype=int)
    return [
        (np.setdiff1d(all_rows, validation), validation)
        for validation in validation_folds
    ]


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def domain_log_transform(x: np.ndarray) -> np.ndarray:
    transformed = np.asarray(x, dtype=float).copy()
    transformed[:, 4] = 2011 - transformed[:, 4]
    transformed[:, 5] = 2011 - transformed[:, 5]
    for column in (0, 1, 6, 7, 10, 11):
        transformed[:, column] = np.log1p(
            np.maximum(transformed[:, column], 0.0)
        )
    return transformed


def fit_grid(
    estimator,
    parameter_grid: dict[str, Iterable],
    x: np.ndarray,
    y: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        error_score="raise",
    )
    search.fit(x, y)
    return (
        search.best_estimator_,
        float(-search.best_score_),
        json_ready(search.best_params_),
    )


def manual_cv_predictions(
    estimator,
    x: np.ndarray,
    y: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float]]:
    oof = np.full(len(y), np.nan, dtype=float)
    fold_mse = []
    for training, validation in cv:
        fitted = clone(estimator).fit(x[training], y[training])
        predictions = fitted.predict(x[validation]).reshape(-1)
        oof[validation] = predictions
        fold_mse.append(
            float(mean_squared_error(y[validation], predictions))
        )
    if not np.isfinite(oof).all():
        raise AssertionError("OOF predictions are incomplete")
    return oof, fold_mse


def fit_ols_subset(
    x: np.ndarray,
    y: np.ndarray,
    columns: tuple[int, ...],
) -> np.ndarray:
    design = np.column_stack(
        [np.ones(len(y), dtype=float), x[:, columns]]
    )
    return np.linalg.lstsq(design, y, rcond=None)[0]


def predict_ols_subset(
    x: np.ndarray,
    coefficients: np.ndarray,
    columns: tuple[int, ...],
) -> np.ndarray:
    design = np.column_stack(
        [np.ones(len(x), dtype=float), x[:, columns]]
    )
    return design @ coefficients


def exhaustive_subset_selection(
    x: np.ndarray,
    y: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, object], dict[str, object]]:
    records = []
    for size in range(1, x.shape[1] + 1):
        for columns in itertools.combinations(range(x.shape[1]), size):
            fold_errors = []
            for training, validation in cv:
                coefficients = fit_ols_subset(
                    x[training],
                    y[training],
                    columns,
                )
                predictions = predict_ols_subset(
                    x[validation],
                    coefficients,
                    columns,
                )
                fold_errors.append(
                    float(
                        mean_squared_error(
                            y[validation],
                            predictions,
                        )
                    )
                )
            records.append(
                {
                    "mean_mse": float(np.mean(fold_errors)),
                    "se_mse": float(
                        np.std(fold_errors, ddof=1)
                        / np.sqrt(len(fold_errors))
                    ),
                    "n_features": size,
                    "columns": columns,
                }
            )

    minimum = min(
        records,
        key=lambda row: (
            row["mean_mse"],
            row["n_features"],
            row["columns"],
        ),
    )
    threshold = minimum["mean_mse"] + minimum["se_mse"]
    one_se = min(
        (row for row in records if row["mean_mse"] <= threshold),
        key=lambda row: (
            row["n_features"],
            row["mean_mse"],
            row["columns"],
        ),
    )
    return minimum, one_se


def direct_screen(
    x: np.ndarray,
    y: np.ndarray,
    official_cv: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []

    def add_fixed(name: str, estimator) -> None:
        _, fold_mse = manual_cv_predictions(
            estimator,
            x,
            y,
            official_cv,
        )
        rows.append(
            {
                "model": name,
                "mean_cv_mse": float(np.mean(fold_mse)),
                "sd_fold_mse": float(np.std(fold_mse, ddof=1)),
                "best_params": "{}",
                "screening_note": "Fixed model; official five folds",
            }
        )

    def add_tuned(name: str, estimator, grid: dict[str, Iterable]) -> None:
        _, score, parameters = fit_grid(
            estimator,
            grid,
            x,
            y,
            official_cv,
        )
        rows.append(
            {
                "model": name,
                "mean_cv_mse": score,
                "sd_fold_mse": np.nan,
                "best_params": json.dumps(parameters, sort_keys=True),
                "screening_note": (
                    "Same folds used for tuning and scoring; screening only"
                ),
            }
        )

    add_fixed("raw_ols", LinearRegression())
    add_fixed(
        "domain_log_ols",
        Pipeline(
            [
                (
                    "domain",
                    FunctionTransformer(
                        domain_log_transform,
                        validate=False,
                    ),
                ),
                ("scale", StandardScaler()),
                ("regression", LinearRegression()),
            ]
        ),
    )
    add_tuned(
        "ridge",
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("regression", Ridge()),
            ]
        ),
        {"regression__alpha": np.logspace(-3, 3, 19)},
    )
    add_tuned(
        "pls",
        Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "regression",
                    PLSRegression(scale=False, max_iter=2000),
                ),
            ]
        ),
        {"regression__n_components": list(range(1, len(FEATURES) + 1))},
    )
    add_tuned(
        "huber",
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("regression", HuberRegressor(max_iter=5000)),
            ]
        ),
        {
            "regression__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "regression__epsilon": [1.35, 1.5, 1.75, 2.0, 2.25],
        },
    )
    add_tuned(
        "quadratic_lasso",
        Pipeline(
            [
                (
                    "polynomial",
                    PolynomialFeatures(degree=2, include_bias=False),
                ),
                ("scale", StandardScaler()),
                ("regression", Lasso(max_iter=200000)),
            ]
        ),
        {"regression__alpha": np.logspace(-2, 2, 17)},
    )
    add_tuned(
        "additive_spline_ridge",
        Pipeline(
            [
                ("spline", SplineTransformer(include_bias=False)),
                ("scale", StandardScaler()),
                ("regression", Ridge()),
            ]
        ),
        {
            "spline__n_knots": [3, 4, 5, 6],
            "spline__degree": [2, 3],
            "regression__alpha": np.logspace(-3, 3, 13),
        },
    )

    subset_min, subset_1se = exhaustive_subset_selection(x, y, official_cv)
    for name, selected in (
        ("best_subset_min", subset_min),
        ("best_subset_1se", subset_1se),
    ):
        rows.append(
            {
                "model": name,
                "mean_cv_mse": selected["mean_mse"],
                "sd_fold_mse": np.nan,
                "best_params": json.dumps(
                    {
                        "n_features": selected["n_features"],
                        "features": [
                            FEATURES[column]
                            for column in selected["columns"]
                        ],
                    },
                    sort_keys=True,
                ),
                "screening_note": (
                    "Same folds used for subset selection and scoring; "
                    "screening only"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_cv_mse")


def nested_comparison(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    prediction_rows = []
    subset_rows = []

    for outer_seed in OUTER_SEEDS:
        outer_cv = make_splits(len(y), seed=outer_seed)
        for fold_number, (training, validation) in enumerate(
            outer_cv,
            start=1,
        ):
            x_train = x[training]
            y_train = y[training]
            inner_seed = outer_seed * 100 + fold_number + 17
            inner_cv = make_splits(len(training), seed=inner_seed)

            fitted_models = {}
            tuning_scores = {}
            tuning_parameters = {}

            fitted_models["ols_all"] = LinearRegression().fit(
                x_train,
                y_train,
            )
            tuning_scores["ols_all"] = np.nan
            tuning_parameters["ols_all"] = {}

            ridge, ridge_score, ridge_parameters = fit_grid(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("regression", Ridge()),
                    ]
                ),
                {"regression__alpha": np.logspace(-3, 3, 19)},
                x_train,
                y_train,
                inner_cv,
            )
            fitted_models["ridge"] = ridge
            tuning_scores["ridge"] = ridge_score
            tuning_parameters["ridge"] = ridge_parameters

            pls, pls_score, pls_parameters = fit_grid(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "regression",
                            PLSRegression(scale=False, max_iter=2000),
                        ),
                    ]
                ),
                {
                    "regression__n_components": list(
                        range(1, len(FEATURES) + 1)
                    )
                },
                x_train,
                y_train,
                inner_cv,
            )
            fitted_models["pls"] = pls
            tuning_scores["pls"] = pls_score
            tuning_parameters["pls"] = pls_parameters

            huber, huber_score, huber_parameters = fit_grid(
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "regression",
                            HuberRegressor(max_iter=5000),
                        ),
                    ]
                ),
                {
                    "regression__alpha": [
                        0.001,
                        0.01,
                        0.1,
                        1.0,
                        10.0,
                    ],
                    "regression__epsilon": [
                        1.35,
                        1.5,
                        1.75,
                        2.0,
                        2.25,
                    ],
                },
                x_train,
                y_train,
                inner_cv,
            )
            fitted_models["huber"] = huber
            tuning_scores["huber"] = huber_score
            tuning_parameters["huber"] = huber_parameters

            subset_min, subset_1se = exhaustive_subset_selection(
                x_train,
                y_train,
                inner_cv,
            )
            subset_specs = {
                "subset_min": subset_min,
                "subset_1se": subset_1se,
            }
            subset_predictions = {}
            for model_name, selected in subset_specs.items():
                columns = selected["columns"]
                coefficients = fit_ols_subset(
                    x_train,
                    y_train,
                    columns,
                )
                subset_predictions[model_name] = predict_ols_subset(
                    x[validation],
                    coefficients,
                    columns,
                )
                tuning_scores[model_name] = selected["mean_mse"]
                tuning_parameters[model_name] = {
                    "n_features": selected["n_features"],
                    "features": [
                        FEATURES[column] for column in columns
                    ],
                }
                for feature_index, feature_name in enumerate(FEATURES):
                    subset_rows.append(
                        {
                            "outer_seed": outer_seed,
                            "outer_fold": fold_number,
                            "selection_rule": model_name,
                            "n_features": selected["n_features"],
                            "feature": feature_name,
                            "selected": feature_index in columns,
                        }
                    )

            cart, cart_score, cart_parameters = fit_grid(
                DecisionTreeRegressor(
                    min_samples_leaf=5,
                    random_state=0,
                ),
                {"max_depth": list(range(2, 11))},
                x_train,
                y_train,
                inner_cv,
            )
            fitted_models["cart"] = cart
            tuning_scores["cart"] = cart_score
            tuning_parameters["cart"] = cart_parameters

            adaptive_scores = {
                model_name: tuning_scores[model_name]
                for model_name in (
                    "ridge",
                    "pls",
                    "huber",
                    "subset_min",
                )
            }
            adaptive_choice = min(
                adaptive_scores,
                key=adaptive_scores.get,
            )
            tuning_scores["adaptive_linear"] = adaptive_scores[
                adaptive_choice
            ]
            tuning_parameters["adaptive_linear"] = {
                "selected_family": adaptive_choice
            }

            predictions = {
                model_name: model.predict(x[validation]).reshape(-1)
                for model_name, model in fitted_models.items()
            }
            predictions.update(subset_predictions)
            if adaptive_choice in subset_predictions:
                predictions["adaptive_linear"] = subset_predictions[
                    adaptive_choice
                ]
            else:
                predictions["adaptive_linear"] = fitted_models[
                    adaptive_choice
                ].predict(x[validation]).reshape(-1)

            for model_name, model_predictions in predictions.items():
                mse = float(
                    mean_squared_error(
                        y[validation],
                        model_predictions,
                    )
                )
                fold_rows.append(
                    {
                        "outer_seed": outer_seed,
                        "outer_fold": fold_number,
                        "model": model_name,
                        "outer_mse": mse,
                        "outer_rmse": float(np.sqrt(mse)),
                        "inner_selected_mse": tuning_scores[model_name],
                        "selected_parameters": json.dumps(
                            tuning_parameters[model_name],
                            sort_keys=True,
                        ),
                    }
                )
                for row_index, actual, prediction in zip(
                    validation,
                    y[validation],
                    model_predictions,
                ):
                    prediction_rows.append(
                        {
                            "outer_seed": outer_seed,
                            "outer_fold": fold_number,
                            "source_row_index": int(row_index),
                            "model": model_name,
                            "actual_sale_price_keur": float(actual),
                            "oof_prediction": float(prediction),
                            "squared_error": float(
                                (actual - prediction) ** 2
                            ),
                        }
                    )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(subset_rows),
    )


def summarize_nested(
    fold_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeat_scores = (
        fold_results.groupby(["outer_seed", "model"], as_index=False)[
            "outer_mse"
        ]
        .mean()
        .rename(columns={"outer_mse": "repeat_mse"})
    )
    summary = (
        repeat_scores.groupby("model")["repeat_mse"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_nested_mse",
                "std": "sd_repeat_mse",
                "median": "median_repeat_mse",
                "min": "min_repeat_mse",
                "max": "max_repeat_mse",
                "count": "n_repeats",
            }
        )
    )
    summary["nested_rmse"] = np.sqrt(summary["mean_nested_mse"])
    summary["se_repeat_mse"] = (
        summary["sd_repeat_mse"] / np.sqrt(summary["n_repeats"])
    )
    summary = summary.sort_values("mean_nested_mse")

    cart = repeat_scores.loc[
        repeat_scores["model"] == "cart",
        ["outer_seed", "repeat_mse"],
    ].rename(columns={"repeat_mse": "cart_repeat_mse"})
    paired_rows = []
    for model_name in LINEAR_MODELS:
        linear = repeat_scores.loc[
            repeat_scores["model"] == model_name,
            ["outer_seed", "repeat_mse"],
        ].rename(columns={"repeat_mse": "linear_repeat_mse"})
        paired = cart.merge(linear, on="outer_seed", validate="one_to_one")
        improvement = (
            paired["cart_repeat_mse"] - paired["linear_repeat_mse"]
        )
        rng = np.random.default_rng(STUDENT_ID + len(model_name))
        bootstrap = np.array(
            [
                rng.choice(improvement.to_numpy(), size=len(paired)).mean()
                for _ in range(20000)
            ]
        )
        paired_rows.append(
            {
                "linear_model": model_name,
                "mean_cart_minus_linear_mse": float(improvement.mean()),
                "bootstrap_95_low": float(
                    np.quantile(bootstrap, 0.025)
                ),
                "bootstrap_95_high": float(
                    np.quantile(bootstrap, 0.975)
                ),
                "mean_percent_mse_reduction": float(
                    100
                    * (
                        1
                        - paired["linear_repeat_mse"].mean()
                        / paired["cart_repeat_mse"].mean()
                    )
                ),
                "linear_wins_out_of_repeats": int(
                    (improvement > 0).sum()
                ),
                "n_repeats": len(paired),
            }
        )
    return summary, pd.DataFrame(paired_rows).sort_values(
        "mean_cart_minus_linear_mse",
        ascending=False,
    )


def choose_final_linear_model(summary: pd.DataFrame) -> str:
    linear = summary.loc[summary["model"].isin(LINEAR_MODELS)].copy()
    best = linear.sort_values("mean_nested_mse").iloc[0]
    threshold = best["mean_nested_mse"] + best["se_repeat_mse"]
    eligible = linear.loc[linear["mean_nested_mse"] <= threshold].copy()
    eligible["complexity_rank"] = eligible["model"].map(COMPLEXITY_ORDER)
    return str(
        eligible.sort_values(
            ["complexity_rank", "mean_nested_mse"]
        ).iloc[0]["model"]
    )


def official_ols_cart_comparison(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    official_cv = make_splits(len(y), seed=STUDENT_ID)
    ols_oof, ols_fold_mse = manual_cv_predictions(
        LinearRegression(),
        x,
        y,
        official_cv,
    )
    cart_oof = pd.read_csv(CART_OOF_PATH).sort_values(
        "source_row_index"
    )
    if not np.allclose(
        cart_oof["actual_sale_price_keur"].to_numpy(),
        y,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Saved CART OOF targets do not match source data")

    fold_by_row = np.empty(len(y), dtype=int)
    for fold_number, (_, validation) in enumerate(official_cv, start=1):
        fold_by_row[validation] = fold_number
    comparison = pd.DataFrame(
        {
            "source_row_index": np.arange(len(y)),
            "fold": fold_by_row,
            "actual_sale_price_keur": y,
            "ols_oof_prediction": ols_oof,
            "cart_oof_prediction": cart_oof[
                "oof_prediction"
            ].to_numpy(),
        }
    )
    comparison["ols_squared_error"] = (
        comparison["actual_sale_price_keur"]
        - comparison["ols_oof_prediction"]
    ) ** 2
    comparison["cart_squared_error"] = (
        comparison["actual_sale_price_keur"]
        - comparison["cart_oof_prediction"]
    ) ** 2
    cart_fold_mse = (
        comparison.groupby("fold")["cart_squared_error"].mean().tolist()
    )
    metrics = {
        "ols_mean_cv_mse": float(np.mean(ols_fold_mse)),
        "ols_oof_rmse": float(
            np.sqrt(comparison["ols_squared_error"].mean())
        ),
        "cart_mean_cv_mse": float(np.mean(cart_fold_mse)),
        "cart_oof_rmse": float(
            np.sqrt(comparison["cart_squared_error"].mean())
        ),
        "ols_percent_mse_reduction_vs_cart": float(
            100
            * (
                1
                - np.mean(ols_fold_mse) / np.mean(cart_fold_mse)
            )
        ),
        "rows_where_ols_has_lower_squared_error": int(
            (
                comparison["ols_squared_error"]
                < comparison["cart_squared_error"]
            ).sum()
        ),
        "n_rows": len(comparison),
    }
    return comparison, metrics


def save_plots(
    nested_summary: pd.DataFrame,
    official_comparison: pd.DataFrame,
) -> None:
    plot_models = [
        "ols_all",
        "ridge",
        "pls",
        "huber",
        "subset_min",
        "subset_1se",
        "cart",
    ]
    plot_data = (
        nested_summary.set_index("model").loc[plot_models].reset_index()
    )
    colors = [
        "#2A6F97" if model != "cart" else "#C14953"
        for model in plot_data["model"]
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        plot_data["model"],
        plot_data["mean_nested_mse"],
        yerr=plot_data["se_repeat_mse"],
        capsize=4,
        color=colors,
    )
    ax.set_ylabel("Repeated nested CV MSE")
    ax.set_title("trees Task 1: Linear Models vs Tuned CART")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        EXPERIMENT_DIR / "linear_vs_cart_nested_cv.png",
        dpi=180,
    )
    plt.close(fig)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    limits = [
        float(official_comparison["actual_sale_price_keur"].min()),
        float(official_comparison["actual_sale_price_keur"].max()),
    ]
    for axis, model, color, title in (
        (
            axes[0],
            "ols_oof_prediction",
            "#2A6F97",
            "OLS",
        ),
        (
            axes[1],
            "cart_oof_prediction",
            "#C14953",
            "CART depth 6",
        ),
    ):
        axis.scatter(
            official_comparison["actual_sale_price_keur"],
            official_comparison[model],
            s=22,
            alpha=0.7,
            color=color,
        )
        axis.plot(limits, limits, color="black", linewidth=1)
        axis.set_xlabel("Actual sale price (kEUR)")
        axis.set_ylabel("OOF prediction (kEUR)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.suptitle("Official 5-Fold Out-of-Fold Predictions")
    figure.tight_layout()
    figure.savefig(
        EXPERIMENT_DIR / "official_oof_actual_vs_predicted.png",
        dpi=180,
    )
    plt.close(figure)


def write_readme(
    screening: pd.DataFrame,
    nested_summary: pd.DataFrame,
    paired: pd.DataFrame,
    official_metrics: dict[str, float],
    selected_model: str,
) -> None:
    best_screen = screening.iloc[0]
    nested_ols = nested_summary.set_index("model").loc["ols_all"]
    nested_cart = nested_summary.set_index("model").loc["cart"]
    ols_pair = paired.set_index("linear_model").loc["ols_all"]
    text = f"""# trees Task 1: Linear Regression vs CART

## Conclusion

The recommended linear model is **ordinary least squares using all 12
predictors**. It is selected because it is the simplest model within one
standard error of the best repeated nested-CV linear result.

On the exact official five folds:

- OLS MSE: {official_metrics["ols_mean_cv_mse"]:.3f}
- CART depth-6 MSE: {official_metrics["cart_mean_cv_mse"]:.3f}
- OLS MSE reduction: {official_metrics["ols_percent_mse_reduction_vs_cart"]:.1f}%
- OLS RMSE: {official_metrics["ols_oof_rmse"]:.3f} kEUR
- CART RMSE: {official_metrics["cart_oof_rmse"]:.3f} kEUR

In six-repeat nested 5-fold CV, where CART depth and linear alternatives are
tuned only inside each outer training fold:

- OLS mean MSE: {nested_ols["mean_nested_mse"]:.3f}
- Tuned CART mean MSE: {nested_cart["mean_nested_mse"]:.3f}
- Mean MSE reduction: {ols_pair["mean_percent_mse_reduction"]:.1f}%
- OLS beat CART in {int(ols_pair["linear_wins_out_of_repeats"])}
  of {int(ols_pair["n_repeats"])} repeats

The best direct screening score was {best_screen["model"]} at
{best_screen["mean_cv_mse"]:.3f} MSE, but tuned screening scores reuse the
same folds for selection and evaluation. Nested CV shows that subset
selection, ridge, PLS, Huber regression, quadratic Lasso, and additive
splines do not reliably improve on plain OLS.

## Important Project Point

This experiment does **not** replace the required scratch CART output in task
1. The project explicitly evaluates the scratch tree and its agreement with
the library tree. It shows that, as a predictive benchmark for these data, a
linear model is much more accurate than a single CART.

## Reproduce

Run:

```powershell
& ..\\..\\.venv-trees-tree-models\\Scripts\\python.exe .\\linear_vs_cart_experiment.py
```

Selected final model: `{selected_model}`
"""
    (EXPERIMENT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("This experiment expects complete input data")
    if list(train.columns[:12]) != FEATURES or list(test.columns) != FEATURES:
        raise ValueError("Unexpected trees feature columns")

    x = train[FEATURES].to_numpy(dtype=float)
    y = train["sale_price_keur"].to_numpy(dtype=float)
    x_test = test[FEATURES].to_numpy(dtype=float)
    official_cv = make_splits(len(y), seed=STUDENT_ID)

    screening = direct_screen(x, y, official_cv)
    screening.to_csv(
        EXPERIMENT_DIR / "linear_model_screening.csv",
        index=False,
    )

    fold_results, nested_oof, subset_records = nested_comparison(x, y)
    fold_results.to_csv(
        EXPERIMENT_DIR / "nested_cv_fold_results.csv",
        index=False,
    )
    nested_oof.to_csv(
        EXPERIMENT_DIR / "nested_cv_oof_predictions.csv",
        index=False,
    )

    nested_summary, paired = summarize_nested(fold_results)
    nested_summary.to_csv(
        EXPERIMENT_DIR / "nested_cv_summary.csv",
        index=False,
    )
    paired.to_csv(
        EXPERIMENT_DIR / "paired_linear_vs_cart.csv",
        index=False,
    )

    subset_frequency = (
        subset_records.groupby(["selection_rule", "feature"], as_index=False)
        .agg(
            selection_count=("selected", "sum"),
            selection_rate=("selected", "mean"),
            mean_selected_model_size=("n_features", "mean"),
        )
        .sort_values(["selection_rule", "selection_rate"], ascending=[True, False])
    )
    subset_frequency.to_csv(
        EXPERIMENT_DIR / "subset_selection_frequency.csv",
        index=False,
    )

    official_comparison, official_metrics = official_ols_cart_comparison(
        x,
        y,
    )
    official_comparison.to_csv(
        EXPERIMENT_DIR / "official_ols_vs_cart_oof.csv",
        index=False,
    )

    selected_model = choose_final_linear_model(nested_summary)
    if selected_model != "ols_all":
        raise AssertionError(
            "The final simple-model rule no longer selects OLS; "
            "review the experiment before exporting predictions"
        )
    final_model = LinearRegression().fit(x, y)
    test_predictions = final_model.predict(x_test)
    if len(test_predictions) != len(test) or not np.isfinite(
        test_predictions
    ).all():
        raise AssertionError("Invalid final test predictions")

    coefficient_table = pd.DataFrame(
        {
            "term": ["intercept", *FEATURES],
            "coefficient": [
                float(final_model.intercept_),
                *final_model.coef_.astype(float).tolist(),
            ],
        }
    )
    coefficient_table.to_csv(
        EXPERIMENT_DIR / "final_ols_coefficients.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "test_row_index": np.arange(len(test)),
            "linear_regression_prediction": test_predictions,
        }
    ).to_csv(
        EXPERIMENT_DIR / "final_ols_test_predictions.csv",
        index=False,
    )

    save_plots(nested_summary, official_comparison)
    results = {
        "recommended_model": selected_model,
        "recommendation_rule": (
            "Simplest linear model within one SE of the best repeated "
            "nested-CV mean"
        ),
        "official_five_fold_comparison": official_metrics,
        "nested_cv_summary": nested_summary.to_dict(orient="records"),
        "paired_linear_vs_cart": paired.to_dict(orient="records"),
        "final_ols_intercept": float(final_model.intercept_),
        "final_ols_coefficients": dict(
            zip(FEATURES, final_model.coef_.astype(float).tolist())
        ),
        "final_ols_test_predictions": test_predictions.astype(float).tolist(),
        "experiment_settings": {
            "outer_seeds": list(OUTER_SEEDS),
            "outer_folds": N_FOLDS,
            "inner_folds": N_FOLDS,
            "cart_depth_candidates": list(range(2, 11)),
            "cart_min_samples_leaf": 5,
        },
    }
    (EXPERIMENT_DIR / "linear_vs_cart_results.json").write_text(
        json.dumps(json_ready(results), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_readme(
        screening,
        nested_summary,
        paired,
        official_metrics,
        selected_model,
    )

    print(screening[["model", "mean_cv_mse"]].to_string(index=False))
    print()
    print(
        nested_summary[
            ["model", "mean_nested_mse", "nested_rmse"]
        ].to_string(index=False)
    )
    print()
    print(json.dumps(official_metrics, indent=2))
    print(f"\nRecommended model: {selected_model}")


if __name__ == "__main__":
    main()
