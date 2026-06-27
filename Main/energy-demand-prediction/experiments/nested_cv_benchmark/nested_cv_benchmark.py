import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import ConvergenceWarning


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from energy_demand_code import (  # noqa: E402
    BOUNDARY_LEFT,
    BOUNDARY_RIGHT,
    RANDOM_SEED,
    brier_score,
    bspline_basis_1d,
    fit_logistic_model,
    fit_ols,
    load_data,
    log_loss_binary,
    make_clamped_knot_vector,
    make_folds,
    make_internal_knots,
    mse,
)


N_OUTER_REPEATS = 30
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
KNOT_COUNTS = tuple(range(0, 7))

OUTER_SCORES_PATH = HERE / "nested_outer_scores.csv"
FIXED_SCORES_PATH = HERE / "fixed_spline_candidate_scores.csv"
SELECTION_PATH = HERE / "nested_selection_records.csv"
SELECTION_FREQUENCY_PATH = HERE / "nested_selection_frequency.csv"
PERFORMANCE_PATH = HERE / "nested_performance_summary.csv"
FIXED_SUMMARY_PATH = HERE / "fixed_spline_performance_summary.csv"
PAIRED_PATH = HERE / "paired_model_comparisons.csv"
OOF_PATH = HERE / "repeated_oof_predictions.csv"
OOF_AGG_PATH = HERE / "aggregated_oof_predictions.csv"
REPORT_PATH = HERE / "nested_cv_report.txt"
PERFORMANCE_PLOT_PATH = HERE / "nested_performance_comparison.png"
SELECTION_PLOT_PATH = HERE / "nested_selection_frequency.png"
DIAGNOSTICS_PLOT_PATH = HERE / "nested_oof_diagnostics.png"
FINAL_RECOMMENDATION_PATH = HERE / "recommended_configuration.json"


@dataclass(frozen=True)
class SplineSpec:
    placement: str
    n_internal_knots: int

    @property
    def name(self) -> str:
        return f"spline_{self.placement}_m{self.n_internal_knots}"

    @property
    def complexity(self) -> int:
        return self.n_internal_knots + 6


SPLINE_SPECS = [
    SplineSpec("quantile", n_knots)
    for n_knots in KNOT_COUNTS
] + [
    SplineSpec("equal", n_knots)
    for n_knots in KNOT_COUNTS
    if n_knots > 0
]


def scaled_covariates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperature = (df["temperature"].to_numpy(dtype=float) - 20.0) / 20.0
    humidity = (df["humidity"].to_numpy(dtype=float) - 55.0) / 35.0
    weekend = df["weekend"].to_numpy(dtype=float)
    return temperature, humidity, weekend


def spline_knots(train: pd.DataFrame, spec: SplineSpec) -> np.ndarray:
    if spec.n_internal_knots == 0:
        return np.array([], dtype=float)
    if spec.placement == "quantile":
        return make_internal_knots(
            train["temperature"].to_numpy(dtype=float),
            spec.n_internal_knots,
        )
    return np.linspace(
        BOUNDARY_LEFT,
        BOUNDARY_RIGHT,
        spec.n_internal_knots + 2,
    )[1:-1]


def spline_design(df: pd.DataFrame, knots: np.ndarray) -> np.ndarray:
    bmat = bspline_basis_1d(
        df["temperature"].to_numpy(dtype=float),
        make_clamped_knot_vector(knots),
    )
    _, humidity, weekend = scaled_covariates(df)
    return np.column_stack([bmat, humidity, weekend])


def baseline_design(df: pd.DataFrame, model_name: str) -> np.ndarray:
    temperature, humidity, weekend = scaled_covariates(df)
    if model_name == "linear":
        return np.column_stack(
            [np.ones(len(df)), temperature, humidity, weekend]
        )
    if model_name == "cubic_polynomial":
        return np.column_stack(
            [
                np.ones(len(df)),
                temperature,
                temperature**2,
                temperature**3,
                humidity,
                weekend,
            ]
        )
    raise ValueError(f"Unknown baseline model: {model_name}")


def fit_logistic(x: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return fit_logistic_model(x, y)


def fit_spline_model(train: pd.DataFrame, task: str, spec: SplineSpec):
    knots = spline_knots(train, spec)
    x_train = spline_design(train, knots)
    if task == "regression":
        fitted = fit_ols(
            x_train,
            train["consumption_kwh"].to_numpy(dtype=float),
        )
    else:
        fitted = fit_logistic(
            x_train,
            train["high_demand_alert"].to_numpy(dtype=int),
        )
    return knots, fitted


def predict_spline_model(
    df: pd.DataFrame,
    task: str,
    knots: np.ndarray,
    fitted,
) -> np.ndarray:
    x = spline_design(df, knots)
    if task == "regression":
        return x @ fitted
    return fitted.predict_proba(x)[:, 1]


def fit_baseline(train: pd.DataFrame, task: str, model_name: str):
    if model_name == "null":
        if task == "regression":
            return float(train["consumption_kwh"].mean())
        return float(train["high_demand_alert"].mean())

    x_train = baseline_design(train, model_name)
    if task == "regression":
        return fit_ols(
            x_train,
            train["consumption_kwh"].to_numpy(dtype=float),
        )
    return fit_logistic(
        x_train,
        train["high_demand_alert"].to_numpy(dtype=int),
    )


def predict_baseline(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    fitted,
) -> np.ndarray:
    if model_name == "null":
        return np.full(len(df), fitted, dtype=float)
    x = baseline_design(df, model_name)
    if task == "regression":
        return x @ fitted
    return fitted.predict_proba(x)[:, 1]


def score_predictions(
    task: str,
    valid: pd.DataFrame,
    prediction: np.ndarray,
) -> tuple[float, float | None]:
    if task == "regression":
        truth = valid["consumption_kwh"].to_numpy(dtype=float)
        return mse(truth, prediction), float(np.mean(np.abs(truth - prediction)))
    truth = valid["high_demand_alert"].to_numpy(dtype=int)
    return (
        log_loss_binary(truth, prediction),
        brier_score(truth, prediction),
    )


def inner_evaluate(
    outer_train: pd.DataFrame,
    task: str,
    repeat: int,
    outer_fold: int,
) -> pd.DataFrame:
    seed = RANDOM_SEED + 10000 * repeat + 101 * outer_fold
    inner_folds = make_folds(
        len(outer_train),
        n_folds=N_INNER_FOLDS,
        seed=seed,
    )
    all_idx = np.arange(len(outer_train))
    rows = []

    for spec in SPLINE_SPECS:
        fold_scores = []
        for inner_fold, valid_idx in enumerate(inner_folds):
            train_idx = np.setdiff1d(all_idx, valid_idx)
            inner_train = outer_train.iloc[train_idx].reset_index(drop=True)
            inner_valid = outer_train.iloc[valid_idx].reset_index(drop=True)
            try:
                knots, fitted = fit_spline_model(inner_train, task, spec)
                pred = predict_spline_model(
                    inner_valid,
                    task,
                    knots,
                    fitted,
                )
                primary, _ = score_predictions(task, inner_valid, pred)
            except Exception:
                primary = np.inf
            fold_scores.append(primary)

        finite_scores = np.asarray(fold_scores, dtype=float)
        mean_score = float(np.mean(finite_scores))
        sd_score = float(np.std(finite_scores, ddof=1))
        rows.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "task": task,
                "candidate": spec.name,
                "placement": spec.placement,
                "n_internal_knots": spec.n_internal_knots,
                "complexity": spec.complexity,
                "inner_mean_score": mean_score,
                "inner_sd_score": sd_score,
                "inner_se_score": sd_score / np.sqrt(N_INNER_FOLDS),
            }
        )
    return pd.DataFrame(rows)


def select_specs(inner_results: pd.DataFrame) -> tuple[SplineSpec, SplineSpec]:
    best_row = inner_results.loc[inner_results["inner_mean_score"].idxmin()]
    best_spec = SplineSpec(
        str(best_row["placement"]),
        int(best_row["n_internal_knots"]),
    )
    threshold = (
        float(best_row["inner_mean_score"])
        + float(best_row["inner_se_score"])
    )
    eligible = inner_results[
        inner_results["inner_mean_score"] <= threshold
    ].copy()
    eligible["placement_priority"] = eligible["placement"].map(
        {"quantile": 0, "equal": 1}
    )
    one_se_row = eligible.sort_values(
        [
            "complexity",
            "n_internal_knots",
            "placement_priority",
            "inner_mean_score",
        ]
    ).iloc[0]
    one_se_spec = SplineSpec(
        str(one_se_row["placement"]),
        int(one_se_row["n_internal_knots"]),
    )
    return best_spec, one_se_spec


def current_spec(task: str) -> SplineSpec:
    return SplineSpec(
        "quantile",
        2 if task == "regression" else 0,
    )


def append_oof_rows(
    rows: list[dict],
    valid: pd.DataFrame,
    repeat: int,
    outer_fold: int,
    task: str,
    pipeline: str,
    prediction: np.ndarray,
) -> None:
    truth_col = (
        "consumption_kwh"
        if task == "regression"
        else "high_demand_alert"
    )
    for row_id, truth, pred in zip(
        valid["_row_id"].to_numpy(dtype=int),
        valid[truth_col].to_numpy(),
        prediction,
    ):
        rows.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "row_id": int(row_id),
                "task": task,
                "pipeline": pipeline,
                "truth": float(truth),
                "prediction": float(pred),
            }
        )


def run_nested_benchmark(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = train.copy()
    data["_row_id"] = np.arange(len(data))

    outer_score_rows = []
    fixed_score_rows = []
    selection_rows = []
    oof_rows = []

    for repeat in range(N_OUTER_REPEATS):
        outer_folds = make_folds(
            len(data),
            n_folds=N_OUTER_FOLDS,
            seed=RANDOM_SEED + repeat,
        )
        all_idx = np.arange(len(data))

        for outer_fold, valid_idx in enumerate(outer_folds):
            train_idx = np.setdiff1d(all_idx, valid_idx)
            outer_train = data.iloc[train_idx].reset_index(drop=True)
            outer_valid = data.iloc[valid_idx].reset_index(drop=True)

            for task in ["regression", "classification"]:
                inner_results = inner_evaluate(
                    outer_train,
                    task,
                    repeat,
                    outer_fold,
                )
                best_spec, one_se_spec = select_specs(inner_results)
                best_inner = inner_results.loc[
                    inner_results["candidate"] == best_spec.name
                ].iloc[0]
                one_se_inner = inner_results.loc[
                    inner_results["candidate"] == one_se_spec.name
                ].iloc[0]

                for rule, spec, row in [
                    ("minimum", best_spec, best_inner),
                    ("one_se", one_se_spec, one_se_inner),
                ]:
                    selection_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "task": task,
                            "selection_rule": rule,
                            "candidate": spec.name,
                            "placement": spec.placement,
                            "n_internal_knots": spec.n_internal_knots,
                            "inner_mean_score": row["inner_mean_score"],
                            "inner_se_score": row["inner_se_score"],
                        }
                    )

                pipeline_specs = {
                    "nested_minimum": best_spec,
                    "nested_one_se": one_se_spec,
                    "current_solution": current_spec(task),
                }
                for pipeline, spec in pipeline_specs.items():
                    knots, fitted = fit_spline_model(
                        outer_train,
                        task,
                        spec,
                    )
                    pred = predict_spline_model(
                        outer_valid,
                        task,
                        knots,
                        fitted,
                    )
                    primary, secondary = score_predictions(
                        task,
                        outer_valid,
                        pred,
                    )
                    outer_score_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "task": task,
                            "pipeline": pipeline,
                            "model": spec.name,
                            "primary_score": primary,
                            "secondary_score": secondary,
                        }
                    )
                    append_oof_rows(
                        oof_rows,
                        outer_valid,
                        repeat,
                        outer_fold,
                        task,
                        pipeline,
                        pred,
                    )

                for model_name in ["null", "linear", "cubic_polynomial"]:
                    fitted = fit_baseline(
                        outer_train,
                        task,
                        model_name,
                    )
                    pred = predict_baseline(
                        outer_valid,
                        task,
                        model_name,
                        fitted,
                    )
                    primary, secondary = score_predictions(
                        task,
                        outer_valid,
                        pred,
                    )
                    outer_score_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "task": task,
                            "pipeline": model_name,
                            "model": model_name,
                            "primary_score": primary,
                            "secondary_score": secondary,
                        }
                    )
                    append_oof_rows(
                        oof_rows,
                        outer_valid,
                        repeat,
                        outer_fold,
                        task,
                        model_name,
                        pred,
                    )

                for spec in SPLINE_SPECS:
                    knots, fitted = fit_spline_model(
                        outer_train,
                        task,
                        spec,
                    )
                    pred = predict_spline_model(
                        outer_valid,
                        task,
                        knots,
                        fitted,
                    )
                    primary, secondary = score_predictions(
                        task,
                        outer_valid,
                        pred,
                    )
                    fixed_score_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "task": task,
                            "candidate": spec.name,
                            "placement": spec.placement,
                            "n_internal_knots": spec.n_internal_knots,
                            "primary_score": primary,
                            "secondary_score": secondary,
                        }
                    )

    return (
        pd.DataFrame(outer_score_rows),
        pd.DataFrame(fixed_score_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(oof_rows),
    )


def summarize_outer_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["task", "pipeline"], as_index=False)
        .agg(
            mean_primary=("primary_score", "mean"),
            sd_primary=("primary_score", "std"),
            se_primary=(
                "primary_score",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)),
            ),
            median_primary=("primary_score", "median"),
            mean_secondary=("secondary_score", "mean"),
            sd_secondary=("secondary_score", "std"),
            n_outer_folds=("primary_score", "size"),
        )
        .sort_values(["task", "mean_primary"])
        .reset_index(drop=True)
    )


def summarize_fixed_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(
            [
                "task",
                "candidate",
                "placement",
                "n_internal_knots",
            ],
            as_index=False,
        )
        .agg(
            mean_primary=("primary_score", "mean"),
            sd_primary=("primary_score", "std"),
            se_primary=(
                "primary_score",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)),
            ),
            median_primary=("primary_score", "median"),
            mean_secondary=("secondary_score", "mean"),
            sd_secondary=("secondary_score", "std"),
        )
        .sort_values(["task", "mean_primary"])
        .reset_index(drop=True)
    )


def selection_frequency(selection: pd.DataFrame) -> pd.DataFrame:
    frequency = (
        selection.groupby(
            [
                "task",
                "selection_rule",
                "candidate",
                "placement",
                "n_internal_knots",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "times_selected"})
    )
    frequency["selection_frequency"] = (
        frequency["times_selected"]
        / (N_OUTER_REPEATS * N_OUTER_FOLDS)
    )
    return frequency.sort_values(
        ["task", "selection_rule", "selection_frequency"],
        ascending=[True, True, False],
    )


def paired_comparisons(scores: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    for task in ["regression", "classification"]:
        task_scores = scores[scores["task"] == task].pivot_table(
            index=["repeat", "outer_fold"],
            columns="pipeline",
            values="primary_score",
        )
        current = task_scores["current_solution"]
        for challenger in [
            "nested_minimum",
            "nested_one_se",
            "linear",
            "cubic_polynomial",
            "null",
        ]:
            diff = task_scores[challenger] - current
            mean_diff = float(diff.mean())
            se_diff = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            critical = float(
                stats.t.ppf(0.975, df=len(diff) - 1)
            )
            comparisons.append(
                {
                    "task": task,
                    "challenger": challenger,
                    "reference": "current_solution",
                    "mean_score_difference": mean_diff,
                    "ci95_low": mean_diff - critical * se_diff,
                    "ci95_high": mean_diff + critical * se_diff,
                    "paired_t_pvalue": float(
                        stats.ttest_rel(
                            task_scores[challenger],
                            current,
                        ).pvalue
                    ),
                    "fraction_challenger_better": float(
                        np.mean(diff < 0)
                    ),
                }
            )
    return pd.DataFrame(comparisons)


def aggregate_oof(oof: pd.DataFrame) -> pd.DataFrame:
    return (
        oof.groupby(
            ["row_id", "task", "pipeline", "truth"],
            as_index=False,
        )
        .agg(
            mean_prediction=("prediction", "mean"),
            sd_prediction=("prediction", "std"),
        )
    )


def make_performance_plot(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    pipelines = [
        "null",
        "linear",
        "cubic_polynomial",
        "current_solution",
        "nested_minimum",
        "nested_one_se",
    ]
    labels = {
        "null": "Null",
        "linear": "Linear",
        "cubic_polynomial": "Cubic poly",
        "current_solution": "Current",
        "nested_minimum": "Nested min",
        "nested_one_se": "Nested 1-SE",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, task, ylabel in [
        (axes[0], "regression", "outer-fold MSE"),
        (axes[1], "classification", "outer-fold log loss"),
    ]:
        data = scores[scores["task"] == task]
        values = [
            data.loc[data["pipeline"] == pipeline, "primary_score"]
            for pipeline in pipelines
        ]
        box = ax.boxplot(
            values,
            labels=[labels[p] for p in pipelines],
            patch_artist=True,
            showfliers=False,
        )
        colors = [
            "#999999",
            "#4c78a8",
            "#f58518",
            "#54a24b",
            "#e45756",
            "#b279a2",
        ]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
        means = summary[summary["task"] == task].set_index(
            "pipeline"
        )["mean_primary"]
        ax.scatter(
            np.arange(1, len(pipelines) + 1),
            [means[p] for p in pipelines],
            color="black",
            s=25,
            zorder=4,
            label="mean",
        )
        ax.set_title(
            "Regression performance"
            if task == "regression"
            else "Classification performance",
            fontweight="bold",
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "30 x Repeated Nested 5-Fold Validation",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        PERFORMANCE_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_selection_plot(frequency: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("regression", "minimum"),
        ("regression", "one_se"),
        ("classification", "minimum"),
        ("classification", "one_se"),
    ]
    for ax, (task, rule) in zip(axes.ravel(), panels):
        subset = frequency[
            (frequency["task"] == task)
            & (frequency["selection_rule"] == rule)
        ].sort_values(
            ["n_internal_knots", "placement"]
        )
        colors = subset["placement"].map(
            {"quantile": "#2f6f9f", "equal": "#bf8f2f"}
        )
        ax.bar(
            subset["candidate"],
            subset["selection_frequency"],
            color=colors,
            alpha=0.82,
        )
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{task.capitalize()} - {rule.replace('_', ' ')}",
            fontweight="bold",
        )
        ax.set_ylabel("selection frequency")
        ax.tick_params(axis="x", rotation=55, labelsize=8)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Nested-CV Spline Selection Stability",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        SELECTION_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def calibration_points(
    truth: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 8,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"truth": truth, "probability": probability}
    )
    frame["bin"] = pd.qcut(
        frame["probability"],
        q=n_bins,
        duplicates="drop",
    )
    return (
        frame.groupby("bin", observed=True)
        .agg(
            mean_probability=("probability", "mean"),
            observed_rate=("truth", "mean"),
            n=("truth", "size"),
        )
        .reset_index()
    )


def make_diagnostics_plot(aggregated: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    reg = aggregated[
        (aggregated["task"] == "regression")
        & (aggregated["pipeline"] == "current_solution")
    ]
    residual = reg["truth"] - reg["mean_prediction"]
    axes[0, 0].scatter(
        reg["mean_prediction"],
        residual,
        s=22,
        alpha=0.62,
        color="#2f6f9f",
    )
    axes[0, 0].axhline(0, color="#c94f37", linewidth=1.5)
    axes[0, 0].set_title(
        "Current regression: OOF residuals",
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("mean repeated OOF prediction")
    axes[0, 0].set_ylabel("observed - predicted")
    axes[0, 0].grid(alpha=0.22)

    axes[0, 1].scatter(
        reg["truth"],
        reg["mean_prediction"],
        s=22,
        alpha=0.62,
        color="#3a8f60",
    )
    low = min(reg["truth"].min(), reg["mean_prediction"].min())
    high = max(reg["truth"].max(), reg["mean_prediction"].max())
    axes[0, 1].plot(
        [low, high],
        [low, high],
        color="#c94f37",
        linestyle="--",
    )
    axes[0, 1].set_title(
        "Current regression: predicted vs actual",
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("observed consumption")
    axes[0, 1].set_ylabel("mean repeated OOF prediction")
    axes[0, 1].grid(alpha=0.22)

    for pipeline, color, label in [
        ("current_solution", "#7257a8", "Current"),
        ("nested_minimum", "#c94f37", "Nested minimum"),
    ]:
        cls = aggregated[
            (aggregated["task"] == "classification")
            & (aggregated["pipeline"] == pipeline)
        ]
        calibration = calibration_points(
            cls["truth"].to_numpy(dtype=float),
            cls["mean_prediction"].to_numpy(dtype=float),
        )
        axes[1, 0].plot(
            calibration["mean_probability"],
            calibration["observed_rate"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axes[1, 0].plot(
        [0, 1],
        [0, 1],
        color="#555555",
        linestyle="--",
        label="perfect",
    )
    axes[1, 0].set_title(
        "Repeated OOF calibration",
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("mean predicted probability")
    axes[1, 0].set_ylabel("observed alert rate")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.22)

    cls = aggregated[
        (aggregated["task"] == "classification")
        & (aggregated["pipeline"] == "current_solution")
    ]
    axes[1, 1].hist(
        cls.loc[cls["truth"] == 0, "mean_prediction"],
        bins=16,
        alpha=0.62,
        color="#2f6f9f",
        label="alert=0",
    )
    axes[1, 1].hist(
        cls.loc[cls["truth"] == 1, "mean_prediction"],
        bins=16,
        alpha=0.62,
        color="#c94f37",
        label="alert=1",
    )
    axes[1, 1].set_title(
        "Current classifier: OOF probabilities",
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("mean repeated OOF probability")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.22)

    fig.suptitle(
        "Out-of-Fold Diagnostics",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        DIAGNOSTICS_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_report(
    performance: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    frequency: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict:
    recommendations = {}
    lines = [
        "REPEATED NESTED CROSS-VALIDATION BENCHMARK",
        "",
        f"Design: {N_OUTER_REPEATS} repetitions x "
        f"{N_OUTER_FOLDS} outer folds, with "
        f"{N_INNER_FOLDS}-fold inner tuning.",
        "Spline candidates: cubic B-splines with 0-6 internal knots,",
        "using quantile or equally spaced knot locations.",
        "Baselines: null, linear, and global cubic polynomial.",
        "All reported scores are from real outer validation rows.",
        "",
    ]

    for task in ["regression", "classification"]:
        task_perf = performance[
            performance["task"] == task
        ].sort_values("mean_primary")
        current = task_perf[
            task_perf["pipeline"] == "current_solution"
        ].iloc[0]
        nested_min = task_perf[
            task_perf["pipeline"] == "nested_minimum"
        ].iloc[0]
        nested_one_se = task_perf[
            task_perf["pipeline"] == "nested_one_se"
        ].iloc[0]
        best_fixed = fixed_summary[
            fixed_summary["task"] == task
        ].sort_values("mean_primary").iloc[0]
        paired_task = paired[paired["task"] == task].set_index(
            "challenger"
        )

        lines.extend(
            [
                task.upper(),
                f"- current solution mean score: "
                f"{current['mean_primary']:.6f}",
                f"- nested minimum pipeline: "
                f"{nested_min['mean_primary']:.6f}",
                f"- nested one-SE pipeline: "
                f"{nested_one_se['mean_primary']:.6f}",
                f"- best fixed spline candidate: "
                f"{best_fixed['candidate']} "
                f"({best_fixed['mean_primary']:.6f})",
                f"- nested minimum minus current paired difference: "
                f"{paired_task.loc['nested_minimum', 'mean_score_difference']:.6f}",
                f"- 95% CI: ["
                f"{paired_task.loc['nested_minimum', 'ci95_low']:.6f}, "
                f"{paired_task.loc['nested_minimum', 'ci95_high']:.6f}]",
                "",
            ]
        )

        if (
            paired_task.loc[
                "nested_minimum",
                "ci95_high",
            ]
            < 0
        ):
            recommendation = "nested_minimum"
        elif (
            paired_task.loc[
                "nested_one_se",
                "ci95_high",
            ]
            < 0
        ):
            recommendation = "nested_one_se"
        else:
            recommendation = "current_solution"

        recommendations[task] = {
            "recommended_pipeline": recommendation,
            "best_fixed_candidate": str(best_fixed["candidate"]),
            "best_fixed_placement": str(best_fixed["placement"]),
            "best_fixed_n_internal_knots": int(
                best_fixed["n_internal_knots"]
            ),
            "current_mean_primary_score": float(
                current["mean_primary"]
            ),
            "nested_minimum_mean_primary_score": float(
                nested_min["mean_primary"]
            ),
            "nested_one_se_mean_primary_score": float(
                nested_one_se["mean_primary"]
            ),
        }

    lines.extend(
        [
            "SELECTION STABILITY",
            "The full selection-frequency table is stored in",
            SELECTION_FREQUENCY_PATH.name + ".",
            "",
            "Decision rule:",
            "Change the final model only if a nested pipeline has a",
            "paired outer-fold improvement whose 95% confidence interval",
            "is entirely below zero. Otherwise retain the current simpler",
            "solution because the observed difference is not convincing.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    with open(
        FINAL_RECOMMENDATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(recommendations, file, indent=2)
    return recommendations


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    train, _ = load_data()
    outer_scores, fixed_scores, selection, oof = (
        run_nested_benchmark(train)
    )

    outer_scores.to_csv(OUTER_SCORES_PATH, index=False)
    fixed_scores.to_csv(FIXED_SCORES_PATH, index=False)
    selection.to_csv(SELECTION_PATH, index=False)
    oof.to_csv(OOF_PATH, index=False)

    performance = summarize_outer_scores(outer_scores)
    fixed_summary = summarize_fixed_scores(fixed_scores)
    frequency = selection_frequency(selection)
    paired = paired_comparisons(outer_scores)
    aggregated = aggregate_oof(oof)

    performance.to_csv(PERFORMANCE_PATH, index=False)
    fixed_summary.to_csv(FIXED_SUMMARY_PATH, index=False)
    frequency.to_csv(SELECTION_FREQUENCY_PATH, index=False)
    paired.to_csv(PAIRED_PATH, index=False)
    aggregated.to_csv(OOF_AGG_PATH, index=False)

    make_performance_plot(outer_scores, performance)
    make_selection_plot(frequency)
    make_diagnostics_plot(aggregated)
    recommendations = write_report(
        performance,
        fixed_summary,
        frequency,
        paired,
    )

    print("Performance summary:")
    print(performance.to_string(index=False))
    print("\nBest fixed spline candidates:")
    print(
        fixed_summary.groupby("task", as_index=False)
        .first()
        .to_string(index=False)
    )
    print("\nPaired comparisons:")
    print(paired.to_string(index=False))
    print("\nRecommendations:")
    print(json.dumps(recommendations, indent=2))
    print(f"\nOutputs saved under: {HERE}")


if __name__ == "__main__":
    main()
