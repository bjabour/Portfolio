import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning


HERE = Path(__file__).resolve().parent
ASSIGNMENT_DIR = HERE.parents[1]
sys.path.insert(0, str(ASSIGNMENT_DIR))

from splines_code import (  # noqa: E402
    BOUNDARY_LEFT,
    BOUNDARY_RIGHT,
    RANDOM_SEED,
    brier_score,
    design_matrix,
    fit_logistic_model,
    fit_ols,
    load_data,
    log_loss_binary,
    make_folds,
    make_internal_knots,
    mse,
)


N_REPEATS = 10
N_FOLDS = 5
KNOT_COUNTS = tuple(range(0, 7))
AUGMENTATION_RATIOS = (0, 1, 3)
N_STABILITY_REFITS = 100

RAW_SCORES_PATH = HERE / "mc_real_validation_scores.csv"
SUMMARY_PATH = HERE / "mc_performance_summary.csv"
SELECTION_PATH = HERE / "mc_selection_frequency.csv"
STABILITY_PATH = HERE / "mc_stability_summary.csv"
REPORT_PATH = HERE / "mc_experiment_report.txt"
PLOT_PATH = HERE / "mc_stability_comparison.png"
CANDIDATE_PATH = HERE / "mc_candidate_test_predictions.json"


def fit_regression_fixed(
    df: pd.DataFrame,
    y: np.ndarray,
    knots: np.ndarray,
) -> np.ndarray:
    return fit_ols(design_matrix(df, knots), y)


def predict_regression_fixed(
    df: pd.DataFrame,
    knots: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    return design_matrix(df, knots) @ beta


def fit_logistic_fixed(
    df: pd.DataFrame,
    y: np.ndarray,
    knots: np.ndarray,
):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return fit_logistic_model(design_matrix(df, knots), y)


def predict_logistic_fixed(df: pd.DataFrame, knots: np.ndarray, model) -> np.ndarray:
    return model.predict_proba(design_matrix(df, knots))[:, 1]


def fit_pseudo_dgp(train: pd.DataFrame) -> dict:
    temperature = train["temperature"].to_numpy(dtype=float)

    reg_knots = make_internal_knots(temperature, 2)
    reg_y = train["consumption_kwh"].to_numpy(dtype=float)
    reg_beta = fit_regression_fixed(train, reg_y, reg_knots)
    reg_fitted = predict_regression_fixed(train, reg_knots, reg_beta)
    residuals = reg_y - reg_fitted
    residuals = residuals - residuals.mean()

    cls_knots = make_internal_knots(temperature, 0)
    cls_y = train["high_demand_alert"].to_numpy(dtype=int)
    cls_model = fit_logistic_fixed(train, cls_y, cls_knots)

    return {
        "reg_knots": reg_knots,
        "reg_beta": reg_beta,
        "reg_residuals": residuals,
        "cls_knots": cls_knots,
        "cls_model": cls_model,
    }


def generate_pseudo_data(
    train: pd.DataFrame,
    dgp: dict,
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled_idx = rng.integers(0, len(train), size=n_rows)
    pseudo = train.iloc[sampled_idx][["temperature", "humidity", "weekend"]].reset_index(drop=True)

    pseudo["temperature"] = np.clip(
        pseudo["temperature"].to_numpy(dtype=float) + rng.normal(0.0, 0.55, n_rows),
        BOUNDARY_LEFT,
        BOUNDARY_RIGHT,
    )
    pseudo["humidity"] = np.clip(
        pseudo["humidity"].to_numpy(dtype=float) + rng.normal(0.0, 1.5, n_rows),
        20.0,
        90.0,
    )

    reg_mean = predict_regression_fixed(pseudo, dgp["reg_knots"], dgp["reg_beta"])
    pseudo["consumption_kwh"] = reg_mean + rng.choice(
        dgp["reg_residuals"],
        size=n_rows,
        replace=True,
    )

    cls_prob = predict_logistic_fixed(pseudo, dgp["cls_knots"], dgp["cls_model"])
    pseudo["high_demand_alert"] = rng.binomial(1, cls_prob, size=n_rows)
    return pseudo


def candidate_knots(real_train: pd.DataFrame, n_internal_knots: int) -> np.ndarray:
    return make_internal_knots(
        real_train["temperature"].to_numpy(dtype=float),
        n_internal_knots,
    )


def evaluate_fold(
    real_train: pd.DataFrame,
    real_valid: pd.DataFrame,
    repeat: int,
    fold: int,
) -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED + 1000 * repeat + fold)
    dgp = fit_pseudo_dgp(real_train)
    max_pseudo = generate_pseudo_data(
        real_train,
        dgp,
        n_rows=len(real_train) * max(AUGMENTATION_RATIOS),
        rng=rng,
    )

    rows = []
    y_reg_valid = real_valid["consumption_kwh"].to_numpy(dtype=float)
    y_cls_valid = real_valid["high_demand_alert"].to_numpy(dtype=int)

    for n_knots in KNOT_COUNTS:
        knots = candidate_knots(real_train, n_knots)

        for ratio in AUGMENTATION_RATIOS:
            if ratio == 0:
                model_train = real_train
            else:
                n_pseudo = ratio * len(real_train)
                model_train = pd.concat(
                    [real_train, max_pseudo.iloc[:n_pseudo]],
                    ignore_index=True,
                )

            y_reg_train = model_train["consumption_kwh"].to_numpy(dtype=float)
            reg_beta = fit_regression_fixed(model_train, y_reg_train, knots)
            reg_pred = predict_regression_fixed(real_valid, knots, reg_beta)
            rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "task": "regression",
                    "augmentation_ratio": ratio,
                    "n_internal_knots": n_knots,
                    "metric": "mse",
                    "score": mse(y_reg_valid, reg_pred),
                }
            )

            y_cls_train = model_train["high_demand_alert"].to_numpy(dtype=int)
            try:
                cls_model = fit_logistic_fixed(model_train, y_cls_train, knots)
                cls_prob = predict_logistic_fixed(real_valid, knots, cls_model)
                logloss = log_loss_binary(y_cls_valid, cls_prob)
                brier = brier_score(y_cls_valid, cls_prob)
            except Exception:
                logloss = np.inf
                brier = np.inf

            rows.extend(
                [
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "task": "classification",
                        "augmentation_ratio": ratio,
                        "n_internal_knots": n_knots,
                        "metric": "log_loss",
                        "score": logloss,
                    },
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "task": "classification",
                        "augmentation_ratio": ratio,
                        "n_internal_knots": n_knots,
                        "metric": "brier",
                        "score": brier,
                    },
                ]
            )

    return rows


def run_repeated_validation(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repeat in range(N_REPEATS):
        folds = make_folds(
            len(train),
            n_folds=N_FOLDS,
            seed=RANDOM_SEED + repeat,
        )
        all_idx = np.arange(len(train))
        for fold_number, valid_idx in enumerate(folds):
            train_idx = np.setdiff1d(all_idx, valid_idx)
            real_train = train.iloc[train_idx].reset_index(drop=True)
            real_valid = train.iloc[valid_idx].reset_index(drop=True)
            rows.extend(evaluate_fold(real_train, real_valid, repeat, fold_number))
    return pd.DataFrame(rows)


def summarize_scores(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            ["task", "metric", "augmentation_ratio", "n_internal_knots"],
            as_index=False,
        )
        .agg(
            mean_score=("score", "mean"),
            sd_score=("score", "std"),
            median_score=("score", "median"),
            n_real_validation_folds=("score", "size"),
        )
        .sort_values(["task", "metric", "mean_score"])
        .reset_index(drop=True)
    )


def selection_frequency(raw: pd.DataFrame) -> pd.DataFrame:
    primary = raw[
        ((raw["task"] == "regression") & (raw["metric"] == "mse"))
        | ((raw["task"] == "classification") & (raw["metric"] == "log_loss"))
    ]
    per_repeat = (
        primary.groupby(
            ["repeat", "task", "augmentation_ratio", "n_internal_knots"],
            as_index=False,
        )["score"]
        .mean()
    )

    selected_rows = []
    for (repeat, task), group in per_repeat.groupby(["repeat", "task"]):
        best = group.loc[group["score"].idxmin()]
        selected_rows.append(
            {
                "repeat": repeat,
                "task": task,
                "augmentation_ratio": int(best["augmentation_ratio"]),
                "n_internal_knots": int(best["n_internal_knots"]),
            }
        )

    selected = pd.DataFrame(selected_rows)
    frequency = (
        selected.groupby(
            ["task", "augmentation_ratio", "n_internal_knots"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "times_selected"})
    )
    frequency["selection_frequency"] = frequency["times_selected"] / N_REPEATS
    return frequency.sort_values(
        ["task", "selection_frequency"],
        ascending=[True, False],
    )


def best_config(
    summary: pd.DataFrame,
    task: str,
    augmented: bool,
) -> pd.Series:
    metric = "mse" if task == "regression" else "log_loss"
    subset = summary[(summary["task"] == task) & (summary["metric"] == metric)]
    if augmented:
        subset = subset[subset["augmentation_ratio"] > 0]
    else:
        subset = subset[subset["augmentation_ratio"] == 0]
    return subset.loc[subset["mean_score"].idxmin()]


def fit_candidate(
    real_train: pd.DataFrame,
    task: str,
    n_knots: int,
    augmentation_ratio: int,
    rng: np.random.Generator,
):
    knots = candidate_knots(real_train, n_knots)
    model_train = real_train

    if augmentation_ratio > 0:
        dgp = fit_pseudo_dgp(real_train)
        pseudo = generate_pseudo_data(
            real_train,
            dgp,
            n_rows=augmentation_ratio * len(real_train),
            rng=rng,
        )
        model_train = pd.concat([real_train, pseudo], ignore_index=True)

    if task == "regression":
        beta = fit_regression_fixed(
            model_train,
            model_train["consumption_kwh"].to_numpy(dtype=float),
            knots,
        )
        return knots, beta

    model = fit_logistic_fixed(
        model_train,
        model_train["high_demand_alert"].to_numpy(dtype=int),
        knots,
    )
    return knots, model


def stability_curves(
    train: pd.DataFrame,
    task: str,
    config: pd.Series,
    humidity: float,
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED + (17 if task == "regression" else 29))
    grid = pd.DataFrame(
        {
            "temperature": np.linspace(BOUNDARY_LEFT, BOUNDARY_RIGHT, 181),
            "humidity": humidity,
            "weekend": 0,
        }
    )
    curves = []

    for _ in range(N_STABILITY_REFITS):
        idx = rng.integers(0, len(train), size=len(train))
        boot = train.iloc[idx].reset_index(drop=True)
        knots, fitted = fit_candidate(
            boot,
            task=task,
            n_knots=int(config["n_internal_knots"]),
            augmentation_ratio=int(config["augmentation_ratio"]),
            rng=rng,
        )
        if task == "regression":
            curves.append(predict_regression_fixed(grid, knots, fitted))
        else:
            curves.append(predict_logistic_fixed(grid, knots, fitted))

    return np.asarray(curves)


def stability_summary(
    train: pd.DataFrame,
    baseline_reg: pd.Series,
    augmented_reg: pd.Series,
    baseline_cls: pd.Series,
    augmented_cls: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    humidity = float(train["humidity"].median())
    temp_grid = np.linspace(BOUNDARY_LEFT, BOUNDARY_RIGHT, 181)
    configs = {
        "regression_baseline": ("regression", baseline_reg),
        "regression_augmented": ("regression", augmented_reg),
        "classification_baseline": ("classification", baseline_cls),
        "classification_augmented": ("classification", augmented_cls),
    }

    rows = []
    curve_data = {"temperature": temp_grid}
    for label, (task, config) in configs.items():
        curves = stability_curves(train, task, config, humidity)
        q05 = np.quantile(curves, 0.05, axis=0)
        q50 = np.quantile(curves, 0.50, axis=0)
        q95 = np.quantile(curves, 0.95, axis=0)
        width = q95 - q05
        rows.append(
            {
                "model": label,
                "task": task,
                "augmentation_ratio": int(config["augmentation_ratio"]),
                "n_internal_knots": int(config["n_internal_knots"]),
                "median_90pct_band_width": float(np.median(width)),
                "mean_90pct_band_width": float(np.mean(width)),
                "max_90pct_band_width": float(np.max(width)),
            }
        )
        curve_data[label] = {
            "q05": q05,
            "q50": q50,
            "q95": q95,
        }
    return pd.DataFrame(rows), curve_data


def make_stability_plot(curve_data: dict) -> None:
    temp = curve_data["temperature"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("regression_baseline", "Regression: no augmentation", "#2f6f9f"),
        ("regression_augmented", "Regression: pseudo-DGP augmentation", "#bf8f2f"),
        ("classification_baseline", "Classification: no augmentation", "#7257a8"),
        ("classification_augmented", "Classification: pseudo-DGP augmentation", "#3a8f60"),
    ]
    for ax, (key, title, color) in zip(axes.ravel(), panels):
        values = curve_data[key]
        ax.fill_between(temp, values["q05"], values["q95"], color=color, alpha=0.24)
        ax.plot(temp, values["q50"], color=color, linewidth=2.2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("temperature")
        ax.set_ylabel("prediction" if "regression" in key else "probability")
        if "classification" in key:
            ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.22)
    fig.suptitle("Monte Carlo / Bootstrap Curve Stability", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_augmented_candidate_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    augmented_reg: pd.Series,
    augmented_cls: pd.Series,
) -> dict:
    rng = np.random.default_rng(RANDOM_SEED + 909)
    reg_knots, reg_beta = fit_candidate(
        train,
        task="regression",
        n_knots=int(augmented_reg["n_internal_knots"]),
        augmentation_ratio=int(augmented_reg["augmentation_ratio"]),
        rng=rng,
    )
    cls_knots, cls_model = fit_candidate(
        train,
        task="classification",
        n_knots=int(augmented_cls["n_internal_knots"]),
        augmentation_ratio=int(augmented_cls["augmentation_ratio"]),
        rng=rng,
    )
    pred_y = predict_regression_fixed(test, reg_knots, reg_beta)
    pred_p = predict_logistic_fixed(test, cls_knots, cls_model)
    pred_c = (pred_p >= 0.5).astype(int)
    candidate = {
        "pred_consumption_kwh": pred_y.tolist(),
        "pred_high_demand_prob": pred_p.tolist(),
        "pred_high_demand_class": pred_c.tolist(),
    }
    with open(CANDIDATE_PATH, "w", encoding="utf-8") as file:
        json.dump(candidate, file, indent=2)
    return candidate


def write_report(
    baseline_reg: pd.Series,
    augmented_reg: pd.Series,
    baseline_cls: pd.Series,
    augmented_cls: pd.Series,
    stability: pd.DataFrame,
) -> None:
    reg_improvement = 100.0 * (
        baseline_reg["mean_score"] - augmented_reg["mean_score"]
    ) / baseline_reg["mean_score"]
    cls_improvement = 100.0 * (
        baseline_cls["mean_score"] - augmented_cls["mean_score"]
    ) / baseline_cls["mean_score"]

    reg_stability = stability[stability["task"] == "regression"].set_index("model")
    cls_stability = stability[stability["task"] == "classification"].set_index("model")
    reg_width_change = 100.0 * (
        reg_stability.loc["regression_baseline", "median_90pct_band_width"]
        - reg_stability.loc["regression_augmented", "median_90pct_band_width"]
    ) / reg_stability.loc["regression_baseline", "median_90pct_band_width"]
    cls_width_change = 100.0 * (
        cls_stability.loc["classification_baseline", "median_90pct_band_width"]
        - cls_stability.loc["classification_augmented", "median_90pct_band_width"]
    ) / cls_stability.loc["classification_baseline", "median_90pct_band_width"]

    lines = [
        "PSEUDO-DGP MONTE CARLO EXPERIMENT",
        "",
        "Validation design:",
        f"- {N_REPEATS} repetitions of manual {N_FOLDS}-fold CV",
        "- pseudo-DGP estimated only on each real training fold",
        "- synthetic rows generated from resampled/jittered fold covariates",
        "- regression outcomes use fitted mean plus residual-bootstrap noise",
        "- classification outcomes use Bernoulli draws from fitted probabilities",
        "- all performance scores are computed on untouched real validation rows",
        "",
        "Best unaugmented regression:",
        f"- knots: {int(baseline_reg['n_internal_knots'])}",
        f"- repeated-CV MSE: {baseline_reg['mean_score']:.6f}",
        "",
        "Best augmented regression:",
        f"- augmentation ratio: {int(augmented_reg['augmentation_ratio'])}x",
        f"- knots: {int(augmented_reg['n_internal_knots'])}",
        f"- repeated-CV MSE: {augmented_reg['mean_score']:.6f}",
        f"- relative MSE improvement: {reg_improvement:.3f}%",
        f"- median stability-band width reduction: {reg_width_change:.3f}%",
        "",
        "Best unaugmented classification:",
        f"- knots: {int(baseline_cls['n_internal_knots'])}",
        f"- repeated-CV log loss: {baseline_cls['mean_score']:.6f}",
        "",
        "Best augmented classification:",
        f"- augmentation ratio: {int(augmented_cls['augmentation_ratio'])}x",
        f"- knots: {int(augmented_cls['n_internal_knots'])}",
        f"- repeated-CV log loss: {augmented_cls['mean_score']:.6f}",
        f"- relative log-loss improvement: {cls_improvement:.3f}%",
        f"- median stability-band width reduction: {cls_width_change:.3f}%",
        "",
        "Interpretation:",
        "Synthetic data do not create new information. They can only regularize the fit",
        "toward the estimated pseudo-DGP. An augmented model should be preferred only",
        "when it improves performance on real held-out rows and also improves stability.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    train, test = load_data()

    raw = run_repeated_validation(train)
    raw.to_csv(RAW_SCORES_PATH, index=False)

    summary = summarize_scores(raw)
    summary.to_csv(SUMMARY_PATH, index=False)

    frequency = selection_frequency(raw)
    frequency.to_csv(SELECTION_PATH, index=False)

    baseline_reg = best_config(summary, "regression", augmented=False)
    augmented_reg = best_config(summary, "regression", augmented=True)
    baseline_cls = best_config(summary, "classification", augmented=False)
    augmented_cls = best_config(summary, "classification", augmented=True)

    stability, curve_data = stability_summary(
        train,
        baseline_reg,
        augmented_reg,
        baseline_cls,
        augmented_cls,
    )
    stability.to_csv(STABILITY_PATH, index=False)
    make_stability_plot(curve_data)
    create_augmented_candidate_predictions(
        train,
        test,
        augmented_reg,
        augmented_cls,
    )
    write_report(
        baseline_reg,
        augmented_reg,
        baseline_cls,
        augmented_cls,
        stability,
    )

    comparison = pd.DataFrame(
        [
            {
                "task": "regression",
                "version": "unaugmented",
                "augmentation_ratio": int(baseline_reg["augmentation_ratio"]),
                "n_internal_knots": int(baseline_reg["n_internal_knots"]),
                "metric": "MSE",
                "mean_real_validation_score": baseline_reg["mean_score"],
                "sd_real_validation_score": baseline_reg["sd_score"],
            },
            {
                "task": "regression",
                "version": "pseudo-DGP augmented",
                "augmentation_ratio": int(augmented_reg["augmentation_ratio"]),
                "n_internal_knots": int(augmented_reg["n_internal_knots"]),
                "metric": "MSE",
                "mean_real_validation_score": augmented_reg["mean_score"],
                "sd_real_validation_score": augmented_reg["sd_score"],
            },
            {
                "task": "classification",
                "version": "unaugmented",
                "augmentation_ratio": int(baseline_cls["augmentation_ratio"]),
                "n_internal_knots": int(baseline_cls["n_internal_knots"]),
                "metric": "log loss",
                "mean_real_validation_score": baseline_cls["mean_score"],
                "sd_real_validation_score": baseline_cls["sd_score"],
            },
            {
                "task": "classification",
                "version": "pseudo-DGP augmented",
                "augmentation_ratio": int(augmented_cls["augmentation_ratio"]),
                "n_internal_knots": int(augmented_cls["n_internal_knots"]),
                "metric": "log loss",
                "mean_real_validation_score": augmented_cls["mean_score"],
                "sd_real_validation_score": augmented_cls["sd_score"],
            },
        ]
    )
    comparison.to_csv(HERE / "mc_best_model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print("\nStability:")
    print(stability.to_string(index=False))
    print(f"\nOutputs saved under: {HERE}")


if __name__ == "__main__":
    main()
