import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ASSIGNMENT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ASSIGNMENT_DIR / "splines_train.csv"
TEST_PATH = ASSIGNMENT_DIR / "splines_test.csv"
RESULTS_PATH = ASSIGNMENT_DIR / "splines_results.json"
CV_SUMMARY_PATH = ASSIGNMENT_DIR / "experiments" / "splines_cv_summary.csv"

BOUNDARY_LEFT = 0.0
BOUNDARY_RIGHT = 40.0
SPLINE_DEGREE = 3
RANDOM_SEED = 9618
N_FOLDS = 5
KNOT_CANDIDATES = tuple(range(0, 11))


@dataclass
class RegressionFit:
    internal_knots: np.ndarray
    beta: np.ndarray


@dataclass
class LogisticFit:
    internal_knots: np.ndarray
    model: LogisticRegression


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def make_internal_knots(temperature: np.ndarray, n_internal_knots: int) -> np.ndarray:
    if n_internal_knots == 0:
        return np.array([], dtype=float)
    probs = np.linspace(0.0, 1.0, n_internal_knots + 2)[1:-1]
    knots = np.quantile(temperature, probs)
    return np.asarray(knots, dtype=float)


def make_clamped_knot_vector(internal_knots: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.repeat(BOUNDARY_LEFT, SPLINE_DEGREE + 1),
            np.asarray(internal_knots, dtype=float),
            np.repeat(BOUNDARY_RIGHT, SPLINE_DEGREE + 1),
        ]
    )


def bspline_basis_1d(x: np.ndarray, knot_vector: np.ndarray, degree: int = SPLINE_DEGREE) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    knots = np.asarray(knot_vector, dtype=float)
    x_eval = np.clip(x, knots[degree], knots[-degree - 1])

    basis = ((x_eval[:, None] >= knots[:-1]) & (x_eval[:, None] < knots[1:])).astype(float)
    at_right = np.isclose(x_eval, knots[-1])
    if np.any(at_right):
        basis[at_right, :] = 0.0
        basis[at_right, -1] = 1.0

    for current_degree in range(1, degree + 1):
        n_cols = len(knots) - current_degree - 1
        next_basis = np.zeros((len(x_eval), n_cols), dtype=float)

        for j in range(n_cols):
            left_den = knots[j + current_degree] - knots[j]
            right_den = knots[j + current_degree + 1] - knots[j + 1]

            if left_den > 0:
                next_basis[:, j] += (
                    (x_eval - knots[j]) / left_den
                ) * basis[:, j]
            if right_den > 0:
                next_basis[:, j] += (
                    (knots[j + current_degree + 1] - x_eval) / right_den
                ) * basis[:, j + 1]

        basis = next_basis

    at_left = np.isclose(x_eval, knots[degree])
    at_right = np.isclose(x_eval, knots[-degree - 1])
    if np.any(at_left):
        basis[at_left, :] = 0.0
        basis[at_left, 0] = 1.0
    if np.any(at_right):
        basis[at_right, :] = 0.0
        basis[at_right, -1] = 1.0

    return basis


def design_matrix(df: pd.DataFrame, internal_knots: np.ndarray) -> np.ndarray:
    knot_vector = make_clamped_knot_vector(internal_knots)
    bmat = bspline_basis_1d(df["temperature"].to_numpy(dtype=float), knot_vector)
    humidity = df["humidity"].to_numpy(dtype=float)[:, None]
    weekend = df["weekend"].to_numpy(dtype=float)[:, None]
    return np.column_stack([bmat, humidity, weekend])


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def fit_regression(train: pd.DataFrame, n_internal_knots: int) -> RegressionFit:
    internal_knots = make_internal_knots(
        train["temperature"].to_numpy(dtype=float),
        n_internal_knots,
    )
    x_train = design_matrix(train, internal_knots)
    y_train = train["consumption_kwh"].to_numpy(dtype=float)
    beta = fit_ols(x_train, y_train)
    return RegressionFit(internal_knots=internal_knots, beta=beta)


def predict_regression(df: pd.DataFrame, fit: RegressionFit) -> np.ndarray:
    return design_matrix(df, fit.internal_knots) @ fit.beta


def fit_logistic_model(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=np.inf,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-10,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Setting penalty=None will ignore")
        warnings.filterwarnings("ignore", message="'penalty' was deprecated")
        model.fit(x, y)
    return model


def fit_logistic(train: pd.DataFrame, n_internal_knots: int) -> LogisticFit:
    internal_knots = make_internal_knots(
        train["temperature"].to_numpy(dtype=float),
        n_internal_knots,
    )
    x_train = design_matrix(train, internal_knots)
    y_train = train["high_demand_alert"].to_numpy(dtype=int)
    model = fit_logistic_model(x_train, y_train)
    return LogisticFit(internal_knots=internal_knots, model=model)


def predict_logistic_probability(df: pd.DataFrame, fit: LogisticFit) -> np.ndarray:
    return fit.model.predict_proba(design_matrix(df, fit.internal_knots))[:, 1]


def make_folds(n_rows: int, n_folds: int = N_FOLDS, seed: int = RANDOM_SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(shuffled, n_folds)]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def log_loss_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_true - y_prob) ** 2))


def cross_validate_models(train: pd.DataFrame) -> pd.DataFrame:
    folds = make_folds(len(train))
    rows = []

    for n_knots in KNOT_CANDIDATES:
        reg_scores = []
        cls_log_losses = []
        cls_brier_scores = []

        for validation_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(train)), validation_idx)
            fold_train = train.iloc[train_idx].reset_index(drop=True)
            fold_valid = train.iloc[validation_idx].reset_index(drop=True)

            reg_fit = fit_regression(fold_train, n_knots)
            y_reg = fold_valid["consumption_kwh"].to_numpy(dtype=float)
            pred_reg = predict_regression(fold_valid, reg_fit)
            reg_scores.append(mse(y_reg, pred_reg))

            cls_fit = fit_logistic(fold_train, n_knots)
            y_cls = fold_valid["high_demand_alert"].to_numpy(dtype=int)
            prob_cls = predict_logistic_probability(fold_valid, cls_fit)
            cls_log_losses.append(log_loss_binary(y_cls, prob_cls))
            cls_brier_scores.append(brier_score(y_cls, prob_cls))

        rows.append(
            {
                "n_internal_knots": n_knots,
                "regression_cv_mse": float(np.mean(reg_scores)),
                "regression_cv_mse_sd": float(np.std(reg_scores, ddof=1)),
                "classification_cv_log_loss": float(np.mean(cls_log_losses)),
                "classification_cv_log_loss_sd": float(np.std(cls_log_losses, ddof=1)),
                "classification_cv_brier": float(np.mean(cls_brier_scores)),
                "classification_cv_brier_sd": float(np.std(cls_brier_scores, ddof=1)),
            }
        )

    return pd.DataFrame(rows)


def choose_knot_counts(cv_summary: pd.DataFrame) -> tuple[int, int]:
    reg_idx = int(cv_summary["regression_cv_mse"].idxmin())
    cls_idx = int(cv_summary["classification_cv_log_loss"].idxmin())
    return (
        int(cv_summary.loc[reg_idx, "n_internal_knots"]),
        int(cv_summary.loc[cls_idx, "n_internal_knots"]),
    )


def temperature_effect_grid(
    fit: RegressionFit | LogisticFit,
    humidity: float,
    weekend: int,
    n_points: int = 241,
) -> pd.DataFrame:
    grid = pd.DataFrame(
        {
            "temperature": np.linspace(BOUNDARY_LEFT, BOUNDARY_RIGHT, n_points),
            "humidity": humidity,
            "weekend": weekend,
        }
    )
    if isinstance(fit, RegressionFit):
        grid["prediction"] = predict_regression(grid, fit)
    else:
        grid["prediction"] = predict_logistic_probability(grid, fit)
    return grid


def solve_assignment(save_outputs: bool = True) -> tuple[dict, dict]:
    train, test = load_data()
    cv_summary = cross_validate_models(train)
    n_knots_reg, n_knots_cls = choose_knot_counts(cv_summary)

    reg_fit = fit_regression(train, n_knots_reg)
    cls_fit = fit_logistic(train, n_knots_cls)

    pred_consumption = predict_regression(test, reg_fit)
    pred_prob = predict_logistic_probability(test, cls_fit)
    pred_class = (pred_prob >= 0.5).astype(int)

    results = {
        "pred_consumption_kwh": pred_consumption.astype(float).tolist(),
        "pred_high_demand_prob": pred_prob.astype(float).tolist(),
        "pred_high_demand_class": pred_class.astype(int).tolist(),
    }

    diagnostics = {
        "cv_summary": cv_summary,
        "n_knots_regression": n_knots_reg,
        "n_knots_classification": n_knots_cls,
        "regression_internal_knots": reg_fit.internal_knots.tolist(),
        "classification_internal_knots": cls_fit.internal_knots.tolist(),
        "regression_beta": reg_fit.beta.tolist(),
        "classification_beta": cls_fit.model.coef_.ravel().tolist(),
        "train_consumption_mean": float(train["consumption_kwh"].mean()),
        "train_high_demand_rate": float(train["high_demand_alert"].mean()),
    }

    if save_outputs:
        with open(RESULTS_PATH, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

        CV_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv_summary.to_csv(CV_SUMMARY_PATH, index=False)

    return results, diagnostics


if __name__ == "__main__":
    solve_assignment(save_outputs=True)
