import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from energy_demand_code import (  # noqa: E402
    design_matrix,
    fit_ols,
    load_data,
    make_folds,
    make_internal_knots,
    mse,
)


FIXED_KNOTS = np.array([9.0, 21.5, 34.0])
OUT_PATH = Path(__file__).resolve().parent / "energy_demand_fixed_regression_knots_comparison.csv"


def evaluate_knots(train: pd.DataFrame, label: str, knot_rule) -> dict:
    fold_scores = []
    folds = make_folds(len(train))

    for valid_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(train)), valid_idx)
        fold_train = train.iloc[train_idx].reset_index(drop=True)
        fold_valid = train.iloc[valid_idx].reset_index(drop=True)

        knots = np.asarray(knot_rule(fold_train), dtype=float)
        x_train = design_matrix(fold_train, knots)
        x_valid = design_matrix(fold_valid, knots)
        y_train = fold_train["consumption_kwh"].to_numpy(dtype=float)
        y_valid = fold_valid["consumption_kwh"].to_numpy(dtype=float)

        beta = fit_ols(x_train, y_train)
        fold_scores.append(mse(y_valid, x_valid @ beta))

    return {
        "model": label,
        "internal_knots": ", ".join(f"{k:.4f}" for k in knot_rule(train)),
        "n_internal_knots": len(knot_rule(train)),
        "n_spline_basis": len(knot_rule(train)) + 4,
        "cv_mse_mean": np.mean(fold_scores),
        "cv_mse_sd": np.std(fold_scores, ddof=1),
        **{f"fold_{i + 1}_mse": score for i, score in enumerate(fold_scores)},
    }


def main() -> None:
    train, _ = load_data()

    rows = [
        evaluate_knots(
            train,
            "current selected: 2 quantile knots",
            lambda df: make_internal_knots(
                df["temperature"].to_numpy(dtype=float),
                2,
            ),
        ),
        evaluate_knots(
            train,
            "proposed fixed: 9, 21.5, 34",
            lambda _df: FIXED_KNOTS.copy(),
        ),
    ]

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_PATH, index=False)
    print(comparison.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
