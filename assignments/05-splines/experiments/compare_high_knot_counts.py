import sys
from pathlib import Path

import numpy as np
import pandas as pd


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT_DIR))

from splines_code import (  # noqa: E402
    brier_score,
    fit_logistic,
    fit_regression,
    load_data,
    log_loss_binary,
    make_folds,
    mse,
    predict_logistic_probability,
    predict_regression,
)


KNOT_COUNTS = [10, 15, 20, 25, 30]
OUT_PATH = Path(__file__).resolve().parent / "splines_high_knot_cv_comparison.csv"


def main() -> None:
    train, _ = load_data()
    folds = make_folds(len(train))
    rows = []

    for n_knots in KNOT_COUNTS:
        reg_mse = []
        cls_log_loss = []
        cls_brier = []

        for valid_idx in folds:
            train_idx = np.setdiff1d(np.arange(len(train)), valid_idx)
            fold_train = train.iloc[train_idx].reset_index(drop=True)
            fold_valid = train.iloc[valid_idx].reset_index(drop=True)

            reg_fit = fit_regression(fold_train, n_knots)
            reg_pred = predict_regression(fold_valid, reg_fit)
            reg_true = fold_valid["consumption_kwh"].to_numpy(dtype=float)
            reg_mse.append(mse(reg_true, reg_pred))

            cls_fit = fit_logistic(fold_train, n_knots)
            cls_prob = predict_logistic_probability(fold_valid, cls_fit)
            cls_true = fold_valid["high_demand_alert"].to_numpy(dtype=int)
            cls_log_loss.append(log_loss_binary(cls_true, cls_prob))
            cls_brier.append(brier_score(cls_true, cls_prob))

        rows.append(
            {
                "n_internal_knots": n_knots,
                "n_spline_basis": n_knots + 4,
                "n_design_columns": n_knots + 6,
                "regression_cv_mse_mean": np.mean(reg_mse),
                "regression_cv_mse_sd": np.std(reg_mse, ddof=1),
                "classification_cv_log_loss_mean": np.mean(cls_log_loss),
                "classification_cv_log_loss_sd": np.std(cls_log_loss, ddof=1),
                "classification_cv_brier_mean": np.mean(cls_brier),
                "classification_cv_brier_sd": np.std(cls_brier, ddof=1),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
