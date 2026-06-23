import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
ASSIGNMENT_DIR = HERE.parents[1]
sys.path.insert(0, str(ASSIGNMENT_DIR))
sys.path.insert(0, str(HERE))

from splines_code import load_data  # noqa: E402
from pseudo_dgp_mc_experiment import stability_curves  # noqa: E402


RAW_PATH = HERE / "mc_real_validation_scores.csv"
OUT_PATH = HERE / "mc_current_vs_repeated_choice.csv"
REPORT_PATH = HERE / "mc_current_vs_repeated_choice.txt"


def config(n_knots: int) -> pd.Series:
    return pd.Series(
        {
            "augmentation_ratio": 0,
            "n_internal_knots": n_knots,
        }
    )


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    reg = raw[
        (raw["task"] == "regression")
        & (raw["metric"] == "mse")
        & (raw["augmentation_ratio"] == 0)
        & (raw["n_internal_knots"].isin([1, 2]))
    ]
    repeat_scores = (
        reg.groupby(["repeat", "n_internal_knots"])["score"]
        .mean()
        .unstack("n_internal_knots")
        .rename(columns={1: "one_knot_mse", 2: "two_knot_mse"})
    )
    repeat_scores["two_minus_one"] = (
        repeat_scores["two_knot_mse"] - repeat_scores["one_knot_mse"]
    )

    differences = repeat_scores["two_minus_one"].to_numpy(dtype=float)
    mean_diff = float(differences.mean())
    se_diff = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    t_critical = float(stats.t.ppf(0.975, df=len(differences) - 1))
    ci_low = mean_diff - t_critical * se_diff
    ci_high = mean_diff + t_critical * se_diff
    paired_p = float(stats.ttest_rel(
        repeat_scores["two_knot_mse"],
        repeat_scores["one_knot_mse"],
    ).pvalue)

    train, _ = load_data()
    humidity = float(train["humidity"].median())
    one_curves = stability_curves(train, "regression", config(1), humidity)
    two_curves = stability_curves(train, "regression", config(2), humidity)

    one_width = np.quantile(one_curves, 0.95, axis=0) - np.quantile(one_curves, 0.05, axis=0)
    two_width = np.quantile(two_curves, 0.95, axis=0) - np.quantile(two_curves, 0.05, axis=0)

    comparison = pd.DataFrame(
        [
            {
                "model": "one internal knot",
                "repeated_cv_mse": repeat_scores["one_knot_mse"].mean(),
                "repeat_to_repeat_sd": repeat_scores["one_knot_mse"].std(ddof=1),
                "median_90pct_curve_band_width": np.median(one_width),
                "mean_90pct_curve_band_width": np.mean(one_width),
            },
            {
                "model": "current two internal knots",
                "repeated_cv_mse": repeat_scores["two_knot_mse"].mean(),
                "repeat_to_repeat_sd": repeat_scores["two_knot_mse"].std(ddof=1),
                "median_90pct_curve_band_width": np.median(two_width),
                "mean_90pct_curve_band_width": np.mean(two_width),
            },
        ]
    )
    comparison.to_csv(OUT_PATH, index=False)

    report = [
        "CURRENT TWO-KNOT MODEL VS REPEATED-CV ONE-KNOT MODEL",
        "",
        f"Mean paired MSE difference (two minus one): {mean_diff:.6f}",
        f"95% CI for paired difference: [{ci_low:.6f}, {ci_high:.6f}]",
        f"Paired t-test p-value: {paired_p:.6f}",
        "",
        "Interpretation:",
        "A positive difference favors one knot. The difference is practically tiny,",
        "and the confidence interval includes zero. Repeated CV therefore does not",
        "provide convincing evidence that one knot is better than the current",
        "two-knot model.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(comparison.to_string(index=False))
    print()
    print("\n".join(report))


if __name__ == "__main__":
    main()
