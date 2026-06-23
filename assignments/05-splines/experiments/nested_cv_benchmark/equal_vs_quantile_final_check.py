import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
ASSIGNMENT_DIR = HERE.parents[1]
sys.path.insert(0, str(ASSIGNMENT_DIR))
sys.path.insert(0, str(HERE))

from splines_code import RESULTS_PATH, load_data  # noqa: E402
from nested_cv_benchmark import (  # noqa: E402
    RANDOM_SEED,
    SplineSpec,
    fit_spline_model,
    predict_spline_model,
)


N_BOOTSTRAP = 300
FIXED_SCORES_PATH = HERE / "fixed_spline_candidate_scores.csv"
COMPARISON_PATH = HERE / "equal_vs_quantile_m2_comparison.csv"
STABILITY_PATH = HERE / "equal_vs_quantile_m2_stability.csv"
PLOT_PATH = HERE / "equal_vs_quantile_m2_stability.png"
CANDIDATE_RESULTS_PATH = HERE / "equal_m2_candidate_results.json"
REPORT_PATH = HERE / "equal_vs_quantile_m2_report.txt"


def main() -> None:
    train, test = load_data()
    fixed = pd.read_csv(FIXED_SCORES_PATH)
    reg = fixed[
        (fixed["task"] == "regression")
        & fixed["candidate"].isin(
            ["spline_quantile_m2", "spline_equal_m2"]
        )
    ]
    paired = reg.pivot_table(
        index=["repeat", "outer_fold"],
        columns="candidate",
        values=["primary_score", "secondary_score"],
    )

    mse_diff = (
        paired[("primary_score", "spline_equal_m2")]
        - paired[("primary_score", "spline_quantile_m2")]
    )
    mae_diff = (
        paired[("secondary_score", "spline_equal_m2")]
        - paired[("secondary_score", "spline_quantile_m2")]
    )

    critical = stats.t.ppf(0.975, len(mse_diff) - 1)
    mse_se = mse_diff.std(ddof=1) / np.sqrt(len(mse_diff))
    mae_se = mae_diff.std(ddof=1) / np.sqrt(len(mae_diff))

    comparison = pd.DataFrame(
        [
            {
                "metric": "MSE",
                "quantile_m2_mean": paired[
                    ("primary_score", "spline_quantile_m2")
                ].mean(),
                "equal_m2_mean": paired[
                    ("primary_score", "spline_equal_m2")
                ].mean(),
                "equal_minus_quantile": mse_diff.mean(),
                "ci95_low": mse_diff.mean() - critical * mse_se,
                "ci95_high": mse_diff.mean() + critical * mse_se,
                "paired_pvalue": stats.ttest_rel(
                    paired[
                        ("primary_score", "spline_equal_m2")
                    ],
                    paired[
                        ("primary_score", "spline_quantile_m2")
                    ],
                ).pvalue,
                "fraction_equal_better": np.mean(mse_diff < 0),
            },
            {
                "metric": "MAE",
                "quantile_m2_mean": paired[
                    ("secondary_score", "spline_quantile_m2")
                ].mean(),
                "equal_m2_mean": paired[
                    ("secondary_score", "spline_equal_m2")
                ].mean(),
                "equal_minus_quantile": mae_diff.mean(),
                "ci95_low": mae_diff.mean() - critical * mae_se,
                "ci95_high": mae_diff.mean() + critical * mae_se,
                "paired_pvalue": stats.ttest_rel(
                    paired[
                        ("secondary_score", "spline_equal_m2")
                    ],
                    paired[
                        ("secondary_score", "spline_quantile_m2")
                    ],
                ).pvalue,
                "fraction_equal_better": np.mean(mae_diff < 0),
            },
        ]
    )
    comparison.to_csv(COMPARISON_PATH, index=False)

    quantile_spec = SplineSpec("quantile", 2)
    equal_spec = SplineSpec("equal", 2)
    grid = pd.DataFrame(
        {
            "temperature": np.linspace(0.0, 40.0, 181),
            "humidity": float(train["humidity"].median()),
            "weekend": 0,
        }
    )
    rng = np.random.default_rng(RANDOM_SEED + 707)
    quantile_curves = []
    equal_curves = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[idx].reset_index(drop=True)
        q_knots, q_fit = fit_spline_model(
            sample,
            "regression",
            quantile_spec,
        )
        e_knots, e_fit = fit_spline_model(
            sample,
            "regression",
            equal_spec,
        )
        quantile_curves.append(
            predict_spline_model(
                grid,
                "regression",
                q_knots,
                q_fit,
            )
        )
        equal_curves.append(
            predict_spline_model(
                grid,
                "regression",
                e_knots,
                e_fit,
            )
        )

    quantile_curves = np.asarray(quantile_curves)
    equal_curves = np.asarray(equal_curves)
    stability_rows = []
    curve_summary = {}
    for name, curves in [
        ("quantile_m2", quantile_curves),
        ("equal_m2", equal_curves),
    ]:
        q05 = np.quantile(curves, 0.05, axis=0)
        q50 = np.quantile(curves, 0.50, axis=0)
        q95 = np.quantile(curves, 0.95, axis=0)
        width = q95 - q05
        curve_summary[name] = (q05, q50, q95)
        stability_rows.append(
            {
                "model": name,
                "median_90pct_band_width": np.median(width),
                "mean_90pct_band_width": np.mean(width),
                "max_90pct_band_width": np.max(width),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(STABILITY_PATH, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, name, title, color in [
        (
            axes[0],
            "quantile_m2",
            "Quantile knots: 14.57, 25.76",
            "#2f6f9f",
        ),
        (
            axes[1],
            "equal_m2",
            "Equal knots: 13.33, 26.67",
            "#3a8f60",
        ),
    ]:
        q05, q50, q95 = curve_summary[name]
        ax.fill_between(
            grid["temperature"],
            q05,
            q95,
            color=color,
            alpha=0.25,
        )
        ax.plot(
            grid["temperature"],
            q50,
            color=color,
            linewidth=2.3,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("temperature")
        ax.set_ylabel("predicted consumption")
        ax.grid(alpha=0.22)
    fig.suptitle(
        "Two-Knot Regression Bootstrap Stability",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    q_knots, q_fit = fit_spline_model(
        train,
        "regression",
        quantile_spec,
    )
    e_knots, e_fit = fit_spline_model(
        train,
        "regression",
        equal_spec,
    )
    q_test = predict_spline_model(
        test,
        "regression",
        q_knots,
        q_fit,
    )
    e_test = predict_spline_model(
        test,
        "regression",
        e_knots,
        e_fit,
    )
    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        current_results = json.load(file)
    candidate_results = dict(current_results)
    candidate_results["pred_consumption_kwh"] = e_test.tolist()
    with open(
        CANDIDATE_RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(candidate_results, file, indent=2)

    test_difference = e_test - q_test
    report = [
        "EQUAL-SPACED VS QUANTILE-SPACED TWO-KNOT REGRESSION",
        "",
        "Real-data repeated outer validation:",
        comparison.to_string(index=False),
        "",
        "Bootstrap curve stability:",
        stability.to_string(index=False),
        "",
        "Test-prediction sensitivity:",
        f"- RMSE between regression predictions: "
        f"{np.sqrt(np.mean(test_difference**2)):.6f}",
        f"- mean absolute difference: "
        f"{np.mean(np.abs(test_difference)):.6f}",
        f"- maximum absolute difference: "
        f"{np.max(np.abs(test_difference)):.6f}",
        f"- correlation: {np.corrcoef(q_test, e_test)[0, 1]:.9f}",
        "",
        "Interpretation:",
        "Equal spacing has a small but consistent validation advantage.",
        "The practical difference is tiny, so this is a refinement rather",
        "than a fundamentally better model.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))


if __name__ == "__main__":
    main()
