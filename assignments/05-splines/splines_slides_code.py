import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from splines_code import (
    ASSIGNMENT_DIR,
    BOUNDARY_LEFT,
    BOUNDARY_RIGHT,
    CV_SUMMARY_PATH,
    RANDOM_SEED,
    RESULTS_PATH,
    choose_knot_counts,
    fit_logistic,
    fit_regression,
    load_data,
    predict_logistic_probability,
    predict_regression,
    solve_assignment,
    temperature_effect_grid,
)


PDF_PATH = Path(os.environ.get("splines_SLIDES_PDF_PATH", ASSIGNMENT_DIR / "splines_slides.pdf"))
EXPERIMENTS_DIR = ASSIGNMENT_DIR / "experiments"
CONSUMPTION_EDA_PATH = EXPERIMENTS_DIR / "consumption_vs_covariates.png"
ALERT_RATE_EDA_PATH = EXPERIMENTS_DIR / "high_demand_alert_rate_by_covariate_bins.png"
BOOTSTRAP_SUMMARY_PATH = EXPERIMENTS_DIR / "splines_bootstrap_curve_summary.csv"
B_BOOTSTRAP = 200

BLUE = "#2f6f9f"
RED = "#c94f37"
GREEN = "#3a8f60"
PURPLE = "#7257a8"
GOLD = "#bf8f2f"
GRAY = "#555555"
LIGHT_GRAY = "#e8eef3"
DARK = "#222222"


def draw_table(ax, df, title, fontsize=7.5, scale_x=1.0, scale_y=1.2, col_widths=None):
    ### Draws a pandas table inside a slide axis
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(scale_x, scale_y)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(LIGHT_GRAY)
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#b9c2cb")
        cell.set_linewidth(0.50)
    if col_widths is not None:
        for col_idx, width in enumerate(col_widths):
            for row_idx in range(len(df) + 1):
                table[(row_idx, col_idx)].set_width(width)


def text_panel(ax, title, lines, fontsize=8.5, family="monospace"):
    ### Writes text inside a slide panel
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.text(
        0.03,
        0.52,
        "\n".join(lines),
        fontsize=fontsize,
        family=family,
        transform=ax.transAxes,
        va="center",
        color=DARK,
        linespacing=1.20,
    )


def binned_alert_rate(train: pd.DataFrame, covariate: str, bins: int = 10) -> pd.DataFrame:
    ### Computes empirical alert rates in quantile bins
    work = train[[covariate, "high_demand_alert"]].copy()
    work["bin"] = pd.qcut(work[covariate], q=bins, duplicates="drop")
    return (
        work.groupby("bin", observed=True)
        .agg(
            alert_rate=("high_demand_alert", "mean"),
            n=("high_demand_alert", "size"),
            x_mid=(covariate, "mean"),
        )
        .reset_index()
    )


def create_eda_plot_images(train: pd.DataFrame) -> None:
    ### Creates the two EDA images used on Slide 1
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    for ax, cov in zip(axes, ["temperature", "humidity", "weekend"]):
        if cov == "weekend":
            jitter = np.random.default_rng(RANDOM_SEED).normal(0, 0.035, len(train))
            ax.scatter(train[cov] + jitter, train["consumption_kwh"], s=18, alpha=0.50, color=BLUE, edgecolor="none")
            means = train.groupby(cov)["consumption_kwh"].mean()
            ax.scatter(means.index, means.values, s=85, color=RED, edgecolor="white", linewidth=0.8, zorder=5)
            ax.set_xticks([0, 1])
        else:
            ax.scatter(train[cov], train["consumption_kwh"], s=18, alpha=0.55, color=BLUE, edgecolor="none")
            bins = pd.qcut(train[cov], 12, duplicates="drop")
            smooth = train.groupby(bins, observed=True).agg(x=(cov, "mean"), y=("consumption_kwh", "mean"))
            ax.plot(smooth["x"], smooth["y"], color=RED, linewidth=2.2)
        ax.set_title(f"Consumption vs {cov}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(cov)
        ax.set_ylabel("consumption_kwh")
        ax.grid(alpha=0.22, linewidth=0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(CONSUMPTION_EDA_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    for ax, cov in zip(axes[:2], ["temperature", "humidity"]):
        grouped = binned_alert_rate(train, cov)
        ax.plot(grouped["x_mid"], grouped["alert_rate"], color=PURPLE, linewidth=2.4, marker="o", markersize=6.4)
        for _, row in grouped.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["x_mid"], row["alert_rate"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_title(f"Alert rate vs {cov}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(cov)
        ax.set_ylabel("mean high_demand_alert")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.22, linewidth=0.5)

    weekend = train.groupby("weekend")["high_demand_alert"].agg(["mean", "count"]).reset_index()
    bars = axes[2].bar(weekend["weekend"].astype(str), weekend["mean"], color=[GREEN, GOLD], width=0.55)
    for bar, (_, row) in zip(bars, weekend.iterrows()):
        axes[2].annotate(
            f"{row['mean']:.2f}\nn={int(row['count'])}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )
    axes[2].set_title("Alert rate vs weekend", fontsize=14, fontweight="bold", pad=10)
    axes[2].set_xlabel("weekend")
    axes[2].set_ylabel("mean high_demand_alert")
    axes[2].set_ylim(0, 1.08)
    axes[2].grid(axis="y", alpha=0.22, linewidth=0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(ALERT_RATE_EDA_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def show_image(ax, image_path: Path) -> None:
    ### Places a saved EDA figure onto a slide
    ax.imshow(plt.imread(image_path))
    ax.axis("off")


def summary_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    ### Creates compact data summaries for Slide 1
    rows = []
    for col, label, role in [
        ("temperature", "temp", "covariate"),
        ("humidity", "humidity", "covariate"),
        ("weekend", "weekend", "covariate"),
        ("consumption_kwh", "consump", "response"),
        ("high_demand_alert", "alert", "response"),
    ]:
        rows.append(
            {
                "variable": label,
                "role": role,
                "mean": f"{train[col].mean():.2f}",
                "sd": f"{train[col].std(ddof=1):.2f}",
                "min": f"{train[col].min():.2f}",
                "max": f"{train[col].max():.2f}",
            }
        )
    rows.append(
        {
            "variable": "test",
            "role": "hidden y",
            "mean": f"n={len(test)}",
            "sd": "",
            "min": f"T {test.temperature.min():.2f}",
            "max": f"T {test.temperature.max():.2f}",
        }
    )
    return pd.DataFrame(rows)


def correlation_table(train: pd.DataFrame) -> pd.DataFrame:
    ### Reports correlations among targets and covariates
    corr = train[["temperature", "humidity", "weekend", "consumption_kwh", "high_demand_alert"]].corr()
    pairs = [
        ("temperature", "consumption_kwh"),
        ("humidity", "consumption_kwh"),
        ("weekend", "consumption_kwh"),
        ("temperature", "high_demand_alert"),
        ("humidity", "high_demand_alert"),
        ("weekend", "high_demand_alert"),
    ]
    rows = []
    for x, y in pairs:
        rows.append({"x": x, "y": y, "corr": f"{corr.loc[x, y]:.3f}"})
    return pd.DataFrame(rows)


def bootstrap_curves(train: pd.DataFrame, n_knots_reg: int, n_knots_cls: int, humidity: float, weekend: int):
    ### Refits selected spline models on row-bootstrap samples for curve stability
    rng = np.random.default_rng(RANDOM_SEED + 77)
    temp_grid = np.linspace(BOUNDARY_LEFT, BOUNDARY_RIGHT, 181)
    grid = pd.DataFrame({"temperature": temp_grid, "humidity": humidity, "weekend": weekend})

    reg_curves = []
    cls_curves = []
    for _ in range(B_BOOTSTRAP):
        idx = rng.integers(0, len(train), size=len(train))
        sample = train.iloc[idx].reset_index(drop=True)
        reg_curves.append(predict_regression(grid, fit_regression(sample, n_knots_reg)))
        cls_curves.append(predict_logistic_probability(grid, fit_logistic(sample, n_knots_cls)))

    reg_curves = np.asarray(reg_curves)
    cls_curves = np.asarray(cls_curves)
    return {
        "temperature": temp_grid,
        "reg_curves": reg_curves,
        "cls_curves": cls_curves,
        "reg_q05": np.quantile(reg_curves, 0.05, axis=0),
        "reg_q50": np.quantile(reg_curves, 0.50, axis=0),
        "reg_q95": np.quantile(reg_curves, 0.95, axis=0),
        "cls_q05": np.quantile(cls_curves, 0.05, axis=0),
        "cls_q50": np.quantile(cls_curves, 0.50, axis=0),
        "cls_q95": np.quantile(cls_curves, 0.95, axis=0),
    }


def save_bootstrap_summary(curves: dict[str, np.ndarray]) -> None:
    ### Writes bootstrap curve quantiles used in the slide interpretation
    out = pd.DataFrame(
        {
            "temperature": curves["temperature"],
            "consumption_q05": curves["reg_q05"],
            "consumption_q50": curves["reg_q50"],
            "consumption_q95": curves["reg_q95"],
            "high_demand_prob_q05": curves["cls_q05"],
            "high_demand_prob_q50": curves["cls_q50"],
            "high_demand_prob_q95": curves["cls_q95"],
        }
    )
    out.to_csv(BOOTSTRAP_SUMMARY_PATH, index=False)


def coefficient_table(reg_fit, cls_fit) -> tuple[pd.DataFrame, pd.DataFrame]:
    ### Creates compact coefficient tables for Slide 2
    reg_names = [f"B{j + 1}(T)" for j in range(len(reg_fit.beta) - 2)] + ["humidity", "weekend"]
    cls_beta = cls_fit.model.coef_.ravel()
    cls_names = [f"B{j + 1}(T)" for j in range(len(cls_beta) - 2)] + ["humidity", "weekend"]

    reg_table = pd.DataFrame(
        {
            "term": reg_names,
            "OLS beta": [f"{b:.3f}" for b in reg_fit.beta],
        }
    )
    cls_table = pd.DataFrame(
        {
            "term": cls_names,
            "logit beta": [f"{b:.3f}" for b in cls_beta],
        }
    )
    return reg_table, cls_table


def main():
    ### Creates the required three-slide PDF
    results, _ = solve_assignment(save_outputs=True)
    train, test = load_data()
    cv_summary = pd.read_csv(CV_SUMMARY_PATH)
    n_knots_reg, n_knots_cls = choose_knot_counts(cv_summary)
    reg_fit = fit_regression(train, n_knots_reg)
    cls_fit = fit_logistic(train, n_knots_cls)
    median_humidity = float(train["humidity"].median())
    reference_weekend = 0
    curves = bootstrap_curves(train, n_knots_reg, n_knots_cls, median_humidity, reference_weekend)
    save_bootstrap_summary(curves)
    create_eda_plot_images(train)

    reg_coef_table, cls_coef_table = coefficient_table(reg_fit, cls_fit)
    pred_y = np.asarray(results["pred_consumption_kwh"], dtype=float)
    pred_p = np.asarray(results["pred_high_demand_prob"], dtype=float)
    pred_c = np.asarray(results["pred_high_demand_class"], dtype=int)

    with PdfPages(PDF_PATH) as pdf:
        #===========================================================
        #   SLIDE 1: DATA

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[0.62, 1.28, 1.28], hspace=0.28, wspace=0.30)
        fig.suptitle("Data", fontsize=18, fontweight="bold", y=0.965)

        ax_overview = fig.add_subplot(gs[0, 0])
        overview = (
            "Data files:\n"
            "  splines_train.csv\n"
            "  splines_test.csv\n\n"
            f"Training observations: {len(train)}\n"
            f"Test observations: {len(test)}\n"
            "Covariates: temperature, humidity, weekend\n"
            "Responses: consumption_kwh, high_demand_alert"
        )
        ax_overview.text(0.02, 0.50, overview, fontsize=7.2, family="monospace", va="center")
        ax_overview.set_title("Sample size and structure", fontsize=10.5, fontweight="bold")
        ax_overview.axis("off")

        ax_summary = fig.add_subplot(gs[0, 1])
        draw_table(ax_summary, summary_table(train, test), "Y and covariate summary", fontsize=5.7, scale_y=1.05)

        ax_response = fig.add_subplot(gs[0, 2])
        ax_response.hist(train["consumption_kwh"], bins=18, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.6)
        ax_response.axvline(train["consumption_kwh"].mean(), color=RED, linewidth=1.5, label="mean")
        ax_response.set_title("Response distribution", fontsize=10.5, fontweight="bold")
        ax_response.set_xlabel("consumption_kwh", fontsize=8)
        ax_response.set_ylabel("count", fontsize=8)
        ax_response.tick_params(labelsize=7)
        ax_response.text(
            0.55,
            0.78,
            f"alert rate = {train.high_demand_alert.mean():.3f}",
            transform=ax_response.transAxes,
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cccccc"},
        )
        ax_response.grid(alpha=0.20, linewidth=0.5)

        ax_consumption = fig.add_subplot(gs[1, :])
        show_image(ax_consumption, CONSUMPTION_EDA_PATH)

        ax_alert = fig.add_subplot(gs[2, :])
        show_image(ax_alert, ALERT_RATE_EDA_PATH)

        fig.subplots_adjust(left=0.045, right=0.985, top=0.905, bottom=0.025)
        pdf.savefig(fig)
        plt.close(fig)

        #===========================================================
        #   SLIDE 2: MODEL, METHOD AND RESULTS

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.34)
        fig.suptitle("Model, Method and Results", fontsize=18, fontweight="bold", y=0.965)

        formula_lines = [
            "Regression model:",
            "  y_i = sum_j beta_j B_j(T_i) + beta_H H_i + beta_W W_i + eps_i",
            "",
            "Classification model:",
            "  Pr(C_i=1) = sigmoid(sum_j beta_j B_j(T_i) + beta_H H_i + beta_W W_i)",
            "",
            "Estimated tuning:",
            f"  regression internal knots: {n_knots_reg}",
            f"  classification internal knots: {n_knots_cls}",
            "  boundary knots: 0 and 40 C",
            "  no intercept column",
        ]
        text_panel(fig.add_subplot(gs[0, 0]), "Model formula", formula_lines, fontsize=7.0)

        method_lines = [
            "Method:",
            "  1. Build clamped cubic knot vector",
            "  2. Construct B-spline basis manually",
            "     by Cox-de Boor recursion",
            "  3. Fit OLS manually with np.linalg.lstsq",
            "  4. Fit no-intercept logistic regression",
            "     on the hand-built design matrix",
            "  5. Choose knot count by manual 5-fold CV",
            "",
            "Classification threshold: 0.5",
        ]
        text_panel(fig.add_subplot(gs[0, 1]), "Method", method_lines, fontsize=7.6)

        draw_table(fig.add_subplot(gs[1, 0]), reg_coef_table, "Estimated regression coefficients", fontsize=7.1, scale_y=1.12)
        draw_table(fig.add_subplot(gs[1, 1]), cls_coef_table, "Estimated logistic coefficients", fontsize=7.6, scale_y=1.16)

        results_table = pd.DataFrame(
            [
                {
                    "quantity": "regression CV MSE",
                    "value": f"{cv_summary.loc[cv_summary.regression_cv_mse.idxmin(), 'regression_cv_mse']:.4f}",
                },
                {
                    "quantity": "classification CV log loss",
                    "value": f"{cv_summary.loc[cv_summary.classification_cv_log_loss.idxmin(), 'classification_cv_log_loss']:.4f}",
                },
                {
                    "quantity": "classification CV Brier",
                    "value": f"{cv_summary.loc[cv_summary.classification_cv_log_loss.idxmin(), 'classification_cv_brier']:.4f}",
                },
                {"quantity": "test consumption mean", "value": f"{pred_y.mean():.3f}"},
                {"quantity": "test probability mean", "value": f"{pred_p.mean():.3f}"},
                {"quantity": "predicted alert rate", "value": f"{pred_c.mean():.3f}"},
            ]
        )
        draw_table(fig.add_subplot(gs[2, 0]), results_table, "Numerical results", fontsize=7.4, scale_y=1.22)

        library_lines = [
            "Assumptions:",
            "  independent rows, not a time series",
            "  train and hidden test rows share the same process",
            "  temperature effect is smooth but nonlinear",
            "  humidity and weekend enter additively and linearly",
            "  bootstrap resamples full rows",
            "",
            "Libraries:",
            "  pandas, numpy, json",
            "  sklearn.linear_model.LogisticRegression",
            "  matplotlib for slides/plots",
        ]
        text_panel(fig.add_subplot(gs[2, 1]), "Assumptions and libraries", library_lines, fontsize=7.2)

        fig.subplots_adjust(left=0.065, right=0.965, top=0.90, bottom=0.06)
        pdf.savefig(fig)
        plt.close(fig)

        #===========================================================
        #   SLIDE 3: VISUALIZATION

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.36)
        fig.suptitle("Visualization", fontsize=18, fontweight="bold", y=0.965)

        n_grid = len(curves["temperature"])
        grid_reg = temperature_effect_grid(reg_fit, median_humidity, reference_weekend, n_points=n_grid)
        grid_cls = temperature_effect_grid(cls_fit, median_humidity, reference_weekend, n_points=n_grid)

        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(train["temperature"], train["consumption_kwh"], s=13, alpha=0.42, color=BLUE, edgecolor="none")
        ax.plot(grid_reg["temperature"], grid_reg["prediction"], color=RED, linewidth=2.3)
        ax.set_title("Consumption spline fit", fontsize=9.4, fontweight="bold")
        ax.set_xlabel("temperature", fontsize=8)
        ax.set_ylabel("consumption", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.20, linewidth=0.5)

        ax = fig.add_subplot(gs[0, 1])
        binned = binned_alert_rate(train, "temperature")
        ax.scatter(binned["x_mid"], binned["alert_rate"], s=38, color=PURPLE, zorder=3)
        ax.plot(binned["x_mid"], binned["alert_rate"], color=PURPLE, linewidth=1.4, alpha=0.85)
        ax.plot(grid_cls["temperature"], grid_cls["prediction"], color=RED, linewidth=2.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Alert probability spline fit", fontsize=9.4, fontweight="bold")
        ax.set_xlabel("temperature", fontsize=8)
        ax.set_ylabel("alert probability", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.20, linewidth=0.5)

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(cv_summary["n_internal_knots"], cv_summary["regression_cv_mse"], marker="o", color=BLUE, linewidth=1.9)
        best_reg = cv_summary.loc[cv_summary["regression_cv_mse"].idxmin()]
        ax.scatter(best_reg["n_internal_knots"], best_reg["regression_cv_mse"], s=58, color=RED, zorder=4)
        ax.set_title("Regression CV error", fontsize=9.4, fontweight="bold")
        ax.set_xlabel("internal knots", fontsize=8)
        ax.set_ylabel("CV MSE", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.20, linewidth=0.5)

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(cv_summary["n_internal_knots"], cv_summary["classification_cv_log_loss"], marker="o", color=PURPLE, linewidth=1.9)
        best_cls = cv_summary.loc[cv_summary["classification_cv_log_loss"].idxmin()]
        ax.scatter(best_cls["n_internal_knots"], best_cls["classification_cv_log_loss"], s=58, color=RED, zorder=4)
        ax.set_title("Classifier CV log loss", fontsize=9.4, fontweight="bold")
        ax.set_xlabel("internal knots", fontsize=8)
        ax.set_ylabel("CV log loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.20, linewidth=0.5)

        ax = fig.add_subplot(gs[1, 1])
        for curve in curves["reg_curves"][::10]:
            ax.plot(curves["temperature"], curve, color="#aeb7bf", alpha=0.22, linewidth=0.8)
        ax.fill_between(curves["temperature"], curves["reg_q05"], curves["reg_q95"], color=GOLD, alpha=0.28)
        ax.plot(curves["temperature"], grid_reg["prediction"], color=RED, linewidth=2.4)
        ax.set_title("Bootstrap stability: consumption", fontsize=9.4, fontweight="bold")
        ax.set_xlabel("temperature", fontsize=8)
        ax.set_ylabel("predicted consumption", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.20, linewidth=0.5)

        ax = fig.add_subplot(gs[1, 2])
        for curve in curves["cls_curves"][::10]:
            ax.plot(curves["temperature"], curve, color="#aeb7bf", alpha=0.22, linewidth=0.8)
        ax.fill_between(curves["temperature"], curves["cls_q05"], curves["cls_q95"], color=GREEN, alpha=0.24)
        ax.plot(curves["temperature"], grid_cls["prediction"], color=RED, linewidth=2.4)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Bootstrap stability: alert prob", fontsize=9.4, fontweight="bold")
        ax.set_xlabel("temperature", fontsize=8)
        ax.set_ylabel("predicted probability", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.20, linewidth=0.5)

        fig.subplots_adjust(left=0.065, right=0.965, top=0.90, bottom=0.07)
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
