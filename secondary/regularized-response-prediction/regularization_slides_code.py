import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path

from regularization_code import (
    B_BOOTSTRAP,
    LAMBDA_GRID_PATH,
    RESULTS_PATH,
    TEST_PATH,
    TRAIN_PATH,
    feature_names,
    fit_lasso_cv,
    fit_ridge_closed_form,
    solve_project,
    standardize_apply,
    standardize_fit,
)


#===========================================================
#   FILE PATHS AND PLOT SETTINGS

PDF_PATH = Path(os.environ.get("regularization_SLIDES_PDF_PATH", "regularization_slides.pdf"))

BLUE = "#2f6f9f"
RED = "#c94f37"
GREEN = "#3a8f60"
PURPLE = "#7257a8"
GOLD = "#bf8f2f"
GRAY = "#555555"
LIGHT_GRAY = "#edf1f4"
DARK = "#222222"


#===========================================================
#   RESULTS AND TABLE HELPERS

def load_results() -> dict:
    ### Loads existing JSON results or regenerates them if missing
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    results, _ = solve_project()
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


def draw_table(ax, df, title, fontsize=7.2, scale_y=1.15, col_widths=None):
    ### Draws a compact table inside a slide axis
    ax.axis("off")
    ax.set_title(title, fontsize=10.2, fontweight="bold", pad=7)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, scale_y)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(LIGHT_GRAY)
            cell.set_text_props(weight="bold", color=DARK)
        cell.set_edgecolor("#b9c2cb")
        cell.set_linewidth(0.45)
    if col_widths is not None:
        for col_idx, width in enumerate(col_widths):
            for row_idx in range(len(df) + 1):
                table[(row_idx, col_idx)].set_width(width)


def text_panel(ax, title, lines, fontsize=8.4, width=58, family="monospace"):
    ### Writes wrapped text inside a slide panel
    ax.axis("off")
    ax.set_title(title, fontsize=10.2, fontweight="bold", pad=7)
    wrapped_lines = []
    for line in lines:
        if line.strip() == "":
            wrapped_lines.append("")
        elif line.startswith("  "):
            wrapped_lines.append(textwrap.fill(line, width=width, subsequent_indent="  "))
        else:
            wrapped_lines.append(textwrap.fill(line, width=width))
    ax.text(
        0.02,
        0.98,
        "\n".join(wrapped_lines),
        transform=ax.transAxes,
        fontsize=fontsize,
        family=family,
        va="top",
        ha="left",
        color=DARK,
        linespacing=1.24,
    )


def format_float(value, digits=4):
    ### Formats a number for display on slides
    return f"{float(value):.{digits}f}"


def predictor_summary_table(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    ### Creates sample-size and distribution summary rows for Slide 1
    train_x = train[features].to_numpy(dtype=float)
    test_x = test[features].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "Quantity": "train rows",
                "Value": f"{len(train)}",
                "Notes": "observed y",
            },
            {
                "Quantity": "test rows",
                "Value": f"{len(test)}",
                "Notes": "hidden y",
            },
            {
                "Quantity": "predictors",
                "Value": f"{len(features)}",
                "Notes": "x1 through x40",
            },
            {
                "Quantity": "y mean / SD",
                "Value": f"{y.mean():.3f} / {y.std(ddof=1):.3f}",
                "Notes": f"range {y.min():.2f} to {y.max():.2f}",
            },
            {
                "Quantity": "train x mean",
                "Value": f"{train_x.mean():.3f}",
                "Notes": f"feature means {train_x.mean(axis=0).min():.2f} to {train_x.mean(axis=0).max():.2f}",
            },
            {
                "Quantity": "train x SD",
                "Value": f"{train_x.std(axis=0, ddof=1).mean():.3f}",
                "Notes": f"feature SDs {train_x.std(axis=0, ddof=1).min():.2f} to {train_x.std(axis=0, ddof=1).max():.2f}",
            },
            {
                "Quantity": "test x mean",
                "Value": f"{test_x.mean():.3f}",
                "Notes": f"similar scale to train",
            },
        ]
    )


def top_correlation_table(train: pd.DataFrame, features: list[str], n_rows=8) -> pd.DataFrame:
    ### Finds predictors with the largest absolute marginal correlation with y
    rows = []
    y = train["y"].to_numpy(dtype=float)
    for feature in features:
        corr = np.corrcoef(train[feature].to_numpy(dtype=float), y)[0, 1]
        rows.append({"Variable": feature, "corr(x,y)": corr, "abs corr": abs(corr)})
    out = pd.DataFrame(rows).sort_values("abs corr", ascending=False).head(n_rows)
    out["corr(x,y)"] = out["corr(x,y)"].map(lambda v: f"{v:.3f}")
    out["abs corr"] = out["abs corr"].map(lambda v: f"{v:.3f}")
    return out[["Variable", "corr(x,y)", "abs corr"]]


def lasso_cv_mse_from_training_data(train: pd.DataFrame, features: list[str]) -> float:
    ### Refits lasso CV and returns the selected alpha's mean validation MSE
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    _, lasso_fit = fit_lasso_cv(x_train, y_train)
    mean_mse_path = np.mean(lasso_fit.mse_path_, axis=1)
    selected_index = int(np.argmin(np.abs(lasso_fit.alphas_ - lasso_fit.alpha_)))
    return float(mean_mse_path[selected_index])


def results_table(results: dict, lasso_cv_mse: float) -> pd.DataFrame:
    ### Builds the numerical results table for Slide 2
    ridge_pred = np.asarray(results["ridge_pred_test"], dtype=float)
    ridge_gd_pred = np.asarray(results["ridge_gd_pred_test"], dtype=float)
    lasso_pred = np.asarray(results["lasso_pred_test"], dtype=float)
    return pd.DataFrame(
        [
            {"Result": "ridge lambda", "Value": f"{results['ridge_lambda']:.6f}"},
            {"Result": "min ridge CV MSE", "Value": f"{min(results['ridge_cv_mse']):.4f}"},
            {
                "Result": "GD vs closed-form RMSE",
                "Value": f"{np.sqrt(np.mean((ridge_pred - ridge_gd_pred) ** 2)):.2e}",
            },
            {"Result": "lasso alpha", "Value": f"{results['lasso_alpha']:.6f}"},
            {"Result": "lasso CV MSE", "Value": f"{lasso_cv_mse:.4f}"},
            {"Result": "lasso nonzero betas", "Value": f"{results['lasso_n_nonzero']} of 40"},
            {"Result": "ridge test pred mean", "Value": f"{ridge_pred.mean():.3f}"},
            {"Result": "lasso test pred mean", "Value": f"{lasso_pred.mean():.3f}"},
        ]
    )


#===========================================================
#   MODEL PATHS AND GRADIENT DESCENT DIAGNOSTICS

def ridge_path(x_train: np.ndarray, y_train: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    ### Computes ridge coefficients across the required lambda grid
    return np.vstack([fit_ridge_closed_form(x_train, y_train, float(lam)).beta_raw for lam in lambdas])


def lasso_coeff_path(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ### Computes lasso coefficients across sklearn's alpha path for visualization
    x_mean, x_scale = standardize_fit(x_train)
    x_z = standardize_apply(x_train, x_mean, x_scale)
    alphas, coef_path, _ = lasso_path(x_z, y_train, max_iter=100_000, tol=1e-8)
    coef_path_raw = coef_path / x_scale[:, None]
    return alphas, coef_path_raw


def gradient_descent_history(x_train: np.ndarray, y_train: np.ndarray, lam: float, closed_beta_std: np.ndarray):
    ### Records how quickly gradient descent approaches the closed-form ridge coefficients
    x_mean, x_scale = standardize_fit(x_train)
    x_z = standardize_apply(x_train, x_mean, x_scale)
    y_centered = y_train - y_train.mean()
    n, p = x_z.shape
    beta = np.zeros(p)
    largest_eigenvalue = float(np.linalg.eigvalsh((x_z.T @ x_z) / n).max())
    step_size = 0.95 / (2.0 * (largest_eigenvalue + lam))

    iterations = []
    beta_distance = []
    checkpoints = set(np.unique(np.r_[np.arange(0, 601, 10), np.arange(650, 2001, 50)]))

    for iteration in range(0, 2001):
        if iteration in checkpoints:
            iterations.append(iteration)
            beta_distance.append(np.linalg.norm(beta - closed_beta_std))

        residual = y_centered - x_z @ beta
        gradient = -(2.0 / n) * (x_z.T @ residual) + 2.0 * lam * beta
        beta = beta - step_size * gradient

    return np.asarray(iterations), np.asarray(beta_distance)


#===========================================================
#   BUILDING THE THREE-SLIDE PDF

def main():
    ### Creates the project slide deck PDF

    #===========================================================
    #   LOADING DATA AND RESULTS

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    lambdas = pd.read_csv(LAMBDA_GRID_PATH)["lambda"].to_numpy(dtype=float)
    results = load_results()

    features = feature_names()
    y = train["y"].to_numpy(dtype=float)
    train_x = train[features].to_numpy(dtype=float)
    corr_table = top_correlation_table(train, features)
    summary_table = predictor_summary_table(train, test, features)
    lasso_cv_mse = lasso_cv_mse_from_training_data(train, features)
    result_table = results_table(results, lasso_cv_mse)

    ridge_beta = np.asarray(results["ridge_beta"], dtype=float)
    lasso_beta = np.asarray(results["lasso_beta"], dtype=float)
    bootstrap_freq = np.asarray(results["bootstrap_selection_freq"], dtype=float)
    ridge_cv = np.asarray(results["ridge_cv_mse"], dtype=float)
    ridge_path_beta = ridge_path(train_x, y, lambdas)
    lasso_alphas, lasso_path_beta = lasso_coeff_path(train_x, y)
    closed_ridge = fit_ridge_closed_form(train_x, y, float(results["ridge_lambda"]))
    gd_iter, gd_dist = gradient_descent_history(
        train_x, y, float(results["ridge_lambda"]), closed_ridge.beta_std
    )

    #===========================================================
    #   WRITING PDF SLIDES

    with PdfPages(PDF_PATH) as pdf:
        #===========================================================
        #   SLIDE 1: DATA

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
        fig.suptitle("Data", fontsize=18, fontweight="bold", y=0.965)

        # Sample size and summary statistics
        ax_summary = fig.add_subplot(gs[0, 0])
        draw_table(ax_summary, summary_table, "Sample size and summary statistics", fontsize=6.7, scale_y=1.18)

        # Largest raw marginal correlations
        ax_corr = fig.add_subplot(gs[0, 1])
        draw_table(ax_corr, corr_table, "Largest marginal correlations with y", fontsize=7.3, scale_y=1.23)

        # Response histogram
        ax_y = fig.add_subplot(gs[1, 0])
        ax_y.hist(y, bins=18, color=BLUE, alpha=0.80, edgecolor="white", linewidth=0.7)
        ax_y.axvline(y.mean(), color=RED, linewidth=2.0, label=f"mean = {y.mean():.2f}")
        ax_y.set_title("Response distribution", fontsize=10.2, fontweight="bold")
        ax_y.set_xlabel("y")
        ax_y.set_ylabel("Count")
        ax_y.legend(fontsize=7.5)
        ax_y.grid(alpha=0.20, linewidth=0.5)

        # Pooled predictor histogram
        ax_x = fig.add_subplot(gs[1, 1])
        ax_x.hist(train_x.ravel(), bins=28, color=GREEN, alpha=0.72, edgecolor="white", linewidth=0.6)
        ax_x.axvline(0, color=GRAY, linestyle="--", linewidth=1.1)
        ax_x.set_title("Pooled predictor distribution", fontsize=10.2, fontweight="bold")
        ax_x.set_xlabel("All x values from the training set")
        ax_x.set_ylabel("Count")
        ax_x.grid(alpha=0.20, linewidth=0.5)

        fig.subplots_adjust(left=0.075, right=0.965, top=0.90, bottom=0.08)
        pdf.savefig(fig)
        plt.close(fig)

        #===========================================================
        #   SLIDE 2: MODEL, METHOD AND RESULTS

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 2, figure=fig, hspace=0.54, wspace=0.30)
        fig.suptitle("Model, Method and Results", fontsize=18, fontweight="bold", y=0.965)

        # Regularized regression formulas
        ax_formula = fig.add_subplot(gs[0, 0])
        text_panel(
            ax_formula,
            "Model formula",
            [
                "Ridge minimizes:",
                "(1/n) sum_i (y_i - beta_0 - x_i'beta)^2 + lambda sum_j beta_j^2",
                "",
                "Lasso uses the same squared-error loss but replaces the penalty with alpha sum_j |beta_j|.",
                "The intercept is fitted separately and is not penalized.",
            ],
            fontsize=8.2,
            width=54,
        )

        # Main modeling workflow
        ax_method = fig.add_subplot(gs[0, 1])
        text_panel(
            ax_method,
            "Method",
            [
                "Ridge: closed-form matrix solve plus explicit gradient descent.",
                "CV: manual 5-fold validation over the provided 80-lambda grid.",
                "Ridge choice: select the lambda with the smallest average MSE across all validation folds.",
                "Scaling: every ridge fold standardizes x using that fold's training rows only.",
                "Lasso: sklearn.linear_model.LassoCV with standardized predictors.",
            ],
            fontsize=8.2,
            width=55,
        )

        # Main numeric results
        ax_results = fig.add_subplot(gs[1, 0])
        draw_table(ax_results, result_table, "Numerical results", fontsize=7.2, scale_y=1.17)

        # Lasso-selected variables
        ax_selected = fig.add_subplot(gs[1, 1])
        selected_vars = ", ".join(results["lasso_selected_vars"])
        text_panel(
            ax_selected,
            "Selected variables",
            [
                f"Lasso selected {results['lasso_n_nonzero']} variables:",
                selected_vars,
                "",
                "Ridge keeps all 40 coefficients but shrinks them toward zero.",
                "Lasso sets some coefficients exactly to zero, giving a sparse model.",
            ],
            fontsize=8.0,
            width=55,
        )

        # Modeling assumptions
        ax_assumptions = fig.add_subplot(gs[2, 0])
        text_panel(
            ax_assumptions,
            "Model assumptions",
            [
                "Rows are treated as independent training examples.",
                "Train and hidden test rows are assumed to come from the same data-generating process.",
                "The conditional mean is approximated by a linear combination of standardized predictors.",
                "Correlated predictors may make lasso variable selection unstable.",
            ],
            fontsize=8.0,
            width=56,
        )

        # Library usage
        ax_libraries = fig.add_subplot(gs[2, 1])
        text_panel(
            ax_libraries,
            "Libraries",
            [
                "Ridge implementation: numpy matrix operations and loops only.",
                "Lasso implementation: sklearn.linear_model.LassoCV.",
                "Data and output: pandas, json.",
                "Slides and figures: matplotlib.",
                f"Bootstrap stability: B = {B_BOOTSTRAP} lasso refits.",
            ],
            fontsize=8.0,
            width=56,
        )

        fig.subplots_adjust(left=0.075, right=0.965, top=0.90, bottom=0.075)
        pdf.savefig(fig)
        plt.close(fig)

        #===========================================================
        #   SLIDE 3: VISUALIZATION

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 3, figure=fig, hspace=0.58, wspace=0.42)
        fig.suptitle("Visualization", fontsize=18, fontweight="bold", y=0.965)

        # Ridge cross-validation curve
        ax_cv = fig.add_subplot(gs[0, 0])
        ax_cv.semilogx(lambdas, ridge_cv, color=BLUE, linewidth=2.1)
        ax_cv.axvline(results["ridge_lambda"], color=RED, linestyle="--", linewidth=1.8)
        ax_cv.scatter([results["ridge_lambda"]], [ridge_cv.min()], color=RED, s=36, zorder=5)
        ax_cv.set_title("Cross-validation error", fontsize=9.4, fontweight="bold")
        ax_cv.set_xlabel("lambda")
        ax_cv.set_ylabel("validation MSE")
        ax_cv.legend(["ridge CV MSE", "selected lambda"], fontsize=6.3, loc="upper left")
        ax_cv.grid(alpha=0.22, linewidth=0.5)

        # Ridge and lasso coefficient paths
        ax_path = fig.add_subplot(gs[0, 1])
        path_order = np.argsort(np.abs(lasso_beta))[-6:][::-1]
        path_colors = [BLUE, RED, GREEN, PURPLE, GOLD, GRAY]
        for color, idx in zip(path_colors, path_order):
            ax_path.semilogx(lambdas, ridge_path_beta[:, idx], color=color, linewidth=1.5, label=f"{features[idx]} ridge")
        for color, idx in zip(path_colors[:3], path_order[:3]):
            ax_path.semilogx(
                lasso_alphas,
                lasso_path_beta[idx, :],
                color=color,
                linestyle=":",
                linewidth=1.7,
                label=f"{features[idx]} lasso",
            )
        ax_path.axhline(0, color="#888888", linewidth=0.7)
        ax_path.set_title("Coefficient path", fontsize=9.4, fontweight="bold")
        ax_path.set_xlabel("penalty strength")
        ax_path.set_ylabel("coefficient")
        ax_path.legend(fontsize=5.5, ncol=2, loc="best", frameon=True)
        ax_path.grid(alpha=0.22, linewidth=0.5)

        # Side-by-side final ridge and lasso coefficients
        ax_coef = fig.add_subplot(gs[0, 2])
        beta_abs_order = np.argsort(np.maximum(np.abs(ridge_beta), np.abs(lasso_beta)))[-12:]
        labels = [features[i] for i in beta_abs_order]
        x_pos = np.arange(len(labels))
        ax_coef.bar(x_pos - 0.18, ridge_beta[beta_abs_order], width=0.36, color=BLUE, alpha=0.78, label="Ridge")
        ax_coef.bar(x_pos + 0.18, lasso_beta[beta_abs_order], width=0.36, color=GOLD, alpha=0.82, label="Lasso")
        ax_coef.axhline(0, color=GRAY, linewidth=0.9)
        ax_coef.set_title("Ridge vs lasso betas", fontsize=9.4, fontweight="bold")
        ax_coef.set_xticks(x_pos)
        ax_coef.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.2)
        ax_coef.set_ylabel("raw-scale beta")
        ax_coef.legend(fontsize=6.3, loc="best")
        ax_coef.grid(axis="y", alpha=0.22, linewidth=0.5)

        # Bootstrap lasso selection frequencies
        ax_boot = fig.add_subplot(gs[1, 0])
        freq_order = np.argsort(bootstrap_freq)[::-1]
        sorted_freq = bootstrap_freq[freq_order]
        sorted_labels = [features[i] for i in freq_order]
        colors = [PURPLE if label in results["lasso_selected_vars"] else GRAY for label in sorted_labels]
        ax_boot.bar(np.arange(len(sorted_freq)), sorted_freq, color=colors, alpha=0.82, edgecolor="white", linewidth=0.35)
        ax_boot.axhline(0.5, color=RED, linestyle="--", linewidth=1.0)
        ax_boot.set_title("Bootstrap lasso stability", fontsize=9.4, fontweight="bold")
        ax_boot.set_xlabel("variables sorted by frequency")
        ax_boot.set_ylabel("frequency")
        tick_index = np.arange(0, len(sorted_labels), 4)
        ax_boot.set_xticks(tick_index)
        ax_boot.set_xticklabels([sorted_labels[i] for i in tick_index], rotation=45, ha="right", fontsize=6.0)
        ax_boot.set_ylim(0, 1.06)
        ax_boot.legend(
            handles=[
                Patch(facecolor=PURPLE, alpha=0.82, label="Selected by full-data lasso"),
                Patch(facecolor=GRAY, alpha=0.82, label="Not selected by full-data lasso"),
                Line2D([0], [0], color=RED, linestyle="--", linewidth=1.0, label="50% frequency"),
            ],
            fontsize=5.8,
            loc="upper right",
            frameon=True,
        )
        ax_boot.grid(axis="y", alpha=0.22, linewidth=0.5)

        # Gradient descent convergence to the closed-form ridge solution
        ax_gd = fig.add_subplot(gs[1, 1])
        ax_gd.semilogy(gd_iter, gd_dist, color=GREEN, linewidth=1.9, label="||beta_GD - beta_closed||")
        ax_gd.set_title("Gradient descent convergence", fontsize=9.4, fontweight="bold")
        ax_gd.set_xlabel("iteration")
        ax_gd.set_ylabel("beta distance")
        ax_gd.tick_params(axis="both", labelsize=7.0)
        ax_gd.legend(fontsize=5.9, loc="upper right")
        ax_gd.grid(alpha=0.22, linewidth=0.5)

        # Conceptual penalty-geometry cartoon
        ax_geo = fig.add_subplot(gs[1, 2])
        grid = np.linspace(-2.2, 2.2, 160)
        b1, b2 = np.meshgrid(grid, grid)
        loss = (b1 - 1.05) ** 2 + 0.55 * (b2 + 0.55) ** 2 + 0.42 * (b1 - 1.05) * (b2 + 0.55)
        ax_geo.contour(b1, b2, loss, levels=8, colors="#b8c1ca", linewidths=0.75)
        theta = np.linspace(0, 2 * np.pi, 300)
        ax_geo.plot(1.1 * np.cos(theta), 1.1 * np.sin(theta), color=BLUE, linewidth=1.8, label="ridge L2 circle")
        diamond = np.array([[0, 1.1], [1.1, 0], [0, -1.1], [-1.1, 0], [0, 1.1]])
        ax_geo.plot(diamond[:, 0], diamond[:, 1], color=GOLD, linewidth=1.8, label="lasso L1 diamond")
        ax_geo.scatter([1.1], [0.0], color=GOLD, s=25, zorder=5, label="lasso corner")
        ax_geo.scatter([0.96], [-0.38], color=BLUE, s=25, zorder=5, label="ridge smooth shrink")
        ax_geo.axhline(0, color="#dddddd", linewidth=0.7)
        ax_geo.axvline(0, color="#dddddd", linewidth=0.7)
        ax_geo.set_aspect("equal", adjustable="box")
        ax_geo.set_title("Penalty geometry", fontsize=9.4, fontweight="bold")
        ax_geo.set_xlabel("beta 1")
        ax_geo.set_ylabel("beta 2")
        ax_geo.set_xlim(-2.0, 2.0)
        ax_geo.set_ylim(-2.0, 2.0)
        ax_geo.legend(fontsize=5.7, loc="upper left", frameon=True)
        ax_geo.grid(alpha=0.12, linewidth=0.4)

        fig.subplots_adjust(left=0.060, right=0.970, top=0.90, bottom=0.085)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
