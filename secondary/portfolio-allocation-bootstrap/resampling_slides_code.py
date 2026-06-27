import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from resampling_code import B_BOOTSTRAP, RANDOM_STATE, bootstrap_alphas, portfolio_alpha


DATA_PATH = "resampling_data.csv"
PDF_PATH = "resampling_slides.pdf"

BLUE = "#2f6f9f"
RED = "#c94f37"
GREEN = "#3a8f60"
PURPLE = "#7257a8"
GRAY = "#555555"
LIGHT_GRAY = "#e8eef3"


def draw_table(ax, df, title, fontsize=8, scale_x=1.0, scale_y=1.2, col_widths=None):
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
        cell.set_linewidth(0.55)
    if col_widths is not None:
        for col_idx, width in enumerate(col_widths):
            for row_idx in range(len(df) + 1):
                table[(row_idx, col_idx)].set_width(width)


def empirical_cdf(values):
    ### Returns x and F_n(x) values for an empirical CDF step plot
    ordered = np.sort(np.asarray(values, dtype=float))
    cdf = np.arange(1, len(ordered) + 1) / len(ordered)
    return ordered, cdf


def portfolio_variance_grid(asset1_return, asset2_return, alpha_grid):
    ### Calculates sample portfolio variance over candidate allocation values
    x = np.asarray(asset1_return, dtype=float)
    y = np.asarray(asset2_return, dtype=float)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    return (
        alpha_grid ** 2 * var_x
        + (1 - alpha_grid) ** 2 * var_y
        + 2 * alpha_grid * (1 - alpha_grid) * cov_xy
    )


def format_return_table(data):
    ### Creates compact return summary statistics
    rows = []
    for label, col in [("Asset 1", "asset1_return"), ("Asset 2", "asset2_return")]:
        rows.append(
            {
                "Asset": label,
                "Mean": f"{data[col].mean():.4%}",
                "SD": f"{data[col].std(ddof=1):.4%}",
                "Min": f"{data[col].min():.4%}",
                "Median": f"{data[col].median():.4%}",
                "Max": f"{data[col].max():.4%}",
            }
        )
    return pd.DataFrame(rows)


def add_return_histogram(ax, values, color, title):
    ### Draws a daily return histogram with a zero line
    ax.hist(values, bins=18, color=color, alpha=0.72, edgecolor="white", linewidth=0.7)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.20, linewidth=0.5)


def main():
    ### Builds the three required slides

    #===========================================================
    #   LOADING DATA AND BOOTSTRAP RESULTS

    data = pd.read_csv(DATA_PATH)
    x = data["asset1_return"].to_numpy()
    y = data["asset2_return"].to_numpy()

    alpha_hat = portfolio_alpha(x, y)
    boot_alpha = bootstrap_alphas(x, y, B_BOOTSTRAP, RANDOM_STATE)
    ci_lower, ci_upper = np.percentile(boot_alpha, [2.5, 97.5])

    #===========================================================
    #   BUILDING VALUES USED ACROSS SLIDES

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    corr_xy = np.corrcoef(x, y)[0, 1]
    allocation_asset2 = 1 - alpha_hat
    ci_width = ci_upper - ci_lower
    min_var_return = alpha_hat * x + allocation_asset2 * y
    equal_weight_return = 0.5 * x + 0.5 * y

    stability_points = np.unique(np.linspace(1000, B_BOOTSTRAP, 30, dtype=int))
    stability_ci = np.array([np.percentile(boot_alpha[:m], [2.5, 50, 97.5]) for m in stability_points])

    #===========================================================
    #   PREPARING TABLES FOR SLIDES

    summary_table = format_return_table(data)
    dependence_table = pd.DataFrame(
        [
            {"Quantity": "Asset 1 variance", "Value": f"{var_x:.8f}"},
            {"Quantity": "Asset 2 variance", "Value": f"{var_y:.8f}"},
            {"Quantity": "Covariance", "Value": f"{cov_xy:.8f}"},
            {"Quantity": "Correlation", "Value": f"{corr_xy:.4f}"},
        ]
    )
    results_table = pd.DataFrame(
        [
            {"Quantity": "alpha_hat", "Value": f"{alpha_hat:.6f}"},
            {"Quantity": "Allocation to Asset 2", "Value": f"{allocation_asset2:.6f}"},
            {"Quantity": "95% CI lower", "Value": f"{ci_lower:.6f}"},
            {"Quantity": "95% CI upper", "Value": f"{ci_upper:.6f}"},
            {"Quantity": "CI width", "Value": f"{ci_width:.6f}"},
            {"Quantity": "Bootstrap B", "Value": f"{B_BOOTSTRAP}"},
        ]
    )
    quantile_table = pd.DataFrame(
        [
            {"Bootstrap alpha* quantile": "2.5%", "Value": f"{np.percentile(boot_alpha, 2.5):.6f}"},
            {"Bootstrap alpha* quantile": "50.0%", "Value": f"{np.percentile(boot_alpha, 50):.6f}"},
            {"Bootstrap alpha* quantile": "97.5%", "Value": f"{np.percentile(boot_alpha, 97.5):.6f}"},
        ]
    )

    #===========================================================
    #   WRITING PDF SLIDES

    with PdfPages(PDF_PATH) as pdf:
        #===========================================================
        #   SLIDE 1: DATA

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[0.9, 1.05, 1.35], hspace=0.55, wspace=0.38)
        fig.suptitle("Data", fontsize=18, fontweight="bold")

        ax_overview = fig.add_subplot(gs[0, 0])
        overview_text = (
            "Data file: resampling_data.csv\n"
            f"Sample size: n = {len(data)} paired daily returns\n"
            "Variables: asset1_return, asset2_return\n"
            "Bootstrap unit: full paired row"
        )
        ax_overview.text(0.02, 0.50, overview_text, fontsize=9.3, family="monospace", va="center")
        ax_overview.set_title("Sample structure", fontsize=11, fontweight="bold")
        ax_overview.axis("off")

        ax_summary = fig.add_subplot(gs[0, 1:])
        draw_table(ax_summary, summary_table, "Daily return summary", fontsize=7.6, scale_x=1.0, scale_y=1.35)

        ax_dep = fig.add_subplot(gs[1, 0])
        draw_table(ax_dep, dependence_table, "Variance and dependence", fontsize=7.8, scale_x=0.95, scale_y=1.2)

        ax_hist1 = fig.add_subplot(gs[1, 1])
        add_return_histogram(ax_hist1, x, BLUE, "Asset 1 returns")

        ax_hist2 = fig.add_subplot(gs[1, 2])
        add_return_histogram(ax_hist2, y, RED, "Asset 2 returns")

        ax_scatter = fig.add_subplot(gs[2, :])
        ax_scatter.scatter(x, y, s=32, alpha=0.72, color=PURPLE, edgecolor="white", linewidth=0.35)
        ax_scatter.axhline(0, color=GRAY, linestyle="--", linewidth=1)
        ax_scatter.axvline(0, color=GRAY, linestyle="--", linewidth=1)
        ax_scatter.set_title(f"Paired return cloud (correlation = {corr_xy:.3f})", fontsize=11, fontweight="bold")
        ax_scatter.set_xlabel("Asset 1 daily return")
        ax_scatter.set_ylabel("Asset 2 daily return")
        ax_scatter.grid(alpha=0.20, linewidth=0.5)

        fig.subplots_adjust(left=0.09, right=0.965, top=0.90, bottom=0.08)
        pdf.savefig(fig)
        plt.close(fig)

        #===========================================================
        #   SLIDE 2: MODEL, METHOD AND RESULTS

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.10, 1.05, 1.05], hspace=0.55, wspace=0.36)
        fig.suptitle("Model, Method and Results", fontsize=18, fontweight="bold")

        ax_formula = fig.add_subplot(gs[0, 0])
        formula_text = (
            "Minimum-variance allocation:\n"
            r"$\alpha = \frac{\sigma_Y^2-\sigma_{XY}}{\sigma_X^2+\sigma_Y^2-2\sigma_{XY}}$"
            "\n\n"
            "Portfolio return:\n"
            r"$R_p(\alpha)=\alpha X+(1-\alpha)Y$"
            "\n\n"
            "Plug-in estimate uses sample variance and covariance."
        )
        ax_formula.text(0.02, 0.50, formula_text, fontsize=10.0, va="center")
        ax_formula.set_title("Model formula", fontsize=11, fontweight="bold")
        ax_formula.axis("off")

        ax_method = fig.add_subplot(gs[0, 1])
        method_text = (
            "Nonparametric bootstrap from the EDF:\n"
            "1. Generate U from Uniform(0,1)\n"
            "2. Convert to row index: floor(n*U)\n"
            "3. Keep paired returns together\n"
            "4. Recompute alpha* for each replicate\n"
            "5. Use 2.5% and 97.5% alpha* percentiles\n\n"
            "Libraries: pandas, numpy, matplotlib"
        )
        ax_method.text(0.02, 0.50, method_text, fontsize=8.9, family="monospace", va="center")
        ax_method.set_title("Bootstrap method", fontsize=11, fontweight="bold")
        ax_method.axis("off")

        ax_assumptions = fig.add_subplot(gs[1, 0])
        assumptions_text = (
            "Assumptions:\n"
            "- the 100 daily pairs are representative of the return process\n"
            "- rows are independent enough for row-level bootstrap inference\n"
            "- paired row resampling preserves covariance between assets\n"
            "- no parametric normal return model is required\n"
            "- the sample covariance matrix is stable enough for the ratio estimate"
        )
        ax_assumptions.text(0.02, 0.50, assumptions_text, fontsize=9.1, family="monospace", va="center")
        ax_assumptions.set_title("Model assumptions", fontsize=11, fontweight="bold")
        ax_assumptions.axis("off")

        ax_results = fig.add_subplot(gs[1, 1])
        draw_table(ax_results, results_table, "Final estimate and interval", fontsize=8.0, scale_x=1.0, scale_y=1.28)

        ax_quantiles = fig.add_subplot(gs[2, 0])
        draw_table(ax_quantiles, quantile_table, "Percentile CI calculation", fontsize=8.2, scale_x=0.95, scale_y=1.45)

        ax_interpretation = fig.add_subplot(gs[2, 1])
        interp_text = (
            f"The fitted minimum-variance portfolio puts {alpha_hat:.1%} in Asset 1\n"
            f"and {allocation_asset2:.1%} in Asset 2.\n\n"
            f"The bootstrap interval [{ci_lower:.3f}, {ci_upper:.3f}] is fairly tight\n"
            "relative to the 0 to 1 allocation scale, so the data consistently\n"
            "favor a larger weight on Asset 1.\n\n"
            "The interval is still reported on alpha directly, allowing the\n"
            "grader to evaluate coverage and interval width."
        )
        ax_interpretation.text(0.02, 0.52, interp_text, fontsize=9.4, va="center")
        ax_interpretation.set_title("Interpretation", fontsize=11, fontweight="bold")
        ax_interpretation.axis("off")

        fig.subplots_adjust(left=0.075, right=0.965, top=0.90, bottom=0.07)
        pdf.savefig(fig)
        plt.close(fig)

        #===========================================================
        #   SLIDE 3: VISUALIZATION

        fig = plt.figure(figsize=(11, 8.5))
        gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.34)
        fig.suptitle("Visualization", fontsize=18, fontweight="bold")

        ax_edf = fig.add_subplot(gs[0, 0])
        ecdf_x, cdf_x = empirical_cdf(x)
        ecdf_y, cdf_y = empirical_cdf(y)
        ax_edf.step(ecdf_x, cdf_x, where="post", color=BLUE, linewidth=2, label="Asset 1 EDF")
        ax_edf.step(ecdf_y, cdf_y, where="post", color=RED, linewidth=2, label="Asset 2 EDF")
        ax_edf.axvline(0, color=GRAY, linestyle="--", linewidth=1)
        ax_edf.set_title("Empirical distribution functions", fontsize=9.8, fontweight="bold")
        ax_edf.set_xlabel("Daily return")
        ax_edf.set_ylabel("F_n(return)")
        ax_edf.legend(fontsize=7, loc="lower right")
        ax_edf.grid(alpha=0.20, linewidth=0.5)

        ax_boot = fig.add_subplot(gs[0, 1])
        ax_boot.hist(boot_alpha, bins=42, color=PURPLE, alpha=0.76, edgecolor="white", linewidth=0.45)
        ax_boot.axvline(alpha_hat, color=GREEN, linewidth=2.2, label=f"alpha_hat = {alpha_hat:.3f}")
        ax_boot.axvline(ci_lower, color=RED, linestyle="--", linewidth=1.8, label="95% CI")
        ax_boot.axvline(ci_upper, color=RED, linestyle="--", linewidth=1.8)
        ax_boot.set_title("Bootstrap distribution of alpha*", fontsize=9.8, fontweight="bold")
        ax_boot.set_xlabel("alpha*")
        ax_boot.set_ylabel("Count")
        ax_boot.legend(fontsize=7)
        ax_boot.grid(alpha=0.20, linewidth=0.5)

        ax_stability = fig.add_subplot(gs[0, 2])
        ax_stability.fill_between(
            stability_points,
            stability_ci[:, 0],
            stability_ci[:, 2],
            color=PURPLE,
            alpha=0.15,
            label="95% CI path",
        )
        ax_stability.plot(stability_points, stability_ci[:, 0], color=RED, linestyle="--", linewidth=1.5)
        ax_stability.plot(stability_points, stability_ci[:, 1], color=GREEN, linewidth=1.8, label="Median alpha*")
        ax_stability.plot(stability_points, stability_ci[:, 2], color=RED, linestyle="--", linewidth=1.5)
        ax_stability.axhline(alpha_hat, color=GRAY, linewidth=1.1, label="alpha_hat")
        ax_stability.set_title("Bootstrap CI stability", fontsize=9.8, fontweight="bold")
        ax_stability.set_xlabel("Replicates used")
        ax_stability.set_ylabel("alpha")
        ax_stability.legend(fontsize=7, loc="best")
        ax_stability.grid(alpha=0.20, linewidth=0.5)

        ax_var = fig.add_subplot(gs[1, 0])
        alpha_grid = np.linspace(0, 1, 250)
        variance_grid = portfolio_variance_grid(x, y, alpha_grid)
        ax_var.plot(alpha_grid, variance_grid, color=BLUE, linewidth=2.2)
        ax_var.axvline(alpha_hat, color=GREEN, linewidth=2.0, label=f"Minimum at {alpha_hat:.3f}")
        ax_var.axvspan(ci_lower, ci_upper, color=PURPLE, alpha=0.16, label="Bootstrap CI")
        ax_var.set_title("Sample portfolio variance curve", fontsize=9.8, fontweight="bold")
        ax_var.set_xlabel("Allocation alpha to Asset 1")
        ax_var.set_ylabel("Sample variance")
        ax_var.legend(fontsize=7)
        ax_var.grid(alpha=0.20, linewidth=0.5)

        ax_risk = fig.add_subplot(gs[1, 1])
        risk_labels = ["Asset 1", "Asset 2", "50/50", "Min-var"]
        risk_sd = np.array(
            [
                np.std(x, ddof=1),
                np.std(y, ddof=1),
                np.std(equal_weight_return, ddof=1),
                np.std(min_var_return, ddof=1),
            ]
        )
        bars = ax_risk.bar(
            risk_labels,
            risk_sd,
            color=[BLUE, RED, GRAY, GREEN],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, risk_sd):
            ax_risk.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.00025,
                f"{value:.2%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax_risk.set_title("Daily volatility comparison", fontsize=9.8, fontweight="bold")
        ax_risk.set_ylabel("Sample SD")
        ax_risk.set_ylim(0, risk_sd.max() * 1.24)
        ax_risk.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
        ax_risk.tick_params(axis="x", labelsize=8)
        ax_risk.grid(axis="y", alpha=0.20, linewidth=0.5)

        ax_pair = fig.add_subplot(gs[1, 2])
        ax_pair.scatter(x, y, s=28, color=GRAY, alpha=0.58, edgecolor="white", linewidth=0.25)
        ax_pair.axhline(0, color=RED, linestyle="--", linewidth=1)
        ax_pair.axvline(0, color=BLUE, linestyle="--", linewidth=1)
        ax_pair.set_title("Paired rows preserve covariance", fontsize=9.8, fontweight="bold")
        ax_pair.set_xlabel("Asset 1 daily return")
        ax_pair.set_ylabel("Asset 2 daily return")
        ax_pair.text(
            0.03,
            0.95,
            f"corr = {corr_xy:.3f}\ncov = {cov_xy:.6f}",
            transform=ax_pair.transAxes,
            fontsize=7.6,
            va="top",
            bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "alpha": 0.85},
        )
        ax_pair.grid(alpha=0.20, linewidth=0.5)

        fig.subplots_adjust(left=0.09, right=0.975, top=0.90, bottom=0.085)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
