import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import stats

PREDICTORS = ["x1", "x2", "x3", "x4", "x5"]
ADDITIVE_MODEL = ["x1", "x3", "x5"]
CENTERED_NO_MAIN_X4 = ["x1", "x3_c", "x5", "x3c_x4c"]
CENTERED_HIERARCHICAL = ["x1", "x3_c", "x4_c", "x5", "x3c_x4c"]
TRAIN_PATH = "regression_train.csv"
TEST_PATH = "regression_test.csv"
PDF_PATH = "regression_slides.pdf"


def add_centered_terms(df: pd.DataFrame, x3_mean: float, x4_mean: float) -> pd.DataFrame:
    ### Adds centered covariates and their interaction
    out = df.copy()
    out["x3_c"] = out["x3"] - x3_mean
    out["x4_c"] = out["x4"] - x4_mean
    out["x3c_x4c"] = out["x3_c"] * out["x4_c"]
    return out


def fit_ols(df: pd.DataFrame, predictors: list[str]):
    ### This function returns regression parameters
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[predictors].to_numpy()])  # defining covariate matrix
    coef = np.linalg.lstsq(X, y, rcond=None)[0]  # getting regression coefs
    fitted = X @ coef
    resid = y - fitted
    n = len(df)
    p = X.shape[1]  # includes intercept
    dof = n - p

    sigma = float(np.sqrt((resid @ resid) / dof))
    xtx_inv = np.linalg.inv(X.T @ X)  # variance-covariance matrix
    se = sigma * np.sqrt(np.diag(xtx_inv))
    t_stat = coef / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))  # two-sided p values

    nll = float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + resid ** 2 / (2 * sigma ** 2)))
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof
    f_stat = ((ss_tot - ss_res) / (p - 1)) / (ss_res / dof)
    f_p = 1 - stats.f.cdf(f_stat, p - 1, dof)

    # AIC and BIC
    loglik = -n / 2 * (np.log(2 * np.pi) + 1 + np.log(ss_res / n))
    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + np.log(n) * p

    return {
        "predictors": predictors,
        "coef": coef,
        "fitted": fitted,
        "resid": resid,
        "sigma": sigma,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "nll": nll,
        "r2": r2,
        "adj_r2": adj_r2,
        "f_stat": f_stat,
        "f_p": f_p,
        "aic": aic,
        "bic": bic,
    }


def formula_lines(model: dict):
    ### Writes model formula on two lines to fit the slide block
    names = ["intercept"] + model["predictors"]
    parts = [
        f"{coef:.3f}" if name == "intercept" else f"({coef:.3f})*{name}"
        for name, coef in zip(names, model["coef"])
    ]
    first_line = "y_hat = " + " + ".join(parts[:3])
    second_line = "        + " + " + ".join(parts[3:])
    return [first_line, second_line]


raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
x3_mean = raw_train["x3"].mean()
x4_mean = raw_train["x4"].mean()
train = add_centered_terms(raw_train, x3_mean, x4_mean)
test = add_centered_terms(raw_test, x3_mean, x4_mean)
corr = raw_train.corr()

full_main_model = fit_ols(raw_train, PREDICTORS)
additive_fit = fit_ols(raw_train, ADDITIVE_MODEL)
centered_no_main_fit = fit_ols(train, CENTERED_NO_MAIN_X4)
model = fit_ols(train, CENTERED_HIERARCHICAL)
selected = model["predictors"]
interaction_p = model["p_value"][-1]
x4_p = model["p_value"][3]

#===========================================================
#   CREATING SLIDES

# SLIDE 1 - Parameters & Plots

with PdfPages(PDF_PATH) as pdf:
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Data", fontsize=18, fontweight="bold")

    # Sample size
    info = (
        f"Training observations: {len(train)}\n"
        f"Test observations: {len(test)}\n"
        f"Predictors: {', '.join(PREDICTORS)}\n"
        "Response: y (continuous)"
    )
    axes[0, 0].text(0.05, 0.55, info, fontsize=10, family="monospace", transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("Sample Size & Structure")
    axes[0, 0].axis("off")

    # Summary stats
    summary = raw_train.describe().loc[["mean", "std", "min", "max"]].round(2)
    axes[0, 1].text(0.02, 0.45, summary.to_string(), fontsize=7, family="monospace", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Summary Statistics")
    axes[0, 1].axis("off")

    # Correlation
    corr_y = corr["y"].drop("y").sort_values()
    colors = ["#d95f02" if value < 0 else "#1b9e77" for value in corr_y.values]
    axes[0, 2].barh(corr_y.index, corr_y.values, color=colors)
    axes[0, 2].axvline(0, color="black", linewidth=1)
    axes[0, 2].set_xlim(-1, 1)
    axes[0, 2].set_title("Correlation With y")

    # Histograms
    for ax, col in zip(axes[1], ["x3", "x4", "y"]):
        ax.hist(raw_train[col], bins=20, color="#4c78a8", edgecolor="white")
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

#========================================================
# SLIDE 2 - MODELLING & RESULTS

    # Header
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Model, Method and Results", fontsize=18, fontweight="bold")

    # Model and assumptions
    assumptions = (
        "Model:\n"
        "  y = beta_0 + sum(beta_j * x_j) + epsilon\n"
        "  use centered x3, x4, and x3*x4\n"
        "  epsilon ~ N(0, sigma^2)\n\n"
        "Assumptions:\n"
        "  1. linear mean structure\n"
        "  2. independent observations\n"
        "  3. constant error variance\n"
        "  4. approximately normal residuals"
    )
    axes[0, 0].text(0.03, 0.5, assumptions, fontsize=8.7, family="monospace", transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("Model & Assumptions")
    axes[0, 0].axis("off")

    # Method
    method = (
        "Method:\n"
        "  ordinary least squares\n"
        "  center x3 and x4 before interaction\n\n"
        "Packages:\n"
        "  pandas\n"
        "  numpy\n"
        "  scipy.stats\n"
        "  matplotlib"
    )
    axes[0, 1].text(0.05, 0.5, method, fontsize=8.7, family="monospace", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("Method")
    axes[0, 1].axis("off")

    # Variable selection
    selection_lines = ["Full-model p-values:"]
    for name, p_value in zip(PREDICTORS, full_main_model["p_value"][1:]):
        tag = "keep" if p_value < 0.05 else "drop"
        selection_lines.append(f"  {name:>7s}: p={p_value:.4f} ({tag})")
    selection_lines.append(f"  {'x3_c*x4_c':>7s}: p={interaction_p:.4f} (keep)")
    selection_lines.append("")
    selection_lines.append("x4_c kept by hierarchy")
    selection_lines.append(f"Selected: {', '.join(selected)}")
    axes[0, 2].text(0.03, 0.5, "\n".join(selection_lines), fontsize=7.2, family="monospace", transform=axes[0, 2].transAxes)
    axes[0, 2].set_title("Variable Selection")
    axes[0, 2].axis("off")

    # Selected model
    coef_lines = formula_lines(model) + ["", f"sigma_hat = {model['sigma']:.4f}", "", "Coefficients:"]
    for name, coef, se in zip(["intercept"] + selected, model["coef"], model["se"]):
        coef_lines.append(f"  {name}: {coef:.4f} (SE {se:.4f})")
    axes[1, 0].text(0.03, 0.56, "\n".join(coef_lines), fontsize=6.8, family="monospace", transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Estimated Model")
    axes[1, 0].axis("off")

    # Coefficient tests
    table_lines = ["name        coef      t       p", "--------------------------------"]
    for name, coef, t_stat, p_value in zip(["intercept"] + selected, model["coef"], model["t_stat"], model["p_value"]):
        table_lines.append(f"{name:<10}{coef:>7.3f}{t_stat:>8.2f}{p_value:>9.4f}")
    axes[1, 1].text(0.03, 0.56, "\n".join(table_lines), fontsize=6.9, family="monospace", transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Coefficient Tests")
    axes[1, 1].axis("off")

    # Model comparison summary
    comparison_lines = [
        "Model comparison:",
        "spec             adjR2    AIC     BIC",
        f"additive        {additive_fit['adj_r2']:>6.3f}{additive_fit['aic']:>8.1f}{additive_fit['bic']:>8.1f}",
        f"centered no x4  {centered_no_main_fit['adj_r2']:>6.3f}{centered_no_main_fit['aic']:>8.1f}{centered_no_main_fit['bic']:>8.1f}",
        f"hierarchical*   {model['adj_r2']:>6.3f}{model['aic']:>8.1f}{model['bic']:>8.1f}",
        "",
        f"x4_c p-value:   {x4_p:.4f}",
        "kept to preserve hierarchy",
        "with centered x3_c*x4_c",
    ]
    axes[1, 2].text(0.02, 0.56, "\n".join(comparison_lines), fontsize=6.6, family="monospace", transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Fit Summary")
    axes[1, 2].axis("off")

#===================================================================
# SLIDE 3 - VISUALIZATION

    # Header
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
    fig.suptitle("Visualization", fontsize=18, fontweight="bold")

    # x3 scatter plot
    axes[0, 0].scatter(raw_train["x3"], raw_train["y"], alpha=0.7, color="#4c78a8")
    x_grid = np.linspace(raw_train["x3"].min(), raw_train["x3"].max(), 200)
    slope = np.polyfit(raw_train["x3"], raw_train["y"], 1)
    axes[0, 0].plot(x_grid, slope[0] * x_grid + slope[1], color="#d62728", linewidth=2)
    axes[0, 0].set_title("y vs x3")
    axes[0, 0].set_xlabel("x3")
    axes[0, 0].set_ylabel("y")

    # centered interaction scatter plot
    axes[0, 1].scatter(train["x3c_x4c"], train["y"], alpha=0.7, color="#59a14f")
    x_grid = np.linspace(train["x3c_x4c"].min(), train["x3c_x4c"].max(), 200)
    slope = np.polyfit(train["x3c_x4c"], train["y"], 1)
    axes[0, 1].plot(x_grid, slope[0] * x_grid + slope[1], color="#d62728", linewidth=2)
    axes[0, 1].set_title("y vs x3_c*x4_c")
    axes[0, 1].set_xlabel("x3_c*x4_c")
    axes[0, 1].set_ylabel("y")

    # residuals vs fitted
    axes[0, 2].scatter(model["fitted"], model["resid"], alpha=0.7, color="#f28e2b")
    axes[0, 2].axhline(0, color="black", linewidth=1)
    axes[0, 2].set_title("Residuals vs Fitted")
    axes[0, 2].set_xlabel("Fitted")
    axes[0, 2].set_ylabel("Residual")

    # QQ plot
    (osm, osr), (slope, intercept, r) = stats.probplot(model["resid"], dist="norm")
    axes[1, 0].scatter(osm, osr, s=12, alpha=0.35)
    axes[1, 0].plot(osm, slope * osm + intercept)
    axes[1, 0].set_title("QQ Plot")
    axes[1, 0].set_xlabel("Theoretical Quantiles")
    axes[1, 0].set_ylabel("Sample Quantiles")

    # predicted vs actual
    axes[1, 1].scatter(model["fitted"], raw_train["y"], alpha=0.7, color="#e15759")
    limits = [min(model["fitted"].min(), raw_train["y"].min()), max(model["fitted"].max(), raw_train["y"].max())]
    axes[1, 1].plot(limits, limits, color="black", linestyle="--")
    axes[1, 1].set_title("Predicted vs Actual")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")

    # coefficient size
    axes[1, 2].bar(selected, np.abs(model["coef"][1:]), color="#b07aa1")
    axes[1, 2].set_title("Coefficient Size")
    axes[1, 2].set_xlabel("Predictor")
    axes[1, 2].set_ylabel("|Estimate|")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

print(f"selected predictors: {selected}")
print(f"saved: {PDF_PATH}")
