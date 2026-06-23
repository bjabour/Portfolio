import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


TRAIN_PATH = "classification_train.csv"
TEST_PATH = "classification_test.csv"
PDF_PATH = "classification_slides.pdf"
RANDOM_STATE = 9618

BASE_FEATURES = ["x1", "x2", "x3"]
LOGISTIC_TERMS = ["x1", "x1_sq", "x2_x3"]
HIERARCHICAL_LOGISTIC_TERMS = ["x1", "x2", "x3", "x1_sq", "x2_x3"]
FULL_LOGISTIC_TERMS = [
    "x1", "x2", "x3", "x1_sq", "x2_sq", "x3_sq", "x1_x2", "x1_x3", "x2_x3"
]

BLUE = "#2f6f9f"
RED = "#c94f37"
GREEN = "#3a8f60"
PURPLE = "#7257a8"
GRAY = "#555555"


def add_model_terms(df: pd.DataFrame) -> pd.DataFrame:
    ### Adds squared terms and interactions used for model comparison
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_sq"] = out["x2"] ** 2
    out["x3_sq"] = out["x3"] ** 2
    out["x1_x2"] = out["x1"] * out["x2"]
    out["x1_x3"] = out["x1"] * out["x3"]
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


def design_matrix(df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    ### Builds the logistic regression design matrix with an intercept
    return sm.add_constant(df[terms], has_constant="add").to_numpy()


def fit_logistic_glm(df: pd.DataFrame, terms: list[str]):
    ### Fits logistic regression
    return sm.GLM(df["y"].to_numpy(), design_matrix(df, terms), family=sm.families.Binomial()).fit(maxiter=200)


def metrics(y_true, prob_class1):
    ### Calculates validation metrics from predicted class-1 probabilities
    pred = (prob_class1 >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Log-loss": log_loss(y_true, np.clip(prob_class1, 1e-12, 1 - 1e-12)),
        "AUC": roc_auc_score(y_true, prob_class1),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


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
            cell.set_facecolor("#e8eef3")
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.55)
    if col_widths is not None:
        for col_idx, width in enumerate(col_widths):
            for row_idx in range(len(df) + 1):
                table[(row_idx, col_idx)].set_width(width)


def draw_confusion(ax, cm, title):
    ### Draws a 2 by 2 confusion matrix
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=8)
    ax.set_yticklabels(["0", "1"], fontsize=8)
    ax.set_xlabel("Pred", fontsize=8, labelpad=1)
    ax.set_ylabel("True", fontsize=8, labelpad=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")


def draw_covariance_ellipse(ax, subset, x_col, y_col, color, n_std=1.6):
    ### Draws covariance ellipse to show each class cloud shape
    coords = subset[[x_col, y_col]].dropna().to_numpy()
    if len(coords) < 3:
        return
    mean = coords.mean(axis=0)
    cov = np.cov(coords, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ellipse = Ellipse(
        mean,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(ellipse)


def plot_hist_by_class(ax, df, feature, title=None):
    ### Plots overlapping histograms split by the binary class
    ax.hist(df.loc[df["y"] == 0, feature], bins=18, alpha=0.58, color=BLUE, label="y=0")
    ax.hist(df.loc[df["y"] == 1, feature], bins=18, alpha=0.58, color=RED, label="y=1")
    ax.set_title(title or f"Histogram of {feature}", fontsize=10.5, fontweight="bold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)


def validation_model_selection(train_df, valid_df):
    ### Compares logistic candidates and QDA on the validation split
    candidates = {
        "Full logistic": ("logistic", FULL_LOGISTIC_TERMS, "all squares + interactions"),
        "Chosen logistic": ("logistic", LOGISTIC_TERMS, "x1, x1^2, x2*x3"),
        "Hierarchical logistic": ("logistic", HIERARCHICAL_LOGISTIC_TERMS, "x1,x2,x3,x1^2,x2*x3"),
        "QDA": ("qda", BASE_FEATURES, "class-specific covariance"),
    }
    rows = []
    for name, (model_type, terms, description) in candidates.items():
        if model_type == "logistic":
            fit = fit_logistic_glm(train_df, terms)
            prob = fit.predict(design_matrix(valid_df, terms))
        else:
            fit = QuadraticDiscriminantAnalysis()
            fit.fit(train_df[BASE_FEATURES], train_df["y"])
            prob = fit.predict_proba(valid_df[BASE_FEATURES])[:, 1]
        model_metrics = metrics(valid_df["y"], prob)
        rows.append(
            {
                "Model": name,
                "Specification": description,
                "Log-loss": f"{model_metrics['Log-loss']:.3f}",
                "AUC": f"{model_metrics['AUC']:.3f}",
                "Accuracy": f"{model_metrics['Accuracy']:.3f}",
                "Precision": f"{model_metrics['Precision']:.3f}",
                "Recall": f"{model_metrics['Recall']:.3f}",
                "F1": f"{model_metrics['F1']:.3f}",
                "FP/FN": f"{model_metrics['FP']}/{model_metrics['FN']}",
            }
        )
    return pd.DataFrame(rows)


def predicted_probability_grid(fit, train, x1_value=None):
    ### Creates an x2 by x3 grid for the fitted probability surface
    x2_grid = np.linspace(train["x2"].quantile(0.02), train["x2"].quantile(0.98), 120)
    x3_grid = np.linspace(train["x3"].quantile(0.02), train["x3"].quantile(0.98), 120)
    xx, yy = np.meshgrid(x2_grid, x3_grid)
    if x1_value is None:
        x1_value = train["x1"].median()
    grid = pd.DataFrame({"x1": x1_value, "x2": xx.ravel(), "x3": yy.ravel()})
    grid = add_model_terms(grid)
    prob = fit.predict(design_matrix(grid, LOGISTIC_TERMS)).reshape(xx.shape)
    return xx, yy, prob


def fitted_vs_y_grid(fit, train):
    ### Creates fitted probabilities over an x1 grid
    x1_grid = np.linspace(train["x1"].quantile(0.02), train["x1"].quantile(0.98), 160)
    grid = pd.DataFrame(
        {
            "x1": x1_grid,
            "x2": train["x2"].median(),
            "x3": train["x3"].median(),
        }
    )
    grid = add_model_terms(grid)
    prob = fit.predict(design_matrix(grid, LOGISTIC_TERMS))
    return x1_grid, prob


def binned_residuals(df, feature, n_bins=10):
    ### Averages residuals inside quantile bins for diagnostic plots
    tmp = df[[feature, "residual"]].copy()
    tmp["bin"] = pd.qcut(tmp[feature], q=n_bins, duplicates="drop")
    grouped = tmp.groupby("bin", observed=True).agg(
        x_mid=(feature, "mean"),
        resid_mean=("residual", "mean"),
    )
    return grouped["x_mid"].to_numpy(), grouped["resid_mean"].to_numpy()


def reliability_points(df, n_bins=10):
    ### Groups fitted probabilities into bins for calibration checking
    tmp = df[["fitted", "y"]].copy()
    tmp["bin"] = pd.qcut(tmp["fitted"], q=n_bins, duplicates="drop")
    grouped = tmp.groupby("bin", observed=True).agg(
        pred_mean=("fitted", "mean"),
        observed_rate=("y", "mean"),
        count=("y", "size"),
    )
    return grouped


#===========================================================
#   LOADING DATA AND ADDING MODEL TERMS

raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
train = add_model_terms(raw_train)
test = add_model_terms(raw_test)

#===========================================================
#   VALIDATION COMPARISON

train_part, valid_part = train_test_split(
    train,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=train["y"],
)

logistic_valid_fit = fit_logistic_glm(train_part, LOGISTIC_TERMS)
valid_logistic_prob = logistic_valid_fit.predict(design_matrix(valid_part, LOGISTIC_TERMS))

qda = QuadraticDiscriminantAnalysis()
qda.fit(train_part[BASE_FEATURES], train_part["y"])
valid_qda_prob = qda.predict_proba(valid_part[BASE_FEATURES])[:, 1]

logistic_metrics = metrics(valid_part["y"], valid_logistic_prob)
qda_metrics = metrics(valid_part["y"], valid_qda_prob)
metric_table = pd.DataFrame(
    [
        {"Method": "Logistic", **logistic_metrics},
        {"Method": "QDA", **qda_metrics},
    ]
)
metric_cols = ["Accuracy", "Log-loss", "AUC", "Precision", "Recall", "F1"]
for col in metric_cols:
    metric_table[col] = metric_table[col].map(lambda v: f"{v:.3f}")
metric_table = metric_table[["Method"] + metric_cols + ["TN", "FP", "FN", "TP"]]

selection_table = validation_model_selection(train_part, valid_part)
final_fit = fit_logistic_glm(train, LOGISTIC_TERMS)
train["fitted"] = final_fit.predict(design_matrix(train, LOGISTIC_TERMS))
train["residual"] = train["y"] - train["fitted"]
coef_table = pd.DataFrame(
    {
        "Term": ["Intercept", "x1", "x1^2", "x2*x3"],
        "Coef": [f"{value:.3f}" for value in final_fit.params],
        "p": [f"{value:.3f}" for value in final_fit.pvalues],
    }
)

#===========================================================
#   CREATING SLIDES

with PdfPages(PDF_PATH) as pdf:
    # Slide 1: Data
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.95, 1, 1], hspace=0.65, wspace=0.38)
    fig.suptitle("Data", fontsize=18, fontweight="bold")

    class_counts = raw_train["y"].value_counts().sort_index()
    ax_info = fig.add_subplot(gs[0, 0])
    info = (
        f"Training observations: {len(raw_train)}\n"
        f"Test observations: {len(raw_test)}\n"
        f"Class 0: {class_counts[0]} ({class_counts[0] / len(raw_train):.1%})\n"
        f"Class 1: {class_counts[1]} ({class_counts[1] / len(raw_train):.1%})\n"
        f"Predictors: {', '.join(BASE_FEATURES)}"
    )
    ax_info.text(0.02, 0.50, info, fontsize=9.6, family="monospace", va="center")
    ax_info.set_title("Sample sizes", fontsize=11, fontweight="bold")
    ax_info.axis("off")

    stats_rows = []
    for label, subset in [
        ("Total", raw_train),
        ("y=0", raw_train[raw_train["y"] == 0]),
        ("y=1", raw_train[raw_train["y"] == 1]),
    ]:
        row = {"Group": label, "n": len(subset)}
        for feature in BASE_FEATURES:
            row[f"{feature} mean"] = f"{subset[feature].mean():.2f}"
            row[f"{feature} sd"] = f"{subset[feature].std():.2f}"
        stats_rows.append(row)
    stats_table = pd.DataFrame(stats_rows)
    draw_table(
        fig.add_subplot(gs[0, 1:]),
        stats_table,
        "Summary statistics by class and total",
        fontsize=7.1,
        scale_x=1.02,
        scale_y=1.2,
    )

    hist_features = [
        ("x1", "Histogram of x1"),
        ("x2", "Histogram of x2"),
        ("x3", "Histogram of x3"),
        ("x1_sq", "Histogram of x1^2"),
        ("x2_x3", "Histogram of x2*x3"),
    ]
    hist_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    for ax, (feature, title) in zip(hist_axes, hist_features):
        plot_hist_by_class(ax, train, feature, title=title)

    corr = raw_train[BASE_FEATURES + ["y"]].corr()
    ax_corr = fig.add_subplot(gs[2, 2])
    im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(4))
    ax_corr.set_yticks(range(4))
    ax_corr.set_xticklabels(corr.columns, fontsize=8)
    ax_corr.set_yticklabels(corr.index, fontsize=8)
    ax_corr.set_title("Correlation heatmap", fontsize=10.5, fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax_corr.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # Slide 2: Model, method, and results
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 0.95, 1.15], hspace=0.62, wspace=0.33)
    fig.suptitle("Model, Method and Results", fontsize=18, fontweight="bold")

    ax_method = fig.add_subplot(gs[0, 0])
    method_text = (
        "Logistic model:\n"
        r"$\mathrm{logit}\{P(Y=1\mid x)\} = \beta_0 + \beta_1x_1+ \beta_2x_1^2 + \beta_3(x_2x_3)$" "\n\n"
        "QDA model:\n"
        r"$\delta_k(x) = -\frac{1}{2}\log|\Sigma_k|-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+ \log(\pi_k)$" "\n"
        r"Classify to $\mathrm{argmax}_k\ \delta_k(x)$." "\n\n"
        "Libraries: statsmodels, scikit-learn, pandas, numpy, matplotlib\n"
    )
    ax_method.text(0.02, 0.50, method_text, fontsize=8.8, va="center")
    ax_method.set_title("Model formula and libraries", fontsize=11, fontweight="bold")
    ax_method.axis("off")

    ax_assumptions = fig.add_subplot(gs[0, 1])
    assumptions_text = (
        "Logistic assumptions:\n"
        "- independent binary outcomes\n"
        "- log-odds are linear in selected engineered terms\n"
        "- no perfect separation; sufficient sample size for SEs\n\n"
        "QDA assumptions:\n"
        "- predictors are approximately Gaussian within each class\n"
        "- each class has its own covariance matrix\n"
        "- observations are independent"
    )
    ax_assumptions.text(0.02, 0.50, assumptions_text, fontsize=9.0, family="monospace", va="center")
    ax_assumptions.set_title("Assumptions", fontsize=11, fontweight="bold")
    ax_assumptions.axis("off")

    ax_selection = fig.add_subplot(gs[1, :])
    draw_table(
        ax_selection,
        selection_table,
        "Model Selection",
        fontsize=5.8,
        scale_x=1.0,
        scale_y=1.5,
        col_widths=[0.14, 0.27, 0.10, 0.09, 0.10, 0.10, 0.09, 0.08, 0.08],
    )
    ax_selection.text(
        0.5,
        (-0.17),
        "For model selection, the training data is split into fitting and validation parts; metrics are computed on the validation part."
        " \n The hierarchical logistic model is retained for comparison; final predictions use the logistic model with the lowest validation log-loss.",
        ha="center",
        va="bottom",
        fontsize=7.8,
    )

    ax_coef = fig.add_subplot(gs[2, 0])
    ax_coef.axis("off")
    ax_coef.set_title("Logistic coefficients", fontsize=11, fontweight="bold", pad=8)
    coef_ax = ax_coef.inset_axes([0.16, 0.38, 0.52, 0.58])
    draw_table(coef_ax, coef_table, "", fontsize=7.4, scale_x=0.95, scale_y=1.05)
    coef_ax.set_title("")
    insight = (
        "Insight: x1 raises class-1 odds; x1^2 bends\n"
        "that effect downward; x2*x3 is the key\n"
        "interaction signal."
    )
    ax_coef.text(0.08, 0.18, insight, fontsize=8.2, family="monospace", va="center")

    ax_roc = fig.add_subplot(gs[2, 1])
    fpr_log, tpr_log, _ = roc_curve(valid_part["y"], valid_logistic_prob)
    fpr_qda, tpr_qda, _ = roc_curve(valid_part["y"], valid_qda_prob)
    ax_roc.plot(fpr_log, tpr_log, color=GREEN, linewidth=2, label=f"Logistic AUC={auc(fpr_log, tpr_log):.3f}")
    ax_roc.plot(fpr_qda, tpr_qda, color=RED, linewidth=2, label=f"QDA AUC={auc(fpr_qda, tpr_qda):.3f}")
    ax_roc.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    ax_roc.set_title("ROC / AUC", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # Slide 3: Visualization
    fig = plt.figure(figsize=(11, 8.5))
    gs3 = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.42)
    fig.suptitle("Visualization", fontsize=18, fontweight="bold")

    ax_prob_density = fig.add_subplot(gs3[0, 0])
    bins = np.linspace(0, 1, 22)
    ax_prob_density.hist(
        train.loc[train["y"] == 0, "fitted"],
        bins=bins,
        density=True,
        alpha=0.58,
        color=BLUE,
        label="y=0",
    )
    ax_prob_density.hist(
        train.loc[train["y"] == 1, "fitted"],
        bins=bins,
        density=True,
        alpha=0.58,
        color=RED,
        label="y=1",
    )
    ax_prob_density.axvline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax_prob_density.set_title("Predicted probability density", fontsize=10.4, fontweight="bold")
    ax_prob_density.set_xlabel("Fitted P(y=1)")
    ax_prob_density.set_ylabel("Density")
    ax_prob_density.legend(fontsize=7, loc="upper center")

    ax_class_box = fig.add_subplot(gs3[0, 1])
    class_positions = []
    class_box_data = []
    class_colors = []
    for idx, feature in enumerate(BASE_FEATURES, start=1):
        class_positions.extend([idx - 0.17, idx + 0.17])
        class_box_data.extend([
            raw_train.loc[raw_train["y"] == 0, feature],
            raw_train.loc[raw_train["y"] == 1, feature],
        ])
        class_colors.extend([BLUE, RED])
    bp2 = ax_class_box.boxplot(
        class_box_data,
        positions=class_positions,
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
    )
    for patch, color in zip(bp2["boxes"], class_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    ax_class_box.set_xticks([1, 2, 3])
    ax_class_box.set_xticklabels(BASE_FEATURES)
    ax_class_box.set_title("Covariates by class", fontsize=10.4, fontweight="bold")
    ax_class_box.legend([bp2["boxes"][0], bp2["boxes"][1]], ["y=0", "y=1"], fontsize=7, loc="upper right")

    ax_reliability = fig.add_subplot(gs3[0, 2])
    rel = reliability_points(train)
    ax_reliability.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Ideal")
    ax_reliability.plot(
        rel["pred_mean"],
        rel["observed_rate"],
        marker="o",
        linewidth=1.8,
        color=PURPLE,
        label="Model",
    )
    for _, row in rel.iterrows():
        ax_reliability.text(row["pred_mean"], row["observed_rate"] + 0.035, str(int(row["count"])), fontsize=6.5, ha="center")
    ax_reliability.set_xlim(0, 1)
    ax_reliability.set_ylim(0, 1)
    ax_reliability.set_title("Reliability plot", fontsize=10.4, fontweight="bold")
    ax_reliability.set_xlabel("Mean fitted P(y=1)")
    ax_reliability.set_ylabel("Observed y=1 rate")
    ax_reliability.legend(fontsize=7, loc="upper left")

    ax_cloud = fig.add_subplot(gs3[1, 0])
    ax_cloud.axis("off")
    ax_cloud.set_title("Pairwise class clouds", fontsize=10.4, fontweight="bold", pad=6)
    cloud_specs = [
        ("x1", "x2", [0.00, 0.53, 0.47, 0.38]),
        ("x1", "x3", [0.53, 0.53, 0.47, 0.38]),
        ("x2", "x3", [0.27, 0.05, 0.47, 0.38]),
    ]
    for x_col, y_col, bounds in cloud_specs:
        cloud_ax = ax_cloud.inset_axes(bounds)
        for cls, color in [(0, BLUE), (1, RED)]:
            subset = raw_train[raw_train["y"] == cls]
            cloud_ax.scatter(
                subset[x_col],
                subset[y_col],
                s=7,
                alpha=0.34,
                color=color,
                edgecolor="none",
                label=f"y={cls}",
            )
            draw_covariance_ellipse(cloud_ax, subset, x_col, y_col, color)
        cloud_ax.set_xlabel(x_col, fontsize=7, labelpad=0)
        cloud_ax.set_ylabel(y_col, fontsize=7, labelpad=0)
        cloud_ax.tick_params(axis="both", labelsize=6, pad=1)
        cloud_ax.grid(alpha=0.18, linewidth=0.5)
    ax_cloud.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="y=0", markersize=5),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, label="y=1", markersize=5),
        ],
        loc="lower right",
        fontsize=7,
        frameon=True,
    )

    ax_cm = fig.add_subplot(gs3[1, 1])
    ax_cm.axis("off")
    ax_cm.set_title("Validation confusion matrices", fontsize=10.4, fontweight="bold", pad=6)
    log_cm_ax = ax_cm.inset_axes([-0.2, 0.2, 0.5, 1])
    qda_cm_ax = ax_cm.inset_axes([0.61, 0.2, 0.5, 1])
    log_cm = confusion_matrix(valid_part["y"], (valid_logistic_prob >= 0.5).astype(int), labels=[0, 1])
    qda_cm = confusion_matrix(valid_part["y"], (valid_qda_prob >= 0.5).astype(int), labels=[0, 1])
    draw_confusion(log_cm_ax, log_cm, "Logistic")
    draw_confusion(qda_cm_ax, qda_cm, "QDA")

    ax_surface = fig.add_subplot(gs3[1, 2])
    xx, yy, zz = predicted_probability_grid(final_fit, train, x1_value=raw_train["x1"].median())
    contour = ax_surface.contourf(xx, yy, zz, levels=np.linspace(0, 1, 12), cmap="RdBu_r", alpha=0.80)
    ax_surface.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.1)
    ax_surface.scatter(
        raw_train["x2"],
        raw_train["x3"],
        c=raw_train["y"].map({0: BLUE, 1: RED}),
        alpha=0.54,
        s=20,
        edgecolor="white",
        linewidth=0.2,
    )
    ax_surface.set_title("x2*x3 probability surface", fontsize=10.4, fontweight="bold")
    ax_surface.set_xlabel("x2")
    ax_surface.set_ylabel("x3")
    fig.colorbar(contour, ax=ax_surface, fraction=0.046, pad=0.04, label="P(y=1)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

print(f"Wrote {PDF_PATH}")
