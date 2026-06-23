import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
PDF = ROOT / "trees_slides.pdf"
TRAIN = ROOT / "trees_train.csv"
RESULTS = ROOT / "trees_results.json"
BLUE, RED, GREEN, PURPLE, GOLD, GRAY = "#2f6f9f", "#c94f37", "#3a8f60", "#7257a8", "#bf8f2f", "#555555"
LIGHT, DARK = "#e8eef3", "#222222"
FEATURES = ["GrLivArea", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
            "TotalBsmtSF", "GarageArea", "BedroomAbvGr", "Fireplaces", "WoodDeckSF", "OpenPorchSF"]


def table(ax, frame, title, size=8, yscale=1.18, widths=None):
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)
    obj = ax.table(cellText=frame.values, colLabels=frame.columns, loc="center",
                   cellLoc="center", colLoc="center")
    obj.auto_set_font_size(False)
    obj.set_fontsize(size)
    obj.scale(1, yscale)
    for (row, col), cell in obj.get_celld().items():
        cell.set_edgecolor("#b9c2cb")
        cell.set_linewidth(.45)
        if row == 0:
            cell.set_facecolor(LIGHT)
            cell.set_text_props(weight="bold")
        if widths:
            cell.set_width(widths[col])
    return obj


def text_panel(ax, title, lines, size=8, body_y=.97, linespacing=1.22):
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=6)
    ax.text(.02, body_y, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=size, color=DARK, linespacing=linespacing)


def save(pdf, fig, number, bottom=.045, top=.915):
    fig.subplots_adjust(left=.06, right=.985, top=top, bottom=bottom)
    pdf.savefig(fig)
    fig.savefig(ROOT / f"trees_slide_preview_{number}.png", dpi=150)
    plt.close(fig)


def slide1(pdf):
    data = pd.read_csv(TRAIN)
    distressed = ["OverallQual", "GarageArea", "GrLivArea"]
    price = ["OverallQual", "GrLivArea", "GarageArea"]
    fig = plt.figure(figsize=(11, 8.5))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.46, .96, .96], hspace=.43)
    fig.suptitle("Data", fontsize=18, fontweight="bold", y=.975)

    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[.72, 2.28], wspace=.12)
    ax = fig.add_subplot(top[0])
    price_values = data.sale_price_keur
    lines = [
        "Train / test: 300 / 200",
        "Predictors: 12 numeric variables",
        "Missing values: 0",
        "",
        f"Sale price mean: {price_values.mean():.1f} kEUR",
        f"Median: {price_values.median():.1f} kEUR",
        f"Range: {price_values.min():.1f}-{price_values.max():.1f}",
        "",
        "Distressed:",
        f"47 / 300 = {data.distressed.mean():.1%}",
    ]
    text_panel(ax, "Sample and targets", lines, 9.0)

    ax = fig.add_subplot(top[1])
    corr = data[FEATURES + ["sale_price_keur", "distressed"]].corr()
    corr_table = pd.DataFrame({
        "Predictor": FEATURES,
        "Corr. with price": [f"{corr.loc[f, 'sale_price_keur']:.3f}" for f in FEATURES],
        "Corr. with distressed": [f"{corr.loc[f, 'distressed']:.3f}" for f in FEATURES],
    })
    table(ax, corr_table, "Numerical Pearson correlations", 7.5, 1.02, [.42, .29, .29])

    row = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=.32)
    for i, feature in enumerate(distressed):
        ax = fig.add_subplot(row[i])
        sns.boxplot(data=data, x="distressed", y=feature, ax=ax, color="#72B7B2",
                    width=.62, fliersize=1.5)
        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel("distressed", fontsize=9, labelpad=2)
        ax.set_ylabel("")
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", alpha=.18)

    row = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=.32)
    for i, feature in enumerate(price):
        ax = fig.add_subplot(row[i])
        sns.regplot(data=data, x=feature, y="sale_price_keur", ax=ax,
                    scatter_kws={"s": 8, "alpha": .35}, line_kws={"color": RED, "linewidth": 1.4})
        ax.set_title(feature, fontsize=11, fontweight="bold", pad=7)
        ax.set_xlabel("")
        ax.set_ylabel("sale price (kEUR)" if i == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8.2)
        ax.grid(alpha=.18)
    save(pdf, fig, 1, bottom=.035, top=.895)


def slide2(pdf, results):
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.16, 1.16, 1.02], hspace=.27, wspace=.25)
    fig.suptitle("Model Selection & Final Choices", fontsize=18,
                 fontweight="bold", y=.975)

    text_panel(fig.add_subplot(gs[0, 0]), "Libraries", [
        "• numpy / pandas: data, arrays, JSON",
        "• sklearn: permitted tree, RF classifier,",
        "  sigmoid calibration and validation",
        "",
        "• imbalanced-learn: fold-local oversampling",
        "• XGBoost 3.2: Tasks 3 and 5",
        "",
        "• matplotlib / seaborn: plots and slides",
        "• Tasks 1-2 use self-written CART/RF",
    ], 9.6, body_y=.86, linespacing=1.28)
    text_panel(fig.add_subplot(gs[0, 1]), "Assumptions & Notes", [
        "• Train and test share the same mechanism.",
        "• Rows are independent properties.",
        "",
        "• Original year and area values are retained.",
        "• Trees model nonlinearities and interactions.",
        "",
        "• Outer validation is untouched by tuning,",
        "  resampling, early stopping or calibration.",
        "• Importance is predictive, not causal.",
    ], 9.6, body_y=.86, linespacing=1.28)
    text_panel(fig.add_subplot(gs[0, 2]), "Methods", [
        "1. CART tests midpoints and maximizes SSE drop.",
        "2. RF bootstraps rows, samples mtry per node,",
        "   averages predictions and aggregates MDI.",
        "3. Boosting screens structure and regularization;",
        "   early stopping uses training data only.",
        "4. Regression selection: validation MSE.",
        "5. Classification selection: stratified log loss;",
        "   AUC, Brier and prevalence bias support it.",
        "6. Resampling/calibration stay inside train folds.",
        "7. Finalists: six seeds, 30 folds, full-data refit.",
    ], 8.8, body_y=.86, linespacing=1.25)

    model_rows = pd.DataFrame([
        ["1 CART", "5-fold MSE, depth 2-10", "depth 6", "Required output is depth 3; sklearn agreement exact"],
        ["2 Scratch RF", "OOB screen + 30 CV folds", "MSE 1484.85", "Best stable setting: depth 8, leaf 1"],
        ["3 XGB reg", "early-stop screen + 30 folds", "MSE 1362.70", "Lowest regression error; shallow additive trees"],
        ["4 RF class", "stratified probability CV", "log loss 0.37046", "Ratio 0.20 resampling beat weighting/calibration"],
        ["5 XGB class", "nested probability CV", "log loss 0.38167", "Weight 1.5 + sigmoid gave calibrated probabilities"],
    ], columns=["Task", "Comparison", "Selected score", "Why selected"])
    table(fig.add_subplot(gs[1, :]), model_rows, "Validation evidence", 7.8, 1.44,
          [.13, .23, .14, .50])

    rf_oob = pd.read_csv(EXP / "Random-Forest" / "rf_final_convergence.csv").iloc[-1]
    rf_clf = pd.read_csv(EXP / "Random-Forest-Classification" / "rf_clf_repeated_cv_summary.csv").iloc[0]
    gb_clf = pd.read_json(EXP / "XGBoost-Classification" / "xgb_clf_selected_configuration.json", typ="series")
    def top_value(key):
        name, value = max(results[key].items(), key=lambda item: item[1])
        return f"{name} ({value:.3f})"
    numeric = pd.DataFrame([
        ["1 CART", f"{min(results['tree_cv_mse_curve']):.2f}", "equiv. max Δ", f"{max(abs(np.array(results['tree_pred_test'])-np.array(results['tree_lib_pred_test']))):.1e}"],
        ["2 RF reg", "1484.85", "OOB MSE", f"{rf_oob.oob_mse:.2f}"],
        ["3 XGB reg", "1362.70", "top total gain", top_value("gbm_reg_var_importance")],
        ["4 RF class", "log loss 0.37046", f"AUC {rf_clf.mean_roc_auc:.3f}", f"Brier {rf_clf.mean_brier_score:.3f}"],
        ["5 XGB class", "log loss 0.38167", f"AUC {gb_clf.mean_roc_auc:.3f}", f"Brier {gb_clf.mean_brier_score:.3f}"],
    ], columns=["Model", "Validation result", "Probability / purity diagnostic", "Value"])
    table(fig.add_subplot(gs[2, :2]), numeric, "Numerical results", 7.6, 1.38,
          [.17, .24, .30, .29])

    text_panel(fig.add_subplot(gs[2, 2]), "Selected model parameters", [
        "T1  output depth 3; CV depth 6; leaf 5",
        "     SSE drop = purity improvement",
        "T2  B=2000; mtry=4; depth 8; leaf 1",
        f"     top MDI: {top_value('rf_reg_var_importance')}",
        "T3  B=386; eta=.05; depth 1; gamma=.1",
        f"     top gain: {top_value('gbm_reg_var_importance')}",
        "T4  B=2000; mtry=6; depth 5; log loss",
        f"     top MDI: {top_value('rf_clf_var_importance')}",
        "T5  B=122; eta=.03; depth 3; weight=1.5",
        f"     top gain: {top_value('gbm_clf_var_importance')}",
    ], 8.8, body_y=.95, linespacing=1.14)
    save(pdf, fig, 2, bottom=.035, top=.895)


def top_bars(ax, frame, value, labels, title, color):
    view = frame.nsmallest(6, value).copy()
    view["label"] = [labels(row) for _, row in view.iterrows()]
    ax.barh(np.arange(len(view)), view[value], color=color, alpha=.82)
    ax.set_yticks(np.arange(len(view)))
    ax.set_yticklabels(view["label"], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=.2)
    low, high = float(view[value].min()), float(view[value].max())
    pad = max((high - low) * .22, abs(low) * .004)
    ax.set_xlim(low - pad, high + pad)


def comparison_bars(ax, labels, values, title, xlabel, colors):
    order = np.argsort(values)[::-1]
    labels = np.asarray(labels)[order]
    values = np.asarray(values, dtype=float)[order]
    colors = np.asarray(colors)[order]
    ax.barh(np.arange(len(values)), values, color=colors, alpha=.84)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=7.8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=.2)
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * .08, abs(low) * .004)
    ax.set_xlim(low - pad, high + pad)


def slide3(pdf, results):
    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(2, 4, figure=fig, hspace=.58, wspace=.62)
    fig.suptitle("Diagnostics and Modeling Insights", fontsize=18, fontweight="bold", y=.975)

    ax = fig.add_subplot(gs[0, 0])
    depths = np.arange(2, 11)
    curve = np.asarray(results["tree_cv_mse_curve"])
    ax.plot(depths, curve, marker="o", color=BLUE)
    ax.scatter([results["tree_max_depth"]], [curve.min()], color=RED, s=40, zorder=4)
    ax.set_xlabel("max depth", fontsize=8)
    ax.set_title("Task 1: CART depth", fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=.22)

    ax = fig.add_subplot(gs[0, 1])
    rf_conv = pd.read_csv(EXP / "Random-Forest" / "rf_final_convergence.csv")
    ax.plot(rf_conv.n_trees, rf_conv.oob_mse, marker="o", color=GREEN)
    ax.axvline(200, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("scratch trees", fontsize=8)
    ax.set_ylabel("OOB MSE", fontsize=8)
    ax.set_title("Task 2: RF convergence", fontsize=9.5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=.22)

    ax = fig.add_subplot(gs[0, 2])
    xgb_reg = pd.read_csv(EXP / "XGBoost-Regression" /
                          "xgb_reg_repeated_cv_summary.csv")
    top_bars(ax, xgb_reg, "mean_mse",
             lambda r: f"d{int(r.max_depth)} lr{r.learning_rate:g} "
                       f"B{int(r.median_best_n_estimators)}",
             "Task 3: XGB regression", PURPLE)
    ax.set_xlabel("validation MSE", fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    linear = pd.read_csv(EXP / "linear-regression" / "nested_cv_summary.csv")
    rf_reg = pd.read_csv(EXP / "Random-Forest" / "rf_repeated_cv_summary.csv")
    price_labels = ["OLS", "XGB", "Scratch RF", "CART"]
    price_values = [
        float(linear.loc[linear.model == "ols_all", "mean_nested_mse"].iloc[0]),
        float(xgb_reg.mean_mse.min()),
        float(rf_reg.mean_validation_mse.min()),
        float(linear.loc[linear.model == "cart", "mean_nested_mse"].iloc[0]),
    ]
    comparison_bars(ax, price_labels, price_values, "Price benchmark",
                    "repeated-CV MSE", [GRAY, PURPLE, GREEN, BLUE])

    ax = fig.add_subplot(gs[1, 0])
    rf_clf = pd.read_csv(EXP / "Random-Forest-Classification" /
                         "rf_clf_repeated_cv_summary.csv")
    top_bars(
        ax, rf_clf, "mean_log_loss",
        lambda r: f"{'ROS' if r.strategy == 'random_over' else 'SM' if r.strategy == 'smote' else r.strategy}"
                  f" d{int(r.max_depth)}",
        "Task 4: RF probabilities", BLUE
    )
    ax.set_xlabel("validation log loss", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    xgb_clf = pd.read_csv(EXP / "XGBoost-Classification" /
                          "xgb_clf_complete_model_summary.csv")
    top_bars(ax, xgb_clf, "mean_log_loss",
             lambda r: f"{str(r.imbalance_strategy)[:4]} "
                       f"{str(r.calibration)[:3]}-{int(r.calibration_folds)}",
             "Task 5: XGB calibration", GOLD)
    ax.set_xlabel("validation log loss", fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    models = {
        "RF reg": results["rf_reg_var_importance"],
        "XGB reg": results["gbm_reg_var_importance"],
        "RF clf": results["rf_clf_var_importance"],
        "XGB clf": results["gbm_clf_var_importance"],
    }
    importance = pd.DataFrame(models).T[FEATURES]
    top = importance.max(axis=0).nlargest(5).index
    sns.heatmap(importance[top], cmap="Blues", annot=True, fmt=".2f", cbar=False,
                ax=ax, linewidths=.4, annot_kws={"fontsize": 6.8})
    ax.set_title("Cross-model importance", fontsize=9.5, fontweight="bold")
    short = {
        "OverallQual": "Quality", "GrLivArea": "Living", "LotArea": "Lot",
        "YearBuilt": "Built", "GarageArea": "Garage", "TotalBsmtSF": "Basement",
        "OverallCond": "Condition",
    }
    ax.set_xticklabels([short.get(name, name) for name in top], rotation=32, ha="right")
    ax.tick_params(axis="x", labelsize=6.7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    ax = fig.add_subplot(gs[1, 3])
    clf_compare = pd.read_csv(
        EXP / "Logistic-vs-Tree-Classifiers" /
        "classifier_comparison_summary.csv"
    ).set_index("model")
    clf_labels = ["Logistic + RF", "Task 4 RF", "Task 5 XGB"]
    clf_values = [
        float(clf_compare.loc["blend_raw_l2_rf", "mean_log_loss"]),
        float(clf_compare.loc["random_forest", "mean_log_loss"]),
        float(clf_compare.loc["xgboost", "mean_log_loss"]),
    ]
    comparison_bars(ax, clf_labels, clf_values, "Probability benchmark",
                    "repeated-CV log loss", [RED, BLUE, GOLD])
    save(pdf, fig, 3, bottom=.09, top=.875)


def main():
    if not RESULTS.exists():
        raise FileNotFoundError("Run trees_code.py before generating slides.")
    sns.set_theme(style="whitegrid")
    results = json.loads(RESULTS.read_text())
    with PdfPages(PDF) as pdf:
        slide1(pdf)
        slide2(pdf, results)
        slide3(pdf, results)
    print(f"Wrote {PDF}")


if __name__ == "__main__":
    main()
