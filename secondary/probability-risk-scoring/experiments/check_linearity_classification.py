import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN_PATH = os.path.join("..", "classification_train.csv")
OUT_DIR = "."
FEATURES = ["x1", "x2", "x3"]
RANDOM_STATE = 9618


MODEL_SPECS = {
    "linear": ["x1", "x2", "x3"],
    "interactions": ["x1", "x2", "x3", "x1_x2", "x1_x3", "x2_x3"],
    "quadratic": ["x1", "x2", "x3", "x1_sq", "x2_sq", "x3_sq"],
    "quadratic_interactions": [
        "x1",
        "x2",
        "x3",
        "x1_sq",
        "x2_sq",
        "x3_sq",
        "x1_x2",
        "x1_x3",
        "x2_x3",
    ],
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_terms(df):
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_sq"] = out["x2"] ** 2
    out["x3_sq"] = out["x3"] ** 2
    out["x1_x2"] = out["x1"] * out["x2"]
    out["x1_x3"] = out["x1"] * out["x3"]
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


def scale_design(train_df, valid_df, terms):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[terms])
    x_valid = scaler.transform(valid_df[terms])
    return sm.add_constant(x_train, has_constant="add"), sm.add_constant(x_valid, has_constant="add")


def fit_predict(train_df, valid_df, terms):
    x_train, x_valid = scale_design(train_df, valid_df, terms)
    y_train = train_df["y"].to_numpy()
    model = sm.GLM(y_train, x_train, family=sm.families.Binomial())
    fit = model.fit(maxiter=200)
    proba = np.asarray(fit.predict(x_valid))
    return np.clip(proba, 1e-6, 1 - 1e-6), fit


def metric_row(model_name, y_true, proba):
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "log_loss": log_loss(y_true, proba),
        "auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def savefig(name, tight=True):
    if tight:
        plt.tight_layout()
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


raw = pd.read_csv(TRAIN_PATH)
data = add_terms(raw)

train_df, valid_df = train_test_split(
    data,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=data["y"],
)

single_split_rows = []
roc_rows = {}
fits = {}
for name, terms in MODEL_SPECS.items():
    proba, fit = fit_predict(train_df, valid_df, terms)
    single_split_rows.append(metric_row(name, valid_df["y"].to_numpy(), proba))
    roc_rows[name] = roc_curve(valid_df["y"].to_numpy(), proba)
    fits[name] = fit

single_split = pd.DataFrame(single_split_rows).sort_values(["log_loss", "model"])
single_split.to_csv("classification_logistic_linearity_single_split.csv", index=False)

# Repeated out-of-fold comparison gives a less noisy read than one validation split.
cv_rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr_idx, va_idx) in enumerate(skf.split(data[FEATURES], data["y"]), start=1):
    cv_train = data.iloc[tr_idx].copy()
    cv_valid = data.iloc[va_idx].copy()
    y_valid = cv_valid["y"].to_numpy()
    for name, terms in MODEL_SPECS.items():
        proba, _ = fit_predict(cv_train, cv_valid, terms)
        row = metric_row(name, y_valid, proba)
        row["fold"] = fold
        cv_rows.append(row)

cv = pd.DataFrame(cv_rows)
cv.to_csv("classification_logistic_linearity_cv_folds.csv", index=False)
cv_summary = (
    cv.groupby("model")
    .agg(
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
    )
    .sort_values("log_loss_mean")
)
cv_summary.to_csv("classification_logistic_linearity_cv_summary.csv")

# Bar chart of cross-validated probability performance.
plot_df = cv.melt(
    id_vars=["model", "fold"],
    value_vars=["log_loss", "auc", "accuracy", "f1"],
    var_name="metric",
    value_name="value",
)
plt.figure(figsize=(11, 5.5))
sns.barplot(data=plot_df, x="model", y="value", hue="metric", errorbar="sd")
plt.title("Logistic model comparison: linear vs nonlinear terms")
plt.xlabel("")
plt.xticks(rotation=20, ha="right")
savefig("classification_linearity_model_metrics.png")

# ROC curves for the held-out validation split.
plt.figure(figsize=(6.4, 5.2))
for name, (fpr, tpr, _) in roc_rows.items():
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Validation ROC: logistic model variants")
plt.legend(fontsize=8)
savefig("classification_linearity_roc.png")

# Predicted-probability surfaces for linear and best nonlinear model.
best_model = cv_summary.index[0]
surface_models = ["linear"]
if best_model != "linear":
    surface_models.append(best_model)

pairs = [("x1", "x2", "x3"), ("x1", "x3", "x2"), ("x2", "x3", "x1")]
fig, axes = plt.subplots(len(surface_models), 3, figsize=(15, 4.6 * len(surface_models)))
if len(surface_models) == 1:
    axes = np.array([axes])

for row_idx, model_name in enumerate(surface_models):
    terms = MODEL_SPECS[model_name]
    for col_idx, (a, b, held) in enumerate(pairs):
        ax = axes[row_idx, col_idx]
        xx, yy = np.meshgrid(
            np.linspace(data[a].quantile(0.02), data[a].quantile(0.98), 120),
            np.linspace(data[b].quantile(0.02), data[b].quantile(0.98), 120),
        )
        grid = pd.DataFrame({a: xx.ravel(), b: yy.ravel(), held: data[held].median()})
        grid = add_terms(grid)
        # Refit surface model on full data for a smoother display.
        scaler = StandardScaler()
        x_full = sm.add_constant(scaler.fit_transform(data[terms]), has_constant="add")
        fit_full = sm.GLM(data["y"], x_full, family=sm.families.Binomial()).fit(maxiter=200)
        x_grid = sm.add_constant(scaler.transform(grid[terms]), has_constant="add")
        zz = fit_full.predict(x_grid).reshape(xx.shape)
        contour = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 13), cmap="RdBu_r", alpha=0.75)
        ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.3)
        sns.scatterplot(
            data=data,
            x=a,
            y=b,
            hue="y",
            palette={0: "#2f6f9f", 1: "#c94f37"},
            s=22,
            alpha=0.60,
            edgecolor="white",
            linewidth=0.2,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{model_name}: {a} vs {b}, {held}=median")
fig.subplots_adjust(right=0.90, hspace=0.35, wspace=0.25)
cbar_ax = fig.add_axes([0.925, 0.18, 0.016, 0.64])
fig.colorbar(contour, cax=cbar_ax, label="Predicted P(y=1)")
savefig("classification_linearity_probability_surfaces.png", tight=False)

summary_lines = [
    "Checking whether a linear decision boundary is sufficient",
    "=========================================================",
    "",
    "Single validation split metrics:",
    single_split.round(4).to_string(index=False),
    "",
    "Five-fold cross-validation summary:",
    cv_summary.round(4).to_string(),
    "",
    f"Best model by mean CV log-loss: {best_model}",
    "",
    "Interpretation guide:",
    (
        "If nonlinear models reduce validation/CV log-loss and increase AUC/F1 relative "
        "to the linear model, that is evidence that a purely linear boundary is not "
        "sufficient. If improvements are tiny or unstable across folds, the simpler "
        "linear model may be preferable."
    ),
]

with open("classification_linearity_analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Wrote linearity-check outputs:")
for name in [
    "classification_logistic_linearity_single_split.csv",
    "classification_logistic_linearity_cv_folds.csv",
    "classification_logistic_linearity_cv_summary.csv",
    "classification_linearity_model_metrics.png",
    "classification_linearity_roc.png",
    "classification_linearity_probability_surfaces.png",
    "classification_linearity_analysis_summary.txt",
]:
    print(f"  {name}")
