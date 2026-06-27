import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN_PATH = os.path.join("..", "classification_train.csv")
RANDOM_STATE = 9618
MAIN_TERMS = ["x1", "x2", "x3"]
OPTIONAL_TERMS = ["x1_sq", "x2_sq", "x3_sq", "x1_x2", "x1_x3", "x2_x3"]


def add_terms(df):
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_sq"] = out["x2"] ** 2
    out["x3_sq"] = out["x3"] ** 2
    out["x1_x2"] = out["x1"] * out["x2"]
    out["x1_x3"] = out["x1"] * out["x3"]
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


def model_label(optional_terms):
    if not optional_terms:
        return "main_effects_only"
    return "main + " + " + ".join(optional_terms)


def fit_predict(train_df, valid_df, terms):
    scaler = StandardScaler()
    x_train = sm.add_constant(scaler.fit_transform(train_df[terms]), has_constant="add")
    x_valid = sm.add_constant(scaler.transform(valid_df[terms]), has_constant="add")
    fit = sm.GLM(train_df["y"], x_train, family=sm.families.Binomial()).fit(maxiter=200)
    proba = np.clip(fit.predict(x_valid), 1e-6, 1 - 1e-6)
    return proba


def score_model(data, optional_terms):
    terms = MAIN_TERMS + list(optional_terms)
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(data[MAIN_TERMS], data["y"]), start=1):
        train_df = data.iloc[train_idx].copy()
        valid_df = data.iloc[valid_idx].copy()
        y_true = valid_df["y"].to_numpy()
        proba = fit_predict(train_df, valid_df, terms)
        pred = (proba >= 0.5).astype(int)
        rows.append(
            {
                "fold": fold,
                "log_loss": log_loss(y_true, proba),
                "auc": roc_auc_score(y_true, proba),
                "accuracy": accuracy_score(y_true, pred),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def full_data_fit_stats(data, optional_terms):
    terms = MAIN_TERMS + list(optional_terms)
    scaler = StandardScaler()
    x = sm.add_constant(scaler.fit_transform(data[terms]), has_constant="add")
    fit = sm.GLM(data["y"], x, family=sm.families.Binomial()).fit(maxiter=200)
    return fit.aic, fit.bic_deviance, fit.deviance


def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=180, bbox_inches="tight")
    plt.close()


raw = pd.read_csv(TRAIN_PATH)
data = add_terms(raw)

all_fold_rows = []
summary_rows = []

for k in range(len(OPTIONAL_TERMS) + 1):
    for optional in itertools.combinations(OPTIONAL_TERMS, k):
        optional = list(optional)
        label = model_label(optional)
        fold_scores = score_model(data, optional)
        fold_scores["model"] = label
        fold_scores["n_optional_terms"] = len(optional)
        fold_scores["optional_terms"] = ", ".join(optional) if optional else "none"
        all_fold_rows.append(fold_scores)
        aic, bic_deviance, deviance = full_data_fit_stats(data, optional)
        summary_rows.append(
            {
                "model": label,
                "optional_terms": ", ".join(optional) if optional else "none",
                "n_optional_terms": len(optional),
                "cv_log_loss_mean": fold_scores["log_loss"].mean(),
                "cv_log_loss_sd": fold_scores["log_loss"].std(),
                "cv_auc_mean": fold_scores["auc"].mean(),
                "cv_auc_sd": fold_scores["auc"].std(),
                "cv_accuracy_mean": fold_scores["accuracy"].mean(),
                "cv_f1_mean": fold_scores["f1"].mean(),
                "full_data_aic": aic,
                "full_data_bic_deviance": bic_deviance,
                "full_data_deviance": deviance,
            }
        )

folds = pd.concat(all_fold_rows, ignore_index=True)
summary = pd.DataFrame(summary_rows).sort_values(["cv_log_loss_mean", "n_optional_terms", "model"])

folds.to_csv("classification_term_selection_cv_folds.csv", index=False)
summary.to_csv("classification_term_selection_all_models.csv", index=False)
summary.head(20).to_csv("classification_term_selection_top20.csv", index=False)

top = summary.head(15).copy()
plt.figure(figsize=(10.5, 6.5))
sns.barplot(data=top, y="model", x="cv_log_loss_mean", hue="n_optional_terms", dodge=False, palette="viridis")
plt.title("Top logistic term subsets by 5-fold CV log-loss")
plt.xlabel("Mean CV log-loss, lower is better")
plt.ylabel("")
plt.legend(title="optional terms", loc="lower right")
savefig("classification_term_selection_top15_logloss.png")

plt.figure(figsize=(8.8, 5.2))
sns.scatterplot(
    data=summary,
    x="n_optional_terms",
    y="cv_log_loss_mean",
    size="cv_auc_mean",
    hue="cv_auc_mean",
    palette="mako",
    sizes=(35, 160),
)
plt.title("All 64 logistic term subsets")
plt.xlabel("Number of optional nonlinear terms")
plt.ylabel("Mean CV log-loss")
savefig("classification_term_selection_complexity_tradeoff.png")

best = summary.iloc[0]
simple_near_best = summary[summary["cv_log_loss_mean"] <= best["cv_log_loss_mean"] + 0.005]
simple_near_best = simple_near_best.sort_values(["n_optional_terms", "cv_log_loss_mean"]).head(1).iloc[0]

lines = [
    "Logistic nonlinear term selection",
    "=================================",
    "",
    "Method: all models keep x1, x2, x3. I tested all 64 subsets of:",
    ", ".join(OPTIONAL_TERMS),
    "",
    "Top 20 models by 5-fold CV log-loss:",
    summary.head(20)[
        [
            "optional_terms",
            "n_optional_terms",
            "cv_log_loss_mean",
            "cv_log_loss_sd",
            "cv_auc_mean",
            "cv_accuracy_mean",
            "cv_f1_mean",
            "full_data_aic",
        ]
    ].round(4).to_string(index=False),
    "",
    f"Best by CV log-loss: {best['optional_terms']} ({best['cv_log_loss_mean']:.4f})",
    (
        "Simplest model within 0.005 log-loss of best: "
        f"{simple_near_best['optional_terms']} ({simple_near_best['cv_log_loss_mean']:.4f})"
    ),
]

with open("classification_term_selection_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Wrote term selection outputs:")
for filename in [
    "classification_term_selection_all_models.csv",
    "classification_term_selection_top20.csv",
    "classification_term_selection_cv_folds.csv",
    "classification_term_selection_top15_logloss.png",
    "classification_term_selection_complexity_tradeoff.png",
    "classification_term_selection_summary.txt",
]:
    print(f"  {filename}")
