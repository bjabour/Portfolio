import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


warnings.filterwarnings("ignore")

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


def label(optional_terms):
    return "main effects only" if not optional_terms else "main + " + " + ".join(optional_terms)


def predict_glm(train_df, valid_df, terms):
    x_train = sm.add_constant(train_df[terms], has_constant="add")
    x_valid = sm.add_constant(valid_df[terms], has_constant="add")
    fit = sm.GLM(train_df["y"], x_train, family=sm.families.Binomial()).fit(maxiter=200)
    return np.clip(fit.predict(x_valid), 1e-12, 1 - 1e-12)


def metric_row(model_name, optional_terms, fold, y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "model": model_name,
        "optional_terms": ", ".join(optional_terms) if optional_terms else "none",
        "n_optional_terms": len(optional_terms),
        "fold": fold,
        "log_loss": log_loss(y_true, prob),
        "auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


raw = pd.read_csv(TRAIN_PATH)
data = add_terms(raw)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

all_subsets = []
for k in range(len(OPTIONAL_TERMS) + 1):
    all_subsets.extend(list(itertools.combinations(OPTIONAL_TERMS, k)))

rows = []
for fold, (train_idx, valid_idx) in enumerate(cv.split(data[MAIN_TERMS], data["y"]), start=1):
    train_df = data.iloc[train_idx].copy()
    valid_df = data.iloc[valid_idx].copy()
    y_valid = valid_df["y"].to_numpy()
    for optional in all_subsets:
        optional = list(optional)
        terms = MAIN_TERMS + optional
        prob = predict_glm(train_df, valid_df, terms)
        rows.append(metric_row(label(optional), optional, fold, y_valid, prob))

folds = pd.DataFrame(rows)
folds.to_csv("classification_glm_repeated_term_selection_folds.csv", index=False)

summary = (
    folds.groupby(["model", "optional_terms", "n_optional_terms"])
    .agg(
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
    )
    .reset_index()
    .sort_values(["log_loss_mean", "n_optional_terms", "model"])
)
summary.to_csv("classification_glm_repeated_term_selection_all.csv", index=False)
summary.head(20).to_csv("classification_glm_repeated_term_selection_top20.csv", index=False)

top = summary.head(15)
plt.figure(figsize=(10.8, 6.5))
plt.barh(top["model"], top["log_loss_mean"], xerr=top["log_loss_sd"], color="#2f6f9f", alpha=0.86)
plt.gca().invert_yaxis()
plt.xlabel("Repeated 5-fold CV log-loss, lower is better")
plt.title("Top unpenalized logistic GLM term subsets")
plt.tight_layout()
plt.savefig("classification_glm_repeated_term_selection_top15.png", dpi=180, bbox_inches="tight")
plt.close()

best = summary.iloc[0]
near = summary[summary["log_loss_mean"] <= best["log_loss_mean"] + 0.003]
simple_near = near.sort_values(["n_optional_terms", "log_loss_mean"]).iloc[0]

lines = [
    "Repeated-CV logistic GLM term selection",
    "======================================",
    "",
    "All models keep x1, x2, x3. Tested every subset of:",
    ", ".join(OPTIONAL_TERMS),
    "",
    "Repeated CV: 5 folds x 5 repeats = 25 validation folds.",
    "",
    "Top 20 models:",
    summary.head(20)[
        [
            "optional_terms",
            "n_optional_terms",
            "log_loss_mean",
            "log_loss_sd",
            "auc_mean",
            "accuracy_mean",
            "f1_mean",
        ]
    ].round(4).to_string(index=False),
    "",
    f"Best GLM by repeated-CV log-loss: {best['optional_terms']} ({best['log_loss_mean']:.4f})",
    (
        "Simplest GLM within 0.003 log-loss of best: "
        f"{simple_near['optional_terms']} ({simple_near['log_loss_mean']:.4f})"
    ),
]

with open("classification_glm_repeated_term_selection_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Wrote repeated GLM term-selection outputs:")
for filename in [
    "classification_glm_repeated_term_selection_folds.csv",
    "classification_glm_repeated_term_selection_all.csv",
    "classification_glm_repeated_term_selection_top20.csv",
    "classification_glm_repeated_term_selection_top15.png",
    "classification_glm_repeated_term_selection_summary.txt",
]:
    print(f"  {filename}")
print()
print(f"Best GLM: {best['optional_terms']} ({best['log_loss_mean']:.4f})")
