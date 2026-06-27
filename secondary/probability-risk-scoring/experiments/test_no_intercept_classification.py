import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split


warnings.filterwarnings("ignore")

TRAIN_PATH = os.path.join("..", "classification_train.csv")
RANDOM_STATE = 9618
TERMS = ["x1", "x2", "x3", "x1_sq", "x2_x3"]


def add_terms(df):
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


def design(df, intercept=True):
    x = df[TERMS]
    if intercept:
        x = sm.add_constant(x, has_constant="add")
    return x


def fit_predict(train_df, valid_df, intercept=True):
    fit = sm.GLM(train_df["y"], design(train_df, intercept), family=sm.families.Binomial()).fit(maxiter=200)
    prob = fit.predict(design(valid_df, intercept))
    return np.clip(prob, 1e-12, 1 - 1e-12), fit


def score(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "log_loss": log_loss(y_true, prob),
        "auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


raw = pd.read_csv(TRAIN_PATH)
data = add_terms(raw)

train_part, valid_part = train_test_split(
    data,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=data["y"],
)

single_rows = []
for label, intercept in [("with_intercept", True), ("no_intercept", False)]:
    prob, fit = fit_predict(train_part, valid_part, intercept)
    row = {"model": label, "validation": "single_split", **score(valid_part["y"], prob), "aic_train_fit": fit.aic}
    single_rows.append(row)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
cv_rows = []
for fold, (tr_idx, va_idx) in enumerate(cv.split(data[["x1", "x2", "x3"]], data["y"]), start=1):
    tr = data.iloc[tr_idx].copy()
    va = data.iloc[va_idx].copy()
    for label, intercept in [("with_intercept", True), ("no_intercept", False)]:
        prob, _ = fit_predict(tr, va, intercept)
        cv_rows.append({"model": label, "fold": fold, **score(va["y"], prob)})

cv_df = pd.DataFrame(cv_rows)
cv_summary = (
    cv_df.groupby("model")
    .agg(
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        auc_mean=("auc", "mean"),
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
    )
    .reset_index()
    .sort_values("log_loss_mean")
)

single = pd.DataFrame(single_rows).sort_values("log_loss")
single.to_csv("classification_intercept_single_split.csv", index=False)
cv_summary.to_csv("classification_intercept_repeated_cv_summary.csv", index=False)
cv_df.to_csv("classification_intercept_repeated_cv_folds.csv", index=False)

full_with = sm.GLM(data["y"], design(data, True), family=sm.families.Binomial()).fit(maxiter=200)
full_without = sm.GLM(data["y"], design(data, False), family=sm.families.Binomial()).fit(maxiter=200)

lines = [
    "Intercept test for chosen logistic model",
    "=======================================",
    "",
    "Chosen terms: x1 + x2 + x3 + x1^2 + x2*x3",
    "",
    "Single validation split:",
    single.round(4).to_string(index=False),
    "",
    "Repeated 5-fold CV x 10 repeats:",
    cv_summary.round(4).to_string(index=False),
    "",
    "Full-data fit:",
    f"with intercept:    AIC={full_with.aic:.4f}, intercept={full_with.params.iloc[0]:.4f}, p={full_with.pvalues.iloc[0]:.4f}",
    f"without intercept: AIC={full_without.aic:.4f}",
]

with open("classification_intercept_test_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
