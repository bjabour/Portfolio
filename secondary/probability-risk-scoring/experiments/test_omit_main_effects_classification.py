import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


warnings.filterwarnings("ignore")

TRAIN_PATH = os.path.join("..", "classification_train.csv")
RANDOM_STATE = 9618

MODELS = {
    "hierarchical_selected": ["x1", "x2", "x3", "x1_sq", "x2_x3"],
    "omit_x2_x3_main_effects": ["x1", "x1_sq", "x2_x3"],
    "interaction_only_plus_mains": ["x1", "x2", "x3", "x2_x3"],
    "interaction_only_no_x2_x3_mains": ["x1", "x2_x3"],
}


def add_terms(df):
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


def fit_predict(train_df, valid_df, terms):
    x_train = sm.add_constant(train_df[terms], has_constant="add")
    x_valid = sm.add_constant(valid_df[terms], has_constant="add")
    fit = sm.GLM(train_df["y"], x_train, family=sm.families.Binomial()).fit(maxiter=200)
    return np.clip(fit.predict(x_valid), 1e-12, 1 - 1e-12)


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
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)

rows = []
for fold, (train_idx, valid_idx) in enumerate(cv.split(data[["x1", "x2", "x3"]], data["y"]), start=1):
    train_df = data.iloc[train_idx].copy()
    valid_df = data.iloc[valid_idx].copy()
    y_valid = valid_df["y"].to_numpy()
    for model_name, terms in MODELS.items():
        prob = fit_predict(train_df, valid_df, terms)
        row = {"model": model_name, "terms": " + ".join(terms), "fold": fold}
        row.update(score(y_valid, prob))
        rows.append(row)

folds = pd.DataFrame(rows)
folds.to_csv("classification_omit_main_effects_cv_folds.csv", index=False)

summary = (
    folds.groupby(["model", "terms"])
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
summary.to_csv("classification_omit_main_effects_summary.csv", index=False)

hier = summary.loc[summary["model"] == "hierarchical_selected"].iloc[0]
omit = summary.loc[summary["model"] == "omit_x2_x3_main_effects"].iloc[0]
delta = omit["log_loss_mean"] - hier["log_loss_mean"]

lines = [
    "Omitting x2 and x3 while keeping x2*x3",
    "======================================",
    "",
    "Repeated CV: 5 folds x 10 repeats = 50 validation folds.",
    "",
    summary.round(4).to_string(index=False),
    "",
    f"Log-loss change from omitting x2 and x3 main effects: {delta:+.4f}",
]
with open("classification_omit_main_effects_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(lines[-1])
print(summary.round(4).to_string(index=False))
