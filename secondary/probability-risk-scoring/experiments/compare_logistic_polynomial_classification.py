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
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)

TRAIN_PATH = os.path.join("..", "classification_train.csv")
RANDOM_STATE = 9618

FEATURES = ["x1", "x2", "x3"]
MODEL_SPECS = {
    "linear_main_effects": ["x1", "x2", "x3"],
    "full_interactions": ["x1", "x2", "x3", "x1_x2", "x1_x3", "x2_x3"],
    "x1_squared_only": ["x1", "x2", "x3", "x1_sq"],
    "x2_squared_only": ["x1", "x2", "x3", "x2_sq"],
    "x3_squared_only": ["x1", "x2", "x3", "x3_sq"],
    "all_squared_terms": ["x1", "x2", "x3", "x1_sq", "x2_sq", "x3_sq"],
    "squares_plus_interactions": [
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


def add_polynomial_terms(df):
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_sq"] = out["x2"] ** 2
    out["x3_sq"] = out["x3"] ** 2
    out["x1_x2"] = out["x1"] * out["x2"]
    out["x1_x3"] = out["x1"] * out["x3"]
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


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


def fit_scaled_glm(train_df, valid_df, terms):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[terms])
    x_valid = scaler.transform(valid_df[terms])
    x_train = sm.add_constant(x_train, has_constant="add")
    x_valid = sm.add_constant(x_valid, has_constant="add")
    fit = sm.GLM(train_df["y"], x_train, family=sm.families.Binomial()).fit(maxiter=200)
    proba = np.clip(fit.predict(x_valid), 1e-6, 1 - 1e-6)
    return proba, fit, scaler


def fit_full_data_model(data, terms):
    scaler = StandardScaler()
    x = scaler.fit_transform(data[terms])
    x = sm.add_constant(x, has_constant="add")
    fit = sm.GLM(data["y"], x, family=sm.families.Binomial()).fit(maxiter=200)
    names = ["const"] + terms
    conf = np.asarray(fit.conf_int())
    coef = pd.DataFrame(
        {
            "term": names,
            "coef": fit.params,
            "std_err": fit.bse,
            "z": fit.tvalues,
            "p_value": fit.pvalues,
            "ci_lower": conf[:, 0],
            "ci_upper": conf[:, 1],
        }
    )
    return fit, coef


def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=180, bbox_inches="tight")
    plt.close()


raw = pd.read_csv(TRAIN_PATH)
data = add_polynomial_terms(raw)

train_df, valid_df = train_test_split(
    data,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=data["y"],
)

single_rows = []
for model_name, terms in MODEL_SPECS.items():
    proba, _, _ = fit_scaled_glm(train_df, valid_df, terms)
    single_rows.append(metric_row(model_name, valid_df["y"].to_numpy(), proba))

single = pd.DataFrame(single_rows).sort_values(["log_loss", "model"])
single.to_csv("classification_polynomial_single_split_comparison.csv", index=False)

cv_rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_index, valid_index) in enumerate(skf.split(data[FEATURES], data["y"]), start=1):
    cv_train = data.iloc[train_index].copy()
    cv_valid = data.iloc[valid_index].copy()
    y_valid = cv_valid["y"].to_numpy()
    for model_name, terms in MODEL_SPECS.items():
        proba, _, _ = fit_scaled_glm(cv_train, cv_valid, terms)
        row = metric_row(model_name, y_valid, proba)
        row["fold"] = fold
        cv_rows.append(row)

cv = pd.DataFrame(cv_rows)
cv.to_csv("classification_polynomial_cv_folds.csv", index=False)

cv_summary = (
    cv.groupby("model")
    .agg(
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
    )
    .sort_values("log_loss_mean")
)
cv_summary.to_csv("classification_polynomial_cv_summary.csv")

fit_summaries = []
coefs_to_write = {}
for model_name, terms in MODEL_SPECS.items():
    fit, coef = fit_full_data_model(data, terms)
    coefs_to_write[model_name] = coef
    coef.to_csv(f"classification_{model_name}_coefficients.csv", index=False)
    fit_summaries.append(
        {
            "model": model_name,
            "df_model": int(fit.df_model),
            "aic": fit.aic,
            "bic_deviance": fit.bic_deviance,
            "deviance": fit.deviance,
        }
    )
pd.DataFrame(fit_summaries).sort_values("aic").to_csv("classification_polynomial_full_data_fit_stats.csv", index=False)

plot_df = cv.melt(
    id_vars=["model", "fold"],
    value_vars=["log_loss", "auc", "accuracy", "f1"],
    var_name="metric",
    value_name="value",
)
plt.figure(figsize=(12, 5.8))
sns.barplot(data=plot_df, x="model", y="value", hue="metric", errorbar="sd")
plt.title("Logistic polynomial comparison")
plt.xlabel("")
plt.xticks(rotation=24, ha="right")
savefig("classification_polynomial_cv_metrics.png")

plt.figure(figsize=(8.6, 5.2))
ranked = cv_summary.reset_index()
sns.pointplot(data=ranked, x="log_loss_mean", y="model", join=False, color="#2f6f9f")
for i, row in ranked.iterrows():
    plt.hlines(
        y=i,
        xmin=row["log_loss_mean"] - row["log_loss_sd"],
        xmax=row["log_loss_mean"] + row["log_loss_sd"],
        color="#2f6f9f",
        alpha=0.7,
    )
plt.title("Mean CV log-loss with +/- 1 SD")
plt.xlabel("Lower is better")
plt.ylabel("")
savefig("classification_polynomial_logloss_rank.png")

best = cv_summary.index[0]
summary_lines = [
    "Focused logistic polynomial comparison",
    "======================================",
    "",
    "Question:",
    "Can interactions or squared covariates improve logistic regression over a purely linear boundary?",
    "",
    "Why x and x^2 together are justified:",
    (
        "Including x and x^2 is standard in polynomial logistic regression. The x term captures "
        "a directional/asymmetric shift, while x^2 captures curvature: cases where unusually "
        "large positive or negative values change the class probability. Keeping both terms "
        "also preserves hierarchy, so the squared effect is not forced to be symmetric around zero "
        "unless the data support that."
    ),
    "",
    "Single validation split:",
    single.round(4).to_string(index=False),
    "",
    "Five-fold cross-validation summary:",
    cv_summary.round(4).to_string(),
    "",
    f"Current best by mean CV log-loss: {best}",
    "",
    "Practical interpretation:",
    (
        "If the full interaction model beats the square-only models, the main nonlinearity is likely "
        "from variables modifying each other. If square-only models beat interactions, curvature in "
        "individual predictors is more important. If squares_plus_interactions wins but only slightly, "
        "we should balance the small gain against wider confidence intervals and possible instability."
    ),
]
with open("classification_polynomial_comparison_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("Wrote polynomial comparison outputs:")
for filename in [
    "classification_polynomial_single_split_comparison.csv",
    "classification_polynomial_cv_folds.csv",
    "classification_polynomial_cv_summary.csv",
    "classification_polynomial_full_data_fit_stats.csv",
    "classification_polynomial_cv_metrics.png",
    "classification_polynomial_logloss_rank.png",
    "classification_polynomial_comparison_summary.txt",
]:
    print(f"  {filename}")
