import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


warnings.filterwarnings("ignore")

TRAIN_PATH = os.path.join("..", "classification_train.csv")
TEST_PATH = os.path.join("..", "classification_test.csv")
RANDOM_STATE = 9618
BASE_FEATURES = ["x1", "x2", "x3"]


def add_terms(df):
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_sq"] = out["x2"] ** 2
    out["x3_sq"] = out["x3"] ** 2
    out["x1_x2"] = out["x1"] * out["x2"]
    out["x1_x3"] = out["x1"] * out["x3"]
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


def glm_predict(train_df, valid_df, terms):
    x_train = sm.add_constant(train_df[terms], has_constant="add")
    x_valid = sm.add_constant(valid_df[terms], has_constant="add")
    fit = sm.GLM(train_df["y"], x_train, family=sm.families.Binomial()).fit(maxiter=200)
    return np.clip(fit.predict(x_valid), 1e-12, 1 - 1e-12)


def sklearn_pipeline(degree, penalty, c_value):
    if penalty == "l2":
        clf = LogisticRegression(
            penalty="l2",
            C=c_value,
            solver="lbfgs",
            max_iter=6000,
            random_state=RANDOM_STATE,
        )
    elif penalty == "l1":
        clf = LogisticRegression(
            penalty="l1",
            C=c_value,
            solver="liblinear",
            max_iter=6000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported penalty: {penalty}")

    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("logistic", clf),
        ]
    )


def score_prob(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "log_loss": log_loss(y_true, prob),
        "auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
train = add_terms(raw_train)
test = add_terms(raw_test)

glm_candidates = {
    "glm_current_simple": ["x1", "x2", "x3", "x1_sq", "x2_x3"],
    "glm_best_subset": ["x1", "x2", "x3", "x1_sq", "x3_sq", "x1_x3", "x2_x3"],
    "glm_full_degree2": [
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

c_values = [0.03, 0.1, 0.3, 1, 3, 10, 30]
sklearn_candidates = []
for degree in range(1, 7):
    for c_value in c_values:
        sklearn_candidates.append((f"ridge_poly{degree}_C{c_value}", degree, "l2", c_value))
for degree in range(2, 7):
    for c_value in c_values:
        sklearn_candidates.append((f"lasso_poly{degree}_C{c_value}", degree, "l1", c_value))

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
fold_rows = []

for fold_id, (train_idx, valid_idx) in enumerate(cv.split(train[BASE_FEATURES], train["y"]), start=1):
    train_fold = train.iloc[train_idx].copy()
    valid_fold = train.iloc[valid_idx].copy()
    y_valid = valid_fold["y"].to_numpy()

    for name, terms in glm_candidates.items():
        prob = glm_predict(train_fold, valid_fold, terms)
        row = {"model": name, "family": "statsmodels_glm", "fold": fold_id, "degree": "", "penalty": "none", "C": ""}
        row.update(score_prob(y_valid, prob))
        fold_rows.append(row)

    for name, degree, penalty, c_value in sklearn_candidates:
        pipe = sklearn_pipeline(degree, penalty, c_value)
        pipe.fit(train_fold[BASE_FEATURES], train_fold["y"])
        prob = np.clip(pipe.predict_proba(valid_fold[BASE_FEATURES])[:, 1], 1e-12, 1 - 1e-12)
        row = {"model": name, "family": "sklearn_regularized", "fold": fold_id, "degree": degree, "penalty": penalty, "C": c_value}
        row.update(score_prob(y_valid, prob))
        fold_rows.append(row)

fold_scores = pd.DataFrame(fold_rows)
fold_scores.to_csv("classification_best_logistic_cv_folds.csv", index=False)

summary = (
    fold_scores.groupby(["model", "family", "degree", "penalty", "C"], dropna=False)
    .agg(
        log_loss_mean=("log_loss", "mean"),
        log_loss_sd=("log_loss", "std"),
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
    )
    .reset_index()
    .sort_values(["log_loss_mean", "log_loss_sd", "model"])
)
summary.to_csv("classification_best_logistic_cv_all_models.csv", index=False)
summary.head(25).to_csv("classification_best_logistic_cv_top25.csv", index=False)

best = summary.iloc[0]
top = summary.head(15).copy()
plt.figure(figsize=(10.8, 6.5))
colors = ["#2f6f9f" if family == "statsmodels_glm" else "#3a8f60" for family in top["family"]]
plt.barh(top["model"], top["log_loss_mean"], xerr=top["log_loss_sd"], color=colors, alpha=0.86)
plt.gca().invert_yaxis()
plt.xlabel("Repeated 5-fold CV log-loss, lower is better")
plt.title("Best logistic-regression candidates")
plt.tight_layout()
plt.savefig("classification_best_logistic_top15_logloss.png", dpi=180, bbox_inches="tight")
plt.close()

best_model_name = best["model"]
best_test_probs = None
best_feature_table = None

if best["family"] == "statsmodels_glm":
    best_terms = glm_candidates[best_model_name]
    full_fit = sm.GLM(train["y"], sm.add_constant(train[best_terms], has_constant="add"), family=sm.families.Binomial()).fit(maxiter=200)
    best_test_probs = full_fit.predict(sm.add_constant(test[best_terms], has_constant="add"))
    best_feature_table = pd.DataFrame(
        {
            "term": ["Intercept"] + best_terms,
            "coef": full_fit.params,
            "p_value": full_fit.pvalues,
        }
    )
else:
    best_degree = int(best["degree"])
    best_penalty = best["penalty"]
    best_c = float(best["C"])
    best_pipe = sklearn_pipeline(best_degree, best_penalty, best_c)
    best_pipe.fit(train[BASE_FEATURES], train["y"])
    best_test_probs = best_pipe.predict_proba(raw_test[BASE_FEATURES])[:, 1]

    poly = best_pipe.named_steps["poly"]
    feature_names = poly.get_feature_names_out(BASE_FEATURES)
    coefs = best_pipe.named_steps["logistic"].coef_[0]
    coef_order = np.argsort(np.abs(coefs))[::-1]
    best_feature_table = pd.DataFrame(
        {
            "term": feature_names[coef_order],
            "scaled_coef": coefs[coef_order],
        }
    )

pd.DataFrame({"prob_class1": np.clip(best_test_probs, 1e-12, 1 - 1e-12)}).to_csv(
    "classification_best_logistic_temp_test_probabilities.csv",
    index=False,
    float_format="%.10f",
)
best_feature_table.to_csv("classification_best_logistic_coefficients.csv", index=False)

current = summary.loc[summary["model"] == "glm_current_simple"].iloc[0]
improvement = current["log_loss_mean"] - best["log_loss_mean"]
if best["family"] == "statsmodels_glm":
    interpretation = (
        "The richer regularized polynomial models did not beat the simpler GLM on repeated-CV "
        "log-loss. This suggests that the useful nonlinear signal is already captured by "
        "x1^2 and x2*x3, while higher-degree terms mostly add variance or overconfidence. "
        "For both prediction and the assignment's required GLM confidence intervals, the "
        "simple selected statsmodels model remains the best choice from this experiment."
    )
else:
    interpretation = (
        "The best regularized polynomial logistic model beat the unpenalized GLM. "
        "Regularization lets the model use richer nonlinear features while controlling "
        "overconfidence. For the submitted assignment, however, an unpenalized statsmodels "
        "GLM is still cleaner because the required probability confidence intervals come "
        "directly from the GLM covariance matrix."
    )

lines = [
    "Best temp-only logistic regression experiment",
    "============================================",
    "",
    "Goal: minimize validation log-loss using logistic regression only.",
    "Comparison used repeated stratified 5-fold CV with 5 repeats (25 validation folds).",
    "",
    "Models compared:",
    "- statsmodels GLMs: current simple model, previous best subset, full degree-2 GLM",
    "- regularized sklearn logistic regressions with polynomial features up to degree 6",
    "- ridge (L2) and lasso (L1) penalties over C = " + ", ".join(str(v) for v in c_values),
    "",
    "Top 25 models:",
    summary.head(25)[
        [
            "model",
            "family",
            "degree",
            "penalty",
            "C",
            "log_loss_mean",
            "log_loss_sd",
            "auc_mean",
            "accuracy_mean",
            "f1_mean",
        ]
    ].round(4).to_string(index=False),
    "",
    f"Best model: {best_model_name}",
    f"Best mean CV log-loss: {best['log_loss_mean']:.4f}",
    f"Best log-loss SD: {best['log_loss_sd']:.4f}",
    f"Current selected GLM mean CV log-loss: {current['log_loss_mean']:.4f}",
    f"Improvement over current selected GLM: {improvement:.4f}",
    "",
    "Interpretation:",
    interpretation,
]

with open("classification_best_logistic_experiment_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Wrote best-logistic experiment outputs:")
for filename in [
    "classification_best_logistic_cv_folds.csv",
    "classification_best_logistic_cv_all_models.csv",
    "classification_best_logistic_cv_top25.csv",
    "classification_best_logistic_top15_logloss.png",
    "classification_best_logistic_temp_test_probabilities.csv",
    "classification_best_logistic_coefficients.csv",
    "classification_best_logistic_experiment_summary.txt",
]:
    print(f"  {filename}")
print()
print(f"Best model: {best_model_name}")
print(f"Best mean CV log-loss: {best['log_loss_mean']:.4f}")
