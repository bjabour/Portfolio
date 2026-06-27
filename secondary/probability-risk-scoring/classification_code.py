import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


TRAIN_PATH = "classification_train.csv"
TEST_PATH = "classification_test.csv"
RESULTS_PATH = "classification_results.csv"
RANDOM_STATE = 9618

BASE_FEATURES = ["x1", "x2", "x3"]
LOGISTIC_TERMS = ["x1", "x1_sq", "x2_x3"]


def sigmoid(z):
    ### Converts log-odds to probabilities
    z = np.clip(z, -700, 700)
    return 1 / (1 + np.exp(-z))


def add_model_terms(df: pd.DataFrame) -> pd.DataFrame:
    ### Adds the nonlinear terms selected during validation.
    out = df.copy()
    out["x1_sq"] = out["x1"] ** 2
    out["x2_x3"] = out["x2"] * out["x3"]
    return out


def design_matrix(df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    ### Builds the logistic regression design matrix with an intercept
    return sm.add_constant(df[terms], has_constant="add").to_numpy()


def fit_logistic_glm(df: pd.DataFrame, terms: list[str]):
    ### Fits logistic regression
    y = df["y"].to_numpy()
    x = design_matrix(df, terms)
    model = sm.GLM(y, x, family=sm.families.Binomial())
    return model.fit(maxiter=200)


def classification_metrics(y_true, prob_class1, threshold=0.5) -> dict:
    ### Calculates validation metrics from predicted class-1 probabilities
    pred = (prob_class1 >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "log_loss": log_loss(y_true, np.clip(prob_class1, 1e-12, 1 - 1e-12)),
        "auc": roc_auc_score(y_true, prob_class1),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


#===========================================================
#   LOADING DATA AND ADDING MODEL TERMS

raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
train = add_model_terms(raw_train)
test = add_model_terms(raw_test)

#===========================================================
#   VALIDATION COMPARISON

# Validation comparison between the required logistic method and QDA.
train_part, valid_part = train_test_split(
    train,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=train["y"],
)

logistic_valid_fit = fit_logistic_glm(train_part, LOGISTIC_TERMS)
valid_logistic_x = design_matrix(valid_part, LOGISTIC_TERMS)
valid_logistic_prob = logistic_valid_fit.predict(valid_logistic_x)

qda = QuadraticDiscriminantAnalysis()
qda.fit(train_part[BASE_FEATURES], train_part["y"])
valid_qda_prob = qda.predict_proba(valid_part[BASE_FEATURES])[:, 1]

comparison = pd.DataFrame(
    [
        {"method": "Selected logistic regression", **classification_metrics(valid_part["y"], valid_logistic_prob)},
        {"method": "QDA", **classification_metrics(valid_part["y"], valid_qda_prob)},
    ]
)

#===========================================================
#   FITTING FINAL LOGISTIC MODEL

# Final predictions must come from logistic regression, fit on all training data.
final_fit = fit_logistic_glm(train, LOGISTIC_TERMS)
test_x = design_matrix(test, LOGISTIC_TERMS)
eta = test_x @ np.asarray(final_fit.params)
cov_beta = np.asarray(final_fit.cov_params())
eta_var = np.sum((test_x @ cov_beta) * test_x, axis=1)
eta_se = np.sqrt(np.maximum(eta_var, 0))

z_value = 1.96
prob_class1 = sigmoid(eta)
ci_lower = sigmoid(eta - z_value * eta_se)
ci_upper = sigmoid(eta + z_value * eta_se)

#===========================================================
#   SAVING SUBMISSION FILE

results = pd.DataFrame(
    {
        "prob_class1": prob_class1,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }
)
results.to_csv(RESULTS_PATH, index=False, float_format="%.10f")

print("Validation comparison:")
print(comparison.round(4).to_string(index=False))
print()
print(f"Wrote {RESULTS_PATH} with {len(results)} test-set predictions.")
