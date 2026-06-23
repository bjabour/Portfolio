import numpy as np
import pandas as pd
from scipy import stats

PREDICTORS = ["x1", "x2", "x3", "x4", "x5"]
FINAL_MODEL = ["x1", "x3_c", "x4_c", "x5", "x3c_x4c"]
TRAIN_PATH = "regression_train.csv"
TEST_PATH = "regression_test.csv"
RESULTS_PATH = "regression_results.csv"


def add_centered_terms(df: pd.DataFrame, x3_mean: float, x4_mean: float) -> pd.DataFrame:
    ### Adds centered covariates and their interaction
    out = df.copy()
    out["x3_c"] = out["x3"] - x3_mean
    out["x4_c"] = out["x4"] - x4_mean
    out["x3c_x4c"] = out["x3_c"] * out["x4_c"]
    return out


def fit_ols(df: pd.DataFrame, predictors: list[str]) -> dict:
    ### This function returns regression parameters
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[predictors].to_numpy()])  # defining covariate matrix
    coef = np.linalg.lstsq(X, y, rcond=None)[0]  # getting regression coefs
    fitted = X @ coef
    resid = y - fitted
    n = len(df)
    p = X.shape[1]  # includes intercept
    dof = n - p
    sse = float(resid @ resid)

    sigma = float(np.sqrt(sse / dof))  # estimate of error SD
    xtx_inv = np.linalg.inv(X.T @ X)  # variance-covariance matrix
    se = sigma * np.sqrt(np.diag(xtx_inv))

    # t statistics and two-sided p-values
    t_stat = coef / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=dof))

    # adjusted R-squared
    sst = float(((y - y.mean()) @ (y - y.mean())))
    r2 = 1 - sse / sst
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof

    # AIC and BIC
    loglik = -n / 2 * (np.log(2 * np.pi) + 1 + np.log(sse / n))
    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + np.log(n) * p

    return {
        "predictors": predictors,
        "coef": coef,
        "fitted": fitted,
        "resid": resid,
        "sigma": sigma,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "adj_r2": adj_r2,
        "aic": aic,
        "bic": bic,
    }


def design_matrix(df: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    ### Builds the design matrix for prediction
    return np.column_stack([np.ones(len(df)), df[predictors].to_numpy()])


raw_train = pd.read_csv(TRAIN_PATH)
raw_test = pd.read_csv(TEST_PATH)
x3_mean = raw_train["x3"].mean()
x4_mean = raw_train["x4"].mean()
train = add_centered_terms(raw_train, x3_mean, x4_mean)
test = add_centered_terms(raw_test, x3_mean, x4_mean)

final_fit = fit_ols(train, FINAL_MODEL)

#===========================================================
#   GENERATING PREDICTIONS

predicted_mean = design_matrix(test, FINAL_MODEL) @ final_fit["coef"]
predicted_sd = np.full(len(test), final_fit["sigma"])

#===========================================================
#   SAVING SUBMISSION FILE

pd.DataFrame(
    {
        "predicted_mean": np.round(predicted_mean, 4),
        "predicted_sd": np.round(predicted_sd, 4),
    }
).to_csv(RESULTS_PATH, index=False)

print(f"train rows: {len(train)}")
print(f"test rows: {len(test)}")
print(f"selected predictors: {FINAL_MODEL}")
print(f"sigma_hat: {final_fit['sigma']:.4f}")
print(f"saved: {RESULTS_PATH}")
