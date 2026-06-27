import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


#===========================================================
#   FILE PATHS AND RUN SETTINGS

TRAIN_PATH = Path("regularization_train.csv")
TEST_PATH = Path("regularization_test.csv")
LAMBDA_GRID_PATH = Path("regularization_lambda_grid.csv")
RESULTS_PATH = Path("regularization_results.json")

STUDENT_ID = 9618
RANDOM_STATE = STUDENT_ID
N_FOLDS = 5
B_BOOTSTRAP = 200
NONZERO_TOL = 1e-8


#===========================================================
#   STANDARDIZED MODEL CONTAINER

@dataclass
class StandardizedLinearModel:
    ### Stores coefficients fitted on standardized x and converts them back to raw scale
    x_mean: np.ndarray
    x_scale: np.ndarray
    intercept_std: float
    beta_std: np.ndarray

    @property
    def beta_raw(self) -> np.ndarray:
        ### Converts standardized coefficients to coefficients on the original x scale
        return self.beta_std / self.x_scale

    @property
    def intercept_raw(self) -> float:
        ### Converts the standardized intercept to the original x scale
        return float(self.intercept_std - np.dot(self.x_mean / self.x_scale, self.beta_std))

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        ### Predicts y for raw, unstandardized predictor values
        x_z = standardize_apply(x_raw, self.x_mean, self.x_scale)
        return self.intercept_std + x_z @ self.beta_std


#===========================================================
#   FEATURE AND STANDARDIZATION HELPERS

def feature_names() -> list[str]:
    ### Returns the required x1 through x40 feature names in order
    return [f"x{i}" for i in range(1, 41)]


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ### Computes column means and SDs used to standardize predictors
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0, ddof=0)
    x_scale = np.where(x_scale == 0, 1.0, x_scale)
    return x_mean, x_scale


def standardize_apply(x: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray) -> np.ndarray:
    ### Applies a previously fitted standardization rule
    return (x - x_mean) / x_scale


#===========================================================
#   RIDGE REGRESSION FROM SCRATCH

def fit_ridge_closed_form(x_raw: np.ndarray, y: np.ndarray, lam: float) -> StandardizedLinearModel:
    ### Fits ridge using the closed-form matrix solution
    x_mean, x_scale = standardize_fit(x_raw)
    x_z = standardize_apply(x_raw, x_mean, x_scale)
    y_mean = float(y.mean())
    y_centered = y - y_mean

    n, p = x_z.shape
    gram = (x_z.T @ x_z) / n
    rhs = (x_z.T @ y_centered) / n
    beta_std = np.linalg.solve(gram + lam * np.eye(p), rhs)
    return StandardizedLinearModel(x_mean, x_scale, y_mean, beta_std)


def fit_ridge_gradient_descent(
    x_raw: np.ndarray,
    y: np.ndarray,
    lam: float,
    tolerance: float = 1e-12,
    max_iter: int = 750_000,
) -> tuple[StandardizedLinearModel, int, float]:
    ### Fits ridge using explicit gradient descent on the ridge objective
    x_mean, x_scale = standardize_fit(x_raw)
    x_z = standardize_apply(x_raw, x_mean, x_scale)
    y_mean = float(y.mean())
    y_centered = y - y_mean

    n, p = x_z.shape
    beta = np.zeros(p)
    largest_eigenvalue = float(np.linalg.eigvalsh((x_z.T @ x_z) / n).max())
    step_size = 0.95 / (2.0 * (largest_eigenvalue + lam))

    # gradient descent update: beta <- beta - step_size * gradient
    last_step_norm = np.inf
    for iteration in range(1, max_iter + 1):
        residual = y_centered - x_z @ beta
        gradient = -(2.0 / n) * (x_z.T @ residual) + 2.0 * lam * beta
        beta_next = beta - step_size * gradient
        last_step_norm = float(np.linalg.norm(beta_next - beta))
        beta_scale = max(1.0, float(np.linalg.norm(beta_next)))
        beta = beta_next
        if last_step_norm <= tolerance * beta_scale:
            break
    else:
        print("Warning: ridge gradient descent reached max_iter before tolerance.")

    model = StandardizedLinearModel(x_mean, x_scale, y_mean, beta)
    return model, iteration, last_step_norm


#===========================================================
#   MANUAL 5-FOLD CROSS-VALIDATION FOR RIDGE

def make_cv_folds(n_rows: int, n_folds: int = N_FOLDS, random_state: int = RANDOM_STATE) -> list[np.ndarray]:
    ### Creates deterministic shuffled folds for cross-validation
    rng = np.random.default_rng(random_state)
    shuffled_index = rng.permutation(n_rows)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(shuffled_index, n_folds)]


def ridge_cv_mse(x: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    ### Computes mean validation MSE for every lambda in the required grid
    n_rows = len(y)
    all_index = np.arange(n_rows)
    folds = make_cv_folds(n_rows)
    cv_mse = np.zeros(len(lambdas))

    for lambda_index, lam in enumerate(lambdas):
        fold_mse = []
        for validation_index in folds:
            is_validation = np.zeros(n_rows, dtype=bool)
            is_validation[validation_index] = True
            train_index = all_index[~is_validation]

            # Standardization is fitted only on the fold's training rows
            model = fit_ridge_closed_form(x[train_index], y[train_index], float(lam))
            validation_pred = model.predict(x[validation_index])
            fold_mse.append(np.mean((y[validation_index] - validation_pred) ** 2))
        cv_mse[lambda_index] = float(np.mean(fold_mse))

    return cv_mse


#===========================================================
#   LASSO REGRESSION USING LIBRARIES

def fit_lasso_cv(x_raw: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> tuple[StandardizedLinearModel, LassoCV]:
    ### Fits lasso with sklearn cross-validation, allowed for Tasks 2 and 3
    x_mean, x_scale = standardize_fit(x_raw)
    x_z = standardize_apply(x_raw, x_mean, x_scale)
    lasso = LassoCV(
        cv=N_FOLDS,
        random_state=random_state,
        max_iter=200_000,
        tol=1e-8,
    )
    lasso.fit(x_z, y)
    model = StandardizedLinearModel(x_mean, x_scale, float(lasso.intercept_), lasso.coef_.copy())
    return model, lasso


#===========================================================
#   BOOTSTRAP LASSO STABILITY

def bootstrap_lasso_selection_frequencies(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ### Repeatedly refits lasso on bootstrap samples and records variable selection
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows, n_features = x.shape
    selected_counts = np.zeros(n_features, dtype=float)

    for _ in range(B_BOOTSTRAP):
        bootstrap_index = rng.integers(0, n_rows, size=n_rows)
        bootstrap_model, _ = fit_lasso_cv(x[bootstrap_index], y[bootstrap_index])
        selected_counts += (np.abs(bootstrap_model.beta_std) > NONZERO_TOL).astype(float)

    return selected_counts / B_BOOTSTRAP


#===========================================================
#   OUTPUT FORMATTING

def to_float_list(values: np.ndarray) -> list[float]:
    ### Converts numpy arrays into JSON-safe Python float lists
    return [float(v) for v in np.asarray(values, dtype=float)]


#===========================================================
#   FULL PROJECT SOLUTION

def solve_project() -> dict:
    ### Runs ridge, lasso, and bootstrap stability and returns the JSON payload

    #===========================================================
    #   LOADING DATA

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    lambdas = pd.read_csv(LAMBDA_GRID_PATH)["lambda"].to_numpy(dtype=float)

    features = feature_names()
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)

    #===========================================================
    #   FITTING RIDGE MODEL

    # Manual 5-fold CV chooses the lambda with the smallest average validation MSE
    cv_mse = ridge_cv_mse(x_train, y_train, lambdas)
    best_lambda = float(lambdas[int(np.argmin(cv_mse))])
    ridge_model = fit_ridge_closed_form(x_train, y_train, best_lambda)
    ridge_pred_test = ridge_model.predict(x_test)

    # Gradient descent uses the same selected lambda and checks the closed-form solution
    ridge_gd_model, gd_iterations, gd_last_step = fit_ridge_gradient_descent(x_train, y_train, best_lambda)
    ridge_gd_pred_test = ridge_gd_model.predict(x_test)

    #===========================================================
    #   FITTING LASSO MODEL

    # Lasso library CV is allowed by the project for Tasks 2 and 3
    lasso_model, lasso_fit = fit_lasso_cv(x_train, y_train)
    lasso_pred_test = lasso_model.predict(x_test)
    lasso_selected_mask = np.abs(lasso_model.beta_std) > NONZERO_TOL
    lasso_selected_vars = [name for name, selected in zip(features, lasso_selected_mask) if selected]

    #===========================================================
    #   COMPUTING BOOTSTRAP SELECTION FREQUENCIES

    bootstrap_freq = bootstrap_lasso_selection_frequencies(x_train, y_train)

    #===========================================================
    #   BUILDING REQUIRED JSON STRUCTURE

    results = {
        "ridge_pred_test": to_float_list(ridge_pred_test),
        "ridge_gd_pred_test": to_float_list(ridge_gd_pred_test),
        "ridge_lambda": best_lambda,
        "ridge_cv_mse": to_float_list(cv_mse),
        "ridge_beta": to_float_list(ridge_model.beta_raw),
        "ridge_gd_beta": to_float_list(ridge_gd_model.beta_raw),
        "lasso_pred_test": to_float_list(lasso_pred_test),
        "lasso_alpha": float(lasso_fit.alpha_),
        "lasso_n_nonzero": int(lasso_selected_mask.sum()),
        "lasso_selected_vars": lasso_selected_vars,
        "lasso_beta": to_float_list(lasso_model.beta_raw),
        "bootstrap_selection_freq": to_float_list(bootstrap_freq),
    }

    diagnostic = {
        "ridge_gd_iterations": int(gd_iterations),
        "ridge_gd_last_step_norm": float(gd_last_step),
        "ridge_gd_vs_closed_form_rmse": float(np.sqrt(np.mean((ridge_gd_pred_test - ridge_pred_test) ** 2))),
        "ridge_train_intercept": ridge_model.intercept_raw,
        "lasso_train_intercept": lasso_model.intercept_raw,
    }
    return results, diagnostic


def main():
    ### Writes the required JSON results file and prints a short run summary

    #===========================================================
    #   RUNNING SOLUTION AND SAVING FILE

    results, diagnostic = solve_project()
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    #===========================================================
    #   PRINTING RUN SUMMARY

    print(f"saved: {RESULTS_PATH}")
    print(f"ridge lambda: {results['ridge_lambda']:.10f}")
    print(f"ridge minimum CV MSE: {min(results['ridge_cv_mse']):.6f}")
    print(f"ridge GD iterations: {diagnostic['ridge_gd_iterations']}")
    print(f"ridge GD vs closed-form test RMSE: {diagnostic['ridge_gd_vs_closed_form_rmse']:.12f}")
    print(f"lasso alpha: {results['lasso_alpha']:.10f}")
    print(f"lasso nonzero coefficients: {results['lasso_n_nonzero']}")
    print(f"lasso selected variables: {', '.join(results['lasso_selected_vars'])}")


if __name__ == "__main__":
    main()
