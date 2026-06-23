from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

from regularization_code import (
    LAMBDA_GRID_PATH,
    N_FOLDS,
    RANDOM_STATE,
    TRAIN_PATH,
    feature_names,
    fit_ridge_closed_form,
    standardize_apply,
    standardize_fit,
)


#===========================================================
#   EXPERIMENT SETTINGS

RESULTS_PATH = Path("regularization_error_decomposition_results.csv")
SUMMARY_PATH = Path("regularization_error_decomposition_summary.txt")
ONE_SE_OBSERVED_PATH = Path("regularization_lasso_one_se_observed.csv")

N_SIMULATIONS = 50
N_TRAIN = 150
N_TEST = 5000
REPEATED_CV_SPLITS = 5

TRUE_ACTIVE = ["x1", "x8", "x12", "x15", "x20", "x24", "x31", "x34"]
TRUE_ACTIVE_BETAS = np.array([1.82658, 1.09470, -2.3844, -0.74875, -2.39174, 1.1904, 1.83936, 2.48072])
TRUE_INTERCEPT = -0.95
NOISE_SD = 2.8758
NOISE_VARIANCE = 8.2703
TARGET_SNR = 3.73
POPULATION_RIDGE_LAMBDA = 0.08396


#===========================================================
#   CALIBRATED PSEUDO DATA-GENERATING PROCESS

def positive_definite_covariance(x: np.ndarray) -> np.ndarray:
    ### Estimates predictor covariance and clips tiny eigenvalues for stable simulation
    covariance = np.cov(x, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    clipped = np.maximum(eigenvalues, 1e-8)
    return (eigenvectors * clipped) @ eigenvectors.T


def oracle_beta_vector() -> np.ndarray:
    ### Places the exact oracle coefficients in the full 40-variable coefficient vector
    features = feature_names()
    beta = np.zeros(len(features))
    active_index = [features.index(name) for name in TRUE_ACTIVE]
    beta[active_index] = TRUE_ACTIVE_BETAS
    return beta


def simulate_dataset(
    rng: np.random.Generator,
    x_mean: np.ndarray,
    x_covariance: np.ndarray,
    beta_true: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ### Generates correlated predictors, noiseless means, and noisy responses
    x = rng.multivariate_normal(x_mean, x_covariance, size=n_rows)
    mean_y = TRUE_INTERCEPT + x @ beta_true
    y = mean_y + rng.normal(0.0, NOISE_SD, size=n_rows)
    return x, y, mean_y


#===========================================================
#   RIDGE TUNING AND EVALUATION

def make_folds(n_rows: int, seed: int) -> list[np.ndarray]:
    ### Creates deterministic shuffled 5-fold validation indexes
    rng = np.random.default_rng(seed)
    return [np.asarray(fold, dtype=int) for fold in np.array_split(rng.permutation(n_rows), N_FOLDS)]


def ridge_cv_curve(x: np.ndarray, y: np.ndarray, lambdas: np.ndarray, seed: int) -> np.ndarray:
    ### Computes a leakage-safe ridge CV curve for one fold partition
    all_index = np.arange(len(y))
    curve = np.zeros(len(lambdas))

    for lambda_index, lam in enumerate(lambdas):
        fold_mse = []
        for validation_index in make_folds(len(y), seed):
            is_validation = np.zeros(len(y), dtype=bool)
            is_validation[validation_index] = True
            train_index = all_index[~is_validation]

            model = fit_ridge_closed_form(x[train_index], y[train_index], float(lam))
            prediction = model.predict(x[validation_index])
            fold_mse.append(np.mean((y[validation_index] - prediction) ** 2))
        curve[lambda_index] = float(np.mean(fold_mse))

    return curve


def ridge_test_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    ### Evaluates all full-training ridge fits on the large simulated test set
    test_mse = np.zeros(len(lambdas))
    for index, lam in enumerate(lambdas):
        model = fit_ridge_closed_form(x_train, y_train, float(lam))
        test_mse[index] = np.mean((y_test - model.predict(x_test)) ** 2)
    return test_mse


#===========================================================
#   LASSO TUNING AND EVALUATION

def fit_current_lasso(x: np.ndarray, y: np.ndarray) -> tuple[LassoCV, np.ndarray, np.ndarray]:
    ### Reproduces the submitted global-scaling plus LassoCV workflow
    x_mean, x_scale = standardize_fit(x)
    x_z = standardize_apply(x, x_mean, x_scale)
    model = LassoCV(cv=N_FOLDS, max_iter=100_000, tol=1e-7)
    model.fit(x_z, y)
    return model, x_mean, x_scale


def one_se_alpha(alphas: np.ndarray, fold_mse: np.ndarray) -> tuple[float, float, float]:
    ### Selects the largest alpha within one standard error of minimum mean CV MSE
    mean_mse = fold_mse.mean(axis=1)
    minimum_index = int(np.argmin(mean_mse))
    minimum_se = float(fold_mse[minimum_index].std(ddof=1) / np.sqrt(fold_mse.shape[1]))
    threshold = float(mean_mse[minimum_index] + minimum_se)
    eligible = alphas[mean_mse <= threshold]
    return float(np.max(eligible)), threshold, minimum_se


def lasso_alpha_grid(x: np.ndarray, y: np.ndarray, n_alphas: int = 100) -> np.ndarray:
    ### Reconstructs the standard log-spaced lasso alpha range
    x_mean, x_scale = standardize_fit(x)
    x_z = standardize_apply(x, x_mean, x_scale)
    y_centered = y - y.mean()
    alpha_max = np.max(np.abs(x_z.T @ y_centered)) / len(y)
    return np.geomspace(alpha_max, alpha_max * 1e-3, n_alphas)


def strict_lasso_cv_curve(
    x: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    seed: int,
) -> np.ndarray:
    ### Returns fold-level MSE after standardizing separately inside each fold
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(splitter.split(x))
    fold_mse = np.zeros((len(alphas), N_FOLDS))

    for alpha_index, alpha in enumerate(alphas):
        for fold_index, (train_index, validation_index) in enumerate(splits):
            x_mean, x_scale = standardize_fit(x[train_index])
            x_train_z = standardize_apply(x[train_index], x_mean, x_scale)
            x_validation_z = standardize_apply(x[validation_index], x_mean, x_scale)

            model = Lasso(alpha=float(alpha), max_iter=100_000, tol=1e-7)
            model.fit(x_train_z, y[train_index])
            prediction = model.predict(x_validation_z)
            fold_mse[alpha_index, fold_index] = np.mean((y[validation_index] - prediction) ** 2)

    return fold_mse


def fit_lasso_at_alpha(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[Lasso, np.ndarray, np.ndarray]:
    ### Fits one lasso model on all training rows using full-training scaling
    x_mean, x_scale = standardize_fit(x)
    x_z = standardize_apply(x, x_mean, x_scale)
    model = Lasso(alpha=float(alpha), max_iter=100_000, tol=1e-7)
    model.fit(x_z, y)
    return model, x_mean, x_scale


def lasso_test_mse(
    model: Lasso,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    ### Computes test MSE using the training-set standardization parameters
    prediction = model.predict(standardize_apply(x_test, x_mean, x_scale))
    return float(np.mean((y_test - prediction) ** 2))


def selection_metrics(coef: np.ndarray) -> dict:
    ### Compares a fitted lasso support with the eight exact active predictors
    features = feature_names()
    selected = np.abs(coef) > 1e-8
    true_active = np.array([name in TRUE_ACTIVE for name in features])
    return {
        "nonzero": int(selected.sum()),
        "tp": int(np.sum(selected & true_active)),
        "fp": int(np.sum(selected & ~true_active)),
        "fn": int(np.sum(~selected & true_active)),
    }


#===========================================================
#   ONE MONTE CARLO REPLICATION

def run_replication(
    simulation_index: int,
    x_mean: np.ndarray,
    x_covariance: np.ndarray,
    beta_true: np.ndarray,
    lambdas: np.ndarray,
) -> dict:
    ### Separates oracle noise, finite-sample/model cost, and tuning cost
    rng = np.random.default_rng(RANDOM_STATE + 10_000 + simulation_index)
    x_train, y_train, _ = simulate_dataset(rng, x_mean, x_covariance, beta_true, N_TRAIN)
    x_test, y_test, mean_test = simulate_dataset(rng, x_mean, x_covariance, beta_true, N_TEST)

    oracle_mse = float(np.mean((y_test - mean_test) ** 2))

    # Known-support OLS is a benchmark using the true eight-variable active set.
    active_index = [feature_names().index(name) for name in TRUE_ACTIVE]
    active_design = np.column_stack([np.ones(N_TRAIN), x_train[:, active_index]])
    active_beta = np.linalg.lstsq(active_design, y_train, rcond=None)[0]
    active_test_design = np.column_stack([np.ones(N_TEST), x_test[:, active_index]])
    active_ols_mse = float(np.mean((y_test - active_test_design @ active_beta) ** 2))

    # Ridge oracle tuning diagnoses the best achievable ridge fit for this training sample.
    ridge_path_mse = ridge_test_path(x_train, y_train, x_test, y_test, lambdas)
    ridge_oracle_index = int(np.argmin(ridge_path_mse))
    ridge_oracle_mse = float(ridge_path_mse[ridge_oracle_index])

    ridge_single_curve = ridge_cv_curve(x_train, y_train, lambdas, RANDOM_STATE)
    ridge_single_index = int(np.argmin(ridge_single_curve))
    ridge_single_mse = float(ridge_path_mse[ridge_single_index])

    repeated_curves = np.vstack(
        [
            ridge_cv_curve(x_train, y_train, lambdas, RANDOM_STATE + 100 * repeat)
            for repeat in range(REPEATED_CV_SPLITS)
        ]
    )
    ridge_repeated_index = int(np.argmin(repeated_curves.mean(axis=0)))
    ridge_repeated_mse = float(ridge_path_mse[ridge_repeated_index])

    fixed_index = int(np.argmin(np.abs(lambdas - POPULATION_RIDGE_LAMBDA)))
    ridge_fixed_mse = float(ridge_path_mse[fixed_index])

    # Current lasso reproduces the submitted preprocessing and default CV behavior.
    current_lasso, current_mean, current_scale = fit_current_lasso(x_train, y_train)
    lasso_current_mse = lasso_test_mse(current_lasso, current_mean, current_scale, x_test, y_test)
    current_selection = selection_metrics(current_lasso.coef_)

    current_one_se_alpha, _, _ = one_se_alpha(current_lasso.alphas_, current_lasso.mse_path_)
    current_one_se_model, one_se_mean, one_se_scale = fit_lasso_at_alpha(
        x_train, y_train, current_one_se_alpha
    )
    lasso_one_se_mse = lasso_test_mse(current_one_se_model, one_se_mean, one_se_scale, x_test, y_test)
    one_se_selection = selection_metrics(current_one_se_model.coef_)

    # Oracle lasso tuning chooses alpha using the large test set only as a diagnostic.
    alphas = lasso_alpha_grid(x_train, y_train)
    lasso_path_mse = np.zeros(len(alphas))
    for alpha_index, alpha in enumerate(alphas):
        model, model_mean, model_scale = fit_lasso_at_alpha(x_train, y_train, float(alpha))
        lasso_path_mse[alpha_index] = lasso_test_mse(model, model_mean, model_scale, x_test, y_test)
    lasso_oracle_index = int(np.argmin(lasso_path_mse))
    lasso_oracle_mse = float(lasso_path_mse[lasso_oracle_index])

    # Strict CV moves standardization inside each validation split.
    strict_fold_mse = strict_lasso_cv_curve(x_train, y_train, alphas, RANDOM_STATE)
    strict_curve = strict_fold_mse.mean(axis=1)
    strict_index = int(np.argmin(strict_curve))
    strict_model, strict_mean, strict_scale = fit_lasso_at_alpha(x_train, y_train, float(alphas[strict_index]))
    lasso_strict_mse = lasso_test_mse(strict_model, strict_mean, strict_scale, x_test, y_test)

    strict_one_se_alpha, _, _ = one_se_alpha(alphas, strict_fold_mse)
    strict_one_se_model, strict_one_se_mean, strict_one_se_scale = fit_lasso_at_alpha(
        x_train, y_train, strict_one_se_alpha
    )
    lasso_strict_one_se_mse = lasso_test_mse(
        strict_one_se_model, strict_one_se_mean, strict_one_se_scale, x_test, y_test
    )

    return {
        "simulation": simulation_index,
        "oracle_mse": oracle_mse,
        "active_ols_mse": active_ols_mse,
        "ridge_oracle_mse": ridge_oracle_mse,
        "ridge_single_cv_mse": ridge_single_mse,
        "ridge_repeated_cv_mse": ridge_repeated_mse,
        "ridge_fixed_population_lambda_mse": ridge_fixed_mse,
        "ridge_oracle_lambda": float(lambdas[ridge_oracle_index]),
        "ridge_single_cv_lambda": float(lambdas[ridge_single_index]),
        "ridge_repeated_cv_lambda": float(lambdas[ridge_repeated_index]),
        "lasso_oracle_mse": lasso_oracle_mse,
        "lasso_current_cv_mse": lasso_current_mse,
        "lasso_one_se_mse": lasso_one_se_mse,
        "lasso_strict_fold_scaled_cv_mse": lasso_strict_mse,
        "lasso_strict_one_se_mse": lasso_strict_one_se_mse,
        "lasso_oracle_alpha": float(alphas[lasso_oracle_index]),
        "lasso_current_alpha": float(current_lasso.alpha_),
        "lasso_one_se_alpha": current_one_se_alpha,
        "lasso_strict_alpha": float(alphas[strict_index]),
        "lasso_strict_one_se_alpha": strict_one_se_alpha,
        "lasso_current_nonzero": current_selection["nonzero"],
        "lasso_current_tp": current_selection["tp"],
        "lasso_current_fp": current_selection["fp"],
        "lasso_current_fn": current_selection["fn"],
        "lasso_one_se_nonzero": one_se_selection["nonzero"],
        "lasso_one_se_tp": one_se_selection["tp"],
        "lasso_one_se_fp": one_se_selection["fp"],
        "lasso_one_se_fn": one_se_selection["fn"],
    }


#===========================================================
#   SUMMARIZING THE EXPERIMENT

def mean_and_se(values: pd.Series) -> tuple[float, float]:
    ### Returns Monte Carlo mean and standard error
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def observed_lasso_one_se(train: pd.DataFrame) -> pd.DataFrame:
    ### Compares alpha.min and alpha.1se on the original 150-row training data
    features = feature_names()
    x = train[features].to_numpy(dtype=float)
    y = train["y"].to_numpy(dtype=float)
    minimum_model, x_mean, x_scale = fit_current_lasso(x, y)
    alpha_one_se, threshold, minimum_se = one_se_alpha(minimum_model.alphas_, minimum_model.mse_path_)
    one_se_model, _, _ = fit_lasso_at_alpha(x, y, alpha_one_se)

    rows = []
    for label, alpha, coef in [
        ("alpha.min", float(minimum_model.alpha_), minimum_model.coef_),
        ("alpha.1se", alpha_one_se, one_se_model.coef_),
    ]:
        metrics = selection_metrics(coef)
        selected_vars = [name for name, value in zip(features, coef) if abs(value) > 1e-8]
        rows.append(
            {
                "rule": label,
                "alpha": alpha,
                "nonzero": metrics["nonzero"],
                "true_positives": metrics["tp"],
                "false_positives": metrics["fp"],
                "false_negatives": metrics["fn"],
                "selected_variables": ", ".join(selected_vars),
                "one_se_threshold": threshold if label == "alpha.1se" else np.nan,
                "se_at_minimum": minimum_se if label == "alpha.1se" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    results: pd.DataFrame,
    beta_true: np.ndarray,
    x_covariance: np.ndarray,
    observed_one_se: pd.DataFrame,
):
    ### Writes an interpretable decomposition of average prediction error
    metrics = {
        "oracle": mean_and_se(results["oracle_mse"]),
        "active_ols": mean_and_se(results["active_ols_mse"]),
        "ridge_oracle": mean_and_se(results["ridge_oracle_mse"]),
        "ridge_single": mean_and_se(results["ridge_single_cv_mse"]),
        "ridge_repeated": mean_and_se(results["ridge_repeated_cv_mse"]),
        "ridge_fixed": mean_and_se(results["ridge_fixed_population_lambda_mse"]),
        "lasso_oracle": mean_and_se(results["lasso_oracle_mse"]),
        "lasso_current": mean_and_se(results["lasso_current_cv_mse"]),
        "lasso_one_se": mean_and_se(results["lasso_one_se_mse"]),
        "lasso_strict": mean_and_se(results["lasso_strict_fold_scaled_cv_mse"]),
        "lasso_strict_one_se": mean_and_se(results["lasso_strict_one_se_mse"]),
    }

    ridge_noise = metrics["oracle"][0]
    true_support_estimation_cost = metrics["active_ols"][0] - ridge_noise
    ridge_model_cost = metrics["ridge_oracle"][0] - metrics["active_ols"][0]
    ridge_tuning_cost = metrics["ridge_single"][0] - metrics["ridge_oracle"][0]
    lasso_model_cost = metrics["lasso_oracle"][0] - metrics["active_ols"][0]
    lasso_tuning_cost = metrics["lasso_current"][0] - metrics["lasso_oracle"][0]
    lasso_one_se_cost = metrics["lasso_one_se"][0] - metrics["lasso_oracle"][0]

    active_beta_text = ", ".join(
        f"{name}={beta_true[feature_names().index(name)]:.3f}" for name in TRUE_ACTIVE
    )
    implied_snr = float(beta_true @ x_covariance @ beta_true / NOISE_VARIANCE)

    lines = [
        "regularization MONTE CARLO ERROR-DECOMPOSITION EXPERIMENT",
        "",
        "Important limitation:",
        "The experiment now uses the exact oracle intercept, active variables, coefficients,",
        "and noise level. The hidden predictor-generation mechanism is still unavailable,",
        "so predictor means and covariance are estimated from the observed training data.",
        "It is therefore exact for the conditional response model but semi-parametric for X.",
        "",
        f"Simulations: {N_SIMULATIONS}; train n={N_TRAIN}; large test n={N_TEST}",
        f"Exact oracle active coefficients: {active_beta_text}",
        f"Reported SNR: {TARGET_SNR:.4f}",
        f"SNR implied by exact betas and empirical X covariance: {implied_snr:.4f}",
        "",
        "Average test MSE (Monte Carlo mean +/- simulation SE):",
    ]
    for label, key in [
        ("True conditional mean (irreducible noise)", "oracle"),
        ("Known-support OLS", "active_ols"),
        ("Ridge, best test-set lambda diagnostic", "ridge_oracle"),
        ("Ridge, submitted single 5-fold CV style", "ridge_single"),
        ("Ridge, repeated 5-fold CV", "ridge_repeated"),
        ("Ridge, fixed reported population lambda", "ridge_fixed"),
        ("Lasso, best test-set alpha diagnostic", "lasso_oracle"),
        ("Lasso, submitted LassoCV style", "lasso_current"),
        ("Lasso, alpha.1se rule", "lasso_one_se"),
        ("Lasso, fold-scaled strict CV", "lasso_strict"),
        ("Lasso, strict fold-scaled alpha.1se", "lasso_strict_one_se"),
    ]:
        mean_value, se_value = metrics[key]
        lines.append(f"  {label:<46} {mean_value:8.4f} +/- {se_value:.4f}")

    lines.extend(
        [
            "",
            "Approximate gap decomposition:",
            f"  Ridge irreducible noise:                 {ridge_noise:.4f}",
            f"  Finite-sample cost with true support:    {true_support_estimation_cost:.4f}",
            f"  Ridge unknown-support/model-family cost: {ridge_model_cost:.4f}",
            f"  Ridge single-CV tuning cost:             {ridge_tuning_cost:.4f}",
            f"  Lasso irreducible noise:                 {ridge_noise:.4f}",
            f"  Finite-sample cost with true support:    {true_support_estimation_cost:.4f}",
            f"  Lasso unknown-support/model-family cost: {lasso_model_cost:.4f}",
            f"  Lasso current-CV tuning/preprocess cost: {lasso_tuning_cost:.4f}",
            f"  Lasso alpha.1se tuning/sparsity cost:    {lasso_one_se_cost:.4f}",
            "",
            "Average variable selection across simulations:",
            f"  alpha.min: TP={results['lasso_current_tp'].mean():.2f}, "
            f"FP={results['lasso_current_fp'].mean():.2f}, "
            f"FN={results['lasso_current_fn'].mean():.2f}, "
            f"nonzero={results['lasso_current_nonzero'].mean():.2f}",
            f"  alpha.1se: TP={results['lasso_one_se_tp'].mean():.2f}, "
            f"FP={results['lasso_one_se_fp'].mean():.2f}, "
            f"FN={results['lasso_one_se_fn'].mean():.2f}, "
            f"nonzero={results['lasso_one_se_nonzero'].mean():.2f}",
            "",
            "Observed training-data comparison:",
            f"  alpha.min={observed_one_se.loc[0, 'alpha']:.6f}: "
            f"TP={int(observed_one_se.loc[0, 'true_positives'])}, "
            f"FP={int(observed_one_se.loc[0, 'false_positives'])}, "
            f"FN={int(observed_one_se.loc[0, 'false_negatives'])}, "
            f"nonzero={int(observed_one_se.loc[0, 'nonzero'])}",
            f"  alpha.1se={observed_one_se.loc[1, 'alpha']:.6f}: "
            f"TP={int(observed_one_se.loc[1, 'true_positives'])}, "
            f"FP={int(observed_one_se.loc[1, 'false_positives'])}, "
            f"FN={int(observed_one_se.loc[1, 'false_negatives'])}, "
            f"nonzero={int(observed_one_se.loc[1, 'nonzero'])}",
            "",
            "Interpretation:",
            "The oracle-tuned model rows are deliberately optimistic and cannot be used in",
            "practice; they isolate the part of excess error that remains even if tuning were",
            "perfect. The difference between the current-CV row and its oracle-tuned row is",
            "the estimated tuning/testing contribution under this calibrated pseudo-DGP.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


#===========================================================
#   RUNNING THE EXPERIMENT

def main():
    ### Runs all Monte Carlo replications and writes CSV and text summaries
    observed_train = pd.read_csv(TRAIN_PATH)
    features = feature_names()
    observed_x = observed_train[features].to_numpy(dtype=float)
    x_mean = observed_x.mean(axis=0)
    x_covariance = positive_definite_covariance(observed_x)
    beta_true = oracle_beta_vector()
    lambdas = pd.read_csv(LAMBDA_GRID_PATH)["lambda"].to_numpy(dtype=float)

    rows = []
    for simulation_index in range(N_SIMULATIONS):
        rows.append(run_replication(simulation_index, x_mean, x_covariance, beta_true, lambdas))
        if (simulation_index + 1) % 5 == 0:
            print(f"completed {simulation_index + 1}/{N_SIMULATIONS}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, float_format="%.10f")
    observed_one_se = observed_lasso_one_se(observed_train)
    observed_one_se.to_csv(ONE_SE_OBSERVED_PATH, index=False, float_format="%.10f")
    write_summary(results, beta_true, x_covariance, observed_one_se)

    print(f"saved: {RESULTS_PATH}")
    print(f"saved: {SUMMARY_PATH}")
    print(f"saved: {ONE_SE_OBSERVED_PATH}")


if __name__ == "__main__":
    main()
