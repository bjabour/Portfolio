import numpy as np
import pandas as pd


DATA_PATH = "resampling_data.csv"
RESULTS_PATH = "resampling_bootstrap.csv"
RANDOM_STATE = 9618
B_BOOTSTRAP = 50000


def portfolio_alpha(asset1_return: np.ndarray, asset2_return: np.ndarray) -> float:
    ### Calculates the minimum-variance allocation to Asset 1
    x = np.asarray(asset1_return, dtype=float)
    y = np.asarray(asset2_return, dtype=float)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    denominator = var_x + var_y - 2 * cov_xy
    return float((var_y - cov_xy) / denominator)


def bootstrap_alphas(asset1_return: np.ndarray, asset2_return: np.ndarray, b: int, random_state: int) -> np.ndarray:
    ### Implements EDF bootstrap by converting uniforms into row indices
    x = np.asarray(asset1_return, dtype=float)
    y = np.asarray(asset2_return, dtype=float)
    n = len(x)
    rng = np.random.default_rng(random_state)

    u = rng.random((b, n))
    row_index = np.floor(n * u).astype(int)

    boot_x = x[row_index]
    boot_y = y[row_index]

    var_x = np.var(boot_x, axis=1, ddof=1)
    var_y = np.var(boot_y, axis=1, ddof=1)
    centered_x = boot_x - boot_x.mean(axis=1, keepdims=True)
    centered_y = boot_y - boot_y.mean(axis=1, keepdims=True)
    cov_xy = np.sum(centered_x * centered_y, axis=1) / (n - 1)

    denominator = var_x + var_y - 2 * cov_xy
    return (var_y - cov_xy) / denominator


def save_results(results: pd.DataFrame, path: str):
    ### Saves results, allowing reruns when an identical CSV is open locally
    try:
        results.to_csv(path, index=False, float_format="%.10f")
    except PermissionError:
        existing = pd.read_csv(path)
        same_columns = list(existing.columns) == list(results.columns)
        same_shape = existing.shape == results.shape
        same_values = np.allclose(existing.to_numpy(dtype=float), results.to_numpy(dtype=float), atol=1e-10)
        if not (same_columns and same_shape and same_values):
            raise
        print(f"{path} is open or locked; existing file already matches computed results.")


def main():
    ### Runs the full assignment solution and writes the required CSV

    #===========================================================
    #   LOADING DATA

    data = pd.read_csv(DATA_PATH)
    x = data["asset1_return"].to_numpy()
    y = data["asset2_return"].to_numpy()

    #===========================================================
    #   ESTIMATING ALPHA FROM THE ORIGINAL DATA

    alpha_hat = portfolio_alpha(x, y)

    #===========================================================
    #   BOOTSTRAPPING A 95% PERCENTILE CONFIDENCE INTERVAL

    boot_alpha = bootstrap_alphas(x, y, B_BOOTSTRAP, RANDOM_STATE)
    ci_lower, ci_upper = np.percentile(boot_alpha, [2.5, 97.5])

    #===========================================================
    #   SAVING SUBMISSION FILE

    results = pd.DataFrame(
        {
            "alpha_hat": [alpha_hat],
            "ci_lower": [ci_lower],
            "ci_upper": [ci_upper],
        }
    )
    save_results(results, RESULTS_PATH)

    #===========================================================
    #   PRINTING RUN SUMMARY

    print(f"rows: {len(data)}")
    print(f"bootstrap replicates: {B_BOOTSTRAP}")
    print(f"alpha_hat: {alpha_hat:.6f}")
    print(f"95% percentile CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
