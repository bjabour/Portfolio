import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import matplotlib
import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
ASSIGNMENT_DIR = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from scratch_random_forest import ScratchRandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDENT_ID = 9618
TRAIN_PATH = ASSIGNMENT_DIR / "trees_train.csv"
TEST_PATH = ASSIGNMENT_DIR / "trees_test.csv"
CART_DIR = EXPERIMENTS_DIR / "CART"

FEATURE_NAMES = [
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GarageArea",
    "BedroomAbvGr",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF",
]

MTRY = 4
SCREEN_TREES = 500
CV_TREES = 500
FINAL_TREES = 2000
SCREEN_DEPTHS = (4, 5, 6, 7, 8, 10, 12, 15)
SCREEN_LEAF_SIZES = (1, 2, 3, 5)
CV_SEEDS = (0, 42, 2026, STUDENT_ID, 19618, 31415)
FINAL_CHECKPOINTS = (50, 100, 200, 300, 500, 800, 1000, 1500, 2000)
NEAR_MINIMUM_FRACTION = 0.0025
CV_WORKERS = 6

LOCKED_TASK1_HASHES = {
    "official_depth_cv.csv": (
        "4F7A282AAF5F0D05388BC39338A607D1BC9E5764B7E9A4C28E5FEA8BB83D314F"
    ),
    "cart_test_predictions.csv": (
        "5B939A1AA42874BBC0E1BC497C0B556CFF6FB9AB2E0DEB3FB6D8D8C5ACE040BF"
    ),
    "cart_task1_results.json": (
        "67AC10A3FCBC3A35CE78ECF6ACCD704769C2F37434FD8201EFA04D84BAE2843D"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_original_data() -> dict[str, object]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if list(train.columns[:12]) != FEATURE_NAMES:
        raise ValueError("Unexpected training feature order")
    if list(test.columns) != FEATURE_NAMES:
        raise ValueError("Unexpected test feature order")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError("Task 2 expects complete input data")
    if any("years_since" in column for column in train.columns):
        raise ValueError("Transformed year columns are prohibited in Task 2")
    return {
        "train": train,
        "test": test,
        "x_train": train[FEATURE_NAMES].to_numpy(dtype=np.float32),
        "x_test": test[FEATURE_NAMES].to_numpy(dtype=np.float32),
        "y_train": train["sale_price_keur"].to_numpy(dtype=np.float64),
    }


def seed_from_parts(*parts: int) -> int:
    sequence = np.random.SeedSequence([int(part) for part in parts])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def make_manual_folds(
    n_rows: int,
    seed: int,
    n_folds: int = 5,
) -> list[np.ndarray]:
    shuffled = np.random.default_rng(seed).permutation(n_rows)
    folds = [
        np.asarray(fold, dtype=int)
        for fold in np.array_split(shuffled, n_folds)
    ]
    combined = np.concatenate(folds)
    if len(np.unique(combined)) != n_rows:
        raise AssertionError("Manual folds do not cover each row exactly once")
    return folds


def rank_vector(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    rank_a = rank_vector(a)
    rank_b = rank_vector(b)
    if np.std(rank_a) == 0.0 or np.std(rank_b) == 0.0:
        return 1.0 if np.array_equal(rank_a, rank_b) else 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def top_feature_indices(values: np.ndarray, count: int = 5) -> np.ndarray:
    return np.argsort(-np.asarray(values), kind="mergesort")[:count]


def pairwise_importance_stability(
    importance_matrix: np.ndarray,
) -> tuple[float, float]:
    correlations = []
    overlaps = []
    top_sets = [
        set(top_feature_indices(row).tolist())
        for row in importance_matrix
    ]
    for first in range(len(importance_matrix)):
        for second in range(first + 1, len(importance_matrix)):
            correlations.append(
                spearman_correlation(
                    importance_matrix[first],
                    importance_matrix[second],
                )
            )
            overlaps.append(
                len(top_sets[first] & top_sets[second]) / 5.0
            )
    return float(np.mean(correlations)), float(np.mean(overlaps))


def importance_columns(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}{feature_name}": float(value)
        for feature_name, value in zip(FEATURE_NAMES, values)
    }


def fit_screen_configuration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, object]:
    started = time.perf_counter()
    forest = ScratchRandomForestRegressor(
        n_estimators=SCREEN_TREES,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=MTRY,
        random_state=STUDENT_ID,
    ).fit(x_train, y_train, FEATURE_NAMES)
    diagnostics = forest.tree_diagnostics_frame()
    row = {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "n_trees": SCREEN_TREES,
        "mtry": MTRY,
        "seed": STUDENT_ID,
        "oob_mse": forest.oob_mse_,
        "oob_rmse": forest.oob_rmse_,
        "oob_coverage": forest.oob_coverage_,
        "min_oob_count": int(forest.oob_counts_.min()),
        "mean_oob_count": float(forest.oob_counts_.mean()),
        "max_oob_count": int(forest.oob_counts_.max()),
        "mean_tree_depth": float(diagnostics["tree_depth"].mean()),
        "mean_node_count": float(diagnostics["node_count"].mean()),
        "mean_leaf_count": float(diagnostics["leaf_count"].mean()),
        "mean_unique_inbag_rows": float(
            diagnostics["unique_inbag_rows"].mean()
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    row.update(importance_columns("mdi_", forest.feature_importances_))
    row.update(importance_columns("raw_sse_share_", forest.raw_sse_share_))
    return row


def run_oob_screen(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    output_path = EXPERIMENT_DIR / "rf_oob_screening.csv"
    if output_path.exists():
        existing = pd.read_csv(output_path)
        expected_keys = {
            (depth, leaf)
            for depth in SCREEN_DEPTHS
            for leaf in SCREEN_LEAF_SIZES
        }
        existing_keys = {
            (int(row.max_depth), int(row.min_samples_leaf))
            for row in existing.itertuples()
        }
        if len(existing) == len(expected_keys) and existing_keys == expected_keys:
            print("Reusing complete 32-setting OOB screen.", flush=True)
            return existing.sort_values(
                ["oob_mse", "max_depth", "min_samples_leaf"],
                kind="mergesort",
            )

    records = []
    total = len(SCREEN_DEPTHS) * len(SCREEN_LEAF_SIZES)
    completed = 0
    for max_depth in SCREEN_DEPTHS:
        for min_samples_leaf in SCREEN_LEAF_SIZES:
            completed += 1
            print(
                "OOB screen "
                f"{completed}/{total}: depth={max_depth}, "
                f"leaf={min_samples_leaf}",
                flush=True,
            )
            records.append(
                fit_screen_configuration(
                    x_train,
                    y_train,
                    max_depth,
                    min_samples_leaf,
                )
            )
            pd.DataFrame(records).to_csv(
                output_path,
                index=False,
            )
    screening = pd.DataFrame(records).sort_values(
        ["oob_mse", "max_depth", "min_samples_leaf"],
        kind="mergesort",
    )
    screening.insert(0, "oob_rank", np.arange(1, len(screening) + 1))
    screening.to_csv(
        output_path,
        index=False,
    )
    return screening


def choose_shortlist(screening: pd.DataFrame) -> pd.DataFrame:
    selected_keys = set()
    for row in screening.head(6).itertuples():
        selected_keys.add((int(row.max_depth), int(row.min_samples_leaf)))
    for leaf_size in SCREEN_LEAF_SIZES:
        best_leaf = screening.loc[
            screening["min_samples_leaf"] == leaf_size
        ].iloc[0]
        selected_keys.add(
            (int(best_leaf["max_depth"]), int(best_leaf["min_samples_leaf"]))
        )

    shortlist = screening.loc[
        screening.apply(
            lambda row: (
                int(row["max_depth"]),
                int(row["min_samples_leaf"]),
            )
            in selected_keys,
            axis=1,
        )
    ].copy()
    shortlist = shortlist.sort_values(
        ["oob_mse", "max_depth", "min_samples_leaf"],
        kind="mergesort",
    )
    shortlist.insert(
        0, "shortlist_order", np.arange(1, len(shortlist) + 1)
    )
    shortlist.to_csv(
        EXPERIMENT_DIR / "rf_cv_shortlist.csv",
        index=False,
    )
    return shortlist


def fit_cv_job(
    job: tuple[
        int,
        int,
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> dict[str, object]:
    (
        max_depth,
        min_samples_leaf,
        cv_seed,
        fold_number,
        forest_seed,
        x_training,
        y_training,
        x_validation,
        y_validation,
    ) = job
    started = time.perf_counter()
    forest = ScratchRandomForestRegressor(
        n_estimators=CV_TREES,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=MTRY,
        random_state=forest_seed,
    ).fit(x_training, y_training, FEATURE_NAMES)
    prediction = forest.predict(x_validation)
    mse = float(np.mean((y_validation - prediction) ** 2))
    record = {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "cv_seed": cv_seed,
        "fold": fold_number,
        "forest_seed": forest_seed,
        "training_rows": len(y_training),
        "validation_rows": len(y_validation),
        "validation_mse": mse,
        "validation_rmse": float(np.sqrt(mse)),
        "training_fold_oob_mse": forest.oob_mse_,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    record.update(importance_columns("mdi_", forest.feature_importances_))
    return record


def run_repeated_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    output_path = EXPERIMENT_DIR / "rf_repeated_cv_fold_results.csv"
    if output_path.exists():
        existing = pd.read_csv(output_path)
        records = existing.to_dict("records")
    else:
        records = []
    completed_keys = {
        (
            int(row["max_depth"]),
            int(row["min_samples_leaf"]),
            int(row["cv_seed"]),
            int(row["fold"]),
        )
        for row in records
    }
    configurations = [
        (int(row.max_depth), int(row.min_samples_leaf))
        for row in shortlist.itertuples()
    ]
    total_fits = len(configurations) * len(CV_SEEDS) * 5
    completed = len(completed_keys)
    all_indices = np.arange(len(y_train), dtype=int)
    jobs = []

    for max_depth, min_samples_leaf in configurations:
        for cv_seed in CV_SEEDS:
            folds = make_manual_folds(len(y_train), cv_seed)
            for fold_number, validation_indices in enumerate(
                folds, start=1
            ):
                key = (
                    max_depth,
                    min_samples_leaf,
                    cv_seed,
                    fold_number,
                )
                if key in completed_keys:
                    continue
                validation_mask = np.zeros(len(y_train), dtype=bool)
                validation_mask[validation_indices] = True
                training_indices = all_indices[~validation_mask]
                forest_seed = seed_from_parts(
                    STUDENT_ID, cv_seed, fold_number
                )
                jobs.append(
                    (
                        max_depth,
                        min_samples_leaf,
                        cv_seed,
                        fold_number,
                        forest_seed,
                        x_train[training_indices],
                        y_train[training_indices],
                        x_train[validation_indices],
                        y_train[validation_indices],
                    )
                )

    if jobs:
        print(
            f"Resuming repeated CV at {completed}/{total_fits}; "
            f"running {len(jobs)} remaining fits with {CV_WORKERS} workers.",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=CV_WORKERS) as executor:
            for record in executor.map(fit_cv_job, jobs, chunksize=1):
                records.append(record)
                completed += 1
                print(
                    "Repeated CV "
                    f"{completed}/{total_fits}: "
                    f"depth={record['max_depth']}, "
                    f"leaf={record['min_samples_leaf']}, "
                    f"seed={record['cv_seed']}, "
                    f"fold={record['fold']}",
                    flush=True,
                )
                pd.DataFrame(records).to_csv(output_path, index=False)
    else:
        print("Reusing complete repeated-CV results.", flush=True)
    result = pd.DataFrame(records)
    return result.sort_values(
        [
            "max_depth",
            "min_samples_leaf",
            "cv_seed",
            "fold",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_repeated_cv(
    fold_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_results = fold_results.sort_values(
        [
            "max_depth",
            "min_samples_leaf",
            "cv_seed",
            "fold",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_rows = []
    frequency_rows = []
    mdi_columns = [f"mdi_{name}" for name in FEATURE_NAMES]
    grouped = fold_results.groupby(
        ["max_depth", "min_samples_leaf"], sort=True
    )
    for (max_depth, min_samples_leaf), group in grouped:
        importance_matrix = group[mdi_columns].to_numpy(dtype=float)
        rank_stability, top5_stability = pairwise_importance_stability(
            importance_matrix
        )
        mean_importance = importance_matrix.mean(axis=0)
        sd_importance = importance_matrix.std(axis=0, ddof=1)
        top_counts = np.zeros(len(FEATURE_NAMES), dtype=int)
        for importance in importance_matrix:
            top_counts[top_feature_indices(importance)] += 1

        row = {
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "n_fits": len(group),
            "mean_validation_mse": float(group["validation_mse"].mean()),
            "sd_validation_mse": float(
                group["validation_mse"].std(ddof=1)
            ),
            "validation_rmse_from_mean_mse": float(
                np.sqrt(group["validation_mse"].mean())
            ),
            "min_validation_mse": float(group["validation_mse"].min()),
            "max_validation_mse": float(group["validation_mse"].max()),
            "mean_training_fold_oob_mse": float(
                group["training_fold_oob_mse"].mean()
            ),
            "mdi_pairwise_spearman": rank_stability,
            "top5_pairwise_overlap": top5_stability,
        }
        row.update(importance_columns("mean_mdi_", mean_importance))
        row.update(importance_columns("sd_mdi_", sd_importance))
        summary_rows.append(row)

        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            frequency_rows.append(
                {
                    "max_depth": int(max_depth),
                    "min_samples_leaf": int(min_samples_leaf),
                    "feature": feature_name,
                    "top5_count": int(top_counts[feature_index]),
                    "top5_frequency": float(
                        top_counts[feature_index] / len(group)
                    ),
                    "mean_mdi": float(mean_importance[feature_index]),
                    "sd_mdi": float(sd_importance[feature_index]),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "mean_validation_mse",
            "max_depth",
            "min_samples_leaf",
        ],
        kind="mergesort",
    )
    minimum = float(summary["mean_validation_mse"].min())
    summary["within_0_25_percent"] = (
        summary["mean_validation_mse"]
        <= minimum * (1.0 + NEAR_MINIMUM_FRACTION)
    )
    eligible = summary.loc[summary["within_0_25_percent"]].copy()
    eligible = eligible.sort_values(
        [
            "mdi_pairwise_spearman",
            "sd_validation_mse",
            "max_depth",
            "min_samples_leaf",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    selected_key = (
        int(eligible.iloc[0]["max_depth"]),
        int(eligible.iloc[0]["min_samples_leaf"]),
    )
    summary["selected"] = (
        (summary["max_depth"] == selected_key[0])
        & (summary["min_samples_leaf"] == selected_key[1])
    )
    summary.insert(
        0,
        "mean_mse_rank",
        summary["mean_validation_mse"].rank(
            method="first"
        ).astype(int),
    )
    summary.to_csv(
        EXPERIMENT_DIR / "rf_repeated_cv_summary.csv",
        index=False,
    )

    frequencies = pd.DataFrame(frequency_rows)
    frequencies.to_csv(
        EXPERIMENT_DIR / "rf_importance_stability.csv",
        index=False,
    )
    return summary, frequencies


def run_unit_tests() -> pd.DataFrame:
    records = []

    def record(name: str, passed: bool, detail: str) -> None:
        records.append(
            {"test": name, "passed": bool(passed), "detail": detail}
        )
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    rng = np.random.default_rng(2026)
    x = rng.normal(size=(80, 6)).astype(np.float32)
    y = 7.0 * x[:, 0] + rng.normal(scale=0.2, size=80)
    feature_names = [f"x{index}" for index in range(x.shape[1])]
    forest = ScratchRandomForestRegressor(
        n_estimators=60,
        max_depth=6,
        min_samples_leaf=2,
        max_features=4,
        random_state=STUDENT_ID,
    ).fit(
        x,
        y,
        feature_names,
        checkpoints=(10, 60),
        evaluation_x=x[:5],
    )

    record(
        "bootstrap_samples_have_original_size_and_duplicates",
        all(len(indices) == len(x) for indices in forest.bootstrap_indices_)
        and all(
            len(np.unique(indices)) < len(x)
            for indices in forest.bootstrap_indices_
        ),
        "All 60 bootstrap samples contain 80 draws and duplicates.",
    )

    complement_match = True
    for bootstrap_indices, oob_indices in zip(
        forest.bootstrap_indices_, forest.oob_indices_
    ):
        expected = np.flatnonzero(
            np.bincount(bootstrap_indices, minlength=len(x)) == 0
        )
        complement_match = complement_match and np.array_equal(
            oob_indices, expected
        )
    record(
        "oob_is_exact_inbag_complement",
        complement_match,
        "All OOB index arrays equal the complement of unique in-bag rows.",
    )

    subset_sizes = np.concatenate(
        [
            tree.feature_subset_sizes()
            for tree in forest.estimators_
            if len(tree.feature_subset_sizes())
        ]
    )
    record(
        "every_eligible_node_receives_mtry_four",
        len(subset_sizes) > 0 and np.all(subset_sizes == 4),
        f"candidate_nodes={len(subset_sizes)}",
    )

    direct_prediction = np.mean(
        np.vstack(
            [tree.predict(x[:5]) for tree in forest.estimators_]
        ),
        axis=0,
    )
    record(
        "forest_prediction_is_tree_mean",
        np.array_equal(forest.predict(x[:5]), direct_prediction),
        "Direct and forest prediction arrays are exactly equal.",
    )

    row_index = int(np.argmax(forest.oob_counts_))
    manual_oob = np.mean(
        [
            tree.predict(x[[row_index]])[0]
            for tree, oob_indices in zip(
                forest.estimators_, forest.oob_indices_
            )
            if row_index in oob_indices
        ]
    )
    record(
        "oob_prediction_uses_only_excluding_trees",
        abs(manual_oob - forest.oob_prediction_[row_index]) <= 1e-12,
        f"row={row_index}, trees={forest.oob_counts_[row_index]}",
    )

    constant_forest = ScratchRandomForestRegressor(
        n_estimators=20,
        max_depth=5,
        min_samples_leaf=1,
        max_features=4,
        random_state=3,
    ).fit(x, np.repeat(4.2, len(x)), feature_names)
    record(
        "constant_target_returns_constant_and_zero_mdi",
        np.allclose(constant_forest.predict(x[:5]), 4.2)
        and np.allclose(constant_forest.feature_importances_, 0.0),
        f"mdi_sum={constant_forest.feature_importances_.sum()}",
    )

    record(
        "synthetic_signal_is_dominant",
        int(np.argmax(forest.feature_importances_)) == 0
        and forest.feature_importances_[0] > 0.70,
        f"signal_mdi={forest.feature_importances_[0]:.6f}",
    )

    record(
        "mdi_is_finite_nonnegative_and_normalized",
        np.isfinite(forest.feature_importances_).all()
        and np.all(forest.feature_importances_ >= 0.0)
        and abs(float(forest.feature_importances_.sum()) - 1.0) <= 1e-12,
        f"mdi_sum={forest.feature_importances_.sum():.16f}",
    )

    repeated = ScratchRandomForestRegressor(
        n_estimators=60,
        max_depth=6,
        min_samples_leaf=2,
        max_features=4,
        random_state=STUDENT_ID,
    ).fit(x, y, feature_names)
    record(
        "fixed_seed_is_byte_deterministic",
        forest.predict(x).tobytes() == repeated.predict(x).tobytes()
        and forest.feature_importances_.tobytes()
        == repeated.feature_importances_.tobytes(),
        "Predictions and MDI are byte-identical.",
    )

    checkpoint_direct = np.mean(
        np.vstack(
            [tree.predict(x[:5]) for tree in forest.estimators_[:10]]
        ),
        axis=0,
    )
    record(
        "checkpoint_prediction_is_cumulative_tree_mean",
        np.array_equal(
            forest.checkpoints_[0].evaluation_prediction,
            checkpoint_direct,
        ),
        "Ten-tree checkpoint exactly equals direct ten-tree mean.",
    )
    return pd.DataFrame(records)


def create_convergence_artifacts(
    final_forest: ScratchRandomForestRegressor,
    final_test_prediction: np.ndarray,
) -> pd.DataFrame:
    rows = []
    final_top5 = set(
        top_feature_indices(final_forest.feature_importances_).tolist()
    )
    for checkpoint in final_forest.checkpoints_:
        prediction_difference = (
            checkpoint.evaluation_prediction - final_test_prediction
        )
        checkpoint_top5 = set(top_feature_indices(checkpoint.mdi).tolist())
        row = {
            "n_trees": checkpoint.n_trees,
            "oob_mse": checkpoint.oob_mse,
            "oob_rmse": checkpoint.oob_rmse,
            "oob_coverage": checkpoint.oob_coverage,
            "min_oob_count": checkpoint.min_oob_count,
            "mean_oob_count": checkpoint.mean_oob_count,
            "max_oob_count": checkpoint.max_oob_count,
            "test_prediction_rmse_to_2000": float(
                np.sqrt(np.mean(prediction_difference**2))
            ),
            "test_prediction_max_abs_diff_to_2000": float(
                np.max(np.abs(prediction_difference))
            ),
            "mdi_spearman_to_2000": spearman_correlation(
                checkpoint.mdi,
                final_forest.feature_importances_,
            ),
            "mdi_top5_overlap_to_2000": (
                len(checkpoint_top5 & final_top5) / 5.0
            ),
        }
        row.update(importance_columns("mdi_", checkpoint.mdi))
        row.update(
            importance_columns(
                "raw_sse_share_", checkpoint.raw_sse_share
            )
        )
        rows.append(row)
    convergence = pd.DataFrame(rows)
    convergence.to_csv(
        EXPERIMENT_DIR / "rf_final_convergence.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(
        convergence["n_trees"],
        convergence["oob_mse"],
        marker="o",
        color="#1d4ed8",
    )
    axes[0].axvline(200, color="#64748b", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Number of trees",
        ylabel="OOB MSE",
        title="OOB Error Convergence",
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        convergence["n_trees"],
        convergence["test_prediction_rmse_to_2000"],
        marker="o",
        color="#b91c1c",
    )
    axes[1].set(
        xlabel="Number of trees",
        ylabel="RMSE versus 2,000-tree predictions",
        title="Prediction Stabilization",
    )
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        EXPERIMENT_DIR / "rf_final_convergence.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
    return convergence


def create_importance_plot(importance: pd.DataFrame) -> None:
    ordered = importance.sort_values("mdi", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(ordered["feature"], ordered["mdi"], color="#2563eb")
    ax.set(
        xlabel="Normalized Mean Decrease in Impurity",
        ylabel="",
        title="Scratch Random-Forest Regression Importance",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        EXPERIMENT_DIR / "rf_final_mdi_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_findings(
    screening: pd.DataFrame,
    shortlist: pd.DataFrame,
    cv_summary: pd.DataFrame,
    selected_depth: int,
    selected_leaf: int,
    final_forest: ScratchRandomForestRegressor,
    convergence: pd.DataFrame,
    importance: pd.DataFrame,
    source_integrity: pd.DataFrame,
    unit_tests: pd.DataFrame,
) -> None:
    selected = cv_summary.loc[cv_summary["selected"]].iloc[0]
    best_screen = screening.iloc[0]
    cv_rows = "\n".join(
        f"| {int(row.max_depth)} | {int(row.min_samples_leaf)} | "
        f"{row.mean_validation_mse:.3f} | "
        f"{row.validation_rmse_from_mean_mse:.3f} | "
        f"{row.mdi_pairwise_spearman:.3f} | "
        f"{row.top5_pairwise_overlap:.3f} | "
        f"{'yes' if row.selected else ''} |"
        for row in cv_summary.itertuples()
    )
    importance_rows = "\n".join(
        f"| {int(row.mdi_rank)} | `{row.feature}` | "
        f"{row.mdi:.6f} | {row.raw_sse_share:.6f} |"
        for row in importance.sort_values("mdi_rank").itertuples()
    )
    findings = f"""# Random-Forest Task 2 Findings

## Scope

This experiment implements only Task 2. It does not create the final trees
submission JSON, slides, slide source, or models for Tasks 3 through 5.

The forest reads only the original supplied train and test CSV files. It uses
the original `YearBuilt` and `YearRemodAdd` columns and retains every original
`LotArea` value, including the training maximum of
{source_integrity.loc[source_integrity.file == "train", "lot_area_max"].iloc[0]:.0f}.

## Scratch Forest Method

- Each tree receives 300 row draws sampled with replacement.
- Duplicate bootstrap observations remain duplicate training cases.
- Out-of-bag rows are the exact complement of unique in-bag rows.
- Exactly four predictors are sampled independently at every eligible node.
- The best valid split among those four uses the Task 1 cumulative-SSE CART
  search.
- Forest predictions are arithmetic means of tree predictions.
- The final MDI follows equal-tree averaging: normalize each non-stump tree's
  SSE reductions, average those vectors, then normalize the forest vector.

The year transformations were intentionally abandoned for Task 2. Tree splits
depend on ordering, so the original calendar years contain the same information
without introducing reversed branch semantics or held-out midpoint-boundary
effects.

## OOB Screening

The 32-setting screen used {SCREEN_TREES} trees per setting, fixed `mtry=4`,
and seed {STUDENT_ID}. The leading OOB setting was depth
{int(best_screen.max_depth)}, leaf size
{int(best_screen.min_samples_leaf)}, with OOB MSE
{best_screen.oob_mse:.3f}. OOB screening was used only to create the
cross-validation shortlist of {len(shortlist)} settings.

## Repeated Five-Fold Selection

Each shortlisted setting was evaluated in 30 untouched validation folds:
five folds for each predeclared seed `{", ".join(str(seed) for seed in CV_SEEDS)}`.
Every fold forest contained {CV_TREES} trees.

| Depth | Leaf size | Mean validation MSE | RMSE | MDI rank stability | Top-5 overlap | Selected |
|---:|---:|---:|---:|---:|---:|:---:|
{cv_rows}

The selected setting is **max depth {selected_depth}, minimum leaf size
{selected_leaf}**. Its mean validation MSE is
**{selected.mean_validation_mse:.3f}**, corresponding to RMSE
**{selected.validation_rmse_from_mean_mse:.3f}**.

Candidates within 0.25% of the minimum were eligible for the stability
tie-break. The order was MDI pairwise Spearman stability, fold-MSE standard
deviation, shallower depth, then smaller leaf size.

## Final Forest

The final experimental forest contains **{FINAL_TREES} trees**, uses
`mtry=4`, max depth {selected_depth}, minimum leaf size {selected_leaf}, and
seed {STUDENT_ID}.

- Final OOB MSE: {final_forest.oob_mse_:.3f}
- Final OOB RMSE: {final_forest.oob_rmse_:.3f}
- OOB coverage: {final_forest.oob_coverage_:.3%}
- Minimum OOB predictions per row: {int(final_forest.oob_counts_.min())}
- Maximum OOB predictions per row: {int(final_forest.oob_counts_.max())}
- OOB MSE at 200 trees: {convergence.loc[convergence.n_trees == 200, "oob_mse"].iloc[0]:.3f}
- Test-prediction RMSE between 200 and 2,000 trees:
  {convergence.loc[convergence.n_trees == 200, "test_prediction_rmse_to_2000"].iloc[0]:.3f}

## Mean Decrease in Impurity

| Rank | Feature | Equal-tree MDI | Global raw-SSE share |
|---:|---|---:|---:|
{importance_rows}

The equal-tree MDI vector is the Task 2 submission candidate because it matches
the current forest convention used by sklearn. The global raw-SSE share is
retained only as a diagnostic. MDI is predictive importance, not causal or
generative importance, and correlated housing predictors can divide impurity
credit.

## Verification

- Focused tests passed: {int(unit_tests["passed"].sum())}/{len(unit_tests)}
- Train source hash unchanged:
  {source_integrity.loc[source_integrity.file == "train", "hash_unchanged"].iloc[0]}
- Test source hash unchanged:
  {source_integrity.loc[source_integrity.file == "test", "hash_unchanged"].iloc[0]}
- Task 1 locked artifacts unchanged:
  {source_integrity.loc[source_integrity.file == "task1_locked_outputs", "hash_unchanged"].iloc[0]}
- Final prediction rows: 200
- Final importance sum: {final_forest.feature_importances_.sum():.16f}

The scratch CART and random-forest modules contain no prohibited library tree
or ensemble imports.

## 2026 Research Insights

1. Breiman's random-forest construction remains the controlling algorithm:
   bootstrap rows, independently randomized per-node feature subsets,
   unpruned or regularized CART trees, OOB monitoring, and prediction averaging.
2. Current sklearn source confirms bootstrap multiplicities, OOB aggregation,
   independent estimator seeds, and equal-tree averaging of normalized
   impurity vectors.
3. Current R `randomForest` documentation defines regression node impurity
   through residual sum of squares.
4. The June 2026 plateau-search work notes that tree-count optimization often
   runs to the search boundary. Therefore, the experiment fixes 2,000 trees
   and uses the OOB/prediction trajectory to demonstrate stabilization instead
   of claiming a favorable smaller count.
5. 2026 variable-importance research distinguishes model importance from
   importance in the underlying data-generating process. The assignment
   explicitly grades MDI, so alternative sensitivity measures are contextual
   diagnostics rather than substitutes.

Sources:

- https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
- https://search.r-project.org/CRAN/refmans/randomForest/html/importance.html
- https://arxiv.org/abs/2606.03549
- https://doi.org/10.1007/s10260-026-00839-y

No external implementation code was copied.

## Files

- `rf_task2_experiment.py`
- `rf_task2_results.json`
- `rf_oob_screening.csv`
- `rf_cv_shortlist.csv`
- `rf_repeated_cv_fold_results.csv`
- `rf_repeated_cv_summary.csv`
- `rf_importance_stability.csv`
- `rf_final_test_predictions.csv`
- `rf_final_oob_predictions.csv`
- `rf_final_feature_importance.csv`
- `rf_final_tree_diagnostics.csv`
- `rf_final_convergence.csv`
- `rf_final_convergence.png`
- `rf_final_mdi_importance.png`
- `rf_unit_tests.csv`
- `rf_source_data_integrity.csv`
"""
    (EXPERIMENT_DIR / "RF_FINDINGS.md").write_text(
        findings, encoding="utf-8"
    )


def main() -> None:
    train_hash_before = sha256(TRAIN_PATH)
    test_hash_before = sha256(TEST_PATH)
    task1_hashes_before = {
        name: sha256(CART_DIR / name)
        for name in LOCKED_TASK1_HASHES
    }
    for name, expected_hash in LOCKED_TASK1_HASHES.items():
        if task1_hashes_before[name] != expected_hash:
            raise AssertionError(
                f"Task 1 locked artifact changed before Task 2: {name}"
            )

    data = load_original_data()
    train = data["train"]
    test = data["test"]
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]

    unit_tests = run_unit_tests()
    unit_tests.to_csv(
        EXPERIMENT_DIR / "rf_unit_tests.csv", index=False
    )

    screening = run_oob_screen(x_train, y_train)
    shortlist = choose_shortlist(screening)
    fold_results = run_repeated_cv(x_train, y_train, shortlist)
    fold_results.to_csv(
        EXPERIMENT_DIR / "rf_repeated_cv_fold_results.csv",
        index=False,
    )
    cv_summary, importance_stability = summarize_repeated_cv(
        fold_results
    )
    selected = cv_summary.loc[cv_summary["selected"]].iloc[0]
    selected_depth = int(selected["max_depth"])
    selected_leaf = int(selected["min_samples_leaf"])

    print(
        "Fitting final forest: "
        f"trees={FINAL_TREES}, depth={selected_depth}, "
        f"leaf={selected_leaf}",
        flush=True,
    )
    final_forest = ScratchRandomForestRegressor(
        n_estimators=FINAL_TREES,
        max_depth=selected_depth,
        min_samples_leaf=selected_leaf,
        max_features=MTRY,
        random_state=STUDENT_ID,
    ).fit(
        x_train,
        y_train,
        FEATURE_NAMES,
        checkpoints=FINAL_CHECKPOINTS,
        evaluation_x=x_test,
    )
    final_test_prediction = final_forest.predict(x_test)
    if (
        len(final_test_prediction) != 200
        or not np.isfinite(final_test_prediction).all()
    ):
        raise AssertionError("Final test predictions are invalid")

    pd.DataFrame(
        {
            "test_row_index": np.arange(len(test)),
            "rf_reg_prediction": final_test_prediction,
        }
    ).to_csv(
        EXPERIMENT_DIR / "rf_final_test_predictions.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "source_row_index": np.arange(len(train)),
            "actual_sale_price_keur": y_train,
            "oob_prediction": final_forest.oob_prediction_,
            "oob_count": final_forest.oob_counts_,
            "oob_residual": y_train - final_forest.oob_prediction_,
        }
    ).to_csv(
        EXPERIMENT_DIR / "rf_final_oob_predictions.csv",
        index=False,
    )

    selected_frequency = importance_stability.loc[
        (importance_stability["max_depth"] == selected_depth)
        & (
            importance_stability["min_samples_leaf"]
            == selected_leaf
        )
    ].set_index("feature")
    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "mdi": final_forest.feature_importances_,
            "raw_sse": final_forest.raw_sse_importances_,
            "raw_sse_share": final_forest.raw_sse_share_,
        }
    )
    importance["mdi_rank"] = (
        importance["mdi"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    importance["cv_top5_frequency"] = importance["feature"].map(
        selected_frequency["top5_frequency"]
    )
    importance["cv_mean_mdi"] = importance["feature"].map(
        selected_frequency["mean_mdi"]
    )
    importance["cv_sd_mdi"] = importance["feature"].map(
        selected_frequency["sd_mdi"]
    )
    importance.to_csv(
        EXPERIMENT_DIR / "rf_final_feature_importance.csv",
        index=False,
    )
    create_importance_plot(importance)

    tree_diagnostics = final_forest.tree_diagnostics_frame()
    tree_diagnostics.to_csv(
        EXPERIMENT_DIR / "rf_final_tree_diagnostics.csv",
        index=False,
    )
    if not (
        (tree_diagnostics["bootstrap_size"] == len(train)).all()
        and (tree_diagnostics["duplicate_draws"] > 0).all()
        and (
            tree_diagnostics.loc[
                tree_diagnostics["candidate_nodes"] > 0,
                "min_candidate_features",
            ]
            == MTRY
        ).all()
        and (
            tree_diagnostics.loc[
                tree_diagnostics["candidate_nodes"] > 0,
                "max_candidate_features",
            ]
            == MTRY
        ).all()
    ):
        raise AssertionError("Final forest bootstrap or mtry diagnostics failed")

    convergence = create_convergence_artifacts(
        final_forest, final_test_prediction
    )

    importance_dictionary = {
        feature_name: float(value)
        for feature_name, value in zip(
            FEATURE_NAMES, final_forest.feature_importances_
        )
    }
    if (
        set(importance_dictionary) != set(FEATURE_NAMES)
        or not np.isfinite(list(importance_dictionary.values())).all()
        or any(value < 0.0 for value in importance_dictionary.values())
        or abs(sum(importance_dictionary.values()) - 1.0) > 1e-12
    ):
        raise AssertionError("Final MDI dictionary is invalid")

    task2_results = {
        "rf_reg_pred_test": final_test_prediction.astype(float).tolist(),
        "rf_reg_var_importance": importance_dictionary,
        "rf_reg_n_trees": FINAL_TREES,
        "rf_reg_mtry": MTRY,
        "rf_reg_max_depth": selected_depth,
        "_experiment_only": {
            "min_samples_leaf": selected_leaf,
            "selection_method": (
                "Repeated 5-fold CV over six fixed seeds after OOB screening"
            ),
            "final_seed": STUDENT_ID,
            "source_feature_names": FEATURE_NAMES,
            "source_train_sha256": train_hash_before,
            "source_test_sha256": test_hash_before,
            "final_oob_mse": final_forest.oob_mse_,
            "final_oob_rmse": final_forest.oob_rmse_,
        },
    }
    with open(
        EXPERIMENT_DIR / "rf_task2_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(task2_results, file, indent=2)

    train_hash_after = sha256(TRAIN_PATH)
    test_hash_after = sha256(TEST_PATH)
    task1_hashes_after = {
        name: sha256(CART_DIR / name)
        for name in LOCKED_TASK1_HASHES
    }
    task1_unchanged = (
        task1_hashes_before == task1_hashes_after == LOCKED_TASK1_HASHES
    )
    source_integrity = pd.DataFrame(
        [
            {
                "file": "train",
                "path": str(TRAIN_PATH),
                "hash_before": train_hash_before,
                "hash_after": train_hash_after,
                "hash_unchanged": train_hash_before == train_hash_after,
                "rows": len(train),
                "lot_area_max": float(train["LotArea"].max()),
            },
            {
                "file": "test",
                "path": str(TEST_PATH),
                "hash_before": test_hash_before,
                "hash_after": test_hash_after,
                "hash_unchanged": test_hash_before == test_hash_after,
                "rows": len(test),
                "lot_area_max": float(test["LotArea"].max()),
            },
            {
                "file": "task1_locked_outputs",
                "path": str(CART_DIR),
                "hash_before": json.dumps(
                    task1_hashes_before, sort_keys=True
                ),
                "hash_after": json.dumps(
                    task1_hashes_after, sort_keys=True
                ),
                "hash_unchanged": task1_unchanged,
                "rows": len(task1_hashes_before),
                "lot_area_max": np.nan,
            },
        ]
    )
    if not source_integrity["hash_unchanged"].all():
        raise AssertionError("Source or Task 1 artifacts changed")
    source_integrity.to_csv(
        EXPERIMENT_DIR / "rf_source_data_integrity.csv",
        index=False,
    )

    write_findings(
        screening=screening,
        shortlist=shortlist,
        cv_summary=cv_summary,
        selected_depth=selected_depth,
        selected_leaf=selected_leaf,
        final_forest=final_forest,
        convergence=convergence,
        importance=importance,
        source_integrity=source_integrity,
        unit_tests=unit_tests,
    )

    print(
        f"Selected depth={selected_depth}, leaf={selected_leaf}; "
        f"CV MSE={selected.mean_validation_mse:.3f}",
        flush=True,
    )
    print(
        f"Final 2,000-tree OOB MSE={final_forest.oob_mse_:.3f}",
        flush=True,
    )
    print(f"Wrote Task 2 artifacts to: {EXPERIMENT_DIR}", flush=True)


if __name__ == "__main__":
    main()
