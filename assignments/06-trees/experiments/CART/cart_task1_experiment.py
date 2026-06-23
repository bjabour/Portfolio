import hashlib
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from scratch_cart import (
    ScratchCARTRegressor,
    find_best_split_bruteforce,
    find_best_split_vectorized,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths and experiment settings

EXPERIMENT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = EXPERIMENT_DIR.parents[1]
TRAIN_PATH = ASSIGNMENT_DIR / "trees_train.csv"
TEST_PATH = ASSIGNMENT_DIR / "trees_test.csv"

STUDENT_ID = 9618
REFERENCE_YEAR = 2011
N_FOLDS = 5
MIN_SAMPLES_LEAF = 5
OFFICIAL_DEPTHS = tuple(range(2, 11))
OFFICIAL_CV_SEED = STUDENT_ID
STABILITY_SEEDS = (0, 42, 2026, STUDENT_ID, 19618, 31415)
SPLIT_TOLERANCE = 1e-12

ORIGINAL_FEATURES = [
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
TRANSFORMED_FEATURES = [
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "years_since_built_2011",
    "years_since_remodel_2011",
    "TotalBsmtSF",
    "GarageArea",
    "BedroomAbvGr",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF",
]
OUTPUT_FEATURE_NAMES = {
    "years_since_built_2011": "YearBuilt",
    "years_since_remodel_2011": "YearRemodAdd",
}


# ============================================================
# Data preparation

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_year_columns(data: pd.DataFrame) -> pd.DataFrame:
    transformed = data.copy()
    transformed["YearBuilt"] = REFERENCE_YEAR - transformed["YearBuilt"]
    transformed["YearRemodAdd"] = (
        REFERENCE_YEAR - transformed["YearRemodAdd"]
    )
    return transformed.rename(
        columns={
            "YearBuilt": "years_since_built_2011",
            "YearRemodAdd": "years_since_remodel_2011",
        }
    )


def load_experiment_data() -> dict[str, object]:
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    train_transformed = transform_year_columns(train_raw)
    test_transformed = transform_year_columns(test_raw)

    if train_raw.isna().any().any() or test_raw.isna().any().any():
        raise ValueError("Task 1 expects complete input data")
    if list(train_raw.columns[:12]) != ORIGINAL_FEATURES:
        raise ValueError("Unexpected training feature order")
    if list(test_raw.columns) != ORIGINAL_FEATURES:
        raise ValueError("Unexpected test feature order")
    if list(train_transformed[TRANSFORMED_FEATURES].columns) != TRANSFORMED_FEATURES:
        raise ValueError("Unexpected transformed feature order")

    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_transformed": train_transformed,
        "test_transformed": test_transformed,
        "x_train_original": train_raw[ORIGINAL_FEATURES].to_numpy(
            dtype=np.float32
        ),
        "x_test_original": test_raw[ORIGINAL_FEATURES].to_numpy(
            dtype=np.float32
        ),
        "x_train_transformed": train_transformed[
            TRANSFORMED_FEATURES
        ].to_numpy(dtype=np.float32),
        "x_test_transformed": test_transformed[
            TRANSFORMED_FEATURES
        ].to_numpy(dtype=np.float32),
        "y_train": train_raw["sale_price_keur"].to_numpy(dtype=np.float64),
    }


# ============================================================
# Cross-validation and comparison helpers

def make_manual_folds(
    n_rows: int,
    n_folds: int = N_FOLDS,
    seed: int = OFFICIAL_CV_SEED,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)
    folds = [
        np.asarray(fold, dtype=int)
        for fold in np.array_split(shuffled, n_folds)
    ]
    combined = np.concatenate(folds)
    if len(combined) != n_rows or len(np.unique(combined)) != n_rows:
        raise AssertionError("CV folds do not cover each row exactly once")
    return folds


def cross_validate_depths(
    x: np.ndarray,
    y: np.ndarray,
    depths: Iterable[int],
    seed: int,
    return_oof_for_depth: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None, pd.DataFrame]:
    folds = make_manual_folds(len(y), seed=seed)
    all_indices = np.arange(len(y), dtype=int)
    summary_rows = []
    assignment_rows = []
    requested_oof = (
        np.full(len(y), np.nan, dtype=np.float64)
        if return_oof_for_depth is not None
        else None
    )

    for fold_number, validation_indices in enumerate(folds, start=1):
        for row_index in validation_indices:
            assignment_rows.append(
                {
                    "source_row_index": int(row_index),
                    "fold": fold_number,
                    "seed": seed,
                }
            )

    for depth in depths:
        fold_mse = []
        oof = np.full(len(y), np.nan, dtype=np.float64)

        for validation_indices in folds:
            validation_mask = np.zeros(len(y), dtype=bool)
            validation_mask[validation_indices] = True
            training_indices = all_indices[~validation_mask]

            tree = ScratchCARTRegressor(
                max_depth=int(depth),
                min_samples_leaf=MIN_SAMPLES_LEAF,
            ).fit(
                x[training_indices],
                y[training_indices],
                TRANSFORMED_FEATURES,
            )
            predictions = tree.predict(x[validation_indices])
            oof[validation_indices] = predictions
            fold_mse.append(
                float(np.mean((y[validation_indices] - predictions) ** 2))
            )

        if not np.isfinite(oof).all():
            raise AssertionError("OOF predictions are incomplete")
        mean_mse = float(np.mean(fold_mse))
        summary_rows.append(
            {
                "seed": seed,
                "max_depth": int(depth),
                **{
                    f"fold_{j}_mse": value
                    for j, value in enumerate(fold_mse, start=1)
                },
                "mean_cv_mse": mean_mse,
                "sd_cv_mse": float(np.std(fold_mse, ddof=1)),
                "oof_rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
            }
        )
        if return_oof_for_depth == depth:
            requested_oof[:] = oof

    return (
        pd.DataFrame(summary_rows),
        requested_oof,
        pd.DataFrame(assignment_rows).sort_values("source_row_index"),
    )


def choose_first_minimum_depth(cv_summary: pd.DataFrame) -> int:
    ordered = cv_summary.sort_values("max_depth").reset_index(drop=True)
    minimum = ordered["mean_cv_mse"].min()
    selected = ordered.loc[
        np.isclose(ordered["mean_cv_mse"], minimum, rtol=0.0, atol=1e-12),
        "max_depth",
    ]
    return int(selected.iloc[0])


def leaf_signatures(
    leaf_ids: np.ndarray,
    predictions: np.ndarray,
) -> dict[tuple[int, ...], float]:
    signatures: dict[tuple[int, ...], float] = {}
    for leaf_id in np.unique(leaf_ids):
        rows = tuple(np.flatnonzero(leaf_ids == leaf_id).tolist())
        signatures[rows] = float(predictions[rows[0]])
    return signatures


def compare_leaf_signatures(
    first_ids: np.ndarray,
    first_predictions: np.ndarray,
    second_ids: np.ndarray,
    second_predictions: np.ndarray,
) -> tuple[bool, float]:
    first = leaf_signatures(first_ids, first_predictions)
    second = leaf_signatures(second_ids, second_predictions)
    if set(first) != set(second):
        return False, float("inf")
    maximum_difference = max(
        abs(first[rows] - second[rows]) for rows in first
    )
    return True, float(maximum_difference)


def sklearn_structure_comparison(
    scratch: ScratchCARTRegressor,
    library: DecisionTreeRegressor,
) -> pd.DataFrame:
    rows = []
    scratch_nodes = scratch.iter_nodes()
    library_tree = library.tree_
    n_rows = max(len(scratch_nodes), library_tree.node_count)

    for node_id in range(n_rows):
        scratch_node = (
            scratch_nodes[node_id] if node_id < len(scratch_nodes) else None
        )
        library_exists = node_id < library_tree.node_count
        library_feature = (
            int(library_tree.feature[node_id]) if library_exists else None
        )
        library_is_leaf = library_feature == -2 if library_exists else None
        rows.append(
            {
                "node_id": node_id,
                "scratch_exists": scratch_node is not None,
                "sklearn_exists": library_exists,
                "scratch_is_leaf": (
                    scratch_node.is_leaf if scratch_node is not None else None
                ),
                "sklearn_is_leaf": library_is_leaf,
                "scratch_feature_index": (
                    scratch_node.feature_index
                    if scratch_node is not None
                    and not scratch_node.is_leaf
                    else None
                ),
                "sklearn_feature_index": (
                    library_feature
                    if library_exists and not library_is_leaf
                    else None
                ),
                "scratch_threshold": (
                    scratch_node.threshold
                    if scratch_node is not None
                    and not scratch_node.is_leaf
                    else None
                ),
                "sklearn_threshold": (
                    float(library_tree.threshold[node_id])
                    if library_exists and not library_is_leaf
                    else None
                ),
                "scratch_prediction": (
                    scratch_node.prediction
                    if scratch_node is not None
                    else None
                ),
                "sklearn_prediction": (
                    float(library_tree.value[node_id].reshape(-1)[0])
                    if library_exists
                    else None
                ),
                "scratch_n_samples": (
                    scratch_node.n_samples
                    if scratch_node is not None
                    else None
                ),
                "sklearn_n_samples": (
                    int(library_tree.n_node_samples[node_id])
                    if library_exists
                    else None
                ),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison["feature_match"] = (
        comparison["scratch_feature_index"].fillna(-2)
        == comparison["sklearn_feature_index"].fillna(-2)
    )
    comparison["leaf_match"] = (
        comparison["scratch_is_leaf"] == comparison["sklearn_is_leaf"]
    )
    comparison["threshold_abs_diff"] = (
        comparison["scratch_threshold"]
        - comparison["sklearn_threshold"]
    ).abs()
    comparison["prediction_abs_diff"] = (
        comparison["scratch_prediction"]
        - comparison["sklearn_prediction"]
    ).abs()
    comparison["sample_count_match"] = (
        comparison["scratch_n_samples"] == comparison["sklearn_n_samples"]
    )
    return comparison


# ============================================================
# Diagnostic tests

def split_comparison_record(
    case: str,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    min_samples_leaf: int,
) -> dict[str, object]:
    vectorized = find_best_split_vectorized(
        x, y, indices, min_samples_leaf
    )
    brute = find_best_split_bruteforce(
        x, y, indices, min_samples_leaf
    )

    both_none = vectorized is None and brute is None
    if both_none:
        return {
            "case": case,
            "vectorized_feature": None,
            "bruteforce_feature": None,
            "vectorized_threshold": None,
            "bruteforce_threshold": None,
            "child_sse_abs_diff": 0.0,
            "partition_match": True,
            "passed": True,
        }
    if vectorized is None or brute is None:
        return {
            "case": case,
            "vectorized_feature": (
                vectorized.feature_index if vectorized else None
            ),
            "bruteforce_feature": brute.feature_index if brute else None,
            "vectorized_threshold": (
                vectorized.threshold if vectorized else None
            ),
            "bruteforce_threshold": brute.threshold if brute else None,
            "child_sse_abs_diff": float("inf"),
            "child_sse_rel_diff": float("inf"),
            "partition_match": False,
            "passed": False,
        }

    partition_match = (
        set(vectorized.left_indices) == set(brute.left_indices)
        and set(vectorized.right_indices) == set(brute.right_indices)
    )
    child_sse_abs_diff = abs(vectorized.child_sse - brute.child_sse)
    child_sse_scale = max(
        abs(vectorized.child_sse),
        abs(brute.child_sse),
        1.0,
    )
    child_sse_rel_diff = child_sse_abs_diff / child_sse_scale
    passed = (
        vectorized.feature_index == brute.feature_index
        and abs(vectorized.threshold - brute.threshold) <= 1e-12
        and np.isclose(
            vectorized.child_sse,
            brute.child_sse,
            rtol=1e-12,
            atol=1e-8,
        )
        and partition_match
    )
    return {
        "case": case,
        "vectorized_feature": vectorized.feature_index,
        "bruteforce_feature": brute.feature_index,
        "vectorized_threshold": vectorized.threshold,
        "bruteforce_threshold": brute.threshold,
        "child_sse_abs_diff": child_sse_abs_diff,
        "child_sse_rel_diff": child_sse_rel_diff,
        "partition_match": partition_match,
        "passed": passed,
    }


def run_split_verification(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> pd.DataFrame:
    records = []

    synthetic_x = np.asarray(
        [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0],
         [4.0, 0.0], [5.0, -1.0]],
        dtype=np.float32,
    )
    synthetic_y = np.asarray([0, 0, 0, 10, 10, 10], dtype=np.float64)
    records.append(
        split_comparison_record(
            "synthetic_known_split",
            synthetic_x,
            synthetic_y,
            np.arange(len(synthetic_y)),
            min_samples_leaf=1,
        )
    )

    rng = np.random.default_rng(2026)
    for sample_number, sample_size in enumerate((60, 90, 140), start=1):
        sampled = np.sort(
            rng.choice(len(y_train), size=sample_size, replace=False)
        )
        records.append(
            split_comparison_record(
                f"real_sample_{sample_number}_n{sample_size}",
                x_train,
                y_train,
                sampled,
                min_samples_leaf=MIN_SAMPLES_LEAF,
            )
        )

    result = pd.DataFrame(records)
    if not result["passed"].all():
        raise AssertionError("Vectorized and brute-force split searches differ")
    return result


def run_unit_tests() -> pd.DataFrame:
    records = []

    def record(name: str, passed: bool, detail: str) -> None:
        records.append(
            {"test": name, "passed": bool(passed), "detail": detail}
        )
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    constant_x = np.arange(20, dtype=np.float32).reshape(-1, 1)
    constant_y = np.repeat(7.5, 20)
    constant_tree = ScratchCARTRegressor(max_depth=5).fit(
        constant_x, constant_y, ["x"]
    )
    record(
        "constant_target_stays_one_leaf",
        constant_tree.get_n_leaves() == 1
        and np.allclose(constant_tree.predict(constant_x), 7.5),
        f"leaves={constant_tree.get_n_leaves()}",
    )

    repeated_x = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)
    repeated_y = np.asarray([0, 0, 0, 10, 10, 10], dtype=np.float64)
    repeated_tree = ScratchCARTRegressor(
        max_depth=1, min_samples_leaf=2
    ).fit(repeated_x, repeated_y, ["x"])
    record(
        "repeated_values_known_threshold",
        not repeated_tree.root_.is_leaf
        and abs(repeated_tree.root_.threshold - 0.5) <= 1e-12,
        f"threshold={repeated_tree.root_.threshold}",
    )

    too_small_x = np.arange(8, dtype=np.float32).reshape(-1, 1)
    too_small_y = np.arange(8, dtype=np.float64)
    too_small_tree = ScratchCARTRegressor(
        max_depth=4, min_samples_leaf=5
    ).fit(too_small_x, too_small_y, ["x"])
    record(
        "minimum_leaf_prevents_invalid_split",
        too_small_tree.get_n_leaves() == 1,
        f"leaves={too_small_tree.get_n_leaves()}",
    )

    known_x = np.arange(6, dtype=np.float32).reshape(-1, 1)
    known_y = np.asarray([0, 0, 0, 10, 10, 10], dtype=np.float64)
    known_tree = ScratchCARTRegressor(
        max_depth=1, min_samples_leaf=1
    ).fit(known_x, known_y, ["x"])
    record(
        "known_optimal_split_and_predictions",
        abs(known_tree.root_.threshold - 2.5) <= 1e-12
        and np.allclose(
            known_tree.predict(known_x),
            [0, 0, 0, 10, 10, 10],
        ),
        f"threshold={known_tree.root_.threshold}",
    )

    return pd.DataFrame(records)


# ============================================================
# Reporting

def save_official_cv_plot(
    official_cv: pd.DataFrame,
    selected_depth: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    depths = official_cv["max_depth"].to_numpy()
    for fold_number in range(1, N_FOLDS + 1):
        ax.plot(
            depths,
            official_cv[f"fold_{fold_number}_mse"],
            color="#9ca3af",
            linewidth=1.0,
            alpha=0.65,
        )
    ax.plot(
        depths,
        official_cv["mean_cv_mse"],
        color="#1d4ed8",
        marker="o",
        linewidth=2.5,
        label="Mean five-fold MSE",
    )
    selected_mse = float(
        official_cv.loc[
            official_cv["max_depth"] == selected_depth,
            "mean_cv_mse",
        ].iloc[0]
    )
    ax.scatter(
        [selected_depth],
        [selected_mse],
        color="#b91c1c",
        s=70,
        zorder=5,
        label=f"First minimum: depth {selected_depth}",
    )
    ax.set(
        xlabel="Maximum tree depth",
        ylabel="Validation MSE",
        title="Scratch CART Manual Five-Fold Cross-Validation",
        xticks=list(OFFICIAL_DEPTHS),
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        EXPERIMENT_DIR / "official_depth_cv_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_depth3_tree_plot(tree: ScratchCARTRegressor) -> None:
    nodes = {node.node_id: node for node in tree.iter_nodes()}
    positions = {
        0: (0.50, 0.91),
        1: (0.25, 0.70),
        8: (0.75, 0.70),
        2: (0.125, 0.48),
        5: (0.375, 0.48),
        9: (0.625, 0.48),
        12: (0.875, 0.48),
        3: (0.0625, 0.19),
        4: (0.1875, 0.19),
        6: (0.3125, 0.19),
        7: (0.4375, 0.19),
        10: (0.5625, 0.19),
        11: (0.6875, 0.19),
        13: (0.8125, 0.19),
        14: (0.9375, 0.19),
    }
    branch_labels = {
        (0, 1): "Yes: quality <= 6",
        (0, 8): "No: quality >= 7",
        (1, 2): "Yes: living area <= 1,112",
        (1, 5): "No: living area > 1,112",
        (8, 9): "Yes: quality = 7",
        (8, 12): "No: quality >= 8",
        (2, 3): "Yes",
        (2, 4): "No",
        (5, 6): "Yes",
        (5, 7): "No",
        (9, 10): "Yes",
        (9, 11): "No",
        (12, 13): "Yes",
        (12, 14): "No",
    }

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.axis("off")

    for parent in nodes.values():
        if parent.is_leaf:
            continue
        for child in (parent.left, parent.right):
            x_parent, y_parent = positions[parent.node_id]
            x_child, y_child = positions[child.node_id]
            ax.annotate(
                "",
                xy=(x_child, y_child + 0.055),
                xytext=(x_parent, y_parent - 0.055),
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#64748b",
                    "linewidth": 1.4,
                },
            )
            ax.text(
                (x_parent + x_child) / 2,
                (y_parent + y_child) / 2 + 0.012,
                branch_labels[(parent.node_id, child.node_id)],
                ha="center",
                va="center",
                fontsize=8.5,
                color="#334155",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
            )

    for node_id, node in nodes.items():
        x_position, y_position = positions[node_id]
        if node.is_leaf:
            label = (
                f"LEAF\nn = {node.n_samples}\n"
                f"Predicted price = {node.prediction:.1f} kEUR"
            )
            facecolor = "#dcfce7"
            edgecolor = "#15803d"
        else:
            feature_name = tree.feature_names_[node.feature_index]
            if feature_name == "years_since_built_2011":
                split_text = "YearBuilt >= 1950?"
            elif feature_name == "OverallQual":
                split_text = f"OverallQual <= {node.threshold:g}?"
            else:
                split_text = f"{feature_name} <= {node.threshold:g}?"
            label = (
                f"{split_text}\n"
                f"n = {node.n_samples} | mean = {node.prediction:.1f} kEUR\n"
                f"SSE reduction = {node.sse_reduction:,.0f}"
            )
            facecolor = "#dbeafe"
            edgecolor = "#1d4ed8"
        ax.text(
            x_position,
            y_position,
            label,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="semibold" if not node.is_leaf else "normal",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": 1.5,
            },
        )

    ax.set_title(
        "Required Depth-3 Scratch CART: How a Property Reaches Its Price Prediction",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0.5,
        0.985,
        (
            "Start at the root, answer each question, and follow the labeled "
            "branch. The leaf mean is the final prediction."
        ),
        ha="center",
        va="top",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(
        EXPERIMENT_DIR / "depth3_cart_tree_intuition.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_findings(
    official_cv: pd.DataFrame,
    selected_depth: int,
    model_diagnostics: pd.DataFrame,
    upper_level_splits: pd.DataFrame,
    equivalence_metrics: dict[str, object],
    stability_summary: pd.DataFrame,
    year_equivalence: pd.DataFrame,
    split_verification: pd.DataFrame,
    unit_tests: pd.DataFrame,
    source_integrity: pd.DataFrame,
) -> None:
    selected_cv = official_cv.loc[
        official_cv["max_depth"] == selected_depth
    ].iloc[0]
    selection_counts = stability_summary.loc[
        stability_summary["selection_count"] > 0,
        ["max_depth", "selection_count"],
    ]
    selection_text = ", ".join(
        f"depth {int(row.max_depth)}: {int(row.selection_count)}"
        for row in selection_counts.itertuples()
    )
    cv_rows = "\n".join(
        f"| {int(row.max_depth)} | {row.mean_cv_mse:.3f} | "
        f"{row.sd_cv_mse:.3f} | {row.oof_rmse:.3f} |"
        for row in official_cv.itertuples()
    )
    diagnostic_rows = "\n".join(
        f"| {row.model} | {int(row.max_depth)} | {int(row.node_count)} | "
        f"{int(row.leaf_count)} | {row.training_mse:.3f} | "
        f"{row.validation_rmse:.3f} |"
        for row in model_diagnostics.itertuples()
    )
    upper_split_rows = "\n".join(
        f"| {int(row.depth)} | {int(row.node_id)} | "
        f"`{row.split_description_transformed}` | "
        f"`{row.split_description_original}` | {int(row.n_samples)} | "
        f"{row.sse_reduction:.3f} |"
        for row in upper_level_splits.loc[
            upper_level_splits["model"] == "required_depth_3_scratch"
        ].itertuples()
    )
    required_year_checks = year_equivalence.loc[
        year_equivalence["required_check"]
    ]
    year_cv_check = year_equivalence.loc[
        year_equivalence["comparison"] == "official_cv_curve"
    ].iloc[0]

    findings = f"""# CART Task 1 Findings

## Scope

This experiment implements only Task 1. It does not create final submission
code, slides, the complete results JSON, or models for Tasks 2 through 5.

The model reads the original supplied CSV files directly. In memory it
replaces `YearBuilt` and `YearRemodAdd` with differences from 2011. The
original `LotArea` values are retained without clamping.

## Scratch CART Method

- Greedy binary recursive partitioning.
- Every valid midpoint between distinct sorted predictor values is tested.
- A split minimizes left-child SSE plus right-child SSE.
- Each terminal prediction is the mean response in that leaf.
- `min_samples_leaf=5`.
- Predictors are represented as `float32`, matching sklearn's documented tree
  input conversion; outcomes and cumulative SSE calculations use `float64`.
- Equal split scores use deterministic feature-order and threshold-order
  tie-breaking.
- The split search uses cumulative sums and cumulative squared sums.

The class also accepts per-node feature subsampling, which will allow the same
tree implementation to serve as the foundation for the scratch random forest
in Task 2.

## Official Five-Fold Depth Search

The official folds use one deterministic shuffle with seed {OFFICIAL_CV_SEED}.
Each of the 300 observations appears in exactly one 60-row validation fold.

| Maximum depth | Mean CV MSE | Fold SD | OOF RMSE |
|---:|---:|---:|---:|
{cv_rows}

The first minimum occurs at **depth {selected_depth}**, with mean CV MSE
**{selected_cv.mean_cv_mse:.3f}** and OOF RMSE
**{selected_cv.oof_rmse:.3f}**.

The required JSON fields have different roles:

- `tree_pred_test` must contain predictions from the required depth-3 scratch
  tree.
- `tree_lib_pred_test` must contain predictions from the fixed depth-3
  sklearn comparison tree.
- `tree_max_depth` records the depth selected by scratch CV, which is
  {selected_depth}.
- `tree_cv_mse_curve` records the nine MSE values for depths 2 through 10.

## Fitted Tree Diagnostics

| Model | Depth | Nodes | Leaves | Training MSE | Validation RMSE |
|---|---:|---:|---:|---:|---:|
{diagnostic_rows}

The depth-3 root split is
`{equivalence_metrics["root_feature"]} <= {equivalence_metrics["root_threshold"]:.6g}`.

### Depth-3 Upper Splits

| Depth | Node | Transformed split | Original-scale report | Rows | SSE reduction |
|---:|---:|---|---|---:|---:|
{upper_split_rows}

## Scratch-versus-sklearn Equivalence

- Test prediction RMSE: {equivalence_metrics["test_prediction_rmse"]:.3e}
- Maximum test prediction difference:
  {equivalence_metrics["test_prediction_max_abs_diff"]:.3e}
- Training prediction RMSE:
  {equivalence_metrics["train_prediction_rmse"]:.3e}
- Matching test rows: {equivalence_metrics["matching_test_rows"]}/200
- Matching depth: {equivalence_metrics["depth_match"]}
- Matching node count: {equivalence_metrics["node_count_match"]}
- Matching leaf count: {equivalence_metrics["leaf_count_match"]}
- Matching training leaf partitions:
  {equivalence_metrics["leaf_partition_match"]}
- Maximum matching-leaf prediction difference:
  {equivalence_metrics["leaf_prediction_max_abs_diff"]:.3e}

The required tolerances were passed: RMSE below `1e-10` and maximum absolute
difference below `1e-9`.

## Robustness Diagnostics

The official answer remains based only on seed {OFFICIAL_CV_SEED}. Additional
fixed seeds were used to assess depth-selection stability, not to search for a
favorable answer.

Selection counts across {len(STABILITY_SEEDS)} fixed seeds: {selection_text}.

All required full-tree year-transformation checks passed:
{int(required_year_checks["passed"].sum())}/{len(required_year_checks)}.
The transformations preserve the training partitions, although their
decreasing direction swaps the visual left and right branches.

The transformed-year and original-year CV searches both selected depth
{int(year_cv_check["selected_depth_transformed"])}. Their fold MSE curves were
not numerically identical: the maximum absolute depth-level difference was
{year_cv_check["prediction_max_abs_diff"]:.3f}. A held-out calendar year can
equal a midpoint that was absent from that fold's training rows. Because CART
routes equality through the `<=` branch, reversing the year axis can route
that boundary row to the complementary branch. This diagnostic does not
change the required official result, which is fitted and selected entirely
with the transformed columns.

The vectorized split search agreed with brute force in
{int(split_verification["passed"].sum())}/{len(split_verification)} cases.
All {int(unit_tests["passed"].sum())} focused unit tests passed.

## Data Integrity

- Train SHA-256 before and after:
  `{source_integrity.loc[source_integrity.file == "train", "hash_before"].iloc[0]}`
- Test SHA-256 before and after:
  `{source_integrity.loc[source_integrity.file == "test", "hash_before"].iloc[0]}`
- Original train `LotArea` maximum:
  {source_integrity.loc[source_integrity.file == "train", "lot_area_max"].iloc[0]:.0f}
- Original test `LotArea` maximum:
  {source_integrity.loc[source_integrity.file == "test", "lot_area_max"].iloc[0]:.0f}

The hashes are unchanged and the capped experimental CSV was never read.

## 2026 Research Insights Used

1. The current 2026 scikit-learn decision-tree documentation describes
   squared error as variance reduction, terminal means as L2 predictions,
   `min_samples_leaf` as a smoothing control, internal `float32` conversion,
   and randomized resolution of exactly tied splits. The scratch
   implementation matches the numerical semantics needed by this dataset,
   while retaining explicit deterministic tie-breaking.
2. The 2026 paper *Regularisation of CART Trees by Summation of p-values*
   formalizes L2 CART as greedy optimal recursive binary splitting and
   emphasizes controlling tree complexity. Its proposed stopping rule was not
   used because this assignment explicitly requires five-fold depth tuning.
3. A 2026 decision-tree pruning project reinforces the practical
   bias-variance role of controlling tree size. Again, the assignment's
   depth-2-to-10 CV rule takes precedence over external pruning schemes.

Sources:

- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
- https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/tree
- https://su.diva-portal.org/smash/get/diva2%3A2043553/FULLTEXT01.pdf
- https://github.com/btboilerplate/Decisiontree_Classification

No external CART source code was copied.

## Files

- `cart_task1_experiment.py`
- `cart_task1_results.json`
- `official_depth_cv.csv`
- `official_depth_cv_curve.png`
- `depth3_cart_tree_intuition.png`
- `official_cv_fold_assignments.csv`
- `official_selected_depth_oof_predictions.csv`
- `repeated_cv_depth_stability.csv`
- `depth_selection_stability_summary.csv`
- `cart_model_diagnostics.csv`
- `cart_tree_structure.csv`
- `cart_upper_level_splits.csv`
- `cart_equivalence_metrics.csv`
- `cart_equivalence_nodes.csv`
- `cart_test_predictions.csv`
- `year_transform_equivalence.csv`
- `year_transform_cv_comparison.csv`
- `split_search_verification.csv`
- `cart_unit_tests.csv`
- `source_data_integrity.csv`
"""
    (EXPERIMENT_DIR / "CART_FINDINGS.md").write_text(
        findings, encoding="utf-8"
    )


# ============================================================
# Main experiment

def main() -> None:
    train_hash_before = sha256(TRAIN_PATH)
    test_hash_before = sha256(TEST_PATH)
    data = load_experiment_data()

    train_raw = data["train_raw"]
    test_raw = data["test_raw"]
    x_train = data["x_train_transformed"]
    x_test = data["x_test_transformed"]
    x_train_original = data["x_train_original"]
    x_test_original = data["x_test_original"]
    y_train = data["y_train"]

    unit_tests = run_unit_tests()
    split_verification = run_split_verification(x_train, y_train)

    initial_cv, _, fold_assignments = cross_validate_depths(
        x_train,
        y_train,
        OFFICIAL_DEPTHS,
        OFFICIAL_CV_SEED,
    )
    selected_depth = choose_first_minimum_depth(initial_cv)
    official_cv, selected_oof, fold_assignments = cross_validate_depths(
        x_train,
        y_train,
        OFFICIAL_DEPTHS,
        OFFICIAL_CV_SEED,
        return_oof_for_depth=selected_depth,
    )
    if not initial_cv.equals(official_cv):
        raise AssertionError("Repeated official CV execution was not deterministic")

    depth3_scratch = ScratchCARTRegressor(
        max_depth=3,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    ).fit(x_train, y_train, TRANSFORMED_FEATURES)
    save_depth3_tree_plot(depth3_scratch)
    selected_tree = ScratchCARTRegressor(
        max_depth=selected_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    ).fit(x_train, y_train, TRANSFORMED_FEATURES)

    # This is the single library tree call permitted by Task 1.
    depth3_library = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        criterion="squared_error",
        random_state=0,
    )
    depth3_library.fit(x_train, y_train)

    scratch_test_pred = depth3_scratch.predict(x_test)
    library_test_pred = depth3_library.predict(x_test)
    scratch_train_pred = depth3_scratch.predict(x_train)
    library_train_pred = depth3_library.predict(x_train)
    selected_test_pred = selected_tree.predict(x_test)
    selected_train_pred = selected_tree.predict(x_train)

    test_diff = scratch_test_pred - library_test_pred
    train_diff = scratch_train_pred - library_train_pred
    test_rmse = float(np.sqrt(np.mean(test_diff * test_diff)))
    test_max_abs = float(np.max(np.abs(test_diff)))
    train_rmse = float(np.sqrt(np.mean(train_diff * train_diff)))

    scratch_train_leaves = depth3_scratch.apply(x_train)
    library_train_leaves = depth3_library.apply(x_train)
    leaf_partition_match, leaf_prediction_difference = (
        compare_leaf_signatures(
            scratch_train_leaves,
            scratch_train_pred,
            library_train_leaves,
            library_train_pred,
        )
    )
    node_comparison = sklearn_structure_comparison(
        depth3_scratch, depth3_library
    )

    equivalence_metrics = {
        "test_prediction_rmse": test_rmse,
        "test_prediction_max_abs_diff": test_max_abs,
        "train_prediction_rmse": train_rmse,
        "matching_test_rows": int(
            np.isclose(
                scratch_test_pred,
                library_test_pred,
                rtol=0.0,
                atol=1e-10,
            ).sum()
        ),
        "depth_match": (
            depth3_scratch.get_depth() == depth3_library.get_depth()
        ),
        "node_count_match": (
            depth3_scratch.node_count_ == depth3_library.tree_.node_count
        ),
        "leaf_count_match": (
            depth3_scratch.get_n_leaves() == depth3_library.get_n_leaves()
        ),
        "leaf_partition_match": leaf_partition_match,
        "leaf_prediction_max_abs_diff": leaf_prediction_difference,
        "all_node_features_match": bool(
            node_comparison["feature_match"].all()
        ),
        "all_node_types_match": bool(node_comparison["leaf_match"].all()),
        "all_node_sample_counts_match": bool(
            node_comparison["sample_count_match"].all()
        ),
        "maximum_node_threshold_difference": float(
            node_comparison["threshold_abs_diff"].fillna(0.0).max()
        ),
        "maximum_node_prediction_difference": float(
            node_comparison["prediction_abs_diff"].fillna(0.0).max()
        ),
        "root_feature": TRANSFORMED_FEATURES[
            depth3_scratch.root_.feature_index
        ],
        "root_threshold": float(depth3_scratch.root_.threshold),
    }

    if test_rmse >= 1e-10 or test_max_abs >= 1e-9:
        raise AssertionError("Scratch depth-3 predictions do not match sklearn")
    if equivalence_metrics["matching_test_rows"] != len(test_raw):
        raise AssertionError("Not all test rows match sklearn predictions")
    if not all(
        [
            equivalence_metrics["depth_match"],
            equivalence_metrics["node_count_match"],
            equivalence_metrics["leaf_count_match"],
            equivalence_metrics["leaf_partition_match"],
            equivalence_metrics["all_node_features_match"],
            equivalence_metrics["all_node_types_match"],
            equivalence_metrics["all_node_sample_counts_match"],
        ]
    ):
        raise AssertionError("Scratch and sklearn tree structures differ")

    official_cv = official_cv.copy()
    official_cv["selected"] = (
        official_cv["max_depth"] == selected_depth
    )
    official_cv.to_csv(
        EXPERIMENT_DIR / "official_depth_cv.csv", index=False
    )
    save_official_cv_plot(official_cv, selected_depth)
    fold_assignments.to_csv(
        EXPERIMENT_DIR / "official_cv_fold_assignments.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "source_row_index": np.arange(len(y_train)),
            "actual_sale_price_keur": y_train,
            "oof_prediction": selected_oof,
            "residual": y_train - selected_oof,
        }
    ).to_csv(
        EXPERIMENT_DIR
        / "official_selected_depth_oof_predictions.csv",
        index=False,
    )

    stability_frames = []
    for seed in STABILITY_SEEDS:
        seed_cv, _, _ = cross_validate_depths(
            x_train, y_train, OFFICIAL_DEPTHS, seed
        )
        seed_selected = choose_first_minimum_depth(seed_cv)
        seed_cv["selected"] = seed_cv["max_depth"] == seed_selected
        stability_frames.append(seed_cv)
    repeated_stability = pd.concat(stability_frames, ignore_index=True)
    repeated_stability.to_csv(
        EXPERIMENT_DIR / "repeated_cv_depth_stability.csv",
        index=False,
    )
    stability_summary = (
        repeated_stability.groupby("max_depth")
        .agg(
            selection_count=("selected", "sum"),
            mean_mse_across_seeds=("mean_cv_mse", "mean"),
            sd_mse_across_seeds=("mean_cv_mse", "std"),
            min_mse_across_seeds=("mean_cv_mse", "min"),
            max_mse_across_seeds=("mean_cv_mse", "max"),
        )
        .reset_index()
    )
    stability_summary.to_csv(
        EXPERIMENT_DIR / "depth_selection_stability_summary.csv",
        index=False,
    )

    original_cv, _, _ = cross_validate_depths(
        x_train_original,
        y_train,
        OFFICIAL_DEPTHS,
        OFFICIAL_CV_SEED,
    )
    original_depth3 = ScratchCARTRegressor(
        max_depth=3,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    ).fit(x_train_original, y_train, ORIGINAL_FEATURES)
    original_selected = ScratchCARTRegressor(
        max_depth=selected_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    ).fit(x_train_original, y_train, ORIGINAL_FEATURES)

    year_comparisons = []
    for model_name, transformed_model, original_model in [
        ("depth_3", depth3_scratch, original_depth3),
        ("cv_selected_depth", selected_tree, original_selected),
    ]:
        for split_name, transformed_x, original_x in [
            ("train", x_train, x_train_original),
            ("test", x_test, x_test_original),
        ]:
            transformed_prediction = transformed_model.predict(transformed_x)
            original_prediction = original_model.predict(original_x)
            difference = transformed_prediction - original_prediction
            transformed_ids = transformed_model.apply(transformed_x)
            original_ids = original_model.apply(original_x)
            partition_match, leaf_difference = compare_leaf_signatures(
                transformed_ids,
                transformed_prediction,
                original_ids,
                original_prediction,
            )
            year_comparisons.append(
                {
                    "comparison": f"{model_name}_{split_name}",
                    "check_type": "full_tree_partition_equivalence",
                    "required_check": True,
                    "prediction_rmse": float(
                        np.sqrt(np.mean(difference * difference))
                    ),
                    "prediction_max_abs_diff": float(
                        np.max(np.abs(difference))
                    ),
                    "leaf_partition_match": partition_match,
                    "leaf_prediction_max_abs_diff": leaf_difference,
                    "passed": (
                        np.max(np.abs(difference)) <= 1e-9
                        and partition_match
                        and leaf_difference <= 1e-9
                    ),
                    "selected_depth_transformed": None,
                    "selected_depth_original": None,
                    "note": (
                        "Equivalent leaf partitions and predictions; "
                        "left/right branch labels may be reversed."
                    ),
                }
            )

    cv_curve_difference = (
        official_cv["mean_cv_mse"].to_numpy()
        - original_cv["mean_cv_mse"].to_numpy()
    )
    original_selected_depth = choose_first_minimum_depth(original_cv)
    year_cv_comparison = pd.DataFrame(
        {
            "max_depth": official_cv["max_depth"].to_numpy(dtype=int),
            "transformed_year_mean_cv_mse": official_cv[
                "mean_cv_mse"
            ].to_numpy(),
            "original_year_mean_cv_mse": original_cv[
                "mean_cv_mse"
            ].to_numpy(),
            "mse_difference_transformed_minus_original": cv_curve_difference,
        }
    )
    year_cv_comparison["absolute_mse_difference"] = np.abs(
        year_cv_comparison[
            "mse_difference_transformed_minus_original"
        ]
    )
    year_cv_comparison.to_csv(
        EXPERIMENT_DIR / "year_transform_cv_comparison.csv",
        index=False,
    )
    year_comparisons.append(
        {
            "comparison": "official_cv_curve",
            "check_type": "held_out_boundary_sensitivity",
            "required_check": False,
            "prediction_rmse": float(
                np.sqrt(np.mean(cv_curve_difference * cv_curve_difference))
            ),
            "prediction_max_abs_diff": float(
                np.max(np.abs(cv_curve_difference))
            ),
            "leaf_partition_match": None,
            "leaf_prediction_max_abs_diff": None,
            "passed": bool(selected_depth == original_selected_depth),
            "selected_depth_transformed": selected_depth,
            "selected_depth_original": original_selected_depth,
            "note": (
                "The selected depth is stable. Fold MSEs can differ when a "
                "held-out year equals an unseen midpoint and <= routing is "
                "reversed by the decreasing transformation."
            ),
        }
    )
    year_equivalence = pd.DataFrame(year_comparisons)
    year_equivalence.to_csv(
        EXPERIMENT_DIR / "year_transform_equivalence.csv",
        index=False,
    )
    required_year_checks = year_equivalence.loc[
        year_equivalence["required_check"]
    ]
    if not required_year_checks["passed"].all():
        raise AssertionError("Year transformation changed CART predictions")

    selected_cv_row = official_cv.loc[
        official_cv["max_depth"] == selected_depth
    ].iloc[0]
    model_diagnostics = pd.DataFrame(
        [
            {
                "model": "required_depth_3_scratch",
                "max_depth": 3,
                "node_count": depth3_scratch.node_count_,
                "leaf_count": depth3_scratch.get_n_leaves(),
                "training_mse": float(
                    np.mean((y_train - scratch_train_pred) ** 2)
                ),
                "validation_rmse": float(
                    official_cv.loc[
                        official_cv["max_depth"] == 3, "oof_rmse"
                    ].iloc[0]
                ),
            },
            {
                "model": "cv_selected_scratch",
                "max_depth": selected_depth,
                "node_count": selected_tree.node_count_,
                "leaf_count": selected_tree.get_n_leaves(),
                "training_mse": float(
                    np.mean((y_train - selected_train_pred) ** 2)
                ),
                "validation_rmse": float(selected_cv_row["oof_rmse"]),
            },
        ]
    )
    model_diagnostics.to_csv(
        EXPERIMENT_DIR / "cart_model_diagnostics.csv",
        index=False,
    )

    tree_structure = pd.concat(
        [
            depth3_scratch.structure_frame("required_depth_3_scratch"),
            selected_tree.structure_frame("cv_selected_scratch"),
        ],
        ignore_index=True,
    )
    tree_structure.to_csv(
        EXPERIMENT_DIR / "cart_tree_structure.csv", index=False
    )
    upper_level_splits = tree_structure.loc[
        (~tree_structure["is_leaf"]) & (tree_structure["depth"] <= 2)
    ].copy()
    upper_level_splits.to_csv(
        EXPERIMENT_DIR / "cart_upper_level_splits.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in equivalence_metrics.items()
        ]
    ).to_csv(
        EXPERIMENT_DIR / "cart_equivalence_metrics.csv",
        index=False,
    )
    node_comparison.to_csv(
        EXPERIMENT_DIR / "cart_equivalence_nodes.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "test_row_index": np.arange(len(test_raw)),
            "scratch_depth3_prediction": scratch_test_pred,
            "sklearn_depth3_prediction": library_test_pred,
            "absolute_difference": np.abs(test_diff),
            "scratch_cv_selected_depth_prediction": selected_test_pred,
        }
    ).to_csv(
        EXPERIMENT_DIR / "cart_test_predictions.csv",
        index=False,
    )

    split_verification.to_csv(
        EXPERIMENT_DIR / "split_search_verification.csv",
        index=False,
    )
    unit_tests.to_csv(
        EXPERIMENT_DIR / "cart_unit_tests.csv", index=False
    )

    train_hash_after = sha256(TRAIN_PATH)
    test_hash_after = sha256(TEST_PATH)
    source_integrity = pd.DataFrame(
        [
            {
                "file": "train",
                "path": str(TRAIN_PATH),
                "hash_before": train_hash_before,
                "hash_after": train_hash_after,
                "hash_unchanged": train_hash_before == train_hash_after,
                "rows": len(train_raw),
                "lot_area_max": float(train_raw["LotArea"].max()),
            },
            {
                "file": "test",
                "path": str(TEST_PATH),
                "hash_before": test_hash_before,
                "hash_after": test_hash_after,
                "hash_unchanged": test_hash_before == test_hash_after,
                "rows": len(test_raw),
                "lot_area_max": float(test_raw["LotArea"].max()),
            },
        ]
    )
    if not source_integrity["hash_unchanged"].all():
        raise AssertionError("Source CSV files changed during the experiment")
    source_integrity.to_csv(
        EXPERIMENT_DIR / "source_data_integrity.csv",
        index=False,
    )

    task1_results = {
        "tree_pred_test": scratch_test_pred.astype(float).tolist(),
        "tree_max_depth": int(selected_depth),
        "tree_cv_mse_curve": official_cv["mean_cv_mse"]
        .astype(float)
        .tolist(),
        "tree_lib_pred_test": library_test_pred.astype(float).tolist(),
        "_experiment_only": {
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "official_cv_seed": OFFICIAL_CV_SEED,
            "feature_names_used": TRANSFORMED_FEATURES,
            "feature_name_output_mapping": OUTPUT_FEATURE_NAMES,
            "source_train_sha256": train_hash_before,
            "source_test_sha256": test_hash_before,
            "scratch_sklearn_test_rmse": test_rmse,
            "scratch_sklearn_test_max_abs_diff": test_max_abs,
        },
    }
    with open(
        EXPERIMENT_DIR / "cart_task1_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(task1_results, file, indent=2)

    write_findings(
        official_cv=official_cv,
        selected_depth=selected_depth,
        model_diagnostics=model_diagnostics,
        upper_level_splits=upper_level_splits,
        equivalence_metrics=equivalence_metrics,
        stability_summary=stability_summary,
        year_equivalence=year_equivalence,
        split_verification=split_verification,
        unit_tests=unit_tests,
        source_integrity=source_integrity,
    )

    print(f"Selected depth: {selected_depth}")
    print(f"Depth-3 scratch/sklearn test RMSE: {test_rmse:.3e}")
    print(f"Wrote CART experiment artifacts to: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()
