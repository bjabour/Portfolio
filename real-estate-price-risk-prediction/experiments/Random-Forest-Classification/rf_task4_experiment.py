from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from modeling_utils import (  # noqa: E402
    CLASSIFICATION_TARGET,
    FEATURES,
    SEEDS,
    build_smote_preprocessor,
    clipped_probabilities,
    config_key,
    deterministic_sample,
    load_original_data,
    normalized_vector,
    package_versions,
    pairwise_spearman,
    probabilities_are_valid,
    source_integrity_frame,
    top_k_overlap,
    transformed_feature_order,
    write_json,
)


SCREEN_SEED = 9618
SCREEN_TREES = 500
FINAL_TREES = 2000
CONVERGENCE_CHECKPOINTS = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000]
warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)


class TargetMinorityRatio:
    def __init__(self, ratio: float):
        self.ratio = float(ratio)

    def __call__(self, y: np.ndarray | pd.Series) -> dict[int, int]:
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        minority_position = int(np.argmin(counts))
        majority_position = int(np.argmax(counts))
        minority_label = int(labels[minority_position])
        current_minority = int(counts[minority_position])
        majority = int(counts[majority_position])
        requested = int(np.ceil(majority * self.ratio))
        return {minority_label: max(current_minority, requested)}


def structural_space() -> list[dict[str, Any]]:
    keys = ["max_depth", "min_samples_leaf", "max_features", "criterion", "max_samples"]
    values = [
        [3, 4, 5, 6, 8, 10, 12],
        [1, 2, 5, 10, 15],
        [3, 4, 6],
        ["gini", "log_loss"],
        [0.7, 1.0],
    ]
    return [dict(zip(keys, values_, strict=True)) for values_ in itertools.product(*values)]


def strategy_candidates() -> list[dict[str, Any]]:
    candidates = [
        {"strategy": "none", "sampling_ratio": None, "k_neighbors": None},
        {"strategy": "balanced", "sampling_ratio": None, "k_neighbors": None},
        {"strategy": "balanced_subsample", "sampling_ratio": None, "k_neighbors": None},
    ]
    candidates.extend(
        {
            "strategy": "random_over",
            "sampling_ratio": ratio,
            "k_neighbors": None,
        }
        for ratio in [0.2, 0.3, 0.4]
    )
    candidates.extend(
        {
            "strategy": "smote",
            "sampling_ratio": ratio,
            "k_neighbors": neighbors,
        }
        for ratio in [0.2, 0.3, 0.4]
        for neighbors in [2, 3, 4]
    )
    candidates.append({"strategy": "smote", "sampling_ratio": 0.7, "k_neighbors": 10})
    return candidates


def make_rf(
    structure: dict[str, Any],
    n_estimators: int,
    seed: int,
    class_weight: str | None = None,
    oob_score: bool = False,
    warm_start: bool = False,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=-1,
        class_weight=class_weight,
        oob_score=oob_score,
        warm_start=warm_start,
        bootstrap=True,
        **structure,
    )


def make_estimator(
    structure: dict[str, Any],
    strategy: dict[str, Any],
    n_estimators: int,
    seed: int,
) -> Any:
    strategy_name = strategy["strategy"]
    if strategy_name == "none":
        return make_rf(structure, n_estimators, seed)
    if strategy_name in {"balanced", "balanced_subsample"}:
        return make_rf(structure, n_estimators, seed, class_weight=strategy_name)
    if strategy_name == "random_over":
        return ImbPipeline(
            [
                (
                    "sampler",
                    RandomOverSampler(
                        sampling_strategy=TargetMinorityRatio(
                            float(strategy["sampling_ratio"])
                        ),
                        random_state=seed,
                    ),
                ),
                ("rf", make_rf(structure, n_estimators, seed)),
            ]
        )
    if strategy_name == "smote":
        return ImbPipeline(
            [
                ("preprocessor", build_smote_preprocessor()),
                (
                    "sampler",
                    SMOTE(
                        sampling_strategy=TargetMinorityRatio(
                            float(strategy["sampling_ratio"])
                        ),
                        k_neighbors=int(strategy["k_neighbors"]),
                        random_state=seed,
                    ),
                ),
                ("rf", make_rf(structure, n_estimators, seed)),
            ]
        )
    raise ValueError(f"Unknown strategy: {strategy_name}")


def extract_rf(estimator: Any) -> tuple[RandomForestClassifier, list[str]]:
    if isinstance(estimator, RandomForestClassifier):
        return estimator, FEATURES
    if hasattr(estimator, "named_steps"):
        order = transformed_feature_order() if "preprocessor" in estimator.named_steps else FEATURES
        return estimator.named_steps["rf"], order
    raise TypeError(f"Cannot extract a forest from {type(estimator)!r}")


def fitted_base_from_calibrator(calibrator: CalibratedClassifierCV) -> Any:
    if len(calibrator.calibrated_classifiers_) != 1:
        raise AssertionError("ensemble=False should leave one full-data calibrated classifier.")
    return calibrator.calibrated_classifiers_[0].estimator


def importance_from_estimator(estimator: Any) -> dict[str, float]:
    forest, order = extract_rf(estimator)
    vector = normalized_vector(forest.feature_importances_)
    mapping = dict(zip(order, vector.tolist(), strict=True))
    return {feature: float(mapping.get(feature, 0.0)) for feature in FEATURES}


def force_serial_prediction(estimator: Any) -> None:
    base = (
        fitted_base_from_calibrator(estimator)
        if isinstance(estimator, CalibratedClassifierCV)
        else estimator
    )
    forest, _ = extract_rf(base)
    forest.set_params(n_jobs=1)


def fit_candidate(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    structure: dict[str, Any],
    strategy: dict[str, Any],
    calibration: dict[str, Any],
    n_estimators: int,
    seed: int,
) -> tuple[Any, Any]:
    estimator = make_estimator(structure, strategy, n_estimators, seed)
    method = calibration["method"]
    if method == "none":
        estimator.fit(x_train, y_train)
        return estimator, estimator
    inner_cv = StratifiedKFold(
        n_splits=int(calibration["folds"]),
        shuffle=True,
        random_state=seed + 700_000,
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=inner_cv,
        ensemble=False,
        n_jobs=1,
    )
    calibrated.fit(x_train, y_train)
    return calibrated, fitted_base_from_calibrator(calibrated)


def evaluate_candidate(
    x: pd.DataFrame,
    y: pd.Series,
    structure: dict[str, Any],
    strategy: dict[str, Any],
    calibration: dict[str, Any],
    complete_id: str,
    structure_id: int,
    cv_seed: int,
    stage: str,
    n_estimators: int = SCREEN_TREES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    for fold, (train_index, validation_index) in enumerate(splitter.split(x, y), start=1):
        fit_seed = cv_seed * 100 + fold
        fitted, fitted_base = fit_candidate(
            x.iloc[train_index],
            y.iloc[train_index],
            structure,
            strategy,
            calibration,
            n_estimators,
            fit_seed,
        )
        probabilities = fitted.predict_proba(x.iloc[validation_index])[:, 1]
        clipped = clipped_probabilities(probabilities)
        importance = importance_from_estimator(fitted_base)
        row = {
            "stage": stage,
            "complete_id": complete_id,
            "structure_id": structure_id,
            "cv_seed": cv_seed,
            "fold": fold,
            "n_train": len(train_index),
            "n_validation": len(validation_index),
            "positive_train": int(y.iloc[train_index].sum()),
            "positive_validation": int(y.iloc[validation_index].sum()),
            "log_loss": float(log_loss(y.iloc[validation_index], clipped, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y.iloc[validation_index], probabilities)),
            "roc_auc": float(roc_auc_score(y.iloc[validation_index], probabilities)),
            "mean_probability": float(probabilities.mean()),
            "observed_prevalence": float(y.iloc[validation_index].mean()),
            "prevalence_bias": float(probabilities.mean() - y.iloc[validation_index].mean()),
            **structure,
            **strategy,
            "calibration": calibration["method"],
            "calibration_folds": calibration["folds"],
        }
        row.update({f"mdi_{feature}": importance[feature] for feature in FEATURES})
        rows.append(row)
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for complete_id, group in frame.groupby("complete_id", sort=False):
        first = group.iloc[0]
        vectors = [
            row[[f"mdi_{feature}" for feature in FEATURES]].to_numpy(dtype=float)
            for _, row in group.iterrows()
        ]
        rows.append(
            {
                "complete_id": complete_id,
                "structure_id": int(first["structure_id"]),
                "mean_log_loss": float(group["log_loss"].mean()),
                "std_log_loss": float(group["log_loss"].std(ddof=1)),
                "mean_brier_score": float(group["brier_score"].mean()),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "mean_probability": float(group["mean_probability"].mean()),
                "observed_prevalence": float(group["observed_prevalence"].mean()),
                "absolute_prevalence_bias": abs(float(group["prevalence_bias"].mean())),
                "mdi_rank_stability": pairwise_spearman(vectors),
                "mdi_top5_overlap": top_k_overlap(vectors),
                **{
                    key: first[key]
                    for key in [
                        "max_depth",
                        "min_samples_leaf",
                        "max_features",
                        "criterion",
                        "max_samples",
                        "strategy",
                        "sampling_ratio",
                        "k_neighbors",
                        "calibration",
                        "calibration_folds",
                    ]
                },
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_log_loss", "complete_id"]).reset_index(drop=True)


def complete_identifier(
    structure_id: int,
    strategy: dict[str, Any],
    calibration: dict[str, Any],
) -> str:
    return (
        f"s{structure_id}|{strategy['strategy']}|r={strategy['sampling_ratio']}|"
        f"k={strategy['k_neighbors']}|cal={calibration['method']}|"
        f"folds={calibration['folds']}"
    )


def row_to_configuration(row: pd.Series) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    structure = {
        "max_depth": int(row["max_depth"]),
        "min_samples_leaf": int(row["min_samples_leaf"]),
        "max_features": int(row["max_features"]),
        "criterion": str(row["criterion"]),
        "max_samples": float(row["max_samples"]),
    }
    sampling_ratio = None if pd.isna(row["sampling_ratio"]) else float(row["sampling_ratio"])
    k_neighbors = None if pd.isna(row["k_neighbors"]) else int(row["k_neighbors"])
    strategy = {
        "strategy": str(row["strategy"]),
        "sampling_ratio": sampling_ratio,
        "k_neighbors": k_neighbors,
    }
    calibration_folds = (
        None if pd.isna(row["calibration_folds"]) else int(row["calibration_folds"])
    )
    calibration = {"method": str(row["calibration"]), "folds": calibration_folds}
    return structure, strategy, calibration


def select_winner(summary: pd.DataFrame) -> pd.Series:
    minimum = float(summary["mean_log_loss"].min())
    eligible = summary[summary["mean_log_loss"] <= minimum + 0.002].copy()
    eligible = eligible.sort_values(
        [
            "mean_brier_score",
            "absolute_prevalence_bias",
            "mdi_rank_stability",
            "std_log_loss",
            "max_depth",
            "complete_id",
        ],
        ascending=[True, True, False, True, True, True],
    )
    return eligible.iloc[0]


def declared_imbalance_method(strategy_name: str) -> str:
    if strategy_name in {"balanced", "balanced_subsample"}:
        return "class_weight_balanced"
    if strategy_name in {"random_over", "smote"}:
        return "resampling"
    if strategy_name == "none":
        return "none"
    raise ValueError(strategy_name)


def convergence_diagnostic(
    x: pd.DataFrame,
    y: pd.Series,
    structure: dict[str, Any],
    strategy: dict[str, Any],
) -> pd.DataFrame:
    class_weight = strategy["strategy"] if strategy["strategy"] in {
        "balanced",
        "balanced_subsample",
    } else None
    diagnostic_strategy = (
        strategy["strategy"] if strategy["strategy"] in {"none", "balanced", "balanced_subsample"}
        else "natural_distribution_diagnostic"
    )
    model = make_rf(
        structure,
        CONVERGENCE_CHECKPOINTS[0],
        SCREEN_SEED,
        class_weight=class_weight,
        oob_score=True,
        warm_start=True,
    )
    rows: list[dict[str, Any]] = []
    for checkpoint in CONVERGENCE_CHECKPOINTS:
        model.set_params(n_estimators=checkpoint)
        model.fit(x, y)
        probabilities = model.oob_decision_function_[:, 1]
        covered = np.isfinite(probabilities)
        importance = normalized_vector(model.feature_importances_)
        row = {
            "n_estimators": checkpoint,
            "diagnostic_strategy": diagnostic_strategy,
            "oob_coverage": float(covered.mean()),
            "oob_log_loss": float(log_loss(y[covered], clipped_probabilities(probabilities[covered]))),
            "oob_brier_score": float(brier_score_loss(y[covered], probabilities[covered])),
            "mean_oob_probability": float(probabilities[covered].mean()),
        }
        row.update({f"mdi_{feature}": importance[index] for index, feature in enumerate(FEATURES)})
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(
    structure_summary: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    y: pd.Series,
    oof_frame: pd.DataFrame,
    importance: dict[str, float],
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=structure_summary,
        x="max_depth",
        y="mean_log_loss",
        hue="min_samples_leaf",
        size="max_features",
        palette="viridis",
    )
    plt.title("Task 4 Random-Forest Structural Screen")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "rf_clf_structural_screen.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    display = repeated_summary.copy()
    display["label"] = display["complete_id"].str.replace("|", "\n", regex=False)
    sns.barplot(data=display, x="label", y="mean_log_loss", color="#4472C4")
    plt.xticks(rotation=35, ha="right")
    plt.title("Task 4 Repeated-CV Log Loss")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "rf_clf_repeated_cv.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=convergence, x="n_estimators", y="oob_log_loss", marker="o")
    plt.title("Random-Forest OOB Log-Loss Convergence")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "rf_clf_oob_convergence.png", dpi=180)
    plt.close()

    fraction_positive, mean_predicted = calibration_curve(
        oof_frame["actual"],
        oof_frame["probability"],
        n_bins=8,
        strategy="quantile",
    )
    calibration_points = pd.DataFrame(
        {"mean_predicted_probability": mean_predicted, "fraction_positive": fraction_positive}
    )
    calibration_points.to_csv(
        EXPERIMENT_DIR / "rf_clf_calibration_curve.csv", index=False
    )
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.plot(mean_predicted, fraction_positive, marker="o")
    plt.title("Selected RF Repeated-OOF Calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed distressed fraction")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "rf_clf_calibration_curve.png", dpi=180)
    plt.close()

    importance_frame = pd.DataFrame(
        {"feature": list(importance), "importance": list(importance.values())}
    ).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_frame, x="importance", y="feature", color="#70AD47")
    plt.title("Final RF Classification MDI")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "rf_clf_final_mdi.png", dpi=180)
    plt.close()


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_original_data()
    x = train[FEATURES]
    y = train[CLASSIFICATION_TARGET].astype(int)
    source_before = source_integrity_frame()
    source_before.to_csv(EXPERIMENT_DIR / "rf_clf_source_data_integrity.csv", index=False)
    write_json(EXPERIMENT_DIR / "rf_clf_environment.json", package_versions())

    sampled_structures = deterministic_sample(structural_space(), 72, SCREEN_SEED)
    no_strategy = {"strategy": "none", "sampling_ratio": None, "k_neighbors": None}
    no_calibration = {"method": "none", "folds": None}
    structure_folds_path = EXPERIMENT_DIR / "rf_clf_structural_screen_fold_results.csv"
    structure_summary_path = EXPERIMENT_DIR / "rf_clf_structural_screen_summary.csv"
    structure_shortlist_path = EXPERIMENT_DIR / "rf_clf_structural_shortlist.csv"
    if (
        structure_folds_path.exists()
        and structure_summary_path.exists()
        and structure_shortlist_path.exists()
    ):
        print("[Task 4] Resuming from structural-screen checkpoints.", flush=True)
        structure_folds = pd.read_csv(structure_folds_path)
        structure_summary = pd.read_csv(structure_summary_path)
        structure_shortlist = pd.read_csv(structure_shortlist_path)
    else:
        structure_rows: list[dict[str, Any]] = []
        for structure_id, structure in enumerate(sampled_structures):
            print(f"[Task 4 structure] {structure_id + 1}/72", flush=True)
            complete_id = complete_identifier(structure_id, no_strategy, no_calibration)
            structure_rows.extend(
                evaluate_candidate(
                    x,
                    y,
                    structure,
                    no_strategy,
                    no_calibration,
                    complete_id,
                    structure_id,
                    SCREEN_SEED,
                    "structure_screen",
                )
            )
        structure_folds = pd.DataFrame(structure_rows)
        structure_summary = summarize(structure_folds)
        structure_shortlist = structure_summary.head(8).copy()
        structure_folds.to_csv(structure_folds_path, index=False)
        structure_summary.to_csv(structure_summary_path, index=False)
        structure_shortlist.to_csv(structure_shortlist_path, index=False)

    imbalance_folds_path = EXPERIMENT_DIR / "rf_clf_imbalance_fold_results.csv"
    imbalance_summary_path = EXPERIMENT_DIR / "rf_clf_imbalance_summary.csv"
    if imbalance_folds_path.exists() and imbalance_summary_path.exists():
        print("[Task 4] Resuming from imbalance-screen checkpoints.", flush=True)
        imbalance_folds = pd.read_csv(imbalance_folds_path)
        imbalance_summary = pd.read_csv(imbalance_summary_path)
    else:
        imbalance_rows: list[dict[str, Any]] = []
        for position, structure_row in structure_shortlist.iterrows():
            structure_id = int(structure_row["structure_id"])
            structure = sampled_structures[structure_id]
            for strategy_position, strategy in enumerate(strategy_candidates(), start=1):
                print(
                    f"[Task 4 imbalance] structure {position + 1}/8, "
                    f"strategy {strategy_position}/{len(strategy_candidates())}",
                    flush=True,
                )
                complete_id = complete_identifier(structure_id, strategy, no_calibration)
                imbalance_rows.extend(
                    evaluate_candidate(
                        x,
                        y,
                        structure,
                        strategy,
                        no_calibration,
                        complete_id,
                        structure_id,
                        SCREEN_SEED,
                        "imbalance_screen",
                    )
                )
        imbalance_folds = pd.DataFrame(imbalance_rows)
        imbalance_summary = summarize(imbalance_folds)
        imbalance_folds.to_csv(imbalance_folds_path, index=False)
        imbalance_summary.to_csv(imbalance_summary_path, index=False)
    base_calibration_shortlist = imbalance_summary.head(12).copy()

    calibration_folds_path = EXPERIMENT_DIR / "rf_clf_calibration_fold_results.csv"
    calibration_summary_path = EXPERIMENT_DIR / "rf_clf_calibration_summary.csv"
    complete_shortlist_path = EXPERIMENT_DIR / "rf_clf_complete_shortlist.csv"
    if (
        calibration_folds_path.exists()
        and calibration_summary_path.exists()
        and complete_shortlist_path.exists()
    ):
        print("[Task 4] Resuming from calibration-screen checkpoints.", flush=True)
        calibration_folds = pd.read_csv(calibration_folds_path)
        calibration_summary = pd.read_csv(calibration_summary_path)
        complete_shortlist = pd.read_csv(complete_shortlist_path)
    else:
        calibration_rows: list[dict[str, Any]] = []
        for base_position, (_, base_row) in enumerate(
            base_calibration_shortlist.iterrows(), start=1
        ):
            structure, strategy, _ = row_to_configuration(base_row)
            structure_id = int(base_row["structure_id"])
            for folds in [3, 5, 8]:
                print(
                    f"[Task 4 calibration] base {base_position}/12, sigmoid folds={folds}",
                    flush=True,
                )
                calibration = {"method": "sigmoid", "folds": folds}
                complete_id = complete_identifier(structure_id, strategy, calibration)
                calibration_rows.extend(
                    evaluate_candidate(
                        x,
                        y,
                        structure,
                        strategy,
                        calibration,
                        complete_id,
                        structure_id,
                        SCREEN_SEED,
                        "calibration_screen",
                    )
                )
        preliminary_calibration = summarize(pd.DataFrame(calibration_rows))
        isotonic_bases = (
            pd.concat([base_calibration_shortlist, preliminary_calibration], ignore_index=True)
            .sort_values("mean_log_loss")
            .drop_duplicates(["structure_id", "strategy", "sampling_ratio", "k_neighbors"])
            .head(4)
        )
        for position, (_, base_row) in enumerate(isotonic_bases.iterrows(), start=1):
            structure, strategy, _ = row_to_configuration(base_row)
            structure_id = int(base_row["structure_id"])
            print(f"[Task 4 calibration] isotonic base {position}/4", flush=True)
            calibration = {"method": "isotonic", "folds": 5}
            complete_id = complete_identifier(structure_id, strategy, calibration)
            calibration_rows.extend(
                evaluate_candidate(
                    x,
                    y,
                    structure,
                    strategy,
                    calibration,
                    complete_id,
                    structure_id,
                    SCREEN_SEED,
                    "calibration_screen",
                )
            )
        calibration_folds = pd.DataFrame(calibration_rows)
        calibration_summary = summarize(calibration_folds)
        all_complete_summary = (
            pd.concat([imbalance_summary, calibration_summary], ignore_index=True)
            .sort_values(["mean_log_loss", "complete_id"])
            .reset_index(drop=True)
        )
        complete_shortlist = all_complete_summary.head(6).copy()
        calibration_folds.to_csv(calibration_folds_path, index=False)
        calibration_summary.to_csv(calibration_summary_path, index=False)
        complete_shortlist.to_csv(complete_shortlist_path, index=False)

    repeated_folds_path = EXPERIMENT_DIR / "rf_clf_repeated_cv_fold_results.csv"
    repeated_summary_path = EXPERIMENT_DIR / "rf_clf_repeated_cv_summary.csv"
    if repeated_folds_path.exists() and repeated_summary_path.exists():
        print("[Task 4] Resuming from repeated-CV checkpoints.", flush=True)
        repeated_folds = pd.read_csv(repeated_folds_path)
        repeated_summary = pd.read_csv(repeated_summary_path)
    else:
        repeated_rows: list[dict[str, Any]] = []
        for candidate_position, (_, candidate) in enumerate(
            complete_shortlist.iterrows(), start=1
        ):
            structure, strategy, calibration = row_to_configuration(candidate)
            structure_id = int(candidate["structure_id"])
            complete_id = complete_identifier(structure_id, strategy, calibration)
            for seed in SEEDS:
                print(
                    f"[Task 4 repeated] candidate {candidate_position}/6, seed={seed}",
                    flush=True,
                )
                repeated_rows.extend(
                    evaluate_candidate(
                        x,
                        y,
                        structure,
                        strategy,
                        calibration,
                        complete_id,
                        structure_id,
                        seed,
                        "repeated",
                    )
                )
        repeated_folds = pd.DataFrame(repeated_rows)
        repeated_summary = summarize(repeated_folds)
        repeated_folds.to_csv(repeated_folds_path, index=False)
        repeated_summary.to_csv(repeated_summary_path, index=False)
    winner = select_winner(repeated_summary)
    structure, strategy, calibration = row_to_configuration(winner)

    selected_id = str(winner["complete_id"])
    selected_oof = repeated_folds[repeated_folds["complete_id"] == selected_id].copy()
    oof_records = []
    splitter_records = []
    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (_, validation_index) in enumerate(splitter.split(x, y), start=1):
            fold_row = selected_oof[
                (selected_oof["cv_seed"] == seed) & (selected_oof["fold"] == fold)
            ].iloc[0]
            splitter_records.append(
                {
                    "cv_seed": seed,
                    "fold": fold,
                    "validation_indices": json.dumps(validation_index.tolist()),
                    "fold_log_loss": fold_row["log_loss"],
                }
            )
    pd.DataFrame(splitter_records).to_csv(
        EXPERIMENT_DIR / "rf_clf_selected_fold_projects.csv", index=False
    )

    oof_path = EXPERIMENT_DIR / "rf_clf_selected_oof_probabilities.csv"
    if oof_path.exists():
        print("[Task 4] Resuming from selected OOF probability checkpoint.", flush=True)
        oof_frame = pd.read_csv(oof_path)
    else:
        # Recreate row-level selected OOF probabilities for reliability diagnostics.
        for seed in SEEDS:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (train_index, validation_index) in enumerate(splitter.split(x, y), start=1):
                fitted, _ = fit_candidate(
                    x.iloc[train_index],
                    y.iloc[train_index],
                    structure,
                    strategy,
                    calibration,
                    SCREEN_TREES,
                    seed * 100 + fold,
                )
                force_serial_prediction(fitted)
                probabilities = fitted.predict_proba(x.iloc[validation_index])[:, 1]
                oof_records.extend(
                    {
                        "cv_seed": seed,
                        "fold": fold,
                        "row_index": int(row_index),
                        "actual": int(y.iloc[row_index]),
                        "probability": float(probability),
                    }
                    for row_index, probability in zip(
                        validation_index, probabilities, strict=True
                    )
                )
        oof_frame = pd.DataFrame(oof_records)
        oof_frame.to_csv(oof_path, index=False)

    final_fitted, final_base = fit_candidate(
        x,
        y,
        structure,
        strategy,
        calibration,
        FINAL_TREES,
        SCREEN_SEED,
    )
    force_serial_prediction(final_fitted)
    test_probabilities = final_fitted.predict_proba(test[FEATURES])[:, 1].astype(float)
    final_importance = importance_from_estimator(final_base)
    convergence = convergence_diagnostic(x, y, structure, strategy)
    convergence.to_csv(EXPERIMENT_DIR / "rf_clf_oob_convergence.csv", index=False)

    importance_rows = [
        {"rank": rank, "feature": feature, "importance": value}
        for rank, (feature, value) in enumerate(
            sorted(final_importance.items(), key=lambda item: item[1], reverse=True), start=1
        )
    ]
    pd.DataFrame(importance_rows).to_csv(
        EXPERIMENT_DIR / "rf_clf_final_feature_importance.csv", index=False
    )
    pd.DataFrame(
        {"row_index": np.arange(len(test_probabilities)), "rf_clf_pred_prob": test_probabilities}
    ).to_csv(EXPERIMENT_DIR / "rf_clf_final_test_probabilities.csv", index=False)

    duplicate_fitted, _ = fit_candidate(
        x,
        y,
        structure,
        strategy,
        calibration,
        FINAL_TREES,
        SCREEN_SEED,
    )
    force_serial_prediction(duplicate_fitted)
    duplicate_probabilities = duplicate_fitted.predict_proba(test[FEATURES])[:, 1]
    source_after = source_integrity_frame()
    tests = pd.DataFrame(
        [
            {"test": "probability_count_200", "passed": len(test_probabilities) == 200},
            {
                "test": "probabilities_valid",
                "passed": probabilities_are_valid(test_probabilities),
            },
            {"test": "training_prevalence_47_of_300", "passed": int(y.sum()) == 47},
            {
                "test": "importance_has_12_features",
                "passed": set(final_importance) == set(FEATURES),
            },
            {
                "test": "importance_sums_to_one",
                "passed": abs(sum(final_importance.values()) - 1.0) < 1e-12,
            },
            {
                "test": "deterministic_probabilities",
                "passed": bool(np.array_equal(test_probabilities, duplicate_probabilities)),
            },
            {
                "test": "source_hashes_unchanged",
                "passed": source_before["sha256"].tolist() == source_after["sha256"].tolist(),
            },
            {"test": "original_lotarea_maximum", "passed": float(x["LotArea"].max()) == 215245.0},
            {
                "test": "declared_method_is_allowed",
                "passed": declared_imbalance_method(strategy["strategy"])
                in {"class_weight_balanced", "resampling", "none"},
            },
            {"test": "final_tree_count_at_least_200", "passed": FINAL_TREES >= 200},
        ]
    )
    tests.to_csv(EXPERIMENT_DIR / "rf_clf_unit_tests.csv", index=False)
    if not bool(tests["passed"].all()):
        raise AssertionError(tests[~tests["passed"]].to_dict(orient="records"))

    result = {
        "rf_clf_pred_prob": test_probabilities.tolist(),
        "rf_clf_var_importance": final_importance,
        "rf_clf_imbalance_method": declared_imbalance_method(strategy["strategy"]),
        "rf_clf_n_trees": FINAL_TREES,
        "rf_clf_mtry": int(structure["max_features"]),
        "rf_clf_max_depth": int(structure["max_depth"]),
    }
    write_json(EXPERIMENT_DIR / "rf_task4_results.json", result)
    write_json(
        EXPERIMENT_DIR / "rf_clf_selected_configuration.json",
        {
            **structure,
            **strategy,
            "calibration_method": calibration["method"],
            "calibration_folds": calibration["folds"],
            "n_estimators": FINAL_TREES,
            "mean_validation_log_loss": float(winner["mean_log_loss"]),
            "mean_brier_score": float(winner["mean_brier_score"]),
            "mean_roc_auc": float(winner["mean_roc_auc"]),
            "mdi_rank_stability": float(winner["mdi_rank_stability"]),
            "declared_imbalance_method": declared_imbalance_method(strategy["strategy"]),
        },
    )
    save_plots(
        structure_summary,
        repeated_summary,
        convergence,
        y,
        oof_frame,
        final_importance,
    )

    findings = f"""# Random-Forest Classification Task 4 Findings

## Scope

This experiment implements only Task 4 with sklearn's library random forest.
It uses the untouched original CSV files, original year columns, uncapped
`LotArea`, and original row order.

## Data And Scoring

- Training rows: 300
- Distressed cases: 47
- Training prevalence: {y.mean():.6%}
- Primary model-selection metric: validation log loss
- Supporting metrics: Brier score, ROC AUC, prevalence bias, and MDI stability
- Threshold accuracy was not used for selection because the project evaluates
  predicted probabilities.

## Search

- Structural screen: 72 deterministic 500-tree configurations
- Imbalance screen: natural training, class weighting, random oversampling,
  and fold-local SMOTE
- Calibration screen: uncalibrated, sigmoid with 3/5/8 inner folds, and
  five-fold isotonic diagnostics
- Final comparison: six complete configurations over 30 untouched outer folds

All resampling and calibration occurred inside training folds. For SMOTE only,
area variables received fold-local `log1p` plus standard scaling and all other
variables received fold-local standard scaling.

## Selected Model

- Max depth: {structure["max_depth"]}
- Minimum leaf size: {structure["min_samples_leaf"]}
- mtry: {structure["max_features"]}
- Criterion: {structure["criterion"]}
- Bootstrap sample fraction: {structure["max_samples"]}
- Training strategy: {strategy["strategy"]}
- Sampling ratio: {strategy["sampling_ratio"]}
- SMOTE neighbors: {strategy["k_neighbors"]}
- Calibration: {calibration["method"]}
- Calibration folds: {calibration["folds"]}
- Final trees: {FINAL_TREES}
- Declared imbalance method: `{declared_imbalance_method(strategy["strategy"])}`
- Repeated-CV log loss: {winner["mean_log_loss"]:.6f}
- Repeated-CV Brier score: {winner["mean_brier_score"]:.6f}
- Repeated-CV ROC AUC: {winner["mean_roc_auc"]:.6f}

## Variable Importance

The submission candidate is the normalized sklearn impurity-decrease vector
from the fitted underlying forest. Calibration changes probabilities but does
not change the forest's split-based importance.

## Verification

- Focused tests passed: {int(tests["passed"].sum())}/{len(tests)}
- Final probabilities: {len(test_probabilities)}
- Importance sum: {sum(final_importance.values()):.16f}
- Source hashes unchanged: {bool(tests.loc[tests["test"] == "source_hashes_unchanged", "passed"].iloc[0])}

## Research Notes

Current sklearn documentation defines forest probabilities as averages of
tree class probabilities and exposes impurity-based feature importance.
Calibration is evaluated because the evaluation target is cross-entropy rather
than classification accuracy. Imbalanced-learn's leakage guidance supports
placing every sampler inside the training-fold pipeline. MDI remains a
predictive model importance and can divide credit among correlated housing
features.

Sources:

- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/calibration.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
- https://imbalanced-learn.org/stable/common_pitfalls.html
- https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf

No external implementation code was copied.
"""
    (EXPERIMENT_DIR / "RF_CLF_FINDINGS.md").write_text(findings, encoding="utf-8")
    print(json.dumps(result, indent=2)[:1200])
    print(f"Selected repeated-CV log loss: {winner['mean_log_loss']:.6f}")


if __name__ == "__main__":
    main()
