from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier


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
MAX_ROUNDS = 4000
EARLY_STOPPING_ROUNDS = 100


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


@dataclass
class FittedBundle:
    model: XGBClassifier
    preprocessor: Any | None
    feature_order: list[str]

    def transform(self, x: pd.DataFrame) -> Any:
        return self.preprocessor.transform(x) if self.preprocessor is not None else x

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.transform(x))[:, 1]


@dataclass
class ProbabilityCalibrator:
    method: str
    fitted: Any | None

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = clipped_probabilities(probabilities)
        if self.method == "none":
            return probabilities
        if self.method == "sigmoid":
            logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
            return self.fitted.predict_proba(logits)[:, 1]
        if self.method == "isotonic":
            return np.asarray(self.fitted.predict(probabilities), dtype=float)
        raise ValueError(self.method)


def structural_space() -> list[dict[str, Any]]:
    keys = [
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
    ]
    values = [
        [1, 2, 3, 4],
        [0.01, 0.03, 0.05, 0.1],
        [1, 3, 5, 10],
        [0.0, 0.1, 0.5, 1.0],
        [0.7, 0.85, 1.0],
        [0.7, 0.85, 1.0],
        [1.0, 5.0, 10.0],
        [0.0, 0.1, 1.0],
    ]
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]


def imbalance_candidates() -> list[dict[str, Any]]:
    candidates = []
    for weight in [1.0, 1.5, 2.0, 2.32, 3.0, 5.383]:
        for max_delta_step in [0.0, 1.0]:
            candidates.append(
                {
                    "imbalance_strategy": "none" if weight == 1.0 else "scale_pos_weight",
                    "scale_pos_weight": weight,
                    "max_delta_step": max_delta_step,
                    "sampling_ratio": None,
                    "k_neighbors": None,
                }
            )
    candidates.extend(
        {
            "imbalance_strategy": "random_over",
            "scale_pos_weight": 1.0,
            "max_delta_step": 0.0,
            "sampling_ratio": ratio,
            "k_neighbors": None,
        }
        for ratio in [0.2, 0.3, 0.4]
    )
    candidates.extend(
        {
            "imbalance_strategy": "smote",
            "scale_pos_weight": 1.0,
            "max_delta_step": 0.0,
            "sampling_ratio": ratio,
            "k_neighbors": neighbors,
        }
        for ratio in [0.2, 0.3, 0.4]
        for neighbors in [2, 3, 4]
    )
    candidates.append(
        {
            "imbalance_strategy": "smote",
            "scale_pos_weight": 1.0,
            "max_delta_step": 0.0,
            "sampling_ratio": 0.7,
            "k_neighbors": 10,
        }
    )
    return candidates


def natural_imbalance() -> dict[str, Any]:
    return {
        "imbalance_strategy": "none",
        "scale_pos_weight": 1.0,
        "max_delta_step": 0.0,
        "sampling_ratio": None,
        "k_neighbors": None,
    }


def make_model(
    structure: dict[str, Any],
    imbalance: dict[str, Any],
    n_estimators: int,
    seed: int,
    early_stop: bool,
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stop else None,
        scale_pos_weight=float(imbalance["scale_pos_weight"]),
        max_delta_step=float(imbalance["max_delta_step"]),
        **structure,
    )


def prepare_training_data(
    x: pd.DataFrame,
    y: pd.Series,
    imbalance: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, Any | None, list[str]]:
    strategy = imbalance["imbalance_strategy"]
    if strategy == "random_over":
        sampler = RandomOverSampler(
            sampling_strategy=TargetMinorityRatio(float(imbalance["sampling_ratio"])),
            random_state=seed,
        )
        x_resampled, y_resampled = sampler.fit_resample(x, y)
        return x_resampled, np.asarray(y_resampled), None, FEATURES
    if strategy == "smote":
        preprocessor = build_smote_preprocessor()
        transformed = preprocessor.fit_transform(x)
        sampler = SMOTE(
            sampling_strategy=TargetMinorityRatio(float(imbalance["sampling_ratio"])),
            k_neighbors=int(imbalance["k_neighbors"]),
            random_state=seed,
        )
        x_resampled, y_resampled = sampler.fit_resample(transformed, y)
        return x_resampled, np.asarray(y_resampled), preprocessor, transformed_feature_order()
    return x, np.asarray(y), None, FEATURES


def importance_from_bundle(
    bundle: FittedBundle,
    importance_type: str = "total_gain",
) -> dict[str, float]:
    raw = bundle.model.get_booster().get_score(importance_type=importance_type)
    values = []
    for index, feature in enumerate(bundle.feature_order):
        values.append(float(raw.get(feature, raw.get(f"f{index}", 0.0))))
    normalized = normalized_vector(values)
    mapped = dict(zip(bundle.feature_order, normalized.tolist(), strict=True))
    return {feature: float(mapped.get(feature, 0.0)) for feature in FEATURES}


def determine_best_rounds(
    x: pd.DataFrame,
    y: pd.Series,
    structure: dict[str, Any],
    imbalance: dict[str, Any],
    seed: int,
) -> int:
    x_inner, x_early, y_inner, y_early = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    train_values, train_target, preprocessor, _ = prepare_training_data(
        x_inner, y_inner, imbalance, seed
    )
    early_values = preprocessor.transform(x_early) if preprocessor is not None else x_early
    model = make_model(structure, imbalance, MAX_ROUNDS, seed, early_stop=True)
    model.fit(
        train_values,
        train_target,
        eval_set=[(early_values, y_early)],
        verbose=False,
    )
    return int(model.best_iteration) + 1


def fit_bundle(
    x: pd.DataFrame,
    y: pd.Series,
    structure: dict[str, Any],
    imbalance: dict[str, Any],
    n_estimators: int,
    seed: int,
) -> FittedBundle:
    train_values, train_target, preprocessor, order = prepare_training_data(
        x, y, imbalance, seed
    )
    model = make_model(structure, imbalance, n_estimators, seed, early_stop=False)
    model.fit(train_values, train_target, verbose=False)
    return FittedBundle(model=model, preprocessor=preprocessor, feature_order=order)


def base_identifier(structure_id: int, imbalance: dict[str, Any]) -> str:
    return (
        f"s{structure_id}|imb={imbalance['imbalance_strategy']}|"
        f"w={imbalance['scale_pos_weight']}|delta={imbalance['max_delta_step']}|"
        f"r={imbalance['sampling_ratio']}|k={imbalance['k_neighbors']}"
    )


def complete_identifier(base_id: str, method: str, folds: int | None) -> str:
    return f"{base_id}|cal={method}|folds={folds}"


def evaluate_raw_candidate(
    x: pd.DataFrame,
    y: pd.Series,
    structure: dict[str, Any],
    imbalance: dict[str, Any],
    structure_id: int,
    cv_seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_id = base_identifier(structure_id, imbalance)
    complete_id = complete_identifier(base_id, "none", None)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    for fold, (train_index, validation_index) in enumerate(splitter.split(x, y), start=1):
        fit_seed = cv_seed * 100 + fold
        rounds = determine_best_rounds(
            x.iloc[train_index],
            y.iloc[train_index],
            structure,
            imbalance,
            fit_seed + 500_000,
        )
        bundle = fit_bundle(
            x.iloc[train_index],
            y.iloc[train_index],
            structure,
            imbalance,
            rounds,
            fit_seed,
        )
        probabilities = bundle.predict_proba(x.iloc[validation_index])
        importance = importance_from_bundle(bundle)
        row = {
            "stage": stage,
            "base_id": base_id,
            "complete_id": complete_id,
            "structure_id": structure_id,
            "cv_seed": cv_seed,
            "fold": fold,
            "best_n_estimators": rounds,
            "log_loss": float(
                log_loss(y.iloc[validation_index], clipped_probabilities(probabilities))
            ),
            "brier_score": float(brier_score_loss(y.iloc[validation_index], probabilities)),
            "roc_auc": float(roc_auc_score(y.iloc[validation_index], probabilities)),
            "mean_probability": float(probabilities.mean()),
            "observed_prevalence": float(y.iloc[validation_index].mean()),
            "prevalence_bias": float(
                probabilities.mean() - y.iloc[validation_index].mean()
            ),
            **structure,
            **imbalance,
            "calibration": "none",
            "calibration_folds": None,
        }
        row.update({f"mdi_{feature}": importance[feature] for feature in FEATURES})
        rows.append(row)
    return rows


def fit_calibrator(
    probabilities: np.ndarray,
    y: np.ndarray,
    method: str,
) -> ProbabilityCalibrator:
    probabilities = clipped_probabilities(probabilities)
    if method == "none":
        return ProbabilityCalibrator(method="none", fitted=None)
    if method == "sigmoid":
        logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
        model = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=2000)
        model.fit(logits, y)
        return ProbabilityCalibrator(method="sigmoid", fitted=model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probabilities, y)
        return ProbabilityCalibrator(method="isotonic", fitted=model)
    raise ValueError(method)


def fit_calibrated_bundle(
    x: pd.DataFrame,
    y: pd.Series,
    structure: dict[str, Any],
    imbalance: dict[str, Any],
    n_estimators: int,
    calibration_method: str,
    calibration_folds: int,
    seed: int,
) -> tuple[FittedBundle, ProbabilityCalibrator]:
    oof_probabilities = np.full(len(x), np.nan, dtype=float)
    splitter = StratifiedKFold(
        n_splits=calibration_folds,
        shuffle=True,
        random_state=seed + 700_000,
    )
    for fold, (train_index, validation_index) in enumerate(splitter.split(x, y), start=1):
        bundle = fit_bundle(
            x.iloc[train_index],
            y.iloc[train_index],
            structure,
            imbalance,
            n_estimators,
            seed * 100 + fold,
        )
        oof_probabilities[validation_index] = bundle.predict_proba(x.iloc[validation_index])
    if not np.all(np.isfinite(oof_probabilities)):
        raise AssertionError("Calibration OOF probabilities are incomplete.")
    calibrator = fit_calibrator(oof_probabilities, y.to_numpy(), calibration_method)
    full_bundle = fit_bundle(x, y, structure, imbalance, n_estimators, seed)
    return full_bundle, calibrator


def evaluate_calibrated_candidate(
    x: pd.DataFrame,
    y: pd.Series,
    structure: dict[str, Any],
    imbalance: dict[str, Any],
    structure_id: int,
    n_estimators: int,
    calibration_method: str,
    calibration_folds: int,
    cv_seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_id = base_identifier(structure_id, imbalance)
    complete_id = complete_identifier(base_id, calibration_method, calibration_folds)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    for fold, (train_index, validation_index) in enumerate(splitter.split(x, y), start=1):
        fit_seed = cv_seed * 100 + fold
        bundle, calibrator = fit_calibrated_bundle(
            x.iloc[train_index],
            y.iloc[train_index],
            structure,
            imbalance,
            n_estimators,
            calibration_method,
            calibration_folds,
            fit_seed,
        )
        raw = bundle.predict_proba(x.iloc[validation_index])
        probabilities = calibrator.transform(raw)
        importance = importance_from_bundle(bundle)
        row = {
            "stage": stage,
            "base_id": base_id,
            "complete_id": complete_id,
            "structure_id": structure_id,
            "cv_seed": cv_seed,
            "fold": fold,
            "best_n_estimators": n_estimators,
            "log_loss": float(
                log_loss(y.iloc[validation_index], clipped_probabilities(probabilities))
            ),
            "brier_score": float(brier_score_loss(y.iloc[validation_index], probabilities)),
            "roc_auc": float(roc_auc_score(y.iloc[validation_index], probabilities)),
            "mean_probability": float(probabilities.mean()),
            "observed_prevalence": float(y.iloc[validation_index].mean()),
            "prevalence_bias": float(
                probabilities.mean() - y.iloc[validation_index].mean()
            ),
            **structure,
            **imbalance,
            "calibration": calibration_method,
            "calibration_folds": calibration_folds,
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
                "base_id": first["base_id"],
                "complete_id": complete_id,
                "structure_id": int(first["structure_id"]),
                "mean_log_loss": float(group["log_loss"].mean()),
                "std_log_loss": float(group["log_loss"].std(ddof=1)),
                "mean_brier_score": float(group["brier_score"].mean()),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "mean_probability": float(group["mean_probability"].mean()),
                "observed_prevalence": float(group["observed_prevalence"].mean()),
                "absolute_prevalence_bias": abs(float(group["prevalence_bias"].mean())),
                "mean_best_n_estimators": float(group["best_n_estimators"].mean()),
                "median_best_n_estimators": int(np.median(group["best_n_estimators"])),
                "mdi_rank_stability": pairwise_spearman(vectors),
                "mdi_top5_overlap": top_k_overlap(vectors),
                **{
                    key: first[key]
                    for key in [
                        "max_depth",
                        "learning_rate",
                        "min_child_weight",
                        "gamma",
                        "subsample",
                        "colsample_bytree",
                        "reg_lambda",
                        "reg_alpha",
                        "imbalance_strategy",
                        "scale_pos_weight",
                        "max_delta_step",
                        "sampling_ratio",
                        "k_neighbors",
                        "calibration",
                        "calibration_folds",
                    ]
                },
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_log_loss", "complete_id"]).reset_index(drop=True)


def row_to_configuration(
    row: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any], str, int | None]:
    structure = {
        "max_depth": int(row["max_depth"]),
        "learning_rate": float(row["learning_rate"]),
        "min_child_weight": float(row["min_child_weight"]),
        "gamma": float(row["gamma"]),
        "subsample": float(row["subsample"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "reg_lambda": float(row["reg_lambda"]),
        "reg_alpha": float(row["reg_alpha"]),
    }
    imbalance = {
        "imbalance_strategy": str(row["imbalance_strategy"]),
        "scale_pos_weight": float(row["scale_pos_weight"]),
        "max_delta_step": float(row["max_delta_step"]),
        "sampling_ratio": None
        if pd.isna(row["sampling_ratio"])
        else float(row["sampling_ratio"]),
        "k_neighbors": None if pd.isna(row["k_neighbors"]) else int(row["k_neighbors"]),
    }
    folds = None if pd.isna(row["calibration_folds"]) else int(row["calibration_folds"])
    return structure, imbalance, str(row["calibration"]), folds


def uses_nontrivial_imbalance(row: pd.Series) -> bool:
    return bool(
        row["imbalance_strategy"] in {"random_over", "smote", "scale_pos_weight"}
        and (
            row["imbalance_strategy"] in {"random_over", "smote"}
            or float(row["scale_pos_weight"]) > 1.0
        )
    )


def select_winner(summary: pd.DataFrame) -> pd.Series:
    compliant = summary[summary.apply(uses_nontrivial_imbalance, axis=1)].copy()
    if compliant.empty:
        raise AssertionError("No Task-5 candidate implements imbalance handling.")
    minimum = float(compliant["mean_log_loss"].min())
    eligible = compliant[compliant["mean_log_loss"] <= minimum + 0.002].copy()
    eligible = eligible.sort_values(
        [
            "mean_brier_score",
            "absolute_prevalence_bias",
            "mdi_rank_stability",
            "std_log_loss",
            "max_depth",
            "median_best_n_estimators",
            "complete_id",
        ],
        ascending=[True, True, False, True, True, True, True],
    )
    return eligible.iloc[0]


def save_plots(
    screen_summary: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    oof_frame: pd.DataFrame,
    importance: dict[str, float],
    learning_frame: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=screen_summary,
        x="mean_best_n_estimators",
        y="mean_log_loss",
        hue="max_depth",
        palette="viridis",
        s=70,
    )
    plt.title("Task 5 XGBoost Structural Screen")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_clf_structural_screen.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    display = repeated_summary.copy()
    display["label"] = display["base_id"].str.replace("|", "\n", regex=False)
    sns.barplot(data=display, x="label", y="mean_log_loss", color="#4472C4")
    plt.xticks(rotation=35, ha="right")
    plt.title("Task 5 Raw Repeated-CV Log Loss")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_clf_repeated_cv.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    display = calibration_summary.copy()
    display["label"] = display["complete_id"].str.replace("|", "\n", regex=False)
    sns.barplot(data=display, x="label", y="mean_log_loss", color="#ED7D31")
    plt.xticks(rotation=35, ha="right")
    plt.title("Task 5 Calibration Comparison")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_clf_calibration_comparison.png", dpi=180)
    plt.close()

    fraction_positive, mean_predicted = calibration_curve(
        oof_frame["actual"],
        oof_frame["probability"],
        n_bins=8,
        strategy="quantile",
    )
    points = pd.DataFrame(
        {"mean_predicted_probability": mean_predicted, "fraction_positive": fraction_positive}
    )
    points.to_csv(EXPERIMENT_DIR / "xgb_clf_calibration_curve.csv", index=False)
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.plot(mean_predicted, fraction_positive, marker="o")
    plt.title("Selected XGBoost Repeated-OOF Calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed distressed fraction")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_clf_calibration_curve.png", dpi=180)
    plt.close()

    importance_frame = pd.DataFrame(
        {"feature": list(importance), "importance": list(importance.values())}
    ).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_frame, x="importance", y="feature", color="#70AD47")
    plt.title("Final XGBoost Classification Total Gain")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_clf_final_total_gain.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=learning_frame, x="iteration", y="log_loss", hue="dataset")
    plt.title("Selected XGBoost Classification Learning Curve")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_clf_learning_curve.png", dpi=180)
    plt.close()


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_original_data()
    x = train[FEATURES]
    y = train[CLASSIFICATION_TARGET].astype(int)
    source_before = source_integrity_frame()
    source_before.to_csv(EXPERIMENT_DIR / "xgb_clf_source_data_integrity.csv", index=False)
    write_json(EXPERIMENT_DIR / "xgb_clf_environment.json", package_versions())

    sampled_structures = deterministic_sample(structural_space(), 60, SCREEN_SEED)
    structure_folds_path = EXPERIMENT_DIR / "xgb_clf_structural_screen_fold_results.csv"
    structure_summary_path = EXPERIMENT_DIR / "xgb_clf_structural_screen_summary.csv"
    structure_shortlist_path = EXPERIMENT_DIR / "xgb_clf_structural_shortlist.csv"
    if (
        structure_folds_path.exists()
        and structure_summary_path.exists()
        and structure_shortlist_path.exists()
    ):
        print("[Task 5] Resuming from structural-screen checkpoints.", flush=True)
        structure_folds = pd.read_csv(structure_folds_path)
        structure_summary = pd.read_csv(structure_summary_path)
        structure_shortlist = pd.read_csv(structure_shortlist_path)
    else:
        rows = []
        for structure_id, structure in enumerate(sampled_structures):
            print(f"[Task 5 structure] {structure_id + 1}/60", flush=True)
            rows.extend(
                evaluate_raw_candidate(
                    x,
                    y,
                    structure,
                    natural_imbalance(),
                    structure_id,
                    SCREEN_SEED,
                    "structure_screen",
                )
            )
        structure_folds = pd.DataFrame(rows)
        structure_summary = summarize(structure_folds)
        structure_shortlist = structure_summary.head(6).copy()
        structure_folds.to_csv(structure_folds_path, index=False)
        structure_summary.to_csv(structure_summary_path, index=False)
        structure_shortlist.to_csv(structure_shortlist_path, index=False)

    imbalance_folds_path = EXPERIMENT_DIR / "xgb_clf_imbalance_fold_results.csv"
    imbalance_summary_path = EXPERIMENT_DIR / "xgb_clf_imbalance_summary.csv"
    imbalance_shortlist_path = EXPERIMENT_DIR / "xgb_clf_imbalance_shortlist.csv"
    if (
        imbalance_folds_path.exists()
        and imbalance_summary_path.exists()
        and imbalance_shortlist_path.exists()
    ):
        print("[Task 5] Resuming from imbalance-screen checkpoints.", flush=True)
        imbalance_folds = pd.read_csv(imbalance_folds_path)
        imbalance_summary = pd.read_csv(imbalance_summary_path)
        imbalance_shortlist = pd.read_csv(imbalance_shortlist_path)
    else:
        rows = []
        candidates = imbalance_candidates()
        for structure_position, (_, structure_row) in enumerate(
            structure_shortlist.iterrows(), start=1
        ):
            structure_id = int(structure_row["structure_id"])
            structure = sampled_structures[structure_id]
            for imbalance_position, imbalance in enumerate(candidates, start=1):
                print(
                    f"[Task 5 imbalance] structure {structure_position}/6, "
                    f"candidate {imbalance_position}/{len(candidates)}",
                    flush=True,
                )
                rows.extend(
                    evaluate_raw_candidate(
                        x,
                        y,
                        structure,
                        imbalance,
                        structure_id,
                        SCREEN_SEED,
                        "imbalance_screen",
                    )
                )
        imbalance_folds = pd.DataFrame(rows)
        imbalance_summary = summarize(imbalance_folds)
        imbalance_shortlist = imbalance_summary.head(6).copy()
        imbalance_folds.to_csv(imbalance_folds_path, index=False)
        imbalance_summary.to_csv(imbalance_summary_path, index=False)
        imbalance_shortlist.to_csv(imbalance_shortlist_path, index=False)

    repeated_folds_path = EXPERIMENT_DIR / "xgb_clf_repeated_cv_fold_results.csv"
    repeated_summary_path = EXPERIMENT_DIR / "xgb_clf_repeated_cv_summary.csv"
    if repeated_folds_path.exists() and repeated_summary_path.exists():
        print("[Task 5] Resuming from raw repeated-CV checkpoints.", flush=True)
        repeated_folds = pd.read_csv(repeated_folds_path)
        repeated_summary = pd.read_csv(repeated_summary_path)
    else:
        rows = []
        for candidate_position, (_, candidate) in enumerate(
            imbalance_shortlist.iterrows(), start=1
        ):
            structure, imbalance, _, _ = row_to_configuration(candidate)
            structure_id = int(candidate["structure_id"])
            for seed in SEEDS:
                print(
                    f"[Task 5 repeated] candidate {candidate_position}/6, seed={seed}",
                    flush=True,
                )
                rows.extend(
                    evaluate_raw_candidate(
                        x,
                        y,
                        structure,
                        imbalance,
                        structure_id,
                        seed,
                        "repeated_raw",
                    )
                )
        repeated_folds = pd.DataFrame(rows)
        repeated_summary = summarize(repeated_folds)
        repeated_folds.to_csv(repeated_folds_path, index=False)
        repeated_summary.to_csv(repeated_summary_path, index=False)

    calibration_folds_path = EXPERIMENT_DIR / "xgb_clf_calibration_fold_results.csv"
    calibration_summary_path = EXPERIMENT_DIR / "xgb_clf_calibration_summary.csv"
    if calibration_folds_path.exists() and calibration_summary_path.exists():
        print("[Task 5] Resuming from calibration checkpoints.", flush=True)
        calibration_folds = pd.read_csv(calibration_folds_path)
        calibration_summary = pd.read_csv(calibration_summary_path)
    else:
        best_raw_loss = float(repeated_summary["mean_log_loss"].min())
        calibration_bases = repeated_summary[
            repeated_summary["mean_log_loss"] <= best_raw_loss + 0.01
        ].head(4)
        if len(calibration_bases) < 4:
            calibration_bases = repeated_summary.head(4)
        rows = []
        for base_position, (_, base) in enumerate(calibration_bases.iterrows(), start=1):
            structure, imbalance, _, _ = row_to_configuration(base)
            structure_id = int(base["structure_id"])
            rounds = int(base["median_best_n_estimators"])
            for method, folds in [
                ("sigmoid", 3),
                ("sigmoid", 5),
                ("sigmoid", 8),
                ("isotonic", 5),
            ]:
                for seed in SEEDS:
                    print(
                        f"[Task 5 calibration] base {base_position}/4, "
                        f"{method}-{folds}, seed={seed}",
                        flush=True,
                    )
                    rows.extend(
                        evaluate_calibrated_candidate(
                            x,
                            y,
                            structure,
                            imbalance,
                            structure_id,
                            rounds,
                            method,
                            folds,
                            seed,
                            "calibration",
                        )
                    )
        calibration_folds = pd.DataFrame(rows)
        calibration_summary = summarize(calibration_folds)
        calibration_folds.to_csv(calibration_folds_path, index=False)
        calibration_summary.to_csv(calibration_summary_path, index=False)

    combined_summary = (
        pd.concat([repeated_summary, calibration_summary], ignore_index=True)
        .sort_values(["mean_log_loss", "complete_id"])
        .reset_index(drop=True)
    )
    combined_summary.to_csv(
        EXPERIMENT_DIR / "xgb_clf_complete_model_summary.csv", index=False
    )
    winner = select_winner(combined_summary)
    structure, imbalance, calibration_method, calibration_folds_count = row_to_configuration(
        winner
    )
    n_estimators = int(winner["median_best_n_estimators"])

    selected_oof_path = EXPERIMENT_DIR / "xgb_clf_selected_oof_probabilities.csv"
    if selected_oof_path.exists():
        print("[Task 5] Resuming from selected OOF probability checkpoint.", flush=True)
        selected_oof = pd.read_csv(selected_oof_path)
    else:
        records = []
        for seed in SEEDS:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (train_index, validation_index) in enumerate(
                splitter.split(x, y), start=1
            ):
                fit_seed = seed * 100 + fold
                if calibration_method == "none":
                    bundle = fit_bundle(
                        x.iloc[train_index],
                        y.iloc[train_index],
                        structure,
                        imbalance,
                        n_estimators,
                        fit_seed,
                    )
                    calibrator = ProbabilityCalibrator("none", None)
                else:
                    bundle, calibrator = fit_calibrated_bundle(
                        x.iloc[train_index],
                        y.iloc[train_index],
                        structure,
                        imbalance,
                        n_estimators,
                        calibration_method,
                        int(calibration_folds_count),
                        fit_seed,
                    )
                probabilities = calibrator.transform(
                    bundle.predict_proba(x.iloc[validation_index])
                )
                records.extend(
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
        selected_oof = pd.DataFrame(records)
        selected_oof.to_csv(selected_oof_path, index=False)

    if calibration_method == "none":
        final_bundle = fit_bundle(
            x, y, structure, imbalance, n_estimators, SCREEN_SEED
        )
        final_calibrator = ProbabilityCalibrator("none", None)
    else:
        final_bundle, final_calibrator = fit_calibrated_bundle(
            x,
            y,
            structure,
            imbalance,
            n_estimators,
            calibration_method,
            int(calibration_folds_count),
            SCREEN_SEED,
        )
    raw_test_probabilities = final_bundle.predict_proba(test[FEATURES])
    test_probabilities = final_calibrator.transform(raw_test_probabilities)
    final_importance = importance_from_bundle(final_bundle, "total_gain")

    diagnostics = {
        importance_type: importance_from_bundle(final_bundle, importance_type)
        for importance_type in ["total_gain", "gain", "weight", "cover", "total_cover"]
    }
    importance_rows = []
    for rank, (feature, value) in enumerate(
        sorted(final_importance.items(), key=lambda item: item[1], reverse=True), start=1
    ):
        importance_rows.append(
            {
                "rank": rank,
                "feature": feature,
                "total_gain": value,
                "gain": diagnostics["gain"][feature],
                "weight": diagnostics["weight"][feature],
                "cover": diagnostics["cover"][feature],
                "total_cover": diagnostics["total_cover"][feature],
            }
        )
    pd.DataFrame(importance_rows).to_csv(
        EXPERIMENT_DIR / "xgb_clf_final_feature_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "row_index": np.arange(len(test_probabilities)),
            "raw_probability": raw_test_probabilities,
            "gbm_clf_pred_prob": test_probabilities,
        }
    ).to_csv(EXPERIMENT_DIR / "xgb_clf_final_test_probabilities.csv", index=False)

    # One fixed split provides a learning-curve diagnostic only; it does not select the model.
    x_learning, x_learning_validation, y_learning, y_learning_validation = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SCREEN_SEED,
    )
    learning_values, learning_target, learning_preprocessor, _ = prepare_training_data(
        x_learning, y_learning, imbalance, SCREEN_SEED
    )
    learning_validation_values = (
        learning_preprocessor.transform(x_learning_validation)
        if learning_preprocessor is not None
        else x_learning_validation
    )
    learning_model = make_model(
        structure, imbalance, MAX_ROUNDS, SCREEN_SEED, early_stop=True
    )
    learning_model.fit(
        learning_values,
        learning_target,
        eval_set=[
            (learning_values, learning_target),
            (learning_validation_values, y_learning_validation),
        ],
        verbose=False,
    )
    learning_rows = []
    for dataset_name, metrics in learning_model.evals_result().items():
        for iteration, value in enumerate(metrics["logloss"], start=1):
            learning_rows.append(
                {"dataset": dataset_name, "iteration": iteration, "log_loss": float(value)}
            )
    learning_frame = pd.DataFrame(learning_rows)
    learning_frame.to_csv(
        EXPERIMENT_DIR / "xgb_clf_learning_curve.csv", index=False
    )

    if calibration_method == "none":
        duplicate_bundle = fit_bundle(
            x, y, structure, imbalance, n_estimators, SCREEN_SEED
        )
        duplicate_calibrator = ProbabilityCalibrator("none", None)
    else:
        duplicate_bundle, duplicate_calibrator = fit_calibrated_bundle(
            x,
            y,
            structure,
            imbalance,
            n_estimators,
            calibration_method,
            int(calibration_folds_count),
            SCREEN_SEED,
        )
    duplicate_probabilities = duplicate_calibrator.transform(
        duplicate_bundle.predict_proba(test[FEATURES])
    )
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
                "test": "imbalance_handling_is_nontrivial",
                "passed": imbalance["imbalance_strategy"] in {"random_over", "smote"}
                or float(imbalance["scale_pos_weight"]) > 1.0,
            },
            {"test": "positive_estimator_count", "passed": n_estimators > 0},
        ]
    )
    tests.to_csv(EXPERIMENT_DIR / "xgb_clf_unit_tests.csv", index=False)
    if not bool(tests["passed"].all()):
        raise AssertionError(tests[~tests["passed"]].to_dict(orient="records"))

    full_hyperparameters = {
        **structure,
        "n_estimators": n_estimators,
        **imbalance,
        "calibration_method": calibration_method,
        "calibration_folds": calibration_folds_count,
    }
    result = {
        "gbm_clf_pred_prob": test_probabilities.tolist(),
        "gbm_clf_var_importance": final_importance,
        "gbm_clf_hyperparams": full_hyperparameters,
    }
    write_json(EXPERIMENT_DIR / "xgb_task5_results.json", result)
    write_json(
        EXPERIMENT_DIR / "xgb_clf_selected_configuration.json",
        {
            **full_hyperparameters,
            "mean_validation_log_loss": float(winner["mean_log_loss"]),
            "mean_brier_score": float(winner["mean_brier_score"]),
            "mean_roc_auc": float(winner["mean_roc_auc"]),
            "mdi_rank_stability": float(winner["mdi_rank_stability"]),
        },
    )
    save_plots(
        structure_summary,
        repeated_summary,
        calibration_summary,
        selected_oof,
        final_importance,
        learning_frame,
    )

    findings = f"""# XGBoost Classification Task 5 Findings

## Scope

This experiment implements only Task 5 with XGBoost. It reads the untouched
original train and test CSV files, preserves original year columns and row
order, and retains the uncapped `LotArea` maximum.

## Data And Scoring

- Training rows: 300
- Distressed cases: 47
- Training prevalence: {y.mean():.6%}
- Primary metric: validation log loss
- Supporting metrics: Brier score, ROC AUC, prevalence bias, and total-gain
  rank stability

Threshold accuracy was not used because the project evaluates probability
cross-entropy.

## Search

- Structural screen: 60 deterministic XGBoost configurations
- Imbalance screen: `scale_pos_weight`, `max_delta_step`, random oversampling,
  and fold-local scaled SMOTE
- Raw finalists: six configurations over 30 outer folds
- Calibration finalists: uncalibrated, sigmoid with 3/5/8 OOF folds, and
  five-fold isotonic calibration

Early stopping used an inner split of each outer training fold. Resampling,
scaling, and calibration were always fitted without access to the outer
validation observations.

## Selected Model

- Max depth: {structure["max_depth"]}
- Learning rate: {structure["learning_rate"]}
- Estimators: {n_estimators}
- Min child weight: {structure["min_child_weight"]}
- Gamma: {structure["gamma"]}
- Subsample: {structure["subsample"]}
- Column subsample: {structure["colsample_bytree"]}
- L2 regularization: {structure["reg_lambda"]}
- L1 regularization: {structure["reg_alpha"]}
- Imbalance strategy: {imbalance["imbalance_strategy"]}
- Positive-class weight: {imbalance["scale_pos_weight"]}
- Max delta step: {imbalance["max_delta_step"]}
- Sampling ratio: {imbalance["sampling_ratio"]}
- SMOTE neighbors: {imbalance["k_neighbors"]}
- Calibration: {calibration_method}
- Calibration folds: {calibration_folds_count}
- Repeated-CV log loss: {winner["mean_log_loss"]:.6f}
- Repeated-CV Brier score: {winner["mean_brier_score"]:.6f}
- Repeated-CV ROC AUC: {winner["mean_roc_auc"]:.6f}

The final estimator count is the median leakage-free best iteration associated
with the winning base configuration.

## Variable Importance

The submission candidate uses normalized XGBoost `total_gain`, the summed loss
reduction assigned to each feature. Gain, split count, cover, and total cover
are retained as diagnostics only.

## Verification

- Focused tests passed: {int(tests["passed"].sum())}/{len(tests)}
- Final probabilities: {len(test_probabilities)}
- Importance sum: {sum(final_importance.values()):.16f}
- Source hashes unchanged: {bool(tests.loc[tests["test"] == "source_hashes_unchanged", "passed"].iloc[0])}

## Research Notes

XGBoost documentation describes `scale_pos_weight` as a useful imbalance
starting point, not a guarantee of calibrated probabilities. This experiment
therefore tunes weighting conservatively and evaluates OOF calibration
separately. The full class ratio and aggressive SMOTE setting are retained as
stress diagnostics. Total gain is predictive importance rather than causal
importance, particularly when housing predictors are correlated.

Sources:

- https://xgboost.readthedocs.io/en/stable/python/python_api.html
- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- https://scikit-learn.org/stable/modules/calibration.html
- https://imbalanced-learn.org/stable/common_pitfalls.html
- https://doi.org/10.3389/frai.2026.1752632
- https://arxiv.org/abs/2602.10524

No external implementation code was copied.
"""
    (EXPERIMENT_DIR / "XGB_CLF_FINDINGS.md").write_text(findings, encoding="utf-8")
    print(json.dumps(full_hyperparameters, indent=2))
    print(f"Selected repeated-CV log loss: {winner['mean_log_loss']:.6f}")


if __name__ == "__main__":
    main()
