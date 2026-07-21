from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from modeling_utils import (  # noqa: E402
    FEATURES,
    REGRESSION_TARGET,
    SEEDS,
    config_key,
    deterministic_sample,
    load_original_data,
    normalized_xgb_importance,
    package_versions,
    pairwise_spearman,
    rmse_from_mse,
    source_integrity_frame,
    top_k_overlap,
    write_json,
)


SCREEN_SEED = 9618
MAX_ROUNDS = 4000
EARLY_STOPPING_ROUNDS = 100


def candidate_space() -> list[dict]:
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
        [1, 2, 3, 4, 5],
        [0.01, 0.03, 0.05, 0.1],
        [1, 3, 5, 10],
        [0.0, 0.1, 0.5, 1.0],
        [0.7, 0.85, 1.0],
        [0.7, 0.85, 1.0],
        [1.0, 5.0, 10.0],
        [0.0, 0.1, 1.0],
    ]
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]


def make_model(config: dict, n_estimators: int, random_state: int, early_stop: bool) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stop else None,
        **config,
    )


def evaluate_config(
    x: pd.DataFrame,
    y: pd.Series,
    config: dict,
    cv_seed: int,
    candidate_id: int,
    stage: str,
) -> tuple[list[dict], list[np.ndarray]]:
    rows: list[dict] = []
    importance_vectors: list[np.ndarray] = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    for fold, (train_index, validation_index) in enumerate(splitter.split(x), start=1):
        x_outer_train = x.iloc[train_index]
        y_outer_train = y.iloc[train_index]
        x_validation = x.iloc[validation_index]
        y_validation = y.iloc[validation_index]
        inner_seed = cv_seed * 100 + fold
        x_inner_train, x_early, y_inner_train, y_early = train_test_split(
            x_outer_train,
            y_outer_train,
            test_size=0.2,
            random_state=inner_seed,
        )
        early_model = make_model(config, MAX_ROUNDS, inner_seed, early_stop=True)
        early_model.fit(
            x_inner_train,
            y_inner_train,
            eval_set=[(x_early, y_early)],
            verbose=False,
        )
        best_rounds = int(early_model.best_iteration) + 1
        model = make_model(config, best_rounds, inner_seed, early_stop=False)
        model.fit(x_outer_train, y_outer_train, verbose=False)
        predictions = model.predict(x_validation)
        mse = float(mean_squared_error(y_validation, predictions))
        importance = normalized_xgb_importance(model, "total_gain")
        importance_vector = np.asarray([importance[feature] for feature in FEATURES])
        importance_vectors.append(importance_vector)
        row = {
            "stage": stage,
            "candidate_id": candidate_id,
            "config_key": config_key(config),
            "cv_seed": cv_seed,
            "fold": fold,
            "n_train": len(train_index),
            "n_validation": len(validation_index),
            "best_n_estimators": best_rounds,
            "mse": mse,
            "rmse": rmse_from_mse(mse),
        }
        row.update(config)
        row.update({f"mdi_{feature}": importance[feature] for feature in FEATURES})
        rows.append(row)
    return rows, importance_vectors


def summarize_folds(frame: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict] = []
    for (candidate_id, key), group in frame.groupby(["candidate_id", "config_key"], sort=False):
        vectors = [
            row[[f"mdi_{feature}" for feature in FEATURES]].to_numpy(dtype=float)
            for _, row in group.iterrows()
        ]
        first = group.iloc[0]
        summaries.append(
            {
                "candidate_id": int(candidate_id),
                "config_key": key,
                "mean_mse": float(group["mse"].mean()),
                "std_mse": float(group["mse"].std(ddof=1)),
                "rmse_from_mean_mse": rmse_from_mse(group["mse"].mean()),
                "mean_best_n_estimators": float(group["best_n_estimators"].mean()),
                "median_best_n_estimators": int(np.median(group["best_n_estimators"])),
                "mdi_rank_stability": pairwise_spearman(vectors),
                "mdi_top5_overlap": top_k_overlap(vectors),
                **{key: first[key] for key in [
                    "max_depth",
                    "learning_rate",
                    "min_child_weight",
                    "gamma",
                    "subsample",
                    "colsample_bytree",
                    "reg_lambda",
                    "reg_alpha",
                ]},
            }
        )
    return pd.DataFrame(summaries).sort_values(["mean_mse", "candidate_id"]).reset_index(drop=True)


def select_shortlist(screen_summary: pd.DataFrame) -> pd.DataFrame:
    selected_ids = set(screen_summary.head(8)["candidate_id"].astype(int))
    for _, depth_group in screen_summary.groupby("max_depth"):
        selected_ids.add(int(depth_group.iloc[0]["candidate_id"]))
    return (
        screen_summary[screen_summary["candidate_id"].isin(selected_ids)]
        .sort_values(["mean_mse", "candidate_id"])
        .reset_index(drop=True)
    )


def select_winner(summary: pd.DataFrame) -> pd.Series:
    minimum = float(summary["mean_mse"].min())
    eligible = summary[summary["mean_mse"] <= minimum * 1.0025].copy()
    eligible = eligible.sort_values(
        [
            "mdi_rank_stability",
            "std_mse",
            "max_depth",
            "median_best_n_estimators",
            "candidate_id",
        ],
        ascending=[False, True, True, True, True],
    )
    return eligible.iloc[0]


def save_plots(
    screen_summary: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    final_model: XGBRegressor,
    x: pd.DataFrame,
    y: pd.Series,
    selected_config: dict,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=screen_summary,
        x="mean_best_n_estimators",
        y="mean_mse",
        hue="max_depth",
        palette="viridis",
        s=70,
    )
    plt.title("Task 3 XGBoost Regression Screening")
    plt.xlabel("Mean early-stopped boosting rounds")
    plt.ylabel("Five-fold validation MSE")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_reg_screening.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    plot_frame = repeated_summary.copy()
    plot_frame["label"] = plot_frame.apply(
        lambda row: f"id={int(row.candidate_id)}, d={int(row.max_depth)}", axis=1
    )
    sns.barplot(data=plot_frame, x="label", y="mean_mse", color="#4472C4")
    plt.xticks(rotation=45, ha="right")
    plt.title("Repeated Five-Fold Validation MSE")
    plt.xlabel("Shortlisted configuration")
    plt.ylabel("Mean MSE across 30 folds")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_reg_repeated_cv.png", dpi=180)
    plt.close()

    split_x_train, split_x_validation, split_y_train, split_y_validation = train_test_split(
        x, y, test_size=0.2, random_state=SCREEN_SEED
    )
    diagnostic = make_model(selected_config, MAX_ROUNDS, SCREEN_SEED, early_stop=True)
    diagnostic.fit(
        split_x_train,
        split_y_train,
        eval_set=[(split_x_train, split_y_train), (split_x_validation, split_y_validation)],
        verbose=False,
    )
    evaluations = diagnostic.evals_result()
    learning_rows = []
    for dataset_name, values in evaluations.items():
        for iteration, value in enumerate(values["rmse"], start=1):
            learning_rows.append(
                {"dataset": dataset_name, "iteration": iteration, "rmse": float(value)}
            )
    learning_frame = pd.DataFrame(learning_rows)
    learning_frame.to_csv(EXPERIMENT_DIR / "xgb_reg_learning_curve.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=learning_frame, x="iteration", y="rmse", hue="dataset")
    plt.title("Selected XGBoost Learning Curve")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_reg_learning_curve.png", dpi=180)
    plt.close()

    fitted = final_model.predict(x)
    residuals = y.to_numpy() - fitted
    residual_frame = pd.DataFrame(
        {"row_index": np.arange(len(y)), "actual": y, "fitted": fitted, "residual": residuals}
    )
    residual_frame.to_csv(EXPERIMENT_DIR / "xgb_reg_training_residuals.csv", index=False)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=fitted, y=residuals, s=45)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Final XGBoost Training Residuals")
    plt.xlabel("Fitted sale price (kEUR)")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(EXPERIMENT_DIR / "xgb_reg_residuals.png", dpi=180)
    plt.close()


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_original_data()
    x = train[FEATURES]
    y = train[REGRESSION_TARGET]

    source_before = source_integrity_frame()
    source_before.to_csv(EXPERIMENT_DIR / "xgb_reg_source_data_integrity.csv", index=False)
    write_json(EXPERIMENT_DIR / "xgb_reg_environment.json", package_versions())

    sampled_configs = deterministic_sample(candidate_space(), 60, SCREEN_SEED)
    screen_folds_path = EXPERIMENT_DIR / "xgb_reg_screening_fold_results.csv"
    screen_summary_path = EXPERIMENT_DIR / "xgb_reg_screening_summary.csv"
    shortlist_path = EXPERIMENT_DIR / "xgb_reg_shortlist.csv"
    if screen_folds_path.exists() and screen_summary_path.exists() and shortlist_path.exists():
        print("[Task 3] Resuming from saved screening results.", flush=True)
        screen_folds = pd.read_csv(screen_folds_path)
        screen_summary = pd.read_csv(screen_summary_path)
        shortlist = pd.read_csv(shortlist_path)
    else:
        screen_rows: list[dict] = []
        for candidate_id, config in enumerate(sampled_configs):
            print(f"[Task 3 screen] candidate {candidate_id + 1}/60", flush=True)
            rows, _ = evaluate_config(x, y, config, SCREEN_SEED, candidate_id, "screen")
            screen_rows.extend(rows)
        screen_folds = pd.DataFrame(screen_rows)
        screen_summary = summarize_folds(screen_folds)
        shortlist = select_shortlist(screen_summary)
        screen_folds.to_csv(screen_folds_path, index=False)
        screen_summary.to_csv(screen_summary_path, index=False)
        shortlist.to_csv(shortlist_path, index=False)

    config_lookup = {index: config for index, config in enumerate(sampled_configs)}
    repeated_folds_path = EXPERIMENT_DIR / "xgb_reg_repeated_cv_fold_results.csv"
    repeated_summary_path = EXPERIMENT_DIR / "xgb_reg_repeated_cv_summary.csv"
    if repeated_folds_path.exists() and repeated_summary_path.exists():
        print("[Task 3] Resuming from saved repeated-CV results.", flush=True)
        repeated_folds = pd.read_csv(repeated_folds_path)
        repeated_summary = pd.read_csv(repeated_summary_path)
    else:
        repeated_rows: list[dict] = []
        for shortlist_position, candidate_id in enumerate(
            shortlist["candidate_id"].astype(int), start=1
        ):
            config = config_lookup[candidate_id]
            for seed in SEEDS:
                print(
                    f"[Task 3 repeated] candidate {shortlist_position}/{len(shortlist)}, seed={seed}",
                    flush=True,
                )
                rows, _ = evaluate_config(x, y, config, seed, candidate_id, "repeated")
                repeated_rows.extend(rows)
        repeated_folds = pd.DataFrame(repeated_rows)
        repeated_summary = summarize_folds(repeated_folds)
        repeated_folds.to_csv(repeated_folds_path, index=False)
        repeated_summary.to_csv(repeated_summary_path, index=False)
    winner = select_winner(repeated_summary)
    selected_id = int(winner["candidate_id"])
    selected_config = config_lookup[selected_id]
    selected_rounds = int(winner["median_best_n_estimators"])
    selected_fold_rows = repeated_folds[repeated_folds["candidate_id"] == selected_id]
    selected_fold_rows[
        ["cv_seed", "fold", "best_n_estimators", "mse", "rmse"]
    ].to_csv(EXPERIMENT_DIR / "xgb_reg_best_iteration_stability.csv", index=False)

    final_model = make_model(selected_config, selected_rounds, SCREEN_SEED, early_stop=False)
    final_model.fit(x, y, verbose=False)
    test_predictions = final_model.predict(test[FEATURES]).astype(float)
    final_importance = normalized_xgb_importance(final_model, "total_gain")

    importance_rows = []
    diagnostics = {
        importance_type: normalized_xgb_importance(final_model, importance_type)
        for importance_type in ["total_gain", "gain", "weight", "cover", "total_cover"]
    }
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
    importance_frame = pd.DataFrame(importance_rows)
    importance_frame.to_csv(
        EXPERIMENT_DIR / "xgb_reg_final_feature_importance.csv", index=False
    )
    pd.DataFrame(
        {"row_index": np.arange(len(test_predictions)), "gbm_reg_pred_test": test_predictions}
    ).to_csv(EXPERIMENT_DIR / "xgb_reg_final_test_predictions.csv", index=False)

    duplicate_model = make_model(selected_config, selected_rounds, SCREEN_SEED, early_stop=False)
    duplicate_model.fit(x, y, verbose=False)
    duplicate_predictions = duplicate_model.predict(test[FEATURES])
    source_after = source_integrity_frame()
    tests = pd.DataFrame(
        [
            {"test": "prediction_count_200", "passed": len(test_predictions) == 200},
            {"test": "predictions_finite", "passed": bool(np.all(np.isfinite(test_predictions)))},
            {
                "test": "importance_has_12_features",
                "passed": set(final_importance) == set(FEATURES),
            },
            {
                "test": "importance_sums_to_one",
                "passed": abs(sum(final_importance.values()) - 1.0) < 1e-12,
            },
            {
                "test": "deterministic_predictions",
                "passed": bool(np.array_equal(test_predictions, duplicate_predictions)),
            },
            {
                "test": "source_hashes_unchanged",
                "passed": source_before["sha256"].tolist() == source_after["sha256"].tolist(),
            },
            {"test": "original_lotarea_maximum", "passed": float(x["LotArea"].max()) == 215245.0},
            {
                "test": "original_year_columns",
                "passed": "YearBuilt" in x and "YearRemodAdd" in x,
            },
        ]
    )
    tests.to_csv(EXPERIMENT_DIR / "xgb_reg_unit_tests.csv", index=False)
    if not bool(tests["passed"].all()):
        raise AssertionError(tests[~tests["passed"]].to_dict(orient="records"))

    result = {
        "gbm_reg_pred_test": test_predictions.tolist(),
        "gbm_reg_hyperparams": {
            "n_estimators": selected_rounds,
            "learning_rate": float(selected_config["learning_rate"]),
            "max_depth": int(selected_config["max_depth"]),
        },
        "gbm_reg_var_importance": final_importance,
    }
    write_json(EXPERIMENT_DIR / "xgb_task3_results.json", result)
    write_json(
        EXPERIMENT_DIR / "xgb_reg_selected_configuration.json",
        {
            **selected_config,
            "n_estimators": selected_rounds,
            "candidate_id": selected_id,
            "mean_validation_mse": float(winner["mean_mse"]),
            "validation_rmse": float(winner["rmse_from_mean_mse"]),
            "mdi_rank_stability": float(winner["mdi_rank_stability"]),
        },
    )
    save_plots(screen_summary, repeated_summary, final_model, x, y, selected_config)

    findings = f"""# XGBoost Regression Task 3 Findings

## Scope

This experiment implements only Task 3. It reads the untouched original train
and test CSV files, preserves the original year columns, and retains the
uncapped training `LotArea` maximum of 215245.

## Selection Method

- Library: XGBoost {package_versions()["xgboost"]}
- Objective: `reg:squarederror`
- Primary metric: validation MSE
- Initial screen: 60 deterministic configurations, five folds, seed 9618
- Robust comparison: {len(shortlist)} shortlisted configurations over six
  predeclared seeds and 30 validation folds per configuration
- Early stopping was performed on an inner split of each outer training fold.
  The outer validation fold never selected the number of boosting rounds.

## Selected Model

- Candidate: {selected_id}
- Max depth: {selected_config["max_depth"]}
- Learning rate: {selected_config["learning_rate"]}
- Number of estimators: {selected_rounds}
- Min child weight: {selected_config["min_child_weight"]}
- Gamma: {selected_config["gamma"]}
- Subsample: {selected_config["subsample"]}
- Column subsample: {selected_config["colsample_bytree"]}
- L2 regularization: {selected_config["reg_lambda"]}
- L1 regularization: {selected_config["reg_alpha"]}
- Repeated-CV mean MSE: {winner["mean_mse"]:.6f}
- Repeated-CV RMSE: {winner["rmse_from_mean_mse"]:.6f}
- MDI rank stability: {winner["mdi_rank_stability"]:.6f}

The final number of estimators is the median leakage-free best iteration from
the selected configuration's 30 validation fits. The model was then refit on
all 300 training rows without an evaluation set.

## Variable Importance

The submission candidate uses normalized XGBoost `total_gain`, which sums the
loss reduction attributable to each feature across all splits. Average gain,
split count, cover, and total cover are saved only as diagnostics.

## Verification

- Focused tests passed: {int(tests["passed"].sum())}/{len(tests)}
- Test predictions: {len(test_predictions)}
- Importance sum: {sum(final_importance.values()):.16f}
- Deterministic duplicate fit: {bool(tests.loc[tests["test"] == "deterministic_predictions", "passed"].iloc[0])}
- Source hashes unchanged: {bool(tests.loc[tests["test"] == "source_hashes_unchanged", "passed"].iloc[0])}

## Limitations And Research Notes

The sample contains only 300 observations, so deep boosted trees can overfit
despite regularization. Repeated validation and conservative tie-breaking were
used to prefer stable loss and importance rankings. XGBoost's current
documentation confirms that early stopping identifies a best iteration but
does not replace the need for an explicit full-data refit at a fixed round
count. Current 2026 literature continues to emphasize careful validation and
the distinction between predictive tree importance and causal importance.

Sources:

- https://xgboost.readthedocs.io/en/stable/python/python_api.html
- https://xgboost.readthedocs.io/en/stable/parameter.html
- https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- https://doi.org/10.3389/frai.2026.1752632
- https://arxiv.org/abs/2602.10524

No external implementation code was copied.
"""
    (EXPERIMENT_DIR / "XGB_REG_FINDINGS.md").write_text(findings, encoding="utf-8")
    print(json.dumps(result["gbm_reg_hyperparams"], indent=2))
    print(f"Selected repeated-CV MSE: {winner['mean_mse']:.6f}")


if __name__ == "__main__":
    main()
