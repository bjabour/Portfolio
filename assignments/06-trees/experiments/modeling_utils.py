from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Iterable

import imblearn
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = ASSIGNMENT_DIR / "trees_train.csv"
TEST_PATH = ASSIGNMENT_DIR / "trees_test.csv"
STUDENT_ID = ""
SEEDS = (0, 42, 2026, 9618, 19618, 31415)
REGRESSION_TARGET = "sale_price_keur"
CLASSIFICATION_TARGET = "distressed"
FEATURES = [
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
LOG_COLUMNS = [
    "LotArea",
    "TotalBsmtSF",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
]
STANDARD_COLUMNS = [feature for feature in FEATURES if feature not in LOG_COLUMNS]


def load_original_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    expected_train = FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
    if train.columns.tolist() != expected_train:
        raise AssertionError(f"Unexpected training columns: {train.columns.tolist()}")
    if test.columns.tolist() != FEATURES:
        raise AssertionError(f"Unexpected test columns: {test.columns.tolist()}")
    if train.shape != (300, 14) or test.shape != (200, 12):
        raise AssertionError(f"Unexpected shapes: train={train.shape}, test={test.shape}")
    if float(train["LotArea"].max()) != 215245.0:
        raise AssertionError("The original, uncapped LotArea maximum is not present.")
    if int(train[CLASSIFICATION_TARGET].sum()) != 47:
        raise AssertionError("Unexpected distressed count.")
    return train, test


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_integrity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file": TRAIN_PATH.name,
                "path": str(TRAIN_PATH),
                "sha256": sha256(TRAIN_PATH),
                "bytes": TRAIN_PATH.stat().st_size,
            },
            {
                "file": TEST_PATH.name,
                "path": str(TEST_PATH),
                "sha256": sha256(TEST_PATH),
                "bytes": TEST_PATH.stat().st_size,
            },
        ]
    )


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "imbalanced_learn": imblearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized_xgb_importance(
    model: Any,
    importance_type: str = "total_gain",
) -> dict[str, float]:
    raw = model.get_booster().get_score(importance_type=importance_type)
    values = np.asarray([float(raw.get(feature, 0.0)) for feature in FEATURES])
    total = float(values.sum())
    if total > 0:
        values /= total
    return dict(zip(FEATURES, values.tolist(), strict=True))


def normalized_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(list(values), dtype=float)
    vector[~np.isfinite(vector)] = 0.0
    vector = np.maximum(vector, 0.0)
    total = float(vector.sum())
    return vector / total if total > 0 else vector


def pairwise_spearman(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 1.0
    correlations: list[float] = []
    for left_index in range(len(vectors)):
        left_rank = pd.Series(vectors[left_index]).rank(method="average").to_numpy()
        for right_index in range(left_index + 1, len(vectors)):
            right_rank = pd.Series(vectors[right_index]).rank(method="average").to_numpy()
            correlation = np.corrcoef(left_rank, right_rank)[0, 1]
            correlations.append(0.0 if not np.isfinite(correlation) else float(correlation))
    return float(np.mean(correlations))


def top_k_overlap(vectors: list[np.ndarray], k: int = 5) -> float:
    if len(vectors) < 2:
        return 1.0
    sets = [set(np.argsort(vector)[-k:]) for vector in vectors]
    overlaps: list[float] = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            overlaps.append(len(sets[left_index] & sets[right_index]) / k)
    return float(np.mean(overlaps))


def rmse_from_mse(mse: float) -> float:
    return math.sqrt(float(mse))


def build_smote_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "log_standard",
                Pipeline(
                    [
                        (
                            "log1p",
                            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                        ),
                        ("standard_scale", StandardScaler()),
                    ]
                ),
                LOG_COLUMNS,
            ),
            ("standard_scale", StandardScaler(), STANDARD_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_feature_order() -> list[str]:
    return LOG_COLUMNS + STANDARD_COLUMNS


def probabilities_are_valid(probabilities: np.ndarray) -> bool:
    values = np.asarray(probabilities, dtype=float)
    return bool(
        values.ndim == 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
    )


def clipped_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0 - 1e-15)


def config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def deterministic_sample(
    candidates: list[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if size > len(candidates):
        raise ValueError("Requested sample is larger than the candidate collection.")
    rng = np.random.default_rng(seed)
    indexes = rng.choice(len(candidates), size=size, replace=False)
    return [candidates[int(index)] for index in indexes]
