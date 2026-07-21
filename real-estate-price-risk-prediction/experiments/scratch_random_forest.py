from dataclasses import dataclass

import numpy as np
import pandas as pd

from scratch_cart import ScratchCARTRegressor


@dataclass
class ForestCheckpoint:
    n_trees: int
    oob_mse: float
    oob_rmse: float
    oob_coverage: float
    min_oob_count: int
    mean_oob_count: float
    max_oob_count: int
    mdi: np.ndarray
    raw_sse_share: np.ndarray
    evaluation_prediction: np.ndarray | None


class ScratchRandomForestRegressor:
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        max_features: int = 4,
        random_state: int = 9618,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if max_features < 1:
            raise ValueError("max_features must be at least 1")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = int(max_features)
        self.random_state = int(random_state)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        checkpoints: tuple[int, ...] = (),
        evaluation_x: np.ndarray | None = None,
    ) -> "ScratchRandomForestRegressor":
        x_array = np.asarray(x, dtype=np.float32, order="C")
        y_array = np.asarray(y, dtype=np.float64)
        if x_array.ndim != 2:
            raise ValueError("x must be a two-dimensional array")
        if y_array.ndim != 1 or len(y_array) != len(x_array):
            raise ValueError("y must be one-dimensional and match x")
        if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
            raise ValueError("x and y must contain only finite values")
        if self.max_features > x_array.shape[1]:
            raise ValueError("max_features cannot exceed n_features")

        evaluation_array = None
        if evaluation_x is not None:
            evaluation_array = np.asarray(
                evaluation_x, dtype=np.float32, order="C"
            )
            if (
                evaluation_array.ndim != 2
                or evaluation_array.shape[1] != x_array.shape[1]
            ):
                raise ValueError(
                    "evaluation_x must have the same number of features as x"
                )

        checkpoint_set = set(int(value) for value in checkpoints)
        if any(value < 1 or value > self.n_estimators for value in checkpoint_set):
            raise ValueError("checkpoints must lie between 1 and n_estimators")

        self.x_ = x_array
        self.y_ = y_array
        self.n_features_in_ = x_array.shape[1]
        self.feature_names_ = (
            list(feature_names)
            if feature_names is not None
            else [f"x{j}" for j in range(self.n_features_in_)]
        )
        self.estimators_: list[ScratchCARTRegressor] = []
        self.bootstrap_indices_: list[np.ndarray] = []
        self.oob_indices_: list[np.ndarray] = []
        self.tree_seeds_: list[int] = []
        self.checkpoints_: list[ForestCheckpoint] = []

        n_samples = len(y_array)
        oob_sum = np.zeros(n_samples, dtype=np.float64)
        oob_count = np.zeros(n_samples, dtype=np.int64)
        evaluation_sum = (
            np.zeros(len(evaluation_array), dtype=np.float64)
            if evaluation_array is not None
            else None
        )
        mdi_sum = np.zeros(self.n_features_in_, dtype=np.float64)
        raw_sse_sum = np.zeros(self.n_features_in_, dtype=np.float64)
        non_stump_count = 0

        seed_sequence = np.random.SeedSequence(self.random_state)
        for tree_number, child_sequence in enumerate(
            seed_sequence.spawn(self.n_estimators), start=1
        ):
            rng = np.random.default_rng(child_sequence)
            bootstrap_indices = rng.integers(
                0,
                n_samples,
                size=n_samples,
                dtype=np.int64,
            )
            inbag_counts = np.bincount(
                bootstrap_indices, minlength=n_samples
            )
            oob_indices = np.flatnonzero(inbag_counts == 0)
            tree_seed = int(
                rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64)
            )

            tree = ScratchCARTRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            ).fit(
                x_array[bootstrap_indices],
                y_array[bootstrap_indices],
                self.feature_names_,
            )

            self.estimators_.append(tree)
            self.bootstrap_indices_.append(bootstrap_indices)
            self.oob_indices_.append(oob_indices)
            self.tree_seeds_.append(tree_seed)

            if len(oob_indices):
                oob_sum[oob_indices] += tree.predict(x_array[oob_indices])
                oob_count[oob_indices] += 1
            if evaluation_sum is not None:
                evaluation_sum += tree.predict(evaluation_array)

            tree_raw_importance = tree.raw_feature_importances()
            raw_sse_sum += tree_raw_importance
            tree_total = float(tree_raw_importance.sum())
            if tree_total > 0.0:
                mdi_sum += tree_raw_importance / tree_total
                non_stump_count += 1

            if tree_number in checkpoint_set:
                self.checkpoints_.append(
                    self._make_checkpoint(
                        tree_number=tree_number,
                        oob_sum=oob_sum,
                        oob_count=oob_count,
                        mdi_sum=mdi_sum,
                        non_stump_count=non_stump_count,
                        raw_sse_sum=raw_sse_sum,
                        evaluation_sum=evaluation_sum,
                    )
                )

        self.oob_counts_ = oob_count
        self.oob_prediction_ = np.full(n_samples, np.nan, dtype=np.float64)
        covered = oob_count > 0
        self.oob_prediction_[covered] = oob_sum[covered] / oob_count[covered]
        self.oob_mse_ = float(
            np.mean(
                (
                    y_array[covered]
                    - self.oob_prediction_[covered]
                )
                ** 2
            )
        )
        self.oob_rmse_ = float(np.sqrt(self.oob_mse_))
        self.oob_coverage_ = float(np.mean(covered))
        self.raw_sse_importances_ = raw_sse_sum
        self.raw_sse_share_ = self._normalize(raw_sse_sum)
        self.feature_importances_ = (
            self._normalize(mdi_sum / non_stump_count)
            if non_stump_count
            else np.zeros(self.n_features_in_, dtype=np.float64)
        )
        self.non_stump_count_ = non_stump_count
        return self

    def _make_checkpoint(
        self,
        tree_number: int,
        oob_sum: np.ndarray,
        oob_count: np.ndarray,
        mdi_sum: np.ndarray,
        non_stump_count: int,
        raw_sse_sum: np.ndarray,
        evaluation_sum: np.ndarray | None,
    ) -> ForestCheckpoint:
        covered = oob_count > 0
        oob_prediction = oob_sum[covered] / oob_count[covered]
        oob_mse = float(
            np.mean((self.y_[covered] - oob_prediction) ** 2)
        )
        mdi = (
            self._normalize(mdi_sum / non_stump_count)
            if non_stump_count
            else np.zeros(self.n_features_in_, dtype=np.float64)
        )
        return ForestCheckpoint(
            n_trees=tree_number,
            oob_mse=oob_mse,
            oob_rmse=float(np.sqrt(oob_mse)),
            oob_coverage=float(np.mean(covered)),
            min_oob_count=int(oob_count.min()),
            mean_oob_count=float(oob_count.mean()),
            max_oob_count=int(oob_count.max()),
            mdi=mdi.copy(),
            raw_sse_share=self._normalize(raw_sse_sum),
            evaluation_prediction=(
                evaluation_sum.copy() / tree_number
                if evaluation_sum is not None
                else None
            ),
        )

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        total = float(array.sum())
        if total <= 0.0:
            return np.zeros_like(array)
        return array / total

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("The forest must be fitted before prediction")
        x_array = np.asarray(x, dtype=np.float32, order="C")
        prediction_sum = np.zeros(len(x_array), dtype=np.float64)
        for tree in self.estimators_:
            prediction_sum += tree.predict(x_array)
        return prediction_sum / len(self.estimators_)

    def tree_diagnostics_frame(self) -> pd.DataFrame:
        rows = []
        for tree_number, (
            tree,
            bootstrap_indices,
            oob_indices,
            tree_seed,
        ) in enumerate(
            zip(
                self.estimators_,
                self.bootstrap_indices_,
                self.oob_indices_,
                self.tree_seeds_,
            ),
            start=1,
        ):
            subset_sizes = tree.feature_subset_sizes()
            rows.append(
                {
                    "tree_number": tree_number,
                    "tree_seed": tree_seed,
                    "bootstrap_size": len(bootstrap_indices),
                    "unique_inbag_rows": len(np.unique(bootstrap_indices)),
                    "duplicate_draws": (
                        len(bootstrap_indices)
                        - len(np.unique(bootstrap_indices))
                    ),
                    "oob_rows": len(oob_indices),
                    "tree_depth": tree.get_depth(),
                    "node_count": tree.node_count_,
                    "leaf_count": tree.get_n_leaves(),
                    "candidate_nodes": len(subset_sizes),
                    "min_candidate_features": (
                        int(subset_sizes.min()) if len(subset_sizes) else 0
                    ),
                    "max_candidate_features": (
                        int(subset_sizes.max()) if len(subset_sizes) else 0
                    ),
                }
            )
        return pd.DataFrame(rows)

    def checkpoint_frame(self) -> pd.DataFrame:
        rows = []
        for checkpoint in self.checkpoints_:
            row = {
                "n_trees": checkpoint.n_trees,
                "oob_mse": checkpoint.oob_mse,
                "oob_rmse": checkpoint.oob_rmse,
                "oob_coverage": checkpoint.oob_coverage,
                "min_oob_count": checkpoint.min_oob_count,
                "mean_oob_count": checkpoint.mean_oob_count,
                "max_oob_count": checkpoint.max_oob_count,
            }
            row.update(
                {
                    f"mdi_{feature_name}": float(value)
                    for feature_name, value in zip(
                        self.feature_names_, checkpoint.mdi
                    )
                }
            )
            row.update(
                {
                    f"raw_sse_share_{feature_name}": float(value)
                    for feature_name, value in zip(
                        self.feature_names_, checkpoint.raw_sse_share
                    )
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)
