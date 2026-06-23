from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REFERENCE_YEAR = 2011
SPLIT_TOLERANCE = 1e-12
OUTPUT_FEATURE_NAMES = {
    "years_since_built_2011": "YearBuilt",
    "years_since_remodel_2011": "YearRemodAdd",
}


@dataclass
class SplitResult:
    child_sse: float
    feature_index: int
    threshold: float
    left_indices: np.ndarray
    right_indices: np.ndarray
    sse_reduction: float


@dataclass
class TreeNode:
    prediction: float
    depth: int
    n_samples: int
    sse: float
    node_id: int = -1
    feature_index: int | None = None
    threshold: float | None = None
    sse_reduction: float = 0.0
    candidate_features: tuple[int, ...] = ()
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


def node_sse(y: np.ndarray) -> float:
    values = np.asarray(y, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    total = float(values.sum())
    total_sq = float(np.dot(values, values))
    return max(0.0, total_sq - total * total / len(values))


def split_is_better(
    child_sse: float,
    feature_index: int,
    threshold: float,
    current: SplitResult | None,
) -> bool:
    if current is None:
        return True
    if child_sse < current.child_sse - SPLIT_TOLERANCE:
        return True
    if abs(child_sse - current.child_sse) <= SPLIT_TOLERANCE:
        if feature_index < current.feature_index:
            return True
        if feature_index == current.feature_index and threshold < current.threshold:
            return True
    return False


def find_best_split_vectorized(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    min_samples_leaf: int,
    feature_indices: Iterable[int] | None = None,
) -> SplitResult | None:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2 * min_samples_leaf:
        return None

    parent_sse = node_sse(y[indices])
    if parent_sse <= SPLIT_TOLERANCE:
        return None

    if feature_indices is None:
        feature_indices = range(x.shape[1])

    best: SplitResult | None = None
    for feature_index in feature_indices:
        order = np.argsort(x[indices, feature_index], kind="mergesort")
        sorted_indices = indices[order]
        sorted_x = x[sorted_indices, feature_index]
        sorted_y = y[sorted_indices]
        n_samples = len(sorted_indices)

        candidate_positions = np.arange(
            min_samples_leaf - 1,
            n_samples - min_samples_leaf,
            dtype=int,
        )
        if len(candidate_positions) == 0:
            continue

        distinct_mask = (
            sorted_x[candidate_positions]
            < sorted_x[candidate_positions + 1]
        )
        candidate_positions = candidate_positions[distinct_mask]
        if len(candidate_positions) == 0:
            continue

        cumulative_y = np.cumsum(sorted_y, dtype=np.float64)
        cumulative_y_sq = np.cumsum(sorted_y * sorted_y, dtype=np.float64)
        total_y = cumulative_y[-1]
        total_y_sq = cumulative_y_sq[-1]

        n_left = candidate_positions + 1
        n_right = n_samples - n_left
        sum_left = cumulative_y[candidate_positions]
        sum_right = total_y - sum_left
        sum_sq_left = cumulative_y_sq[candidate_positions]
        sum_sq_right = total_y_sq - sum_sq_left

        left_sse = sum_sq_left - sum_left * sum_left / n_left
        right_sse = sum_sq_right - sum_right * sum_right / n_right
        child_sse_values = np.maximum(0.0, left_sse) + np.maximum(
            0.0, right_sse
        )
        local_offset = int(np.argmin(child_sse_values))
        split_position = int(candidate_positions[local_offset])
        child_sse = float(child_sse_values[local_offset])

        left_value = np.float64(sorted_x[split_position])
        right_value = np.float64(sorted_x[split_position + 1])
        threshold = float(left_value / 2.0 + right_value / 2.0)

        if split_is_better(
            child_sse,
            int(feature_index),
            threshold,
            best,
        ):
            best = SplitResult(
                child_sse=child_sse,
                feature_index=int(feature_index),
                threshold=threshold,
                left_indices=sorted_indices[: split_position + 1].copy(),
                right_indices=sorted_indices[split_position + 1 :].copy(),
                sse_reduction=max(0.0, parent_sse - child_sse),
            )

    if best is None or best.sse_reduction <= SPLIT_TOLERANCE:
        return None
    return best


def find_best_split_bruteforce(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    min_samples_leaf: int,
    feature_indices: Iterable[int] | None = None,
) -> SplitResult | None:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2 * min_samples_leaf:
        return None

    parent_sse = node_sse(y[indices])
    if parent_sse <= SPLIT_TOLERANCE:
        return None

    if feature_indices is None:
        feature_indices = range(x.shape[1])

    best: SplitResult | None = None
    for feature_index in feature_indices:
        unique_values = np.unique(x[indices, feature_index])
        if len(unique_values) < 2:
            continue
        thresholds = (
            unique_values[:-1].astype(np.float64)
            + unique_values[1:].astype(np.float64)
        ) / 2.0

        for threshold in thresholds:
            left_mask = x[indices, feature_index] <= threshold
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
            if (
                len(left_indices) < min_samples_leaf
                or len(right_indices) < min_samples_leaf
            ):
                continue

            child_sse = node_sse(y[left_indices]) + node_sse(y[right_indices])
            if split_is_better(
                child_sse,
                int(feature_index),
                float(threshold),
                best,
            ):
                best = SplitResult(
                    child_sse=float(child_sse),
                    feature_index=int(feature_index),
                    threshold=float(threshold),
                    left_indices=left_indices.copy(),
                    right_indices=right_indices.copy(),
                    sse_reduction=max(0.0, parent_sse - child_sse),
                )

    if best is None or best.sse_reduction <= SPLIT_TOLERANCE:
        return None
    return best


class ScratchCARTRegressor:
    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int = 5,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "ScratchCARTRegressor":
        x_array = np.asarray(x, dtype=np.float32, order="C")
        y_array = np.asarray(y, dtype=np.float64)
        if x_array.ndim != 2:
            raise ValueError("x must be a two-dimensional array")
        if y_array.ndim != 1 or len(y_array) != len(x_array):
            raise ValueError("y must be one-dimensional and match x")
        if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
            raise ValueError("x and y must contain only finite values")
        if self.max_features is not None:
            if not 1 <= self.max_features <= x_array.shape[1]:
                raise ValueError("max_features must be between 1 and n_features")

        self.x_ = x_array
        self.y_ = y_array
        self.n_features_in_ = x_array.shape[1]
        self.feature_names_ = (
            list(feature_names)
            if feature_names is not None
            else [f"x{j}" for j in range(self.n_features_in_)]
        )
        self.rng_ = np.random.default_rng(self.random_state)
        self.root_ = self._build(np.arange(len(y_array), dtype=int), depth=0)
        self._assign_node_ids()
        return self

    def _features_for_node(self) -> np.ndarray:
        if self.max_features is None:
            return np.arange(self.n_features_in_, dtype=int)
        selected = self.rng_.choice(
            self.n_features_in_,
            size=self.max_features,
            replace=False,
        )
        return np.sort(selected)

    def _build(self, indices: np.ndarray, depth: int) -> TreeNode:
        values = self.y_[indices]
        current_sse = node_sse(values)
        node = TreeNode(
            prediction=float(values.mean()),
            depth=depth,
            n_samples=len(indices),
            sse=current_sse,
        )

        if (
            depth >= self.max_depth
            or len(indices) < 2 * self.min_samples_leaf
            or current_sse <= SPLIT_TOLERANCE
        ):
            return node

        candidate_features = self._features_for_node()
        node.candidate_features = tuple(
            int(feature_index) for feature_index in candidate_features
        )
        split = find_best_split_vectorized(
            self.x_,
            self.y_,
            indices,
            self.min_samples_leaf,
            candidate_features,
        )
        if split is None:
            return node

        node.feature_index = split.feature_index
        node.threshold = split.threshold
        node.sse_reduction = split.sse_reduction
        node.left = self._build(split.left_indices, depth + 1)
        node.right = self._build(split.right_indices, depth + 1)
        return node

    def _assign_node_ids(self) -> None:
        next_id = 0

        def visit(node: TreeNode) -> None:
            nonlocal next_id
            node.node_id = next_id
            next_id += 1
            if not node.is_leaf:
                visit(node.left)
                visit(node.right)

        visit(self.root_)
        self.node_count_ = next_id

    def _predict_row(self, row: np.ndarray) -> tuple[float, int]:
        node = self.root_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction, node.node_id

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=np.float32, order="C")
        return np.asarray(
            [self._predict_row(row)[0] for row in x_array],
            dtype=np.float64,
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x, dtype=np.float32, order="C")
        return np.asarray(
            [self._predict_row(row)[1] for row in x_array],
            dtype=int,
        )

    def get_depth(self) -> int:
        return max(node.depth for node in self.iter_nodes())

    def get_n_leaves(self) -> int:
        return sum(node.is_leaf for node in self.iter_nodes())

    def iter_nodes(self) -> list[TreeNode]:
        nodes: list[TreeNode] = []

        def visit(node: TreeNode) -> None:
            nodes.append(node)
            if not node.is_leaf:
                visit(node.left)
                visit(node.right)

        visit(self.root_)
        return nodes

    def raw_feature_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for node in self.iter_nodes():
            if not node.is_leaf:
                importances[node.feature_index] += node.sse_reduction
        return importances

    def normalized_feature_importances(self) -> np.ndarray:
        importances = self.raw_feature_importances()
        total = float(importances.sum())
        if total <= 0.0:
            return importances
        return importances / total

    def feature_subset_sizes(self) -> np.ndarray:
        return np.asarray(
            [
                len(node.candidate_features)
                for node in self.iter_nodes()
                if node.candidate_features
            ],
            dtype=int,
        )

    def structure_frame(self, model_name: str) -> pd.DataFrame:
        rows = []
        for node in self.iter_nodes():
            transformed_name = (
                self.feature_names_[node.feature_index]
                if not node.is_leaf
                else ""
            )
            original_name = OUTPUT_FEATURE_NAMES.get(
                transformed_name, transformed_name
            )
            transformed_operator = "<=" if not node.is_leaf else ""
            original_operator = transformed_operator
            original_threshold = node.threshold
            if transformed_name in OUTPUT_FEATURE_NAMES:
                original_operator = ">="
                original_threshold = REFERENCE_YEAR - node.threshold
            transformed_description = (
                f"{transformed_name} <= {node.threshold:.12g}"
                if not node.is_leaf
                else ""
            )
            original_description = (
                f"{original_name} {original_operator} "
                f"{original_threshold:.12g}"
                if not node.is_leaf
                else ""
            )
            rows.append(
                {
                    "model": model_name,
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "is_leaf": node.is_leaf,
                    "n_samples": node.n_samples,
                    "prediction": node.prediction,
                    "sse": node.sse,
                    "feature_transformed": transformed_name,
                    "feature_original": original_name,
                    "threshold": node.threshold,
                    "operator_transformed": transformed_operator,
                    "threshold_original_scale": original_threshold,
                    "operator_original_scale": original_operator,
                    "split_description_transformed": transformed_description,
                    "split_description_original": original_description,
                    "sse_reduction": node.sse_reduction,
                    "left_node_id": (
                        node.left.node_id if node.left is not None else None
                    ),
                    "right_node_id": (
                        node.right.node_id if node.right is not None else None
                    ),
                }
            )
        return pd.DataFrame(rows)
