from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from run_deep_type2_experiments import (
    as_binary,
    build_feature_matrix_batched,
    dementia_metrics,
    fit_transform,
    make_mlp,
    positive_scores_from_two_stage,
)
from run_plan_d_solution import (
    CLASS_NAMES,
    SPLIT_DIRS,
    dataframe_to_markdown,
    list_image_rows,
    metric_row,
    package_versions,
    save_confusion_matrix,
)


OUTPUT_DIR = Path("results") / "experiments" / "type2_constraints"
CONSTRAINTS = {
    "no_type1_limit": None,
    "type1_le_0_05": 0.05,
    "type1_le_0_025": 0.025,
}


@dataclass
class ModelScores:
    name: str
    val_scores: np.ndarray
    val_severity: np.ndarray
    test_scores: np.ndarray
    test_severity: np.ndarray
    payload: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Type II-focused MLP/CNN models under Type I error caps."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mlp-max-iter", type=int, default=120)
    parser.add_argument("--cnn-epochs", type=int, default=10)
    parser.add_argument("--cnn-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores.astype(np.float64), 0.0, 1.0)
    eps = np.finfo(np.float64).eps * 128
    return np.unique(np.clip(np.concatenate([[0.0, 0.5, 1.0], scores, scores - eps, scores + eps]), 0.0, 1.0))


def apply_threshold(scores: np.ndarray, severity: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores >= threshold, severity, 0).astype(int)


def select_threshold_for_cap(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    severity_val: np.ndarray,
    type1_cap: float | None,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in threshold_grid(scores_val):
        pred = apply_threshold(scores_val, severity_val, float(threshold))
        row: dict[str, float | int | str | bool] = {"threshold": float(threshold)}
        row.update(metric_row(y_val, pred, "val"))
        row.update(dementia_metrics(y_val, pred, "val"))
        rows.append(row)
    sweep = pd.DataFrame(rows)
    eligible = sweep if type1_cap is None else sweep[sweep["val_type1_error"] <= type1_cap]
    if eligible.empty:
        eligible = sweep.iloc[[int(sweep["val_type1_error"].argmin())]]
    selected = eligible.sort_values(
        ["val_type2_error", "val_macro_f1", "val_balanced_accuracy", "val_specificity"],
        ascending=[True, False, False, False],
    ).iloc[0]
    sweep["selected"] = False
    sweep.loc[selected.name, "selected"] = True
    cols = ["threshold", "selected"] + [c for c in sweep.columns if c not in {"threshold", "selected"}]
    return float(selected["threshold"]), sweep[cols].sort_values("threshold")


def fit_two_stage_mlp(
    project_root: Path,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    train_domain: np.ndarray,
    val_domain: np.ndarray,
    test_domain: np.ndarray,
    seed: int,
    max_iter: int,
) -> ModelScores:
    domain = fit_transform("domain_pca", train_domain, val_domain, test_domain, 120, seed)

    binary_model = make_mlp(seed + 20, (128, 64), alpha=7e-4, learning_rate_init=8e-4, max_iter=max_iter)
    y_train_binary = as_binary(y_train)
    binary_weights = compute_sample_weight(class_weight="balanced", y=y_train_binary).astype(np.float64)
    binary_weights[y_train_binary == 1] *= 3.0
    binary_model.fit(domain.z_train, y_train_binary, sample_weight=binary_weights)

    severity_model = make_mlp(seed + 21, (96, 48), alpha=1e-3, learning_rate_init=8e-4, max_iter=max_iter)
    dementia_train = y_train != 0
    severity_model.fit(
        domain.z_train[dementia_train],
        y_train[dementia_train],
        sample_weight=compute_sample_weight(class_weight="balanced", y=y_train[dementia_train]),
    )

    val_scores, val_severity, _ = positive_scores_from_two_stage(binary_model, severity_model, domain.z_val)
    test_scores, test_severity, _ = positive_scores_from_two_stage(binary_model, severity_model, domain.z_test)
    return ModelScores(
        "two_stage_mlp",
        val_scores,
        val_severity,
        test_scores,
        test_severity,
        {
            "kind": "two_stage_mlp",
            "feature_bundle": domain,
            "binary_model": binary_model,
            "severity_model": severity_model,
        },
    )


def load_rows_and_domain_features(project_root: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    from run_plan_d_solution import load_split

    rows, images = load_split(project_root, split)
    domain_features, _ = build_feature_matrix_batched(images)
    del images
    return rows, domain_features


class MRIDataset:
    def __init__(self, rows: pd.DataFrame, size: int):
        self.paths = rows["path"].tolist()
        self.labels = rows["label"].to_numpy(dtype=np.int64)
        self.size = size
        self.resample = getattr(Image, "Resampling", Image).BILINEAR

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        import torch

        with Image.open(self.paths[index]) as img:
            img = img.convert("L").resize((self.size, self.size), self.resample)
            array = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array[None, :, :]), int(self.labels[index])


def make_cnn(size: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Dropout(0.25),
        nn.Linear(64 * 4 * 4, 96),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(96, 3),
    )


def predict_cnn(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    proba = np.vstack(probs)
    scores = proba[:, 1] + proba[:, 2]
    severity = np.where(proba[:, 2] >= proba[:, 1], 2, 1).astype(int)
    if not np.isfinite(proba).all():
        raise ValueError("CNN produced non-finite probabilities.")
    return scores, severity, proba


class NumpySmallCNN:
    def __init__(self, image_size: int, seed: int, filters: int = 8, hidden: int = 64):
        self.image_size = image_size
        self.filters = filters
        self.hidden = hidden
        self.kernel_size = 5
        self.pool = 4
        pooled = image_size // self.pool
        rng = np.random.default_rng(seed)
        self.params = {
            "wc": rng.normal(0.0, 0.08, size=(filters, self.kernel_size, self.kernel_size)).astype(np.float32),
            "bc": np.zeros(filters, dtype=np.float32),
            "w1": rng.normal(0.0, 0.04, size=(filters * pooled * pooled, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "w2": rng.normal(0.0, 0.04, size=(hidden, len(CLASS_NAMES))).astype(np.float32),
            "b2": np.zeros(len(CLASS_NAMES), dtype=np.float32),
        }
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.step_index = 0

    def _conv_windows(self, x: np.ndarray) -> np.ndarray:
        pad = self.kernel_size // 2
        padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
        return np.lib.stride_tricks.sliding_window_view(
            padded, (self.kernel_size, self.kernel_size), axis=(1, 2)
        )

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        windows = self._conv_windows(x)
        conv = np.einsum("bhwkl,fkl->bfhw", windows, self.params["wc"], optimize=True)
        conv += self.params["bc"][None, :, None, None]
        act = np.maximum(conv, 0.0)
        pooled_side = self.image_size // self.pool
        pooled = act.reshape(x.shape[0], self.filters, pooled_side, self.pool, pooled_side, self.pool).mean(axis=(3, 5))
        flat = pooled.reshape(x.shape[0], -1)
        hidden_pre = flat @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(hidden_pre, 0.0)
        logits = hidden @ self.params["w2"] + self.params["b2"]
        cache = {
            "x": x,
            "windows": windows,
            "conv": conv,
            "pooled": pooled,
            "flat": flat,
            "hidden_pre": hidden_pre,
            "hidden": hidden,
        }
        return logits, cache

    def predict_proba(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        probs = []
        for start in range(0, x.shape[0], batch_size):
            logits, _ = self.forward(x[start : start + batch_size])
            probs.append(softmax(logits))
        return np.vstack(probs)

    def train_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_weights: np.ndarray,
        learning_rate: float,
        weight_decay: float,
    ) -> float:
        logits, cache = self.forward(x)
        probs = softmax(logits)
        sample_weights = class_weights[y].astype(np.float32)
        normalizer = float(sample_weights.sum())
        loss = -np.sum(sample_weights * np.log(probs[np.arange(y.shape[0]), y] + 1e-9)) / normalizer

        y_one_hot = np.zeros_like(probs)
        y_one_hot[np.arange(y.shape[0]), y] = 1.0
        dlogits = (probs - y_one_hot) * (sample_weights[:, None] / normalizer)

        grads = {}
        grads["w2"] = cache["hidden"].T @ dlogits + weight_decay * self.params["w2"]
        grads["b2"] = dlogits.sum(axis=0)
        dhidden = dlogits @ self.params["w2"].T
        dhidden_pre = dhidden * (cache["hidden_pre"] > 0.0)
        grads["w1"] = cache["flat"].T @ dhidden_pre + weight_decay * self.params["w1"]
        grads["b1"] = dhidden_pre.sum(axis=0)
        dflat = dhidden_pre @ self.params["w1"].T
        pooled_side = self.image_size // self.pool
        dpooled = dflat.reshape(x.shape[0], self.filters, pooled_side, pooled_side)
        dact = np.repeat(np.repeat(dpooled[:, :, :, None, :, None] / (self.pool * self.pool), self.pool, axis=3), self.pool, axis=5)
        dact = dact.reshape(x.shape[0], self.filters, self.image_size, self.image_size)
        dconv = dact * (cache["conv"] > 0.0)
        grads["wc"] = np.einsum("bfhw,bhwkl->fkl", dconv, cache["windows"], optimize=True) + weight_decay * self.params["wc"]
        grads["bc"] = dconv.sum(axis=(0, 2, 3))

        self.adam_step(grads, learning_rate)
        return float(loss)

    def adam_step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        self.step_index += 1
        beta1 = 0.9
        beta2 = 0.999
        for key, grad in grads.items():
            grad = np.clip(grad, -5.0, 5.0).astype(np.float32)
            self.m[key] = beta1 * self.m[key] + (1.0 - beta1) * grad
            self.v[key] = beta2 * self.v[key] + (1.0 - beta2) * (grad * grad)
            m_hat = self.m[key] / (1.0 - beta1**self.step_index)
            v_hat = self.v[key] / (1.0 - beta2**self.step_index)
            self.params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

    def state_copy(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self.params.items()}

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        self.params = {key: value.copy() for key, value in state.items()}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_cnn_arrays(rows: pd.DataFrame, size: int, mean: float | None = None, std: float | None = None):
    resample = getattr(Image, "Resampling", Image).BILINEAR
    arrays = np.empty((len(rows), size, size), dtype=np.float32)
    for index, path in enumerate(rows["path"]):
        with Image.open(path) as img:
            img = img.convert("L").resize((size, size), resample)
            arrays[index] = np.asarray(img, dtype=np.float32) / 255.0
    if mean is None:
        mean = float(arrays.mean())
    if std is None:
        std = float(arrays.std() + 1e-6)
    arrays = (arrays - mean) / std
    return arrays.astype(np.float32), mean, std


def fit_numpy_cnn(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    seed: int,
    epochs: int,
    size: int,
    batch_size: int,
    backend_error: str,
) -> ModelScores:
    size = min(size, 32)
    x_train, mean, std = load_cnn_arrays(train_rows, size)
    x_val, _, _ = load_cnn_arrays(val_rows, size, mean, std)
    x_test, _, _ = load_cnn_arrays(test_rows, size, mean, std)
    y_train = train_rows["label"].to_numpy(dtype=np.int64)
    y_val = val_rows["label"].to_numpy(dtype=np.int64)

    class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
    class_weights = class_counts.sum() / (len(CLASS_NAMES) * np.maximum(class_counts, 1.0))
    class_weights[1:] *= 2.5

    model = NumpySmallCNN(size, seed)
    rng = np.random.default_rng(seed)
    history = []
    best_state = model.state_copy()
    best_val_type2 = float("inf")
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        order = rng.permutation(y_train.shape[0])
        losses = []
        for start in range(0, order.shape[0], batch_size):
            idx = order[start : start + batch_size]
            losses.append(model.train_batch(x_train[idx], y_train[idx], class_weights, 8e-4, 1e-4))
        val_probs = model.predict_proba(x_val)
        val_scores = val_probs[:, 1] + val_probs[:, 2]
        val_severity = np.where(val_probs[:, 2] >= val_probs[:, 1], 2, 1).astype(int)
        val_pred = apply_threshold(val_scores, val_severity, 0.5)
        val_stats = dementia_metrics(y_val, val_pred, "val")
        val_loss = -np.mean(np.log(val_probs[np.arange(y_val.shape[0]), y_val] + 1e-9))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val_loss),
            "val_type2_error_at_0_5": val_stats["val_type2_error"],
            "val_type1_error_at_0_5": val_stats["val_type1_error"],
        }
        history.append(row)
        if (
            float(row["val_type2_error_at_0_5"]) < best_val_type2
            or (
                float(row["val_type2_error_at_0_5"]) == best_val_type2
                and float(row["val_loss"]) < best_val_loss
            )
        ):
            best_val_type2 = float(row["val_type2_error_at_0_5"])
            best_val_loss = float(row["val_loss"])
            best_state = model.state_copy()

    model.load_state(best_state)
    val_proba = model.predict_proba(x_val)
    test_proba = model.predict_proba(x_test)
    val_scores = val_proba[:, 1] + val_proba[:, 2]
    test_scores = test_proba[:, 1] + test_proba[:, 2]
    val_severity = np.where(val_proba[:, 2] >= val_proba[:, 1], 2, 1).astype(int)
    test_severity = np.where(test_proba[:, 2] >= test_proba[:, 1], 2, 1).astype(int)
    return ModelScores(
        "cnn_numpy_small_weighted",
        val_scores,
        val_severity,
        test_scores,
        test_severity,
        {
            "kind": "cnn_numpy_small_weighted",
            "backend": "numpy",
            "torch_error": backend_error,
            "image_size": size,
            "epochs": epochs,
            "batch_size": batch_size,
            "class_weights": class_weights.tolist(),
            "mean": mean,
            "std": std,
            "history": history,
            "params": best_state,
        },
    )


def fit_cnn(
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    seed: int,
    epochs: int,
    size: int,
    batch_size: int,
) -> ModelScores | None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as exc:
        return fit_numpy_cnn(train_rows, val_rows, test_rows, seed, epochs, size, batch_size, str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        MRIDataset(train_rows, size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(MRIDataset(val_rows, size), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(MRIDataset(test_rows, size), batch_size=batch_size, shuffle=False, num_workers=0)

    model = make_cnn(size).to(device)
    y_train = train_rows["label"].to_numpy(dtype=int)
    class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
    class_weights = class_counts.sum() / (len(CLASS_NAMES) * np.maximum(class_counts, 1.0))
    class_weights[1:] *= 2.5
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    best_state = None
    best_val_type2 = float("inf")
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(labels.shape[0])
            seen += int(labels.shape[0])

        val_loss = 0.0
        val_seen = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                loss = criterion(model(images), labels)
                val_loss += float(loss.item()) * int(labels.shape[0])
                val_seen += int(labels.shape[0])
        val_scores, val_severity, _ = predict_cnn(model, val_loader, device)
        val_pred = apply_threshold(val_scores, val_severity, 0.5)
        val_stats = dementia_metrics(val_rows["label"].to_numpy(dtype=int), val_pred, "val")
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss / max(seen, 1),
            "val_loss": val_loss / max(val_seen, 1),
            "val_type2_error_at_0_5": val_stats["val_type2_error"],
            "val_type1_error_at_0_5": val_stats["val_type1_error"],
        }
        history.append(epoch_row)
        if (
            float(epoch_row["val_type2_error_at_0_5"]) < best_val_type2
            or (
                float(epoch_row["val_type2_error_at_0_5"]) == best_val_type2
                and float(epoch_row["val_loss"]) < best_val_loss
            )
        ):
            best_val_type2 = float(epoch_row["val_type2_error_at_0_5"])
            best_val_loss = float(epoch_row["val_loss"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores, val_severity, val_proba = predict_cnn(model, val_loader, device)
    test_scores, test_severity, test_proba = predict_cnn(model, test_loader, device)
    return ModelScores(
        "cnn_small_weighted",
        val_scores,
        val_severity,
        test_scores,
        test_severity,
        {
            "kind": "cnn_small_weighted",
            "model_state_dict": best_state,
            "image_size": size,
            "epochs": epochs,
            "batch_size": batch_size,
            "class_weights": class_weights.tolist(),
            "history": history,
            "torch_device": str(device),
            "val_proba": val_proba,
            "test_proba": test_proba,
        },
    )


def write_constraint_plot(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.copy()
    labels = {
        "no_type1_limit": "no cap",
        "type1_le_0_05": "<= 0.05",
        "type1_le_0_025": "<= 0.025",
    }
    plot_data["constraint_label"] = plot_data["constraint"].map(labels)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="constraint_label", y="test_type2_error", hue="model")
    plt.ylabel("Test Type II error")
    plt.xlabel("Validation Type I cap")
    plt.title("Type II Error Under Type I Constraints")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_outputs(
    output_dir: Path,
    model_scores: list[ModelScores],
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    seed: int,
    elapsed_seconds: float,
    cnn_available: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for scores in model_scores:
        for constraint_name, cap in CONSTRAINTS.items():
            threshold, sweep = select_threshold_for_cap(y_val, scores.val_scores, scores.val_severity, cap)
            val_pred = apply_threshold(scores.val_scores, scores.val_severity, threshold)
            test_pred = apply_threshold(scores.test_scores, scores.test_severity, threshold)
            row: dict[str, float | int | str | bool | None] = {
                "model": scores.name,
                "constraint": constraint_name,
                "type1_cap": cap,
                "threshold": threshold,
            }
            row.update(metric_row(y_val, val_pred, "val"))
            row.update(dementia_metrics(y_val, val_pred, "val"))
            row.update(metric_row(y_test, test_pred, "test"))
            row.update(dementia_metrics(y_test, test_pred, "test"))
            summary_rows.append(row)

            stem = f"{scores.name}_{constraint_name}"
            sweep.to_csv(output_dir / f"threshold_sweep_{stem}.csv", index=False)
            save_confusion_matrix(
                y_val,
                val_pred,
                output_dir / f"val_confusion_matrix_{stem}.csv",
                output_dir / f"val_confusion_matrix_{stem}.png",
                f"Validation Confusion Matrix: {stem}",
            )
            save_confusion_matrix(
                y_test,
                test_pred,
                output_dir / f"test_confusion_matrix_{stem}.csv",
                output_dir / f"test_confusion_matrix_{stem}.png",
                f"Test Confusion Matrix: {stem}",
            )
            val_out = val_rows[["split", "relative_path", "filename", "actual", "label"]].copy()
            val_out["predicted"] = [CLASS_NAMES[int(label)] for label in val_pred]
            val_out["predicted_label"] = val_pred
            val_out["dementia_score"] = scores.val_scores
            val_out["threshold"] = threshold
            val_out.to_csv(output_dir / f"predictions_val_{stem}.csv", index=False)
            test_out = test_rows[["split", "relative_path", "filename", "actual", "label"]].copy()
            test_out["predicted"] = [CLASS_NAMES[int(label)] for label in test_pred]
            test_out["predicted_label"] = test_pred
            test_out["dementia_score"] = scores.test_scores
            test_out["threshold"] = threshold
            test_out.to_csv(output_dir / f"predictions_test_{stem}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "constraint_comparison.csv", index=False)
    write_constraint_plot(summary, output_dir / "type2_constraint_comparison.png")
    joblib.dump({scores.name: scores.payload for scores in model_scores}, output_dir / "models.joblib")

    best_by_constraint = summary.sort_values(
        ["constraint", "val_type2_error", "val_macro_f1", "val_specificity"],
        ascending=[True, True, False, False],
    ).groupby("constraint", as_index=False).head(1)

    manifest = {
        "seed": seed,
        "elapsed_seconds": elapsed_seconds,
        "constraints": CONSTRAINTS,
        "selection_rule": "Within each model and constraint: choose validation threshold with minimum validation Type II error; ties prefer macro-F1, balanced accuracy, then specificity.",
        "positive_definition": "dementia = Very_Mild_Demented or Mild_Demented",
        "type1_definition": "actual Non_Demented predicted as dementia",
        "type2_definition": "actual dementia predicted as Non_Demented",
        "cnn_available": cnn_available,
        "packages": package_versions(),
    }
    (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")

    report = [
        "# Type II Constraint Experiment",
        "",
        "Models: `two_stage_mlp` and `cnn_small_weighted` when PyTorch is available.",
        "",
        "A threshold is selected on validation for each Type I constraint. Test is only reported after that threshold is fixed.",
        "",
        "## Best Model Per Constraint",
        "",
        dataframe_to_markdown(best_by_constraint),
        "",
        "## Full Comparison",
        "",
        dataframe_to_markdown(summary),
        "",
    ]
    for _, row in best_by_constraint.iterrows():
        stem = f"{row['model']}_{row['constraint']}"
        test_cm = pd.read_csv(output_dir / f"test_confusion_matrix_{stem}.csv", index_col=0)
        report.extend(
            [
                f"## Test Report: {stem}",
                "",
                "```text",
                classification_report(
                    y_test,
                    apply_threshold(
                        next(s for s in model_scores if s.name == row["model"]).test_scores,
                        next(s for s in model_scores if s.name == row["model"]).test_severity,
                        float(row["threshold"]),
                    ),
                    target_names=CLASS_NAMES,
                    zero_division=0,
                ),
                "```",
                "",
                dataframe_to_markdown(test_cm.reset_index(names="actual")),
                "",
            ]
        )
    (output_dir / "classification_report.md").write_text("\n".join(report), encoding="utf8")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    set_seeds(args.seed)
    project_root = args.project_root.resolve()
    output_dir = project_root / OUTPUT_DIR

    train_rows, train_domain = load_rows_and_domain_features(project_root, "train")
    val_rows, val_domain = load_rows_and_domain_features(project_root, "val")
    test_rows, test_domain = load_rows_and_domain_features(project_root, "test")
    y_train = train_rows["label"].to_numpy(dtype=int)
    y_val = val_rows["label"].to_numpy(dtype=int)
    y_test = test_rows["label"].to_numpy(dtype=int)

    model_scores: list[ModelScores] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model_scores.append(
            fit_two_stage_mlp(
                project_root,
                y_train,
                y_val,
                y_test,
                train_domain,
                val_domain,
                test_domain,
                args.seed,
                args.mlp_max_iter,
            )
        )

    cnn_scores = fit_cnn(
        train_rows,
        val_rows,
        test_rows,
        args.seed,
        args.cnn_epochs,
        args.cnn_size,
        args.batch_size,
    )
    cnn_available = cnn_scores is not None
    if cnn_scores is not None:
        model_scores.append(cnn_scores)

    elapsed = time.perf_counter() - start
    write_outputs(
        output_dir,
        model_scores,
        y_val,
        y_test,
        val_rows,
        test_rows,
        args.seed,
        elapsed,
        cnn_available,
    )
    print(f"Output directory: {output_dir}")
    print(pd.read_csv(output_dir / "constraint_comparison.csv").to_string(index=False))


if __name__ == "__main__":
    main()
