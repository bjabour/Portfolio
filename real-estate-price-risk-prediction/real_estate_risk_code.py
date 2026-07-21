import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv-trees-tree-models" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    try:
        import imblearn
    except ModuleNotFoundError:
        print(f"Re-launching with {VENV_PYTHON.name}...", flush=True)
        raise SystemExit(subprocess.run(
            [str(VENV_PYTHON), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
        ).returncode)

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


TRAIN, TEST = ROOT / "real_estate_risk_train.csv", ROOT / "real_estate_risk_test.csv"
OUT = ROOT / "real_estate_risk_results.json"
FEATURES = ["GrLivArea", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
            "TotalBsmtSF", "GarageArea", "BedroomAbvGr", "Fireplaces", "WoodDeckSF", "OpenPorchSF"]
SEED = 9618


def sse(y):
    total, total_sq = float(y.sum()), float(np.dot(y, y))
    return max(0., total_sq - total * total / len(y))


class ScratchCARTRegressor:
    def __init__(self, max_depth, min_samples_leaf=5, max_features=None, random_state=None):
        self.max_depth, self.min_leaf, self.max_features, self.random_state = max_depth, min_samples_leaf, max_features, random_state

    def fit(self, x, y, names=None):
        self.x, self.y = np.asarray(x, np.float32), np.asarray(y, float)
        self.p, self.rng, self.raw_importance_ = self.x.shape[1], np.random.default_rng(self.random_state), np.zeros(self.x.shape[1])
        self.root = self._grow(np.arange(len(y)), 0)
        return self

    def _grow(self, rows, depth):
        node = [float(self.y[rows].mean()), None, None, None, None]
        parent = sse(self.y[rows])
        if depth >= self.max_depth or len(rows) < 2 * self.min_leaf or parent <= 1e-12:
            return node
        features = np.arange(self.p) if self.max_features is None else np.sort(
            self.rng.choice(self.p, self.max_features, replace=False))
        best = None
        for feature in features:
            order = np.argsort(self.x[rows, feature], kind="mergesort")
            ordered, values, target = rows[order], self.x[rows[order], feature], self.y[rows[order]]
            positions = np.arange(self.min_leaf - 1, len(rows) - self.min_leaf)
            positions = positions[values[positions] < values[positions + 1]]
            if not len(positions):
                continue
            sy, sy2 = np.cumsum(target), np.cumsum(target * target)
            nl, nr = positions + 1, len(rows) - positions - 1
            sl, sql = sy[positions], sy2[positions]
            sr, sqr = sy[-1] - sl, sy2[-1] - sql
            losses = np.maximum(0, sql - sl * sl / nl) + np.maximum(0, sqr - sr * sr / nr)
            offset = int(np.argmin(losses))
            pos, loss = int(positions[offset]), float(losses[offset])
            threshold = float(np.float64(values[pos]) / 2 + np.float64(values[pos + 1]) / 2)
            if best is None or loss < best[0] - 1e-12 or (
                abs(loss - best[0]) <= 1e-12 and (feature, threshold) < (best[1], best[2])):
                best = loss, int(feature), threshold, ordered[:pos + 1], ordered[pos + 1:]
        if best is None or parent - best[0] <= 1e-12:
            return node
        loss, feature, threshold, left, right = best
        self.raw_importance_[feature] += parent - loss
        node[1:] = feature, threshold, self._grow(left, depth + 1), self._grow(right, depth + 1)
        return node

    def predict(self, x):
        def one(row):
            node = self.root
            while node[1] is not None:
                node = node[3] if row[node[1]] <= node[2] else node[4]
            return node[0]
        return np.array([one(row) for row in np.asarray(x, np.float32)])


class ScratchRandomForestRegressor:
    def __init__(self, n_estimators, max_depth, min_samples_leaf, max_features=4, random_state=9618):
        self.n, self.depth, self.leaf, self.mtry, self.seed = n_estimators, max_depth, min_samples_leaf, max_features, random_state

    def fit(self, x, y, names=None):
        x, y, trees, mdi, nonstump = np.asarray(x, np.float32), np.asarray(y, float), [], np.zeros(x.shape[1]), 0
        for child in np.random.SeedSequence(self.seed).spawn(self.n):
            rng = np.random.default_rng(child)
            boot = rng.integers(0, len(y), size=len(y), dtype=np.int64)
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
            tree = ScratchCARTRegressor(self.depth, self.leaf, self.mtry, tree_seed).fit(x[boot], y[boot])
            trees.append(tree)
            if tree.raw_importance_.sum() > 0:
                mdi += tree.raw_importance_ / tree.raw_importance_.sum()
                nonstump += 1
        self.trees = trees
        average = mdi / nonstump
        self.feature_importances_ = average / average.sum()
        return self

    def predict(self, x):
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)


def folds(n, k=5):
    return [np.asarray(x) for x in np.array_split(np.random.default_rng(SEED).permutation(n), k)]


def task1(train, test):
    y = train.sale_price_keur.to_numpy()
    x, xt, all_rows = train[FEATURES].to_numpy(np.float32), test[FEATURES].to_numpy(np.float32), np.arange(len(y))
    curve = []
    for depth in range(2, 11):
        scores = []
        for valid in folds(len(y)):
            fit = np.setdiff1d(all_rows, valid)
            tree = ScratchCARTRegressor(depth, 5).fit(x[fit], y[fit], FEATURES)
            scores.append(np.mean((y[valid] - tree.predict(x[valid])) ** 2))
        curve.append(float(np.mean(scores)))
    scratch = ScratchCARTRegressor(3, 5).fit(x, y, FEATURES).predict(xt)
    library = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5,
                                    criterion="squared_error", random_state=0).fit(x, y).predict(xt)
    return scratch, int(np.argmin(curve) + 2), curve, library


def xgb_importance(model):
    score = model.get_booster().get_score(importance_type="total_gain")
    values = np.array([score.get(f, score.get(f"f{i}", 0.0)) for i, f in enumerate(FEATURES)], float)
    values /= values.sum()
    return dict(zip(FEATURES, values.tolist()))


def task2(train, test):
    x, xt, y = train[FEATURES].to_numpy(np.float32), test[FEATURES].to_numpy(np.float32), train.sale_price_keur.to_numpy()
    model = ScratchRandomForestRegressor(2000, 8, 1, 4, SEED).fit(x, y, FEATURES)
    return model.predict(xt), dict(zip(FEATURES, model.feature_importances_.tolist()))


def task3(train, test):
    params = dict(n_estimators=386, learning_rate=.05, max_depth=1, min_child_weight=1,
                  gamma=.1, subsample=.85, colsample_bytree=.85, reg_lambda=1, reg_alpha=1)
    model = XGBRegressor(objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                         random_state=SEED, n_jobs=1, verbosity=0, **params).fit(
                             train[FEATURES], train.sale_price_keur)
    return model.predict(test[FEATURES]), params, xgb_importance(model)


def task4(train, test):
    x, y = train[FEATURES], train.distressed.astype(int)
    xr, yr = RandomOverSampler(sampling_strategy={1: 51}, random_state=SEED).fit_resample(x, y)
    model = RandomForestClassifier(n_estimators=2000, max_depth=5, min_samples_leaf=1,
                                   max_features=6, criterion="log_loss", max_samples=1.0,
                                   random_state=SEED, n_jobs=1).fit(xr, yr)
    imp = model.feature_importances_ / model.feature_importances_.sum()
    return model.predict_proba(test[FEATURES])[:, 1], dict(zip(FEATURES, imp.tolist()))


def make_xgb_classifier(seed):
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                         n_estimators=122, learning_rate=.03, max_depth=3, min_child_weight=3,
                         gamma=.1, subsample=.85, colsample_bytree=.85, reg_lambda=1,
                         reg_alpha=.1, scale_pos_weight=1.5, max_delta_step=0,
                         random_state=seed, n_jobs=1, verbosity=0)


def task5(train, test):
    x, y = train[FEATURES], train.distressed.astype(int)
    oof = np.empty(len(y))
    from sklearn.model_selection import StratifiedKFold
    for fold, (fit, valid) in enumerate(StratifiedKFold(8, shuffle=True, random_state=709618).split(x, y), 1):
        oof[valid] = make_xgb_classifier(SEED * 100 + fold).fit(x.iloc[fit], y.iloc[fit]).predict_proba(x.iloc[valid])[:, 1]
    def logit(p):
        p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logit(oof)[:, None], y)
    model = make_xgb_classifier(SEED).fit(x, y)
    raw = model.predict_proba(test[FEATURES])[:, 1]
    return calibrator.predict_proba(logit(raw)[:, None])[:, 1], xgb_importance(model)


def run_models(verbose=False):
    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
    tasks = [task1, task2, task3, task4, task5]
    results = []
    for number, task in enumerate(tasks, 1):
        if verbose:
            print(f"Running Task {number}/5: {task.__name__}...", flush=True)
        results.append(task(train, test))
    return tuple(results)


def build_result(verbose=False):
    tree, rf_reg, gb_reg, rf_clf, gb_clf = run_models(verbose)
    tree_pred, depth, curve, tree_lib = tree
    rf_reg_pred, rf_reg_imp = rf_reg
    gb_reg_pred, gb_reg_params, gb_reg_imp = gb_reg
    rf_prob, rf_clf_imp = rf_clf
    gb_prob, gb_clf_imp = gb_clf
    return {
        "_description": "trees results for portfolio project; shape follows real_estate_risk_results_example.json.",
        "tree_pred_test": tree_pred.tolist(), "tree_max_depth": depth,
        "tree_cv_mse_curve": curve, "tree_lib_pred_test": tree_lib.tolist(),
        "rf_reg_pred_test": rf_reg_pred.tolist(), "rf_reg_var_importance": rf_reg_imp,
        "rf_reg_n_trees": 2000, "rf_reg_mtry": 4, "rf_reg_max_depth": 8,
        "gbm_reg_pred_test": gb_reg_pred.tolist(),
        "gbm_reg_hyperparams": {k: gb_reg_params[k] for k in
                                ["n_estimators", "learning_rate", "max_depth"]},
        "gbm_reg_var_importance": gb_reg_imp,
        "rf_clf_pred_prob": rf_prob.tolist(), "rf_clf_var_importance": rf_clf_imp,
        "rf_clf_imbalance_method": "resampling", "rf_clf_n_trees": 2000,
        "rf_clf_mtry": 6, "rf_clf_max_depth": 5,
        "gbm_clf_pred_prob": gb_prob.tolist(),
        "gbm_clf_var_importance": gb_clf_imp,
        "gbm_clf_hyperparams": {
            "n_estimators": 122, "learning_rate": .03, "max_depth": 3,
            "min_child_weight": 3, "gamma": .1, "subsample": .85,
            "colsample_bytree": .85, "reg_lambda": 1, "reg_alpha": .1,
            "imbalance_strategy": "scale_pos_weight", "scale_pos_weight": 1.5,
            "max_delta_step": 0, "sampling_ratio": None, "k_neighbors": None,
            "calibration_method": "sigmoid", "calibration_folds": 8,
        },
    }


def validate(result):
    example = json.loads((ROOT / "real_estate_risk_results_example.json").read_text())
    if set(result) != set(example):
        raise ValueError(f"JSON keys differ from example: {set(result) ^ set(example)}")

    arrays = ["tree_pred_test", "tree_lib_pred_test", "rf_reg_pred_test",
              "gbm_reg_pred_test", "rf_clf_pred_prob", "gbm_clf_pred_prob"]
    for key in arrays:
        values = np.asarray(result[key], float)
        if values.shape != (200,) or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain 200 finite floats")
    for key in ["rf_clf_pred_prob", "gbm_clf_pred_prob"]:
        values = np.asarray(result[key], float)
        if not np.all((values >= 0) & (values <= 1)):
            raise ValueError(f"{key} must be within [0,1]")

    curve = np.asarray(result["tree_cv_mse_curve"], float)
    if curve.shape != (9,) or not np.isfinite(curve).all():
        raise ValueError("tree_cv_mse_curve must contain nine finite values")
    if result["tree_max_depth"] not in range(2, 11):
        raise ValueError("tree_max_depth must be in {2,...,10}")
    if np.max(np.abs(np.asarray(result["tree_pred_test"]) -
                     np.asarray(result["tree_lib_pred_test"]))) >= 1e-9:
        raise ValueError("Scratch and sklearn depth-3 trees do not agree")

    for key in ["rf_reg_var_importance", "gbm_reg_var_importance",
                "rf_clf_var_importance", "gbm_clf_var_importance"]:
        importance = result[key]
        values = np.asarray([importance[name] for name in FEATURES], float)
        if set(importance) != set(FEATURES) or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain exactly the 12 features")
        if (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-8):
            raise ValueError(f"{key} must be nonnegative and sum to one")

    if result["rf_reg_n_trees"] < 200 or result["rf_clf_n_trees"] < 200:
        raise ValueError("Both forests require at least 200 trees")
    if result["rf_reg_mtry"] != 4:
        raise ValueError("Task 2 requires mtry=4")
    if result["rf_clf_imbalance_method"] not in {
            "class_weight_balanced", "resampling", "threshold_tuning", "none"}:
        raise ValueError("Invalid RF classification imbalance method")
    for key in ["gbm_reg_hyperparams", "gbm_clf_hyperparams"]:
        if not {"n_estimators", "learning_rate", "max_depth"} <= set(result[key]):
            raise ValueError(f"{key} lacks required example parameters")
    return result


def main():
    result = validate(build_result(verbose=True))
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Completed all five tasks. CV-selected CART depth: {result['tree_max_depth']}.")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
