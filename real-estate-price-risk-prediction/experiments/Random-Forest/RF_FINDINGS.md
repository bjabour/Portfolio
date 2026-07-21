# Random-Forest Task 2 Findings

## Scope

This experiment implements only Task 2. It does not create the final trees
submission JSON, slides, slide source, or models for Tasks 3 through 5.

The forest reads only the original supplied train and test CSV files. It uses
the original `YearBuilt` and `YearRemodAdd` columns and retains every original
`LotArea` value, including the training maximum of
215245.

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

The 32-setting screen used 500 trees per setting, fixed `mtry=4`,
and seed 9618. The leading OOB setting was depth
8, leaf size
1, with OOB MSE
1424.569. OOB screening was used only to create the
cross-validation shortlist of 7 settings.

## Repeated Five-Fold Selection

Each shortlisted setting was evaluated in 30 untouched validation folds:
five folds for each predeclared seed `0, 42, 2026, 9618, 19618, 31415`.
Every fold forest contained 500 trees.

| Depth | Leaf size | Mean validation MSE | RMSE | MDI rank stability | Top-5 overlap | Selected |
|---:|---:|---:|---:|---:|---:|:---:|
| 8 | 1 | 1484.850 | 38.534 | 0.979 | 1.000 | yes |
| 8 | 2 | 1501.603 | 38.751 | 0.978 | 1.000 |  |
| 12 | 2 | 1502.643 | 38.764 | 0.977 | 1.000 |  |
| 15 | 2 | 1504.727 | 38.791 | 0.978 | 1.000 |  |
| 12 | 3 | 1523.202 | 39.028 | 0.971 | 1.000 |  |
| 15 | 3 | 1523.785 | 39.036 | 0.972 | 1.000 |  |
| 7 | 5 | 1579.598 | 39.744 | 0.963 | 1.000 |  |

The selected setting is **max depth 8, minimum leaf size
1**. Its mean validation MSE is
**1484.850**, corresponding to RMSE
**38.534**.

Candidates within 0.25% of the minimum were eligible for the stability
tie-break. The order was MDI pairwise Spearman stability, fold-MSE standard
deviation, shallower depth, then smaller leaf size.

## Final Forest

The final experimental forest contains **2000 trees**, uses
`mtry=4`, max depth 8, minimum leaf size 1, and
seed 9618.

- Final OOB MSE: 1420.806
- Final OOB RMSE: 37.694
- OOB coverage: 100.000%
- Minimum OOB predictions per row: 681
- Maximum OOB predictions per row: 792
- OOB MSE at 200 trees: 1480.516
- Test-prediction RMSE between 200 and 2,000 trees:
  2.903

## Mean Decrease in Impurity

| Rank | Feature | Equal-tree MDI | Global raw-SSE share |
|---:|---|---:|---:|
| 1 | `OverallQual` | 0.278383 | 0.278079 |
| 2 | `YearBuilt` | 0.197818 | 0.197865 |
| 3 | `GrLivArea` | 0.162616 | 0.163139 |
| 4 | `GarageArea` | 0.135628 | 0.135509 |
| 5 | `TotalBsmtSF` | 0.099525 | 0.099319 |
| 6 | `YearRemodAdd` | 0.035688 | 0.035907 |
| 7 | `LotArea` | 0.030419 | 0.030432 |
| 8 | `OpenPorchSF` | 0.025375 | 0.025209 |
| 9 | `OverallCond` | 0.011796 | 0.011854 |
| 10 | `BedroomAbvGr` | 0.008453 | 0.008395 |
| 11 | `WoodDeckSF` | 0.007916 | 0.007926 |
| 12 | `Fireplaces` | 0.006383 | 0.006366 |

The equal-tree MDI vector is the Task 2 submission candidate because it matches
the current forest convention used by sklearn. The global raw-SSE share is
retained only as a diagnostic. MDI is predictive importance, not causal or
generative importance, and correlated housing predictors can divide impurity
credit.

## Verification

- Focused tests passed: 10/10
- Train source hash unchanged:
  True
- Test source hash unchanged:
  True
- Task 1 locked artifacts unchanged:
  True
- Final prediction rows: 200
- Final importance sum: 1.0000000000000000

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
   importance in the underlying data-generating process. The project
   explicitly evaluates MDI, so alternative sensitivity measures are contextual
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
