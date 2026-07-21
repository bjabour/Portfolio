# CART Task 1 Findings

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

The official folds use one deterministic shuffle with seed 9618.
Each of the 300 observations appears in exactly one 60-row validation fold.

| Maximum depth | Mean CV MSE | Fold SD | OOF RMSE |
|---:|---:|---:|---:|
| 2 | 3918.884 | 502.008 | 62.601 |
| 3 | 2878.913 | 230.015 | 53.656 |
| 4 | 2517.852 | 442.398 | 50.178 |
| 5 | 2365.254 | 352.450 | 48.634 |
| 6 | 2300.548 | 296.951 | 47.964 |
| 7 | 2322.913 | 413.343 | 48.197 |
| 8 | 2336.057 | 478.653 | 48.333 |
| 9 | 2337.087 | 476.287 | 48.343 |
| 10 | 2337.087 | 476.287 | 48.343 |

The first minimum occurs at **depth 6**, with mean CV MSE
**2300.548** and OOF RMSE
**47.964**.

The required JSON fields have different roles:

- `tree_pred_test` must contain predictions from the required depth-3 scratch
  tree.
- `tree_lib_pred_test` must contain predictions from the fixed depth-3
  sklearn comparison tree.
- `tree_max_depth` records the depth selected by scratch CV, which is
  6.
- `tree_cv_mse_curve` records the nine MSE values for depths 2 through 10.

## Fitted Tree Diagnostics

| Model | Depth | Nodes | Leaves | Training MSE | Validation RMSE |
|---|---:|---:|---:|---:|---:|
| required_depth_3_scratch | 3 | 15 | 8 | 2106.611 | 53.656 |
| cv_selected_scratch | 6 | 65 | 33 | 906.921 | 47.964 |

The depth-3 root split is
`OverallQual <= 6.5`.

### Depth-3 Upper Splits

| Depth | Node | Transformed split | Original-scale report | Rows | SSE reduction |
|---:|---:|---|---|---:|---:|
| 0 | 0 | `OverallQual <= 6.5` | `OverallQual <= 6.5` | 300 | 1846980.986 |
| 1 | 1 | `GrLivArea <= 1112` | `GrLivArea <= 1112` | 185 | 278912.038 |
| 2 | 2 | `years_since_built_2011 <= 61.5` | `YearBuilt >= 1949.5` | 66 | 90068.483 |
| 2 | 5 | `GarageArea <= 401` | `GarageArea <= 401` | 119 | 101205.559 |
| 1 | 8 | `OverallQual <= 7.5` | `OverallQual <= 7.5` | 115 | 233371.792 |
| 2 | 9 | `TotalBsmtSF <= 837.5` | `TotalBsmtSF <= 837.5` | 61 | 60374.979 |
| 2 | 12 | `GrLivArea <= 2771.5` | `GrLivArea <= 2771.5` | 54 | 125615.077 |

## Scratch-versus-sklearn Equivalence

- Test prediction RMSE: 1.135e-13
- Maximum test prediction difference:
  2.274e-13
- Training prediction RMSE:
  1.113e-13
- Matching test rows: 200/200
- Matching depth: True
- Matching node count: True
- Matching leaf count: True
- Matching training leaf partitions:
  True
- Maximum matching-leaf prediction difference:
  2.274e-13

The required tolerances were passed: RMSE below `1e-10` and maximum absolute
difference below `1e-9`.

## Robustness Diagnostics

The official answer remains based only on seed 9618. Additional
fixed seeds were used to assess depth-selection stability, not to search for a
favorable answer.

Selection counts across 6 fixed seeds: depth 5: 3, depth 6: 2, depth 7: 1.

All required full-tree year-transformation checks passed:
4/4.
The transformations preserve the training partitions, although their
decreasing direction swaps the visual left and right branches.

The transformed-year and original-year CV searches both selected depth
6. Their fold MSE curves were
not numerically identical: the maximum absolute depth-level difference was
46.052. A held-out calendar year can
equal a midpoint that was absent from that fold's training rows. Because CART
routes equality through the `<=` branch, reversing the year axis can route
that boundary row to the complementary branch. This diagnostic does not
change the required official result, which is fitted and selected entirely
with the transformed columns.

The vectorized split search agreed with brute force in
4/4 cases.
All 4 focused unit tests passed.

## Data Integrity

- Train SHA-256 before and after:
  `85a5ad0b994e5b4202bd22a7e8b7d8d35a00338d64d85312544b65921cf8a242`
- Test SHA-256 before and after:
  `45aa6f83073380302b8a7a7554bc13beb2774d07d34cdb3dfd30d0ef666b6580`
- Original train `LotArea` maximum:
  215245
- Original test `LotArea` maximum:
  159000

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
   used because this project explicitly requires five-fold depth tuning.
3. A 2026 decision-tree pruning project reinforces the practical
   bias-variance role of controlling tree size. Again, the project's
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
- `official_cv_fold_projects.csv`
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
