# XGBoost Regression Task 3 Findings

## Scope

This experiment implements only Task 3. It reads the untouched original train
and test CSV files, preserves the original year columns, and retains the
uncapped training `LotArea` maximum of 215245.

## Selection Method

- Library: XGBoost 3.2.0
- Objective: `reg:squarederror`
- Primary metric: validation MSE
- Initial screen: 60 deterministic configurations, five folds, seed 9618
- Robust comparison: 10 shortlisted configurations over six
  predeclared seeds and 30 validation folds per configuration
- Early stopping was performed on an inner split of each outer training fold.
  The outer validation fold never selected the number of boosting rounds.

## Selected Model

- Candidate: 22
- Max depth: 1
- Learning rate: 0.05
- Number of estimators: 386
- Min child weight: 1
- Gamma: 0.1
- Subsample: 0.85
- Column subsample: 0.85
- L2 regularization: 1.0
- L1 regularization: 1.0
- Repeated-CV mean MSE: 1362.697230
- Repeated-CV RMSE: 36.914729
- MDI rank stability: 0.933234

The final number of estimators is the median leakage-free best iteration from
the selected configuration's 30 validation fits. The model was then refit on
all 300 training rows without an evaluation set.

## Variable Importance

The submission candidate uses normalized XGBoost `total_gain`, which sums the
loss reduction attributable to each feature across all splits. Average gain,
split count, cover, and total cover are saved only as diagnostics.

## Verification

- Focused tests passed: 8/8
- Test predictions: 200
- Importance sum: 1.0000000000000000
- Deterministic duplicate fit: True
- Source hashes unchanged: True

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
