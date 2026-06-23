# XGBoost Classification Task 5 Findings

## Scope

This experiment implements only Task 5 with XGBoost. It reads the untouched
original train and test CSV files, preserves original year columns and row
order, and retains the uncapped `LotArea` maximum.

## Data And Scoring

- Training rows: 300
- Distressed cases: 47
- Training prevalence: 15.666667%
- Primary metric: validation log loss
- Supporting metrics: Brier score, ROC AUC, prevalence bias, and total-gain
  rank stability

Threshold accuracy was not used because the assignment grades probability
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

- Max depth: 3
- Learning rate: 0.03
- Estimators: 122
- Min child weight: 3.0
- Gamma: 0.1
- Subsample: 0.85
- Column subsample: 0.85
- L2 regularization: 1.0
- L1 regularization: 0.1
- Imbalance strategy: scale_pos_weight
- Positive-class weight: 1.5
- Max delta step: 0.0
- Sampling ratio: None
- SMOTE neighbors: None
- Calibration: sigmoid
- Calibration folds: 8
- Repeated-CV log loss: 0.381670
- Repeated-CV Brier score: 0.116792
- Repeated-CV ROC AUC: 0.748222

The final estimator count is the median leakage-free best iteration associated
with the winning base configuration.

## Variable Importance

The submission candidate uses normalized XGBoost `total_gain`, the summed loss
reduction assigned to each feature. Gain, split count, cover, and total cover
are retained as diagnostics only.

## Verification

- Focused tests passed: 10/10
- Final probabilities: 200
- Importance sum: 1.0000000000000000
- Source hashes unchanged: True

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
