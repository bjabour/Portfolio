# Random-Forest Classification Task 4 Findings

## Scope

This experiment implements only Task 4 with sklearn's library random forest.
It uses the untouched original CSV files, original year columns, uncapped
`LotArea`, and original row order.

## Data And Scoring

- Training rows: 300
- Distressed cases: 47
- Training prevalence: 15.666667%
- Primary model-selection metric: validation log loss
- Supporting metrics: Brier score, ROC AUC, prevalence bias, and MDI stability
- Threshold accuracy was not used for selection because the project evaluates
  predicted probabilities.

## Search

- Structural screen: 72 deterministic 500-tree configurations
- Imbalance screen: natural training, class weighting, random oversampling,
  and fold-local SMOTE
- Calibration screen: uncalibrated, sigmoid with 3/5/8 inner folds, and
  five-fold isotonic diagnostics
- Final comparison: six complete configurations over 30 untouched outer folds

All resampling and calibration occurred inside training folds. For SMOTE only,
area variables received fold-local `log1p` plus standard scaling and all other
variables received fold-local standard scaling.

## Selected Model

- Max depth: 5
- Minimum leaf size: 1
- mtry: 6
- Criterion: log_loss
- Bootstrap sample fraction: 1.0
- Training strategy: random_over
- Sampling ratio: 0.2
- SMOTE neighbors: None
- Calibration: none
- Calibration folds: None
- Final trees: 2000
- Declared imbalance method: `resampling`
- Repeated-CV log loss: 0.370461
- Repeated-CV Brier score: 0.113337
- Repeated-CV ROC AUC: 0.756952

## Variable Importance

The submission candidate is the normalized sklearn impurity-decrease vector
from the fitted underlying forest. Calibration changes probabilities but does
not change the forest's split-based importance.

## Verification

- Focused tests passed: 10/10
- Final probabilities: 200
- Importance sum: 0.9999999999999999
- Source hashes unchanged: True

## Research Notes

Current sklearn documentation defines forest probabilities as averages of
tree class probabilities and exposes impurity-based feature importance.
Calibration is evaluated because the evaluation target is cross-entropy rather
than classification accuracy. Imbalanced-learn's leakage guidance supports
placing every sampler inside the training-fold pipeline. MDI remains a
predictive model importance and can divide credit among correlated housing
features.

Sources:

- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/calibration.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
- https://imbalanced-learn.org/stable/common_pitfalls.html
- https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf

No external implementation code was copied.
