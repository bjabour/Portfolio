# Calibration Fold Tuning

- Outer evaluation: 5-fold stratified CV repeated 5 times.
- Candidate inner calibration folds: 2, 3, 4, 5, 6, 8, 10.
- Fixed tree: max depth 4, minimum leaf size 5.
- Moderate SMOTE ratio: 0.5.
- Selection metric: repeated out-of-fold log loss.
- Best candidate: `sigmoid_calibration_k5`.
- Best log loss: 0.398411.
- Best ROC AUC: 0.728450 for the selected candidate.

The number of folds is a genuine fitted-model setting only for sigmoid
calibration. Plain and SMOTE trees are included as fixed reference methods
and are evaluated on exactly the same outer validation folds.
