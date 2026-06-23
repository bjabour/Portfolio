# Tree Imbalance Strategy Comparison

- Dataset: `trees_train_year_diffs_2011.csv`
- Evaluation: 5-fold stratified CV repeated 5 times
- Tree: `criterion="log_loss"`, `max_depth=4`, `min_samples_leaf=5`
- Primary selection metric: out-of-fold log loss
- Training prevalence: 15.667%
- Initial best strategy: `none`
- Sigmoid calibration was then tested for that strategy using inner 3-fold CV.
- Overall best result: `none_sigmoid_calibrated` with log loss 0.4020

SMOTE and all preprocessing were fitted only inside each training fold.
Validation observations remained untouched. The area features used `log1p`
followed by standard scaling; the remaining predictors used standard scaling.
