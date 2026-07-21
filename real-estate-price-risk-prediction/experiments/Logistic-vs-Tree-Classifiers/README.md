# Logistic Regression vs RF and XGBoost

## Conclusion

Regularized logistic regression is competitive with the selected random
forest and better than the selected XGBoost classifier on repeated
out-of-fold log loss.

| Model | Log loss | Brier | ROC AUC |
|---|---:|---:|---:|
| Raw-feature L2 logistic | 0.3696 | 0.1126 | 0.7396 |
| Degree-2 spline logistic | 0.3687 | 0.1113 | 0.7385 |
| Random forest | 0.3705 | 0.1133 | 0.7500 |
| XGBoost | 0.3817 | 0.1168 | 0.7392 |
| Logistic + RF blend | 0.3627 | 0.1107 | 0.7608 |

Plain L2 logistic is recommended over spline logistic because its log-loss
difference is only 0.0009,
which is much smaller than repeat-to-repeat uncertainty.

Against RF, logistic won
4 of
6 repeats. The mean RF-minus-logistic
difference was
0.0009, with a
repeat-bootstrap interval of
[-0.0076,
0.0085]. They should be treated as tied.

Against XGBoost, logistic won
5 of
6 repeats. The mean
XGBoost-minus-logistic difference was
0.0121.

## Blend

A logistic/RF blend was tuned by leaving out one complete CV repeat at a
time. Its mean held-repeat log loss was 0.3627. The
selected logistic weights ranged from
0.48 to
0.58.

For the exported test benchmark, the full OOF optimum places
0.53 weight on logistic and
0.47 on RF.

## Final Logistic Model

The final raw-feature model uses standardized predictors, natural class
prevalence, L2 regularization, and `C=0.316228`. Balanced class weights
were not used because they inflate probabilities and worsen log loss.

## Project Note

This experiment does not replace tasks 4 and 5. trees still requires RF and
XGBoost outputs and their MDI importances. Logistic regression and the blend
are diagnostic benchmarks outside the required JSON contract.
