# Interview Talking Points

## Project In 30 Seconds

I modeled real estate sale price and distressed-sale probability using a mix of scratch tree algorithms and production libraries. The key constraint was that CART and random forest regression had to be implemented from scratch, while gradient boosting and classification models could use libraries. I focused on leakage-safe validation, probability quality, and clear feature-importance reporting.

## Why The Project Is Strong

- It combines algorithmic implementation with practical model selection.
- It handles both regression and imbalanced probabilistic classification.
- It avoids validation leakage in resampling, early stopping, and calibration.
- It compares model families using the metrics aligned with grading: MSE for price and cross-entropy/log loss for probabilities.
- It produced a perfect assignment score while retaining interpretable diagnostics.

## CART Explanation

CART recursively splits the data into two groups. At each node, it tests midpoint thresholds and chooses the split that most reduces SSE. A leaf predicts the mean sale price of the rows inside it. The submitted depth-3 tree was required for equivalence checking, while depth 6 was selected by cross-validation as the best standalone CART depth.

## Scratch Random Forest Explanation

The random forest averages many noisy trees. Each tree trains on a bootstrap sample of 300 rows with replacement. At every split, only four candidate predictors are considered. Out-of-bag rows, meaning rows absent from a tree bootstrap, provide an internal diagnostic error estimate. `min_samples_leaf=1` means a leaf must contain at least one sampled observation, not exactly one.

## XGBoost Explanation

XGBoost builds trees sequentially. Each new tree corrects the current ensemble using gradients and Hessians of the loss. The model is additive:

`prediction = initial prediction + learning_rate * sum(tree corrections)`.

Important controls include `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `gamma`, row/feature subsampling, and L1/L2 regularization.

## Classification Explanation

The classifiers estimate probabilities, not just labels. Cross-entropy/log loss rewards accurate probabilities and heavily penalizes confident wrong predictions. Random oversampling changes how often minority examples appear in training. Class weighting changes how strongly each class affects the loss. Sigmoid calibration fits a logistic mapping from raw scores to better-calibrated probabilities using out-of-fold predictions.

## What I Would Emphasize

The most important engineering choice was avoiding leakage. Every preprocessing, resampling, early-stopping, and calibration decision happened inside the training fold only. That makes the validation scores credible rather than accidentally optimistic.
