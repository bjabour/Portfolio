# Project Profile

## Project

Tree-Based Real Estate Modeling With Scratch Implementations and Calibrated Ensembles

## Context

This course project used 300 training observations and 200 test observations from an Ames housing style dataset with 12 numeric predictors. The goal was to predict sale price and estimate the probability of a distressed sale. The classification target was imbalanced, with 47 distressed sales among 300 training observations.

## What I Built

I implemented a CART regression tree from scratch using greedy midpoint splitting, recursive partitioning, SSE reduction, deterministic tie-breaking, and leaf predictions equal to node means. I then built a scratch random forest on top of that tree using bootstrap sampling, per-node random feature selection with `mtry=4`, out-of-bag diagnostics, and normalized Mean Decrease in Impurity feature importance.

For gradient boosting, I used XGBoost regression and classification models with leakage-safe cross-validation, early stopping, regularization, and feature-importance reporting via normalized total gain. For classification, I compared random oversampling, class weighting, SMOTE diagnostics, and sigmoid calibration using held-out log loss rather than threshold accuracy.

## Validation Design

The project used manual and repeated cross-validation. Resampling, early stopping, scaling, and calibration were restricted to training folds to avoid validation leakage. Regression models were selected by validation MSE/RMSE. Classification models were selected by probability cross-entropy/log loss, with AUC, Brier score, prevalence bias, and importance stability used as supporting diagnostics.

## Selected Results

- Scratch CART: required depth-3 tree matched the permitted sklearn reference tree to numerical precision; CV-selected depth was 6.
- Scratch random forest: 2,000 trees, `mtry=4`, `max_depth=8`, `min_samples_leaf=1`; repeated-CV MSE 1484.850.
- XGBoost regression: 386 trees, learning rate 0.05, depth 1; repeated-CV MSE 1362.697.
- Random-forest classifier: 2,000 trees, depth 5, `mtry=6`, log-loss criterion, fold-local random oversampling ratio 0.20; mean log loss 0.370461.
- XGBoost classifier: 122 trees, learning rate 0.03, depth 3, `scale_pos_weight=1.5`, eight-fold out-of-fold sigmoid calibration; mean log loss 0.381670.

## Outcome

The assignment received a final score of 100/100, including full credit for performance, slides, and code. The project demonstrates scratch algorithm implementation, ensemble learning, imbalanced classification, probability calibration, leakage-safe validation, and professional technical communication.

## Keywords

Python, NumPy, pandas, scikit-learn, XGBoost, CART, random forest, gradient boosting, cross-validation, model calibration, imbalanced classification, log loss, MSE, feature importance, real estate analytics.
