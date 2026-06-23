# Portfolio

This repository contains a curated Statistical Learning portfolio under
`assignments/`. It includes six assignment projects with final slides,
reproducible code, submitted result artifacts, relevant train/test data,
experiment outputs, tests, support notes, and project-level README files. The
structure is intentionally organized so each assignment can be reviewed on its
own while keeping supporting work inside subfolders.

## Contents

- `assignments/01-regression` - linear regression deliverables and data.
- `assignments/02-classification` - logistic classification and model-selection
  experiments.
- `assignments/03-resampling` - bootstrap, simulation, and resampling analysis.
- `assignments/04-regularization` - LASSO/model-selection artifacts and
  error-decomposition checks.
- `assignments/05-splines` - nonlinear modeling with splines, validation
  benchmarks, and Monte Carlo checks.
- `assignments/06-trees` - tree-based modeling with CART, random forests,
  XGBoost, classification comparisons, and validation artifacts.

## Assignment 5 - Splines

Assignment 5 focuses on flexible nonlinear modeling with spline-based
regression. The main folder includes the final submitted slides, result JSON,
summary, train/test data, and reproducible modeling code. Its `experiments/`
folder expands the analysis with exploratory plots, fixed-knot comparisons,
high-knot sensitivity checks, nested cross-validation benchmarks, out-of-fold
diagnostics, and pseudo-DGP Monte Carlo experiments used to test whether the
chosen spline configuration was stable and defensible.

## Assignment 6 - Trees

Assignment 6 is the largest project in the repository. It covers tree-based
modeling for real estate price prediction and distressed-sale probability
estimation, including CART, random forests, XGBoost regression, XGBoost
classification, logistic-vs-tree classifier comparisons, data-cleaning checks,
class-imbalance experiments, calibration diagnostics, feature-importance
outputs, and repeated/nested cross-validation artifacts.
