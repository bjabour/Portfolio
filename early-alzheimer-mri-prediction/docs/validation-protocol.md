# Validation Protocol

## Split Roles

| Split | Permitted use |
| --- | --- |
| Train | Fit scalers, PCA, classifiers, neural-network weights, and class weights. |
| Validation | Compare models, select thresholds, and create train-fitted diagnostic projections. |
| Test | One final audit after the model and threshold are fixed. |

Preprocessing must never be refitted on validation or test.

## Threshold Selection

Threshold sweeps are computed on validation scores. For the original constrained experiment, thresholds were selected by minimum validation beta within each candidate alpha rule; ties preferred macro-F1, balanced accuracy, then specificity.

The final decision reframes the operating points around the practical objective:

1. Minimize beta.
2. Reject the zero-beta point because alpha is extreme.
3. Recommend threshold `0.50`, which gives lower beta than the lower-alpha alternative while keeping test alpha in the single digits.

No threshold is retuned on test in the scripts. However, test results were reviewed across iterations after the initial experiment, so the current test set is no longer a pristine final holdout.

## Required Assertions

The scripts check or should check:

- Expected split counts: 4,435 / 950 / 951.
- Exactly three labels in fixed order.
- Finite feature values and predictions.
- Probability bounds in `[0, 1]`.
- Train-only fitting of scaler and PCA.
- Fixed confusion-matrix label order.
- Repeated runs with seed 42 produce stable outputs within library/hardware constraints.

## Reported Metrics

Primary:

- Beta / missed-dementia rate.
- Sensitivity.

Secondary:

- Alpha / false-positive rate.
- Specificity.
- ROC-AUC.
- Balanced accuracy and macro-F1 for three-class assessment.
- Confusion counts, because percentages can hide clinical tradeoffs.

## Stability and Uncertainty

The regional occlusion analysis uses paired bootstrap intervals over validation images. This captures sampling variation for that split, but it does not address patient duplication, site shift, scanner shift, or label error.

## Stronger Future Evaluation

Before making a final claim:

1. Obtain patient identifiers.
2. Use grouped nested cross-validation, keeping each patient in one fold.
3. Perform model and threshold selection only inside the inner loop.
4. Aggregate untouched outer-fold predictions.
5. Add confidence intervals for beta and alpha at the selected operating rule.
6. Validate on a separate institution or public cohort.
7. Test calibration and subgroup performance by age, sex, scanner, and site.

## Current Evidence Status

The repository preserves the complete development evidence but labels it as a course/research result. The current numbers are appropriate for comparing methods inside this dataset, not for estimating real-world diagnostic performance.
