# Model Card: Two-Stage MLP MRI Classifier

## Model Details

| Field | Value |
| --- | --- |
| Model | Two-stage scikit-learn `MLPClassifier` pipeline |
| Version | Development snapshot, seed 42 |
| Input | One preprocessed 2D brain MRI image |
| Output | Dementia score, binary decision, and very-mild/mild severity label |
| Recommended threshold | 0.50 |
| Training classes | Non-demented, very mild, mild |
| Primary objective | Minimize beta while avoiding extreme alpha |

## Intended Use

- Statistical-learning coursework.
- Reproducible comparison of deterministic MRI features and neural classifiers.
- Research prototyping and threshold-tradeoff education.

## Out-of-Scope Use

- Clinical diagnosis, screening, treatment, triage, or prognosis.
- Patient-level risk communication.
- Use on unseen scanner protocols without external validation.
- Inference from full 3D studies or other modalities without retraining.
- Claims about disease localization from the regional occlusion plot.

## Architecture

1. Convert the image to grayscale, resize to 128 by 128, and normalize to `[0, 1]`.
2. Compute 1,110 deterministic pixel, texture, edge, asymmetry, grid, and regional features.
3. Apply train-fitted standardization and 120-component PCA.
4. Binary MLP `(128, 64)` estimates dementia probability.
5. Severity MLP `(96, 48)` separates very mild from mild for positive predictions.

## Performance

At threshold `0.50`:

| Metric | Validation | Test audit |
| --- | ---: | ---: |
| ROC-AUC | 0.993 | 0.989 |
| Sensitivity | 97.66% | 96.82% |
| Beta | 2.34% | 3.18% |
| Specificity | 95.21% | 92.29% |
| Alpha | 4.79% | 7.71% |

## Training Data

- 6,336 labeled JPEG images across three classes.
- Fixed 70/15/15 train-validation-test split.
- No augmented images in the final MLP training set.
- No available patient identifiers or acquisition metadata.

## Ethical and Safety Considerations

- A false negative may delay further assessment; this motivates beta minimization.
- A false positive may cause anxiety and unnecessary follow-up; alpha remains a monitored secondary cost.
- Dataset labels can encode source-specific preprocessing and selection bias.
- Performance may differ across demographic groups, scanner vendors, sites, and MRI sequences.
- The model has no mechanism for detecting out-of-distribution or low-quality scans.

## Required Controls Before Deployment Research

- Patient-grouped external validation.
- Prospective data-quality checks.
- Calibration assessment.
- Subgroup error analysis.
- Clinician review of failure cases.
- Defined abstention and human-escalation policy.
- Independent governance and medical-device review.
