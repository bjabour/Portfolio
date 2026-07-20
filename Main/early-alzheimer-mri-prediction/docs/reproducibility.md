# Reproducibility Checklist

## Environment

- [ ] Use Python 3.14.3 or a compatible supported version.
- [ ] Install `Pipfile.lock` with `pipenv sync`, or install `requirements.txt`.
- [ ] Record CPU/GPU, operating system, Python, PyTorch, and scikit-learn versions.
- [ ] Set `MPLCONFIGDIR` to a writable project directory when running headless plots.

## Data

- [ ] Restore the expected nested split folders.
- [ ] Verify split counts: 4,435 / 950 / 951.
- [ ] Verify class counts and exact folder names.
- [ ] Hash files and check for cross-split duplicates.
- [ ] Keep augmented data out of validation and test.
- [ ] Use patient-level grouping if identifiers become available.

## Training

- [ ] Set seed 42.
- [ ] Fit the scaler and PCA on train only.
- [ ] Verify finite features after preprocessing.
- [ ] Verify MLP probabilities remain within `[0, 1]`.
- [ ] Save model configuration, threshold sweep, and class order.

## Model Selection

- [ ] Compare candidate models on validation only.
- [ ] Select the threshold on validation only.
- [ ] Report alpha, beta, sensitivity, specificity, and confusion counts together.
- [ ] Reject the maximum-sensitivity point when alpha is operationally extreme.

## Final Audit

- [ ] Freeze preprocessing, model weights, and threshold.
- [ ] Evaluate the untouched test set once.
- [ ] Save prediction rows and confusion matrices.
- [ ] Record any later test-set inspection as development leakage.
- [ ] Obtain a new holdout before a final external claim.

## Presentation

- [ ] Regenerate LDA, ROC, threshold, and regional-occlusion plots from saved artifacts.
- [ ] Rebuild the standalone HTML.
- [ ] Verify all images load.
- [ ] Check desktop and mobile overflow.
- [ ] Confirm no diagnostic or causal claims are made.
