# Early Alzheimer Prediction from Brain MRI

Reproducible statistical-learning project for classifying three MRI image labels:

- `Non_Demented`
- `Very_Mild_Demented`
- `Mild_Demented`

The final recommendation is a two-stage multilayer perceptron (MLP). Stage 1 estimates whether an image belongs to either dementia class. Stage 2 assigns very-mild or mild severity. The operating threshold is chosen to prioritize a low missed-dementia rate while rejecting thresholds with an extreme false-positive rate.

> This is a research and course project, not a medical device or diagnostic system. Its outputs must not be used for patient care.

## Executive Result

The recommended threshold is `0.50`.

| Metric | Validation | Test audit |
| --- | ---: | ---: |
| ROC-AUC, dementia vs non-demented | 0.993 | 0.989 |
| Sensitivity | 97.66% | 96.82% |
| Beta, missed-dementia rate | 2.34% | 3.18% |
| Specificity | 95.21% | 92.29% |
| Alpha, false-positive rate | 4.79% | 7.71% |

The test confusion counts at this threshold are:

| | Predicted non-demented | Predicted dementia |
| --- | ---: | ---: |
| Actual non-demented | 443 | 37 |
| Actual dementia | 15 | 456 |

The zero-miss operating point was rejected because it raised test alpha to 68.13%. A lower-alpha alternative at threshold `0.748` reduced alpha to 5.83%, but increased beta to 4.25%.

## Validation Qualification

The reported test values are development estimates. The test set was originally evaluated after threshold/model experiments and was later reviewed while preparing the presentation. A final scientific or clinical claim requires a new untouched holdout, preferably with patient-level grouping and external-site validation.

See [Validation Protocol](docs/validation-protocol.md) for the complete boundary and limitations.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `Early-Alzheimer-Prediction.html` | Standalone presentation with all assets embedded. |
| `scripts/` | Experiment, analysis, presentation-runtime, and build scripts. |
| `config/` | Reproducible Python and presentation configuration. |
| `data/` | Data setup instructions; raw MRI images are intentionally excluded. |
| `presentation/` | Presentation HTML/CSS source, hosting metadata, and visual assets. |
| `results/tables/` | Compact CSV/JSON evidence retained from completed experiments. |
| `results/figures/` | ROC, threshold, LDA, and regional-occlusion figures. |
| `docs/` | Data setup, methodology, validation, model card, results, and domain knowledge. |

Large image datasets, virtual environments, cached CNN embeddings, and serialized model binaries are intentionally excluded.

## Quick Start

### 1. Create the environment

The recorded environment used Python 3.14.3 and the versions in `config/python/Pipfile.lock`.

```powershell
pip install pipenv
$env:PIPENV_PIPFILE = "$PWD\config\python\Pipfile"
pipenv sync
```

Alternatively:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r config/requirements.txt
```

### 2. Prepare the data

The scripts expect the exact split layout below at the project root:

```text
data/splits/train/train/{Non_Demented,Very_Mild_Demented,Mild_Demented}/
data/splits/val/val/{Non_Demented,Very_Mild_Demented,Mild_Demented}/
data/splits/test/test/{Non_Demented,Very_Mild_Demented,Mild_Demented}/
```

Expected counts are 4,435 training images, 950 validation images, and 951 test images. See [Data Setup](docs/data-setup.md).

### 3. Run the deterministic baseline

```powershell
pipenv run python scripts/run_plan_d_solution.py --stage deterministic --seed 42 --project-root .
```

### 4. Run the two-stage MLP experiment

```powershell
pipenv run python scripts/run_type2_constraint_experiments.py --seed 42 --mlp-max-iter 120 --project-root .
```

This script also attempts the small CNN experiment when PyTorch is available. The final recommendation is based on the two-stage MLP outputs written to `results/experiments/type2_constraints/`.

### 5. Regenerate interpretation figures

This step requires the saved `results/experiments/type2_constraints/models.joblib` produced by step 4.

```powershell
$env:MPLCONFIGDIR = "$PWD\.matplotlib"
pipenv run python scripts/presentation/generate_model_insight_plots.py
```

### 6. Rebuild the standalone presentation

From the project root, run:

```powershell
node scripts/presentation/export_standalone.mjs
node scripts/presentation/build.mjs
```

The standalone output is `Early-Alzheimer-Prediction.html` at the project root. The ignored deployable bundle is written to `results/presentation-build/dist/`.

## Documentation

- [Data Setup](docs/data-setup.md)
- [Methodology](docs/methodology.md)
- [Validation Protocol](docs/validation-protocol.md)
- [Results and Interpretation](docs/results.md)
- [Model Card](docs/model-card.md)
- [Reproducibility Checklist](docs/reproducibility.md)
- [MRI Domain Knowledge](docs/domain-knowledge/mri-indicators.md)
- [Image Attribution](docs/image-attribution.md)

## Data Attribution

The three retained classes and counts correspond to a subset of the commonly used [Alzheimer's Dataset (4 class of Images)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images), with the `Moderate_Demented` class excluded. Confirm the current dataset availability and terms before use. The training corpus is not redistributed; three low-resolution examples are retained only for the project presentation and are documented in [Image Attribution](docs/image-attribution.md).

## Main Limitations

- Images are isolated 2D JPEG slices, not complete 3D MRI studies.
- Patient identifiers are unavailable, so patient-level split independence cannot be verified.
- Slice orientation and acquisition sequence metadata are missing.
- Class labels are dataset labels, not independently adjudicated clinical diagnoses.
- Regional features are coarse rectangles, not anatomical segmentations.
- Test results were observed during development and require confirmation on untouched data.
