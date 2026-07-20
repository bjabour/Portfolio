# Data Setup

## Source and Scope

This project uses three classes from the Kaggle [Alzheimer's Dataset (4 class of Images)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images):

| Class | Total images |
| --- | ---: |
| Non_Demented | 3,200 |
| Very_Mild_Demented | 2,240 |
| Mild_Demented | 896 |
| Total | 6,336 |

The original `Moderate_Demented` class is not included in this project. The training corpus is excluded from Git because it is large and remains subject to the source platform's terms. Three low-resolution presentation examples are the only retained source images; see [Image Attribution](image-attribution.md).

## Required Split Layout

The training scripts use nested split directories:

```text
early-alzheimer-mri-prediction/
|-- train/
|   `-- train/
|       |-- Non_Demented/
|       |-- Very_Mild_Demented/
|       `-- Mild_Demented/
|-- val/
|   `-- val/
|       |-- Non_Demented/
|       |-- Very_Mild_Demented/
|       `-- Mild_Demented/
`-- test/
    `-- test/
        |-- Non_Demented/
        |-- Very_Mild_Demented/
        `-- Mild_Demented/
```

Accepted extensions are `.jpg`, `.jpeg`, and `.png`.

## Expected Counts

| Split | Count | Percentage |
| --- | ---: | ---: |
| Train | 4,435 | 70% |
| Validation | 950 | 15% |
| Test | 951 | 15% |

`run_plan_d_solution.py` asserts these counts and the three-class label order:

1. `Non_Demented`
2. `Very_Mild_Demented`
3. `Mild_Demented`

## Image Loading

The deterministic and MLP path:

1. Opens each image with Pillow.
2. Converts it to one grayscale channel.
3. Resizes it to 128 by 128 pixels using bilinear interpolation.
4. Scales intensity to `[0, 1]`.

The pretrained CNN experiments convert grayscale images to RGB and apply ImageNet normalization.

## Data Integrity Checks

Before training, verify:

- No image is duplicated across train, validation, and test by filename or file hash.
- Every file can be opened and converted to grayscale.
- Class folder names match exactly.
- No augmented images appear in validation or test.
- If patient identifiers become available, all slices from one patient remain in one split.

## Known Data Limitations

- The files do not include patient IDs, scanner site, acquisition parameters, age, sex, or clinical scores.
- Multiple slices may originate from the same person; this cannot be ruled out from filenames alone.
- JPEG conversion can remove quantitative intensity meaning and acquisition metadata.
- A single slice cannot reliably represent whole-brain disease burden.
- The high axial example used in the presentation does not show the hippocampus well.

These limitations prevent the reported metrics from being interpreted as clinical diagnostic performance.
