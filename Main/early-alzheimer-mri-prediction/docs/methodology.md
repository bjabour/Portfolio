# Methodology

## Prediction Tasks

The original label is three-class:

$$
Y \in \{0,1,2\}
=
\{\text{non-demented},\text{very mild},\text{mild}\}.
$$

The final model separates detection from severity:

$$
Y_{\text{binary}}
=
\begin{cases}
0, & Y = 0 \\
1, & Y \in \{1,2\}.
\end{cases}
$$

Stage 1 predicts $\Pr(Y_{\text{binary}}=1 \mid X)$. Stage 2 predicts class 1 versus class 2 only for images assigned to the dementia branch.

## Error Definitions

Alpha is the false-positive rate among non-demented images:

$$
\alpha
=
\frac{\mathrm{FP}}{\mathrm{TN}+\mathrm{FP}}.
$$

Beta is the false-negative rate among dementia images:

$$
\beta
=
\frac{\mathrm{FN}}{\mathrm{TP}+\mathrm{FN}}
=
1-\mathrm{sensitivity}.
$$

The project objective is to minimize $\beta$ while rejecting thresholds that produce an operationally extreme $\alpha$.

## Deterministic Feature Matrix

Each 128 by 128 grayscale image is transformed into 1,110 numeric features:

### Downsampled pixels

- Average pooling from 128 by 128 to 32 by 32.
- Flattening produces 1,024 spatial intensity features.

### Global summaries

- Mean and standard deviation.
- Foreground-area proxy and foreground intensity.
- Horizontal and vertical edge magnitude and edge density.
- Left-right asymmetry mean and standard deviation.
- Intensity percentiles from 5% through 95%.

### Coarse regional summaries

For central, upper, lower, left-lower, and right-lower rectangular regions, the script calculates:

- Mean intensity.
- Standard deviation.
- Dark-pixel fraction.
- Bright-pixel fraction.

### Grid summaries

A 4 by 4 grid contributes regional means, standard deviations, and dark-pixel fractions.

These are deterministic image summaries. They are not anatomical segmentations.

## Train-Only Preprocessing

The `StandardScaler` and PCA are fitted only on training features. Validation and test matrices are transformed with those fitted objects.

The MLP uses 120 principal components. PCA reduces redundancy and limits the number of trainable weights, but it also mixes spatial and handcrafted variables; a single PCA coordinate does not correspond to one anatomical structure.

## Two-Stage MLP

### Dementia detector

- Hidden layers: `(128, 64)`.
- Activation: ReLU.
- Optimizer: Adam.
- Batch size: 64.
- Initial learning rate: 0.0008.
- L2 penalty: 0.0007.
- Maximum iterations: 120.
- No-improvement window: 15 iterations.
- Class-balanced sample weights with an additional factor of 3 for dementia images.

### Severity classifier

- Trained only on `Very_Mild_Demented` and `Mild_Demented` training rows.
- Hidden layers: `(96, 48)`.
- Activation: ReLU.
- Optimizer: Adam.
- Batch size: 64.
- Initial learning rate: 0.0008.
- L2 penalty: 0.001.
- Class-balanced sample weights.

### Threshold rule

For dementia score $s(X)$ and threshold $t$:

$$
\widehat{Y}_{\text{binary}}
=
\mathbb{1}\{s(X) \ge t\}.
$$

If the result is positive, the severity MLP chooses between very mild and mild.

The recommended operating point is $t=0.50$. It is not the point with absolute minimum beta: a near-zero threshold produces zero misses but an unacceptable alpha of 68.13% on the test audit.

## Other Models Explored

The project also evaluated:

- Most-frequent baseline.
- Balanced multinomial logistic regression.
- Balanced linear SVM.
- RBF-SVM on deterministic PCA features.
- Multiclass and two-stage MLPs.
- Small CNNs trained from scratch.
- Frozen ResNet-18 and MobileNetV3 embeddings with deterministic heads.
- Pretrained CNN fine-tuning experiments.

The final presentation focuses on the MLP because it achieved the lowest beta among operating points with non-extreme alpha.

## Interpretation Analyses

### LDA projection

Linear discriminant analysis is fitted on the training PCA matrix and used only to visualize validation cases. It is not the final classifier. Validation balanced accuracy is 56.1%, illustrating substantial three-class overlap under a linear boundary.

### Regional occlusion

For each coarse region, the validation image patch is replaced with the training-mean patch. Features and MLP scores are recalculated. The reported importance is the decrease in validation ROC-AUC relative to the unmodified images, with paired bootstrap intervals.

Occlusion measures sensitivity of this fitted model to an artificial intervention. It does not identify a causal disease site.
