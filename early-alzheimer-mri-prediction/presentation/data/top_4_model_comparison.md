# Top Four Model Approaches

Type I error: a non-demented MRI is flagged as dementia. Type II error: a dementia MRI is missed.

| Model | Operating point | Test Type I | Test Type II | Sensitivity | Specificity | Cap met? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Two-stage MLP | No limit | 68.13% | 0.00% | 100.00% | 31.88% | N/A |
| Two-stage MLP | <= 5% | 7.71% | 3.18% | 96.82% | 92.29% | No |
| Two-stage MLP | <= 2.5% | 5.83% | 4.25% | 95.75% | 94.17% | No |
| Small weighted CNN | No limit | 91.67% | 0.42% | 99.58% | 8.33% | N/A |
| Small weighted CNN | <= 5% | 3.75% | 88.11% | 11.89% | 96.25% | Yes |
| Small weighted CNN | <= 2.5% | 2.50% | 90.02% | 9.98% | 97.50% | Yes |
| Pretrained CNN hybrid, nominal thresholds | No limit | 27.71% | 1.49% | 98.51% | 72.29% | N/A |
| Pretrained CNN hybrid, nominal thresholds | <= 5% | 6.67% | 5.94% | 94.06% | 93.33% | No |
| Pretrained CNN hybrid, nominal thresholds | <= 2.5% | 3.33% | 10.83% | 89.17% | 96.67% | No |
| Pretrained CNN hybrid, 99% guard | No limit | 27.71% | 1.49% | 98.51% | 72.29% | N/A |
| Pretrained CNN hybrid, 99% guard | <= 5% | 4.79% | 9.77% | 90.23% | 95.21% | Yes |
| Pretrained CNN hybrid, 99% guard | <= 2.5% | 1.88% | 18.26% | 81.74% | 98.13% | Yes |

The 99% guard is the recommended constrained CNN policy because it is the only tested CNN approach that meets both strict Type I limits on the test split. The MLP has lower Type II error, but exceeds both strict test Type I limits.

These test metrics were observed during iterative development. Use an untouched holdout set or nested cross-validation for any final performance claim.
