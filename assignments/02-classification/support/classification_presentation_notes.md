# Confidence Intervals and Classification Metrics

## Confidence Interval for Predicted Probability

The logistic model first predicts a **log-odds** value, called `eta`:

```text
eta = beta0 + beta1*x1 + beta2*x1^2 + beta3*(x2*x3)
```

Then we convert the log-odds into a probability using the sigmoid function:

```text
p = 1 / (1 + exp(-eta))
```

So `prob_class1` is the estimated probability that the observation belongs to class `y=1`.

The confidence interval is calculated on the **log-odds scale**, not directly on the probability scale, because logistic regression coefficients are approximately normal on the log-odds scale.

For each test observation:

```text
eta_hat = x beta_hat
SE(eta_hat) = sqrt(x Cov(beta_hat) x')
```

The 95% confidence interval on the log-odds scale is:

```text
eta_hat - 1.96 * SE(eta_hat)
eta_hat + 1.96 * SE(eta_hat)
```

Then both endpoints are transformed back to probabilities:

```text
ci_lower = sigmoid(eta_hat - 1.96 * SE)
ci_upper = sigmoid(eta_hat + 1.96 * SE)
```

This gives a probability interval between 0 and 1.

## Classification Metrics

### Accuracy

Accuracy is the fraction of all predictions that were correct.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

Precision asks: among the observations predicted as `y=1`, how many were actually `y=1`?

```text
Precision = TP / (TP + FP)
```

High precision means fewer false positives.

### Recall

Recall asks: among the observations that truly were `y=1`, how many did the model correctly catch?

```text
Recall = TP / (TP + FN)
```

High recall means fewer false negatives.

### F1 Score

F1 score combines precision and recall into one number.

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

It is useful when we want one score that balances false positives and false negatives.
