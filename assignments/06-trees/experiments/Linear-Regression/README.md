# trees Task 1: Linear Regression vs CART

## Conclusion

The recommended linear model is **ordinary least squares using all 12
predictors**. It is selected because it is the simplest model within one
standard error of the best repeated nested-CV linear result.

On the exact official five folds:

- OLS MSE: 982.749
- CART depth-6 MSE: 2300.548
- OLS MSE reduction: 57.3%
- OLS RMSE: 31.349 kEUR
- CART RMSE: 47.964 kEUR

In six-repeat nested 5-fold CV, where CART depth and linear alternatives are
tuned only inside each outer training fold:

- OLS mean MSE: 997.033
- Tuned CART mean MSE: 2708.340
- Mean MSE reduction: 63.2%
- OLS beat CART in 6
  of 6 repeats

The best direct screening score was best_subset_min at
957.313 MSE, but tuned screening scores reuse the
same folds for selection and evaluation. Nested CV shows that subset
selection, ridge, PLS, Huber regression, quadratic Lasso, and additive
splines do not reliably improve on plain OLS.

## Important Assignment Point

This experiment does **not** replace the required scratch CART output in task
1. The assignment explicitly grades the scratch tree and its agreement with
the library tree. It shows that, as a predictive benchmark for these data, a
linear model is much more accurate than a single CART.

## Reproduce

Run:

```powershell
& ..\..\.venv-trees-tree-models\Scripts\python.exe .\linear_vs_cart_experiment.py
```

Selected final model: `ols_all`
