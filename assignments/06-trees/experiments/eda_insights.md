# trees Exploratory Data Analysis

## Data Shape

- Training rows: 300
- Test rows: 200
- Numeric predictors: 12
- Missing values in train: 0
- Missing values in test: 0
- `YearBuilt` and `YearRemodAdd` are replaced by their differences from 2011.

## Target Summary

- Mean sale price: 356.310 kEUR
- Median sale price: 340.211 kEUR
- Sale price range: 47.272 to 693.382 kEUR
- Distressed prevalence: 15.667% (47/300)

## Strongest Sale-Price Correlations

- OverallQual: 0.855
- GrLivArea: 0.733
- GarageArea: 0.731
- TotalBsmtSF: 0.682
- years_since_built_2011: -0.654
- years_since_remodel_2011: -0.593

## Strongest Distressed Correlations

- OverallQual: -0.280
- OverallCond: -0.255
- BedroomAbvGr: -0.246
- GarageArea: -0.238
- GrLivArea: -0.228
- Fireplaces: -0.178

## Practical Modeling Notes

- Sale price is continuous and moderately spread, so MSE-focused tree tuning should be stable enough with 5-fold CV.
- The distressed label is imbalanced; probability models should use class weighting/resampling and calibration checks rather than optimizing accuracy.
- Several size and quality variables are correlated, so tree ensembles should help capture interactions without requiring manual feature engineering.
- The test file has the same 12 predictor columns as train and no outcomes, matching the required prediction-only workflow.

## Generated Files

- `train_summary_statistics.csv`
- `test_summary_statistics.csv`
- `train_correlation_matrix.csv`
- `corr_heatmap.png`
- `target_distributions.png`
- `top_price_relationships.png`
- `sale_price_feature_correlations.png`
- `distressed_feature_boxplots.png`
