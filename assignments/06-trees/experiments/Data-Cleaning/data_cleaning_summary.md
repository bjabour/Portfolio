# trees Data Cleaning

## Changes

- Added `years_since_remodel_2011 = 2011 - YearRemodAdd`.
- Preserved every source row; no observations were removed or modified.
- Added `source_row_index` to the augmented datasets for traceability.

## Potential Outlier Rule

For each numeric variable, values below `Q1 - 1.5 * IQR` or above
`Q3 + 1.5 * IQR` are flagged as potential outliers. A flag is diagnostic,
not proof that a value is erroneous. Large lots, houses, decks, or sale
prices can be legitimate observations and may be useful to tree models.
Binary variables are excluded because a rare class is not a numeric outlier.

## Results

- Train: 300 rows; 122 feature-level flags across
  87 unique rows.
- Test: 200 rows; 65 feature-level flags across
  48 unique rows.
- `source_row_index` is the zero-based pandas/CSV data-row index.
- `csv_row_number` is the physical CSV line number, including the header.

## Output Files

- `train_with_remodel_age.csv`
- `test_with_remodel_age.csv`
- `train_potential_outliers.csv`
- `test_potential_outliers.csv`
- `train_outlier_summary.csv`
- `test_outlier_summary.csv`
