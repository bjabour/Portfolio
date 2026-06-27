from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIR / "real_estate_risk_train.csv"
TEST_PATH = PROJECT_DIR / "real_estate_risk_test.csv"

ROW_INDEX_COLUMN = "source_row_index"
REMODEL_AGE_COLUMN = "years_since_remodel_2011"
REFERENCE_YEAR = 2011


def add_remodel_age(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    insert_at = result.columns.get_loc("YearRemodAdd") + 1
    remodel_age = REFERENCE_YEAR - result["YearRemodAdd"]
    result.insert(insert_at, REMODEL_AGE_COLUMN, remodel_age)
    return result


def find_iqr_outliers(
    data: pd.DataFrame, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if data[column].nunique(dropna=True) > 2
    ]
    detail_records = []
    summary_records = []

    for column in numeric_columns:
        values = data[column]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (values < lower_bound) | (values > upper_bound)
        flagged = data.loc[mask, column]

        summary_records.append(
            {
                "dataset": dataset_name,
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": int(mask.sum()),
                "outlier_percent": 100 * mask.mean(),
            }
        )

        for row_index, value in flagged.items():
            detail_records.append(
                {
                    "dataset": dataset_name,
                    ROW_INDEX_COLUMN: row_index,
                    "csv_row_number": row_index + 2,
                    "feature": column,
                    "value": value,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "direction": "low" if value < lower_bound else "high",
                }
            )

    detail_columns = [
        "dataset",
        ROW_INDEX_COLUMN,
        "csv_row_number",
        "feature",
        "value",
        "q1",
        "q3",
        "iqr",
        "lower_bound",
        "upper_bound",
        "direction",
    ]
    details = pd.DataFrame(detail_records, columns=detail_columns)
    summary = pd.DataFrame(summary_records)
    return details, summary


def write_notes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_details: pd.DataFrame,
    test_details: pd.DataFrame,
) -> None:
    train_unique_rows = train_details[ROW_INDEX_COLUMN].nunique()
    test_unique_rows = test_details[ROW_INDEX_COLUMN].nunique()

    notes = f"""# trees Data Cleaning

## Changes

- Added `{REMODEL_AGE_COLUMN} = {REFERENCE_YEAR} - YearRemodAdd`.
- Preserved every source row; no observations were removed or modified.
- Added `{ROW_INDEX_COLUMN}` to the augmented datasets for traceability.

## Potential Outlier Rule

For each numeric variable, values below `Q1 - 1.5 * IQR` or above
`Q3 + 1.5 * IQR` are flagged as potential outliers. A flag is diagnostic,
not proof that a value is erroneous. Large lots, houses, decks, or sale
prices can be legitimate observations and may be useful to tree models.
Binary variables are excluded because a rare class is not a numeric outlier.

## Results

- Train: {len(train)} rows; {len(train_details)} feature-level flags across
  {train_unique_rows} unique rows.
- Test: {len(test)} rows; {len(test_details)} feature-level flags across
  {test_unique_rows} unique rows.
- `source_row_index` is the zero-based pandas/CSV data-row index.
- `csv_row_number` is the physical CSV line number, including the header.

## Output Files

- `train_with_remodel_age.csv`
- `test_with_remodel_age.csv`
- `train_potential_outliers.csv`
- `test_potential_outliers.csv`
- `train_outlier_summary.csv`
- `test_outlier_summary.csv`
"""
    (OUTPUT_DIR / "data_cleaning_summary.md").write_text(notes, encoding="utf-8")


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_augmented = add_remodel_age(train)
    test_augmented = add_remodel_age(test)
    train_augmented.insert(0, ROW_INDEX_COLUMN, train_augmented.index)
    test_augmented.insert(0, ROW_INDEX_COLUMN, test_augmented.index)

    # Outlier detection is performed on source variables, excluding the
    # derived remodel-age feature and traceability index.
    train_details, train_summary = find_iqr_outliers(train, "train")
    test_details, test_summary = find_iqr_outliers(test, "test")

    train_augmented.to_csv(OUTPUT_DIR / "train_with_remodel_age.csv", index=False)
    test_augmented.to_csv(OUTPUT_DIR / "test_with_remodel_age.csv", index=False)
    train_details.to_csv(OUTPUT_DIR / "train_potential_outliers.csv", index=False)
    test_details.to_csv(OUTPUT_DIR / "test_potential_outliers.csv", index=False)
    train_summary.to_csv(OUTPUT_DIR / "train_outlier_summary.csv", index=False)
    test_summary.to_csv(OUTPUT_DIR / "test_outlier_summary.csv", index=False)

    write_notes(train, test, train_details, test_details)


if __name__ == "__main__":
    main()
