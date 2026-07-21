from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "real_estate_risk_train_year_diffs_2011.csv"
TEST_PATH = BASE_DIR / "real_estate_risk_test.csv"

FEATURES = [
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "years_since_built_2011",
    "years_since_remodel_2011",
    "TotalBsmtSF",
    "GarageArea",
    "BedroomAbvGr",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF",
]
REG_TARGET = "sale_price_keur"
CLF_TARGET = "distressed"


def replace_year_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    built_position = result.columns.get_loc("YearBuilt")
    remodel_position = result.columns.get_loc("YearRemodAdd")
    years_since_built = 2011 - result.pop("YearBuilt")
    years_since_remodel = 2011 - result.pop("YearRemodAdd")
    result.insert(built_position, "years_since_built_2011", years_since_built)
    result.insert(remodel_position, "years_since_remodel_2011", years_since_remodel)
    return result


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def write_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_summary = train[FEATURES + [REG_TARGET, CLF_TARGET]].describe().T
    train_summary["missing"] = train[FEATURES + [REG_TARGET, CLF_TARGET]].isna().sum()
    train_summary.to_csv(OUT_DIR / "train_summary_statistics.csv", index_label="feature")

    test_summary = test[FEATURES].describe().T
    test_summary["missing"] = test[FEATURES].isna().sum()
    test_summary.to_csv(OUT_DIR / "test_summary_statistics.csv", index_label="feature")

    train[FEATURES + [REG_TARGET, CLF_TARGET]].corr(numeric_only=True).to_csv(
        OUT_DIR / "train_correlation_matrix.csv"
    )


def plot_correlation_heatmap(train: pd.DataFrame) -> None:
    corr = train[FEATURES + [REG_TARGET, CLF_TARGET]].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
    )
    plt.title("trees Train Correlation Heatmap: 2011 Year Differences")
    save_fig("corr_heatmap.png")


def plot_target_distributions(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train[REG_TARGET], bins=25, kde=True, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Sale Price Distribution")
    axes[0].set_xlabel("sale_price_keur")

    distressed_rate = train[CLF_TARGET].mean()
    sns.countplot(data=train, x=CLF_TARGET, ax=axes[1], color="#F58518")
    axes[1].set_title(f"Distressed Class Balance (rate={distressed_rate:.3f})")
    axes[1].set_xlabel("distressed")
    axes[1].set_ylabel("count")
    save_fig("target_distributions.png")


def plot_regression_relationships(train: pd.DataFrame) -> None:
    corr_to_price = (
        train[FEATURES + [REG_TARGET]]
        .corr(numeric_only=True)[REG_TARGET]
        .drop(REG_TARGET)
        .abs()
        .sort_values(ascending=False)
    )
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    for ax, feature in zip(axes.ravel(), FEATURES):
        sns.regplot(
            data=train,
            x=feature,
            y=REG_TARGET,
            ax=ax,
            scatter_kws={"alpha": 0.55, "s": 22},
            line_kws={"color": "#E45756"},
        )
        ax.set_title(f"{feature} vs sale price")
    fig.suptitle("All Predictors vs Sale Price", fontsize=16, y=1.01)
    save_fig("top_price_relationships.png")

    plt.figure(figsize=(9, 5))
    corr_to_price.sort_values().plot(kind="barh", color="#4C78A8")
    plt.title("Absolute Correlation with Sale Price")
    plt.xlabel("|Pearson correlation|")
    save_fig("sale_price_feature_correlations.png")


def plot_classification_relationships(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    for ax, feature in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=train, x=CLF_TARGET, y=feature, ax=ax, color="#72B7B2")
        sns.stripplot(
            data=train,
            x=CLF_TARGET,
            y=feature,
            ax=ax,
            color="black",
            size=2,
            alpha=0.35,
        )
        ax.set_title(f"{feature} by distressed")
    fig.suptitle("All Predictors by Distressed Class", fontsize=16, y=1.01)
    save_fig("distressed_feature_boxplots.png")


def write_insights(train: pd.DataFrame, test: pd.DataFrame) -> None:
    corr_to_price = (
        train[FEATURES + [REG_TARGET]]
        .corr(numeric_only=True)[REG_TARGET]
        .drop(REG_TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    corr_to_distressed = (
        train[FEATURES + [CLF_TARGET]]
        .corr(numeric_only=True)[CLF_TARGET]
        .drop(CLF_TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    distressed_rate = train[CLF_TARGET].mean()
    price = train[REG_TARGET]
    lines = [
        "# trees Exploratory Data Analysis",
        "",
        "## Data Shape",
        "",
        f"- Training rows: {len(train)}",
        f"- Test rows: {len(test)}",
        f"- Numeric predictors: {len(FEATURES)}",
        f"- Missing values in train: {int(train.isna().sum().sum())}",
        f"- Missing values in test: {int(test.isna().sum().sum())}",
        "- `YearBuilt` and `YearRemodAdd` are replaced by their differences from 2011.",
        "",
        "## Target Summary",
        "",
        f"- Mean sale price: {price.mean():.3f} kEUR",
        f"- Median sale price: {price.median():.3f} kEUR",
        f"- Sale price range: {price.min():.3f} to {price.max():.3f} kEUR",
        f"- Distressed prevalence: {distressed_rate:.3%} ({int(train[CLF_TARGET].sum())}/{len(train)})",
        "",
        "## Strongest Sale-Price Correlations",
        "",
    ]
    lines.extend(f"- {name}: {value:.3f}" for name, value in corr_to_price.head(6).items())
    lines.extend(
        [
            "",
            "## Strongest Distressed Correlations",
            "",
        ]
    )
    lines.extend(f"- {name}: {value:.3f}" for name, value in corr_to_distressed.head(6).items())
    lines.extend(
        [
            "",
            "## Practical Modeling Notes",
            "",
            "- Sale price is continuous and moderately spread, so MSE-focused tree tuning should be stable enough with 5-fold CV.",
            "- The distressed label is imbalanced; probability models should use class weighting/resampling and calibration checks rather than optimizing accuracy.",
            "- Several size and quality variables are correlated, so tree ensembles should help capture interactions without requiring manual feature engineering.",
            "- The test file has the same 12 predictor columns as train and no outcomes, matching the required prediction-only workflow.",
            "",
            "## Generated Files",
            "",
            "- `train_summary_statistics.csv`",
            "- `test_summary_statistics.csv`",
            "- `train_correlation_matrix.csv`",
            "- `corr_heatmap.png`",
            "- `target_distributions.png`",
            "- `top_price_relationships.png`",
            "- `sale_price_feature_correlations.png`",
            "- `distressed_feature_boxplots.png`",
        ]
    )
    (OUT_DIR / "eda_insights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    train = pd.read_csv(TRAIN_PATH)
    test = replace_year_columns(pd.read_csv(TEST_PATH))

    write_summary(train, test)
    plot_correlation_heatmap(train)
    plot_target_distributions(train)
    plot_regression_relationships(train)
    plot_classification_relationships(train)
    write_insights(train, test)


if __name__ == "__main__":
    main()
