from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ASSIGNMENT_DIR / "trees_train_year_diffs_2011.csv"
OUTPUT_PATH = ASSIGNMENT_DIR / "trees_train_year_diffs_2011_corr_heatmap.png"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    correlation = data.corr(numeric_only=True)
    mask = np.triu(np.ones_like(correlation, dtype=bool))

    sns.set_theme(style="white")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap: Train Data with 2011 Year Differences")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
