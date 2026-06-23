from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
ASSIGNMENT_DIR = HERE.parent
TRAIN_PATH = ASSIGNMENT_DIR / "splines_train.csv"


def save_consumption_plots(df: pd.DataFrame) -> None:
    covariates = ["temperature", "humidity", "weekend"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, cov in zip(axes, covariates):
        if cov == "weekend":
            sns.stripplot(
                data=df,
                x=cov,
                y="consumption_kwh",
                jitter=0.18,
                alpha=0.65,
                ax=ax,
            )
            sns.pointplot(
                data=df,
                x=cov,
                y="consumption_kwh",
                color="black",
                errorbar="sd",
                markers="D",
                linestyles="",
                ax=ax,
            )
        else:
            sns.scatterplot(
                data=df,
                x=cov,
                y="consumption_kwh",
                alpha=0.72,
                edgecolor=None,
                ax=ax,
            )
            sns.regplot(
                data=df,
                x=cov,
                y="consumption_kwh",
                scatter=False,
                lowess=True,
                color="black",
                ax=ax,
            )
        ax.set_title(f"Consumption vs {cov}")
        ax.set_xlabel(cov)
        ax.set_ylabel("consumption_kwh")

    fig.suptitle("Consumption vs Covariates", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "consumption_vs_covariates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_high_demand_plots(df: pd.DataFrame) -> None:
    covariates = ["temperature", "humidity", "weekend"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, cov in zip(axes, covariates):
        if cov == "weekend":
            grouped = df.groupby(cov)["high_demand_alert"].mean().reset_index()
            sns.barplot(
                data=grouped,
                x=cov,
                y="high_demand_alert",
                ax=ax,
                color="#4C78A8",
            )
            ax.set_ylim(0, 1)
        else:
            sns.stripplot(
                data=df,
                x=cov,
                y="high_demand_alert",
                jitter=0.08,
                alpha=0.45,
                ax=ax,
            )
            sns.regplot(
                data=df,
                x=cov,
                y="high_demand_alert",
                scatter=False,
                logistic=True,
                ci=None,
                color="black",
                ax=ax,
            )
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"High demand vs {cov}")
        ax.set_xlabel(cov)
        ax.set_ylabel("high_demand_alert")

    fig.suptitle("High-Demand Alert vs Covariates", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "high_demand_vs_covariates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_high_demand_rate_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, cov, bins in [
        (axes[0], "temperature", 10),
        (axes[1], "humidity", 10),
    ]:
        work = df[[cov, "high_demand_alert"]].copy()
        work["bin"] = pd.qcut(work[cov], q=bins, duplicates="drop")
        grouped = (
            work.groupby("bin", observed=True)
            .agg(
                alert_rate=("high_demand_alert", "mean"),
                n=("high_demand_alert", "size"),
                x_mid=(cov, "mean"),
            )
            .reset_index()
        )

        ax.plot(
            grouped["x_mid"],
            grouped["alert_rate"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color="#1F77B4",
        )
        ax.fill_between(
            grouped["x_mid"],
            grouped["alert_rate"],
            alpha=0.16,
            color="#1F77B4",
        )
        for _, row in grouped.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["x_mid"], row["alert_rate"]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
            )
        ax.set_title(f"Binned alert rate vs {cov}")
        ax.set_xlabel(cov)
        ax.set_ylabel("mean high_demand_alert")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, alpha=0.28)

    weekend_rate = (
        df.groupby("weekend")["high_demand_alert"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "alert_rate", "count": "n"})
    )
    bars = axes[2].bar(
        weekend_rate["weekend"].astype(str),
        weekend_rate["alert_rate"],
        color=["#59A14F", "#F28E2B"],
        width=0.58,
    )
    for bar, (_, row) in zip(bars, weekend_rate.iterrows()):
        axes[2].annotate(
            f"{row['alert_rate']:.2f}\nn={int(row['n'])}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
        )
    axes[2].set_title("Alert rate by weekend")
    axes[2].set_xlabel("weekend")
    axes[2].set_ylabel("mean high_demand_alert")
    axes[2].set_ylim(0, 1.08)
    axes[2].grid(True, axis="y", alpha=0.28)

    fig.suptitle("High-Demand Alert Rate by Covariate Bins", fontsize=15, y=1.04)
    fig.tight_layout()
    fig.savefig(HERE / "high_demand_alert_rate_by_covariate_bins.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "temperature",
        "humidity",
        "weekend",
        "consumption_kwh",
        "high_demand_alert",
    ]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.82},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(HERE / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    df = pd.read_csv(TRAIN_PATH)
    save_consumption_plots(df)
    save_high_demand_plots(df)
    save_high_demand_rate_plots(df)
    save_correlation_heatmap(df)


if __name__ == "__main__":
    main()
