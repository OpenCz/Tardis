import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

_HEATMAP_COLS = [
    "Average delay of all trains at arrival",
    "Average delay of all trains at departure",
    "Average delay of late trains at arrival",
    "Number of trains delayed > 15min",
    "Number of trains delayed > 30min",
    "Number of trains delayed > 60min",
    "cancellation_rate",
    "punctuality_rate",
    "Average journey time",
]


def plot_delay_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(
        df["Average delay of all trains at arrival"].dropna(), kde=True, ax=axes[0]
    )
    axes[0].set_title("Distribution of average arrival delay")
    axes[0].set_xlabel("Delay (minutes)")
    sns.boxplot(y=df["Average delay of all trains at arrival"].dropna(), ax=axes[1])
    axes[1].set_title("Arrival delay — spread and outliers")
    plt.tight_layout()
    plt.show()


def plot_seasonal_delay(df: pd.DataFrame) -> None:
    sns.boxplot(
        x="season",
        y="Average delay of all trains at arrival",
        data=df,
        order=["winter", "spring", "summer", "autumn"],
    )
    plt.title("Arrival delay by season")
    plt.xlabel("Season")
    plt.ylabel("Delay (minutes)")
    plt.tight_layout()
    plt.show()


def plot_top_stations_by_delay(df: pd.DataFrame, n: int = 10) -> None:
    top_stations = (
        df.groupby("Departure station")["Average delay of all trains at arrival"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    top_stations.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top {n} departure stations by average arrival delay")
    ax.set_xlabel("Average delay (minutes)")
    plt.tight_layout()
    plt.show()


def plot_monthly_trend(df: pd.DataFrame) -> None:
    monthly = df.groupby(df["Date"].dt.to_period("M"))[
        "Average delay of all trains at arrival"
    ].mean()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly.index, monthly.values)
    ax.axhline(monthly.mean(), color="red", linestyle="--", label="Overall mean")
    ax.set_title("Monthly average arrival delay over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Delay (minutes)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_delay_categories(df: pd.DataFrame) -> None:
    delay_counts = df["delay_category"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(delay_counts.values, labels=delay_counts.index, autopct="%1.1f%%")
    axes[0].set_title("Delay category proportions")
    delay_counts.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Delay category counts")
    axes[1].set_xlabel("Category")
    axes[1].set_ylabel("Number of services")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    cols = [c for c in _HEATMAP_COLS if c in df.columns]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        df[cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlation matrix — key numeric features")
    plt.tight_layout()
    plt.show()


def plot_cancellation_by_service(df: pd.DataFrame) -> None:
    sns.boxplot(x="Service", y="cancellation_rate", data=df)
    plt.title("Cancellation rate by service type")
    plt.ylabel("Cancellation rate")
    plt.tight_layout()
    plt.show()


def run_all_eda(df: pd.DataFrame) -> None:
    plot_delay_distribution(df)
    plot_seasonal_delay(df)
    plot_top_stations_by_delay(df)
    plot_monthly_trend(df)
    plot_delay_categories(df)
    plot_correlation_matrix(df)
    plot_cancellation_by_service(df)
