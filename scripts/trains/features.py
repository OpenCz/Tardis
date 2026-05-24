import pandas as pd

_SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

_COL_SCHEDULED = "Number of scheduled trains"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    print(f'Years  : {sorted(df["year"].unique().tolist())}')
    print(f'Months : {sorted(df["month"].unique().tolist())}')
    return df


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df["season"] = df["month"].map(_SEASON_MAP)
    print("Season distribution:")
    print(df["season"].value_counts().to_string())
    return df


def add_delay_category(df: pd.DataFrame) -> pd.DataFrame:
    df["delay_category"] = pd.cut(
        df["Average delay of all trains at arrival"],
        bins=[float("-inf"), 0, 5, 15, 30, float("inf")],
        labels=["early", "on_time", "slight", "moderate", "severe"],
    )
    print("Delay category distribution:")
    print(df["delay_category"].value_counts().sort_index().to_string())
    return df


def add_cancellation_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["cancellation_rate"] = df["Number of cancelled trains"] / df[
        _COL_SCHEDULED
    ].replace(0, float("nan"))
    print(
        f'cancellation_rate — min: {df["cancellation_rate"].min():.4f}'
        f'  mean: {df["cancellation_rate"].mean():.4f}'
        f'  max: {df["cancellation_rate"].max():.4f}'
    )
    return df


def add_punctuality_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["punctuality_rate"] = 1 - (
        df["Number of trains delayed at arrival"] / df[_COL_SCHEDULED].replace(0, float("nan"))
    )
    print(
        f'punctuality_rate — min: {df["punctuality_rate"].min():.4f}'
        f'  mean: {df["punctuality_rate"].mean():.4f}'
        f'  max: {df["punctuality_rate"].max():.4f}'
    )
    return df
