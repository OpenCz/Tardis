import pandas as pd

_SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year and month integer columns from the parsed date."""
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    print(f'Years  : {sorted(df["year"].unique().tolist())}')
    print(f'Months : {sorted(df["month"].unique().tolist())}')
    return df


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    df["season"] = df["month"].map(_SEASON_MAP)
    print("Season distribution:")
    print(df["season"].value_counts().to_string())
    return df


def add_temperature_amplitude(df: pd.DataFrame) -> pd.DataFrame:
    """Diurnal temperature range TX - TN."""
    if "TX" in df.columns and "TN" in df.columns:
        df["temp_amplitude"] = df["TX"] - df["TN"]
        print(
            f'temp_amplitude — min: {df["temp_amplitude"].min():.2f}'
            f'  mean: {df["temp_amplitude"].mean():.2f}'
            f'  max: {df["temp_amplitude"].max():.2f}'
        )
    else:
        print("[add_temperature_amplitude] TX or TN not found — skipped")
    return df


def add_wind_category(df: pd.DataFrame) -> pd.DataFrame:
    """Beaufort-inspired wind category from daily mean wind speed (FFM, m/s)."""
    if "FFM" not in df.columns:
        print("[add_wind_category] FFM not found — skipped")
        return df

    df["wind_category"] = pd.cut(
        df["FFM"],
        bins=[float("-inf"), 0.5, 3.3, 7.9, 13.8, float("inf")],
        labels=["calm", "light", "moderate", "strong", "storm"],
    )
    print("Wind category distribution:")
    print(df["wind_category"].value_counts().sort_index().to_string())
    return df


def add_precipitation_category(df: pd.DataFrame) -> pd.DataFrame:
    """Daily precipitation category from RR (mm)."""
    if "RR" not in df.columns:
        print("[add_precipitation_category] RR not found — skipped")
        return df

    df["precip_category"] = pd.cut(
        df["RR"],
        bins=[float("-inf"), 0.0, 2.0, 10.0, 30.0, float("inf")],
        labels=["dry", "trace", "light", "moderate", "heavy"],
    )
    print("Precipitation category distribution:")
    print(df["precip_category"].value_counts().sort_index().to_string())
    return df
