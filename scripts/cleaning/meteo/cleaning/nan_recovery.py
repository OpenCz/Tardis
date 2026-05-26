import pandas as pd

# Key continuous weather variables eligible for linear interpolation
_INTERPOLATE_COLS = [
    "RR",    # precipitation (mm)
    "TN",    # min temperature (°C)
    "TX",    # max temperature (°C)
    "TM",    # mean temperature (°C)
    "FFM",   # mean wind speed (m/s)
    "FF2M",  # mean wind speed at 2 m (m/s)
    "PMERM", # mean sea-level pressure (hPa)
    "UN",    # min relative humidity (%)
    "UX",    # max relative humidity (%)
    "UM",    # mean relative humidity (%)
    "INST",  # sunshine duration (h)
    "GLOT",  # global radiation (J/cm²)
]

_INTERPOLATION_LIMIT = 3  # max consecutive missing days to fill


def recover_weather_interpolation(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    limit: int = _INTERPOLATION_LIMIT,
) -> tuple[pd.DataFrame, int]:
    """Fill short NaN gaps within each station using linear interpolation.

    Only fills runs of up to `limit` consecutive missing values per station
    (NUM_POSTE) sorted by date.  Longer gaps and truly absent sensors remain NaN.
    """
    if cols is None:
        cols = _INTERPOLATE_COLS

    df = df.sort_values(["NUM_POSTE", "date"]).reset_index(drop=True)
    total_recovered = 0

    for col in cols:
        if col not in df.columns:
            continue
        before = int(df[col].isna().sum())
        df[col] = df.groupby("NUM_POSTE")[col].transform(
            lambda x: x.interpolate(method="linear", limit=limit, limit_direction="both")
        )
        after = int(df[col].isna().sum())
        recovered = before - after
        total_recovered += recovered
        if recovered:
            print(f"  {col}: {recovered:,} NaN(s) recovered by interpolation")

    print(f"Total recovered by interpolation: {total_recovered:,} cell(s)")
    return df, total_recovered
