import pandas as pd

# Physically non-negative quantities (precipitation, wind speed)
_NON_NEGATIVE_COLS = [
    "RR",     # precipitation (mm)
    "FFM",    # mean wind speed (m/s)
    "FF2M",   # mean wind speed 2 m (m/s)
    "FXY",    # max wind gust (m/s)
    "FXI",    # max instantaneous wind (m/s)
    "FXI2",   # max instantaneous wind 2 m (m/s)
    "FXI3S",  # max 3-second gust (m/s)
    "INST",   # sunshine duration (h)
    "GLOT",   # global radiation (J/cm²)
    "DIFT",   # diffuse radiation (J/cm²)
    "DIRT",   # direct radiation (J/cm²)
    "INFRART",# infrared radiation (J/cm²)
]

# Humidity is constrained to [0, 100]
_HUMIDITY_COLS = ["UN", "UX", "UM"]


def fix_negative_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Set physically impossible negative values to NaN."""
    total_fixed = 0
    for col in _NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna() & (df[col] < 0)
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = float("nan")
            total_fixed += n
            print(f"[FIXED] {col}: {n} negative value(s) → NaN")

    print(f"Total: {total_fixed} fix(es) — negative impossible values removed")
    return df, total_fixed


def fix_humidity_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clip humidity values outside [0, 100] to NaN."""
    total_fixed = 0
    for col in _HUMIDITY_COLS:
        if col not in df.columns:
            continue
        mask = df[col].notna() & ((df[col] < 0) | (df[col] > 100))
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = float("nan")
            total_fixed += n
            print(f"[FIXED] {col}: {n} out-of-range value(s) → NaN")

    print(f"Total: {total_fixed} fix(es) — humidity out-of-range values removed")
    return df, total_fixed


def fix_temperature_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Set TN to NaN when TN > TX (min temperature cannot exceed max temperature)."""
    if "TN" not in df.columns or "TX" not in df.columns:
        return df, 0
    mask = df["TN"].notna() & df["TX"].notna() & (df["TN"] > df["TX"])
    n = int(mask.sum())
    if n:
        df.loc[mask, "TN"] = float("nan")
        print(f"[FIXED] TN > TX: {n} row(s) — TN set to NaN")
    else:
        print("Temperature consistency OK (TN ≤ TX everywhere)")
    return df, n
