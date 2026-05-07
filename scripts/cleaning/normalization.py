import pandas as pd

_LABEL_COLS = ["Departure station", "Arrival station", "Service"]


def normalize_labels(
    df: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    if cols is None:
        cols = _LABEL_COLS
    for col in cols:
        df[col] = df[col].str.strip().str.upper()
    for col in cols:
        print(f"Unique {col}: {df[col].nunique()}")
    return df
