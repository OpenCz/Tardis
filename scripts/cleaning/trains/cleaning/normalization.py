import pandas as pd

_LABEL_COLS = ["Departure station", "Arrival station", "Service"]


def normalize_labels(df, cols=None):
    if cols is None:
        cols = _LABEL_COLS

    missing = [c for c in cols if c not in df.columns]
    present = [c for c in cols if c in df.columns]

    if missing:
        print(f"[normalize_labels] Missing columns skipped: {missing}")

    df = df.copy()
    for col in present:
        df[col] = df[col].astype("string").str.strip().str.upper()

    return df
