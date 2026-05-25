import pandas as pd

_LABEL_COLS = ["NOM_USUEL"]


def normalize_labels(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Uppercase + strip station name labels, then store as 'category'.

    Using 'category' after normalization ensures:
    - All ~1 500 unique station names are stored exactly once.
    - Every row holds a cheap integer code instead of a repeated string.
    - The dtype is consistent whether NOM_USUEL was originally object, string,
      or already category (cast_string_columns runs before this step).
    """
    if cols is None:
        cols = _LABEL_COLS

    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]

    if missing:
        print(f"[normalize_labels] Missing columns skipped: {missing}")

    for col in present:
        # Convert to plain str first (works regardless of object/string/category),
        # normalize, then re-encode as category to keep the memory benefit.
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
            .astype("category")
        )

    for col in present:
        print(f"Unique {col}: {df[col].nunique()}")

    return df
