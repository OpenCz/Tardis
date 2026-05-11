import pandas as pd

CRITICAL_COLS = [
    "Date",
    "Service",
    "Departure station",
    "Arrival station",
    "Number of scheduled trains",
]


def drop_comment_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    comment_cols = df.filter(regex="comments$").columns.tolist()
    df = df.drop(columns=comment_cols)
    print(f"Dropped {len(comment_cols)} column(s): {comment_cols}")
    print(f"Remaining: {df.shape[1]} columns")
    return df, comment_cols


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    nb_dup = df.duplicated().sum()
    if nb_dup > 0:
        df = df.drop_duplicates()
    else:
        print("No duplicates found.")
    return df, int(nb_dup)


def drop_critical_nan(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLS)
    return df, before - len(df)
