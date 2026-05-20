import pandas as pd

_TEXT_COLS = ["Date", "Service", "Departure station", "Arrival station", "Nom_Gare", "Trigramme", "Segment(s) DRG", "Id_Gare"]

_STR_COLS = ["Service", "Departure station", "Arrival station", "Nom_Gare", "Trigramme", "Segment(s) DRG", "Id_Gare"]

def parse_dates(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], format="mixed")
    print(f"dtype : {df[col].dtype}")
    print(f"Range : {df[col].min().date()} → {df[col].max().date()}")
    return df


def convert_numerics(
    df: pd.DataFrame, text_cols: list[str] | None = None
) -> pd.DataFrame:
    if text_cols is None:
        text_cols = _TEXT_COLS
    numeric_cols = [c for c in df.columns if c not in text_cols]

    for col in numeric_cols:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().str.replace(",", ".")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    still_object = [c for c in numeric_cols if df[c].dtype == object]
    if still_object:
        print(f"WARNING — still object after conversion: {still_object}")
    else:
        print(f"All {len(numeric_cols)} numeric columns successfully converted.")
    return df


def cast_string_columns(
    df: pd.DataFrame, str_cols: list[str] | None = None
) -> pd.DataFrame:
    if str_cols is None:
        str_cols = _STR_COLS
        str_cols = [col for col in str_cols if col in df.columns]
    if str_cols:
        df[str_cols] = df[str_cols].astype("string")
        print("Dtypes after cast:")
        print(df[str_cols].dtypes.to_string())
    return df
