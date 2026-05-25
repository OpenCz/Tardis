import pandas as pd

# Columns that must NOT be coerced to numeric
_TEXT_COLS = ["NOM_USUEL", "STATUS_FXI3S", "STATUS_DXI3S"]
# "AAAAMMJJ" is kept as int until parse_dates; date is the parsed column
_DATE_RAW_COL = "AAAAMMJJ"
_DATE_COL = "date"

# Columns to cast to low-cardinality "category" dtype (saves memory vs string)
_CAT_COLS = ["NOM_USUEL", "STATUS_FXI3S", "STATUS_DXI3S"]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse AAAAMMJJ (integer YYYYMMDD) into a proper datetime column 'date'."""
    # Int32 nullable integers render as "<NA>" when missing → coerce handles them
    df[_DATE_COL] = pd.to_datetime(
        df[_DATE_RAW_COL].astype(str).str.zfill(8), format="%Y%m%d", errors="coerce"
    )
    valid = df[_DATE_COL].notna().sum()
    print(f"parse_dates : {valid:,}/{len(df):,} rows parsed successfully")
    print(f"  dtype : {df[_DATE_COL].dtype}")
    print(f"  Range : {df[_DATE_COL].min().date()} → {df[_DATE_COL].max().date()}")
    return df


def convert_numerics(
    df: pd.DataFrame, text_cols: list[str] | None = None
) -> pd.DataFrame:
    """Coerce object columns that should be numeric.

    Columns already holding a numeric dtype (float32, float64, Int32 …) are
    skipped — this preserves the float32 types set at read time instead of
    silently up-casting them back to float64.
    """
    if text_cols is None:
        text_cols = _TEXT_COLS + [_DATE_RAW_COL, _DATE_COL]

    numeric_cols = [c for c in df.columns if c not in text_cols]
    converted = 0
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Already numeric (float32, Int32, float64 …) — leave untouched
            continue
        # Object column: clean European decimal comma then coerce
        df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        converted += 1

    still_object = [c for c in numeric_cols if c in df.columns and df[c].dtype == object]
    if still_object:
        print(f"WARNING — still object after conversion: {still_object}")
    else:
        print(
            f"convert_numerics: {converted} object column(s) coerced to numeric"
            f" ({len(numeric_cols) - converted} already-numeric column(s) left as-is)."
        )
    return df


def cast_string_columns(
    df: pd.DataFrame, str_cols: list[str] | None = None
) -> pd.DataFrame:
    """Cast low-cardinality text columns to 'category' dtype.

    'category' stores each unique value once and uses integer codes for every
    row — far more memory-efficient than 'object' or 'string' when cardinality
    is low (e.g. ~1 500 station names repeated millions of times).
    """
    if str_cols is None:
        str_cols = [c for c in _CAT_COLS if c in df.columns]
    if str_cols:
        for col in str_cols:
            df[col] = df[col].astype("category")
        print("Dtypes after cast:")
        print(df[str_cols].dtypes.to_string())
    return df
