import pandas as pd

CRITICAL_COLS = ["NUM_POSTE", "AAAAMMJJ", "NOM_USUEL"]

# Columns that are never dropped by drop_sparse_columns, regardless of completeness.
# These are identifiers, coordinates, and derived time/season features that are
# structurally always present and always meaningful.
_PROTECTED_COLS: frozenset[str] = frozenset({
    "NUM_POSTE", "AAAAMMJJ", "NOM_USUEL",
    "LAT", "LON", "ALTI",
    "date", "year", "month", "season",
})


def drop_sparse_columns(
    df: pd.DataFrame,
    threshold: float = 0.60,
    protected: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns where fewer than `threshold` fraction of values are non-null.

    Columns that are always structurally present (identifiers, coordinates,
    date, and derived time/season features) are protected and never dropped.

    Parameters
    ----------
    df        : Input DataFrame.
    threshold : Minimum non-null fraction required to keep a column (default 0.60).
    protected : Override the default protected-column set.

    Returns
    -------
    (cleaned_df, list_of_dropped_column_names)
    """
    if protected is None:
        protected = _PROTECTED_COLS
    if len(df) == 0:
        return df, []

    completeness = df.notna().mean()
    to_drop = [
        col for col in df.columns
        if col not in protected and completeness[col] < threshold
    ]

    if to_drop:
        df = df.drop(columns=to_drop)
        preview = ", ".join(to_drop[:6]) + ("…" if len(to_drop) > 6 else "")
        print(
            f"drop_sparse_columns: removed {len(to_drop)} column(s) "
            f"(< {threshold:.0%} complete)\n"
            f"  Dropped: {preview}"
        )
    else:
        print(f"drop_sparse_columns: all columns ≥ {threshold:.0%} complete — nothing removed.")

    return df, to_drop


def invalidate_bad_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Set observation values to NaN wherever the corresponding Q* flag == 0.

    Météo-France quality codes:
        0 → suspicious / wrong data
        1 → valid data
        9 → unchecked / missing quality info
    """
    total_invalidated = 0
    for q_col in [c for c in df.columns if c.startswith("Q")]:
        obs_col = q_col[1:]  # strip leading "Q"
        if obs_col not in df.columns:
            continue
        bad = df[q_col] == 0
        n = int(bad.sum())
        if n:
            df.loc[bad, obs_col] = float("nan")
            total_invalidated += n

    print(f"Quality invalidation: {total_invalidated:,} cell(s) set to NaN (Q=0 → suspicious)")
    return df, total_invalidated


def drop_quality_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop all Météo-France quality-flag columns (names starting with Q)."""
    q_cols = [c for c in df.columns if c.startswith("Q")]
    df = df.drop(columns=q_cols)
    print(f"Dropped {len(q_cols)} quality-flag column(s): {q_cols[:5]}{'…' if len(q_cols) > 5 else ''}")
    print(f"Remaining: {df.shape[1]} columns")
    return df, q_cols


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    nb_dup = int(df.duplicated().sum())
    if nb_dup > 0:
        df = df.drop_duplicates()
        print(f"Removed {nb_dup} duplicate row(s).")
    else:
        print("No duplicates found.")
    return df, nb_dup


def drop_critical_nan(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.dropna(subset=[c for c in CRITICAL_COLS if c in df.columns])
    dropped = before - len(df)
    print(f"drop_critical_nan: {dropped} row(s) removed (missing {CRITICAL_COLS})")
    return df, dropped
