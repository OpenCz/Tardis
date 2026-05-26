import gc

import pandas as pd

_JOIN_KEYS = ["NUM_POSTE", "AAAAMMJJ"]
_SHARED_META = ["NOM_USUEL", "LAT", "LON", "ALTI"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outer_merge(
    vent_df: pd.DataFrame, param_slim: pd.DataFrame
) -> pd.DataFrame:
    """Single outer-merge + cleanup of _param suffix duplicates."""
    df = pd.merge(
        vent_df,
        param_slim,
        on=_JOIN_KEYS,
        how="outer",
        suffixes=("", "_param"),
    )
    leftover = [c for c in df.columns if c.endswith("_param")]
    if leftover:
        df = df.drop(columns=leftover)
    return df


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def merge_meteo(
    vent_df: pd.DataFrame,
    param_df: pd.DataFrame,
    chunked: bool = True,
) -> pd.DataFrame:
    """Outer-merge vent and autres-paramètres observations on station + date.

    Both files share metadata columns (NOM_USUEL, LAT, LON, ALTI) besides the
    join keys.  We keep the vent values for those shared columns and drop the
    duplicate parameter versions.

    Parameters
    ----------
    chunked : bool, default True
        If True, the merge is performed year by year so the temporary structures
        created by pd.merge are proportional to a single year of data rather
        than the full dataset.  Results are concatenated at the end.
        Set to False to revert to the original single-pass merge.
    """
    # Drop shared metadata from param to avoid _param duplicates in the result
    param_slim = param_df.drop(
        columns=[c for c in _SHARED_META if c in param_df.columns],
        errors="ignore",
    )

    if not chunked:
        df = _outer_merge(vent_df, param_slim)

    else:
        # Derive year from AAAAMMJJ (integer YYYYMMDD) for both frames.
        # Cast to int64 to avoid float32 precision loss (20231231 has 8 digits,
        # float32 only holds 7 significant digits accurately).
        vent_year = (vent_df["AAAAMMJJ"].dropna().astype("int64") // 10000)
        param_year = (param_slim["AAAAMMJJ"].dropna().astype("int64") // 10000)
        years = sorted(set(vent_year.unique()) | set(param_year.unique()))

        # Pre-compute per-row year for fast boolean indexing
        vent_year_full = vent_df["AAAAMMJJ"].fillna(0).astype("int64") // 10000
        param_year_full = param_slim["AAAAMMJJ"].fillna(0).astype("int64") // 10000

        chunks: list[pd.DataFrame] = []
        for year in years:
            v = vent_df[vent_year_full == year]
            p = param_slim[param_year_full == year]
            if v.empty and p.empty:
                continue
            chunks.append(_outer_merge(v, p))
            print(f"  year {year}: {len(v):,} vent + {len(p):,} param rows merged")

        df = pd.concat(chunks, ignore_index=True)
        del chunks, vent_year_full, param_year_full
        gc.collect()

    print(
        f"merge_meteo : {len(vent_df):,} vent rows × {len(param_df):,} param rows"
        f" → {len(df):,} merged rows, {df.shape[1]} columns"
    )
    return df
