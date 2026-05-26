import gc
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Column & dtype definitions
# ---------------------------------------------------------------------------

# Measurement columns read as float32 → saves ~50% vs pandas float64 default.
# NUM_POSTE and AAAAMMJJ are intentionally kept as "Int32" (nullable int) so
# that the 8-digit date value 20231231 isn't rounded by float32 (only 7 sig figs).
_FLOAT32_COLS: set[str] = {
    # Coordinates
    "LAT", "LON", "ALTI",
    # Wind — vent file (RR-T-Vent)
    "FF", "DD",
    "FXY", "DXY", "HXY",
    "FXI", "DXI", "HXI",
    "FXI2", "DXI2", "HXI2",
    "FXI3S", "DXI3S", "HXI3S",
    "FFM", "FF2M",
    # Temperature & precipitation — vent file
    "RR", "DRR",
    "TN", "HTN", "TX", "HTX", "TM", "TNTXM", "TAMPLI", "TNSOL", "TN50",
    "DG",
    # Pressure — param file (autres-paramètres)
    "PMERM", "PMERMIN", "PMER",
    # Humidity — param file
    "UN", "HUN", "UX", "HUX", "UM", "DHUMEC",
    # Radiation / solar — param file
    "INST", "GLOT", "DIFT", "DIRT", "INFRART", "UV", "UV_INDICEX", "SIGMA",
    # Snow / other continuous — param file
    "DHUMI40", "DHUMI80", "TSVM",
    "ETPMON", "ETPGRILLE", "ETPAZ", "ETPE", "GEOP",
    "ECOULEMENTM", "HNEIGEF", "NEIGETOTX", "NEIGETOT06",
}

_METEO_DTYPES: dict[str, str] = {col: "float32" for col in _FLOAT32_COLS}
_METEO_DTYPES["NUM_POSTE"] = "Int32"   # nullable int — no precision loss
_METEO_DTYPES["AAAAMMJJ"] = "Int32"    # nullable int — float32 can't hold 8-digit dates

# Columns to keep from each source file (Q* quality flags kept via lambda).
# ─ Vent file (RR-T-Vent): precipitation + temperature + wind
_VENT_KEEP: set[str] = {
    "NUM_POSTE", "AAAAMMJJ", "NOM_USUEL", "LAT", "LON", "ALTI",
    # Precipitation
    "RR", "DRR",
    # Temperature
    "TN", "HTN", "TX", "HTX", "TM", "TNTXM", "TAMPLI", "TNSOL", "TN50", "DG",
    # Wind
    "FF", "DD",
    "FXY", "DXY", "HXY",
    "FXI", "DXI", "HXI",
    "FXI2", "DXI2", "HXI2",
    "FXI3S", "DXI3S", "HXI3S",
    "STATUS_FXI3S", "STATUS_DXI3S",
    "FFM", "FF2M",
}

# ─ Param file (autres-paramètres): pressure, humidity, radiation, snow …
_PARAM_KEEP: set[str] = {
    "NUM_POSTE", "AAAAMMJJ",
    # Pressure
    "PMERM", "PMERMIN", "PMER",
    # Humidity
    "UN", "HUN", "UX", "HUX", "UM", "DHUMEC",
    # Radiation / solar
    "INST", "GLOT", "DIFT", "DIRT", "INFRART", "UV", "UV_INDICEX", "SIGMA",
    # Snow & hydrology
    "DHUMI40", "DHUMI80", "TSVM",
    "ETPMON", "ETPGRILLE", "ETPAZ", "ETPE", "GEOP",
    "ECOULEMENTM", "HNEIGEF", "NEIGETOTX", "NEIGETOT06",
    # Binary weather indicators
    "NEIG", "BROU", "ORAG", "GRESIL", "GRELE",
    "ROSEE", "VERGLAS", "SOLNEIGE", "GELEE",
}

# Regex to extract the département code from Météo-France filenames
# e.g. "Q_86_previous-1950-2024_RR-T-Vent.csv"  →  "86"
_DEPT_RE = re.compile(r"Q_([^_]+)_")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generic single-file loader (used by train pipeline)."""
    df = pd.read_csv(path, sep=";")
    original_file = df.copy()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, original_file


def get_paths(index_path: str) -> list[str]:
    """Read the index file and return absolute paths.

    Lines may be bare filenames (relative to the index file's directory)
    or already-absolute paths — both are handled.
    """
    base_dir = os.path.dirname(os.path.abspath(index_path))
    with open(index_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().splitlines() if line.strip()]
    return [
        line if os.path.isabs(line) else os.path.join(base_dir, line)
        for line in lines
    ]


def _dept_code(path: str) -> str | None:
    """Extract département code from a Météo-France CSV filename."""
    m = _DEPT_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def group_by_dept(
    vent_index_path: str,
    param_index_path: str,
) -> dict[str, tuple[list[str], list[str]]]:
    """Group vent and param file paths by département code.

    Returns
    -------
    dict mapping dept_code → (vent_paths, param_paths)
    Sorted by département code so processing is deterministic.
    """
    vent_paths = get_paths(vent_index_path)
    param_paths = get_paths(param_index_path)

    vent_by_dept: dict[str, list[str]] = {}
    for p in vent_paths:
        dept = _dept_code(p)
        if dept:
            vent_by_dept.setdefault(dept, []).append(p)

    param_by_dept: dict[str, list[str]] = {}
    for p in param_paths:
        dept = _dept_code(p)
        if dept:
            param_by_dept.setdefault(dept, []).append(p)

    all_depts = sorted(set(vent_by_dept) | set(param_by_dept))
    return {
        dept: (vent_by_dept.get(dept, []), param_by_dept.get(dept, []))
        for dept in all_depts
    }


# ---------------------------------------------------------------------------
# Low-level file loader (used by the streaming pipeline)
# ---------------------------------------------------------------------------

def load_files(paths: list[str], is_vent: bool) -> pd.DataFrame:
    """Load a specific list of CSV files with column filtering and dtype optimisation.

    Parameters
    ----------
    paths   : list of absolute CSV paths to concatenate
    is_vent : True → apply vent column filter, False → param column filter
    """
    if not paths:
        return pd.DataFrame()

    keep = _VENT_KEEP if is_vent else _PARAM_KEEP
    usecols = lambda col: col in keep or col.startswith("Q")

    frames = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] File not found, skipped: {p}")
            continue
        frames.append(
            pd.read_csv(
                p,
                sep=";",
                usecols=usecols,
                dtype=_METEO_DTYPES,
                low_memory=False,
            )
        )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    del frames
    return df


# ---------------------------------------------------------------------------
# Bulk loaders (kept for backward-compat / notebook use — load EVERYTHING)
# ---------------------------------------------------------------------------

def load_vent(csv_vent_path: str) -> pd.DataFrame:
    """Load ALL wind observation files. Returns a single optimised DataFrame.

    ⚠ This loads the full dataset (~97 M rows) at once.
    Use the streaming Pipeline instead for large-memory environments.
    """
    paths = get_paths(csv_vent_path)
    df = load_files(paths, is_vent=True)
    gc.collect()
    print(
        f"Vent loaded  : {df.shape[0]:,} rows, {df.shape[1]} columns"
        f" from {len(paths)} file(s)"
    )
    return df


def load_parameters(csv_parameter_path: str) -> pd.DataFrame:
    """Load ALL weather-parameter observation files.

    ⚠ This loads the full dataset (~58 M rows) at once.
    Use the streaming Pipeline instead for large-memory environments.
    """
    paths = get_paths(csv_parameter_path)
    df = load_files(paths, is_vent=False)
    gc.collect()
    print(
        f"Params loaded: {df.shape[0]:,} rows, {df.shape[1]} columns"
        f" from {len(paths)} file(s)"
    )
    return df


def load_meteo(
    csv_vent_path: str,
    csv_parameter_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Public helper: load, merge, and return (merged_df, copy_for_comparison).

    ⚠ Loads everything at once — for small datasets or notebook exploration only.
    """
    from .merging import merge_meteo

    vent_df = load_vent(csv_vent_path)
    param_df = load_parameters(csv_parameter_path)
    df = merge_meteo(vent_df, param_df)
    del vent_df, param_df
    gc.collect()

    original = df.copy()
    print(f"Merged meteo : {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df, original
