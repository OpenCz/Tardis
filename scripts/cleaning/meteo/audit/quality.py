from __future__ import annotations

import pandas as pd

from .tracker import AuditReport

_DEFAULT_DIST_COLS = [
    "RR",      # precipitation (mm)
    "TN",      # min temperature (°C)
    "TX",      # max temperature (°C)
    "TM",      # mean temperature (°C)
    "TAMPLI",  # diurnal temperature range (°C)
    "FFM",     # mean wind speed (m/s)
    "FXY",     # max wind gust (m/s)
    "PMERM",   # mean sea-level pressure (hPa)
    "UN",      # min relative humidity (%)
    "UX",      # max relative humidity (%)
    "INST",    # sunshine duration (h)
    "GLOT",    # global radiation (J/cm²)
    "temp_amplitude",  # engineered feature
]


def check_completeness(df: pd.DataFrame, report: AuditReport) -> float:
    total_cells = df.shape[0] * df.shape[1]
    null_cells = int(df.isnull().sum().sum())
    completeness = round(1 - null_cells / total_cells, 4)
    col_completeness = (1 - df.isnull().mean()).sort_values()

    report.add("overall_completeness", completeness, "quality_score",
               "Non-null cells / total cells (1.0 = fully complete)")
    for col in df.columns:
        rate = round(float(1 - df[col].isnull().mean()), 4)
        report.add(f"completeness__{col}", rate, "completeness_per_column", col)

    print(
        f"Overall completeness : {completeness:.2%}"
        f"  ({total_cells - null_cells:,} / {total_cells:,} cells non-null)"
    )
    print("\nLeast complete columns:")
    for col, rate in col_completeness.head(5).items():
        bar = "░" * int((1 - rate) * 30)
        print(f"  {rate:>6.1%}  {bar}  {col}")
    print("\nMost complete columns:")
    for col, rate in col_completeness.tail(5).items():
        print(f"  {rate:>6.1%}  {col}")
    return completeness


def check_distributions(
    df: pd.DataFrame,
    report: AuditReport,
    dist_cols: list[str] | None = None,
) -> pd.DataFrame:
    if dist_cols is None:
        dist_cols = _DEFAULT_DIST_COLS

    stat_rows = []
    for col in dist_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        row = {
            "column": col,
            "n": len(s),
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "p25": round(float(s.quantile(0.25)), 3),
            "median": round(float(s.median()), 3),
            "p75": round(float(s.quantile(0.75)), 3),
            "max": round(float(s.max()), 3),
        }
        stat_rows.append(row)
        for stat, val in row.items():
            if stat == "column":
                continue
            report.add(f"dist__{col}__{stat}", val, "numeric_distribution", col)

    stats_df = pd.DataFrame(stat_rows).set_index("column") if stat_rows else pd.DataFrame()
    if not stats_df.empty:
        print(stats_df.to_string())
    return stats_df


def check_outliers(
    df: pd.DataFrame, report: AuditReport
) -> list[tuple[str, int, float]]:
    outlier_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in ("year", "month", "NUM_POSTE", "AAAAMMJJ")
    ]
    outlier_rows: list[tuple[str, int, float]] = []

    for col in outlier_cols:
        s = df[col].dropna()
        if len(s) < 10 or s.std() == 0:
            continue
        z = (s - s.mean()) / s.std()
        n_out = int((z.abs() > 3).sum())
        rate = round(n_out / len(df), 4)
        outlier_rows.append((col, n_out, rate))
        report.add(
            f"outlier_rate__{col}",
            rate,
            "outlier_analysis",
            f"{n_out} values |z|>3 out of {len(df)} rows",
        )

    outlier_rows.sort(key=lambda x: -x[2])
    print(f'{"Column":<40} {"N outliers":>10}  {"Rate":>7}')
    print("─" * 65)
    for col, n, rate in outlier_rows:
        bar = "█" * min(int(rate * 300), 30)
        print(f"{col:<40} {n:>10,}  {rate:>6.2%}  {bar}")
    return outlier_rows


def check_validity(df: pd.DataFrame, report: AuditReport) -> dict[str, int]:
    checks: dict[str, int] = {}

    # Temperature consistency
    if "TN" in df.columns and "TX" in df.columns:
        both = df[["TN", "TX"]].dropna()
        checks["tn_le_tx"] = int((both["TN"] <= both["TX"]).all())

    # Precipitation non-negative
    if "RR" in df.columns:
        checks["rr_non_negative"] = int((df["RR"].dropna() >= 0).all())

    # Wind speed non-negative
    for wcol in ["FFM", "FXY", "FXI"]:
        if wcol in df.columns:
            checks[f"{wcol.lower()}_non_negative"] = int((df[wcol].dropna() >= 0).all())

    # Humidity in [0, 100]
    for hcol in ["UN", "UX", "UM"]:
        if hcol in df.columns:
            checks[f"{hcol.lower()}_in_0_100"] = int(
                df[hcol].dropna().between(0, 100).all()
            )

    # Sane temperature range [-60, 60]
    if "TM" in df.columns:
        checks["tm_range_sane"] = int(
            (df["TM"].dropna().between(-60, 60)).all()
        )

    # Pressure range [900, 1100] hPa
    if "PMERM" in df.columns:
        checks["pmerm_range_sane"] = int(
            (df["PMERM"].dropna().between(900, 1100)).all()
        )

    # Date range
    if "date" in df.columns:
        checks["date_range_sane"] = int(
            (df["date"] >= "1950-01-01").all() and (df["date"] <= "2026-12-31").all()
        )

    # No fully null rows
    checks["no_fully_null_rows"] = int(df.isnull().all(axis=1).sum() == 0)

    all_passed = sum(checks.values())
    for k, v in checks.items():
        report.add(f"validity__{k}", v, "validity_checks", "1=pass  0=fail")
    report.add("validity_checks_passed", all_passed, "validity_checks",
               f"{all_passed}/{len(checks)} checks passed")
    report.add("validity_checks_total", len(checks), "validity_checks",
               "Total validity rules evaluated")

    print("Validity checks:")
    for k, v in checks.items():
        print(f'  {"✓" if v else "✗"} {k}')
    print(f"\n{all_passed}/{len(checks)} checks passed")
    return checks


def check_effectiveness(df: pd.DataFrame, report: AuditReport) -> dict:
    total_corrections = sum(
        report.get(m)
        for m in [
            "values_fixed_negative",
            "values_fixed_humidity_bounds",
            "values_fixed_temp_consistency",
        ]
    )
    total_recovered = report.get("values_recovered_interpolation")
    total_cells_final = df.shape[0] * df.shape[1]
    correction_rate = round(total_corrections / max(total_cells_final, 1), 6)
    initial_null = report.get("initial_null_total")
    recovery_vs_null = round(total_recovered / max(initial_null, 1), 4)

    report.add_many(
        [
            {"metric": "total_corrections", "value": total_corrections, "category": "pipeline_effectiveness", "reason": "Negative + humidity-bounds + temperature-consistency fixes"},
            {"metric": "total_recovered_cells", "value": total_recovered, "category": "pipeline_effectiveness", "reason": "Cells recovered by linear interpolation within station"},
            {"metric": "correction_rate_per_cell", "value": correction_rate, "category": "pipeline_effectiveness", "reason": "Corrections / total final cells — data noise density"},
            {"metric": "recovery_vs_initial_null", "value": recovery_vs_null, "category": "pipeline_effectiveness", "reason": "Recovered / initial null count — interpolation coverage"},
        ]
    )
    print(f"Total corrections : {total_corrections:,}")
    print(f"Total recovered   : {total_recovered:,}")
    print(f"Correction rate   : {correction_rate:.4%} of all cells")
    print(f"Recovery vs nulls : {recovery_vs_null:.1%} of initial nulls recovered by interpolation")
    return {
        "total_corrections": total_corrections,
        "total_recovered": total_recovered,
        "correction_rate": correction_rate,
        "recovery_vs_null": recovery_vs_null,
    }
