from __future__ import annotations

import pandas as pd

from scripts.audit.tracker import AuditReport

_DEFAULT_DIST_COLS = [
    "Number of scheduled trains",
    "Average delay of all trains at arrival",
    "Average delay of all trains at departure",
    "Average delay of late trains at arrival",
    "Average delay of late trains at departure",
    "cancellation_rate",
    "punctuality_rate",
    "Average journey time",
    "Pct delay due to external causes",
    "Pct delay due to infrastructure",
    "Pct delay due to traffic management",
    "Pct delay due to rolling stock",
]


def check_completeness(df: pd.DataFrame, report: AuditReport) -> float:
    total_cells = df.shape[0] * df.shape[1]
    null_cells = int(df.isnull().sum().sum())
    completeness = round(1 - null_cells / total_cells, 4)
    col_completeness = (1 - df.isnull().mean()).sort_values()

    report.add("overall_completeness", completeness, "quality_score", "Non-null cells / total cells (1.0 = fully complete)")
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

    stats_df = pd.DataFrame(stat_rows).set_index("column")
    print(stats_df.to_string())
    return stats_df


def check_outliers(
    df: pd.DataFrame, report: AuditReport
) -> list[tuple[str, int, float]]:
    outlier_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in ("year", "month")
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
    print(f'{"Column":<58} {"N outliers":>10}  {"Rate":>7}')
    print("─" * 82)
    for col, n, rate in outlier_rows:
        bar = "█" * min(int(rate * 300), 30)
        print(f"{col:<58} {n:>10,}  {rate:>6.2%}  {bar}")
    return outlier_rows


def check_validity(df: pd.DataFrame, report: AuditReport) -> dict[str, int]:
    checks: dict[str, int] = {}

    checks["cancellation_rate_in_0_1"] = int(
        df["cancellation_rate"].dropna().between(0, 1).all()
    )
    checks["punctuality_rate_in_0_1"] = int(
        df["punctuality_rate"].dropna().between(0, 1).all()
    )
    checks["no_negative_scheduled"] = int(
        (df["Number of scheduled trains"].dropna() >= 0).all()
    )

    hier_pairs = [
        ("Number of trains delayed > 15min", "Number of trains delayed > 30min"),
        ("Number of trains delayed > 30min", "Number of trains delayed > 60min"),
    ]
    for col_lo, col_hi in hier_pairs:
        both = df[[col_lo, col_hi]].dropna()
        key = (
            f'hierarchy_{col_lo.split(">")[1].strip().replace(" ", "")}'
            f'_{col_hi.split(">")[1].strip().replace(" ", "")}'
        )
        checks[key] = int((both.iloc[:, 0] >= both.iloc[:, 1]).all())

    pct_cols = [c for c in df.columns if c.startswith("Pct delay due to")]
    if pct_cols:
        row_pct_sum = df[pct_cols].sum(axis=1, skipna=True)
        checks["pct_delay_sum_le_105"] = int((row_pct_sum.dropna() <= 105).all())

    checks["date_range_sane"] = int(
        (df["Date"] >= "2015-01-01").all() and (df["Date"] <= "2026-12-31").all()
    )
    checks["service_only_2_values"] = int(
        df["Service"].dropna().isin(["NATIONAL", "INTERNATIONAL"]).all()
    )
    checks["no_fully_null_rows"] = int(df.isnull().all(axis=1).sum() == 0)

    all_passed = sum(checks.values())
    for k, v in checks.items():
        report.add(f"validity__{k}", v, "validity_checks", "1=pass  0=fail")
    report.add("validity_checks_passed", all_passed, "validity_checks", f"{all_passed}/{len(checks)} checks passed")
    report.add("validity_checks_total", len(checks), "validity_checks", "Total validity rules evaluated")

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
            "values_fixed_count_vs_scheduled",
            "values_fixed_delay_hierarchy",
        ]
    )
    total_recovered = sum(
        report.get(m)
        for m in [
            "values_filled_dep_late",
            "values_filled_dep_all",
            "values_filled_arr_late",
            "values_filled_arr_all",
        ]
    )
    total_cells_final = df.shape[0] * df.shape[1]
    correction_rate = round(total_corrections / total_cells_final, 6)
    initial_null = report.get("initial_null_total")
    recovery_vs_null = round(total_recovered / max(initial_null, 1), 4)

    report.add_many(
        [
            {"metric": "total_corrections", "value": total_corrections, "category": "pipeline_effectiveness", "reason": "Negatives + count overflows + hierarchy violations corrected"},
            {"metric": "total_recovered_cells", "value": total_recovered, "category": "pipeline_effectiveness", "reason": "Cells recovered by algebraic derivation (all 4 delay columns)"},
            {"metric": "correction_rate_per_cell", "value": correction_rate, "category": "pipeline_effectiveness", "reason": "Corrections / total final cells — data noise density"},
            {"metric": "recovery_vs_initial_null", "value": recovery_vs_null, "category": "pipeline_effectiveness", "reason": "Recovered / initial null count — algebraic coverage"},
        ]
    )
    print(f"Total corrections : {total_corrections:,}")
    print(f"Total recovered   : {total_recovered:,}")
    print(f"Correction rate   : {correction_rate:.4%} of all cells")
    print(
        f"Recovery vs nulls : {recovery_vs_null:.1%} of initial nulls recovered algebraically"
    )
    return {
        "total_corrections": total_corrections,
        "total_recovered": total_recovered,
        "correction_rate": correction_rate,
        "recovery_vs_null": recovery_vs_null,
    }
