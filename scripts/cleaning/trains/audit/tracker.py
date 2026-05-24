import pandas as pd


class AuditReport:
    def __init__(self) -> None:
        self._entries: list[dict] = []

    def add(self, metric: str, value, category: str, reason: str) -> None:
        self._entries.append(
            {"metric": metric, "value": value, "category": category, "reason": reason}
        )

    def add_many(self, entries: list[dict]) -> None:
        self._entries.extend(entries)

    def get(self, metric: str):
        for entry in self._entries:
            if entry["metric"] == metric:
                return entry["value"]
        raise KeyError(f"Metric not found: {metric!r}")

    def record_initial_state(self, df: pd.DataFrame) -> None:
        initial_null_total = df.isnull().sum().sum()
        perfect_rows = df.dropna().shape[0]
        perfect_cols = df.dropna(axis=1).shape[1]
        nb_rows = df.shape[0]

        self.add_many(
            [
                {"metric": "initial_rows", "value": df.shape[0], "category": "initial_state", "reason": "Raw dataset before any cleaning"},
                {"metric": "initial_columns", "value": df.shape[1], "category": "initial_state", "reason": "Raw dataset before any cleaning"},
                {"metric": "initial_null_total", "value": int(initial_null_total), "category": "initial_state", "reason": "Total missing values across all cells"},
                {"metric": "perfect_rows", "value": perfect_rows, "category": "initial_state", "reason": "Rows with zero NaN — no cleaning needed"},
                {"metric": "rows_lost_if_strict_dropna", "value": nb_rows - perfect_rows, "category": "initial_state", "reason": "Rows lost with a strict zero-NaN policy"},
                {"metric": "perfect_columns", "value": perfect_cols, "category": "initial_state", "reason": "Columns with zero NaN"},
                {"metric": "cols_lost_if_strict_dropna", "value": df.shape[1] - perfect_cols, "category": "initial_state", "reason": "Columns lost with a strict zero-NaN policy"},
            ]
        )
        for col in df.columns:
            self.add(f"initial_null__{col}", int(df[col].isnull().sum()), "initial_null_per_column", col)

    def record_comment_cols(self, comment_cols: list[str]) -> None:
        for col in comment_cols:
            self.add("column_dropped", col, "cleaning", "Irrelevant data — free text comment column")
        self.add("columns_dropped_total", len(comment_cols), "cleaning", "Irrelevant data — free text columns removed")

    def record_duplicates(self, nb_dup: int) -> None:
        self.add("rows_dropped_duplicates", int(nb_dup), "cleaning", "Duplicate rows — same observation counted more than once")

    def record_critical_nan(self, crit_dropped: int) -> None:
        self.add("rows_dropped_critical_nan", crit_dropped, "cleaning", "Unusable for analysis — missing key identifiers (CRITICAL_COLS)")

    def record_station_clustering(self, stats: dict, station_columns: list[str]) -> None:
        self.add_many(
            [
                {"metric": "stationclusters_total", "value": stats["total"], "category": "station_clustering", "reason": "Unique canonical station names after fuzzy deduplication"},
                {"metric": "stationclusters_with_merge", "value": stats["merged"], "category": "station_clustering", "reason": "Canonical names that absorbed at least one spelling variant"},
                {"metric": "station_variants_merged", "value": stats["variants"], "category": "station_clustering", "reason": "Non-canonical spellings unified into a canonical form"},
                {"metric": "station_fuzzy_threshold", "value": stats["threshold"], "category": "station_clustering", "reason": "token_sort_ratio threshold used for fuzzy matching"},
                {"metric": "station_columns_processed", "value": len(station_columns), "category": "station_clustering", "reason": "Number of station-type columns processed"},
            ]
        )

    def record_delay_recovery(self, dep_late: int, dep_all: int, arr_late: int, arr_all: int) -> None:
        self.add_many(
            [
                {"metric": "values_filled_dep_late", "value": dep_late, "category": "data_recovery", "reason": "Computed from average departure delay + scheduled/delayed counts"},
                {"metric": "values_filled_dep_all", "value": dep_all, "category": "data_recovery", "reason": "Computed from late-train departure delay + delayed/scheduled counts"},
                {"metric": "values_filled_arr_late", "value": arr_late, "category": "data_recovery", "reason": "Computed from average arrival delay + scheduled/delayed counts"},
                {"metric": "values_filled_arr_all", "value": arr_all, "category": "data_recovery", "reason": "Computed from late-train arrival delay + delayed/scheduled counts"},
            ]
        )

    def record_recovery_summary(self, df: pd.DataFrame) -> None:
        null_post_recovery = int(df.isnull().sum().sum())
        initial_null = self.get("initial_null_total")
        algebraic_total = sum(
            self.get(m)
            for m in [
                "values_filled_dep_late",
                "values_filled_dep_all",
                "values_filled_arr_late",
                "values_filled_arr_all",
            ]
        )
        algebraic_recovery_rate = round(algebraic_total / max(initial_null, 1), 4)
        self.add_many(
            [
                {"metric": "null_post_recovery", "value": null_post_recovery, "category": "recovery_summary", "reason": "Remaining nulls after algebraic recovery (steps 4.6–4.7)"},
                {"metric": "null_recovered_algebraic", "value": algebraic_total, "category": "recovery_summary", "reason": "Total cells recovered via algebraic derivation"},
                {"metric": "algebraic_recovery_rate", "value": algebraic_recovery_rate, "category": "recovery_summary", "reason": "Recovered / initial_null_total — algebraic coverage of initial nulls"},
            ]
        )
        print(f"Nulls post recovery   : {null_post_recovery}")
        print(f"Recovered algebraic   : {algebraic_total}")
        print(f"Algebraic coverage    : {algebraic_recovery_rate:.1%} of initial nulls")

    def record_pipeline_summary(
        self,
        df: pd.DataFrame,
        original_rows: int,
        original_col_count: int,
        cols_after_comment_drop: int,
    ) -> None:
        row_retention = round(len(df) / original_rows, 4)
        rows_lost_dedup = self.get("rows_dropped_duplicates")
        rows_lost_crit = self.get("rows_dropped_critical_nan")
        cols_added = df.shape[1] - cols_after_comment_drop

        self.add_many(
            [
                {"metric": "row_retention_rate", "value": row_retention, "category": "pipeline_summary", "reason": "Fraction of original rows kept (higher = more data preserved)"},
                {"metric": "pct_rows_lost_dedup", "value": round(rows_lost_dedup / original_rows, 4), "category": "pipeline_summary", "reason": "Fraction of rows lost to exact deduplication"},
                {"metric": "pct_rows_lost_critical_nan", "value": round(rows_lost_crit / original_rows, 4), "category": "pipeline_summary", "reason": "Fraction of rows lost to missing critical identifiers"},
                {"metric": "cols_added_feature_eng", "value": cols_added, "category": "pipeline_summary", "reason": "Net columns added by feature engineering (year, month, season, etc.)"},
            ]
        )

    def record_corrections(self, neg_fixed: int, overflow_fixed: int, hier_fixed: int) -> None:
        self.add("values_fixed_negative", neg_fixed, "data_correction", "Impossible negative count values — replaced by route-level median")
        self.add("values_fixed_count_vs_scheduled", overflow_fixed, "data_correction", "Counts exceeding scheduled trains — logically impossible")
        self.add("values_fixed_delay_hierarchy", hier_fixed, "data_correction", "Delay hierarchy violation — higher threshold exceeded lower threshold")

    def record_final_state(self, df: pd.DataFrame, original_rows: int) -> None:
        self.add_many(
            [
                {"metric": "final_rows", "value": len(df), "category": "final_state", "reason": "Rows after full cleaning pipeline"},
                {"metric": "final_columns", "value": len(df.columns), "category": "final_state", "reason": "Columns after cleaning + feature engineering"},
                {"metric": "final_null_total", "value": int(df.isnull().sum().sum()), "category": "final_state", "reason": "Intentionally kept null — non-derivable columns"},
                {"metric": "total_rows_removed", "value": original_rows - len(df), "category": "final_state", "reason": "Total rows removed across all cleaning steps"},
            ]
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._entries, columns=["metric", "value", "category", "reason"]
        )

    def export(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"Report saved: {len(df)} entries → {path}")
        print(df["category"].value_counts().to_string())
