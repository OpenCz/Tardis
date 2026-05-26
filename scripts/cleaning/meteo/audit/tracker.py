import pandas as pd


class AuditReport:
    def __init__(self) -> None:
        self._entries: list[dict] = []

    # ------------------------------------------------------------------ #
    # Core primitives                                                      #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Recording helpers                                                    #
    # ------------------------------------------------------------------ #

    def record_initial_state(self, df: pd.DataFrame) -> None:
        initial_null_total = int(df.isnull().sum().sum())
        perfect_rows = df.dropna().shape[0]
        perfect_cols = df.dropna(axis=1).shape[1]
        nb_rows = df.shape[0]

        self.add_many(
            [
                {"metric": "initial_rows", "value": nb_rows, "category": "initial_state", "reason": "Raw dataset before any cleaning"},
                {"metric": "initial_columns", "value": df.shape[1], "category": "initial_state", "reason": "Raw dataset before any cleaning"},
                {"metric": "initial_null_total", "value": initial_null_total, "category": "initial_state", "reason": "Total missing values across all cells"},
                {"metric": "perfect_rows", "value": perfect_rows, "category": "initial_state", "reason": "Rows with zero NaN — no cleaning needed"},
                {"metric": "rows_lost_if_strict_dropna", "value": nb_rows - perfect_rows, "category": "initial_state", "reason": "Rows lost with a strict zero-NaN policy"},
                {"metric": "perfect_columns", "value": perfect_cols, "category": "initial_state", "reason": "Columns with zero NaN"},
                {"metric": "cols_lost_if_strict_dropna", "value": df.shape[1] - perfect_cols, "category": "initial_state", "reason": "Columns lost with a strict zero-NaN policy"},
            ]
        )
        for col in df.columns:
            self.add(f"initial_null__{col}", int(df[col].isnull().sum()), "initial_null_per_column", col)

    def record_quality_invalidation(self, n_invalidated: int, q_cols: list[str]) -> None:
        self.add("cells_invalidated_by_quality_flag", n_invalidated, "cleaning",
                 "Observation values set to NaN where Météo-France quality flag == 0 (suspicious)")
        self.add("quality_columns_dropped", len(q_cols), "cleaning",
                 "Q* columns removed after use — no longer needed downstream")

    def record_duplicates(self, nb_dup: int) -> None:
        self.add("rows_dropped_duplicates", int(nb_dup), "cleaning",
                 "Duplicate rows — same station-day observation counted more than once")

    def record_critical_nan(self, crit_dropped: int) -> None:
        self.add("rows_dropped_critical_nan", crit_dropped, "cleaning",
                 "Unusable for analysis — missing key identifiers (NUM_POSTE / AAAAMMJJ / NOM_USUEL)")

    def record_recovery_summary(self, df: pd.DataFrame, total_recovered: int) -> None:
        null_post_recovery = int(df.isnull().sum().sum())
        initial_null = self.get("initial_null_total")
        recovery_rate = round(total_recovered / max(initial_null, 1), 4)

        self.add_many(
            [
                {"metric": "values_recovered_interpolation", "value": total_recovered, "category": "data_recovery", "reason": "Cells filled by linear interpolation within the same station (limit=3 days)"},
                {"metric": "null_post_recovery", "value": null_post_recovery, "category": "recovery_summary", "reason": "Remaining nulls after interpolation (sensors absent / gaps too large)"},
                {"metric": "interpolation_recovery_rate", "value": recovery_rate, "category": "recovery_summary", "reason": "Recovered / initial_null_total — interpolation coverage of initial nulls"},
            ]
        )
        print(f"Nulls post recovery   : {null_post_recovery:,}")
        print(f"Recovered by interp.  : {total_recovered:,}")
        print(f"Coverage of init nulls: {recovery_rate:.1%}")

    def record_corrections(self, neg_fixed: int, hum_fixed: int, temp_fixed: int) -> None:
        self.add("values_fixed_negative", neg_fixed, "data_correction",
                 "Physically impossible negative values (precipitation, wind speed) → NaN")
        self.add("values_fixed_humidity_bounds", hum_fixed, "data_correction",
                 "Humidity values outside [0, 100] → NaN")
        self.add("values_fixed_temp_consistency", temp_fixed, "data_correction",
                 "TN > TX violation (min temperature exceeded max) — TN set to NaN")

    def record_pipeline_summary(
        self,
        df: pd.DataFrame,
        original_rows: int,
        original_col_count: int,
        cols_after_quality_drop: int,
    ) -> None:
        row_retention = round(len(df) / original_rows, 4) if original_rows else 0
        rows_lost_dedup = self.get("rows_dropped_duplicates")
        rows_lost_crit = self.get("rows_dropped_critical_nan")
        cols_added = df.shape[1] - cols_after_quality_drop

        self.add_many(
            [
                {"metric": "row_retention_rate", "value": row_retention, "category": "pipeline_summary", "reason": "Fraction of original rows kept (higher = more data preserved)"},
                {"metric": "pct_rows_lost_dedup", "value": round(rows_lost_dedup / max(original_rows, 1), 4), "category": "pipeline_summary", "reason": "Fraction of rows lost to exact deduplication"},
                {"metric": "pct_rows_lost_critical_nan", "value": round(rows_lost_crit / max(original_rows, 1), 4), "category": "pipeline_summary", "reason": "Fraction of rows lost to missing critical identifiers"},
                {"metric": "cols_added_feature_eng", "value": cols_added, "category": "pipeline_summary", "reason": "Net columns added by feature engineering (year, month, season, etc.)"},
            ]
        )

    def record_final_state(self, df: pd.DataFrame, original_rows: int) -> None:
        self.add_many(
            [
                {"metric": "final_rows", "value": len(df), "category": "final_state", "reason": "Rows after full cleaning pipeline"},
                {"metric": "final_columns", "value": len(df.columns), "category": "final_state", "reason": "Columns after cleaning + feature engineering"},
                {"metric": "final_null_total", "value": int(df.isnull().sum().sum()), "category": "final_state", "reason": "Remaining nulls — sensors absent at station / gaps too large for interpolation"},
                {"metric": "total_rows_removed", "value": original_rows - len(df), "category": "final_state", "reason": "Total rows removed across all cleaning steps"},
            ]
        )

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._entries, columns=["metric", "value", "category", "reason"]
        )

    def export(self, path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"Report saved: {len(df)} entries → {path}")
        print(df["category"].value_counts().to_string())
