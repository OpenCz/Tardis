"""Streaming météo cleaning pipeline.

Memory model
------------
Instead of loading the full ~155 M-row dataset at once, the pipeline
processes one département at a time (≤ ~700 K rows each for the largest
files) and streams cleaned rows directly to the output CSV.

Peak RAM per iteration:
  vent_df  (one dept, Q cols not yet dropped) : ~50–200 MB
  param_df (one dept, Q cols not yet dropped) : ~30–120 MB
  merged_df (Q cols dropped before merge)     : ~30–150 MB
  ────────────────────────────────────────────────────────
  Total peak                                  : < 500 MB

Comparison with the all-at-once approach (float32):
  vent_all + param_all + merged               : ~70 GB → crash
"""

import gc
import os

from . import cleaning, features, loading, merging


class Pipeline:
    def __init__(
        self,
        data_vent_path: str,
        data_parameter_path: str,
        output_path: str = "data/processed/meteo/cleaned_meteo_dataset.csv",
        report_path: str = "data/processed/audit/cleaning_meteo_report.csv",
    ):
        self.data_vent_path = data_vent_path
        self.data_parameter_path = data_parameter_path
        self.output_path = output_path
        self.report_path = report_path

    # ------------------------------------------------------------------ #
    # Public entry-point                                                   #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Stream-clean all départements and write to a single output CSV.

        Returns a summary dict with aggregate counters.
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)

        dept_groups = loading.group_by_dept(
            self.data_vent_path, self.data_parameter_path
        )
        n_depts = len(dept_groups)
        print(f"Found {n_depts} département(s) to process.\n")

        # Aggregate counters
        total_input_rows = 0
        total_output_rows = 0
        total_invalidated = 0
        total_duplicates = 0
        total_crit_dropped = 0
        total_recovered = 0
        first_write = True

        for i, (dept, (vent_paths, param_paths)) in enumerate(dept_groups.items(), 1):
            print(f"[{i:>3}/{n_depts}] Département {dept} ─────────────────────")

            result = self._process_dept(vent_paths, param_paths)
            if result is None:
                print(f"  → skipped (no data)\n")
                continue

            df, stats = result
            total_input_rows += stats["input_rows"]
            total_invalidated += stats["invalidated"]
            total_duplicates += stats["duplicates"]
            total_crit_dropped += stats["crit_dropped"]
            total_recovered += stats["recovered"]
            total_output_rows += len(df)

            # Append to output CSV (header only on first write)
            df.to_csv(
                self.output_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False

            print(
                f"  → {stats['input_rows']:>8,} input rows"
                f"  |  {len(df):>8,} output rows"
                f"  |  {stats['recovered']:,} interpolated\n"
            )
            del df, result
            gc.collect()

        summary = {
            "total_input_rows": total_input_rows,
            "total_output_rows": total_output_rows,
            "total_invalidated_cells": total_invalidated,
            "total_duplicates_removed": total_duplicates,
            "total_critical_nan_removed": total_crit_dropped,
            "total_cells_recovered": total_recovered,
            "row_retention_rate": round(total_output_rows / max(total_input_rows, 1), 4),
            "output_path": self.output_path,
        }

        self._print_summary(summary)
        self._write_report(summary)
        return summary

    # ------------------------------------------------------------------ #
    # Per-département processing                                           #
    # ------------------------------------------------------------------ #

    def _process_dept(
        self,
        vent_paths: list[str],
        param_paths: list[str],
    ) -> tuple | None:
        """Full clean + engineer cycle for one département.

        Returns (cleaned_df, stats_dict) or None if no data.
        Key memory trick: Q columns are invalidated and dropped on each
        source frame BEFORE the merge — this halves the column count (and
        thus RAM) that enters pd.merge.
        """
        stats = {
            "input_rows": 0,
            "invalidated": 0,
            "duplicates": 0,
            "crit_dropped": 0,
            "recovered": 0,
        }

        # ── 1. Load vent ──────────────────────────────────────────────
        vent_df = loading.load_files(vent_paths, is_vent=True)
        if vent_df.empty:
            # Try to continue with param only if available
            if not param_paths:
                return None
        else:
            stats["input_rows"] += len(vent_df)
            # Pre-clean: invalidate Q flags and drop Q cols BEFORE merge
            vent_df, n_inv = cleaning.invalidate_bad_quality(vent_df)
            stats["invalidated"] += n_inv
            vent_df, _ = cleaning.drop_quality_columns(vent_df)

        # ── 2. Load param ─────────────────────────────────────────────
        param_df = loading.load_files(param_paths, is_vent=False)
        if not param_df.empty:
            stats["input_rows"] += len(param_df)
            param_df, n_inv = cleaning.invalidate_bad_quality(param_df)
            stats["invalidated"] += n_inv
            param_df, _ = cleaning.drop_quality_columns(param_df)

        if vent_df.empty and param_df.empty:
            return None

        # ── 3. Merge (chunked=False — already dept-sized, no need) ────
        df = merging.merge_meteo(vent_df, param_df, chunked=False)
        del vent_df, param_df
        gc.collect()

        # ── 4. Core cleaning (Q steps already done above) ─────────────
        df, nb_dup = cleaning.deduplicate(df)
        stats["duplicates"] += nb_dup

        df, n_crit = cleaning.drop_critical_nan(df)
        stats["crit_dropped"] += n_crit

        if df.empty:
            return None

        df = cleaning.parse_dates(df)
        df = cleaning.convert_numerics(df)
        df = cleaning.cast_string_columns(df)
        df = cleaning.normalize_labels(df)

        df, n_rec = cleaning.recover_weather_interpolation(df)
        stats["recovered"] += n_rec

        # ── 5. Feature engineering ────────────────────────────────────
        df = features.add_time_features(df)
        df = features.add_season(df)
        df = features.add_temperature_amplitude(df)
        df = features.add_wind_category(df)
        df = features.add_precipitation_category(df)

        # ── 6. Consistency fixes ──────────────────────────────────────
        df, _ = cleaning.fix_negative_values(df)
        df, _ = cleaning.fix_humidity_bounds(df)
        df, _ = cleaning.fix_temperature_consistency(df)

        df = df.reset_index(drop=True)
        return df, stats

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def _print_summary(self, s: dict) -> None:
        print("\n" + "═" * 60)
        print("PIPELINE COMPLETE")
        print("═" * 60)
        print(f"  Input rows (vent + param)  : {s['total_input_rows']:>12,}")
        print(f"  Output rows                : {s['total_output_rows']:>12,}")
        print(f"  Row retention rate         : {s['row_retention_rate']:>12.1%}")
        print(f"  Cells invalidated (Q=0)    : {s['total_invalidated_cells']:>12,}")
        print(f"  Duplicates removed         : {s['total_duplicates_removed']:>12,}")
        print(f"  Critical-NaN rows dropped  : {s['total_critical_nan_removed']:>12,}")
        print(f"  Cells recovered by interp  : {s['total_cells_recovered']:>12,}")
        print(f"  Output → {s['output_path']}")
        print("═" * 60)

    def _write_report(self, s: dict) -> None:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(self.report_path)), exist_ok=True)
        with open(self.report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            for k, v in s.items():
                writer.writerow({"metric": k, "value": v})
        print(f"Report saved → {self.report_path}")
