import scripts.audit as audit
import scripts.cleaning as cleaning
import scripts.features as features
import scripts.loading as loading


class Pipeline:
    def __init__(
        self,
        data_path: str,
        output_path: str = "data/proessed/trains/cleaned_dataset.csv",
        report_path: str = "data/proessed/audit/cleaning_report.csv",
        station_threshold: int = 80,
    ):
        self.data_path = data_path
        self.output_path = output_path
        self.report_path = report_path
        self.station_threshold = station_threshold

        self.df = None
        self.original = None
        self.report = audit.AuditReport()

    def run(self):
        self._load()
        self._clean()
        self._engineer_features()
        self._fix_consistency()
        self._export()
        self._audit_quality()
        self.report.record_final_state(self.df, original_rows=self.original.shape[0])
        self.report.export(self.report_path)
        return self.df, self.report

    def _load(self):
        self.df, self.original = loading.load_data(self.data_path)
        self.report.record_initial_state(self.original)

    def _clean(self):
        self.df, comment_cols = cleaning.drop_comment_columns(self.df)
        self.report.record_comment_cols(comment_cols)
        self._cols_after_comment_drop = self.df.shape[1]

        self.df, nb_dup = cleaning.deduplicate(self.df)
        self.report.record_duplicates(nb_dup)

        self.df, crit_dropped = cleaning.drop_critical_nan(self.df)
        self.report.record_critical_nan(crit_dropped)

        self.df = cleaning.parse_dates(self.df)
        self.df = cleaning.convert_numerics(self.df)
        self.df = cleaning.cast_string_columns(self.df)
        self.df = cleaning.normalize_labels(self.df)

        clusterer = cleaning.StationClusterer(threshold=self.station_threshold)
        self.df, station_cols, cluster_stats = clusterer.fit_transform(self.df)
        self.report.record_station_clustering(cluster_stats, station_cols)

        self.df, dep_late, dep_all = cleaning.recover_departure_delay(self.df)
        self.df, arr_late, arr_all = cleaning.recover_arrival_delay(self.df)
        self.report.record_delay_recovery(dep_late, dep_all, arr_late, arr_all)
        self.report.record_recovery_summary(self.df)

        self.report.record_pipeline_summary(
            self.df,
            original_rows=self.original.shape[0],
            original_col_count=self.original.shape[1],
            cols_after_comment_drop=self._cols_after_comment_drop,
        )

    def _engineer_features(self):
        self.df = features.add_time_features(self.df)
        self.df = features.add_season(self.df)
        self.df = features.add_delay_category(self.df)
        self.df = features.add_cancellation_rate(self.df)
        self.df = features.add_punctuality_rate(self.df)

    def _fix_consistency(self):
        self.df, neg_fixed = cleaning.fix_negative_counts(self.df)
        self.df, overflow_fixed = cleaning.fix_count_overflow(self.df)
        self.df, hier_fixed = cleaning.fix_delay_hierarchy(self.df)
        self.df = cleaning.recompute_rates(self.df)
        self.report.record_corrections(neg_fixed, overflow_fixed, hier_fixed)

    def _export(self):
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(self.output_path, index=False)
        print(f"Exported: {len(self.df)} rows, {len(self.df.columns)} columns → {self.output_path}")
        print(f"Columns: {self.df.columns.tolist()}")

    def _audit_quality(self):
        audit.check_completeness(self.df, self.report)
        audit.check_distributions(self.df, self.report)
        audit.check_outliers(self.df, self.report)
        audit.check_validity(self.df, self.report)
        audit.check_effectiveness(self.df, self.report)
