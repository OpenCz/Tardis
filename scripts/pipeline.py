import scripts.audit as audit
import scripts.cleaning as cleaning
import scripts.features as features
import scripts.loading as loading


def run_pipeline(
    data_path: str,
    output_path: str = "cleaned_dataset.csv",
    report_path: str = "cleaning_report.csv",
    station_threshold: int = 80,
):
    report = audit.AuditReport()

    df, original = loading.load_data(data_path)
    report.record_initial_state(original)

    df, comment_cols = cleaning.drop_comment_columns(df)
    report.record_comment_cols(comment_cols)
    cols_after_comment_drop = df.shape[1]

    df, nb_dup = cleaning.deduplicate(df)
    report.record_duplicates(nb_dup)

    df, crit_dropped = cleaning.drop_critical_nan(df)
    report.record_critical_nan(crit_dropped)

    df = cleaning.parse_dates(df)
    df = cleaning.convert_numerics(df)
    df = cleaning.cast_string_columns(df)
    df = cleaning.normalize_labels(df)

    clusterer = cleaning.StationClusterer(threshold=station_threshold)
    df, station_cols, cluster_stats = clusterer.fit_transform(df)
    report.record_station_clustering(cluster_stats, station_cols)

    df, dep_late, dep_all = cleaning.recover_departure_delay(df)
    df, arr_late, arr_all = cleaning.recover_arrival_delay(df)
    report.record_delay_recovery(dep_late, dep_all, arr_late, arr_all)
    report.record_recovery_summary(df)

    df = features.add_time_features(df)
    df = features.add_season(df)
    df = features.add_delay_category(df)
    df = features.add_cancellation_rate(df)
    df = features.add_punctuality_rate(df)

    df, neg_fixed = cleaning.fix_negative_counts(df)
    df, overflow_fixed = cleaning.fix_count_overflow(df)
    df, hier_fixed = cleaning.fix_delay_hierarchy(df)
    df = cleaning.recompute_rates(df)
    report.record_corrections(neg_fixed, overflow_fixed, hier_fixed)

    report.record_pipeline_summary(
        df,
        original_rows=original.shape[0],
        original_col_count=original.shape[1],
        cols_after_comment_drop=cols_after_comment_drop,
    )

    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Exported: {len(df)} rows, {len(df.columns)} columns → {output_path}")
    print(f"Columns: {df.columns.tolist()}")

    audit.check_completeness(df, report)
    audit.check_distributions(df, report)
    audit.check_outliers(df, report)
    audit.check_validity(df, report)
    audit.check_effectiveness(df, report)

    report.record_final_state(df, original_rows=original.shape[0])
    report.export(report_path)

    return df, report
