from scripts.trains.visualization.cleaning_plots import (
    plot_algebraic_recovery,
    plot_corrections,
    plot_critical_nan_drop,
    plot_current_nan,
    plot_dedup,
    plot_null_overview,
    plot_outlier_rates,
    plot_retention_funnel,
    plot_station_clustering,
    plot_validity_checks,
)
from scripts.trains.visualization.eda_plots import (
    plot_cancellation_by_service,
    plot_correlation_matrix,
    plot_delay_categories,
    plot_delay_distribution,
    plot_monthly_trend,
    plot_seasonal_delay,
    plot_top_stations_by_delay,
    run_all_eda,
)

__all__ = [
    # cleaning plots
    "plot_null_overview",
    "plot_current_nan",
    "plot_dedup",
    "plot_critical_nan_drop",
    "plot_station_clustering",
    "plot_algebraic_recovery",
    "plot_retention_funnel",
    "plot_corrections",
    "plot_outlier_rates",
    "plot_validity_checks",
    # eda plots
    "plot_delay_distribution",
    "plot_seasonal_delay",
    "plot_top_stations_by_delay",
    "plot_monthly_trend",
    "plot_delay_categories",
    "plot_correlation_matrix",
    "plot_cancellation_by_service",
    "run_all_eda",
]
