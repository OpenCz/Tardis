import matplotlib.pyplot as plt
import pandas as pd


def plot_null_overview(
    df: pd.DataFrame, initial_null_total: int, nb_rows: int, nb_slots: int
) -> None:
    perfect_rows = df.dropna().shape[0]
    nan_per_col = df.isnull().sum()
    nan_per_col = nan_per_col[nan_per_col > 0].sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(25, 6))

    axes[0].pie(
        [initial_null_total, nb_slots - initial_null_total],
        labels=["Null", "Healthy slots"],
        autopct="%1.1f%%",
    )
    axes[0].set_title("NaN global")

    axes[1].pie(
        [perfect_rows, nb_rows - perfect_rows],
        labels=["Perfect rows", "Rows with NaN"],
        autopct="%1.1f%%",
    )
    axes[1].set_title("Perfect rows")

    nan_per_col.plot(kind="barh", ax=axes[2])
    axes[2].set_title("NaN per column")

    plt.tight_layout()
    plt.show()


def plot_current_nan(df: pd.DataFrame) -> None:
    nan_counts = df.isnull().sum()
    nan_counts.plot(kind="barh", figsize=(10, 6))
    plt.title("NaN counts per column")
    plt.tight_layout()
    plt.show()


def plot_dedup(nb_dup: int, nb_rows: int) -> None:
    plt.pie(
        [nb_dup, nb_rows - nb_dup],
        labels=["Duplicated", "Unique"],
        autopct="%1.1f%%",
    )
    plt.title("Duplicate rows")
    plt.tight_layout()
    plt.show()


def plot_critical_nan_drop(useful: int, unrecoverable: int) -> None:
    plt.pie(
        [useful, unrecoverable],
        labels=["Useful", "Unrecoverable"],
        autopct="%1.1f%%",
    )
    plt.title("Rows kept after critical NaN drop")
    plt.tight_layout()
    plt.show()


def plot_station_clustering(
    n_merged: int, nclusters: int, clusters: dict[str, list[str]]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    n_unique = nclusters - n_merged
    if n_merged == 0:
        axes[0].text(0.5, 0.5, "No variants merged\n(all stations already unique)",
                     ha="center", va="center", fontsize=12, transform=axes[0].transAxes)
    elif n_unique == 0:
        axes[0].text(0.5, 0.5, "All station clusters\nhave merged variants",
                     ha="center", va="center", fontsize=12, transform=axes[0].transAxes)
    else:
        axes[0].pie(
            [n_merged, n_unique],
            labels=["Stations with variants", "Unique stations"],
            autopct="%1.1f%%",
        )
    axes[0].set_title("Stations merged vs unique")

    cluster_sizes = {
        canon: len(variants)
        for canon, variants in clusters.items()
        if len(variants) > 1
    }
    if cluster_sizes:
        pd.Series(cluster_sizes).sort_values().plot(kind="barh", ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "No merged clusters", ha="center", va="center",
                     fontsize=12, transform=axes[1].transAxes)
    axes[1].set_title("Variants merged per canonical station")
    axes[1].set_xlabel("Number of variants")

    plt.tight_layout()
    plt.show()


def plot_algebraic_recovery(algebraic_total: int, null_post_recovery: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [algebraic_total, null_post_recovery],
        labels=["Recovered algebraically", "Still null (intentional)"],
        autopct="%1.1f%%",
    )
    ax.set_title("NaN recovery — algebraic derivation coverage")
    plt.tight_layout()
    plt.show()


def plot_retention_funnel(
    steps: list[tuple[str, int]], original_rows: int
) -> None:
    step_names = [s[0] for s in steps][::-1]
    step_counts = [s[1] for s in steps][::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(step_names, step_counts)
    for bar, count in zip(bars, step_counts):
        pct = count / original_rows
        ax.text(
            bar.get_width() + 30,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1%}",
            va="center",
        )
    ax.set_xlabel("Number of rows")
    ax.set_title("Row retention funnel across cleaning pipeline")
    plt.tight_layout()
    plt.show()


def plot_corrections(neg_fixed: int, overflow_fixed: int, hier_fixed: int) -> None:
    labels = ["Negative values", "Count > scheduled", "Hierarchy violations"]
    values = [neg_fixed, overflow_fixed, hier_fixed]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha="center"
        )
    ax.set_ylabel("Number of corrections")
    ax.set_title("Impossible values corrected per type")
    plt.tight_layout()
    plt.show()


def plot_outlier_rates(outlier_rows: list[tuple[str, int, float]]) -> None:
    filtered = [(col, rate) for col, _n, rate in outlier_rows if rate > 0]
    if not filtered:
        return
    cols_labels = [c for c, _r in filtered]
    rates = [r for _c, r in filtered]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(cols_labels, rates)
    ax.axvline(0.05, color="red", linestyle="--", label="5% threshold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Outlier rate per column (|z-score| > 3)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_validity_checks(checks: dict[str, int]) -> None:
    names = list(checks.keys())
    vals = list(checks.values())
    colors = ["green" if v == 1 else "red" for v in vals]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, vals, color=colors)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.set_title("Validity checks — pipeline invariants")
    plt.tight_layout()
    plt.show()


def plot_comp_null_overview(comp_df: pd.DataFrame) -> None:
    null_counts = comp_df.isnull().sum()
    total_nulls = int(null_counts.sum())
    nb_rows, nb_cols = comp_df.shape
    nb_slots = nb_rows * nb_cols

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].pie(
        [total_nulls, nb_slots - total_nulls],
        labels=["Null", "Healthy slots"],
        autopct="%1.1f%%",
    )
    axes[0].set_title("Complementary dataset — NaN global")

    null_nonzero = null_counts[null_counts > 0]
    if null_nonzero.empty:
        axes[1].text(0.5, 0.5, "No null values found", ha="center", va="center", fontsize=14,
                     transform=axes[1].transAxes)
        axes[1].set_title("NaN per column")
    else:
        null_nonzero.sort_values().plot(kind="barh", ax=axes[1])
        axes[1].set_title("NaN per column")

    plt.tight_layout()
    plt.show()


def plot_comp_segment_distribution(comp_df: pd.DataFrame) -> None:
    if "Segment(s) DRG" not in comp_df.columns:
        return
    # Simplify multi-value segments to their first letter for grouping
    seg = comp_df["Segment(s) DRG"].str.split(";").str[0]
    seg_counts = seg.value_counts().sort_index()
    labels = {"A": "A — Grande vitesse (TGV)", "B": "B — Intercités", "C": "C — Réseau local"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(
        seg_counts.values,
        labels=[labels.get(k, k) for k in seg_counts.index],
        autopct="%1.1f%%",
    )
    axes[0].set_title("Station distribution by segment type")
    bars = axes[1].bar([labels.get(k, k) for k in seg_counts.index], seg_counts.values,
                       color=["gold", "steelblue", "mediumseagreen"])
    for bar, val in zip(bars, seg_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(val), ha="center", fontsize=10)
    axes[1].set_title("Station count by segment")
    axes[1].set_ylabel("Number of stations")
    plt.tight_layout()
    plt.show()
