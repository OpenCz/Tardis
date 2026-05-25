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
    axes[0].set_title("NaN global (meteo)")

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
    nan_counts = nan_counts[nan_counts > 0].sort_values()
    if nan_counts.empty:
        print("No NaN values remaining.")
        return
    nan_counts.plot(kind="barh", figsize=(10, max(6, len(nan_counts) * 0.3)))
    plt.title("NaN counts per column (after cleaning)")
    plt.tight_layout()
    plt.show()


def plot_dedup(nb_dup: int, nb_rows: int) -> None:
    plt.pie(
        [nb_dup, nb_rows - nb_dup],
        labels=["Duplicated", "Unique"],
        autopct="%1.1f%%",
    )
    plt.title("Duplicate rows — meteorological data")
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


def plot_quality_invalidation(n_invalidated: int, total_cells: int) -> None:
    valid = total_cells - n_invalidated
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [n_invalidated, valid],
        labels=["Invalidated (Q=0)", "Valid"],
        autopct="%1.2f%%",
    )
    ax.set_title("Cells invalidated by Météo-France quality flags")
    plt.tight_layout()
    plt.show()


def plot_interpolation_recovery(total_recovered: int, null_post_recovery: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        [total_recovered, null_post_recovery],
        labels=["Recovered (interpolation)", "Still null (intentional)"],
        autopct="%1.1f%%",
    )
    ax.set_title("NaN recovery — linear interpolation within station")
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
    ax.set_title("Row retention funnel across meteo cleaning pipeline")
    plt.tight_layout()
    plt.show()


def plot_corrections(neg_fixed: int, hum_fixed: int, temp_fixed: int) -> None:
    labels = ["Negative values", "Humidity out-of-range", "TN > TX"]
    values = [neg_fixed, hum_fixed, temp_fixed]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["steelblue", "coral", "gold"])
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha="center"
        )
    ax.set_ylabel("Number of corrections")
    ax.set_title("Impossible values corrected per type (meteo)")
    plt.tight_layout()
    plt.show()


def plot_outlier_rates(outlier_rows: list[tuple[str, int, float]]) -> None:
    filtered = [(col, rate) for col, _n, rate in outlier_rows if rate > 0]
    if not filtered:
        return
    cols_labels = [c for c, _r in filtered]
    rates = [r for _c, r in filtered]
    fig, ax = plt.subplots(figsize=(10, max(6, len(filtered) * 0.4)))
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
    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.5)))
    ax.barh(names, vals, color=colors)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FAIL", "PASS"])
    ax.set_title("Validity checks — meteo pipeline invariants")
    plt.tight_layout()
    plt.show()
