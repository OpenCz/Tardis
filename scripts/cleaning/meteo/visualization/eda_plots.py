import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

# ── Palette & style ──────────────────────────────────────────────────────────
_SEASON_ORDER  = ["winter", "spring", "summer", "autumn"]
_SEASON_COLORS = {"winter": "#4575b4", "spring": "#74add1",
                  "summer": "#d73027", "autumn": "#fdae61"}
_MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

_HEATMAP_COLS = [
    "RR", "TN", "TX", "TM", "temp_amplitude",
    "FFM", "FXY", "PMERM", "UN", "UX", "INST", "GLOT",
]


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING PLOTS (kept intact, minor style polish)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_temperature_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE for TN, TM, TX."""
    cols = [c for c in ["TN", "TM", "TX"] if c in df.columns]
    if not cols:
        print("[plot_temperature_distribution] No temperature columns found.")
        return
    labels = {"TN": "Min temp (°C)", "TM": "Mean temp (°C)", "TX": "Max temp (°C)"}
    colors = {"TN": "#4575b4", "TM": "#74add1", "TX": "#d73027"}
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color=colors.get(col, "steelblue"))
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(labels.get(col, col))
    plt.suptitle("Daily temperature distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_seasonal_temperature(df: pd.DataFrame) -> None:
    """Box + strip plot of mean temperature by season."""
    if "season" not in df.columns or "TM" not in df.columns:
        print("[plot_seasonal_temperature] season or TM column missing.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="season", y="TM", data=df,
                   order=_SEASON_ORDER, palette=_SEASON_COLORS,
                   inner="box", ax=ax)
    ax.set_title("Mean temperature distribution by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean temperature (°C)")
    plt.tight_layout()
    plt.show()


def plot_precipitation_distribution(df: pd.DataFrame) -> None:
    if "RR" not in df.columns:
        print("[plot_precipitation_distribution] RR column missing.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rr_nonzero = df.loc[df["RR"] > 0, "RR"].dropna()
    sns.histplot(rr_nonzero, kde=True, ax=axes[0], log_scale=True, color="#4575b4")
    axes[0].set_title("Precipitation (rainy days only, log scale)")
    axes[0].set_xlabel("Precipitation (mm)")

    if "precip_category" in df.columns:
        cat_order = ["dry", "trace", "light", "moderate", "heavy"]
        cat_colors = ["#ffffcc","#a1dab4","#41b6c4","#2c7fb8","#253494"]
        cat_counts = df["precip_category"].value_counts().reindex(cat_order).dropna()
        axes[1].pie(cat_counts.values, labels=cat_counts.index,
                    autopct="%1.1f%%", colors=cat_colors, startangle=90)
        axes[1].set_title("Precipitation category proportions")
    else:
        sns.boxplot(y=df["RR"].dropna(), ax=axes[1])
        axes[1].set_title("Precipitation spread and outliers")
    plt.tight_layout()
    plt.show()


def plot_wind_distribution(df: pd.DataFrame) -> None:
    if "FFM" not in df.columns:
        print("[plot_wind_distribution] FFM column missing.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["FFM"].dropna(), kde=True, ax=axes[0], color="#1a9641")
    axes[0].set_title("Mean wind speed distribution (FFM)")
    axes[0].set_xlabel("Wind speed (m/s)")

    if "wind_category" in df.columns:
        cat_order  = ["calm","light","moderate","strong","storm"]
        cat_colors = ["#ffffb2","#fecc5c","#fd8d3c","#f03b20","#bd0026"]
        cat_counts = df["wind_category"].value_counts().reindex(cat_order).dropna()
        bars = axes[1].bar(cat_counts.index, cat_counts.values,
                           color=cat_colors[:len(cat_counts)])
        for bar, val in zip(bars, cat_counts.values):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max(cat_counts)*0.01,
                         f"{val:,}", ha="center", fontsize=9)
        axes[1].set_title("Wind category distribution")
        axes[1].set_xlabel("Category")
        axes[1].set_ylabel("Station-days")
    plt.tight_layout()
    plt.show()


def plot_monthly_temperature_trend(df: pd.DataFrame) -> None:
    if "date" not in df.columns or "TM" not in df.columns:
        print("[plot_monthly_temperature_trend] date or TM column missing.")
        return
    monthly = df.groupby(df["date"].dt.to_period("M"))["TM"].mean()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly.index, monthly.values, alpha=0.4, color="#d73027", linewidth=0.8)
    rolling = monthly.rolling(12, center=True).mean()
    ax.plot(rolling.index, rolling.values, color="#d73027", linewidth=2, label="12-month rolling mean")
    ax.axhline(monthly.mean(), color="black", linestyle="--",
               label=f"Overall mean ({monthly.mean():.1f} °C)")
    ax.set_title("Monthly mean temperature over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean temperature (°C)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_monthly_precipitation_trend(df: pd.DataFrame) -> None:
    if "date" not in df.columns or "RR" not in df.columns:
        print("[plot_monthly_precipitation_trend] date or RR column missing.")
        return
    monthly = df.groupby(df["date"].dt.to_period("M"))["RR"].sum()
    monthly.index = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(monthly.index, monthly.values, width=20, color="#4575b4", alpha=0.7)
    rolling = monthly.rolling(12, center=True).mean()
    ax.plot(rolling.index, rolling.values, color="#253494", linewidth=2, label="12-month rolling mean")
    ax.set_title("Monthly total precipitation over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total precipitation (mm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_top_stations_by_temperature(df: pd.DataFrame, n: int = 10) -> None:
    if "NOM_USUEL" not in df.columns or "TM" not in df.columns:
        print("[plot_top_stations_by_temperature] NOM_USUEL or TM column missing.")
        return
    top = df.groupby("NOM_USUEL")["TM"].mean().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(top.values.min(), top.values.max())
    colors = [cmap(norm(v)) for v in top.sort_values().values]
    bars = top.sort_values().plot(kind="barh", ax=ax, color=colors)
    for i, (name, val) in enumerate(top.sort_values().items()):
        ax.text(val + 0.05, i, f"{val:.1f} °C", va="center", fontsize=9)
    ax.set_title(f"Top {n} stations by mean temperature")
    ax.set_xlabel("Mean temperature (°C)")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    cols = [c for c in _HEATMAP_COLS if c in df.columns]
    if not cols:
        print("[plot_correlation_matrix] No relevant numeric columns found.")
        return
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle only
    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation matrix — key meteorological features")
    plt.tight_layout()
    plt.show()


def plot_temperature_heatmap_year_month(df: pd.DataFrame) -> None:
    if "year" not in df.columns or "month" not in df.columns or "TM" not in df.columns:
        print("[plot_temperature_heatmap_year_month] Required columns missing.")
        return
    pivot = df.pivot_table(values="TM", index="year", columns="month", aggfunc="mean")
    pivot.columns = [_MONTH_NAMES[c - 1] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.35)))
    sns.heatmap(pivot, annot=False, cmap="RdYlBu_r", ax=ax,
                linewidths=0, cbar_kws={"label": "Mean temperature (°C)"})
    ax.set_title("Mean temperature (°C) — Year × Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    plt.tight_layout()
    plt.show()


def plot_seasonal_wind(df: pd.DataFrame) -> None:
    if "season" not in df.columns or "FFM" not in df.columns:
        print("[plot_seasonal_wind] season or FFM column missing.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="season", y="FFM", data=df,
                   order=_SEASON_ORDER, palette=_SEASON_COLORS,
                   inner="box", ax=ax)
    ax.set_title("Mean wind speed by season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Wind speed (m/s)")
    plt.tight_layout()
    plt.show()


def plot_station_coverage(df: pd.DataFrame) -> None:
    if "NOM_USUEL" not in df.columns:
        print("[plot_station_coverage] NOM_USUEL column missing.")
        return
    counts = df["NOM_USUEL"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = counts.sort_values().plot(kind="barh", ax=ax, color="#2c7fb8")
    for i, v in enumerate(counts.sort_values().values):
        ax.text(v + counts.max() * 0.005, i, f"{v:,}", va="center", fontsize=8)
    ax.set_title("Top 20 stations by record count")
    ax.set_xlabel("Number of daily records")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# NEW PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_temperature_anomaly(
    df: pd.DataFrame,
    baseline_start: int = 1961,
    baseline_end:   int = 1990,
) -> None:
    """Annual temperature anomaly vs climatological baseline (bar chart + rolling mean)."""
    if "year" not in df.columns or "TM" not in df.columns:
        print("[plot_temperature_anomaly] year or TM column missing.")
        return

    annual = df.groupby("year")["TM"].mean()
    mask   = (annual.index >= baseline_start) & (annual.index <= baseline_end)
    baseline = annual[mask].mean() if mask.any() else annual.mean()
    anomaly  = annual - baseline

    colors  = ["#d73027" if v >= 0 else "#4575b4" for v in anomaly.values]
    rolling = anomaly.rolling(10, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(anomaly.index, anomaly.values, color=colors, width=0.8, alpha=0.85)
    ax.plot(rolling.index, rolling.values, color="black", linewidth=2,
            label="10-yr rolling mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(rolling.index, rolling.values, 0,
                    where=rolling.values >= 0, alpha=0.15, color="#d73027")
    ax.fill_between(rolling.index, rolling.values, 0,
                    where=rolling.values < 0,  alpha=0.15, color="#4575b4")
    ax.set_title(f"Annual temperature anomaly  (baseline {baseline_start}–{baseline_end},"
                 f" ref = {baseline:.2f} °C)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Anomaly (°C)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f"))
    plt.tight_layout()
    plt.show()


def plot_station_map(df: pd.DataFrame) -> None:
    """Scatter map of stations (LON × LAT), coloured by mean temperature."""
    needed = {"LAT", "LON", "NOM_USUEL", "TM"}
    if not needed.issubset(df.columns):
        print("[plot_station_map] LAT, LON, NOM_USUEL or TM column missing.")
        return

    agg = {"LAT": ("LAT", "first"), "LON": ("LON", "first"),
           "mean_TM": ("TM", "mean"), "n_records": ("TM", "count")}
    if "ALTI" in df.columns:
        agg["ALTI"] = ("ALTI", "first")
    stats = df.groupby("NOM_USUEL").agg(**agg).reset_index()

    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        stats["LON"], stats["LAT"],
        c=stats["mean_TM"], cmap="RdYlBu_r",
        s=stats["n_records"].clip(upper=stats["n_records"].quantile(0.95)) / 200,
        alpha=0.75, edgecolors="grey", linewidths=0.3,
    )
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Mean temperature (°C)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Station network — {len(stats):,} stations  (dot size ∝ record count)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weather_events_by_month(df: pd.DataFrame) -> None:
    """Monthly frequency of binary weather phenomena: snow, storm, fog, frost, ice."""
    event_cols = {
        "NEIG":    "Snow",
        "ORAG":    "Thunderstorm",
        "BROU":    "Fog",
        "GELEE":   "Frost",
        "VERGLAS": "Ice on ground",
    }
    present = {k: v for k, v in event_cols.items() if k in df.columns}
    if not present or "month" not in df.columns:
        print("[plot_weather_events_by_month] No binary event columns or month missing.")
        return

    monthly = pd.DataFrame({
        label: df[col].fillna(0).astype(bool).groupby(df["month"]).sum()
        for col, label in present.items()
    })
    monthly.index = _MONTH_NAMES

    fig, ax = plt.subplots(figsize=(14, 6))
    monthly.plot(kind="bar", ax=ax, width=0.75,
                 color=["#4575b4","#d73027","#74add1","#313695","#fee090"])
    ax.set_title("Weather phenomena frequency by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Station-days with event")
    ax.legend(title="Phenomenon", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()


def plot_seasonal_precipitation(df: pd.DataFrame) -> None:
    """Violin (log) + % rainy days per season."""
    if "season" not in df.columns or "RR" not in df.columns:
        print("[plot_seasonal_precipitation] season or RR column missing.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rainy = df[df["RR"] > 0].copy()
    sns.violinplot(x="season", y="RR", data=rainy, order=_SEASON_ORDER,
                   palette=_SEASON_COLORS, inner="box", ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Precipitation distribution (rainy days, log scale)")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Daily precipitation (mm)")

    pct = (
        df.groupby("season")["RR"]
        .apply(lambda s: (s > 0).mean() * 100, include_groups=False)
        .reindex(_SEASON_ORDER)
    )
    bars = axes[1].bar(_SEASON_ORDER, pct.values,
                       color=[_SEASON_COLORS[s] for s in _SEASON_ORDER])
    for bar, val in zip(bars, pct.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3, f"{val:.1f}%",
                     ha="center", fontsize=10)
    axes[1].set_title("Percentage of rainy days by season")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("% of station-days with RR > 0")
    axes[1].set_ylim(0, pct.max() * 1.15)

    plt.tight_layout()
    plt.show()


def plot_sunshine_overview(df: pd.DataFrame) -> None:
    """Sunshine hours: histogram + monthly boxplot."""
    if "INST" not in df.columns:
        print("[plot_sunshine_overview] INST (sunshine hours) column missing.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["INST"].dropna(), kde=True, bins=50,
                 color="#fdae61", ax=axes[0])
    axes[0].set_title("Daily sunshine duration distribution")
    axes[0].set_xlabel("Sunshine hours (h)")

    if "month" in df.columns:
        df_plot = df[["month", "INST"]].dropna().copy()
        df_plot["month_name"] = df_plot["month"].map(
            dict(enumerate(_MONTH_NAMES, 1))
        )
        sns.boxplot(x="month_name", y="INST", data=df_plot,
                    order=_MONTH_NAMES, showfliers=False,
                    palette="YlOrRd", ax=axes[1])
        axes[1].set_title("Sunshine hours by month")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Sunshine hours (h)")
        axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("Sunshine (INST) overview", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_temp_precip_scatter(df: pd.DataFrame, sample_n: int = 50_000) -> None:
    """Scatter of TM vs RR (rainy days) coloured by season."""
    needed = {"TM", "RR", "season"}
    if not needed.issubset(df.columns):
        print("[plot_temp_precip_scatter] TM, RR, or season column missing.")
        return

    data = df[list(needed)].dropna()
    data = data[data["RR"] > 0]
    if len(data) > sample_n:
        data = data.sample(sample_n, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    for season in _SEASON_ORDER:
        grp = data[data["season"] == season]
        ax.scatter(grp["TM"], grp["RR"],
                   c=_SEASON_COLORS[season], label=season,
                   alpha=0.25, s=7, rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel("Mean temperature (°C)")
    ax.set_ylabel("Daily precipitation (mm, log scale)")
    ax.set_title("Temperature vs precipitation on rainy days")
    ax.legend(title="Season", markerscale=4)
    plt.tight_layout()
    plt.show()


def plot_long_term_trends(df: pd.DataFrame) -> None:
    """3-panel annual trend chart: temperature, precipitation, wind speed."""
    if "year" not in df.columns:
        print("[plot_long_term_trends] year column missing.")
        return

    panels = [
        ("TM",  "mean", "Annual mean temperature (°C)", "#d73027"),
        ("RR",  "sum",  "Annual total precipitation (mm)", "#4575b4"),
        ("FFM", "mean", "Annual mean wind speed (m/s)",  "#1a9641"),
    ]
    panels = [(c, a, l, col) for c, a, l, col in panels if c in df.columns]
    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 4 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, agg, label, color) in zip(axes, panels):
        annual = df.groupby("year")[col].agg(agg)
        ax.plot(annual.index, annual.values, color=color, alpha=0.5, linewidth=1)
        rolling = annual.rolling(10, center=True).mean()
        ax.plot(rolling.index, rolling.values, color=color,
                linewidth=2.5, label="10-yr rolling mean")

        valid = annual.dropna()
        if len(valid) > 15:
            z = np.polyfit(valid.index.astype(float), valid.values, 1)
            ax.plot(valid.index, np.polyval(z, valid.index.astype(float)),
                    "--", color="black", linewidth=1.2,
                    label=f"Trend ({z[0]:+.3f} per yr)")

        ax.set_ylabel(label)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Year")
    fig.suptitle("Long-term meteorological trends (annual aggregates)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_seasonal_stats_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of seasonal statistics for TM, RR, FFM — replaces the text table."""
    cols_avail = [c for c in ["TM", "RR", "FFM"] if c in df.columns]
    if "season" not in df.columns or not cols_avail:
        print("[plot_seasonal_stats_heatmap] season or metric columns missing.")
        return

    rows = []
    for col in cols_avail:
        agg = df.groupby("season")[col].agg(["mean", "median", "std"])
        agg.columns = [f"{col} mean", f"{col} median", f"{col} std"]
        rows.append(agg.reindex(_SEASON_ORDER))

    stats = pd.concat(rows, axis=1).round(2)

    fig, axes = plt.subplots(1, len(cols_avail), figsize=(5 * len(cols_avail), 5))
    if len(cols_avail) == 1:
        axes = [axes]

    units = {"TM": "°C", "RR": "mm", "FFM": "m/s"}
    cmaps = {"TM": "RdYlBu_r", "RR": "Blues", "FFM": "Greens"}

    for ax, col in zip(axes, cols_avail):
        sub = stats[[c for c in stats.columns if c.startswith(col)]]
        sub.columns = ["mean", "median", "std"]
        sns.heatmap(sub, annot=True, fmt=".1f", cmap=cmaps.get(col, "viridis"),
                    ax=ax, cbar_kws={"label": units.get(col, "")},
                    linewidths=0.5)
        ax.set_title(f"{col} ({units.get(col, '')})")
        ax.set_xlabel("")

    fig.suptitle("Seasonal statistics — mean, median, std", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_humidity_overview(df: pd.DataFrame) -> None:
    """Humidity (UN, UM, UX) distribution + seasonal pattern."""
    hum_cols = [c for c in ["UN", "UM", "UX"] if c in df.columns]
    if not hum_cols:
        print("[plot_humidity_overview] No humidity columns (UN, UM, UX) found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col in hum_cols:
        sns.kdeplot(df[col].dropna(), ax=axes[0], label=col, fill=True, alpha=0.3)
    axes[0].set_title("Relative humidity distribution")
    axes[0].set_xlabel("Humidity (%)")
    axes[0].legend()

    if "season" in df.columns and "UM" in df.columns:
        sns.boxplot(x="season", y="UM", data=df, order=_SEASON_ORDER,
                    palette=_SEASON_COLORS, ax=axes[1], showfliers=False)
        axes[1].set_title("Mean relative humidity by season")
        axes[1].set_xlabel("Season")
        axes[1].set_ylabel("Mean humidity (%)")

    plt.suptitle("Humidity overview", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_all_eda(df: pd.DataFrame) -> None:
    plot_temperature_distribution(df)
    plot_seasonal_temperature(df)
    plot_precipitation_distribution(df)
    plot_wind_distribution(df)
    plot_monthly_temperature_trend(df)
    plot_monthly_precipitation_trend(df)
    plot_correlation_matrix(df)
    plot_temperature_heatmap_year_month(df)
    plot_seasonal_wind(df)
    plot_station_coverage(df)
    # New plots
    plot_temperature_anomaly(df)
    plot_station_map(df)
    plot_weather_events_by_month(df)
    plot_seasonal_precipitation(df)
    plot_sunshine_overview(df)
    plot_temp_precip_scatter(df)
    plot_long_term_trends(df)
    plot_seasonal_stats_heatmap(df)
    plot_humidity_overview(df)
