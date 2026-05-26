import pandas as pd

_COUNT_COLS = [
    "Number of scheduled trains",
    "Number of cancelled trains",
    "Number of trains delayed at departure",
    "Number of trains delayed at arrival",
    "Number of trains delayed > 15min",
    "Number of trains delayed > 30min",
    "Number of trains delayed > 60min",
]
_COUNTS_VS_SCHEDULED = [
    "Number of cancelled trains",
    "Number of trains delayed at departure",
    "Number of trains delayed at arrival",
]
_DELAY_HIERARCHY = [
    "Number of trains delayed > 15min",
    "Number of trains delayed > 30min",
    "Number of trains delayed > 60min",
]
_ROUTE_COLS = ["Departure station", "Arrival station"]
_COL_SCHEDULED = "Number of scheduled trains"


def fix_negative_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    neg_fixed_total = 0
    for col in _COUNT_COLS:
        mask = df[col] < 0
        if mask.any():
            route_median = df.groupby(_ROUTE_COLS)[col].transform(
                lambda x: x[x >= 0].median()
            )
            df.loc[mask, col] = route_median[mask]
            neg_fixed_total += int(mask.sum())
            print(f"[FIXED] {col}: {mask.sum()} negative value(s) → route median")
    print(f"Total: {neg_fixed_total} fix(es)")
    return df, neg_fixed_total


def fix_count_overflow(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    overflow_fixed = 0
    for col in _COUNTS_VS_SCHEDULED:
        mask = df[col].notna() & (df[col] > df[_COL_SCHEDULED])
        if mask.any():
            route_median = df.groupby(_ROUTE_COLS)[col].transform(
                lambda x, c=col: x[
                    x <= df.loc[x.index, _COL_SCHEDULED]
                ].median()
            )
            df.loc[mask, col] = route_median[mask]
            overflow_fixed += int(mask.sum())
            print(
                f"[FIXED] {col}: {mask.sum()} value(s) exceeded scheduled → route median"
            )
    print(f"Total: {overflow_fixed} fix(es)")
    return df, overflow_fixed


def fix_delay_hierarchy(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    hier_fixed = 0
    for i in range(len(_DELAY_HIERARCHY) - 1):
        col_lower = _DELAY_HIERARCHY[i]
        col_higher = _DELAY_HIERARCHY[i + 1]
        mask = (
            df[col_higher].notna()
            & df[col_lower].notna()
            & (df[col_higher] > df[col_lower])
        )
        if mask.any():
            route_median = df.groupby(_ROUTE_COLS)[col_higher].transform(
                lambda x, cl=col_lower: x[x <= df.loc[x.index, cl]].median()
            )
            df.loc[mask, col_higher] = route_median[mask]
            hier_fixed += int(mask.sum())
            print(f"[FIXED] {col_higher} > {col_lower}: {mask.sum()} row(s)")
    print(f"Total: {hier_fixed} fix(es)")
    return df, hier_fixed


def recompute_rates(df: pd.DataFrame) -> pd.DataFrame:
    df["cancellation_rate"] = df["Number of cancelled trains"] / df[
        _COL_SCHEDULED
    ].replace(0, float("nan"))
    df["punctuality_rate"] = 1 - (
        df["Number of trains delayed at arrival"]
        / df[_COL_SCHEDULED].replace(0, float("nan"))
    )
    print(
        f'cancellation_rate — min: {df["cancellation_rate"].min():.4f}'
        f'  mean: {df["cancellation_rate"].mean():.4f}'
        f'  max: {df["cancellation_rate"].max():.4f}'
    )
    print(
        f'punctuality_rate  — min: {df["punctuality_rate"].min():.4f}'
        f'  mean: {df["punctuality_rate"].mean():.4f}'
        f'  max: {df["punctuality_rate"].max():.4f}'
    )
    return df
