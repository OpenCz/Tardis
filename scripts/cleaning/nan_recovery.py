import pandas as pd

_COL_DEP_LATE = "Average delay of late trains at departure"
_COL_DEP_ALL = "Average delay of all trains at departure"
_COL_N_DEP = "Number of trains delayed at departure"

_COL_ARR_LATE = "Average delay of late trains at arrival"
_COL_ARR_ALL = "Average delay of all trains at arrival"
_COL_N_ARR = "Number of trains delayed at arrival"

_COL_SCHEDULED = "Number of scheduled trains"


def recover_departure_delay(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    mask = (
        df[_COL_DEP_LATE].isna()
        & df[_COL_DEP_ALL].notna()
        & (df[_COL_N_DEP] > 0)
    )
    dep_late_filled = int(mask.sum())
    df.loc[mask, _COL_DEP_LATE] = (
        df.loc[mask, _COL_DEP_ALL]
        * df.loc[mask, _COL_SCHEDULED]
        / df.loc[mask, _COL_N_DEP]
    )

    mask = df[_COL_DEP_ALL].isna() & df[_COL_DEP_LATE].notna() & df[_COL_N_DEP].notna()
    dep_all_filled = int(mask.sum())
    df.loc[mask, _COL_DEP_ALL] = (
        df.loc[mask, _COL_DEP_LATE]
        * df.loc[mask, _COL_N_DEP]
        / df.loc[mask, _COL_SCHEDULED]
    )

    print(f"dep_late filled from dep_all : {dep_late_filled} row(s)")
    print(f"dep_all  filled from dep_late : {dep_all_filled} row(s)")
    return df, dep_late_filled, dep_all_filled


def recover_arrival_delay(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    mask = (
        df[_COL_ARR_LATE].isna()
        & df[_COL_ARR_ALL].notna()
        & (df[_COL_N_ARR] > 0)
    )
    arr_late_filled = int(mask.sum())
    df.loc[mask, _COL_ARR_LATE] = (
        df.loc[mask, _COL_ARR_ALL]
        * df.loc[mask, _COL_SCHEDULED]
        / df.loc[mask, _COL_N_ARR]
    )

    mask = df[_COL_ARR_ALL].isna() & df[_COL_ARR_LATE].notna() & df[_COL_N_ARR].notna()
    arr_all_filled = int(mask.sum())
    df.loc[mask, _COL_ARR_ALL] = (
        df.loc[mask, _COL_ARR_LATE]
        * df.loc[mask, _COL_N_ARR]
        / df.loc[mask, _COL_SCHEDULED]
    )

    print(f"arr_late filled from arr_all : {arr_late_filled} row(s)")
    print(f"arr_all  filled from arr_late : {arr_all_filled} row(s)")
    return df, arr_late_filled, arr_all_filled
