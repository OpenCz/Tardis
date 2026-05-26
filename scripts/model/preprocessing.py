import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ROUTE_STAT_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median of route-level features per (departure, arrival) pair."""
    return (
        df.groupby(["Departure station", "Arrival station"])[ROUTE_STAT_FEATURES + ["Service"]]
        .agg({f: "median" for f in ROUTE_STAT_FEATURES} | {"Service": "first"})
        .reset_index()
    )


def get_numpy_arrays(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit preprocessor on train, return (X_train_np, X_test_np, y_train_np, y_test_np)."""
    preprocessor.fit(X_train)
    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)
    y_tr = y_train.to_numpy(dtype=np.float32)
    y_te = y_test.to_numpy(dtype=np.float32)
    return X_tr, X_te, y_tr, y_te
