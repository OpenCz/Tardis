import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


RANDOM_STATE = 42

NUMERIC_FEATURES = ["LAT", "LON", "ALTI", "year", "month", "day_of_week", "day_of_year"]
CATEGORICAL_FEATURES = ["NUM_POSTE", "NOM_USUEL", "season"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def train_one(df: pd.DataFrame, target: str, out_path: Path, max_rows: int) -> dict:
    df_model = df.dropna(subset=[target]).copy()
    if len(df_model) > max_rows:
        df_model = df_model.sample(max_rows, random_state=RANDOM_STATE)

    X = df_model[ALL_FEATURES]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline([("prep", build_preprocessor()), ("model", model)])
    pipeline.fit(X_train, y_train)
    metrics = compute_metrics(y_test, pipeline.predict(X_test))

    joblib.dump(
        {
            "pipeline": pipeline,
            "target": target,
            "features": ALL_FEATURES,
            "metrics": metrics,
        },
        out_path,
    )
    return metrics


def run_training(
    dataset_path: str = "data/processed/meteo/cleaned_meteo_dataset.csv",
    out_dir: str = "models/meteo_aux",
    max_rows: int = 250_000,
) -> dict:
    usecols = [
        "NUM_POSTE",
        "NOM_USUEL",
        "LAT",
        "LON",
        "ALTI",
        "RR",
        "FFM",
        "date",
        "year",
        "month",
        "season",
    ]
    df = pd.read_csv(dataset_path, usecols=usecols, parse_dates=["date"], low_memory=False)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "RR": train_one(df, "RR", out / "rr_model.joblib", max_rows=max_rows),
        "FFM": train_one(df, "FFM", out / "ffm_model.joblib", max_rows=max_rows),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(results, f, indent=2)

    for target, metrics in results.items():
        print(
            f"{target:3s} | RMSE: {metrics['RMSE']:.3f} "
            f"| MAE: {metrics['MAE']:.3f} | R2: {metrics['R2']:.4f}"
        )
    print(f"Saved auxiliary weather models in {out_dir}/")
    return results


if __name__ == "__main__":
    max_rows = int(os.environ.get("METEO_AUX_MAX_ROWS", "250000"))
    run_training(max_rows=max_rows)
