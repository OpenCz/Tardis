import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def baseline_metrics(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    pred = np.full(len(y_test), y_train.mean())
    return compute_metrics(y_test, pred)


def compare_models(all_results: dict, baseline: dict) -> pd.DataFrame:
    df = pd.DataFrame(all_results).T.sort_values("RMSE")
    df["vs Baseline"] = (
        ((1 - df["RMSE"] / baseline["RMSE"]) * 100).round(1).astype(str) + " %"
    )
    return df
