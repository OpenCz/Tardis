import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .evaluation import compute_metrics

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def _make_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline([("prep", clone(preprocessor)), ("model", model)])


def _eval(pipeline, X_train, y_train, X_test, y_test) -> dict:
    pipeline.fit(X_train, y_train)
    return compute_metrics(y_test.to_numpy(), pipeline.predict(X_test))


def train_all(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, dict]:
    """Return (results, pipelines) dicts keyed by model name."""
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=4
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, tree_method="hist", verbosity=0,
        )
    if HAS_LGB:
        candidates["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=-1,
        )

    results, pipelines = {}, {}
    for name, model in candidates.items():
        pipe = _make_pipeline(preprocessor, model)
        results[name] = _eval(pipe, X_train, y_train, X_test, y_test)
        pipelines[name] = pipe
        print(
            f"  {name:20s} | RMSE: {results[name]['RMSE']:.3f} "
            f"| MAE: {results[name]['MAE']:.3f} | R²: {results[name]['R2']:.4f}"
        )
    return results, pipelines


def tune_best(
    preprocessor,
    best_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict]:
    """GridSearchCV on the best model, return (tuned_pipeline, metrics)."""
    if best_name == "LightGBM" and HAS_LGB:
        base = lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1)
        grid = {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [4, 6],
            "model__num_leaves": [31, 63],
        }
    elif best_name == "XGBoost" and HAS_XGB:
        base = xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0, tree_method="hist")
        grid = {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [4, 5, 6],
            "model__subsample": [0.8, 1.0],
        }
    elif best_name == "Random Forest":
        base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=4)
        grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 20, 30],
            "model__min_samples_split": [2, 5],
        }
    else:
        base = GradientBoostingRegressor(random_state=RANDOM_STATE)
        grid = {
            "model__n_estimators": [200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5],
        }

    pipe = _make_pipeline(preprocessor, base)
    search = GridSearchCV(
        pipe, grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=1, verbose=1
    )
    search.fit(X_train, y_train)
    tuned = search.best_estimator_
    metrics = compute_metrics(y_test.to_numpy(), tuned.predict(X_test))
    print(f"\n  Best params : {search.best_params_}")
    print(f"  CV RMSE     : {-search.best_score_:.3f}")
    print(f"  Test RMSE   : {metrics['RMSE']:.3f}")
    return tuned, metrics
