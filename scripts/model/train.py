import json
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from ..features import add_time_features
from .config import ALL_FEATURES, RANDOM_STATE, TARGET
from .dl_models import HAS_TF, HAS_TORCH, train_keras, train_torch
from .evaluation import baseline_metrics, compare_models
from .preprocessing import build_preprocessor, build_route_stats, get_numpy_arrays
from .sklearn_models import train_all, tune_best


def run_training(
    dataset_path: str = "cleaned_dataset.csv",
    model_out: str = "model.joblib",
    route_stats_out: str = "route_stats.csv",
    models_dir: str = "models",
    enable_torch: bool = False,
    enable_tf: bool = False,
) -> tuple[object, dict, pd.DataFrame]:
    """
    Pipeline d'entraînement complet.
    Retourne (best_pipeline, all_results, results_df).
    Sauvegarde tous les modèles dans models/ et model.joblib.
    """

    print("── Chargement des données ────────────────────────────────")
    df = pd.read_csv(dataset_path, parse_dates=["Date"])
    add_time_features(df)
    route_stats = build_route_stats(df)
    print(f"   {len(df):,} lignes | {len(route_stats)} routes uniques")

    df_model = df.dropna(subset=[TARGET]).copy()
    X = df_model[ALL_FEATURES]
    y = df_model[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"   Train : {len(X_train):,} | Test : {len(X_test):,}")

    print("\n── Prétraitement ─────────────────────────────────────────")
    preprocessor = build_preprocessor()
    X_train_np, X_test_np, y_train_np, y_test_np = get_numpy_arrays(
        preprocessor, X_train, X_test, y_train, y_test
    )
    print(f"   Dimensions après encodage : {X_train_np.shape[1]}")

    baseline = baseline_metrics(y_train_np, y_test_np)
    print(f"\n── Baseline — RMSE : {baseline['RMSE']:.3f} | R² : {baseline['R2']:.4f}")

    print("\n── sklearn / XGBoost / LightGBM ──────────────────────────")
    all_results, all_pipelines = train_all(preprocessor, X_train, y_train, X_test, y_test)

    if enable_torch and HAS_TORCH:
        print("\n── PyTorch MLP ───────────────────────────────────────────")
        _, torch_metrics = train_torch(X_train_np, y_train_np, X_test_np, y_test_np)
        all_results["PyTorch MLP"] = torch_metrics
        print(f"   RMSE : {torch_metrics['RMSE']:.3f} | R² : {torch_metrics['R2']:.4f}")
    elif enable_torch and not HAS_TORCH:
        print("\n── PyTorch MLP : non disponible (pip install torch) ──────")

    if enable_tf and HAS_TF:
        print("\n── TensorFlow MLP ────────────────────────────────────────")
        _, keras_metrics, _ = train_keras(X_train_np, y_train_np, X_test_np, y_test_np)
        all_results["TensorFlow MLP"] = keras_metrics
        print(f"   RMSE : {keras_metrics['RMSE']:.3f} | R² : {keras_metrics['R2']:.4f}")
    elif enable_tf and not HAS_TF:
        print("\n── TensorFlow MLP : non disponible (pip install tensorflow) ─")

    sklearn_only = {k: v for k, v in all_results.items()
                    if k not in ("PyTorch MLP", "TensorFlow MLP")}
    best_name = min(sklearn_only, key=lambda k: sklearn_only[k]["RMSE"])
    print(f"\n── Optimisation : {best_name} ────────────────────────────")
    best_pipeline, tuned_metrics = tune_best(
        preprocessor, best_name, X_train, y_train, X_test, y_test
    )
    tuned_name = f"{best_name} (tuned)"
    all_results[tuned_name] = tuned_metrics
    all_pipelines[tuned_name] = best_pipeline

    results_df = compare_models(all_results, baseline)
    print("\n── Classement final ──────────────────────────────────────")
    print(results_df[["RMSE", "MAE", "R2"]].to_string())

    os.makedirs(models_dir, exist_ok=True)
    metadata = {}

    for name, pipe in all_pipelines.items():
        slug = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        path = os.path.join(models_dir, f"{slug}.joblib")
        joblib.dump({"pipeline": pipe, "model_name": name}, path)
        metadata[name] = {"file": path, **all_results.get(name, {})}
        print(f"   Sauvegardé : {path}")

    joblib.dump({"pipeline": best_pipeline, "model_name": tuned_name}, model_out)
    route_stats.to_csv(route_stats_out, index=False)

    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n   ✓ {len(all_pipelines)} modèles dans {models_dir}/")
    print(f"   ✓ Meilleur modèle : {tuned_name}")
    print(f"   ✓ {model_out} + {route_stats_out}")

    return best_pipeline, all_results, results_df


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cleaned_dataset.csv"
    run_training(dataset_path=dataset)
