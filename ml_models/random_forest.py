"""
random_forest.py — ML Model: Random Forest Regressor.

Ensemble of decision trees on bootstrapped subsamples; random feature subsets per split
reduce tree correlation and improve generalisation. Feature importance = mean decrease
in node impurity (MDI). Robust to outliers common in financial log-return data.
Outputs: data/predictions/RandomForest_predictions.csv
         data/results/ml_metrics.csv
         data/results/RandomForest_feature_importance.csv
         plots/RandomForest_{actual_vs_predicted,residuals,feature_importance}.png
"""

import os, sys, time, logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_preparation.model_utils import (
    load_model_data, compute_metrics, save_metrics_row, save_predictions,
    save_feature_importance, plot_actual_vs_predicted, plot_residuals, plot_feature_importance,
)

METRICS_CSV      = "data/results/ml_metrics.csv"
PLOTS_DIR        = "plots"
MODEL_NAME       = "RandomForest"
RANDOM_STATE     = 42
N_ESTIMATORS     = 300
MAX_DEPTH        = None  # no limit — controlled by min_samples_leaf
MIN_SAMPLES_LEAF = 5     # prevents overfitting on tiny leaf nodes
N_JOBS           = -1    # use all CPU cores

os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run():
    logger.info("=" * 60)
    logger.info(f"ML MODEL: {MODEL_NAME}")
    logger.info(f"  n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF}, seed={RANDOM_STATE}")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_model_data()
    feature_names = X_train.columns.tolist()
    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )
    t0 = time.perf_counter()
    model.fit(X_train.values, y_train.values)
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.2f} s")

    t1 = time.perf_counter()
    y_pred = model.predict(X_test.values)
    pred_time = time.perf_counter() - t1

    metrics     = compute_metrics(y_test.values, y_pred, MODEL_NAME)
    importances = model.feature_importances_

    save_predictions(y_test, y_pred, MODEL_NAME)
    save_metrics_row(metrics, METRICS_CSV)
    fi_df = save_feature_importance(feature_names, importances, MODEL_NAME)
    plot_feature_importance(feature_names, importances, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_feature_importance.png", top_n=20)
    plot_actual_vs_predicted(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_actual_vs_predicted.png")
    plot_residuals(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_residuals.png")

    print("\n" + "=" * 60)
    print(f"  {MODEL_NAME} — Evaluation Summary")
    print("=" * 60)
    print(f"  n_estimators         : {N_ESTIMATORS}")
    print(f"  max_depth            : {MAX_DEPTH}")
    print(f"  min_samples_leaf     : {MIN_SAMPLES_LEAF}")
    print(f"  Training time        : {train_time:.2f} s")
    print(f"  Prediction time      : {pred_time*1000:.2f} ms")
    print(f"  Test observations    : {len(y_test)}\n")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<18s}: {v}")
    print("\n  Top-10 Important Features:")
    print(fi_df.head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    run()