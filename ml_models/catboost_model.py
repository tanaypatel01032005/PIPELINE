"""
catboost_model.py — ML Model: CatBoost Gradient Boosting.

Ordered boosting reduces overfitting; strong regularisation defaults need less tuning.
Feature importance uses PredictionValuesChange (SHAP-like, more reliable ranking).
Outputs: data/predictions/CatBoost_predictions.csv
         data/results/ml_metrics.csv
         data/results/CatBoost_feature_importance.csv
         plots/CatBoost_{actual_vs_predicted,residuals,feature_importance}.png
"""

import os, sys, time, logging
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_preparation.model_utils import (
    load_model_data, compute_metrics, save_metrics_row, save_predictions,
    save_feature_importance, plot_actual_vs_predicted, plot_residuals, plot_feature_importance,
)

METRICS_CSV  = "data/results/ml_metrics.csv"
PLOTS_DIR    = "plots"
MODEL_NAME   = "CatBoost"
RANDOM_STATE = 42

PARAMS = dict(
    iterations          = 500,
    depth               = 6,
    learning_rate       = 0.05,
    l2_leaf_reg         = 3.0,
    random_strength     = 1.0,
    bagging_temperature = 1.0,
    loss_function       = "RMSE",
    random_seed         = RANDOM_STATE,
    verbose             = 0,    # suppress CatBoost training logs
    thread_count        = -1,   # use all CPU cores
)

os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run():
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("CatBoost is not installed.  Install it with:\n    pip install catboost")

    logger.info("=" * 60)
    logger.info(f"ML MODEL: {MODEL_NAME}")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_model_data()
    feature_names = X_train.columns.tolist()
    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    model = CatBoostRegressor(**PARAMS)
    t0 = time.perf_counter()
    model.fit(
        X_train.values, y_train.values,
        eval_set=(X_test.values, y_test.values),
        use_best_model=False,   # use all iterations, not early stopping
    )
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.2f} s")

    t1 = time.perf_counter()
    y_pred = model.predict(X_test.values)
    pred_time = time.perf_counter() - t1

    metrics     = compute_metrics(y_test.values, y_pred, MODEL_NAME)
    importances = model.get_feature_importance()

    save_predictions(y_test, y_pred, MODEL_NAME)
    save_metrics_row(metrics, METRICS_CSV)
    fi_df = save_feature_importance(feature_names, importances, MODEL_NAME)
    plot_feature_importance(feature_names, importances, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_feature_importance.png", top_n=20)
    plot_actual_vs_predicted(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_actual_vs_predicted.png")
    plot_residuals(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_residuals.png")

    print("\n" + "=" * 60)
    print(f"  {MODEL_NAME} — Evaluation Summary")
    print("=" * 60)
    for p_key, p_val in PARAMS.items():
        print(f"  {p_key:<22s}: {p_val}")
    print(f"\n  Training time        : {train_time:.2f} s")
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