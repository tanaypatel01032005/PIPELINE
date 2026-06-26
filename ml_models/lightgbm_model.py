"""
lightgbm_model.py — ML Model: LightGBM (Light Gradient Boosting Machine).

Histogram-based learning and leaf-wise tree growth; faster and lower memory than XGBoost
on large datasets. Leaf-wise splitting controlled by min_child_samples to avoid overfitting.
Outputs: data/predictions/LightGBM_predictions.csv
         data/results/ml_metrics.csv
         data/results/LightGBM_feature_importance.csv
         plots/LightGBM_{actual_vs_predicted,residuals,feature_importance}.png
"""

import os, sys, time, logging
import numpy as np
import pandas as _pd

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
MODEL_NAME   = "LightGBM"
RANDOM_STATE = 42

PARAMS = dict(
    n_estimators      = 500,
    max_depth         = -1,    # unlimited; num_leaves controls complexity
    num_leaves        = 31,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_samples = 20,    # minimum data per leaf (overfitting guard)
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    objective         = "regression",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,    # suppress LightGBM logs
)

os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run():
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("LightGBM is not installed.  Install it with:\n    pip install lightgbm")

    logger.info("=" * 60)
    logger.info(f"ML MODEL: {MODEL_NAME}")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_model_data()
    feature_names = X_train.columns.tolist()
    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    model = LGBMRegressor(**PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)])
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.2f} s")

    t1 = time.perf_counter()
    # Pass DataFrame to avoid LightGBM "feature names" UserWarning
    y_pred = model.predict(_pd.DataFrame(X_test.values, columns=feature_names))
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