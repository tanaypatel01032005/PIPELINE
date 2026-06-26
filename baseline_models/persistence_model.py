"""
persistence_model.py — Baseline: Naive Persistence (Random Walk) Model.

Predicts r_hat_{t+1} = r_t (current log return).  Any advanced model must beat
this baseline to justify its added complexity.  Uses unscaled X_test so predictions
are in the same units as y_test.
Outputs: data/predictions/Persistence_predictions.csv
         data/results/baseline_metrics.csv
         plots/Persistence_{actual_vs_predicted,residuals}.png
"""

import os, sys, time, logging
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_preparation.model_utils import (
    compute_metrics, save_metrics_row, save_predictions,
    plot_actual_vs_predicted, plot_residuals,
)

MODEL_INPUT = "data/model_input"
METRICS_CSV = "data/results/baseline_metrics.csv"
PLOTS_DIR   = "plots"
MODEL_NAME  = "Persistence"

for _d in [PLOTS_DIR, "data/results", "data/predictions"]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run():
    logger.info("=" * 60)
    logger.info(f"BASELINE MODEL: {MODEL_NAME}")
    logger.info("=" * 60)

    x_test_path = f"{MODEL_INPUT}/X_test.csv"
    y_test_path = f"{MODEL_INPUT}/y_test.csv"
    for path in [x_test_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: '{path}'\nRun train_test_split.py first.")

    X_test = pd.read_csv(x_test_path, index_col="Date", parse_dates=True)
    y_test = pd.read_csv(y_test_path, index_col="Date", parse_dates=True).squeeze("columns")
    logger.info(f"X_test: {X_test.shape}  |  y_test: {y_test.shape}")

    if "usd_zar_logret" not in X_test.columns:
        raise KeyError("'usd_zar_logret' not found in X_test — needed as the persistence prediction.")

    # Prediction rule: r_hat(t+1) = r(t)
    t0     = time.perf_counter()
    y_pred = X_test["usd_zar_logret"].values
    pred_time = time.perf_counter() - t0
    logger.info(f"Predictions generated in {pred_time*1000:.2f} ms  |  rule: r_hat(t+1) = r(t)")

    metrics = compute_metrics(y_test.values, y_pred, MODEL_NAME)
    save_predictions(y_test, y_pred, MODEL_NAME)
    save_metrics_row(metrics, METRICS_CSV)
    plot_actual_vs_predicted(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_actual_vs_predicted.png")
    plot_residuals(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_residuals.png")

    print("\n" + "=" * 60)
    print(f"  {MODEL_NAME} Model — Evaluation Summary")
    print("=" * 60)
    print(f"  Training time     : N/A  (no training required)")
    print(f"  Prediction time   : {pred_time*1000:.2f} ms")
    print(f"  Test observations : {len(y_test)}\n")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<18s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    run()