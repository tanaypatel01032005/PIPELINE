"""
linear_regression.py — Baseline: Ordinary Least Squares (OLS) Linear Regression.

No regularisation; closed-form solution via normal equations. Provides the
lower bound on model complexity — if complex models don't beat OLS, the
added complexity is not justified.
Outputs: data/predictions/LinearRegression_predictions.csv
         data/results/baseline_metrics.csv
         plots/LinearRegression_{actual_vs_predicted,residuals}.png
"""

import os, sys, time, logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_preparation.model_utils import (
    load_model_data, compute_metrics, save_metrics_row,
    save_predictions, plot_actual_vs_predicted, plot_residuals,
)

METRICS_CSV = "data/results/baseline_metrics.csv"
PLOTS_DIR   = "plots"
MODEL_NAME  = "LinearRegression"

os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run():
    logger.info("=" * 60)
    logger.info(f"BASELINE MODEL: {MODEL_NAME}")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_model_data()
    logger.info(f"Features: {X_train.shape[1]}  |  Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    # Analytical (closed-form) solution — no hyperparameters, no random state needed
    model = LinearRegression()
    t0 = time.perf_counter()
    model.fit(X_train.values, y_train.values)
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.4f} s")

    t1 = time.perf_counter()
    y_pred = model.predict(X_test.values)
    pred_time = time.perf_counter() - t1

    metrics = compute_metrics(y_test.values, y_pred, MODEL_NAME)
    save_predictions(y_test, y_pred, MODEL_NAME)
    save_metrics_row(metrics, METRICS_CSV)
    plot_actual_vs_predicted(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_actual_vs_predicted.png")
    plot_residuals(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_residuals.png")

    print("\n" + "=" * 60)
    print(f"  {MODEL_NAME} — Evaluation Summary")
    print("=" * 60)
    print(f"  Training time     : {train_time:.4f} s")
    print(f"  Prediction time   : {pred_time*1000:.2f} ms")
    print(f"  Test observations : {len(y_test)}\n")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<18s}: {v}")
    print("=" * 60)

    coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_}) \
                .reindex(pd.Index(np.argsort(np.abs(model.coef_))[::-1])).head(10)
    print("\n  Top-10 coefficients by magnitude:")
    print(coef_df.to_string(index=False))


if __name__ == "__main__":
    run()