"""
lasso_regression.py — Baseline: Lasso (L1-Regularised) Linear Regression.

L1 penalty encourages sparse solutions — many coefficients driven to zero,
performing automatic feature selection among the 56 engineered features.
Outputs: data/predictions/Lasso_predictions.csv
         data/results/baseline_metrics.csv
         plots/Lasso_{actual_vs_predicted,residuals}.png
"""

import os, sys, time, logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

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
MODEL_NAME  = "Lasso"
ALPHA       = 0.001    # smaller than Ridge — log returns have tiny scale
MAX_ITER    = 10_000   # extra iterations to ensure convergence

os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run():
    logger.info("=" * 60)
    logger.info(f"BASELINE MODEL: {MODEL_NAME}  (alpha={ALPHA})")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_model_data()

    model = Lasso(alpha=ALPHA, max_iter=MAX_ITER, random_state=42)
    t0 = time.perf_counter()
    model.fit(X_train.values, y_train.values)
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.4f} s")

    n_zero    = int(np.sum(model.coef_ == 0))
    n_nonzero = int(np.sum(model.coef_ != 0))
    logger.info(f"Non-zero coefficients: {n_nonzero} / {len(model.coef_)}  ({n_zero} set to zero by L1)")

    t1 = time.perf_counter()
    y_pred = model.predict(X_test.values)
    pred_time = time.perf_counter() - t1

    metrics = compute_metrics(y_test.values, y_pred, MODEL_NAME)
    save_predictions(y_test, y_pred, MODEL_NAME)
    save_metrics_row(metrics, METRICS_CSV)
    plot_actual_vs_predicted(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_actual_vs_predicted.png")
    plot_residuals(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_residuals.png")

    print("\n" + "=" * 60)
    print(f"  {MODEL_NAME} (alpha={ALPHA}) — Evaluation Summary")
    print("=" * 60)
    print(f"  Training time          : {train_time:.4f} s")
    print(f"  Prediction time        : {pred_time*1000:.2f} ms")
    print(f"  Test observations      : {len(y_test)}")
    print(f"  Non-zero coefficients  : {n_nonzero} / {len(model.coef_)}")
    print(f"  Zeroed out (by L1)     : {n_zero}\n")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<18s}: {v}")
    print("=" * 60)

    nonzero_df = (
        pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_})
        .query("Coefficient != 0")
        .iloc[np.argsort(np.abs(model.coef_[model.coef_ != 0]))[::-1]]
    )
    print(f"\n  Lasso-selected features ({len(nonzero_df)}):")
    print(nonzero_df.to_string(index=False))


if __name__ == "__main__":
    run()