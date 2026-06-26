"""
elastic_net.py — Baseline: Elastic Net (L1 + L2 Regularisation).

Combines L1 and L2 penalties; l1_ratio controls the mix (1.0 = pure Lasso,
0.0 = pure Ridge). Preferable to pure Lasso when features are highly
correlated (e.g. lagged log returns), as Lasso arbitrarily picks one
from a correlated group.
Outputs: data/predictions/ElasticNet_predictions.csv
         data/results/baseline_metrics.csv
         plots/ElasticNet_{actual_vs_predicted,residuals}.png
"""

import os, sys, time, logging
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

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
MODEL_NAME  = "ElasticNet"
ALPHA       = 0.001   # overall regularisation strength
L1_RATIO    = 0.5     # 50% L1, 50% L2
MAX_ITER    = 10_000

os.makedirs(PLOTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run():
    logger.info("=" * 60)
    logger.info(f"BASELINE MODEL: {MODEL_NAME}  (alpha={ALPHA}, l1_ratio={L1_RATIO})")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = load_model_data()

    model = ElasticNet(alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=MAX_ITER, random_state=42)
    t0 = time.perf_counter()
    model.fit(X_train.values, y_train.values)
    train_time = time.perf_counter() - t0
    logger.info(f"Training complete in {train_time:.4f} s")

    n_zero    = int(np.sum(model.coef_ == 0))
    n_nonzero = int(np.sum(model.coef_ != 0))
    logger.info(f"Non-zero coefficients: {n_nonzero} / {len(model.coef_)}")

    t1 = time.perf_counter()
    y_pred = model.predict(X_test.values)
    pred_time = time.perf_counter() - t1

    metrics = compute_metrics(y_test.values, y_pred, MODEL_NAME)
    save_predictions(y_test, y_pred, MODEL_NAME)
    save_metrics_row(metrics, METRICS_CSV)
    plot_actual_vs_predicted(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_actual_vs_predicted.png")
    plot_residuals(y_test, y_pred, MODEL_NAME, f"{PLOTS_DIR}/{MODEL_NAME}_residuals.png")

    print("\n" + "=" * 60)
    print(f"  {MODEL_NAME} (alpha={ALPHA}, l1_ratio={L1_RATIO}) — Evaluation Summary")
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


if __name__ == "__main__":
    run()