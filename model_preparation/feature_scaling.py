"""
feature_scaling.py — Stage 6: Scale predictor features using StandardScaler.

Fits scaler on X_train ONLY, then applies to both train and test sets (leakage prevention).
Inputs:  data/model_input/{X_train,X_test,y_train,y_test}.csv
Outputs: data/model_input/{X_train_scaled,X_test_scaled}.csv
         models/scalers/standard_scaler.pkl
         data/results/scaling_summary.txt
"""

import os, sys, logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_INPUT = "data/model_input"
SCALER_DIR  = "models/scalers"
SUMMARY_TXT = "data/results/scaling_summary.txt"
SCALER_PKL  = f"{SCALER_DIR}/standard_scaler.pkl"

for _d in [MODEL_INPUT, SCALER_DIR, "data/results"]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def check_no_nan_inf(df, label):
    n_nan = int(df.isna().sum().sum())
    if n_nan > 0:
        raise ValueError(f"{label}: {n_nan} NaN value(s) found in {df.columns[df.isna().any()].tolist()}")
    if int(np.isinf(df.select_dtypes(include="number").values).sum()) > 0:
        raise ValueError(f"{label}: infinite value(s) found.")


def check_column_consistency(X_train, X_test):
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Feature columns differ between X_train and X_test.\n"
            f"  Train ({len(X_train.columns)}): {list(X_train.columns)[:5]} ...\n"
            f"  Test  ({len(X_test.columns)}):  {list(X_test.columns)[:5]} ..."
        )


def run():
    logger.info("=" * 60)
    logger.info("STEP 6 — FEATURE SCALING  (StandardScaler)")
    logger.info("=" * 60)

    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        if not os.path.exists(f"{MODEL_INPUT}/{fname}"):
            raise FileNotFoundError(f"Required file not found: '{MODEL_INPUT}/{fname}'\nRun train_test_split.py first.")

    X_train = pd.read_csv(f"{MODEL_INPUT}/X_train.csv", index_col="Date", parse_dates=True)
    X_test  = pd.read_csv(f"{MODEL_INPUT}/X_test.csv",  index_col="Date", parse_dates=True)
    # y files loaded for shape reporting only
    y_train = pd.read_csv(f"{MODEL_INPUT}/y_train.csv", index_col="Date", parse_dates=True)
    y_test  = pd.read_csv(f"{MODEL_INPUT}/y_test.csv",  index_col="Date", parse_dates=True)
    logger.info(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    check_no_nan_inf(X_train, "X_train")
    check_no_nan_inf(X_test,  "X_test")
    check_column_consistency(X_train, X_test)
    assert X_train.index.max() < X_test.index.min(), \
        "Chronological ordering violation: test data overlaps with training data."
    logger.info("Input validation passed. [OK]")

    # Fit on X_train only — critical leakage-prevention step
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    logger.info("StandardScaler fitted on X_train only. [OK]")

    feature_names = X_train.columns.tolist()

    def _to_df(arr, idx): return pd.DataFrame(arr, index=idx, columns=feature_names)

    X_train_scaled = _to_df(scaler.transform(X_train.values), X_train.index)
    X_test_scaled  = _to_df(scaler.transform(X_test.values),  X_test.index)

    check_no_nan_inf(X_train_scaled, "X_train_scaled")
    check_no_nan_inf(X_test_scaled,  "X_test_scaled")
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape  == X_test.shape
    assert list(X_train_scaled.columns) == feature_names
    logger.info("Scaled output validation passed. [OK]")

    X_train_scaled.to_csv(f"{MODEL_INPUT}/X_train_scaled.csv")
    X_test_scaled.to_csv( f"{MODEL_INPUT}/X_test_scaled.csv")
    logger.info(f"Saved -> {MODEL_INPUT}/X_train_scaled.csv")
    logger.info(f"Saved -> {MODEL_INPUT}/X_test_scaled.csv")

    joblib.dump(scaler, SCALER_PKL)
    logger.info(f"Scaler saved -> {SCALER_PKL}")

    means, stds = scaler.mean_, scaler.scale_
    lines = [
        "=" * 72,
        "FEATURE SCALING SUMMARY  (StandardScaler — fitted on X_train only)",
        "=" * 72,
        f"  Scaler type              : StandardScaler (zero mean, unit variance)",
        f"  Fitted on                : X_train only  (no data leakage)",
        f"  Feature count            : {len(feature_names)}",
        f"  X_train_scaled shape     : {X_train_scaled.shape}",
        f"  X_test_scaled  shape     : {X_test_scaled.shape}",
        "",
        f"  {'Feature':<42s} {'Train Mean':>12s} {'Train Std':>12s}",
        "  " + "-" * 68,
    ]
    for feat, mu, sigma in zip(feature_names, means, stds):
        lines.append(f"  {feat:<42s} {mu:>12.6f} {sigma:>12.6f}")
    lines.append("=" * 72)

    print("\n".join(lines))
    with open(SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info(f"Summary saved -> {SUMMARY_TXT}")


if __name__ == "__main__":
    run()