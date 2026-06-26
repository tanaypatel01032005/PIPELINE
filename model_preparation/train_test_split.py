"""
train_test_split.py — Stage 5: Chronological 80/20 Train/Test Split.

No shuffling — shuffling a time series creates data leakage (future observations in training set).
Input:   data/results/engineered_features.csv
Outputs: data/model_input/{X_train,X_test,y_train,y_test}.csv
         data/results/train_test_split_summary.txt
"""

import os, sys, logging
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_CSV   = "data/results/engineered_features.csv"
OUT_DIR     = "data/model_input"
SUMMARY_TXT = "data/results/train_test_split_summary.txt"
TARGET      = "usd_zar_logret_next"
TRAIN_RATIO = 0.80

os.makedirs(OUT_DIR,        exist_ok=True)
os.makedirs("data/results", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def validate_dataframe(df):
    """Check: target present, no NaN, no duplicate columns, all numeric, sorted DatetimeIndex."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in '{INPUT_CSV}'.\nRun feature_engineering.py first.")

    n_nan = int(df.isna().sum().sum())
    if n_nan > 0:
        raise ValueError(f"{n_nan} NaN value(s) found in {df.columns[df.isna().any()].tolist()}\nRe-run feature_engineering.py.")

    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"Duplicate column names detected: {dupes}")

    non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise TypeError(f"Non-numeric columns found: {non_num}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index is not a DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index is not sorted in ascending chronological order.")

    logger.info("Pre-split validation passed. [OK]")


def chronological_split(df):
    """Split df into (X_train, y_train, X_test, y_test) by row count; no shuffling."""
    n_train = int(len(df) * TRAIN_RATIO)
    X, y    = df.drop(columns=[TARGET]), df[TARGET]

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test,  y_test  = X.iloc[n_train:], y.iloc[n_train:]

    assert X_train.index.max() < X_test.index.min(), "Data leakage: training and test date ranges overlap."
    assert X_train.index.is_monotonic_increasing and X_test.index.is_monotonic_increasing
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert list(X_train.columns) == list(X_test.columns)

    logger.info("Chronological split assertions passed. [OK]")
    return X_train, y_train, X_test, y_test


def run():
    logger.info("=" * 60)
    logger.info("STEP 5 — TRAIN / TEST SPLIT")
    logger.info("=" * 60)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Engineered features not found: '{INPUT_CSV}'\nRun feature_engineering.py first.")

    df = pd.read_csv(INPUT_CSV, index_col="Date", parse_dates=True)
    logger.info(f"Loaded '{INPUT_CSV}'  ({df.shape[0]} rows x {df.shape[1]} cols)")

    validate_dataframe(df)
    X_train, y_train, X_test, y_test = chronological_split(df)

    n_total, n_train, n_test = len(df), len(X_train), len(X_test)

    X_train.to_csv(f"{OUT_DIR}/X_train.csv")
    X_test.to_csv( f"{OUT_DIR}/X_test.csv")
    y_train.to_frame().to_csv(f"{OUT_DIR}/y_train.csv")
    y_test.to_frame().to_csv( f"{OUT_DIR}/y_test.csv")
    for f in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        logger.info(f"Saved -> {OUT_DIR}/{f}")

    lines = [
        "=" * 65,
        "TRAIN / TEST SPLIT SUMMARY",
        "=" * 65,
        f"  Input file           : {INPUT_CSV}",
        f"  Target variable      : {TARGET}",
        f"  Split ratio          : {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test",
        "",
        f"  Total rows           : {n_total:>6}",
        f"  Training rows        : {n_train:>6}  ({100*n_train/n_total:.1f}%)",
        f"  Testing rows         : {n_test:>6}  ({100*n_test/n_total:.1f}%)",
        f"  Number of features   : {X_train.shape[1]:>6}",
        "",
        f"  Training date range  : {X_train.index.min().date()} --> {X_train.index.max().date()}",
        f"  Testing  date range  : {X_test.index.min().date()} --> {X_test.index.max().date()}",
        "",
        "  Data leakage check   :",
        f"    Max train date ({X_train.index.max().date()}) < Min test date ({X_test.index.min().date()})  [OK]",
        "",
        "  Output files:",
        f"    {OUT_DIR}/X_train.csv    ({X_train.shape})",
        f"    {OUT_DIR}/X_test.csv     ({X_test.shape})",
        f"    {OUT_DIR}/y_train.csv    ({y_train.shape})",
        f"    {OUT_DIR}/y_test.csv     ({y_test.shape})",
        "=" * 65,
    ]

    print("\n".join(lines))
    with open(SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info(f"Summary saved -> {SUMMARY_TXT}")


if __name__ == "__main__":
    run()