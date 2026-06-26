"""
feature_engineering.py — Stage 4: Build model-ready feature matrix for USD/ZAR forecasting.

Pipeline: Load prices → Load Granger-selected commodities → Validate → Compute log returns →
          Drop leading NaN → Build lag1-5 + rolling mean(5) + rolling std(10) per series →
          Create 1-step-ahead target → Drop NaN rows → Assert integrity → Save
Inputs:  data/processed/mergedFinalData_preprocessed.csv
         data/results/granger_selected_features.csv
Output:  data/results/engineered_features.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PREPROCESSED_CSV = "data/processed/mergedFinalData_preprocessed.csv"
GRANGER_CSV      = "data/results/granger_selected_features.csv"
OUTPUT_CSV       = "data/results/engineered_features.csv"

os.makedirs("data/results", exist_ok=True)

TARGET        = "usd_zar"
TARGET_LOGRET = "usd_zar_logret"
LAG_STEPS     = range(1, 6)
ROLL_MEAN_W   = 5
ROLL_STD_W    = 10


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_columns(df, required, source):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns missing from '{source}':\n  {missing}\n"
            "Check that preprocess.py and adf_test.py have been run first."
        )


def validate_numeric(df, cols):
    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise TypeError(
            f"Non-numeric columns detected (cannot compute log returns):\n  {bad}\n"
            "Re-run preprocess.py to ensure all price columns are numeric."
        )


def replace_non_positive(df, cols):
    # Replace values ≤ 0 with NaN so log() is always defined
    df = df.copy()
    for col in cols:
        mask = df[col] <= 0
        n    = int(mask.sum())
        if n > 0:
            warnings.warn(f"  [WARN] '{col}': {n} non-positive value(s) replaced with NaN.", UserWarning, stacklevel=2)
            df.loc[mask, col] = np.nan
    return df


# ── Feature builders ──────────────────────────────────────────────────────────

def compute_log_return(series):
    return np.log(series).diff()


def lag_features(logret, name):
    return pd.DataFrame(
        {f"{name}_logret_lag{k}": logret.shift(k) for k in LAG_STEPS},
        index=logret.index,
    )


def rolling_features(logret, name):
    return pd.DataFrame(
        {
            f"{name}_logret_rollmean{ROLL_MEAN_W}": logret.rolling(ROLL_MEAN_W).mean(),
            f"{name}_logret_rollstd{ROLL_STD_W}":  logret.rolling(ROLL_STD_W).std(),
        },
        index=logret.index,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    df = pd.read_csv(PREPROCESSED_CSV, index_col="Date", parse_dates=True)
    print(f"\n[1] Loaded prices from '{PREPROCESSED_CSV}'\n    Rows: {len(df)}  Columns: {len(df.columns)}")

    granger_df    = pd.read_csv(GRANGER_CSV)
    selected_cols = [c.replace("_logret_diff1", "").replace("_logret", "")
                     for c in granger_df["Commodity"].tolist()]
    print(f"\n[2] Granger-selected commodities ({len(selected_cols)}): {selected_cols}")

    working_cols = [TARGET] + [c for c in selected_cols if c != TARGET]
    print(f"\n[3] Working columns ({len(working_cols)}): {working_cols}")

    print("\n[4] Validating inputs …")
    validate_columns(df, working_cols, PREPROCESSED_CSV)
    validate_numeric(df, working_cols)
    df_prices = replace_non_positive(df[working_cols].copy(), working_cols)
    assert list(df_prices.columns) == working_cols
    print("    Validation passed. ✓")

    print("\n[5] Computing log returns …")
    logret_df         = df_prices.apply(compute_log_return)
    logret_df.columns = [f"{c}_logret" for c in working_cols]
    logret_df         = logret_df.iloc[1:].copy()
    print(f"    Dropped 1 leading NaN row.  Log-return shape: {logret_df.shape}")

    assert TARGET_LOGRET in logret_df.columns

    print("\n[6] Building feature blocks (lags + rolling stats) …")
    feature_blocks = []
    for col in working_cols:
        lr = logret_df[f"{col}_logret"]
        feature_blocks.append(lr.to_frame())
        feature_blocks.append(lag_features(lr, col))
        feature_blocks.append(rolling_features(lr, col))

    features = pd.concat(feature_blocks, axis=1)
    assert features.columns.duplicated().sum() == 0

    # 1-step-ahead target; shift(-1) creates a trailing NaN removed by dropna below
    features["usd_zar_logret_next"] = features[TARGET_LOGRET].shift(-1)

    n_before = len(features)
    features.dropna(inplace=True)
    print(f"    Rows before NaN drop: {n_before}  |  Removed: {n_before - len(features)}  |  Remaining: {len(features)}")

    print("\n[7] Running integrity checks …")
    assert features.isna().sum().sum() == 0
    assert features.columns.duplicated().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns)
    assert "usd_zar_logret_next" in features.columns
    print("    All integrity checks passed. ✓")

    feature_cols = [c for c in features.columns if c != "usd_zar_logret_next"]
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"  Input commodities (Granger-selected) : {selected_cols}")
    print(f"  Working columns (incl. target)       : {working_cols}")
    print(f"  Final dataset shape                  : {features.shape}")
    print(f"  Number of input features (excl. target): {len(feature_cols)}")
    print(f"\n  Generated feature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"    {i:>3}. {col}")
    print(f"\n  Forecast target column : 'usd_zar_logret_next'")

    features.to_csv(OUTPUT_CSV)
    print(f"\n✅ Saved → {OUTPUT_CSV}")

    return features


if __name__ == "__main__":
    run()