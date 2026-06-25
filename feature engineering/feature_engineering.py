"""
feature_engineering.py
======================
Builds the model-ready feature matrix for the USD/ZAR exchange-rate
forecasting study, following the methodology of the research paper.

  Pipeline:
    1. Load raw (preprocessed) prices from data/processed/mergedFinalData_preprocessed.csv
    2. Validate columns, dtypes, and values before transformation
    3. Keep only the Granger-selected commodities + target (usd_zar)
    4. Compute log returns:  logret = log(P_t) - log(P_{t-1})
       Non-positive values are masked as NaN before log to avoid -inf / NaN.
    5. For every series build:
         - lag features  : logret_lag1 ... logret_lag5
         - rolling mean  : window = 5  (short-term trend)
         - rolling std   : window = 10 (recent volatility / risk)
    6. Create the 1-step-ahead forecast target:
         usd_zar_logret_next = usd_zar_logret.shift(-1)
    7. Drop all NaN rows (from log-return, lags, rolling windows, forward shift)
    8. Save to data/results/engineered_features.csv

Inputs
------
  data/processed/mergedFinalData_preprocessed.csv   -- raw / cleaned prices
  data/results/granger_selected_features.csv        -- Granger-significant commodities

Output
------
  data/results/engineered_features.csv
"""

import os
import warnings
import numpy as np
import pandas as pd

# == Paths =====================================================================
PREPROCESSED_CSV = "data/processed/mergedFinalData_preprocessed.csv"
GRANGER_CSV      = "data/results/granger_selected_features.csv"
OUTPUT_CSV       = "data/results/engineered_features.csv"

# == Hyper-parameters (consistent with paper) =================================
TARGET       = "usd_zar"
LAG_STEPS    = range(1, 6)   # lag1 ... lag5
ROLL_MEAN_W  = 5             # rolling mean window  -> short-term trend
ROLL_STD_W   = 10            # rolling std  window  -> volatility proxy

os.makedirs("data/results", exist_ok=True)


# =============================================================================
# Validation helpers
# =============================================================================

def validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    """
    Raise KeyError if any required column is absent from df.

    Parameters
    ----------
    df       : pd.DataFrame  -- data to check
    required : list[str]     -- column names that must be present
    source   : str           -- filename string used in the error message
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following columns are missing from {source}: {missing}"
        )


def validate_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Raise TypeError if any column in cols is not numeric.

    Log returns require numeric input; object / string columns must be
    caught early so the error is clear rather than cryptic.

    Parameters
    ----------
    df   : pd.DataFrame  -- price data
    cols : list[str]     -- columns to check
    """
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(
            f"Non-numeric columns found (cannot compute log returns): {non_numeric}"
        )


def validate_positive_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Replace non-positive (zero or negative) price values with NaN.

    log(x) is undefined for x <= 0; replacing with NaN propagates cleanly
    through .diff() and is then removed by the final dropna() call.
    A warning is printed for each affected column so the researcher is aware.

    Parameters
    ----------
    df   : pd.DataFrame  -- price data (modified copy is returned)
    cols : list[str]     -- columns to sanitise

    Returns
    -------
    pd.DataFrame with non-positive entries replaced by NaN
    """
    df = df.copy()
    for col in cols:
        mask = df[col] <= 0
        n_invalid = int(mask.sum())
        if n_invalid > 0:
            warnings.warn(
                f"  [WARN] '{col}': {n_invalid} non-positive value(s) replaced with NaN "
                f"before log transformation.",
                UserWarning,
                stacklevel=2,
            )
            df.loc[mask, col] = np.nan
    return df


# =============================================================================
# Feature builders
# =============================================================================

def compute_log_return(series: pd.Series) -> pd.Series:
    """
    Compute daily log return:  r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Log returns are stationary for most financial price series and are the
    standard input representation used in the paper.

    Non-positive values must already be replaced with NaN before calling this
    function (see validate_positive_values).  NaN entries propagate safely
    through np.log and .diff() without producing -inf.
    """
    return np.log(series).diff()


def lag_features(logret: pd.Series, name: str) -> pd.DataFrame:
    """
    Create lag-1 ... lag-5 features from a log-return series.

    Lag features give the model access to the recent history of each series,
    allowing it to capture auto-regressive dynamics.

    Parameters
    ----------
    logret : pd.Series   -- log-return series for one variable
    name   : str         -- base column name (e.g. 'Gold')

    Returns
    -------
    pd.DataFrame with columns: <name>_logret_lag1 ... <name>_logret_lag5
    """
    return pd.DataFrame(
        {f"{name}_logret_lag{k}": logret.shift(k) for k in LAG_STEPS}
    )


def rolling_features(logret: pd.Series, name: str) -> pd.DataFrame:
    """
    Create rolling-window statistics from a log-return series.

      - Rolling mean  (window=5)  -- recent directional trend
      - Rolling std   (window=10) -- recent volatility / risk level

    Parameters
    ----------
    logret : pd.Series   -- log-return series for one variable
    name   : str         -- base column name

    Returns
    -------
    pd.DataFrame with columns:
        <name>_logret_rollmean5, <name>_logret_rollstd10
    """
    return pd.DataFrame({
        f"{name}_logret_rollmean{ROLL_MEAN_W}": logret.rolling(ROLL_MEAN_W).mean(),
        f"{name}_logret_rollstd{ROLL_STD_W}":  logret.rolling(ROLL_STD_W).std(),
    })


# =============================================================================
# Main pipeline
# =============================================================================

def run() -> pd.DataFrame:
    # -- 1. Load raw preprocessed prices --------------------------------------
    df = pd.read_csv(PREPROCESSED_CSV, index_col="Date", parse_dates=True)

    # -- 2. Load Granger-selected commodities ---------------------------------
    granger_df    = pd.read_csv(GRANGER_CSV)
    selected_cols = granger_df["Commodity"].tolist()

    print("=" * 60)
    print(f"Selected commodities ({len(selected_cols)}): {selected_cols}")
    print("=" * 60)

    # Working set: target variable + Granger-selected commodities.
    # Guard against TARGET accidentally appearing in the selected list.
    working_cols = [TARGET] + [c for c in selected_cols if c != TARGET]

    # -- 3. Validate inputs ---------------------------------------------------

    # 3a. Confirm all required columns are present in the CSV
    validate_columns(df, working_cols, PREPROCESSED_CSV)

    # 3b. Confirm all working columns are numeric (float / int)
    validate_numeric(df, working_cols)

    df_prices = df[working_cols]

    # 3c. Replace any non-positive prices with NaN before log transformation.
    #     This prevents np.log from producing -inf or raising errors.
    df_prices = validate_positive_values(df_prices, working_cols)

    # -- 4. Compute log returns for every series ------------------------------
    # logret_t = ln(P_t) - ln(P_{t-1})
    # The first row becomes NaN after .diff() and is removed later.
    logret_df = df_prices.apply(compute_log_return)
    logret_df.columns = [f"{c}_logret" for c in working_cols]

    # -- 5. Build feature blocks ----------------------------------------------
    feature_blocks = []

    for col in working_cols:
        logret_series = logret_df[f"{col}_logret"]

        # 5a. Log-return level (the series itself as a feature)
        feature_blocks.append(logret_series.to_frame())

        # 5b. Lag features (lag1 ... lag5) of the log-return series
        feature_blocks.append(lag_features(logret_series, col))

        # 5c. Rolling statistics (mean-5, std-10) of the log-return series
        feature_blocks.append(rolling_features(logret_series, col))

    features = pd.concat(feature_blocks, axis=1)

    # -- 6. Create 1-step-ahead forecast target -------------------------------
    # usd_zar_logret_next is what the model is trained to predict.
    features["usd_zar_logret_next"] = features["usd_zar_logret"].shift(-1)

    # -- 7. Drop all NaN rows -------------------------------------------------
    # Sources of NaN:
    #   - log-return .diff()              -> 1 leading NaN
    #   - lag5                            -> 5 leading NaNs
    #   - rolling std (w=10)              -> 9 leading NaNs  (dominant)
    #   - target shift(-1)                -> 1 trailing NaN
    #   - any NaN from non-positive mask  -> propagated and removed here
    n_before = len(features)
    features.dropna(inplace=True)
    n_dropped = n_before - len(features)

    # -- 8. Save --------------------------------------------------------------
    features.to_csv(OUTPUT_CSV)

    # -- Summary --------------------------------------------------------------
    # Feature count excludes the prediction target column
    n_features = len(features.columns) - 1

    print(f"\nLog-return features computed for : {working_cols}")
    print(f"Rows dropped (NaN cleanup)       : {n_dropped}")
    print(f"Number of generated features     : {n_features}")
    print(f"Final dataset shape              : {features.shape}")
    print(f"\nFeature columns:\n  {list(features.columns)}")
    print(f"\nSaved -> {OUTPUT_CSV}")

    return features


if __name__ == "__main__":
    run()
