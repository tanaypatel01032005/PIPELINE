"""
build_features.py — Streamlined Feature Engineering Stage
Constructs key, non-redundant features for ZAR-commodity forecasting.
Uses only the 9 Granger-causality selected commodities at 95% confidence,
applying only their optimal lag and best rolling statistics.

Input : data/processed/preprocessed_data.csv
        data/results/correlation_granger_results.csv
        data/results/log_return_data.csv
Output: data/processed/features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


def select_best(candidates, target):
    """Return the key of the candidate series with highest abs correlation to target."""
    best_key, best_corr = None, -1
    for key, series in candidates.items():
        corr = series.shift(1).corr(target)
        if pd.notna(corr) and abs(corr) > best_corr:
            best_corr = abs(corr)
            best_key = key
    return best_key


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    preprocessed_path = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
    log_return_path = PROJECT_ROOT / "data" / "results" / "log_returns.csv"
    granger_path = PROJECT_ROOT / "data" / "results" / "correlation_granger_analysis.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "engineered_features.csv"

    print("Loading datasets...")
    price_df = pd.read_csv(preprocessed_path)
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    price_df.set_index("Date", inplace=True)

    logret_df = pd.read_csv(log_return_path)
    logret_df["Date"] = pd.to_datetime(logret_df["Date"])
    logret_df.set_index("Date", inplace=True)

    print("Combining datasets...")
    combined_df = price_df.join(logret_df, how="inner")
    combined_df.sort_index(inplace=True)

    # Load Granger results and filter for significant commodities at 95% confidence (alpha = 0.05)
    granger_df = pd.read_csv(granger_path)
    significant_df = granger_df[granger_df['Granger_Significant'] == 'Yes']
    selected_commodities = significant_df['Commodity'].tolist()
    best_lags = dict(zip(significant_df['Commodity'], significant_df['Best_Lag']))

    print(f"Selected {len(selected_commodities)} Granger-significant commodities at 95% confidence:")
    for comm in selected_commodities:
        print(f"  - {comm} (Best Lag: {best_lags[comm]})")

    # Validate required columns are present before building any features
    required_cols = ["usd_zar", "usd_zar_logret"] + selected_commodities
    missing_cols = [c for c in required_cols if c not in combined_df.columns]
    if missing_cols:
        print(f"Error: Missing required column(s): {missing_cols}")
        return

    features_df = pd.DataFrame(index=combined_df.index)
    target = combined_df["usd_zar_logret"]

    features_df["usd_zar_logret"] = target
    features_df["usd_zar_logret_next"] = target.shift(-1)
    features_df["usd_zar"] = combined_df["usd_zar"]

    print("\nBuilding features...")

    # --- 1. Target Lags ---
    features_df["usd_zar_logret_lag_1"] = target.shift(1)
    features_df["usd_zar_logret_lag_2"] = target.shift(2)
    features_df["usd_zar_logret_lag_3"] = target.shift(3)
    features_df["usd_zar_logret_lag_4"] = target.shift(4)
    features_df["usd_zar_logret_lag_5"] = target.shift(5)

    # --- 2. Commodity Lags at optimal Best_Lag ---
    for comm in selected_commodities:
        best_lag = best_lags[comm]
        features_df[f"{comm}_lag_{best_lag}"] = combined_df[comm].shift(best_lag)

    # --- 3. Rolling Mean & Std: search windows 5-60, pick highest abs correlation ---
    ROLLING_WINDOWS = range(5, 61)
    commodity_params = {}  # store selected windows per commodity for the summary

    for comm in selected_commodities:
        series = combined_df[comm]

        mean_candidates = {w: series.rolling(w).mean() for w in ROLLING_WINDOWS}
        best_mean_window = select_best(mean_candidates, target)
        features_df[f"{comm}_roll_mean_{best_mean_window}"] = mean_candidates[best_mean_window].shift(1)

        std_candidates = {w: series.rolling(w).std() for w in ROLLING_WINDOWS}
        best_std_window = select_best(std_candidates, target)
        features_df[f"{comm}_roll_std_{best_std_window}"] = std_candidates[best_std_window].shift(1)

        commodity_params[comm] = (best_mean_window, best_std_window)
        print(f"{comm}: Best Rolling Mean = {best_mean_window}, Best Rolling Std = {best_std_window}")

    # --- 4. Spreads (shifted by 1) ---
    # Gold-Silver spread
    if "Gold_logret" in combined_df.columns and "Silver_logret" in combined_df.columns:
        features_df["gold_silver_spread"] = (combined_df["Gold_logret"] - combined_df["Silver_logret"]).shift(1)
    # Brent-WTI spread
    if "Brent_Oil_logret" in combined_df.columns and "WTI_Crude_Oil_logret" in combined_df.columns:
        features_df["brent_wti_spread"] = (combined_df["Brent_Oil_logret"] - combined_df["WTI_Crude_Oil_logret"]).shift(1)
    # Platinum-Palladium spread
    if "platinum_logret" in combined_df.columns and "Palladium_logret" in combined_df.columns:
        features_df["platinum_palladium_spread"] = (combined_df["platinum_logret"] - combined_df["Palladium_logret"]).shift(1)

    # --- 5. Volatility Regime: search windows 5-60, pick highest abs correlation ---
    VOL_WINDOWS = range(5, 61)
    vol_candidates = {w: target.rolling(w).std() for w in VOL_WINDOWS}
    best_vol_window = select_best(vol_candidates, target)

    vol = vol_candidates[best_vol_window]
    features_df[f"usd_zar_logret_roll_std_{best_vol_window}"] = vol.shift(1)

    vol_lag = vol.shift(1)
    expanding_median = vol_lag.shift(1).expanding(min_periods=20).median()
    features_df["usd_zar_high_vol_flag"] = (vol_lag > expanding_median).astype(float)
    features_df.loc[expanding_median.isna(), "usd_zar_high_vol_flag"] = np.nan

    # --- 6. Technical Indicators (shifted by 1) ---
    price = combined_df["usd_zar"]

    # RSI: search periods 5-30, pick highest abs correlation
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    RSI_PERIODS = range(5, 31)
    rsi_candidates = {}
    for p in RSI_PERIODS:
        avg_gain = gain.ewm(com=p - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=p - 1, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi_candidates[p] = 100 - (100 / (1 + rs))

    best_rsi_period = select_best(rsi_candidates, target)
    print(f"Best RSI Period = {best_rsi_period}")
    features_df[f"usd_zar_rsi_{best_rsi_period}"] = rsi_candidates[best_rsi_period].shift(1)

    # MACD(12,26): fixed, not optimized
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    features_df["usd_zar_macd"] = macd.shift(1)

    # Bollinger Band Width: search windows 10-40, pick highest abs correlation
    BB_WINDOWS = range(10, 41)
    bb_candidates = {}
    for w in BB_WINDOWS:
        middle = price.rolling(w).mean()
        std = price.rolling(w).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        bb_candidates[w] = (upper - lower) / middle

    best_bb_window = select_best(bb_candidates, target)
    print(f"Best Bollinger Window = {best_bb_window}")
    features_df[f"usd_zar_bollinger_width_{best_bb_window}"] = bb_candidates[best_bb_window].shift(1)

    # --- 7. Calendar Indicators ---
    month = combined_df.index.month
    features_df["month_sin"] = np.sin(2 * np.pi * month / 12)
    features_df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Save output
    features_df.to_csv(output_path, index=True)
    print(f"\nSaved {features_df.shape[1]} features to {output_path}")
    print(f"Shape: {features_df.shape}")

    # Clean summary of all selected parameters
    print("\n" + "-" * 41)
    print("Selected Feature Parameters")
    print("-" * 41)
    for comm, (mean_w, std_w) in commodity_params.items():
        print(comm)
        print(f"  Rolling Mean Window : {mean_w}")
        print(f"  Rolling Std Window  : {std_w}")
    print(f"USD/ZAR Volatility Window : {best_vol_window}")
    print(f"RSI Period                : {best_rsi_period}")
    print("MACD                      : (12,26)")
    print(f"Bollinger Window          : {best_bb_window}")
    print("-" * 41)

    # Print leading vs interior NaNs per column
    print("\nNaN count details per column:")
    print(f"{'Column Name':<30} | {'Leading NaNs':<12} | {'Interior NaNs':<13} | {'Total NaNs':<10}")
    print("-" * 75)
    for col in features_df.columns:
        s = features_df[col]
        first_valid = s.first_valid_index()
        if first_valid is None:
            leading = len(s)
            interior = 0
        else:
            valid_pos = s.index.get_loc(first_valid)
            leading = s.iloc[:valid_pos].isna().sum()
            interior = s.iloc[valid_pos:].isna().sum()
        total = leading + interior
        print(f"{col:<30} | {leading:<12} | {interior:<13} | {total:<10}")


if __name__ == "__main__":
    main()