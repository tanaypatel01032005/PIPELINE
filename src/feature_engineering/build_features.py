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

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    preprocessed_path = PROJECT_ROOT / "data" / "processed" / "preprocessed_data.csv"
    log_return_path = PROJECT_ROOT / "data" / "results" / "log_return_data.csv"
    granger_path = PROJECT_ROOT / "data" / "results" / "correlation_granger_results.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    
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
    
    # 1. Load Granger results and filter for significant commodities at 95% confidence (alpha = 0.05)
    granger_df = pd.read_csv(granger_path)
    significant_df = granger_df[granger_df['Granger_Significant'] == 'Yes']
    selected_commodities = significant_df['Commodity'].tolist()
    best_lags = dict(zip(significant_df['Commodity'], significant_df['Best_Lag']))
    
    print(f"Selected {len(selected_commodities)} Granger-significant commodities at 95% confidence:")
    for comm in selected_commodities:
        print(f"  - {comm} (Best Lag: {best_lags[comm]})")
        
    features_df = pd.DataFrame(index=combined_df.index)
    
    print("\nBuilding features...")
    
    # --- 1. Target Lags ---
    features_df["usd_zar_logret_lag_1"] = combined_df["usd_zar_logret"].shift(1)
    features_df["usd_zar_logret_lag_2"] = combined_df["usd_zar_logret"].shift(2)
    
    # --- 2. Commodity Lags at optimal Best_Lag ---
    for comm in selected_commodities:
        best_lag = best_lags[comm]
        features_df[f"{comm}_lag_{best_lag}"] = combined_df[comm].shift(best_lag)
        
    # --- 3. Commodity Rolling Statistics (10-day window, shifted by 1) ---
    for comm in selected_commodities:
        features_df[f"{comm}_roll_mean_10"] = combined_df[comm].rolling(window=10).mean().shift(1)
        features_df[f"{comm}_roll_std_10"] = combined_df[comm].rolling(window=10).std().shift(1)
        
    # --- 4. Spreads (shifted by 1) ---
    # Gold-Silver spread
    if "Gold_logret" in combined_df.columns and "Silver_logret" in combined_df.columns:
        features_df["gold_silver_spread"] = (combined_df["Gold_logret"] - combined_df["Silver_logret"]).shift(1)
    # Brent-WTI spread
    if "Brent_Oil_logret" in combined_df.columns and "WTI Crude Oil_logret" in combined_df.columns:
        features_df["brent_wti_spread"] = (combined_df["Brent_Oil_logret"] - combined_df["WTI Crude Oil_logret"]).shift(1)
    # Platinum-Palladium spread
    if "platinum_logret" in combined_df.columns and "Palladium_logret" in combined_df.columns:
        features_df["platinum_palladium_spread"] = (combined_df["platinum_logret"] - combined_df["Palladium_logret"]).shift(1)
        
    # --- 5. Volatility Regime (shifted by 1) ---
    vol_20 = combined_df["usd_zar_logret"].rolling(window=20).std()
    features_df["usd_zar_logret_roll_std_20"] = vol_20.shift(1)
    
    vol_lag = vol_20.shift(1)
    expanding_median = vol_lag.shift(1).expanding(min_periods=20).median()
    features_df["usd_zar_high_vol_flag"] = (vol_lag > expanding_median).astype(float)
    features_df.loc[expanding_median.isna(), "usd_zar_high_vol_flag"] = np.nan
    
    # --- 6. Technical Indicators (shifted by 1) ---
    # RSI(14)
    price = combined_df["usd_zar"]
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    features_df["usd_zar_rsi_14"] = rsi.shift(1)
    
    # MACD(12,26,9) Line
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    features_df["usd_zar_macd"] = macd.shift(1)
    
    # Bollinger Band Width(20)
    bb_middle = price.rolling(window=20).mean()
    bb_std = price.rolling(window=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    features_df["usd_zar_bollinger_width_20"] = bb_width.shift(1)
    
    # --- 7. Calendar Indicators ---
    month = combined_df.index.month
    features_df["month_sin"] = np.sin(2 * np.pi * month / 12)
    features_df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    # Save output
    features_df.to_csv(output_path, index=True)
    print(f"\nSaved {features_df.shape[1]} features to {output_path}")
    print(f"Shape: {features_df.shape}")
    
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
