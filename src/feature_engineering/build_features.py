"""
build_features.py — Feature Engineering Stage
Constructs lags, rolling window statistics, momentum indicator, calendar signs, 
commodity spreads, volatility regimes, and technical indicators for currency-commodity modeling.

Input : data/processed/preprocessed_data.csv
        data/results/correlation_granger_results.csv
        data/results/log_return_data.csv
Output: data/processed/features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

def lag_features(df, granger_csv):
    """
    1. target self-lags: usd_zar_logret shift 1-5
    2. commodity lags at each Granger-selected commodity's Best_Lag (from CSV, Granger_min_p<0.05)
    3. commodity lags at fixed shift 1-3 for same set
    """
    granger_df = pd.read_csv(granger_csv)
    significant_df = granger_df[granger_df['Granger_Significant'] == 'Yes']
    selected_commodities = significant_df['Commodity'].tolist()
    best_lags = dict(zip(significant_df['Commodity'], significant_df['Best_Lag']))
    
    features = pd.DataFrame(index=df.index)
    
    # 1. target self-lags: usd_zar_logret shift 1-5
    for lag in range(1, 6):
        features[f"usd_zar_logret_lag_{lag}"] = df["usd_zar_logret"].shift(lag)
        
    # 2. commodity lags at each Granger-selected commodity's Best_Lag
    for comm in selected_commodities:
        best_lag = best_lags[comm]
        features[f"{comm}_lag_{best_lag}"] = df[comm].shift(best_lag)
        
    # 3. commodity lags at fixed shift 1-3 for same set
    for comm in selected_commodities:
        for lag in range(1, 4):
            features[f"{comm}_lag_{lag}"] = df[comm].shift(lag)
            
    return features

def rolling_features(df, granger_csv, windows=[5, 10, 20]):
    """
    1. rolling mean, std (shift 1) for usd_zar_logret + Granger-selected commodities
    2. realized volatility: sqrt(rolling sum of squared logret) (shift 1)
    3. expanding mean/std for usd_zar_logret (shift 1)
    """
    granger_df = pd.read_csv(granger_csv)
    significant_df = granger_df[granger_df['Granger_Significant'] == 'Yes']
    selected_commodities = significant_df['Commodity'].tolist()
    
    targets = ["usd_zar_logret"] + selected_commodities
    features = pd.DataFrame(index=df.index)
    
    for col in targets:
        for w in windows:
            # rolling mean, std (shift 1)
            features[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w).mean().shift(1)
            features[f"{col}_roll_std_{w}"] = df[col].rolling(window=w).std().shift(1)
            
            # realized volatility: sqrt(rolling sum of squared logret) (shift 1)
            features[f"{col}_realized_vol_{w}"] = np.sqrt((df[col]**2).rolling(window=w).sum()).shift(1)
            
    # expanding mean/std for usd_zar_logret (shift 1)
    features["usd_zar_logret_expanding_mean"] = df["usd_zar_logret"].expanding().mean().shift(1)
    features["usd_zar_logret_expanding_std"] = df["usd_zar_logret"].expanding().std().shift(1)
    
    return features

def momentum_features(df, granger_csv, windows=[5, 10, 20]):
    """
    1. cumulative log return sum over window (shift 1)
    2. price-level pct momentum (P[t-1]-P[t-w-1])/P[t-w-1]
    """
    granger_df = pd.read_csv(granger_csv)
    significant_df = granger_df[granger_df['Granger_Significant'] == 'Yes']
    selected_commodities = significant_df['Commodity'].tolist()
    
    targets = ["usd_zar_logret"] + selected_commodities
    features = pd.DataFrame(index=df.index)
    
    for col in targets:
        for w in windows:
            # cumulative log return sum over window (shift 1)
            features[f"{col}_cum_logret_{w}"] = df[col].rolling(window=w).sum().shift(1)
            
    # price-level pct momentum (P[t-1]-P[t-w-1])/P[t-w-1]
    price_cols = ["usd_zar"] + [c.replace("_logret", "") for c in selected_commodities]
    for col in price_cols:
        for w in windows:
            features[f"{col}_price_momentum_{w}"] = (df[col].shift(1) - df[col].shift(w + 1)) / df[col].shift(w + 1)
            
    return features

def calendar_features(df):
    """
    day-of-week sin/cos, month sin/cos, month-end flag, quarter-end flag
    """
    features = pd.DataFrame(index=df.index)
    dates = pd.to_datetime(df.index)
    
    day_of_week = dates.dayofweek
    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    
    month = dates.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    features["is_month_end"] = dates.is_month_end.astype(float)
    features["is_quarter_end"] = dates.is_quarter_end.astype(float)
    
    return features

def spread_features(df):
    """
    Gold-Silver, Brent_Oil-WTI_Crude_Oil, platinum-Palladium logret spreads
    skip pair if either leg has <80% overlapping non-null history.
    Spreads are shifted by 1 to prevent leakage.
    """
    features = pd.DataFrame(index=df.index)
    
    pairs = [
        ("Gold_logret", "Silver_logret", "gold_silver_spread"),
        ("Brent_Oil_logret", "WTI Crude Oil_logret", "brent_wti_spread"),
        ("platinum_logret", "Palladium_logret", "platinum_palladium_spread")
    ]
    
    for leg1, leg2, name in pairs:
        if leg1 not in df.columns or leg2 not in df.columns:
            print(f"Skipping spread {name}: one or both legs not found in DataFrame.")
            continue
            
        overlap = (df[leg1].notna() & df[leg2].notna()).mean()
        if overlap >= 0.8:
            print(f"Spread {name} overlap: {overlap:.2%}")
            features[name] = (df[leg1] - df[leg2]).shift(1)
        else:
            print(f"Skipping spread {name} due to low overlap: {overlap:.2%}")
            
    return features

def regime_features(df):
    """
    binary high/low vol flag: usd_zar rolling(20) std vs its own expanding percentile up to t-1
    """
    features = pd.DataFrame(index=df.index)
    
    # 20-day rolling volatility of usd_zar log returns
    vol = df["usd_zar_logret"].rolling(window=20).std()
    
    # Shift to t-1 to prevent leakage
    vol_lag = vol.shift(1)
    
    # Get expanding median (50th percentile) of volatility up to t-2
    expanding_median = vol_lag.shift(1).expanding(min_periods=20).median()
    
    features["usd_zar_high_vol_flag"] = (vol_lag > expanding_median).astype(float)
    features.loc[expanding_median.isna(), "usd_zar_high_vol_flag"] = np.nan
    
    return features

def interaction_features(df, granger_csv, top_n=5):
    """
    pairwise products of top 5 Granger-significant lagged commodity features only (by min Granger_min_p)
    """
    granger_df = pd.read_csv(granger_csv)
    significant_df = granger_df[granger_df['Granger_Significant'] == 'Yes']
    sorted_df = significant_df.sort_values(by=["Granger_min_p", "Pearson_p"])
    top_commodities = sorted_df["Commodity"].head(top_n).tolist()
    
    features = pd.DataFrame(index=df.index)
    
    lagged_features = {}
    for comm in top_commodities:
        lagged_features[comm] = df[comm].shift(1)
        
    for i in range(len(top_commodities)):
        for j in range(i + 1, len(top_commodities)):
            c1 = top_commodities[i]
            c2 = top_commodities[j]
            features[f"{c1}_lag1_x_{c2}_lag1"] = lagged_features[c1] * lagged_features[c2]
            
    return features

def technical_indicators(df):
    """
    RSI(14), MACD(12,26,9), Bollinger width(20) on usd_zar price level.
    All shifted by 1 after computation to prevent leakage.
    """
    features = pd.DataFrame(index=df.index)
    price = df["usd_zar"]
    
    # 1. RSI(14)
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    features["usd_zar_rsi_14"] = rsi.shift(1)
    
    # 2. MACD(12,26,9)
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    
    features["usd_zar_macd"] = macd.shift(1)
    features["usd_zar_macd_signal"] = signal.shift(1)
    features["usd_zar_macd_hist"] = hist.shift(1)
    
    # 3. Bollinger Width(20)
    bb_middle = price.rolling(window=20).mean()
    bb_std = price.rolling(window=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    features["usd_zar_bollinger_width_20"] = bb_width.shift(1)
    
    return features

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
    
    print("Building features...")
    f1 = lag_features(combined_df, granger_path)
    f2 = rolling_features(combined_df, granger_path)
    f3 = momentum_features(combined_df, granger_path)
    f4 = calendar_features(combined_df)
    f5 = spread_features(combined_df)
    f6 = regime_features(combined_df)
    f7 = interaction_features(combined_df, granger_path)
    f8 = technical_indicators(combined_df)
    
    print("Merging features...")
    features_df = f1.join([f2, f3, f4, f5, f6, f7, f8], how="outer")
    features_df.sort_index(inplace=True)
    
    features_df.to_csv(output_path, index=True)
    print(f"\nSaved features to {output_path}")
    print(f"Shape: {features_df.shape}")
    
    # Print leading vs interior NaNs per column
    print("\nNaN count details per column:")
    print(f"{'Column Name':<50} | {'Leading NaNs':<12} | {'Interior NaNs':<13} | {'Total NaNs':<10}")
    print("-" * 95)
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
        print(f"{col:<50} | {leading:<12} | {interior:<13} | {total:<10}")

if __name__ == "__main__":
    main()
